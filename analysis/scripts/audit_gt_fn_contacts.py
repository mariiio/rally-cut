"""Audit ground truth quality by sampling false negative contacts for visual review.

Samples ~80 FNs stratified by failure category, outputs video name + timestamp
so the user can visually verify each in the web editor or video player.

Verdicts:
  - gt_error:   GT label is wrong (not a real contact) → re-label
  - real_miss:  Genuine contact the detector missed → keep GT
  - ambiguous:  Can't tell from the video → flag for second opinion

Usage:
    cd analysis
    uv run python scripts/audit_gt_fn_contacts.py
    uv run python scripts/audit_gt_fn_contacts.py --sample 100
    uv run python scripts/audit_gt_fn_contacts.py --category rejected_by_classifier
    uv run python scripts/audit_gt_fn_contacts.py --all  # Don't sample, show all FNs
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.tracking.contact_classifier import (
    CandidateFeatures,
    load_contact_classifier,
)
from rallycut.tracking.contact_detector import detect_contacts
from scripts.diagnose_fn_contacts import (
    FNDiagnostic,
    _reconstruct_ball_player_data,
    diagnose_rally_fns,
)
from scripts.eval_action_detection import (
    GtLabel,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def _load_video_filenames(video_ids: set[str]) -> dict[str, str]:
    """Load video filename for each video_id."""
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"SELECT id, filename FROM videos WHERE id IN ({placeholders})"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            return {row[0]: row[1] for row in cur.fetchall()}


def _format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS.s for easy video seeking."""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:04.1f}"


def _stratified_sample(
    diagnostics: list[FNDiagnostic],
    n: int,
    category_filter: str | None = None,
) -> list[FNDiagnostic]:
    """Stratified sample across FN categories, proportional to category size."""
    if category_filter:
        diagnostics = [d for d in diagnostics if d.category == category_filter]
    if len(diagnostics) <= n:
        return diagnostics

    by_cat: dict[str, list[FNDiagnostic]] = defaultdict(list)
    for d in diagnostics:
        by_cat[d.category].append(d)

    result: list[FNDiagnostic] = []
    remaining = n
    cats = sorted(by_cat.keys(), key=lambda c: len(by_cat[c]))

    # Proportional allocation with minimum 1 per category
    for i, cat in enumerate(cats):
        pool = by_cat[cat]
        cats_left = len(cats) - i
        alloc = max(1, round(len(pool) / len(diagnostics) * n))
        alloc = min(alloc, remaining - (cats_left - 1), len(pool))  # reserve 1 per remaining cat
        alloc = max(1, alloc)
        sampled = random.sample(pool, min(alloc, len(pool)))
        result.extend(sampled)
        remaining -= len(sampled)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GT quality on FN contacts")
    parser.add_argument("--sample", type=int, default=80, help="Number of FNs to sample (default 80)")
    parser.add_argument("--category", type=str, help="Filter to specific FN category")
    parser.add_argument("--all", action="store_true", help="Show all FNs (no sampling)")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--tolerance-ms", type=int, default=167,
        help="Time tolerance in ms for matching (default: 167)",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    # Load classifier — handle feature dimension mismatch gracefully
    classifier = load_contact_classifier()
    if classifier is None:
        console.print("[yellow]No trained contact classifier — using hand-tuned gates[/yellow]")
    else:
        # Check if model expects fewer features than current CandidateFeatures
        try:
            n_model = classifier.model.n_features_in_
            n_current = len(CandidateFeatures.feature_names())
            if n_model != n_current:
                console.print(
                    f"[yellow]Classifier expects {n_model} features, code produces {n_current} "
                    f"— falling back to hand-tuned gates[/yellow]"
                )
                classifier = None
            else:
                console.print(
                    f"[dim]Loaded contact classifier (threshold={classifier.threshold:.2f})[/dim]"
                )
        except AttributeError:
            console.print(
                f"[dim]Loaded contact classifier (threshold={classifier.threshold:.2f})[/dim]"
            )

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action ground truth.[/red]")
        return

    console.print(f"Diagnosing FN contacts across {len(rallies)} rallies...")

    # Collect all FN diagnostics
    all_diagnostics: list[FNDiagnostic] = []
    rally_data_map: dict[str, tuple[str, int, float]] = {}  # rally_id -> (video_id, start_ms, fps)
    total_gt = 0
    total_tp = 0

    for i, rally in enumerate(rallies):
        if (i + 1) % 20 == 0 or i == len(rallies) - 1:
            console.print(f"  [{i + 1}/{len(rallies)}]")

        rally_data_map[rally.rally_id] = (rally.video_id, rally.start_ms, rally.fps)
        ball_positions, player_positions = _reconstruct_ball_player_data(rally)
        pred_actions: list[dict] = []

        if ball_positions:
            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                net_y=rally.court_split_y,
                frame_count=rally.frame_count or None,
                classifier=classifier,
                use_classifier=classifier is not None,
            )
            pred_actions = [c.to_dict() for c in contacts.contacts]

        tolerance_frames = max(1, round(rally.fps * args.tolerance_ms / 1000))
        matches, _ = match_contacts(
            rally.gt_labels, pred_actions, tolerance=tolerance_frames,
        )

        fn_labels = [
            GtLabel(
                frame=m.gt_frame,
                action=m.gt_action,
                player_track_id=-1,
                ball_x=next(
                    (gt.ball_x for gt in rally.gt_labels if gt.frame == m.gt_frame), None
                ),
                ball_y=next(
                    (gt.ball_y for gt in rally.gt_labels if gt.frame == m.gt_frame), None
                ),
            )
            for m in matches if m.pred_frame is None
        ]

        total_gt += len(matches)
        total_tp += sum(1 for m in matches if m.pred_frame is not None)

        if fn_labels:
            diags = diagnose_rally_fns(rally, fn_labels, classifier, tolerance_frames)
            all_diagnostics.extend(diags)

    total_fn = len(all_diagnostics)
    console.print(
        f"\n[bold]Total: {total_gt} GT, {total_tp} TP, {total_fn} FN "
        f"(Recall {total_tp / max(1, total_gt):.1%})[/bold]"
    )

    # Category breakdown
    cat_counts = Counter(d.category for d in all_diagnostics)
    console.print("\n[bold]FN breakdown:[/bold]")
    for cat, count in cat_counts.most_common():
        console.print(f"  {cat}: {count} ({count / total_fn:.0%})")

    if not all_diagnostics:
        console.print("[green]No false negatives![/green]")
        return

    # Sample
    if args.all:
        sampled = all_diagnostics
        if args.category:
            sampled = [d for d in sampled if d.category == args.category]
    else:
        sampled = _stratified_sample(all_diagnostics, args.sample, args.category)

    console.print(f"\n[bold]Sampled {len(sampled)} FNs for visual audit[/bold]")
    sample_cats = Counter(d.category for d in sampled)
    for cat, count in sample_cats.most_common():
        console.print(f"  {cat}: {count}")

    # Load video filenames
    video_ids = {rally_data_map[d.rally_id][0] for d in sampled}
    filenames = _load_video_filenames(video_ids)

    # Output table sorted by video → timestamp for efficient review
    table = Table(title=f"\nGT Audit: {len(sampled)} FN contacts to verify")
    table.add_column("#", justify="right", style="dim", max_width=4)
    table.add_column("Video", max_width=14)
    table.add_column("Time", justify="right")
    table.add_column("Action", max_width=8)
    table.add_column("Category", max_width=22)
    table.add_column("Vel", justify="right", max_width=7)
    table.add_column("Dir°", justify="right", max_width=5)
    table.add_column("PlrDst", justify="right", max_width=6)
    table.add_column("ClfConf", justify="right", max_width=6)
    table.add_column("Rally", style="dim", max_width=10)

    # Sort by video filename then timestamp
    def sort_key(d: FNDiagnostic) -> tuple[str, float]:
        video_id, start_ms, fps = rally_data_map[d.rally_id]
        fname = filenames.get(video_id, video_id[:8])
        abs_seconds = start_ms / 1000 + d.gt_frame / fps
        return (fname, abs_seconds)

    sorted_samples = sorted(sampled, key=sort_key)

    for idx, d in enumerate(sorted_samples, 1):
        video_id, start_ms, fps = rally_data_map[d.rally_id]
        fname = filenames.get(video_id, video_id[:8])
        # Strip extension for brevity
        if fname.endswith((".mp4", ".mov", ".MP4", ".MOV")):
            fname = fname[:-4]
        abs_seconds = start_ms / 1000 + d.gt_frame / fps
        ts = _format_timestamp(abs_seconds)

        vel_str = f"{d.velocity:.4f}" if d.velocity > 0 else "-"
        dir_str = f"{d.direction_change_deg:.0f}" if d.direction_change_deg > 0 else "-"
        import math
        plr_str = f"{d.player_distance:.3f}" if math.isfinite(d.player_distance) else "inf"
        conf_str = f"{d.classifier_confidence:.2f}" if d.classifier_confidence > 0 else "-"

        table.add_row(
            str(idx), fname, ts, d.gt_action, d.category,
            vel_str, dir_str, plr_str, conf_str, d.rally_id[:8],
        )

    console.print(table)

    # Print compact TSV for copy-paste
    console.print("\n[bold]TSV (copy-paste friendly):[/bold]")
    console.print("idx\tvideo\ttime\taction\tcategory\trally_id")
    for idx, d in enumerate(sorted_samples, 1):
        video_id, start_ms, fps = rally_data_map[d.rally_id]
        fname = filenames.get(video_id, video_id[:8])
        if fname.endswith((".mp4", ".mov", ".MP4", ".MOV")):
            fname = fname[:-4]
        abs_seconds = start_ms / 1000 + d.gt_frame / fps
        ts = _format_timestamp(abs_seconds)
        console.print(f"{idx}\t{fname}\t{ts}\t{d.gt_action}\t{d.category}\t{d.rally_id[:8]}")


if __name__ == "__main__":
    main()
