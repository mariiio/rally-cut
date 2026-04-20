"""Step 1 diagnostic: why do 95 GBM-miss GT contacts have NO candidate?

Of 314 GBM-miss contacts, 219 have a candidate within ±7f (rescuable by
decoder). The other 95 have NO candidate — generator never fired. This
script categorizes those 95 into:

- BALL_DROPOUT: ball detection is sparse/low-confidence around the GT frame
  (occlusion, detection failure). Fix: trajectory gap inpainting.
- BALL_PRESENT_NO_EVENT: ball is well-tracked but no generator fired (no
  velocity peak, inflection, direction change, etc.). Fix: serve-specific
  generator OR relaxed generator thresholds.
- OTHER: edge cases (rally boundary, missing data).

Writes ``reports/no_candidate_fn_diagnosis_2026_04_20.md``.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_action_detection import load_rallies_with_action_gt

console = Console()

WINDOW_FRAMES = 15  # ±15 frames around GT contact
BALL_CONF_THRESHOLD = 0.30
DENSITY_THRESHOLD = 0.50  # fraction of frames with ball detected


def main() -> None:
    rallies = load_rallies_with_action_gt()
    {r.rally_id: r for r in rallies}

    oracle = json.loads(
        Path("reports/orthogonality_events_2026_04_19.json").read_text()
    )
    # Oracle events record gbm_hit but NOT nearest_candidate_dist — need to
    # use the newer oracle_candidate_coverage output.
    oracle_cov = json.loads(
        Path("reports/oracle_candidate_coverage_2026_04_20.json").read_text()
    )

    # Re-derive the no-candidate FNs by running the oracle script's logic
    # again. Actually simpler: load per-GT info directly from the coverage
    # pipeline's intermediate events — not saved. So re-produce the filter
    # using oracle_candidate_coverage.py's stats.
    # Shortcut: just grab per-action totals and sample non-rescuable ones
    # by re-running the precompute that oracle script did.
    console.print(
        f"[dim]Oracle coverage summary: {oracle_cov['n_total']} GT contacts, "
        f"{oracle_cov['n_hit']} hit, {oracle_cov['n_miss']} miss, "
        f"{oracle_cov['miss_covered_7f']} rescuable[/dim]"
    )

    # For diagnosis we need per-GT candidate distance. The oracle script
    # doesn't save that — let's regenerate here by re-running the relevant
    # subset of its logic: for each GT, recompute nearest candidate dist.
    from rallycut.tracking.contact_detector import ContactDetectionConfig
    from scripts.eval_loo_video import _precompute
    from scripts.train_contact_classifier import extract_candidate_features

    contact_cfg = ContactDetectionConfig()

    console.print("[bold]Regenerating per-GT candidate distances + ball density...[/bold]")
    cats = Counter()
    detail_by_category: dict[str, list] = defaultdict(list)

    # Use GBM-miss tags from orthogonality_events JSON
    miss_by_key: dict[tuple[str, int], dict] = {
        (e["rally_id"], e["frame"]): e for e in oracle if not e["gbm_hit"]
    }

    for rally in rallies:
        pre = _precompute(rally, contact_cfg)
        if pre is None:
            continue
        _feats, cand_frames = extract_candidate_features(
            rally, config=contact_cfg,
            gt_frames=[gt.frame for gt in rally.gt_labels],
            sequence_probs=pre.sequence_probs,
        )
        cand_frames_list = list(cand_frames)

        # Build frame → ball-confidence map
        ball_conf_by_frame: dict[int, float] = {}
        for bp in (rally.ball_positions_json or []):
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0:
                ball_conf_by_frame[bp["frameNumber"]] = bp.get("confidence", 1.0)

        fps = rally.fps
        tol_frames = max(1, round(fps * 0.233))

        for gt in rally.gt_labels:
            key = (rally.rally_id, gt.frame)
            ev = miss_by_key.get(key)
            if ev is None:
                continue  # GBM HIT, skip
            # Distance to nearest candidate
            if cand_frames_list:
                nearest = min(abs(gt.frame - cf) for cf in cand_frames_list)
            else:
                nearest = 9999
            if nearest <= tol_frames:
                continue  # rescuable, skip

            # Ball density around GT frame
            window = range(max(0, gt.frame - WINDOW_FRAMES),
                           gt.frame + WINDOW_FRAMES + 1)
            detected = sum(
                1 for f in window if ball_conf_by_frame.get(f, 0.0) >= BALL_CONF_THRESHOLD
            )
            density = detected / (2 * WINDOW_FRAMES + 1)

            # Ball-track gap around GT specifically (any undetected frame in ±5 of GT)
            near_window = range(max(0, gt.frame - 5), gt.frame + 6)
            near_detected = sum(
                1 for f in near_window if ball_conf_by_frame.get(f, 0.0) >= BALL_CONF_THRESHOLD
            )
            near_density = near_detected / 11

            cat = _categorize(density, near_density, gt.action)
            cats[cat] += 1
            detail_by_category[cat].append({
                "rally_id": rally.rally_id,
                "frame": gt.frame,
                "action": gt.action,
                "density": density,
                "near_density": near_density,
                "nearest_cand": nearest,
                "fps": fps,
            })

    # Report
    total = sum(cats.values())
    console.print(f"\n[bold]No-candidate FN categorization (n={total})[/bold]")
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Category", style="bold")
    tbl.add_column("Count", justify="right")
    tbl.add_column("Fraction", justify="right")
    for cat, n in cats.most_common():
        tbl.add_row(cat, str(n), f"{n/max(1,total):.1%}")
    console.print(tbl)

    # Per-action breakdown of BALL_DROPOUT
    for cat in ["BALL_DROPOUT", "BALL_PRESENT_NO_EVENT", "EDGE"]:
        items = detail_by_category.get(cat, [])
        if not items:
            continue
        act_counts = Counter(x["action"] for x in items)
        console.print(f"\n[bold]{cat} (n={len(items)}) per action:[/bold] "
                      f"{dict(act_counts.most_common())}")

    # Serve-specific deep dive: how many no-candidate serve FNs are ball-dropout?
    serve_items = [x for cat in detail_by_category.values() for x in cat
                   if x["action"] == "serve"]
    serve_dropout = sum(
        1 for x in serve_items
        if x["density"] < DENSITY_THRESHOLD or x["near_density"] < 0.3
    )
    serve_present = len(serve_items) - serve_dropout
    console.print("\n[bold]Serve-specific[/bold]")
    console.print(f"  Total no-candidate serve FNs: {len(serve_items)}")
    console.print(f"  Ball-dropout (density<50% or near<30%): {serve_dropout}")
    console.print(f"  Ball-present but no event: {serve_present}")

    # Verdict
    dropout_total = cats.get("BALL_DROPOUT", 0)
    cats.get("BALL_PRESENT_NO_EVENT", 0)
    ratio = dropout_total / max(1, total)

    if ratio >= 0.60:
        verdict = "PROCEED with ball-gap inpainting (Step 2)"
        color = "green"
    elif ratio >= 0.30:
        verdict = "PARTIAL — ball-gap inpainting helps some, serve-generator helps others"
        color = "yellow"
    else:
        verdict = "SKIP ball-gap fix — FNs are not dropout-driven; focus on generator logic"
        color = "red"
    console.print(f"\n[bold][{color}]{verdict}[/{color}][/bold]")
    console.print(f"  BALL_DROPOUT fraction: {ratio:.1%} (threshold: ≥60% for PROCEED)")

    # Write report
    out = Path("reports/no_candidate_fn_diagnosis_2026_04_20.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# No-Candidate FN Diagnosis")
    lines.append("")
    lines.append(f"- Total no-candidate GBM-miss GT contacts: {total}")
    lines.append(f"- Ball-density window: ±{WINDOW_FRAMES} frames (ball_conf ≥ {BALL_CONF_THRESHOLD})")
    lines.append("")
    lines.append("## Categorization")
    lines.append("")
    lines.append("| Category | Count | Fraction |")
    lines.append("|---|---|---|")
    for cat, n in cats.most_common():
        lines.append(f"| {cat} | {n} | {n/max(1,total):.1%} |")
    lines.append("")
    lines.append("## Per-action breakdown")
    lines.append("")
    for cat in ["BALL_DROPOUT", "BALL_PRESENT_NO_EVENT", "EDGE"]:
        items = detail_by_category.get(cat, [])
        if not items:
            continue
        act_counts = Counter(x["action"] for x in items)
        lines.append(f"- **{cat}** (n={len(items)}): "
                     f"{dict(act_counts.most_common())}")
    lines.append("")
    lines.append("## Serve-specific")
    lines.append("")
    lines.append(f"- Total no-candidate serve FNs: {len(serve_items)}")
    lines.append(f"- Ball-dropout: {serve_dropout}")
    lines.append(f"- Ball-present but no event: {serve_present}")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append(f"- BALL_DROPOUT fraction: {ratio:.1%}")
    out.write_text("\n".join(lines))
    console.print(f"\n[green]Report: {out}[/green]")


def _categorize(density: float, near_density: float, action: str) -> str:
    """Bucket each no-candidate FN."""
    # EDGE: total data gap (almost no ball detections)
    if density <= 0.05:
        return "EDGE"
    # Ball dropout: low overall density OR gap near the contact
    if density < DENSITY_THRESHOLD or near_density < 0.3:
        return "BALL_DROPOUT"
    # Ball well tracked but generator didn't fire
    return "BALL_PRESENT_NO_EVENT"


if __name__ == "__main__":
    main()
