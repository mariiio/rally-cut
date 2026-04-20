"""Oracle test: do candidate generators produce candidates near every GT contact?

Answers the decisive question before investing in sequence decoding:

- If candidate generators produce a candidate within ±7 frames of every GT →
  the 200 FN contacts are rejected by the GBM classifier, not missed by
  generation. Sequence decoding (Viterbi on the candidate lattice) could
  plausibly rescue them.
- If generators miss many GT contacts entirely → no downstream decoder can
  help. Real bottleneck is upstream.

For each video, under LOO-per-video semantics, we:
1. Generate all candidates (same 8 generators as production detect_contacts)
   at pre-classifier stage — NO threshold applied.
2. Run the Phase-0 GBM pipeline (LOO-trained, threshold=0.30) to get the
   actual production predictions + per-GT hit/miss.
3. For each GT contact compute:
   - was it caught by the GBM (HIT) or missed (FN)?
   - what's the distance to the NEAREST candidate from the generator?
   - what's the distance to the nearest GBM-accepted contact?
4. For FN contacts, measure the fraction where a candidate exists within
   ±7f — this is the "sequence decoding ceiling": the best a perfect
   structural filter could rescue.

Writes ``reports/oracle_candidate_coverage_2026_04_20.md``.

Usage (cd analysis):
    uv run python scripts/oracle_candidate_coverage.py
    uv run python scripts/oracle_candidate_coverage.py --limit 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from scripts.eval_action_detection import (
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.eval_loo_video import (
    RallyPrecomputed,
    _inject_action_classifier,
    _precompute,
    _reset_action_classifier_cache,
    _train_fold,
)
from scripts.train_contact_classifier import extract_candidate_features

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)
console = Console()


@dataclass
class GTEntry:
    video_id: str
    rally_id: str
    gt_frame: int
    gt_action: str
    gbm_hit: bool
    nearest_candidate_dist: int    # frames to nearest generator candidate
    nearest_accepted_dist: int     # frames to nearest GBM-accepted contact
    fps: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--tolerance-ms", type=int, default=233)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str,
                        default="reports/oracle_candidate_coverage_2026_04_20.md")
    args = parser.parse_args()

    t_start = time.time()
    rallies = load_rallies_with_action_gt()
    contact_cfg = ContactDetectionConfig()

    console.print(f"[bold]Precomputing features for {len(rallies)} rallies...[/bold]")
    t_pre = time.time()
    precomputed: list[RallyPrecomputed] = []
    # Track candidate frames per-rally (not stored in RallyPrecomputed)
    candidate_frames_by_rally: dict[str, list[int]] = {}
    for rally in rallies:
        pre = _precompute(rally, contact_cfg)
        if pre is None:
            continue
        precomputed.append(pre)
        # Re-run the candidate extractor just to get the frames list
        _feats, cand_frames = extract_candidate_features(
            rally, config=contact_cfg,
            gt_frames=[gt.frame for gt in rally.gt_labels],
            sequence_probs=pre.sequence_probs,
        )
        candidate_frames_by_rally[rally.rally_id] = list(cand_frames)
    console.print(f"  {len(precomputed)} rallies, {sum(len(v) for v in candidate_frames_by_rally.values())} candidates ({time.time()-t_pre:.0f}s)")

    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for pre in precomputed:
        by_video[pre.rally.video_id].append(pre)

    video_ids = sorted(by_video.keys())
    if args.limit:
        video_ids = video_ids[: args.limit]

    entries: list[GTEntry] = []
    console.print(f"\n[bold]LOO-per-video GBM tagging ({len(video_ids)} folds)...[/bold]")

    for fold_idx, vid in enumerate(video_ids, start=1):
        t_fold = time.time()
        held = by_video[vid]
        train = [pre for v, rs in by_video.items() if v != vid for pre in rs]
        contact_clf, action_clf = _train_fold(train, args.threshold)

        for pre in held:
            rally = pre.rally
            _inject_action_classifier(action_clf if action_clf.is_trained else None)
            try:
                contact_seq = detect_contacts(
                    ball_positions=pre.ball_positions,
                    player_positions=pre.player_positions,
                    config=contact_cfg,
                    net_y=rally.court_split_y,
                    frame_count=rally.frame_count or None,
                    classifier=contact_clf,
                    use_classifier=True,
                    sequence_probs=pre.sequence_probs,
                )
                rally_actions = classify_rally_actions(
                    contact_seq, rally_id=rally.rally_id,
                    use_classifier=action_clf.is_trained,
                    sequence_probs=pre.sequence_probs,
                )
            finally:
                _reset_action_classifier_cache()

            pred_actions = [a.to_dict() for a in rally_actions.actions]
            real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
            tol_frames = max(1, round(rally.fps * args.tolerance_ms / 1000))
            matches, _ = match_contacts(
                rally.gt_labels, real_pred, tolerance=tol_frames,
            )

            cand_frames = candidate_frames_by_rally.get(rally.rally_id, [])
            accepted_frames = [a.get("frame", 0) for a in real_pred]

            for m in matches:
                gt_f = m.gt_frame
                # Distance to nearest candidate (pre-classifier)
                cand_dist = (
                    min(abs(gt_f - cf) for cf in cand_frames)
                    if cand_frames else 9999
                )
                accepted_dist = (
                    min(abs(gt_f - af) for af in accepted_frames)
                    if accepted_frames else 9999
                )
                entries.append(GTEntry(
                    video_id=rally.video_id,
                    rally_id=rally.rally_id,
                    gt_frame=gt_f,
                    gt_action=m.gt_action,
                    gbm_hit=m.pred_frame is not None,
                    nearest_candidate_dist=cand_dist,
                    nearest_accepted_dist=accepted_dist,
                    fps=rally.fps,
                ))

        console.print(f"  [{fold_idx}/{len(video_ids)}] {vid[:8]} ({time.time()-t_fold:.0f}s)")

    # Analysis
    console.print(f"\n[bold]Analysis (wall-clock {(time.time()-t_start)/60:.1f}m)[/bold]")
    n_total = len(entries)
    n_hit = sum(1 for e in entries if e.gbm_hit)
    n_miss = n_total - n_hit

    # Oracle check: among GBM MISSes, how many have a candidate within ±7f?
    misses = [e for e in entries if not e.gbm_hit]
    miss_covered_7 = sum(
        1 for e in misses
        if e.nearest_candidate_dist <= max(1, round(e.fps * args.tolerance_ms / 1000))
    )
    sum(
        1 for e in misses
        if e.nearest_candidate_dist <= max(1, round(e.fps * 10 / 30 * 1000 / 1000) + 10)
    )

    # Hit coverage (sanity: should be ~100%)
    hits = [e for e in entries if e.gbm_hit]
    hit_covered = sum(
        1 for e in hits
        if e.nearest_candidate_dist <= max(1, round(e.fps * args.tolerance_ms / 1000))
    )

    # Per-action breakdown
    per_action: dict[str, dict[str, int]] = defaultdict(
        lambda: {"miss": 0, "miss_cand_covered": 0})
    for e in misses:
        per_action[e.gt_action]["miss"] += 1
        tol = max(1, round(e.fps * args.tolerance_ms / 1000))
        if e.nearest_candidate_dist <= tol:
            per_action[e.gt_action]["miss_cand_covered"] += 1

    # Report
    lines: list[str] = []
    lines.append("# Oracle Candidate-Coverage Test")
    lines.append("")
    lines.append(f"- GT contacts: {n_total}")
    lines.append(f"- GBM hits (baseline 88.0% F1 reproducing): {n_hit} ({n_hit/max(1,n_total):.1%})")
    lines.append(f"- GBM misses: {n_miss}")
    lines.append("")
    lines.append("## The decisive number")
    lines.append("")
    lines.append(f"**Of {n_miss} GBM-MISS contacts, {miss_covered_7} have a generator candidate within ±7 frames.**")
    lines.append(f"= {miss_covered_7 / max(1, n_miss):.1%} of misses are recoverable by a perfect downstream filter.")
    lines.append("")
    lines.append("## Interpretation")
    if miss_covered_7 / max(1, n_miss) >= 0.70:
        lines.append(f"✅ **PROCEED with sequence decoding.** {miss_covered_7} candidates are waiting to be rescued.")
        lines.append("Upper-bound F1 if a perfect structural filter accepts all of them without FPs: ")
        new_tp = n_hit + miss_covered_7
        new_fn = n_miss - miss_covered_7
        lines.append(f"TP={new_tp}, FN={new_fn}, oracle-F1 upper bound ≈ " +
                     f"{2 * new_tp / (2 * new_tp + 174 + new_fn):.1%} (assuming FP=174 unchanged).")
    elif miss_covered_7 / max(1, n_miss) >= 0.40:
        lines.append(f"⚠️ **Partial signal.** {miss_covered_7} candidates exist for rescue, but {n_miss - miss_covered_7} misses have no candidate nearby — those are generator failures.")
        lines.append("Sequence decoding could help the first group. Candidate-generator improvements needed for the second.")
    else:
        lines.append("❌ **STOP.** Most GBM-misses don't have candidates within ±7f.")
        lines.append("Sequence decoding can't rescue what isn't generated. The bottleneck is candidate generation, not classification.")
    lines.append("")
    lines.append("## Sanity check (HIT coverage — should be ~100%)")
    lines.append("")
    lines.append(f"Of {n_hit} GBM-HIT contacts, {hit_covered} have a generator candidate within ±7f ({hit_covered/max(1,n_hit):.1%}).")
    lines.append("")
    lines.append("## Per-action breakdown of misses")
    lines.append("")
    lines.append("| Action | Misses | Candidate within ±7f | Coverage |")
    lines.append("|---|---|---|---|")
    for act in sorted(per_action.keys()):
        p = per_action[act]
        cov = p["miss_cand_covered"] / max(1, p["miss"])
        lines.append(f"| {act} | {p['miss']} | {p['miss_cand_covered']} | {cov:.1%} |")
    lines.append("")

    # Distance distribution of candidate to nearest GT-miss (diagnostic)
    lines.append("## Distance from GBM-MISS GT to nearest candidate (diagnostic)")
    lines.append("")
    miss_dists = sorted(e.nearest_candidate_dist for e in misses)
    if miss_dists:
        for p in [25, 50, 75, 90, 95]:
            idx = min(len(miss_dists) - 1, int(len(miss_dists) * p / 100))
            lines.append(f"- P{p}: {miss_dists[idx]} frames")
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))

    # Console summary
    table = Table(title="Oracle Candidate Coverage", show_header=True, header_style="bold")
    for col in ("Metric", "Value"):
        table.add_column(col)
    table.add_row("Total GT contacts", str(n_total))
    table.add_row("GBM HIT", f"{n_hit}")
    table.add_row("GBM MISS", f"{n_miss}")
    table.add_row("HIT w/ candidate ≤7f (sanity)", f"{hit_covered} ({hit_covered/max(1,n_hit):.1%})")
    table.add_row("MISS w/ candidate ≤7f [DECISIVE]",
                  f"[bold]{miss_covered_7} ({miss_covered_7/max(1,n_miss):.1%})[/bold]")
    console.print(table)

    # JSON dump
    json_path = Path(args.out.replace(".md", ".json"))
    json_path.write_text(json.dumps({
        "n_total": n_total, "n_hit": n_hit, "n_miss": n_miss,
        "miss_covered_7f": miss_covered_7,
        "hit_covered_7f": hit_covered,
        "per_action": {k: dict(v) for k, v in per_action.items()},
    }, indent=2))

    console.print(f"\n[green]Report: {out}[/green]")
    console.print(f"[green]JSON: {json_path}[/green]")


if __name__ == "__main__":
    main()
