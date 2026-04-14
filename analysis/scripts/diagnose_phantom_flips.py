"""Diagnose whether match_tracker phantom flips are the dominant source
of team_near errors.

For rallies where localize_team_near returns WRONG team_near:
  1. Is the track_to_player consistent within the rally? (no = bad within-rally tracking)
  2. Is the Y-gap marginal (< 0.05)? (yes = threshold issue, not phantom flip)
  3. Does the implied team assignment match the majority of other rallies
     in the same "segment" (between side switches)? (no = phantom flip)

If phantom flips dominate → match_tracker fix has high ROI.
If Y-gap or tracking noise dominate → match_tracker fix won't help much.

Usage:
    cd analysis
    uv run python scripts/diagnose_phantom_flips.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from rich.console import Console
from rich.table import Table

from eval_action_detection import _load_track_to_player_maps  # noqa: E402
from eval_score_tracking import load_score_gt  # noqa: E402
from production_eval import _load_team_templates_by_video  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking.team_identity import localize_team_near  # noqa: E402

console = Console()


def _parse_positions(pos_json: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            track_id=p.get("trackId", p.get("track_id", -1)),
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            x=p["x"], y=p["y"],
            width=p["width"], height=p["height"],
            confidence=p.get("confidence", 1.0),
        )
        for p in pos_json
    ]


def _compute_y_gap(
    positions: list[PlayerPosition],
    t2p: dict[int, int],
    templates,
) -> float:
    """Recompute the Y gap between teams for this rally."""
    t0, t1 = templates
    t0_pids = set(t0.player_ids)
    t1_pids = set(t1.player_ids)

    pid_ys: dict[int, list[float]] = {}
    for p in positions:
        pid = t2p.get(p.track_id)
        if pid is not None:
            pid_ys.setdefault(pid, []).append(p.y + p.height / 2.0)

    t0_y = [float(np.mean(pid_ys[pid])) for pid in t0_pids if pid in pid_ys]
    t1_y = [float(np.mean(pid_ys[pid])) for pid in t1_pids if pid in pid_ys]
    if not t0_y or not t1_y:
        return 0.0

    return abs(float(np.mean(t0_y)) - float(np.mean(t1_y)))


def main() -> int:
    console.print("[bold]Loading data...[/bold]")
    video_rallies = load_score_gt()
    video_ids = set(video_rallies.keys())
    templates_by_vid = _load_team_templates_by_video(video_ids)
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    console.print(f"  {len(templates_by_vid)} videos with templates\n")

    # First pass: compute label_a per video (via majority vote)
    vid_label_a: dict[str, str] = {}
    for vid, rallies in sorted(video_rallies.items()):
        templates = templates_by_vid.get(vid)
        if templates is None:
            continue
        t0, t1 = templates
        votes: dict[str, int] = {t0.team_label: 0, t1.team_label: 0}
        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                continue
            positions = _parse_positions(rally.positions)
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue
            tn = localize_team_near(positions, t2p, templates, min_y_gap=0.0)
            if tn is None:
                continue
            serving_label = tn if formation_side == "near" else (
                t0.team_label if tn == t1.team_label else t1.team_label
            )
            if rally.gt_serving_team == "A":
                votes[serving_label] += 1
            else:
                votes[serving_label] -= 1
        if votes:
            vid_label_a[vid] = max(votes, key=lambda k: votes[k])

    # Per-video, per-"segment" (between side switches) team_near majority
    # For simplicity we use side_flipped flag as the segmenter.
    # Segment 0 = no flip, segment 1 = flipped, etc.
    segment_team_near: dict[tuple[str, bool], Counter] = defaultdict(Counter)

    # First gather all team_near outputs and their segment
    all_results: list[dict] = []
    for vid, rallies in sorted(video_rallies.items()):
        templates = templates_by_vid.get(vid)
        if templates is None or vid not in vid_label_a:
            continue
        t0, t1 = templates
        label_a = vid_label_a[vid]
        label_b = t0.team_label if label_a == t1.team_label else t1.team_label

        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                continue
            positions = _parse_positions(rally.positions)
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue

            if formation_side == "near":
                expected_tn = label_a if rally.gt_serving_team == "A" else label_b
            else:
                expected_tn = label_b if rally.gt_serving_team == "A" else label_a

            tn = localize_team_near(positions, t2p, templates)  # default 0.03
            y_gap = _compute_y_gap(positions, t2p, templates)
            segment_key = (vid, bool(rally.side_flipped))

            all_results.append({
                "vid": vid, "rally_id": rally.rally_id,
                "tn": tn, "expected_tn": expected_tn,
                "y_gap": y_gap, "segment_key": segment_key,
                "n_players": len({t2p[tid] for tid in t2p if tid >= 0}),
            })

            # Contribute to segment majority (using all non-None returns)
            if tn is not None:
                segment_team_near[segment_key][tn] += 1

    # Analyze wrong returns
    wrong = [r for r in all_results if r["tn"] is not None and r["tn"] != r["expected_tn"]]
    correct = [r for r in all_results if r["tn"] is not None and r["tn"] == r["expected_tn"]]
    none_returns = [r for r in all_results if r["tn"] is None]

    total = len(all_results)
    console.print(f"Total gated samples: {total}")
    console.print(f"  Correct: {len(correct)} ({len(correct)/total:.0%})")
    console.print(f"  Wrong:   {len(wrong)} ({len(wrong)/total:.0%})")
    console.print(f"  None:    {len(none_returns)} ({len(none_returns)/total:.0%})")
    console.print()

    # Bucket wrong returns by likely cause
    buckets: Counter = Counter()
    bucket_examples: dict[str, list] = defaultdict(list)

    for r in wrong:
        y_gap = r["y_gap"]
        majority = segment_team_near[r["segment_key"]].most_common(1)
        seg_majority = majority[0][0] if majority else None
        seg_total = sum(segment_team_near[r["segment_key"]].values())

        if y_gap < 0.05:
            bucket = "marginal_y_gap"
        elif seg_majority == r["expected_tn"] and seg_total >= 3:
            # Segment majority agrees with expected, but this rally disagrees
            # → phantom flip: this rally's team assignment is inverted vs segment
            bucket = "phantom_flip_vs_segment"
        elif seg_majority != r["expected_tn"] and seg_total >= 3:
            # Segment majority also wrong → systematic issue, likely template or calibration
            bucket = "systematic_segment_error"
        else:
            bucket = "unknown_or_small_segment"

        buckets[bucket] += 1
        if len(bucket_examples[bucket]) < 3:
            bucket_examples[bucket].append(r)

    table = Table(title="Wrong team_near breakdown")
    table.add_column("Bucket", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("% of wrong", justify="right")
    for b, c in buckets.most_common():
        table.add_row(b, str(c), f"{c/len(wrong):.0%}" if wrong else "—")
    console.print(table)

    console.print("\n[bold]Bucket interpretations:[/bold]")
    console.print("  marginal_y_gap: Y gap < 5% — lowering confidence threshold would filter")
    console.print("  phantom_flip_vs_segment: This rally's team_near inverted vs segment majority")
    console.print("    → match_tracker corruption is real here")
    console.print("  systematic_segment_error: Whole segment wrong")
    console.print("    → Template or calibration issue, not match_tracker")

    # Example rallies
    console.print("\n[bold]Examples by bucket:[/bold]")
    for bucket, examples in bucket_examples.items():
        console.print(f"\n  [cyan]{bucket}[/cyan]:")
        for ex in examples:
            console.print(
                f"    {ex['vid'][:10]}/{ex['rally_id'][:8]}: "
                f"tn={ex['tn']} expected={ex['expected_tn']} "
                f"y_gap={ex['y_gap']:.3f} players={ex['n_players']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
