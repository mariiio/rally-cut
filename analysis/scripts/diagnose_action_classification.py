"""Diagnose action classification errors by tracing the state machine.

Prints for each rally:
- Each contact with its GT action, predicted action, court_side, crossed_net, contact_count
- Highlights mismatches

Usage:
    cd analysis
    uv run python scripts/diagnose_action_classification.py
    uv run python scripts/diagnose_action_classification.py --rally <id>
    uv run python scripts/diagnose_action_classification.py --errors-only
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from rich.console import Console

# Reuse the eval script's data loading
sys.path.insert(0, "scripts")
from eval_action_detection import load_rallies_with_action_gt, match_contacts

from rallycut.tracking.action_classifier import (
    _ball_crossed_net,
    classify_rally_actions,
)
from rallycut.tracking.contact_detector import detect_contacts

console = Console()


def trace_classification(rally) -> list[dict]:
    """Run contact detection and classification, annotating with diagnostic info.

    Uses the real classifier for actions (no reimplementation drift) and
    computes auxiliary diagnostic info (crossed_net, player_dist) for display.
    """
    from rallycut.tracking.ball_tracker import BallPosition as BallPos
    from rallycut.tracking.contact_detector import _get_default_classifier
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

    ball_positions = [
        BallPos(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in (rally.ball_positions_json or [])
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]

    player_positions = []
    if rally.positions_json:
        player_positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"],
                y=pp["y"],
                width=pp["width"],
                height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]

    classifier = _get_default_classifier()
    contact_seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
        classifier=classifier,
    )

    # Use the real classifier — no reimplementation drift
    result = classify_rally_actions(contact_seq, rally_id=rally.rally_id)

    # Build diagnostic trace with auxiliary info
    net_y = contact_seq.net_y
    bp_list = contact_seq.ball_positions
    contact_list = contact_seq.contacts

    trace = []
    for i, (contact, action) in enumerate(zip(contact_list, result.actions)):
        crossed_net = None
        if bp_list and i > 0:
            crossed_net = _ball_crossed_net(
                bp_list,
                from_frame=contact_list[i - 1].frame,
                to_frame=contact.frame,
                net_y=net_y,
            )

        trace.append({
            "i": i, "frame": contact.frame,
            "ball_y": contact.ball_y, "court_side": contact.court_side,
            "is_at_net": contact.is_at_net,
            "crossed_net": crossed_net,
            "action": action.action_type.value,
            "player_dist": contact.player_distance,
        })

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rally", type=str)
    parser.add_argument("--errors-only", action="store_true")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies found[/red]")
        return

    error_counts: dict[str, int] = defaultdict(int)
    crossed_net_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rally in rallies:
        if not rally.ball_positions_json:
            continue

        trace = trace_classification(rally)

        # Convert trace to pred_actions format for matching
        pred_actions = [
            {"frame": t["frame"], "action": t["action"]}
            for t in trace
        ]

        tolerance_frames = max(1, round(rally.fps * 167 / 1000))
        matches, unmatched = match_contacts(rally.gt_labels, pred_actions, tolerance=tolerance_frames)

        has_errors = any(
            m.pred_action is not None
            and m.gt_action != m.pred_action
            for m in matches
            if m.gt_action and m.pred_action
        )
        if args.errors_only and not has_errors:
            continue

        console.print(f"\n[bold]Rally {rally.rally_id[:8]}[/bold] net_y={rally.court_split_y or 0:.3f}")

        # Build a lookup from pred frame to trace entry
        trace_by_frame = {t["frame"]: t for t in trace}

        for m in matches:
            gt_action = m.gt_action
            pred_action = m.pred_action
            is_miss = pred_action is None or m.pred_frame is None
            is_error = not is_miss and gt_action != pred_action

            if args.errors_only and not is_error:
                continue

            color = "red" if is_error else ("yellow" if is_miss else "green")

            if is_miss:
                console.print(f"  [{color}]f{m.gt_frame:4d} GT={gt_action:8s} → MISS[/{color}]")
            else:
                t = trace_by_frame.get(m.pred_frame, {})
                crossed = t.get("crossed_net")
                crossed_str = {True: "YES", False: " NO", None: "n/a"}.get(crossed, "  ?")
                at_net = "NET" if t.get("is_at_net") else "   "
                pdist = t.get("player_dist", float("inf"))
                pdist_str = f"{pdist:.3f}" if pdist < float("inf") else "  inf"

                console.print(
                    f"  [{color}]f{m.gt_frame:4d} GT={gt_action:8s} → {pred_action:8s} "
                    f"| ball_y={t.get('ball_y', 0):.3f} side={t.get('court_side', '?'):4s} "
                    f"{at_net} cross={crossed_str} "
                    f"pdist={pdist_str}"
                    f"[/{color}]"
                )

                if is_error:
                    key = f"{gt_action}→{pred_action}"
                    error_counts[key] += 1
                    crossed_label = {True: "crossed_YES", False: "crossed_NO", None: "crossed_N/A"}.get(crossed, "?")
                    crossed_net_stats[key][crossed_label] += 1

    # Summary
    console.print("\n[bold]Error Summary (excluding MISS)[/bold]")
    for key, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        crossed_detail = ", ".join(f"{k}:{v}" for k, v in sorted(crossed_net_stats[key].items()))
        console.print(f"  {key:20s}: {count:3d}  ({crossed_detail})")


if __name__ == "__main__":
    main()
