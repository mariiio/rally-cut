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
    ActionClassifier,
    ActionClassifierConfig,
    _ball_crossed_net,
)
from rallycut.tracking.contact_detector import detect_contacts

console = Console()


def trace_classification(rally) -> list[dict]:
    """Run contact detection and trace classification state machine."""
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
    contacts = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
        classifier=classifier,
    )

    # Trace through classification manually
    net_y = contacts.net_y
    bp_list = contacts.ball_positions
    contact_list = contacts.contacts

    start_frame = contact_list[0].frame if contact_list else 0
    ac = ActionClassifier()
    serve_index = ac._find_serve_index(
        contact_list, start_frame, net_y, bp_list or None
    )

    trace = []
    serve_detected = False
    receive_detected = False
    serve_side = None
    current_side = None
    contact_count_on_side = 0

    for i, contact in enumerate(contact_list):
        crossed_net = None
        if bp_list and i > 0 and current_side is not None:
            crossed_net = _ball_crossed_net(
                bp_list,
                from_frame=contact_list[i - 1].frame,
                to_frame=contact.frame,
                net_y=net_y,
            )

        prev_side = current_side
        prev_count = contact_count_on_side

        # Pre-serve isolation
        if not serve_detected and i != serve_index:
            trace.append({
                "i": i, "frame": contact.frame,
                "ball_y": contact.ball_y, "court_side": contact.court_side,
                "is_at_net": contact.is_at_net,
                "crossed_net": crossed_net,
                "contact_count": 0,
                "action": "unknown", "reason": "pre-serve",
                "current_side": current_side,
            })
            continue

        # Possession tracking (mirrors action_classifier.py lines 349-376)
        if crossed_net is True:
            current_side = contact.court_side
            contact_count_on_side = 0
        elif contact.court_side != current_side:
            if crossed_net is False:
                if contact_count_on_side >= 4:
                    current_side = contact.court_side
                    contact_count_on_side = 0
                # else: trust trajectory, no reset
            else:
                # None or first contact
                current_side = contact.court_side
                contact_count_on_side = 0

        contact_count_on_side += 1

        # Classification
        action = "unknown"
        if not serve_detected:
            if i == serve_index:
                action = "serve"
                serve_detected = True
                serve_side = contact.court_side
                current_side = contact.court_side
                contact_count_on_side = 1
        elif (
            not receive_detected
            and serve_side is not None
            and (contact.court_side != serve_side or crossed_net is True)
        ):
            action = "receive"
            receive_detected = True
            current_side = contact.court_side
            contact_count_on_side = 1
        elif contact_count_on_side == 1:
            action = "dig"
        elif contact_count_on_side == 2:
            action = "set"
        elif contact_count_on_side >= 3:
            action = "spike"

        trace.append({
            "i": i, "frame": contact.frame,
            "ball_y": contact.ball_y, "court_side": contact.court_side,
            "is_at_net": contact.is_at_net,
            "crossed_net": crossed_net,
            "prev_side": prev_side, "prev_count": prev_count,
            "contact_count": contact_count_on_side,
            "action": action,
            "current_side": current_side,
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
                prev_side = t.get("prev_side", "?")
                prev_cnt = t.get("prev_count", "?")

                console.print(
                    f"  [{color}]f{m.gt_frame:4d} GT={gt_action:8s} → {pred_action:8s} "
                    f"| ball_y={t.get('ball_y', 0):.3f} side={t.get('court_side', '?'):4s} "
                    f"{at_net} cross={crossed_str} "
                    f"cnt={t.get('contact_count', '?')} "
                    f"(prev: {prev_side}/{prev_cnt}) "
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
