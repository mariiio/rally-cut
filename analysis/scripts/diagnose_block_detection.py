"""Diagnose block detection failures.

For each GT block contact, checks which block heuristic conditions fail:
1. contact.is_at_net
2. last_action_type == ATTACK
3. frame gap <= 8 from previous contact
4. court_side != previous contact's court_side

Reports whether the block was even detected as a contact (FN vs misclass).

Usage:
    cd analysis
    uv run python scripts/diagnose_block_detection.py
"""

from __future__ import annotations

import argparse
from collections import Counter

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

BLOCK_MAX_FRAME_GAP = 8  # From ActionClassifierConfig default


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose block detection")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument(
        "--tolerance-ms", type=int, default=167,
        help="Time tolerance in ms for matching (default: 167)",
    )
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    # Filter to rallies that have at least one block GT
    block_rallies = [
        r for r in rallies
        if any(gt.action == "block" for gt in r.gt_labels)
    ]

    if not block_rallies:
        console.print("[yellow]No rallies with block GT labels found.[/yellow]")
        return

    # Load calibrators and match teams
    calibrators: dict[str, CourtCalibrator | None] = {}
    video_ids = {r.video_id for r in block_rallies}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None
    match_teams_by_rally = _load_match_team_assignments(video_ids, min_confidence=0.70)

    total_gt_blocks = sum(
        1 for r in block_rallies for gt in r.gt_labels if gt.action == "block"
    )
    console.print(
        f"\n[bold]Block Detection Diagnosis: {total_gt_blocks} GT blocks "
        f"across {len(block_rallies)} rallies[/bold]\n"
    )

    results_table = Table(title="Block Detection Per-GT-Contact")
    results_table.add_column("Rally", style="dim", max_width=8)
    results_table.add_column("GT Frame", justify="right")
    results_table.add_column("Detected?", justify="center")
    results_table.add_column("Pred Action", max_width=8)
    results_table.add_column("is_at_net", justify="center")
    results_table.add_column("prev_attack", justify="center")
    results_table.add_column("frame_gap", justify="right")
    results_table.add_column("side_changed", justify="center")
    results_table.add_column("Failing Conditions")

    detected_as_block = 0
    detected_wrong_action = 0
    not_detected = 0
    condition_failures: Counter[str] = Counter()

    for rally in block_rallies:
        if not rally.ball_positions_json:
            continue

        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        player_positions: list[PlayerPos] = []
        if rally.positions_json:
            player_positions = [
                PlayerPos(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in rally.positions_json
            ]

        contacts = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=calibrators.get(rally.video_id),
        )

        match_teams = match_teams_by_rally.get(rally.rally_id)
        rally_actions = classify_rally_actions(
            contacts, rally.rally_id,
            match_team_assignments=match_teams,
        )
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

        tolerance_frames = max(1, round(rally.fps * args.tolerance_ms / 1000))
        matches, _ = match_contacts(
            rally.gt_labels, real_pred, tolerance=tolerance_frames,
        )

        contact_list = contacts.contacts

        for m in matches:
            if m.gt_action != "block":
                continue

            if m.pred_frame is None:
                # Contact-level FN — block never detected as a contact
                not_detected += 1
                results_table.add_row(
                    rally.rally_id[:8],
                    str(m.gt_frame),
                    "NO",
                    "-",
                    "-", "-", "-", "-",
                    "CONTACT FN (not detected at all)",
                )
                continue

            # Contact was detected — check if classified as block
            if m.pred_action == "block":
                detected_as_block += 1
                results_table.add_row(
                    rally.rally_id[:8],
                    str(m.gt_frame),
                    "YES",
                    "block",
                    "-", "-", "-", "-",
                    "(correct)",
                )
                continue

            # Misclassified — check which block conditions fail
            detected_wrong_action += 1

            # Find the contact in the contact list
            pred_contact_idx = None
            for ci, c in enumerate(contact_list):
                if c.frame == m.pred_frame:
                    pred_contact_idx = ci
                    break

            if pred_contact_idx is None:
                results_table.add_row(
                    rally.rally_id[:8],
                    str(m.gt_frame),
                    "YES",
                    m.pred_action or "?",
                    "?", "?", "?", "?",
                    "contact not found in list",
                )
                continue

            contact = contact_list[pred_contact_idx]

            # Find previous action type from classified actions
            prev_action_type = None
            prev_frame = None
            prev_court_side = None
            if pred_contact_idx > 0:
                prev_c = contact_list[pred_contact_idx - 1]
                prev_frame = prev_c.frame
                prev_court_side = prev_c.court_side
                # Find the action for the previous contact
                for a in rally_actions.actions:
                    if a.frame == prev_c.frame:
                        prev_action_type = a.action_type.value
                        break

            # Check conditions
            is_at_net = contact.is_at_net
            frame_gap = (contact.frame - prev_frame) if prev_frame is not None else 999
            side_changed = (
                contact.court_side != prev_court_side
                if prev_court_side is not None
                else False
            )
            prev_is_attack = prev_action_type == "attack"

            failures: list[str] = []
            if not is_at_net:
                failures.append("not_at_net")
                condition_failures["not_at_net"] += 1
            if not prev_is_attack:
                failures.append(f"prev={prev_action_type or 'none'}")
                condition_failures["prev_not_attack"] += 1
            if frame_gap > BLOCK_MAX_FRAME_GAP:
                failures.append(f"gap={frame_gap}>8")
                condition_failures["frame_gap_too_large"] += 1
            if not side_changed:
                failures.append("same_side")
                condition_failures["same_side"] += 1

            results_table.add_row(
                rally.rally_id[:8],
                str(m.gt_frame),
                "YES",
                m.pred_action or "?",
                "Y" if is_at_net else "N",
                "Y" if prev_is_attack else f"N({prev_action_type or '?'})",
                str(frame_gap),
                "Y" if side_changed else "N",
                ", ".join(failures) if failures else "(all pass but not classified?)",
            )

    console.print(results_table)

    # Summary
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  GT blocks:              {total_gt_blocks}")
    console.print(f"  Detected as block:      {detected_as_block} ({detected_as_block / max(1, total_gt_blocks):.0%})")
    console.print(f"  Detected, wrong action: {detected_wrong_action}")
    console.print(f"  Not detected (FN):      {not_detected}")

    if condition_failures:
        console.print(f"\n[bold]Condition failures (among misclassified)[/bold]")
        for cond, count in condition_failures.most_common():
            console.print(f"  {cond}: {count}")


if __name__ == "__main__":
    main()
