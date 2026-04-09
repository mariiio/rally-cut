"""Validate team-constrained reattribution ceiling.

Measures:
1. Propagated court_side accuracy (after action classification) against GT
2. Simulated uncapped team-constrained reattribution accuracy
3. Compares per-rally teams (classify_teams) vs match_team_assignments

Usage:
    cd analysis
    uv run python scripts/validate_reattribution_ceiling.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rally", type=str)
    parser.add_argument("--tolerance-ms", type=int, default=167)
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT.[/red]")
        return

    calibrators: dict[str, CourtCalibrator | None] = {}
    video_ids = {r.video_id for r in rallies}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    match_teams_by_rally = _load_match_team_assignments(video_ids, min_confidence=0.70)

    console.print(f"\n[bold]Validating reattribution ceiling across {len(rallies)} rallies[/bold]\n")

    # Counters
    total = 0
    current_correct = 0  # Current system attribution
    # Propagated court_side accuracy
    prop_cs_correct_perally = 0
    prop_cs_correct_match = 0
    prop_cs_total = 0
    # Simulated reattribution
    reattr_perally_uncapped = 0
    reattr_perally_capped15 = 0
    reattr_match_uncapped = 0
    # Per-action breakdown
    per_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        cal = calibrators.get(rally.video_id)
        match_teams = match_teams_by_rally.get(rally.rally_id)

        positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"], y=pp["y"],
                width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        net_y_val = rally.court_split_y or 0.5
        per_rally_teams = classify_teams(positions, net_y_val)

        # Run contact detection (no team_assignments — let it use unconstrained)
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=cal,
        )
        contacts = contact_seq.contacts
        contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

        # Classify actions with match teams (current system)
        rally_actions = classify_rally_actions(
            contact_seq, rally.rally_id,
            match_team_assignments=match_teams,
        )
        pred_actions_list = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions_list if not a.get("isSynthetic")]
        action_by_frame = {a.get("frame"): a for a in real_pred}

        fps = rally.fps or 30.0
        tol = max(1, round(fps * args.tolerance_ms / 1000))
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

        for m in matches:
            if m.pred_frame is None:
                continue

            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            if gt_tid < 0 or gt_tid not in avail_tids:
                continue

            pred_action = action_by_frame.get(m.pred_frame)
            if pred_action is None:
                continue

            pred_tid = pred_action.get("playerTrackId", -1)
            pred_cs = pred_action.get("courtSide", "unknown")
            gt_action = m.gt_action

            contact = contact_by_frame.get(m.pred_frame)
            if contact is None:
                continue

            total += 1

            # Current system attribution
            if pred_tid == gt_tid:
                current_correct += 1

            # Propagated court_side accuracy
            if pred_cs in ("near", "far"):
                prop_cs_total += 1
                # Per-rally teams: team 0 = near, team 1 = far
                if gt_tid in per_rally_teams:
                    gt_team_pr = per_rally_teams[gt_tid]
                    expected_cs_pr = "near" if gt_team_pr == 0 else "far"
                    if pred_cs == expected_cs_pr:
                        prop_cs_correct_perally += 1

                # Match teams
                if match_teams and gt_tid in match_teams:
                    gt_team_mt = match_teams[gt_tid]
                    expected_cs_mt = "near" if gt_team_mt == 0 else "far"
                    if pred_cs == expected_cs_mt:
                        prop_cs_correct_match += 1

            # Simulated reattribution: use propagated court_side + teams
            # to pick best candidate on the expected team
            if pred_cs in ("near", "far") and contact.player_candidates:
                # With per-rally teams, uncapped
                expected_team_pr = 0 if pred_cs == "near" else 1
                same_team_pr = [
                    (tid, dist) for tid, dist in contact.player_candidates
                    if per_rally_teams.get(tid) == expected_team_pr
                ]
                if same_team_pr:
                    best_pr = same_team_pr[0][0]
                    if best_pr == gt_tid:
                        reattr_perally_uncapped += 1
                    per_action[gt_action]["pr_uncapped_total"] += 1
                    if best_pr == gt_tid:
                        per_action[gt_action]["pr_uncapped_correct"] += 1

                    # With 1.5x cap
                    current_dist = contact.player_distance
                    capped_pr = [
                        (tid, dist) for tid, dist in same_team_pr
                        if dist <= 1.5 * current_dist or tid == pred_tid
                    ]
                    if capped_pr:
                        best_capped = capped_pr[0][0]
                        if best_capped == gt_tid:
                            reattr_perally_capped15 += 1

                # With match teams, uncapped
                if match_teams:
                    expected_team_mt = 0 if pred_cs == "near" else 1
                    same_team_mt = [
                        (tid, dist) for tid, dist in contact.player_candidates
                        if match_teams.get(tid) == expected_team_mt
                    ]
                    if same_team_mt:
                        best_mt = same_team_mt[0][0]
                        if best_mt == gt_tid:
                            reattr_match_uncapped += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts processed")

    # Report
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]\n")

    console.print(f"  Current attribution accuracy: {current_correct}/{total} = {current_correct / max(1, total):.1%}")

    console.print(f"\n[bold cyan]Propagated court_side accuracy[/bold cyan]")
    console.print(f"  vs per-rally teams: {prop_cs_correct_perally}/{prop_cs_total} = {prop_cs_correct_perally / max(1, prop_cs_total):.1%}")
    console.print(f"  vs match teams: {prop_cs_correct_match}/{prop_cs_total} = {prop_cs_correct_match / max(1, prop_cs_total):.1%}")

    console.print(f"\n[bold green]Simulated reattribution accuracy[/bold green]")
    console.print(f"  Per-rally teams, uncapped: {reattr_perally_uncapped}/{total} = {reattr_perally_uncapped / max(1, total):.1%}")
    console.print(f"  Per-rally teams, 1.5x cap: {reattr_perally_capped15}/{total} = {reattr_perally_capped15 / max(1, total):.1%}")
    console.print(f"  Match teams, uncapped: {reattr_match_uncapped}/{total} = {reattr_match_uncapped / max(1, total):.1%}")

    # Per-action breakdown for per-rally uncapped
    console.print(f"\n[bold cyan]Per-action breakdown (per-rally teams, uncapped)[/bold cyan]")
    act_table = Table()
    act_table.add_column("Action")
    act_table.add_column("Correct", justify="right")
    act_table.add_column("Total", justify="right")
    act_table.add_column("Accuracy", justify="right")

    for action in ["serve", "receive", "set", "attack", "block", "dig"]:
        ac = per_action[action].get("pr_uncapped_correct", 0)
        at = per_action[action].get("pr_uncapped_total", 0)
        if at > 0:
            act_table.add_row(action, str(ac), str(at), f"{ac / at:.1%}")
    console.print(act_table)


if __name__ == "__main__":
    main()
