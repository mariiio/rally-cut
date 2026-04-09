"""Diagnose: what accuracy would we get with perfect team knowledge?

For each contact, if we knew the GT player's team and picked the nearest
same-team candidate, how often would we get the correct player?
This gives the ceiling for continuous per-frame identity resolving teams.

Usage:
    cd analysis
    uv run python scripts/diagnose_oracle_team.py
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
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
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

    console.print(f"\n[bold]Oracle team diagnosis across {len(rallies)} rallies[/bold]\n")

    total = 0
    correct_proximity = 0
    correct_oracle_team = 0
    correct_oracle_identity = 0  # ceiling: oracle knows exact player

    # Breakdown
    cross_team_fixed = 0  # wrong-team error that oracle fixes
    cross_team_still_wrong = 0  # wrong-team error, oracle picks wrong same-team player
    within_team_errors = 0  # same-team, wrong player (oracle can't help with team alone)
    gt_not_in_candidates = 0

    per_action: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "prox": 0, "oracle": 0})

    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        cal = calibrators.get(rally.video_id)
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

        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=cal,
        )
        contacts = contact_seq.contacts
        contact_by_frame = {c.frame: c for c in contacts}

        rally_actions = classify_rally_actions(contact_seq, rally.rally_id)
        pred_actions_list = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions_list if not a.get("isSynthetic")]

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
            gt_action = next(
                (gt.action for gt in rally.gt_labels if gt.frame == m.gt_frame),
                "?",
            )
            if gt_tid < 0 or gt_tid not in avail_tids or gt_tid not in per_rally_teams:
                continue

            contact = contact_by_frame.get(m.pred_frame)
            if contact is None or not contact.player_candidates:
                continue

            gt_team = per_rally_teams[gt_tid]
            pred_tid = contact.player_candidates[0][0]
            pred_team = per_rally_teams.get(pred_tid)

            total += 1
            per_action[gt_action]["total"] += 1

            # Proximity baseline
            if pred_tid == gt_tid:
                correct_proximity += 1
                correct_oracle_team += 1
                correct_oracle_identity += 1
                per_action[gt_action]["prox"] += 1
                per_action[gt_action]["oracle"] += 1
                continue

            # Error — try oracle team
            # Find nearest same-team candidate
            same_team_cands = [
                (tid, d) for tid, d in contact.player_candidates
                if per_rally_teams.get(tid) == gt_team
            ]

            # Oracle identity: is GT in candidates at all?
            gt_in_cands = any(tid == gt_tid for tid, _ in contact.player_candidates)
            if gt_in_cands:
                correct_oracle_identity += 1

            if same_team_cands:
                oracle_tid = same_team_cands[0][0]  # nearest on correct team
                if oracle_tid == gt_tid:
                    cross_team_fixed += 1
                    correct_oracle_team += 1
                    per_action[gt_action]["oracle"] += 1
                elif pred_team != gt_team:
                    # Cross-team but oracle picks wrong same-team player
                    cross_team_still_wrong += 1
                else:
                    # Same team, wrong player — oracle team doesn't help
                    within_team_errors += 1
            else:
                gt_not_in_candidates += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts")

    # Results
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]")
    console.print(f"  Proximity baseline: {correct_proximity}/{total} = {correct_proximity/max(1,total):.1%}")
    console.print(f"  Oracle team:        {correct_oracle_team}/{total} = {correct_oracle_team/max(1,total):.1%}")
    console.print(f"  Oracle identity:    {correct_oracle_identity}/{total} = {correct_oracle_identity/max(1,total):.1%}")

    errors = total - correct_proximity
    console.print(f"\n[bold cyan]Error decomposition ({errors} errors):[/bold cyan]")
    console.print(f"  Cross-team, oracle fixes:            {cross_team_fixed}")
    console.print(f"  Cross-team, oracle same-team wrong:  {cross_team_still_wrong}")
    console.print(f"  Within-team (oracle team can't fix): {within_team_errors}")
    console.print(f"  GT not in any candidate:             {gt_not_in_candidates}")

    # Per-action breakdown
    act_table = Table(title="Per-action accuracy")
    act_table.add_column("Action")
    act_table.add_column("Total")
    act_table.add_column("Proximity")
    act_table.add_column("Oracle Team")
    for action in ["serve", "receive", "set", "attack", "dig", "block"]:
        a = per_action[action]
        if a["total"] == 0:
            continue
        act_table.add_row(
            action,
            str(a["total"]),
            f"{a['prox']/a['total']:.1%}",
            f"{a['oracle']/a['total']:.1%}",
        )
    console.print(act_table)

    console.print(f"\n[bold green]Improvement from oracle team: +{correct_oracle_team - correct_proximity} contacts (+{(correct_oracle_team - correct_proximity)/max(1,total):.1%})[/bold green]")
    console.print(f"[bold green]Remaining gap to oracle identity: {correct_oracle_identity - correct_oracle_team} contacts (within-team)[/bold green]")


if __name__ == "__main__":
    main()
