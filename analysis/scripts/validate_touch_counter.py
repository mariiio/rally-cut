"""Validate a volleyball-rules touch counter for team assignment.

Instead of using player-identity-based court_side (circular dependency),
infer which team should touch the ball from the volleyball sequence:
1. Identify serve → server's team from per-rally classify_teams
2. Count touches: 1,2,3 → flip to other team → 1,2,3 → flip
3. Blocks are special: opponent touches, ball stays on attacking side

Compare this sequence-based team assignment against GT player's actual team.

Usage:
    cd analysis
    uv run python scripts/validate_touch_counter.py
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


def infer_teams_from_sequence(
    contacts: list[Contact],
    per_rally_teams: dict[int, int],
) -> dict[int, int]:
    """Infer expected team for each contact frame using serve-seeded touch counter.

    Returns: dict of frame -> expected_team (0=near, 1=far)
    """
    if not contacts:
        return {}

    result: dict[int, int] = {}

    # Find serve: first contact (detect_contacts puts serve first)
    # Determine server's team from per-rally teams
    serve_contact = contacts[0]
    server_tid = serve_contact.player_track_id
    serving_team = per_rally_teams.get(server_tid)

    if serving_team is None:
        # Can't determine serving team — fall back to Y-based heuristic
        # Server is at baseline: near-side server has high Y, far-side has low Y
        # Use ball Y at serve as proxy
        # Actually, use player candidates to find any player with known team
        for tid, dist in serve_contact.player_candidates:
            if tid in per_rally_teams:
                serving_team = per_rally_teams[tid]
                break
        if serving_team is None:
            return {}

    current_team = serving_team
    touch_count = 0

    for contact in contacts:
        touch_count += 1

        # Safety valve: max 3 touches per side
        if touch_count > 3:
            current_team = 1 - current_team
            touch_count = 1

        result[contact.frame] = current_team

        # After 3rd touch, flip
        if touch_count == 3:
            current_team = 1 - current_team
            touch_count = 0

    return result


def infer_teams_with_block_awareness(
    contacts: list[Contact],
    per_rally_teams: dict[int, int],
    net_y: float,
) -> dict[int, int]:
    """Same as infer_teams_from_sequence but handles blocks.

    A block happens when:
    - Contact is near the net (is_at_net or ball_y close to net_y)
    - It's the 4th+ touch (touch_count would exceed 3 on same side)
    - After block, ball stays on the attacker's side (no team flip)

    Returns: dict of frame -> expected_team (0=near, 1=far)
    """
    if not contacts:
        return {}

    result: dict[int, int] = {}

    serve_contact = contacts[0]
    server_tid = serve_contact.player_track_id
    serving_team = per_rally_teams.get(server_tid)

    if serving_team is None:
        for tid, dist in serve_contact.player_candidates:
            if tid in per_rally_teams:
                serving_team = per_rally_teams[tid]
                break
        if serving_team is None:
            return {}

    current_team = serving_team
    touch_count = 0

    for i, contact in enumerate(contacts):
        touch_count += 1

        # Detect possible block: touch exceeds 3, contact near net
        is_near_net = contact.is_at_net or abs(contact.ball_y - net_y) < 0.08
        if touch_count > 3 and is_near_net:
            # This is a block by the other team
            # The blocker's team is the current attacking team's opponent
            block_team = 1 - current_team
            result[contact.frame] = block_team
            # After block, ball returns to attacking team's side
            # touch_count resets for the NEW rally on the attacking team's side
            current_team = 1 - block_team  # back to the team that just attacked
            touch_count = 0
            continue

        # Safety valve: if still > 3, assume missed contact — flip
        if touch_count > 3:
            current_team = 1 - current_team
            touch_count = 1

        result[contact.frame] = current_team

        # After 3rd touch, flip
        if touch_count == 3:
            current_team = 1 - current_team
            touch_count = 0

    return result


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

    console.print(f"\n[bold]Validating touch counter team assignment across {len(rallies)} rallies[/bold]\n")

    # Counters
    total = 0
    current_correct = 0
    current_team_correct = 0  # Predicted player on same team as GT
    tc_correct = 0  # Touch counter (simple)
    tc_block_correct = 0  # Touch counter with block awareness
    tc_reattr = 0  # Touch counter + nearest on team
    tc_block_reattr = 0
    # Per-action
    per_action_tc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_action_tc_reattr: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_action_current: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Error analysis
    tc_team_correct_gt_nearest = 0
    tc_team_correct_gt_not_nearest = 0
    tc_team_wrong = 0
    # Cross-team vs within-team errors
    cross_team_errors = 0
    within_team_errors = 0

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
        net_y = contact_seq.net_y
        contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

        # Infer teams from touch counter
        tc_teams = infer_teams_from_sequence(contacts, per_rally_teams)
        tc_block_teams = infer_teams_with_block_awareness(
            contacts, per_rally_teams, net_y,
        )

        # Classify actions for baseline
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
            if gt_tid < 0 or gt_tid not in avail_tids or gt_tid not in per_rally_teams:
                continue

            contact = contact_by_frame.get(m.pred_frame)
            if contact is None:
                continue

            gt_team = per_rally_teams[gt_tid]
            pred_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )

            total += 1
            if pred_tid == gt_tid:
                current_correct += 1

            # Current system team accuracy
            pred_team = per_rally_teams.get(pred_tid)
            if pred_team is not None and pred_team == gt_team:
                current_team_correct += 1
            per_action_current[m.gt_action]["total"] += 1
            if pred_team == gt_team:
                per_action_current[m.gt_action]["team_correct"] += 1
            if pred_tid == gt_tid:
                per_action_current[m.gt_action]["player_correct"] += 1

            # Cross-team vs within-team error analysis
            if pred_tid != gt_tid:
                if pred_team is not None and pred_team != gt_team:
                    cross_team_errors += 1
                else:
                    within_team_errors += 1

            # Touch counter team accuracy
            tc_team = tc_teams.get(m.pred_frame)
            tc_block_team = tc_block_teams.get(m.pred_frame)

            if tc_team is not None:
                if tc_team == gt_team:
                    tc_correct += 1
                per_action_tc[m.gt_action]["total"] += 1
                if tc_team == gt_team:
                    per_action_tc[m.gt_action]["correct"] += 1

            if tc_block_team is not None:
                if tc_block_team == gt_team:
                    tc_block_correct += 1

            # Simulated reattribution with touch counter
            if tc_team is not None and contact.player_candidates:
                same_team = [
                    (tid, dist) for tid, dist in contact.player_candidates
                    if per_rally_teams.get(tid) == tc_team
                ]
                if same_team and same_team[0][0] == gt_tid:
                    tc_reattr += 1
                per_action_tc_reattr[m.gt_action]["total"] += 1
                if same_team and same_team[0][0] == gt_tid:
                    per_action_tc_reattr[m.gt_action]["correct"] += 1

                # Error analysis
                if tc_team == gt_team:
                    if same_team and same_team[0][0] == gt_tid:
                        tc_team_correct_gt_nearest += 1
                    else:
                        tc_team_correct_gt_not_nearest += 1
                else:
                    tc_team_wrong += 1

            if tc_block_team is not None and contact.player_candidates:
                same_team_b = [
                    (tid, dist) for tid, dist in contact.player_candidates
                    if per_rally_teams.get(tid) == tc_block_team
                ]
                if same_team_b and same_team_b[0][0] == gt_tid:
                    tc_block_reattr += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts processed")

    # Report
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]\n")
    console.print(f"  Current attribution: {current_correct}/{total} = {current_correct / max(1, total):.1%}")
    console.print(f"  Current team accuracy: {current_team_correct}/{total} = {current_team_correct / max(1, total):.1%}")
    console.print(f"  Cross-team errors: {cross_team_errors}, Within-team errors: {within_team_errors}")

    # Per-action current system team accuracy
    console.print(f"\n[bold magenta]Current system per-action[/bold magenta]")
    cur_table = Table()
    cur_table.add_column("Action")
    cur_table.add_column("Player Correct", justify="right")
    cur_table.add_column("Team Correct", justify="right")
    cur_table.add_column("Total", justify="right")
    cur_table.add_column("Player Acc", justify="right")
    cur_table.add_column("Team Acc", justify="right")
    for action in ["serve", "receive", "set", "attack", "block", "dig"]:
        pc = per_action_current[action].get("player_correct", 0)
        tc_c = per_action_current[action].get("team_correct", 0)
        t = per_action_current[action].get("total", 0)
        if t > 0:
            cur_table.add_row(
                action, str(pc), str(tc_c), str(t),
                f"{pc / t:.1%}", f"{tc_c / t:.1%}",
            )
    console.print(cur_table)
    console.print(f"\n[bold cyan]Touch counter team accuracy[/bold cyan]")
    console.print(f"  Simple (3-touch flip): {tc_correct}/{total} = {tc_correct / max(1, total):.1%}")
    console.print(f"  Block-aware: {tc_block_correct}/{total} = {tc_block_correct / max(1, total):.1%}")

    console.print(f"\n[bold green]Simulated reattribution (nearest on inferred team)[/bold green]")
    console.print(f"  Simple TC + nearest on team: {tc_reattr}/{total} = {tc_reattr / max(1, total):.1%}")
    console.print(f"  Block-aware TC + nearest: {tc_block_reattr}/{total} = {tc_block_reattr / max(1, total):.1%}")

    console.print(f"\n[bold yellow]Error breakdown (simple TC)[/bold yellow]")
    console.print(f"  TC team correct + GT nearest on team: {tc_team_correct_gt_nearest}")
    console.print(f"  TC team correct + GT NOT nearest: {tc_team_correct_gt_not_nearest}")
    console.print(f"  TC team wrong: {tc_team_wrong}")

    # Per-action
    console.print(f"\n[bold cyan]Per-action touch counter team accuracy[/bold cyan]")
    act_table = Table()
    act_table.add_column("Action")
    act_table.add_column("TC Team Correct", justify="right")
    act_table.add_column("Total", justify="right")
    act_table.add_column("Team Acc", justify="right")
    act_table.add_column("Reattr Correct", justify="right")
    act_table.add_column("Reattr Acc", justify="right")

    for action in ["serve", "receive", "set", "attack", "block", "dig"]:
        tc_c = per_action_tc[action].get("correct", 0)
        tc_t = per_action_tc[action].get("total", 0)
        ra_c = per_action_tc_reattr[action].get("correct", 0)
        ra_t = per_action_tc_reattr[action].get("total", 0)
        if tc_t > 0:
            act_table.add_row(
                action, str(tc_c), str(tc_t),
                f"{tc_c / tc_t:.1%}",
                str(ra_c),
                f"{ra_c / ra_t:.1%}" if ra_t > 0 else "-",
            )
    console.print(act_table)


if __name__ == "__main__":
    main()
