"""Validate post-hoc team violation correction.

After the full pipeline runs (contact detection + action classification),
check the action sequence for volleyball rule violations:
1. After serve/attack, next contact must be on opposite team
2. After receive/set/dig, next contact must be on same team
3. Same player touching twice consecutively (same team) is rare

When a violation is detected, re-attribute the violating contact to the
best candidate on the expected team.

Also test: using candidate #1 (perspective-corrected) as base, then
applying post-hoc corrections.

Usage:
    cd analysis
    uv run python scripts/validate_posthoc_correction.py
"""

from __future__ import annotations

import argparse
import math
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
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

# Actions that cross the net (next contact is on opposite team)
NET_CROSSING = {"serve", "attack"}
# Actions that stay on same team
SAME_SIDE = {"receive", "set", "dig"}


def posthoc_correct(
    actions: list[dict],
    contacts: list[Contact],
    per_rally_teams: dict[int, int],
    use_cand0: bool = False,
) -> list[dict]:
    """Apply post-hoc corrections to player attribution.

    For each pair of consecutive actions, check if the team transition
    follows volleyball rules. If not, re-attribute the second contact
    to a player on the expected team.

    Args:
        actions: Action dicts from classify_rally_actions.
        contacts: Contact objects with player_candidates.
        per_rally_teams: track_id -> team (0=near, 1=far).
        use_cand0: If True, start from candidate #1 instead of current.
    """
    import copy

    actions = copy.deepcopy(actions)
    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

    # Optionally switch to candidate #1
    if use_cand0:
        for action in actions:
            frame = action.get("frame")
            contact = contact_by_frame.get(frame)
            if contact and contact.player_candidates:
                action["playerTrackId"] = contact.player_candidates[0][0]

    # Apply team-transition corrections
    for i in range(len(actions) - 1):
        curr = actions[i]
        nxt = actions[i + 1]

        curr_action = curr.get("action", "")
        curr_tid = curr.get("playerTrackId", -1)
        nxt_tid = nxt.get("playerTrackId", -1)
        nxt_frame = nxt.get("frame")

        if curr_tid < 0 or nxt_tid < 0:
            continue

        curr_team = per_rally_teams.get(curr_tid)
        nxt_team = per_rally_teams.get(nxt_tid)

        if curr_team is None or nxt_team is None:
            continue

        # Determine expected team for next contact
        if curr_action in NET_CROSSING:
            expected_team = 1 - curr_team  # Opposite
        elif curr_action in SAME_SIDE:
            expected_team = curr_team  # Same
        else:
            continue  # Unknown action, skip

        # Check if violation
        if nxt_team == expected_team:
            continue  # No violation

        # Violation detected — re-attribute next contact
        contact = contact_by_frame.get(nxt_frame)
        if contact is None or not contact.player_candidates:
            continue

        # Find best candidate on expected team
        for cand_tid, cand_dist in contact.player_candidates:
            if per_rally_teams.get(cand_tid) == expected_team:
                nxt["playerTrackId"] = cand_tid
                break

    # Second pass: same-player consecutive on same team → swap to teammate
    for i in range(len(actions) - 1):
        curr = actions[i]
        nxt = actions[i + 1]

        curr_tid = curr.get("playerTrackId", -1)
        nxt_tid = nxt.get("playerTrackId", -1)
        nxt_frame = nxt.get("frame")

        if curr_tid < 0 or nxt_tid < 0 or curr_tid != nxt_tid:
            continue

        curr_team = per_rally_teams.get(curr_tid)
        nxt_team = per_rally_teams.get(nxt_tid)

        if curr_team is None or curr_team != nxt_team:
            continue  # Different teams (or unknown)

        # Same player, same team, consecutive → swap to teammate
        contact = contact_by_frame.get(nxt_frame)
        if contact is None or not contact.player_candidates:
            continue

        for cand_tid, cand_dist in contact.player_candidates:
            if cand_tid != nxt_tid and per_rally_teams.get(cand_tid) == curr_team:
                nxt["playerTrackId"] = cand_tid
                break

    return actions


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

    console.print(f"\n[bold]Validating post-hoc corrections across {len(rallies)} rallies[/bold]\n")

    total = 0
    cur_correct = 0
    posthoc_cur_correct = 0  # Post-hoc on current
    posthoc_cand0_correct = 0  # Post-hoc on candidate #1
    cand0_correct = 0
    # Error analysis
    n_violations_fixed = 0
    n_consec_fixed = 0

    per_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

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

        rally_actions = classify_rally_actions(contact_seq, rally.rally_id)
        pred_actions_list = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions_list if not a.get("isSynthetic")]

        # Apply post-hoc corrections
        corrected_cur = posthoc_correct(
            real_pred, contacts, per_rally_teams, use_cand0=False,
        )
        corrected_cand0 = posthoc_correct(
            real_pred, contacts, per_rally_teams, use_cand0=True,
        )

        fps = rally.fps or 30.0
        tol = max(1, round(fps * args.tolerance_ms / 1000))
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

        # Match original
        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

        # Build corrected lookups
        corrected_cur_by_frame = {a.get("frame"): a for a in corrected_cur}
        corrected_cand0_by_frame = {a.get("frame"): a for a in corrected_cand0}

        for m in matches:
            if m.pred_frame is None:
                continue

            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            if gt_tid < 0 or gt_tid not in avail_tids or gt_tid not in per_rally_teams:
                continue

            total += 1

            # Current
            pred_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )
            if pred_tid == gt_tid:
                cur_correct += 1

            # Candidate #1
            contact_by_frame = {c.frame: c for c in contacts}
            contact = contact_by_frame.get(m.pred_frame)
            if contact and contact.player_candidates:
                c0_tid = contact.player_candidates[0][0]
                if c0_tid == gt_tid:
                    cand0_correct += 1

            # Post-hoc on current
            ph_cur = corrected_cur_by_frame.get(m.pred_frame)
            if ph_cur and ph_cur.get("playerTrackId") == gt_tid:
                posthoc_cur_correct += 1

            # Post-hoc on cand0
            ph_cand0 = corrected_cand0_by_frame.get(m.pred_frame)
            if ph_cand0 and ph_cand0.get("playerTrackId") == gt_tid:
                posthoc_cand0_correct += 1

            # Per-action
            per_action[m.gt_action]["total"] += 1
            if pred_tid == gt_tid:
                per_action[m.gt_action]["cur"] += 1
            if ph_cur and ph_cur.get("playerTrackId") == gt_tid:
                per_action[m.gt_action]["ph_cur"] += 1
            if ph_cand0 and ph_cand0.get("playerTrackId") == gt_tid:
                per_action[m.gt_action]["ph_cand0"] += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts")

    # Report
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]\n")
    console.print(f"  Current system:              {cur_correct}/{total} = {cur_correct / max(1, total):.1%}")
    console.print(f"  Candidate #1:                {cand0_correct}/{total} = {cand0_correct / max(1, total):.1%}")
    console.print(f"  Post-hoc on current:         {posthoc_cur_correct}/{total} = {posthoc_cur_correct / max(1, total):.1%}")
    console.print(f"  Post-hoc on candidate #1:    {posthoc_cand0_correct}/{total} = {posthoc_cand0_correct / max(1, total):.1%}")

    # Per-action
    console.print(f"\n[bold cyan]Per-action breakdown[/bold cyan]")
    act_table = Table()
    act_table.add_column("Action")
    act_table.add_column("Current", justify="right")
    act_table.add_column("PostHoc+Cur", justify="right")
    act_table.add_column("PostHoc+Cand0", justify="right")
    act_table.add_column("Total", justify="right")

    for action in ["serve", "receive", "set", "attack", "block", "dig"]:
        cc = per_action[action].get("cur", 0)
        phc = per_action[action].get("ph_cur", 0)
        phca = per_action[action].get("ph_cand0", 0)
        t = per_action[action].get("total", 0)
        if t > 0:
            act_table.add_row(
                action,
                f"{cc} ({cc / t:.0%})",
                f"{phc} ({phc / t:.0%})",
                f"{phca} ({phca / t:.0%})",
                str(t),
            )
    console.print(act_table)


if __name__ == "__main__":
    main()
