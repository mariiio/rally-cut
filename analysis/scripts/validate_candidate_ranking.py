"""Validate using perspective-corrected candidate ranking for player selection.

Compares three attribution strategies:
1. Current: _find_nearest_player (narrow ±5 frame, image-space)
2. Candidate #1: player_candidates[0] (wide ±15 frame, perspective-corrected)
3. Hybrid: use candidate #1 but with team-consistency boost

Usage:
    cd analysis
    uv run python scripts/validate_candidate_ranking.py
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

    console.print(f"\n[bold]Validating candidate ranking strategies across {len(rallies)} rallies[/bold]\n")

    total = 0
    cur_correct = 0
    cand0_correct = 0  # First candidate (perspective-corrected)
    # Hybrid: candidate #1 but demote if same player as previous AND same-team alt exists
    hybrid_correct = 0
    # X-weighted distance
    xweight_correct = 0
    # candidate rank analysis
    gt_rank_in_candidates: dict[int, int] = defaultdict(int)  # rank -> count
    # Team accuracy for each
    cur_team_correct = 0
    cand0_team_correct = 0
    # Per-action
    per_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # When cand0 differs from current
    cand0_diff_total = 0
    cand0_diff_fixes = 0
    cand0_diff_regs = 0

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
        contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

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

        prev_tid = -1  # Track previous player for hybrid

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

            pred_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )

            gt_team = per_rally_teams[gt_tid]
            total += 1

            # Current system
            if pred_tid == gt_tid:
                cur_correct += 1
            if per_rally_teams.get(pred_tid) == gt_team:
                cur_team_correct += 1

            # Candidate #1 (perspective-corrected)
            cand0_tid = -1
            if contact.player_candidates:
                cand0_tid = contact.player_candidates[0][0]
            if cand0_tid == gt_tid:
                cand0_correct += 1
            if per_rally_teams.get(cand0_tid) == gt_team:
                cand0_team_correct += 1

            # When cand0 differs from current
            if cand0_tid != pred_tid:
                cand0_diff_total += 1
                if cand0_tid == gt_tid and pred_tid != gt_tid:
                    cand0_diff_fixes += 1
                elif cand0_tid != gt_tid and pred_tid == gt_tid:
                    cand0_diff_regs += 1

            # Hybrid: use cand0, but if same player as prev AND same team,
            # switch to the OTHER same-team player
            hybrid_tid = cand0_tid
            if hybrid_tid == prev_tid and contact.player_candidates:
                hybrid_team = per_rally_teams.get(hybrid_tid)
                if hybrid_team is not None:
                    for cand_tid, _ in contact.player_candidates:
                        if cand_tid != hybrid_tid and per_rally_teams.get(cand_tid) == hybrid_team:
                            hybrid_tid = cand_tid
                            break
            if hybrid_tid == gt_tid:
                hybrid_correct += 1
            prev_tid = cand0_tid

            # X-weighted distance: re-rank candidates by X-alignment
            # Find all player positions at contact frame and compute X-weighted dist
            import math

            xw_best_tid = -1
            xw_best_score = float("inf")
            contact_frame = contact.frame
            for pp in positions:
                # Search within ±5 frames
                if abs(pp.frame_number - contact_frame) <= 5:
                    # X-weighted distance: emphasize X alignment, reduce Y influence
                    dx = abs(contact.ball_x - pp.x)
                    # Player contact point: upper quarter of bbox (torso/arms)
                    player_y = pp.y - pp.height * 0.25
                    dy = abs(contact.ball_y - player_y)
                    # X-weight = 2x, Y-weight = 0.3x (reduce perspective-biased Y)
                    score = math.sqrt((2.0 * dx) ** 2 + (0.3 * dy) ** 2)
                    if score < xw_best_score:
                        xw_best_score = score
                        xw_best_tid = pp.track_id
            if xw_best_tid == gt_tid:
                xweight_correct += 1
            per_action[m.gt_action]["xw"] = per_action[m.gt_action].get("xw", 0) + (1 if xw_best_tid == gt_tid else 0)

            # GT rank in candidates
            if contact.player_candidates:
                cand_tids = [tid for tid, _ in contact.player_candidates]
                if gt_tid in cand_tids:
                    rank = cand_tids.index(gt_tid)
                    gt_rank_in_candidates[rank] += 1
                else:
                    gt_rank_in_candidates[-1] += 1  # Not in candidates

            # Per-action
            per_action[m.gt_action]["total"] += 1
            if pred_tid == gt_tid:
                per_action[m.gt_action]["cur"] += 1
            if cand0_tid == gt_tid:
                per_action[m.gt_action]["cand0"] += 1
            if hybrid_tid == gt_tid:
                per_action[m.gt_action]["hybrid"] += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts")

    # Report
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]\n")
    console.print(f"  Current (narrow, image-space):     {cur_correct}/{total} = {cur_correct / max(1, total):.1%}")
    console.print(f"  Candidate #1 (wide, corrected):    {cand0_correct}/{total} = {cand0_correct / max(1, total):.1%}")
    console.print(f"  Hybrid (cand0 + alternating):      {hybrid_correct}/{total} = {hybrid_correct / max(1, total):.1%}")
    console.print(f"  X-weighted (2x X, 0.3x Y):         {xweight_correct}/{total} = {xweight_correct / max(1, total):.1%}")

    console.print(f"\n  Team accuracy: current={cur_team_correct}/{total} ({cur_team_correct / max(1, total):.1%}), "
                  f"cand0={cand0_team_correct}/{total} ({cand0_team_correct / max(1, total):.1%})")

    console.print(f"\n  When cand0 differs from current: {cand0_diff_total} contacts")
    console.print(f"    Fixes: {cand0_diff_fixes}, Regressions: {cand0_diff_regs}, Net: {cand0_diff_fixes - cand0_diff_regs}")

    # GT rank distribution
    console.print(f"\n[bold cyan]GT player rank in perspective-corrected candidates[/bold cyan]")
    for rank in sorted(gt_rank_in_candidates.keys()):
        count = gt_rank_in_candidates[rank]
        label = f"Rank {rank}" if rank >= 0 else "Not in list"
        console.print(f"  {label}: {count} ({count / max(1, total):.1%})")

    # Per-action
    console.print(f"\n[bold cyan]Per-action breakdown[/bold cyan]")
    act_table = Table()
    act_table.add_column("Action")
    act_table.add_column("Current", justify="right")
    act_table.add_column("Cand0", justify="right")
    act_table.add_column("Hybrid", justify="right")
    act_table.add_column("X-weight", justify="right")
    act_table.add_column("Total", justify="right")

    for action in ["serve", "receive", "set", "attack", "block", "dig"]:
        cc = per_action[action].get("cur", 0)
        c0 = per_action[action].get("cand0", 0)
        hy = per_action[action].get("hybrid", 0)
        t = per_action[action].get("total", 0)
        if t > 0:
            xw = per_action[action].get("xw", 0)
            act_table.add_row(
                action,
                f"{cc} ({cc / t:.0%})",
                f"{c0} ({c0 / t:.0%})",
                f"{hy} ({hy / t:.0%})",
                f"{xw} ({xw / t:.0%})",
                str(t),
            )
    console.print(act_table)


if __name__ == "__main__":
    main()
