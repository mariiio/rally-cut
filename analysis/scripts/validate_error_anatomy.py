"""Deep anatomy of attribution errors to find discriminating features.

For each wrong-nearest error, examine what's DIFFERENT about the GT player
vs the predicted player. Look for features that cleanly separate them.

Usage:
    cd analysis
    uv run python scripts/validate_error_anatomy.py
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

    console.print(f"\n[bold]Error anatomy across {len(rallies)} rallies[/bold]\n")

    total = 0
    correct = 0
    # Error anatomy
    cross_team_rank1 = 0  # GT is rank 1 in same-team candidates
    cross_team_rank2plus = 0
    within_team = 0
    # Feature distributions
    pred_heights: list[float] = []
    gt_heights: list[float] = []
    pred_is_taller_cross: list[bool] = []
    gt_below_ball_cross: list[bool] = []  # GT player Y > ball Y (below in image = near court)
    pred_below_ball_cross: list[bool] = []

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
        contact_by_frame = {c.frame: c for c in contacts}

        # Get player positions by (track_id, frame)
        pp_lookup: dict[tuple[int, int], PlayerPos] = {}
        for pp in positions:
            key = (pp.track_id, pp.frame_number)
            pp_lookup[key] = pp

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
            if contact is None or not contact.player_candidates:
                continue

            gt_team = per_rally_teams[gt_tid]
            pred_tid = contact.player_candidates[0][0]  # cand0 pick (current system)
            pred_team = per_rally_teams.get(pred_tid)

            total += 1
            if pred_tid == gt_tid:
                correct += 1
                continue

            # Error case — classify
            if pred_team != gt_team:
                # Cross-team error
                # Find GT player's position info
                gt_pp = None
                pred_pp = None
                for f_offset in range(6):
                    if gt_pp is None:
                        gt_pp = pp_lookup.get((gt_tid, m.pred_frame + f_offset)) or pp_lookup.get((gt_tid, m.pred_frame - f_offset))
                    if pred_pp is None:
                        pred_pp = pp_lookup.get((pred_tid, m.pred_frame + f_offset)) or pp_lookup.get((pred_tid, m.pred_frame - f_offset))

                if gt_pp and pred_pp:
                    pred_heights.append(pred_pp.height)
                    gt_heights.append(gt_pp.height)
                    pred_is_taller_cross.append(pred_pp.height > gt_pp.height)
                    gt_below_ball_cross.append(gt_pp.y > contact.ball_y)
                    pred_below_ball_cross.append(pred_pp.y > contact.ball_y)

                # Is GT the nearest on their team?
                same_team_cands = [
                    (tid, d) for tid, d in contact.player_candidates
                    if per_rally_teams.get(tid) == gt_team
                ]
                if same_team_cands and same_team_cands[0][0] == gt_tid:
                    cross_team_rank1 += 1
                else:
                    cross_team_rank2plus += 1
            else:
                within_team += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts, {correct} correct")

    errors = total - correct
    console.print(f"\n[bold]Results: {total} contacts, {correct} correct ({correct/max(1,total):.1%})[/bold]")
    console.print(f"  Errors: {errors}")
    console.print(f"  Cross-team (GT rank 1 on team): {cross_team_rank1}")
    console.print(f"  Cross-team (GT rank 2+ on team): {cross_team_rank2plus}")
    console.print(f"  Within-team: {within_team}")

    n_cross = cross_team_rank1 + cross_team_rank2plus
    console.print(f"\n[bold cyan]Cross-team error features ({n_cross} errors)[/bold cyan]")
    if pred_heights and gt_heights:
        avg_pred_h = sum(pred_heights) / len(pred_heights)
        avg_gt_h = sum(gt_heights) / len(gt_heights)
        console.print(f"  Avg pred player height: {avg_pred_h:.3f}")
        console.print(f"  Avg GT player height: {avg_gt_h:.3f}")
        console.print(f"  Pred is taller (nearer): {sum(pred_is_taller_cross)}/{len(pred_is_taller_cross)} = {sum(pred_is_taller_cross)/max(1,len(pred_is_taller_cross)):.1%}")
        console.print(f"  GT player below ball (Y > ball_y): {sum(gt_below_ball_cross)}/{len(gt_below_ball_cross)} = {sum(gt_below_ball_cross)/max(1,len(gt_below_ball_cross)):.1%}")
        console.print(f"  Pred player below ball: {sum(pred_below_ball_cross)}/{len(pred_below_ball_cross)} = {sum(pred_below_ball_cross)/max(1,len(pred_below_ball_cross)):.1%}")

    console.print(f"\n[bold green]If we could fix ALL cross-team rank1 errors:[/bold green]")
    ceiling = correct + cross_team_rank1
    console.print(f"  {ceiling}/{total} = {ceiling / max(1, total):.1%}")
    console.print(f"  (+ fix cross-team rank2+ too: {correct + n_cross}/{total} = {(correct + n_cross) / max(1, total):.1%})")


if __name__ == "__main__":
    main()
