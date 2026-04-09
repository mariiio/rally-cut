"""Test different perspective correction strengths.

The current _depth_scale_at_y uses scale = near_width / width_at_y (power=1).
This captures X-axis perspective but not the additional Y-axis compression.

Test scale^power with different powers to find optimal correction:
- power=1: current (too weak)
- power=2: 2D perspective (both X and Y)
- power=3+: aggressive, may help for extreme far-court cases

Also test a simple heuristic scale when calibration is unavailable.

Usage:
    cd analysis
    uv run python scripts/validate_scale_power.py
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
    _depth_scale_at_y,
    detect_contacts,
)
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def find_nearest_with_power(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPos],
    court_calibrator: CourtCalibrator | None,
    net_y: float,
    power: float = 1.0,
    search_frames: int = 15,
) -> list[tuple[int, float]]:
    """Find nearest players ranked by scale^power corrected distance.

    Falls back to Y-ratio-based heuristic when calibration unavailable.
    """
    best: dict[int, tuple[float, float]] = {}  # tid -> (rank_dist, img_dist)

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        player_x = p.x
        player_y = p.y - p.height * 0.25
        img_dist = math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)

        # Perspective scale
        scale = _depth_scale_at_y(player_y, court_calibrator)
        if scale == 1.0 and net_y > 0:
            # Heuristic: far-court players (above net_y) get scaled by
            # how far they are from net_y relative to the player at the bottom.
            # Near-court (below net_y): scale=1. Far-court: scale increases.
            if player_y < net_y:
                # Far court — higher scaling
                ratio = net_y / max(player_y, 0.01)
                scale = ratio
            else:
                scale = 1.0

        rank_dist = img_dist * (scale ** power)

        if p.track_id not in best or rank_dist < best[p.track_id][0]:
            best[p.track_id] = (rank_dist, img_dist)

    ranked = sorted(best.items(), key=lambda x: x[1][0])
    return [(tid, img_dist) for tid, (_rd, img_dist) in ranked[:4]]


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
    n_cal = 0
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
            n_cal += 1
        else:
            calibrators[vid] = None

    console.print(f"\n[bold]Testing scale powers across {len(rallies)} rallies[/bold]")
    console.print(f"  Calibrated videos: {n_cal}/{len(video_ids)}\n")

    # Test multiple powers
    powers = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results_by_power: dict[float, int] = {p: 0 for p in powers}
    team_by_power: dict[float, int] = {p: 0 for p in powers}
    total = 0
    cur_correct = 0

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

            gt_team = per_rally_teams[gt_tid]
            contact = next((c for c in contacts if c.frame == m.pred_frame), None)
            if contact is None:
                continue

            total += 1

            # Current
            pred_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )
            if pred_tid == gt_tid:
                cur_correct += 1

            # Test each power
            for power in powers:
                ranked = find_nearest_with_power(
                    contact.frame, contact.ball_x, contact.ball_y,
                    positions, cal, net_y, power=power,
                )
                if ranked and ranked[0][0] == gt_tid:
                    results_by_power[power] += 1
                if ranked:
                    pick_team = per_rally_teams.get(ranked[0][0])
                    if pick_team == gt_team:
                        team_by_power[power] += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts")

    # Report
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]\n")
    console.print(f"  Current system: {cur_correct}/{total} = {cur_correct / max(1, total):.1%}\n")

    table = Table(title="Scale power sweep")
    table.add_column("Power")
    table.add_column("Player Correct", justify="right")
    table.add_column("Player Acc", justify="right")
    table.add_column("Team Correct", justify="right")
    table.add_column("Team Acc", justify="right")

    for power in powers:
        pc = results_by_power[power]
        tc = team_by_power[power]
        style = "bold" if pc > cur_correct else ""
        table.add_row(
            f"{power:.1f}",
            str(pc),
            f"{pc / max(1, total):.1%}",
            str(tc),
            f"{tc / max(1, total):.1%}",
            style=style,
        )
    console.print(table)


if __name__ == "__main__":
    main()
