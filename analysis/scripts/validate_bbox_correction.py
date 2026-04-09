"""Test bbox-height-based perspective correction for player attribution.

Hypothesis: player bbox height directly measures camera perspective.
Near-court players have large bboxes, far-court players have small ones.
Correcting distance by bbox height ratio should be a stronger perspective
correction than the geometric court-width approach.

Corrected distance = image_dist * (reference_height / player_height)^power

Also tests: using ONLY bbox height (no distance) as a ranking signal,
and combining bbox height with distance.

Usage:
    cd analysis
    uv run python scripts/validate_bbox_correction.py
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


def find_nearest_bbox_corrected(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPos],
    search_frames: int = 15,
    power: float = 1.0,
) -> list[tuple[int, float, float, float]]:
    """Find nearest players with bbox-height-based correction.

    Returns: list of (track_id, corrected_distance, image_distance, bbox_height)
    """
    # Find the largest bbox height in this frame window (near-court reference)
    max_height = 0.0
    for p in player_positions:
        if abs(p.frame_number - frame) <= search_frames:
            max_height = max(max_height, p.height)

    if max_height <= 0:
        return []

    best: dict[int, tuple[float, float, float]] = {}
    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        player_x = p.x
        player_y = p.y - p.height * 0.25
        img_dist = math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)

        # Bbox-height correction: small players are far from camera
        height_ratio = max_height / max(p.height, 0.01)
        corrected_dist = img_dist * (height_ratio ** power)

        if p.track_id not in best or corrected_dist < best[p.track_id][0]:
            best[p.track_id] = (corrected_dist, img_dist, p.height)

    ranked = sorted(best.items(), key=lambda x: x[1][0])
    return [(tid, cd, imd, h) for tid, (cd, imd, h) in ranked[:4]]


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

    console.print(f"\n[bold]Testing bbox-height correction across {len(rallies)} rallies[/bold]\n")

    powers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results: dict[float, int] = {p: 0 for p in powers}
    team_results: dict[float, int] = {p: 0 for p in powers}
    total = 0
    cur_correct = 0
    cand0_correct = 0

    # Also measure bbox height ratios
    near_heights: list[float] = []
    far_heights: list[float] = []

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

        # Measure bbox height by team
        for p in positions[:200]:  # Sample first 200 positions
            team = per_rally_teams.get(p.track_id)
            if team == 0:
                near_heights.append(p.height)
            elif team == 1:
                far_heights.append(p.height)

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
            if gt_tid < 0 or gt_tid not in avail_tids or gt_tid not in per_rally_teams:
                continue

            gt_team = per_rally_teams[gt_tid]
            contact = contact_by_frame.get(m.pred_frame)
            if contact is None:
                continue

            total += 1
            pred_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )
            if pred_tid == gt_tid:
                cur_correct += 1
            if contact.player_candidates and contact.player_candidates[0][0] == gt_tid:
                cand0_correct += 1

            # Test each power
            for power in powers:
                ranked = find_nearest_bbox_corrected(
                    contact.frame, contact.ball_x, contact.ball_y,
                    positions, search_frames=15, power=power,
                )
                if ranked and ranked[0][0] == gt_tid:
                    results[power] += 1
                if ranked:
                    pick_team = per_rally_teams.get(ranked[0][0])
                    if pick_team == gt_team:
                        team_results[power] += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts")

    # Report
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]\n")
    console.print(f"  Current system: {cur_correct}/{total} = {cur_correct / max(1, total):.1%}")
    console.print(f"  Cand0 (geom correction): {cand0_correct}/{total} = {cand0_correct / max(1, total):.1%}")

    # Bbox heights
    if near_heights and far_heights:
        avg_near = sum(near_heights) / len(near_heights)
        avg_far = sum(far_heights) / len(far_heights)
        console.print(f"\n  Avg bbox height: near={avg_near:.3f}, far={avg_far:.3f}, ratio={avg_near / max(avg_far, 0.001):.1f}x")

    table = Table(title="Bbox-height correction power sweep")
    table.add_column("Power")
    table.add_column("Player Correct", justify="right")
    table.add_column("Player Acc", justify="right")
    table.add_column("Team Correct", justify="right")
    table.add_column("Team Acc", justify="right")
    table.add_column("vs Current", justify="right")

    for power in powers:
        pc = results[power]
        tc = team_results[power]
        delta = pc - cur_correct
        style = "bold green" if delta > 0 else ("red" if delta < 0 else "")
        table.add_row(
            f"{power:.1f}",
            str(pc),
            f"{pc / max(1, total):.1%}",
            str(tc),
            f"{tc / max(1, total):.1%}",
            f"{delta:+d}",
            style=style,
        )
    console.print(table)


if __name__ == "__main__":
    main()
