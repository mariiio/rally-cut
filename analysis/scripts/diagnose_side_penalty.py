"""Diagnose match_players side assignment for a specific video.

Runs match_players with diagnostics enabled to dump per-rally cost matrices
and check whether SIDE_PENALTY is being overwhelmed by appearance costs.
"""
from __future__ import annotations

import logging
import sys

import numpy as np

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.match_tracker import (
    RallyTrackData,
    match_players_across_rallies,
)
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

logging.basicConfig(level=logging.WARNING)

video_id = sys.argv[1] if len(sys.argv) > 1 else "0a383519-ecaa-411a-8e5e-e0aadc835725"

# Load video and rallies from DB (same as match_players.py)
video_path = get_video_path(video_id)
if not video_path:
    print(f"Video {video_id[:8]} not found locally")
    sys.exit(1)

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id, r.video_id, r.start_ms, r.end_ms,
                   pt.positions_json, pt.primary_track_ids,
                   pt.court_split_y, pt.ball_positions_json,
                   pt.actions_json
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id = %s
            ORDER BY r.start_ms
            """,
            [video_id],
        )
        rows = cur.fetchall()

        cur.execute(
            "SELECT court_calibration_json FROM videos WHERE id = %s",
            [video_id],
        )
        cal_row = cur.fetchone()
        has_cal = cal_row and cal_row[0] and len(cal_row[0]) == 4

rallies: list[RallyTrackData] = []
for row in rows:
    (rid, vid, start_ms, end_ms, pos_json, primary_ids,
     court_split_y, ball_json, actions_json) = row
    if not pos_json:
        continue
    positions = [
        PlayerPosition(
            frame_number=p["frameNumber"],
            track_id=p["trackId"],
            x=p["x"],
            y=p["y"],
            width=p["width"],
            height=p["height"],
            confidence=p.get("confidence", 1.0),
        )
        for p in pos_json
    ]
    ball_positions = []
    if ball_json:
        ball_positions = [
            BallPosition(
                frame_number=b["frameNumber"],
                x=b["x"],
                y=b["y"],
                confidence=b.get("confidence", 1.0),
            )
            for b in ball_json
        ]

    # Extract team_assignments from actions_json
    team_assignments = None
    if actions_json and isinstance(actions_json, dict):
        ta = actions_json.get("teamAssignments")
        if ta:
            # Values are "A"/"B" strings, convert to 0/1
            team_assignments = {
                int(k): (0 if v in ("A", 0) else 1)
                for k, v in ta.items()
            }

    rallies.append(RallyTrackData(
        rally_id=str(rid),
        video_id=str(vid),
        start_ms=int(start_ms),
        end_ms=int(end_ms),
        positions=positions,
        primary_track_ids=list(primary_ids) if primary_ids else [],
        court_split_y=float(court_split_y) if court_split_y is not None else None,
        ball_positions=ball_positions,
        team_assignments=team_assignments,
    ))

print(f"Video {video_id[:8]}: {len(rallies)} rallies, court_calibration={'yes' if has_cal else 'no'}")

# Run match_players with diagnostics
# Build calibrator for court-based side classification
cal_calibrator = None
if has_cal:
    from rallycut.court.calibration import CourtCalibrator
    cal_calibrator = CourtCalibrator()
    cal_calibrator.calibrate([(c["x"], c["y"]) for c in cal_row[0]])
    if not cal_calibrator.is_calibrated:
        cal_calibrator = None

result = match_players_across_rallies(
    video_path=video_path,
    rallies=rallies,
    num_samples=12,
    collect_diagnostics=True,
    calibrator=cal_calibrator,
)

# Analyze diagnostics
print(f"\nDiagnostics: {len(result.diagnostics)} rallies collected\n")

# Also load court calibration for position checking
calibrator = cal_calibrator

n_wrong_side = 0
n_total = 0
for diag in result.diagnostics:
    rally_idx = diag.rally_index
    if rally_idx >= len(rallies):
        continue
    rally = rallies[rally_idx]

    # Check assigned side vs actual position
    assignment = diag.assignment  # track_id -> player_id
    track_sides = diag.track_court_sides  # track_id -> 0/1

    # Compute actual court_y per track if calibrator available
    if calibrator:
        from collections import defaultdict
        track_ys: dict[int, list[float]] = defaultdict(list)
        for p in rally.positions:
            if p.track_id in assignment:
                foot_y = p.y + p.height * 0.5
                try:
                    _, cy = calibrator.image_to_court((p.x, foot_y), 1, 1)
                    track_ys[p.track_id].append(cy)
                except Exception:
                    pass

    # Check each assignment
    wrong_assignments = []
    for tid, pid in assignment.items():
        assigned_side = track_sides.get(tid)  # 0=near, 1=far from classifier
        player_team = 0 if pid <= 2 else 1

        # Check if track is actually on the expected side
        if calibrator and tid in track_ys and track_ys[tid]:
            med_y = float(np.median(track_ys[tid]))
            actual_side = 1 if med_y > 8.0 else 0

            # Side penalty should have been computed with the CLASSIFIER's side,
            # not the actual court position. Check if classifier side matches actual.
            classifier_correct = (assigned_side == actual_side) if assigned_side is not None else None

            n_total += 1
            if actual_side != player_team:
                n_wrong_side += 1
                wrong_assignments.append(
                    f"track {tid}→p{pid}: actual_y={med_y:.1f} "
                    f"(side={actual_side}), player expects side={player_team}, "
                    f"classifier_side={assigned_side}, "
                    f"classifier_correct={classifier_correct}"
                )

    if wrong_assignments:
        # Show cost matrix for this rally
        cm = diag.cost_matrix
        tids = diag.track_ids
        pids = diag.player_ids
        print(f"Rally {rally_idx+1:2d} ({rally.rally_id[:8]}): WRONG assignments")
        for wa in wrong_assignments:
            print(f"  {wa}")

        # Print cost matrix
        header = "        " + "".join(f"  p{p}" for p in pids)
        print(header)
        for i, tid in enumerate(tids):
            row = f"  t{tid:3d}:"
            for j in range(len(pids)):
                val = cm[i, j]
                # Highlight the assigned cell
                assigned = assignment.get(tid)
                marker = " *" if assigned == pids[j] else "  "
                row += f" {val:.2f}{marker}"
            side = track_sides.get(tid, "?")
            row += f"  (side={side})"
            print(row)
        print()

print(f"\nSummary: {n_wrong_side}/{n_total} track assignments on wrong court side")
