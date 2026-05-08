"""Visual debug for 3D ball trajectory — overlay projections on video frames.

Generates annotated images showing:
- Court corners (green circles + lines)
- Net line at ground level and net-top height (cyan / magenta)
- Ball detections (yellow dots)
- Fitted 3D trajectory reprojected (red curve)
- Camera info text overlay

Usage:
    cd analysis
    uv run python scripts/debug_ball_3d.py                   # 3 videos, 1 rally each
    uv run python scripts/debug_ball_3d.py --video-id <id>   # specific video
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH, CourtCalibrator  # noqa: E402
from rallycut.court.camera_model import (  # noqa: E402
    CameraModel,
    calibrate_camera,
    project_3d_to_image,
)
from rallycut.court.trajectory_3d import fit_rally  # noqa: E402
from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.evaluation.tracking.db import get_video_path  # noqa: E402
from rallycut.tracking.action_classifier import ActionType, ClassifiedAction  # noqa: E402
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

COURT_CORNERS: list[tuple[float, float]] = [
    (0.0, 0.0), (COURT_WIDTH, 0.0), (COURT_WIDTH, COURT_LENGTH), (0.0, COURT_LENGTH),
]

OUT_DIR = Path("outputs/ball_3d/debug")


def _to_px(u: float, v: float, w: int, h: int) -> tuple[int, int]:
    return (int(u * w), int(v * h))


def draw_debug_frame(
    frame: np.ndarray,
    cam: CameraModel,
    corners: list[tuple[float, float]],
    ball_positions: list[BallPosition],
    rally_start_frame: int,
    traj_result: Any | None,
    fps: float,
    vid_id: str,
) -> np.ndarray:
    """Draw all debug overlays on a single frame."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # --- Court outline (green) ------------------------------------------------
    court_grid = [
        [(0, 0, 0), (8, 0, 0)],   # near baseline
        [(0, 16, 0), (8, 16, 0)],  # far baseline
        [(0, 0, 0), (0, 16, 0)],   # left sideline
        [(8, 0, 0), (8, 16, 0)],   # right sideline
        [(0, 8, 0), (8, 8, 0)],    # center line (ground)
    ]
    for p1_3d, p2_3d in court_grid:
        u1, v1 = project_3d_to_image(cam, np.array(p1_3d, dtype=np.float64))
        u2, v2 = project_3d_to_image(cam, np.array(p2_3d, dtype=np.float64))
        cv2.line(out, _to_px(u1, v1, w, h), _to_px(u2, v2, w, h), (0, 255, 0), 2)

    # --- Court corners (green circles) ----------------------------------------
    for i, (cx, cy) in enumerate(COURT_CORNERS):
        u, v = project_3d_to_image(cam, np.array([cx, cy, 0.0]))
        px = _to_px(u, v, w, h)
        cv2.circle(out, px, 8, (0, 255, 0), -1)
        # Also show actual corner position (blue)
        actual = _to_px(corners[i][0], corners[i][1], w, h)
        cv2.circle(out, actual, 6, (255, 100, 0), 2)

    # --- Net at ground level (cyan) and net top (magenta) --------------------
    for x in [0, 2, 4, 6, 8]:
        # Ground level
        ug, vg = project_3d_to_image(cam, np.array([float(x), 8.0, 0.0]))
        cv2.circle(out, _to_px(ug, vg, w, h), 4, (255, 255, 0), -1)
        # Net top (2.43m)
        ut, vt = project_3d_to_image(cam, np.array([float(x), 8.0, 2.43]))
        cv2.circle(out, _to_px(ut, vt, w, h), 4, (255, 0, 255), -1)
        # Vertical line connecting ground to net top
        cv2.line(out, _to_px(ug, vg, w, h), _to_px(ut, vt, w, h), (255, 0, 255), 1)

    # Net top line (magenta)
    u1, v1 = project_3d_to_image(cam, np.array([0.0, 8.0, 2.43]))
    u2, v2 = project_3d_to_image(cam, np.array([8.0, 8.0, 2.43]))
    cv2.line(out, _to_px(u1, v1, w, h), _to_px(u2, v2, w, h), (255, 0, 255), 2)

    # --- Ball detections (yellow dots) ----------------------------------------
    for bp in ball_positions:
        px = _to_px(bp.x, bp.y, w, h)
        alpha = max(0.3, bp.confidence)
        color = (0, int(255 * alpha), 255)
        cv2.circle(out, px, 3, color, -1)

    # --- 3D fit: yellow dot = actual, red circle = predicted ------------------
    # If they overlap → fit is good.  Far apart → fit is bad.
    if traj_result and traj_result.arcs:
        for arc_i, arc in enumerate(traj_result.arcs):
            arc_obs = [
                bp for bp in ball_positions
                if arc.start_frame <= bp.frame_number <= arc.end_frame
                and bp.confidence > 0.1
            ]

            for bp in arc_obs:
                t = (bp.frame_number - arc.start_frame) / fps
                pt3d = np.array([
                    arc.initial_position[0] + arc.initial_velocity[0] * t,
                    arc.initial_position[1] + arc.initial_velocity[1] * t,
                    arc.initial_position[2] + arc.initial_velocity[2] * t - 0.5 * 9.81 * t**2,
                ])
                u_pred, v_pred = project_3d_to_image(cam, pt3d)

                if -0.3 < u_pred < 1.3 and -0.3 < v_pred < 1.3:
                    pred_px = _to_px(u_pred, v_pred, w, h)
                    # Red open circle = where model thinks ball is
                    cv2.circle(out, pred_px, 6, (0, 0, 255), 2)

            # Label near the first observation
            if arc_obs:
                lx, ly = _to_px(arc_obs[0].x, arc_obs[0].y, w, h)
                label = f"arc{arc_i + 1} {arc.speed_at_start:.0f}m/s  rmse={arc.reprojection_rmse:.0f}px"
                cv2.putText(out, label, (lx + 8, ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(out, label, (lx + 8, ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # --- Info overlay --------------------------------------------------------
    lines = [
        f"Video: {vid_id[:20]}",
        f"Camera: h={cam.camera_position[2]:.2f}m  f={cam.focal_length_px:.0f}px",
        f"Pos: ({cam.camera_position[0]:.1f}, {cam.camera_position[1]:.1f}, {cam.camera_position[2]:.1f})",
        f"Reproj: {cam.reprojection_error:.2f}px",
    ]
    if traj_result and traj_result.arcs:
        n_valid = sum(1 for a in traj_result.arcs if a.is_valid)
        lines.append(f"Arcs: {n_valid}/{len(traj_result.arcs)} valid")
        if traj_result.serve_speed_mps is not None:
            lines.append(f"Serve: {traj_result.serve_speed_mps:.1f} m/s")

    lines.append("")
    lines.append("Green = court outline")
    lines.append("Magenta = net top (2.43m)")
    lines.append("Yellow dot = actual ball detection")
    lines.append("Red circle = 3D model prediction")
    lines.append("(overlap = good fit)")

    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return out


def _parse_ball_positions(bp_json: Any) -> list[BallPosition]:
    if not bp_json:
        return []
    positions = bp_json.get("positions", []) if isinstance(bp_json, dict) else bp_json
    return [
        BallPosition(
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            x=p.get("x", 0.0), y=p.get("y", 0.0),
            confidence=p.get("confidence", 0.5),
        )
        for p in positions
    ]


def _parse_player_positions(pos_json: Any) -> list[PlayerPosition]:
    if not pos_json:
        return []
    return [
        PlayerPosition(
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            track_id=p.get("trackId", p.get("track_id", 0)),
            x=p.get("x", 0.0), y=p.get("y", 0.0),
            width=p.get("width", 0.0), height=p.get("height", 0.0),
            confidence=p.get("confidence", 0.5),
        )
        for p in pos_json
    ]


def _parse_actions(actions_json: Any) -> list[ClassifiedAction]:
    if not actions_json:
        return []
    actions_list: list[dict[str, Any]] = []
    if isinstance(actions_json, dict):
        inner = actions_json.get("actions", actions_json)
        if isinstance(inner, dict):
            actions_list = inner.get("actions", [])
        elif isinstance(inner, list):
            actions_list = inner
    elif isinstance(actions_json, list):
        actions_list = actions_json

    result: list[ClassifiedAction] = []
    for a in actions_list:
        if not isinstance(a, dict):
            continue
        try:
            result.append(ClassifiedAction(
                action_type=ActionType(a.get("action", "unknown")),
                frame=a.get("frame", 0), ball_x=a.get("ballX", 0.0),
                ball_y=a.get("ballY", 0.0), velocity=a.get("velocity", 0.0),
                player_track_id=a.get("playerTrackId", -1),
                court_side=a.get("courtSide", "unknown"),
                confidence=a.get("confidence", 0.0), team=a.get("team", "unknown"),
            ))
        except (KeyError, ValueError):
            continue
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", help="Specific video ID")
    parser.add_argument("--max-videos", type=int, default=3)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load videos + rallies
    with get_connection() as conn:
        with conn.cursor() as cur:
            where = "AND v.id = %s" if args.video_id else ""
            params = [args.video_id] if args.video_id else []
            cur.execute(f"""
                SELECT v.id, v.court_calibration_json, v.width, v.height,
                       r.id, r.start_ms, pt.ball_positions_json,
                       pt.positions_json, pt.actions_json, pt.fps
                FROM videos v
                JOIN rallies r ON r.video_id = v.id
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE v.court_calibration_json IS NOT NULL
                  AND pt.ball_positions_json IS NOT NULL
                  {where}
                ORDER BY v.id, r.start_ms
            """, params)
            rows = cur.fetchall()

    by_vid: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_vid[row[0]].append(row)

    n_done = 0
    for vid_id, vid_rows in sorted(by_vid.items()):
        if n_done >= args.max_videos:
            break

        row = vid_rows[0]  # first rally
        cal_json = row[1]
        w, h = row[2] or 1920, row[3] or 1080
        rally_id = row[4]
        start_ms = row[5] or 0
        bp_json = row[6]
        pos_json = row[7]
        act_json = row[8]
        fps = row[9] or 30.0

        if not isinstance(cal_json, list) or len(cal_json) != 4:
            continue

        corners = [(c["x"], c["y"]) for c in cal_json]
        cam = calibrate_camera(corners, COURT_CORNERS, w, h)
        if cam is None or not cam.is_valid:
            print(f"[SKIP] {vid_id[:12]}: camera calibration failed")
            continue

        # Download video
        print(f"[{n_done + 1}] Downloading {vid_id[:12]}...")
        video_path = get_video_path(vid_id)
        if video_path is None:
            print("  Could not download video")
            continue

        # Read frame at rally start
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(start_ms / 1000.0 * video_fps)
        # Read a frame 1 second into the rally (skip the very start)
        target_frame = start_frame + int(video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"  Could not read frame {target_frame}")
            continue

        # Parse data and fit trajectory
        ball_positions = _parse_ball_positions(bp_json)
        player_positions = _parse_player_positions(pos_json)
        actions = _parse_actions(act_json)

        calibrator = CourtCalibrator()
        calibrator.calibrate(corners)
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=ContactDetectionConfig(),
            court_calibrator=calibrator,
        )

        traj = fit_rally(
            camera=cam,
            contact_sequence=contact_seq,
            classified_actions=actions if actions else None,
            fps=fps,
            rally_id=rally_id,
            video_id=vid_id,
        )

        # Draw debug overlay
        out = draw_debug_frame(
            frame, cam, corners, ball_positions,
            start_frame, traj, fps, vid_id,
        )

        out_path = OUT_DIR / f"{vid_id[:12]}_debug.jpg"
        cv2.imwrite(str(out_path), out)
        print(f"  Saved: {out_path}")
        print(f"  Camera: h={cam.camera_position[2]:.2f}m, f={cam.focal_length_px:.0f}px")
        if traj.arcs:
            n_valid = sum(1 for a in traj.arcs if a.is_valid)
            print(f"  Arcs: {n_valid}/{len(traj.arcs)} valid")
            if traj.serve_speed_mps:
                print(f"  Serve: {traj.serve_speed_mps:.1f} m/s")

        n_done += 1

    print(f"\nDone. {n_done} debug images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
