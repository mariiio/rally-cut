"""Verify calibrated camera heights using player bboxes as ground truth.

Hypothesis: if camera heights are systematically underestimated, then
projecting player bboxes (feet on ground, head at ~1.8m) will give
implied heights that disagree with reality.

For each calibrated video, for each player tracked in each rally:
1. Take the bbox (image_x, image_y, width, height) at several frames
2. Foot position = (bbox_x + width/2, bbox_y + height)
3. Head position = (bbox_x + width/2, bbox_y)
4. Assume foot is on court ground plane (z=0)
5. Back-project foot through the camera to get court (X, Y)
6. Given court (X, Y) and head image position, solve for head z
7. That is the camera's implied player height

If the camera is correct, implied heights should cluster around 1.80m
(men) or 1.70m (women). If they're systematically ~1.2m or ~2.5m, the
camera calibration is biased.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rallycut.court.camera_model import (  # noqa: E402
    CameraModel,
    calibrate_camera,
    calibrate_camera_with_net,
    image_ray,
)
from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from eval_ball_3d import (  # noqa: E402
    COURT_CORNERS,
    load_calibrated_videos,
    load_rallies_for_videos,
    _build_contact_sequence,
    _parse_ball_positions,
    _parse_player_positions,
)
from rallycut.evaluation.db import get_connection  # noqa: E402


SESSION_SHORT_ID = "41e1f30d-d5bb-4386-9908-fa37216eb535"


def _load_session_video_ids(session_id: str) -> set[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id FROM session_videos WHERE session_id = %s",
                (session_id,),
            )
            return {str(row[0]) for row in cur.fetchall()}


def _ray_hit_ground(camera: CameraModel, u_norm: float, v_norm: float) -> np.ndarray | None:
    """Intersect the camera ray through (u, v) with the ground plane z=0."""
    origin, direction = image_ray(camera, (u_norm, v_norm))
    if abs(direction[2]) < 1e-9:
        return None
    t = -origin[2] / direction[2]
    if t <= 0:
        return None
    return origin + t * direction


def _implied_head_height(
    camera: CameraModel,
    foot_u: float, foot_v: float,
    head_u: float, head_v: float,
) -> float | None:
    """Given foot + head image pixels, compute the 3D height of the head.

    Assumes foot is on the z=0 ground plane. Projects foot to 3D, then
    finds where the head ray reaches that (X, Y) and reads off the Z.
    """
    foot_3d = _ray_hit_ground(camera, foot_u, foot_v)
    if foot_3d is None:
        return None

    # Head ray.
    origin, direction = image_ray(camera, (head_u, head_v))

    # The head's 3D point lies on the line origin + t * direction.
    # We want the point where (X, Y) = (foot_3d[0], foot_3d[1]) approximately.
    # Parameterise: head_x = origin[0] + t * d[0], head_y = origin[1] + t * d[1]
    # Solve for t that best matches foot_3d[:2]. Use the dominant axis.
    dx_target = foot_3d[0] - origin[0]
    dy_target = foot_3d[1] - origin[1]
    if abs(direction[0]) > abs(direction[1]):
        if abs(direction[0]) < 1e-9:
            return None
        t = dx_target / direction[0]
    else:
        if abs(direction[1]) < 1e-9:
            return None
        t = dy_target / direction[1]
    if t <= 0:
        return None

    head_3d = origin + t * direction
    return float(head_3d[2])


def _calibrate_video(vcal: Any, net_y: float | None) -> CameraModel | None:
    cam = None
    if net_y is not None:
        cam = calibrate_camera_with_net(
            vcal.image_corners, COURT_CORNERS, vcal.width, vcal.height,
            net_y_image=net_y,
        )
    if cam is None or not cam.is_valid:
        cam = calibrate_camera(
            vcal.image_corners, COURT_CORNERS, vcal.width, vcal.height,
        )
    return cam if cam is not None and cam.is_valid else None


def main() -> None:
    print("Loading session short videos...")
    session_vids = _load_session_video_ids(SESSION_SHORT_ID)
    print(f"  {len(session_vids)} videos in session short")

    print("Loading calibrated videos + rallies...")
    videos = load_calibrated_videos(None)
    rallies = load_rallies_for_videos(session_vids)
    rallies_by_video: dict[str, list[Any]] = defaultdict(list)
    for r in rallies:
        rallies_by_video[r.video_id].append(r)

    print(f"  loaded {len(rallies)} rallies across {len(rallies_by_video)} videos\n")

    print("=" * 80)
    print("Per-video implied player heights (median across rally samples)")
    print("=" * 80)
    print(f"  {'video':<10s} {'cam_h':>7s} {'focal':>7s} {'n_samples':>10s} "
          f"{'p25':>6s} {'median':>7s} {'p75':>6s} {'mean':>6s}")

    all_heights: list[float] = []
    all_cam_heights: list[float] = []

    for vid_id in sorted(session_vids):
        if vid_id not in videos:
            continue
        vcal = videos[vid_id]
        vid_rallies = rallies_by_video.get(vid_id, [])
        if not vid_rallies:
            continue

        # Estimate net_y.
        calibrator = CourtCalibrator()
        calibrator.calibrate(vcal.image_corners)
        net_ys: list[float] = []
        for rally in vid_rallies[:15]:
            bp = _parse_ball_positions(rally.ball_positions_json)
            pp = _parse_player_positions(rally.positions_json)
            if len(bp) < 20:
                continue
            cs = _build_contact_sequence(bp, pp, calibrator)
            if 0.1 < cs.net_y < 0.9:
                net_ys.append(cs.net_y)
        net_y = float(np.median(net_ys)) if net_ys else None

        cam = _calibrate_video(vcal, net_y)
        if cam is None:
            continue

        # Sample player bboxes across all rallies. Use middle frame of each rally.
        implied_heights: list[float] = []
        for rally in vid_rallies:
            pp = _parse_player_positions(rally.positions_json)
            if not pp:
                continue
            # Pick frames at 25%, 50%, 75% of rally length.
            frames = sorted({p.frame_number for p in pp})
            if not frames:
                continue
            sample_frames = [
                frames[len(frames) // 4],
                frames[len(frames) // 2],
                frames[3 * len(frames) // 4],
            ]
            for sf in sample_frames:
                for p in pp:
                    if p.frame_number != sf:
                        continue
                    # bbox is image-space; use bottom-center for feet, top-center for head.
                    # p.x, p.y are normalized centre coordinates (0-1).
                    # p.width, p.height are normalized.
                    cx = p.x
                    top_y = p.y - p.height / 2
                    bot_y = p.y + p.height / 2
                    foot_u = cx
                    foot_v = bot_y
                    head_u = cx
                    head_v = top_y
                    if not (0 < foot_v < 1 and 0 < head_v < 1):
                        continue
                    h = _implied_head_height(cam, foot_u, foot_v, head_u, head_v)
                    if h is None or not (0.3 < h < 3.5):
                        continue
                    implied_heights.append(h)

        if len(implied_heights) < 5:
            continue
        arr = np.array(implied_heights)
        median = float(np.median(arr))
        all_heights.extend(implied_heights)
        all_cam_heights.append(float(cam.camera_position[2]))
        print(
            f"  {vid_id[:8]}   {cam.camera_position[2]:6.2f}m "
            f"{cam.focal_length_px:6.0f}px   {len(arr):>6d}   "
            f"{np.percentile(arr, 25):5.2f}m  {median:5.2f}m  "
            f"{np.percentile(arr, 75):5.2f}m {arr.mean():5.2f}m"
        )

    if all_heights:
        arr = np.array(all_heights)
        print("\n" + "=" * 80)
        print("AGGREGATE (session short) — implied player heights")
        print("=" * 80)
        print(f"  n={len(arr)} player samples across {len(all_cam_heights)} videos")
        print(f"  Implied heights: "
              f"p10={np.percentile(arr, 10):.2f}m  "
              f"p25={np.percentile(arr, 25):.2f}m  "
              f"median={np.median(arr):.2f}m  "
              f"p75={np.percentile(arr, 75):.2f}m  "
              f"p90={np.percentile(arr, 90):.2f}m")
        print(f"  mean={arr.mean():.2f}m  std={arr.std():.2f}m")
        print()
        print(f"  Camera heights (estimated): "
              f"median={np.median(all_cam_heights):.2f}m  "
              f"mean={np.mean(all_cam_heights):.2f}m")
        print()
        print("  Expected player height: ~1.75-1.85m (men's beach volleyball)")
        print()
        # Interpretation.
        median_h = float(np.median(arr))
        if 1.65 <= median_h <= 1.95:
            print("  → camera calibration appears CONSISTENT with real player heights")
            print("    Cameras really are at the estimated heights.")
        elif median_h < 1.65:
            ratio = 1.80 / median_h
            implied_true_cam = np.median(all_cam_heights) * ratio
            print(f"  → implied heights are LOW by factor ~{ratio:.2f}x")
            print(f"    If players are actually 1.80m, cameras may be at ~{implied_true_cam:.2f}m (not {np.median(all_cam_heights):.2f}m)")
            print("    Calibration may be UNDER-estimating camera height.")
        else:
            ratio = 1.80 / median_h
            print(f"  → implied heights are HIGH by factor ~{ratio:.2f}x")
            print("    Calibration may be OVER-estimating camera height.")


if __name__ == "__main__":
    main()
