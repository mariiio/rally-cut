"""Diagnose the focal-length / camera-height / implied-player-height tradeoff.

For one video, sweep focal length over a wide range and report:
  focal, reproj_error, camera_height, implied_player_height

This tells us whether:
  (a) The focal length / height ambiguity is real but the prior gives
      acceptable reproj at "correct" player heights → fix is to relax reproj
  (b) Reproj is well-conditioned and ALL focal lengths give ~1.47m implied
      heights → calibration is actually correct, players really back-project
      that low (bbox truncation problem)
  (c) There's a sweet spot we're missing
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rallycut.court.camera_model import (  # noqa: E402
    CameraModel,
    _build_model,  # type: ignore[attr-defined]
    image_ray,
)
from eval_ball_3d import (  # noqa: E402
    COURT_CORNERS,
    load_calibrated_videos,
    load_rallies_for_videos,
    _parse_player_positions,
)


# Pick a typical video. 313c6c95 (meme) has 44 arcs and 17 serves —
# best signal among session short.
TARGET_VID = "313c6c95-e586-4585-bfce-a2d293d96815"


def _ray_hit_ground(camera: CameraModel, u: float, v: float) -> np.ndarray | None:
    origin, direction = image_ray(camera, (u, v))
    if abs(direction[2]) < 1e-9:
        return None
    t = -origin[2] / direction[2]
    if t <= 0:
        return None
    return origin + t * direction


def _implied_head_height(
    cam: CameraModel,
    foot_u: float, foot_v: float,
    head_u: float, head_v: float,
) -> float | None:
    foot_3d = _ray_hit_ground(cam, foot_u, foot_v)
    if foot_3d is None:
        return None
    origin, direction = image_ray(cam, (head_u, head_v))
    dx = foot_3d[0] - origin[0]
    dy = foot_3d[1] - origin[1]
    if abs(direction[0]) > abs(direction[1]):
        if abs(direction[0]) < 1e-9:
            return None
        t = dx / direction[0]
    else:
        if abs(direction[1]) < 1e-9:
            return None
        t = dy / direction[1]
    if t <= 0:
        return None
    head_3d = origin + t * direction
    return float(head_3d[2])


def _build_camera_with_focal(
    image_corners: list[tuple[float, float]],
    width: int,
    height: int,
    focal: float,
) -> CameraModel | None:
    img_pts_px = np.array(
        [[x * width, y * height] for x, y in image_corners], dtype=np.float64,
    )
    obj_pts_3d = np.array([[c[0], c[1], 0.0] for c in COURT_CORNERS], dtype=np.float64)
    cx_px = width / 2.0
    cy_px = height / 2.0
    K = np.array([[focal, 0, cx_px], [0, focal, cy_px], [0, 0, 1]], dtype=np.float64)
    best: CameraModel | None = None
    for method in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts_3d, img_pts_px, K, distCoeffs=np.zeros(4), flags=method,
            )
        except cv2.error:
            continue
        if not ok:
            continue
        cand = _build_model(
            K, np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
            obj_pts_3d, img_pts_px, width, height,
        )
        if cand is not None and (best is None or cand.reprojection_error < best.reprojection_error):
            best = cand
    return best


def main() -> None:
    print(f"Target video: {TARGET_VID}")
    videos = load_calibrated_videos(TARGET_VID)
    assert TARGET_VID in videos, "video not found"
    vcal = videos[TARGET_VID]
    print(f"  width={vcal.width} height={vcal.height}")
    print(f"  corners={vcal.image_corners}")

    rallies = load_rallies_for_videos({TARGET_VID})
    print(f"  loaded {len(rallies)} rallies")

    # Collect player samples.
    samples: list[tuple[float, float, float, float, float, float]] = []
    # (foot_u, foot_v, head_u, head_v, bbox_width, bbox_height)
    for rally in rallies:
        pp = _parse_player_positions(rally.positions_json)
        if not pp:
            continue
        frames = sorted({p.frame_number for p in pp})
        if not frames:
            continue
        sample_idx = [len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4]
        sample_frames = {frames[i] for i in sample_idx if 0 <= i < len(frames)}
        for p in pp:
            if p.frame_number not in sample_frames:
                continue
            cx = p.x
            top_y = p.y - p.height / 2
            bot_y = p.y + p.height / 2
            if not (0.02 < bot_y < 0.98 and 0.02 < top_y < 0.98 and 0 < cx < 1):
                continue
            samples.append((cx, bot_y, cx, top_y, p.width, p.height))
    print(f"  collected {len(samples)} player samples (no aspect filter)")

    # Subset by aspect ratio bands for diagnostic.
    upright_samples = [s for s in samples if s[5] > s[4] * 1.5]  # height > 1.5*width
    print(f"  upright (h > 1.5*w): {len(upright_samples)}")

    # Sweep focal length.
    print("\n" + "=" * 100)
    print("FOCAL LENGTH SWEEP — full range, no reproj cutoff")
    print("=" * 100)
    print(f"  {'focal':>7s}  {'reproj':>8s}  {'cam_z':>7s}  {'cam_y':>7s}  {'all_med':>9s}  {'upright_med':>13s}  {'upright_p90':>13s}")

    scale = vcal.width / 1920.0
    min_f = 600 * scale
    max_f = 6000 * scale

    rows: list[tuple[float, float, float, float, float, float]] = []
    for i in range(60):
        focal = min_f + (max_f - min_f) * i / 59
        cam = _build_camera_with_focal(
            vcal.image_corners, vcal.width, vcal.height, focal,
        )
        if cam is None:
            continue
        # All samples implied heights.
        all_h: list[float] = []
        for fu, fv, hu, hv, _, _ in samples:
            h = _implied_head_height(cam, fu, fv, hu, hv)
            if h is not None and 0.2 < h < 4.0:
                all_h.append(h)
        all_med = float(np.median(all_h)) if all_h else float("nan")

        upright_h: list[float] = []
        for fu, fv, hu, hv, _, _ in upright_samples:
            h = _implied_head_height(cam, fu, fv, hu, hv)
            if h is not None and 0.2 < h < 4.0:
                upright_h.append(h)
        upright_med = float(np.median(upright_h)) if upright_h else float("nan")
        upright_p90 = float(np.percentile(upright_h, 90)) if upright_h else float("nan")

        cam_pos = cam.camera_position
        rows.append((focal, cam.reprojection_error, float(cam_pos[2]), float(cam_pos[1]), all_med, upright_med))
        print(
            f"  {focal:>7.0f}  {cam.reprojection_error:>7.2f}px  {cam_pos[2]:>6.2f}m  {cam_pos[1]:>6.2f}m  "
            f"{all_med:>8.2f}m  {upright_med:>12.2f}m  {upright_p90:>12.2f}m"
        )

    print("\n=== ANALYSIS ===")
    if not rows:
        print("  No valid candidates")
        return

    # Find the focal where upright_med is closest to 1.75.
    target = 1.75
    best_for_target = min(rows, key=lambda r: abs(r[5] - target))
    print(f"  Closest match to target {target}m upright median:")
    print(f"    focal={best_for_target[0]:.0f}px  reproj={best_for_target[1]:.2f}px  "
          f"cam_z={best_for_target[2]:.2f}m  upright_med={best_for_target[5]:.2f}m")

    # Find the focal with lowest reproj.
    best_reproj = min(rows, key=lambda r: r[1])
    print(f"  Best 4-corner reprojection:")
    print(f"    focal={best_reproj[0]:.0f}px  reproj={best_reproj[1]:.2f}px  "
          f"cam_z={best_reproj[2]:.2f}m  upright_med={best_reproj[5]:.2f}m")

    # Sensitivity: as focal varies, how much does upright_med change?
    upright_meds = [r[5] for r in rows]
    print(f"\n  Implied height range as focal sweeps:")
    print(f"    min={min(upright_meds):.2f}m  max={max(upright_meds):.2f}m  "
          f"spread={max(upright_meds) - min(upright_meds):.2f}m")

    # Reproj range.
    reprojs = [r[1] for r in rows]
    print(f"  Reproj error range:")
    print(f"    min={min(reprojs):.2f}px  max={max(reprojs):.2f}px")


if __name__ == "__main__":
    main()
