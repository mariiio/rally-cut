"""Player-height-aware camera calibration prototype.

Hypothesis: 4-corner solvePnP has a focal-length / camera-height ambiguity
that current code resolves with reprojection minimization alone, biasing
toward low cameras + short focal lengths. Players are a known scale: if we
pick the focal length whose calibration produces realistic implied player
heights, we get a more correct camera pose.

Procedure (per video):
1. Sample player bboxes across all rallies (foot bottom-center, head top-center)
2. Grid search focal length over a wide range
3. For each focal: solvePnP for (R, t); compute (a) 4-corner reproj error,
   (b) median implied player height
4. Pick focal where reproj ≤ 8 px AND |implied_height − 1.75| is smallest
5. Compare camera height + serve speeds to current calibration

Target player height: 1.75m (midpoint of 1.65-1.85 range provided by user).
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
    calibrate_camera,
    calibrate_camera_with_net,
    image_ray,
)
from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from rallycut.court.trajectory_3d import fit_rally  # noqa: E402
from rallycut.evaluation.db import get_connection  # noqa: E402
from eval_ball_3d import (  # noqa: E402
    COURT_CORNERS,
    load_calibrated_videos,
    load_rallies_for_videos,
    _build_contact_sequence,
    _parse_actions,
    _parse_ball_positions,
    _parse_player_positions,
)


SESSION_SHORT_ID = "41e1f30d-d5bb-4386-9908-fa37216eb535"
TARGET_HEIGHT = 1.75   # midpoint of 1.65-1.85m
HEIGHT_TOLERANCE = 0.10  # acceptable deviation from target
REPROJ_TOLERANCE = 8.0   # max acceptable 4-corner reproj error (px)


def _load_session_video_ids(session_id: str) -> set[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id FROM session_videos WHERE session_id = %s",
                (session_id,),
            )
            return {str(row[0]) for row in cur.fetchall()}


def _ray_hit_ground(camera: CameraModel, u_norm: float, v_norm: float) -> np.ndarray | None:
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
    foot_3d = _ray_hit_ground(camera, foot_u, foot_v)
    if foot_3d is None:
        return None
    origin, direction = image_ray(camera, (head_u, head_v))
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


def _collect_player_samples(
    rallies: list[Any],
) -> list[tuple[float, float, float, float]]:
    """Return (foot_u, foot_v, head_u, head_v) tuples sampled from rallies."""
    samples: list[tuple[float, float, float, float]] = []
    for rally in rallies:
        pp = _parse_player_positions(rally.positions_json)
        if not pp:
            continue
        # Sample 3 frames per rally (25%, 50%, 75%).
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
            # Filter to bboxes with plausible aspect ratio (taller than wide,
            # at least somewhat large) — drops noise/diving poses.
            if p.height < 0.05 or p.height < p.width * 1.2:
                continue
            samples.append((cx, bot_y, cx, top_y))
    return samples


def _compute_implied_heights(
    cam: CameraModel,
    samples: list[tuple[float, float, float, float]],
) -> list[float]:
    out: list[float] = []
    for fu, fv, hu, hv in samples:
        h = _implied_head_height(cam, fu, fv, hu, hv)
        if h is not None and 0.3 < h < 3.5:
            out.append(h)
    return out


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


def _player_aware_calibrate(
    image_corners: list[tuple[float, float]],
    width: int,
    height: int,
    samples: list[tuple[float, float, float, float]],
    target_height: float = TARGET_HEIGHT,
) -> tuple[CameraModel | None, list[tuple[float, float, float, float]]]:
    """Grid search focal length, optimising for (a) reproj ≤ tolerance,
    (b) median implied player height closest to target.

    Returns (best_camera, all_candidates) where each candidate is
    (focal, reproj_err, implied_height_median, score).
    """
    scale = width / 1920.0
    # Wider range than default to allow higher focal lengths
    # (which produce taller implied players + higher cameras).
    min_f = 800 * scale
    max_f = 4500 * scale
    n_steps = 60

    candidates: list[tuple[float, float, float, float]] = []  # (f, reproj, h_med, score)
    best: CameraModel | None = None
    best_score = float("inf")

    for i in range(n_steps + 1):
        f = min_f + (max_f - min_f) * i / n_steps
        cam = _build_camera_with_focal(image_corners, width, height, f)
        if cam is None or not cam.is_valid:
            continue
        if cam.reprojection_error > REPROJ_TOLERANCE:
            continue
        heights = _compute_implied_heights(cam, samples)
        if len(heights) < 5:
            continue
        h_med = float(np.median(heights))
        # Score: weighted combination
        # — small reproj error preferred
        # — small distance to target height preferred
        height_err = abs(h_med - target_height)
        score = cam.reprojection_error / 8.0 + height_err * 5.0
        candidates.append((f, cam.reprojection_error, h_med, score))
        if score < best_score:
            best_score = score
            best = cam

    return best, candidates


def main() -> None:
    print("Loading session short videos...")
    session_vids = _load_session_video_ids(SESSION_SHORT_ID)
    print(f"  {len(session_vids)} videos in session short")

    print("Loading rallies + calibrations...")
    videos = load_calibrated_videos(None)
    rallies = load_rallies_for_videos(session_vids)
    rallies_by_video: dict[str, list[Any]] = defaultdict(list)
    for r in rallies:
        rallies_by_video[r.video_id].append(r)

    # Per-video comparison.
    print(f"\n{'='*100}")
    print("PER-VIDEO COMPARISON: current vs player-aware calibration")
    print(f"{'='*100}")
    print(f"  {'video':<10s}  {'current_h':>10s} {'current_f':>10s} {'curr_imp':>9s}    "
          f"{'new_h':>7s} {'new_f':>9s} {'new_imp':>8s} {'new_rep':>8s}  {'delta_h':>8s}")

    new_cameras: dict[str, CameraModel] = {}
    old_cameras: dict[str, CameraModel] = {}
    summary_stats: list[tuple[str, float, float, float, float]] = []

    for vid_id in sorted(session_vids):
        if vid_id not in videos:
            continue
        vcal = videos[vid_id]
        vid_rallies = rallies_by_video.get(vid_id, [])
        if not vid_rallies:
            continue

        # Current calibration (with net constraint).
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

        old_cam = None
        if net_y is not None:
            old_cam = calibrate_camera_with_net(
                vcal.image_corners, COURT_CORNERS, vcal.width, vcal.height,
                net_y_image=net_y,
            )
        if old_cam is None or not old_cam.is_valid:
            old_cam = calibrate_camera(
                vcal.image_corners, COURT_CORNERS, vcal.width, vcal.height,
            )
        if old_cam is None or not old_cam.is_valid:
            continue
        old_cameras[vid_id] = old_cam

        # Sample players for height check.
        samples = _collect_player_samples(vid_rallies)
        if len(samples) < 10:
            continue
        old_heights = _compute_implied_heights(old_cam, samples)
        old_h_med = float(np.median(old_heights)) if old_heights else float("nan")

        # Player-aware calibration.
        new_cam, _candidates = _player_aware_calibrate(
            vcal.image_corners, vcal.width, vcal.height, samples,
        )
        if new_cam is None:
            print(f"  {vid_id[:8]}    {old_cam.camera_position[2]:7.2f}m {old_cam.focal_length_px:8.0f}px {old_h_med:7.2f}m   (no candidate found)")
            continue
        new_cameras[vid_id] = new_cam

        new_heights = _compute_implied_heights(new_cam, samples)
        new_h_med = float(np.median(new_heights)) if new_heights else float("nan")

        delta = new_cam.camera_position[2] - old_cam.camera_position[2]
        print(
            f"  {vid_id[:8]}    {old_cam.camera_position[2]:7.2f}m {old_cam.focal_length_px:8.0f}px {old_h_med:7.2f}m    "
            f"{new_cam.camera_position[2]:5.2f}m {new_cam.focal_length_px:7.0f}px {new_h_med:6.2f}m {new_cam.reprojection_error:6.2f}px  {delta:+6.2f}m"
        )
        summary_stats.append((
            vid_id,
            float(old_cam.camera_position[2]),
            float(new_cam.camera_position[2]),
            old_h_med,
            new_h_med,
        ))

    if not summary_stats:
        print("\nNo videos calibrated successfully")
        return

    arr = np.array([[s[1], s[2], s[3], s[4]] for s in summary_stats])
    print(f"\n{'='*100}")
    print("AGGREGATE COMPARISON (session short)")
    print(f"{'='*100}")
    print(f"  Videos: {len(summary_stats)}")
    print(f"  Current camera height: median={np.median(arr[:, 0]):.2f}m  mean={arr[:, 0].mean():.2f}m  range=[{arr[:, 0].min():.2f}, {arr[:, 0].max():.2f}]")
    print(f"  New camera height:     median={np.median(arr[:, 1]):.2f}m  mean={arr[:, 1].mean():.2f}m  range=[{arr[:, 1].min():.2f}, {arr[:, 1].max():.2f}]")
    print()
    print(f"  Current implied player height: median={np.median(arr[:, 2]):.2f}m")
    print(f"  New implied player height:     median={np.median(arr[:, 3]):.2f}m  (target: 1.75m)")

    # Now re-run trajectory fitting with new cameras and report serve speed deltas.
    print(f"\n{'='*100}")
    print("TRAJECTORY FITTING with NEW cameras (session short)")
    print(f"{'='*100}")

    old_speeds: list[float] = []
    new_speeds: list[float] = []
    old_arcs = 0
    new_arcs = 0

    for vid_id in sorted(new_cameras.keys()):
        if vid_id not in old_cameras:
            continue
        vcal = videos[vid_id]
        vid_rallies = rallies_by_video[vid_id]
        calibrator = CourtCalibrator()
        calibrator.calibrate(vcal.image_corners)

        for rally in vid_rallies:
            bp = _parse_ball_positions(rally.ball_positions_json)
            pp = _parse_player_positions(rally.positions_json)
            if not bp:
                continue
            cs = _build_contact_sequence(bp, pp, calibrator)
            actions = _parse_actions(rally.actions_json)

            traj_old = fit_rally(
                camera=old_cameras[vid_id],
                contact_sequence=cs,
                classified_actions=actions if actions else None,
                fps=rally.fps,
                rally_id=rally.rally_id,
                video_id=vid_id,
                net_height=2.24,
            )
            traj_new = fit_rally(
                camera=new_cameras[vid_id],
                contact_sequence=cs,
                classified_actions=actions if actions else None,
                fps=rally.fps,
                rally_id=rally.rally_id,
                video_id=vid_id,
                net_height=2.24,
            )
            for arc in traj_old.arcs:
                old_arcs += 1
                old_speeds.append(arc.speed_at_start)
            for arc in traj_new.arcs:
                new_arcs += 1
                new_speeds.append(arc.speed_at_start)

    def report(label: str, speeds: list[float], n_arcs: int) -> None:
        if not speeds:
            print(f"  {label:<10s}  no speeds")
            return
        a = np.array(speeds)
        in_range = int(((a >= 10) & (a <= 35)).sum())
        print(f"  {label:<10s}  arcs={n_arcs:>4d}  serves={len(speeds):>4d}  "
              f"in[10,35]={in_range/len(speeds):>4.0%} ({in_range})  "
              f"median={np.median(a):>5.1f}  mean={a.mean():>5.1f}")

    print()
    report("OLD cams", old_speeds, old_arcs)
    report("NEW cams", new_speeds, new_arcs)

    if old_speeds and new_speeds:
        old_in_range = sum(1 for s in old_speeds if 10 <= s <= 35) / len(old_speeds)
        new_in_range = sum(1 for s in new_speeds if 10 <= s <= 35) / len(new_speeds)
        delta = new_in_range - old_in_range
        print(f"\n  In-range delta: {delta:+.1%}")
        print(f"  Median delta:   {np.median(new_speeds) - np.median(old_speeds):+.1f} m/s")


if __name__ == "__main__":
    main()
