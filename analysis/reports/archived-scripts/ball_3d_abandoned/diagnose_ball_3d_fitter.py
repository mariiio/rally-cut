"""Diagnostic for ball 3D fitter failures.

Picks rallies where the fitter produced 0% physical arcs per the audit report
and instruments the pipeline to report:

    1. Contact detection output: how many contacts, at what frames, are there
       sensible gaps (multiple arcs) or just 1-2 contacts (oversegmentation)?
    2. Per-arc observation count and time span.
    3. Per-arc multi-start analysis: what does _multi_start_guesses produce?
       Which start wins, what RMSE did it achieve vs. others?
    4. Alternative multi-start analysis: if we try 64 much more diverse
       starts, does any of them beat the current best?
    5. Constraint ablation: what RMSE do we get WITHOUT net/landing penalties?

The goal is to identify the dominant failure mode: contact segmentation
bugs, narrow multi-start, or constraint interference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
from scipy.optimize import least_squares

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH, CourtCalibrator
from rallycut.court.camera_model import (
    CameraModel,
    calibrate_camera,
    calibrate_camera_with_net,
    project_3d_to_image,
)
from rallycut.court.trajectory_3d import (
    GRAVITY,
    _collect_observations,
    _multi_start_guesses,
    _residuals,
    fit_rally,
)
from rallycut.evaluation.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition

COURT_CORNERS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (COURT_WIDTH, 0.0),
    (COURT_WIDTH, COURT_LENGTH),
    (0.0, COURT_LENGTH),
]

# Worst 5 rallies from the audit report (0% plausibility).
WORST_RALLIES = [
    "0d84f858",
    "9dbe457a",
    "0a376585",
    "2dff5eeb",
    "1bfcbc4f",
]


def _parse_ball(bp_json: Any) -> list[BallPosition]:
    if not bp_json:
        return []
    positions = bp_json.get("positions", []) if isinstance(bp_json, dict) else bp_json
    return [
        BallPosition(
            frame_number=p.get("frameNumber", 0),
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            confidence=p.get("confidence", 0.5),
        )
        for p in positions
    ]


def _parse_players(pos_json: Any) -> list[PlayerPosition]:
    if not pos_json:
        return []
    return [
        PlayerPosition(
            frame_number=p.get("frameNumber", 0),
            track_id=p.get("trackId", 0),
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            width=p.get("width", 0.0),
            height=p.get("height", 0.0),
            confidence=p.get("confidence", 0.5),
            keypoints=p.get("keypoints"),
        )
        for p in pos_json
    ]


def load_rally(rally_prefix: str) -> dict[str, Any] | None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.video_id, pt.fps,
                   pt.ball_positions_json, pt.positions_json,
                   v.court_calibration_json, v.width, v.height
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE r.id LIKE %s
            LIMIT 1
        """, (rally_prefix + "%",))
        row = cur.fetchone()
    if not row:
        return None
    return {
        "rally_id": str(row[0]),
        "video_id": str(row[1]),
        "fps": float(row[2] or 30.0),
        "ball_positions": _parse_ball(row[3]),
        "player_positions": _parse_players(row[4]),
        "calibration": [(c["x"], c["y"]) for c in row[5]] if row[5] else None,
        "width": int(row[6] or 1920),
        "height": int(row[7] or 1080),
    }


def build_camera(rally: dict[str, Any]) -> tuple[CameraModel, CourtCalibrator] | None:
    calibrator = CourtCalibrator()
    calibrator.calibrate(rally["calibration"])
    cs = detect_contacts(
        ball_positions=rally["ball_positions"],
        player_positions=rally["player_positions"],
        config=ContactDetectionConfig(),
        court_calibrator=calibrator,
    )
    net_y = cs.net_y if 0.1 < cs.net_y < 0.9 else None
    cam = None
    if net_y is not None:
        cam = calibrate_camera_with_net(
            rally["calibration"], COURT_CORNERS, rally["width"], rally["height"],
            net_y_image=net_y,
        )
    if cam is None or not cam.is_valid:
        cam = calibrate_camera(
            rally["calibration"], COURT_CORNERS, rally["width"], rally["height"],
        )
    if cam is None:
        return None
    return cam, calibrator


def fit_one_guess(
    camera: CameraModel,
    times: np.ndarray,
    uv: np.ndarray,
    weights: np.ndarray,
    p0: np.ndarray,
    loss: str = "soft_l1",
    f_scale: float = 0.01,
    use_constraints: bool = True,
) -> tuple[np.ndarray | None, float]:
    lb = np.array([-10, -10, 0.0, -35, -35, -20])
    ub = np.array([18, 26, 6.0, 35, 35, 20])
    x0 = np.clip(p0, lb + 1e-6, ub - 1e-6)

    net_c = (8.0, 2.24, 4.5) if use_constraints else None
    try:
        result = least_squares(
            _residuals, x0,
            args=(camera, times, uv, weights, False, net_c, False),
            bounds=(lb, ub),
            loss=loss,
            f_scale=f_scale,
            max_nfev=2000,
        )
    except Exception:  # noqa: BLE001
        return None, float("inf")

    raw = _residuals(result.x, camera, times, uv, np.ones(len(times)), False)
    rmse_norm = float(np.sqrt(np.mean(raw ** 2)))
    rmse_px = rmse_norm * max(camera.image_size)
    return result.x, rmse_px


def wide_multi_start_guesses(
    camera: CameraModel,
    uv: np.ndarray,
    times: np.ndarray,
    z0_prior: float,
) -> list[np.ndarray]:
    """Generate a much wider grid of initial guesses (64 starts)."""
    dt = float(times[-1])
    if dt < 1e-6:
        return []

    guesses: list[np.ndarray] = []
    # Sweep x_start ∈ {0, 2, 4, 6, 8}, y_start ∈ {0, 4, 8, 12, 16}, z0 ∈ {1.0, 2.5, 4.0},
    # and for direction: allow both forward and backward in y.
    # Use image x as a rough lateral position but sweep the depth.
    x_image_start = float(uv[0, 0]) * COURT_WIDTH
    x_image_end = float(uv[-1, 0]) * COURT_WIDTH
    vx = (x_image_end - x_image_start) / dt

    for y_start in [1.0, 4.0, 8.0, 12.0, 15.0]:
        for y_end in [1.0, 4.0, 8.0, 12.0, 15.0]:
            for z0 in [1.5, 3.0, 4.0]:
                vy = (y_end - y_start) / dt
                # Estimate vz from a rough peak at mid-flight.
                vz = 0.5 * GRAVITY * dt
                guesses.append(np.array([
                    x_image_start, y_start, z0, vx, vy, vz,
                ]))

    # Also include a few explicit "directly at camera" and "away from camera" starts
    # for very short arcs.
    for y_start, y_end in [(0.0, 16.0), (16.0, 0.0), (4.0, 12.0), (12.0, 4.0)]:
        for z0 in [2.0, 4.0]:
            vy = (y_end - y_start) / dt
            vz = 0.5 * GRAVITY * dt
            guesses.append(np.array([x_image_start, y_start, z0, vx, vy, vz]))

    return guesses


def diagnose_rally(rally_prefix: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"Rally prefix: {rally_prefix}")
    print("=" * 78)
    rally = load_rally(rally_prefix)
    if not rally:
        print("  NOT FOUND")
        return

    cam_pair = build_camera(rally)
    if cam_pair is None:
        print("  Camera calibration failed.")
        return
    camera, calibrator = cam_pair
    print(f"  rally_id: {rally['rally_id']}")
    print(f"  video_id: {rally['video_id']}")
    print(f"  camera_height: {camera.camera_position[2]:.2f}m  focal: {camera.focal_length_px:.0f}px")
    print(f"  ball observations: {len(rally['ball_positions'])}")

    # 1. Contact detection output.
    cs = detect_contacts(
        ball_positions=rally["ball_positions"],
        player_positions=rally["player_positions"],
        config=ContactDetectionConfig(),
        court_calibrator=calibrator,
    )
    contacts = cs.contacts
    print(f"\n  --- Contact detection ---")
    print(f"  {len(contacts)} contacts detected:")
    for i, c in enumerate(contacts):
        print(f"    contact[{i}]  frame={c.frame}  player_track={c.player_track_id}")
    last_frame = max(bp.frame_number for bp in rally["ball_positions"])
    first_frame = min(bp.frame_number for bp in rally["ball_positions"])
    print(f"  ball frame range: [{first_frame}, {last_frame}]  span={last_frame - first_frame} frames")

    if len(contacts) < 2:
        print("  *** ONLY 1 CONTACT — fitter will try to fit the entire rally as one arc ***")

    # 2-5. Per-arc analysis.
    arc_specs = []
    for i in range(len(contacts) - 1):
        arc_specs.append((contacts[i].frame, contacts[i + 1].frame, 2.0, False))
    if contacts:
        arc_specs.append((contacts[-1].frame, last_frame, 2.0, True))
    # If no contacts, fit the full range as one arc.
    if not arc_specs:
        arc_specs.append((first_frame, last_frame, 2.0, True))

    for arc_idx, (sf, ef, z0, _) in enumerate(arc_specs):
        print(f"\n  --- Arc {arc_idx}: frames [{sf}, {ef}] ({ef - sf} frames, "
              f"{(ef - sf) / rally['fps']:.2f}s) ---")
        obs_arr = _collect_observations(rally["ball_positions"], sf, ef, rally["fps"])
        if obs_arr is None:
            print("    <5 observations — skipped")
            continue
        times = obs_arr[:, 0]
        uv = obs_arr[:, 1:3]
        weights = obs_arr[:, 3]
        print(f"    {len(times)} observations")
        print(f"    image u range: [{uv[:, 0].min():.3f}, {uv[:, 0].max():.3f}]")
        print(f"    image v range: [{uv[:, 1].min():.3f}, {uv[:, 1].max():.3f}]")

        # A. Current multi-start (9 guesses).
        guesses = _multi_start_guesses(camera, uv, times, z0)
        current_results = []
        for g in guesses:
            params, rmse_px = fit_one_guess(camera, times, uv, weights, g)
            current_results.append((g, params, rmse_px))
        current_results.sort(key=lambda r: r[2])
        best_current = current_results[0]
        print(f"    CURRENT multi-start (9): best RMSE = {best_current[2]:.1f}px")
        print(f"      best start: pos0={best_current[0][:3].round(1)} vel0={best_current[0][3:].round(1)}")
        if best_current[1] is not None:
            print(f"      best result: pos0={best_current[1][:3].round(1)} vel0={best_current[1][3:].round(1)}")
            print(f"      speed={np.linalg.norm(best_current[1][3:]):.1f}m/s")

        # B. Wider multi-start (64+ guesses).
        wide_guesses = wide_multi_start_guesses(camera, uv, times, z0)
        wide_results = []
        for g in wide_guesses:
            params, rmse_px = fit_one_guess(camera, times, uv, weights, g)
            wide_results.append((g, params, rmse_px))
        wide_results.sort(key=lambda r: r[2])
        best_wide = wide_results[0]
        improvement = best_current[2] - best_wide[2]
        print(f"    WIDER multi-start ({len(wide_guesses)}): best RMSE = {best_wide[2]:.1f}px  "
              f"(Δ = {-improvement:+.1f}px)")
        if best_wide[1] is not None:
            print(f"      best result: pos0={best_wide[1][:3].round(1)} vel0={best_wide[1][3:].round(1)}")
            print(f"      speed={np.linalg.norm(best_wide[1][3:]):.1f}m/s")

        # C. Constraint-ablation on current multi-start.
        no_constraint_results = []
        for g in guesses:
            params, rmse_px = fit_one_guess(
                camera, times, uv, weights, g, use_constraints=False,
            )
            no_constraint_results.append((g, params, rmse_px))
        no_constraint_results.sort(key=lambda r: r[2])
        best_nc = no_constraint_results[0]
        ncimprovement = best_current[2] - best_nc[2]
        print(f"    NO CONSTRAINTS (9 starts): best RMSE = {best_nc[2]:.1f}px  "
              f"(Δ vs current = {-ncimprovement:+.1f}px)")
        if best_nc[1] is not None:
            print(f"      pos0={best_nc[1][:3].round(1)} vel0={best_nc[1][3:].round(1)}")
            print(f"      speed={np.linalg.norm(best_nc[1][3:]):.1f}m/s")

        # D. Loss ablation: linear L2 instead of soft_l1.
        l2_results = []
        for g in guesses:
            params, rmse_px = fit_one_guess(
                camera, times, uv, weights, g, loss="linear",
            )
            l2_results.append((g, params, rmse_px))
        l2_results.sort(key=lambda r: r[2])
        best_l2 = l2_results[0]
        l2improvement = best_current[2] - best_l2[2]
        print(f"    L2 LOSS (9 starts): best RMSE = {best_l2[2]:.1f}px  "
              f"(Δ vs current = {-l2improvement:+.1f}px)")


def main() -> None:
    for prefix in WORST_RALLIES:
        diagnose_rally(prefix)


if __name__ == "__main__":
    main()
