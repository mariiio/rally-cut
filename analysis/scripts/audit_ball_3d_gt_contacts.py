"""Ceiling test: run the fitter with ACTION-GT contact frames instead of
contact-detector output, then measure Tier 1-A plausibility.

Motivation: the diagnose_ball_3d_fitter.py run showed 4/5 worst rallies had
only 1 contact detected by the contact detector, causing the fitter to try
to fit 3–4 second spans as single parabolas. Wider multi-start and constraint
ablations do not help when the input isn't actually a parabola. Contact
under-segmentation is the primary failure mode.

This script answers: **if contacts were perfect, would the fitter work?**
If yes, the fix is in contact detection (out of scope for this session,
but the direction is clear). If no, the fitter has additional failure modes
that need work.

Uses:
    1. Action GT to source contacts (available on 340 rallies).
    2. Ball GT to run the plausibility audit (available on 40 rallies).
    3. Intersection: ~40 rallies have both ball GT and action GT — perfect
       for ceiling testing.

Output
------
    outputs/ball_3d_rig/audit_gt_contacts_<date>.md — ceiling vs production
    outputs/ball_3d_rig/audit_gt_contacts_<date>.json — structured data
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH, CourtCalibrator
from rallycut.court.camera_model import (
    CameraModel,
    calibrate_camera,
    calibrate_camera_with_net,
)
from rallycut.court.trajectory_3d import fit_rally
from rallycut.evaluation.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    ContactSequence,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition

COURT_CORNERS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (COURT_WIDTH, 0.0),
    (COURT_WIDTH, COURT_LENGTH),
    (0.0, COURT_LENGTH),
]

# Tier 1-A plausibility gates (same as audit_ball_3d_tier1.py).
PLAUSIBLE_PEAK_MAX_M = 6.0
PLAUSIBLE_LANDING_X_MIN = -2.0
PLAUSIBLE_LANDING_X_MAX = COURT_WIDTH + 2.0
PLAUSIBLE_LANDING_Y_MIN = -3.0
PLAUSIBLE_LANDING_Y_MAX = COURT_LENGTH + 3.0
PLAUSIBLE_SPEED_MIN = 3.0
PLAUSIBLE_SPEED_MAX = 30.0
PLAUSIBLE_RMSE_MAX_PX = 20.0

OUTPUT_DIR = Path("outputs/ball_3d_rig")
NET_TAPE_M = 2.24


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


def _contact_from_gt_label(lab: dict[str, Any]) -> Contact:
    """Build a Contact object from one action GT label."""
    return Contact(
        frame=int(lab["frame"]),
        ball_x=float(lab.get("ballX") or 0.0),
        ball_y=float(lab.get("ballY") or 0.0),
        velocity=0.0,  # unknown from GT
        direction_change_deg=0.0,  # unknown from GT
        player_track_id=int(lab.get("playerTrackId") or -1),
    )


def _build_gt_contact_sequence(
    action_gt: list[dict[str, Any]],
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    detected_net_y: float,
) -> ContactSequence:
    """Construct a ContactSequence with contacts sourced from action GT."""
    contacts = [
        _contact_from_gt_label(lab)
        for lab in sorted(action_gt, key=lambda l: l["frame"])
        if lab.get("action") in {"serve", "receive", "set", "attack", "block", "dig"}
    ]
    first_frame = min(bp.frame_number for bp in ball_positions) if ball_positions else 0
    return ContactSequence(
        contacts=contacts,
        net_y=detected_net_y,
        rally_start_frame=first_frame,
        ball_positions=ball_positions,
        player_positions=player_positions,
    )


def _arc_is_physical(
    peak: float,
    landing: tuple[float, float] | None,
    speed: float,
    rmse_px: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not (0.0 <= peak <= PLAUSIBLE_PEAK_MAX_M):
        reasons.append(f"peak={peak:.1f}m")
    if landing is None:
        reasons.append("no_landing")
    else:
        lx, ly = landing
        if not (PLAUSIBLE_LANDING_X_MIN <= lx <= PLAUSIBLE_LANDING_X_MAX):
            reasons.append(f"land_x={lx:.1f}")
        if not (PLAUSIBLE_LANDING_Y_MIN <= ly <= PLAUSIBLE_LANDING_Y_MAX):
            reasons.append(f"land_y={ly:.1f}")
    if not (PLAUSIBLE_SPEED_MIN <= speed <= PLAUSIBLE_SPEED_MAX):
        reasons.append(f"speed={speed:.1f}m/s")
    if rmse_px >= PLAUSIBLE_RMSE_MAX_PX:
        reasons.append(f"rmse={rmse_px:.0f}px")
    return (len(reasons) == 0, reasons)


def _score_rally(
    rally_id: str,
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    corners: list[tuple[float, float]],
    width: int,
    height: int,
    fps: float,
    action_gt: list[dict[str, Any]] | None,
    video_id: str,
    use_gt_contacts: bool,
) -> dict[str, Any] | None:
    calibrator = CourtCalibrator()
    calibrator.calibrate(corners)
    detected = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=ContactDetectionConfig(),
        court_calibrator=calibrator,
    )
    net_y = detected.net_y if 0.1 < detected.net_y < 0.9 else None

    cam = None
    if net_y is not None:
        cam = calibrate_camera_with_net(
            corners, COURT_CORNERS, width, height, net_y_image=net_y,
        )
    if cam is None or not cam.is_valid:
        cam = calibrate_camera(corners, COURT_CORNERS, width, height)
    if cam is None:
        return None

    if use_gt_contacts and action_gt:
        cs = _build_gt_contact_sequence(
            action_gt=action_gt,
            ball_positions=ball_positions,
            player_positions=player_positions,
            detected_net_y=detected.net_y,
        )
    else:
        cs = detected

    if not cs.contacts:
        return None

    traj = fit_rally(
        camera=cam, contact_sequence=cs, classified_actions=None,
        fps=fps, rally_id=rally_id, video_id=video_id,
        net_height=NET_TAPE_M, joint=True,
    )
    if not traj.arcs:
        return None

    n_physical = 0
    n_is_valid = 0
    for arc in traj.arcs:
        is_phys, _ = _arc_is_physical(
            peak=arc.peak_height,
            landing=arc.landing_position,
            speed=arc.speed_at_start,
            rmse_px=arc.reprojection_rmse,
        )
        if is_phys:
            n_physical += 1
        if arc.is_valid:
            n_is_valid += 1

    plausible_rate = n_physical / len(traj.arcs)
    serve_arc = traj.arcs[0]
    serve_phys, _ = _arc_is_physical(
        serve_arc.peak_height, serve_arc.landing_position,
        serve_arc.speed_at_start, serve_arc.reprojection_rmse,
    )

    return {
        "rally_id": rally_id,
        "n_contacts": len(cs.contacts),
        "n_arcs": len(traj.arcs),
        "n_physical": n_physical,
        "n_is_valid": n_is_valid,
        "plausible_rate": plausible_rate,
        "serve_physical": serve_phys,
        "camera_height_m": float(cam.camera_position[2]),
    }


def main() -> None:
    # Load the 40 rallies with dense ball GT + action GT.
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                r.id, r.video_id, pt.fps,
                pt.ball_positions_json, pt.positions_json,
                pt.action_ground_truth_json,
                v.court_calibration_json, v.width, v.height
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE pt.ground_truth_json IS NOT NULL
              AND pt.action_ground_truth_json IS NOT NULL
              AND pt.ball_positions_json IS NOT NULL
              AND v.court_calibration_json IS NOT NULL
        """)
        rows = cur.fetchall()

    print(f"Loaded {len(rows)} rallies with both ball GT and action GT")

    detected_results: list[dict[str, Any]] = []
    gt_results: list[dict[str, Any]] = []

    for row in rows:
        rid = str(row[0])
        vid = str(row[1])
        fps = float(row[2] or 30.0)
        ball_positions = _parse_ball(row[3])
        player_positions = _parse_players(row[4])
        action_gt = row[5] or []
        cal_json = row[6]
        if not isinstance(cal_json, list) or len(cal_json) != 4:
            continue
        corners = [(c["x"], c["y"]) for c in cal_json]
        width = int(row[7] or 1920)
        height = int(row[8] or 1080)

        det = _score_rally(
            rid, ball_positions, player_positions, corners, width, height,
            fps, action_gt, vid, use_gt_contacts=False,
        )
        gt = _score_rally(
            rid, ball_positions, player_positions, corners, width, height,
            fps, action_gt, vid, use_gt_contacts=True,
        )
        if det is None or gt is None:
            continue
        detected_results.append(det)
        gt_results.append(gt)

        print(
            f"  {rid[:10]}  detected: {det['n_contacts']}c/{det['n_arcs']}a "
            f"phys={det['n_physical']}/{det['n_arcs']}  "
            f"  GT: {gt['n_contacts']}c/{gt['n_arcs']}a "
            f"phys={gt['n_physical']}/{gt['n_arcs']}"
        )

    def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {}
        n = len(results)
        shippable = sum(1 for r in results if r["plausible_rate"] >= 0.80)
        serve_ok = sum(1 for r in results if r["serve_physical"])
        total_arcs = sum(r["n_arcs"] for r in results)
        total_physical = sum(r["n_physical"] for r in results)
        return {
            "n_rallies": n,
            "shippable_rate": shippable / n,
            "shippable_n": f"{shippable}/{n}",
            "serve_pass_rate": serve_ok / n,
            "serve_pass_n": f"{serve_ok}/{n}",
            "arc_physical_rate": total_physical / total_arcs if total_arcs else 0.0,
            "arc_physical_n": f"{total_physical}/{total_arcs}",
            "mean_contacts": float(np.mean([r["n_contacts"] for r in results])),
            "mean_arcs": float(np.mean([r["n_arcs"] for r in results])),
        }

    det_summary = _summarise(detected_results)
    gt_summary = _summarise(gt_results)

    def _pct(x: float) -> str:
        return f"{x * 100:.1f}%" if x is not None else "—"

    print()
    print("=" * 70)
    print("CEILING TEST — DETECTED CONTACTS vs ACTION-GT CONTACTS")
    print("=" * 70)
    print(f"{'':<35s} {'DETECTED':>15s} {'GT':>15s}")
    for metric in (
        "shippable_n", "shippable_rate",
        "serve_pass_n", "serve_pass_rate",
        "arc_physical_n", "arc_physical_rate",
        "mean_contacts", "mean_arcs",
    ):
        dv = det_summary.get(metric, "—")
        gv = gt_summary.get(metric, "—")
        if isinstance(dv, float):
            dv = _pct(dv) if "rate" in metric else f"{dv:.2f}"
        if isinstance(gv, float):
            gv = _pct(gv) if "rate" in metric else f"{gv:.2f}"
        print(f"{metric:<35s} {dv!s:>15s} {gv!s:>15s}")
    print()

    date = datetime.now().strftime("%Y-%m-%d")
    out = OUTPUT_DIR / f"audit_gt_contacts_{date}.json"
    out.write_text(json.dumps({
        "detected": {"summary": det_summary, "per_rally": detected_results},
        "gt": {"summary": gt_summary, "per_rally": gt_results},
    }, indent=2, default=float))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
