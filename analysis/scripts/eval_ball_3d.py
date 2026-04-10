"""Evaluate monocular 3D ball trajectory estimation.

Runs the full pipeline (camera calibration → parabolic trajectory fitting)
on all calibrated videos with action GT and reports physics-based sanity
metrics for a go/no-go decision.

Usage:
    cd analysis
    uv run python scripts/eval_ball_3d.py
    uv run python scripts/eval_ball_3d.py --video-id <id>   # single video
    uv run python scripts/eval_ball_3d.py --verbose          # per-arc detail
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH, CourtCalibrator  # noqa: E402
from rallycut.court.camera_model import (  # noqa: E402
    CameraModel,
    calibrate_camera,
    calibrate_camera_with_net,
)
from rallycut.court.trajectory_3d import FittedArc, fit_rally  # noqa: E402
from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.tracking.action_classifier import ClassifiedAction  # noqa: E402
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    ContactDetectionConfig,
    ContactSequence,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

# Beach volleyball court corners (metres, z=0).
COURT_CORNERS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (COURT_WIDTH, 0.0),
    (COURT_WIDTH, COURT_LENGTH),
    (0.0, COURT_LENGTH),
]

# Ship-gate thresholds.
GATE_CAM_SUCCESS = 0.90
GATE_SERVE_SPEED = 0.80   # fraction in [10, 35] m/s
GATE_NET_CROSSING = 0.70  # fraction in [2.24, 5.0] m
GATE_LANDING_Z = 0.70     # fraction with |Z| < 0.5m
GATE_GRAVITY = 0.70       # fraction with |g_residual| < 0.30
GATE_CONTACT_HEIGHT = 0.80  # fraction in [0, 4] m
GATE_REPROJ_RMSE = 5.0    # median px


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class VideoCalibration:
    """Calibration data for one video."""
    video_id: str
    image_corners: list[tuple[float, float]]
    width: int
    height: int


@dataclass
class RallyRow:
    """One rally's data loaded from DB."""
    rally_id: str
    video_id: str
    ball_positions_json: Any
    positions_json: Any
    contacts_json: Any
    actions_json: Any
    fps: float
    frame_count: int


def load_calibrated_videos(video_id: str | None = None) -> dict[str, VideoCalibration]:
    """Load all videos with court calibration."""
    query = """
        SELECT id, court_calibration_json, width, height
        FROM videos
        WHERE court_calibration_json IS NOT NULL
    """
    params: list[str] = []
    if video_id:
        query += " AND id = %s"
        params.append(video_id)

    result: dict[str, VideoCalibration] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                vid_id, cal_json, w, h = row
                if not isinstance(cal_json, list) or len(cal_json) != 4:
                    continue
                corners = [(c["x"], c["y"]) for c in cal_json]
                result[str(vid_id)] = VideoCalibration(
                    video_id=str(vid_id),
                    image_corners=corners,
                    width=w or 1920,
                    height=h or 1080,
                )
    return result


def load_rallies_for_videos(video_ids: set[str]) -> list[RallyRow]:
    """Load rallies with tracked data for a set of videos."""
    if not video_ids:
        return []

    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT
            r.id, r.video_id,
            pt.ball_positions_json, pt.positions_json,
            pt.contacts_json, pt.actions_json,
            pt.fps, pt.frame_count
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id IN ({placeholders})
          AND pt.ball_positions_json IS NOT NULL
        ORDER BY r.video_id, r.start_ms
    """

    result: list[RallyRow] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            for row in cur.fetchall():
                result.append(RallyRow(
                    rally_id=str(row[0]),
                    video_id=str(row[1]),
                    ball_positions_json=row[2],
                    positions_json=row[3],
                    contacts_json=row[4],
                    actions_json=row[5],
                    fps=row[6] or 30.0,
                    frame_count=row[7] or 0,
                ))
    return result


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _parse_ball_positions(bp_json: Any) -> list[BallPosition]:
    """Parse ball_positions_json → list[BallPosition]."""
    if not bp_json:
        return []
    positions = bp_json.get("positions", []) if isinstance(bp_json, dict) else bp_json
    result: list[BallPosition] = []
    for p in positions:
        result.append(BallPosition(
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            confidence=p.get("confidence", 0.5),
        ))
    return result


def _parse_player_positions(pos_json: Any) -> list[PlayerPosition]:
    """Parse positions_json → list[PlayerPosition]."""
    if not pos_json:
        return []
    result: list[PlayerPosition] = []
    for p in pos_json:
        result.append(PlayerPosition(
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            track_id=p.get("trackId", p.get("track_id", 0)),
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            width=p.get("width", 0.0),
            height=p.get("height", 0.0),
            confidence=p.get("confidence", 0.5),
        ))
    return result


def _parse_actions(actions_json: Any) -> list[ClassifiedAction]:
    """Parse actions_json → list[ClassifiedAction]."""
    if not actions_json:
        return []
    from rallycut.tracking.action_classifier import ActionType

    # actions_json is a dict: {"actions": {"actions": [...]}}
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
                frame=a.get("frame", 0),
                ball_x=a.get("ballX", 0.0),
                ball_y=a.get("ballY", 0.0),
                velocity=a.get("velocity", 0.0),
                player_track_id=a.get("playerTrackId", -1),
                court_side=a.get("courtSide", "unknown"),
                confidence=a.get("confidence", 0.0),
                team=a.get("team", "unknown"),
            ))
        except (KeyError, ValueError):
            continue
    return result


def _build_contact_sequence(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    calibrator: CourtCalibrator | None,
) -> ContactSequence:
    """Run contact detection on ball/player positions."""
    config = ContactDetectionConfig()
    return detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=config,
        court_calibrator=calibrator,
    )


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

@dataclass
class ArcMetrics:
    """Aggregated metrics across all fitted arcs."""
    total_arcs: int = 0
    valid_arcs: int = 0
    serve_speeds: list[float] = field(default_factory=list)
    net_crossings: list[float] = field(default_factory=list)
    landing_zs: list[float] = field(default_factory=list)
    gravity_residuals: list[float] = field(default_factory=list)
    contact_heights: list[float] = field(default_factory=list)
    reproj_rmses: list[float] = field(default_factory=list)


def _collect_arc_metrics(arc: FittedArc, is_serve: bool, metrics: ArcMetrics) -> None:
    """Accumulate metrics from a single fitted arc."""
    metrics.total_arcs += 1
    if arc.is_valid:
        metrics.valid_arcs += 1

    if is_serve:
        metrics.serve_speeds.append(arc.speed_at_start)

    if arc.net_crossing_height is not None:
        metrics.net_crossings.append(arc.net_crossing_height)

    if arc.gravity_residual is not None:
        metrics.gravity_residuals.append(arc.gravity_residual)

    metrics.contact_heights.append(arc.initial_position[2])
    metrics.reproj_rmses.append(arc.reprojection_rmse)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    video_id: str | None = None,
    verbose: bool = False,
    joint: bool = True,
) -> dict[str, Any]:
    """Run 3D trajectory evaluation on all calibrated videos."""
    start_time = time.time()

    # --- Load data -----------------------------------------------------------
    print("Loading calibrated videos...")
    videos = load_calibrated_videos(video_id)
    print(f"  Found {len(videos)} calibrated video(s)")

    print("Loading rallies...")
    rallies = load_rallies_for_videos(set(videos.keys()))
    print(f"  Found {len(rallies)} rally(ies) with ball tracking data")

    # Group rallies by video.
    rallies_by_video: dict[str, list[RallyRow]] = defaultdict(list)
    for r in rallies:
        rallies_by_video[r.video_id].append(r)

    # --- Estimate net Y per video from ball trajectories -----------------------
    print("\n=== NET-Y ESTIMATION ===")
    net_y_per_video: dict[str, float] = {}

    for vid_id, vid_rallies in sorted(rallies_by_video.items()):
        if vid_id not in videos:
            continue

        net_ys: list[float] = []
        vcal = videos[vid_id]
        calibrator = CourtCalibrator()
        calibrator.calibrate(vcal.image_corners)

        for rally in vid_rallies[:15]:
            bp = _parse_ball_positions(rally.ball_positions_json)
            pp = _parse_player_positions(rally.positions_json)
            if len(bp) < 20:
                continue
            cs = _build_contact_sequence(bp, pp, calibrator)
            if cs.net_y > 0.1 and cs.net_y < 0.9:
                net_ys.append(cs.net_y)

        if net_ys:
            net_y_per_video[vid_id] = float(np.median(net_ys))

    print(f"  Estimated net Y for {len(net_y_per_video)}/{len(videos)} videos")
    if net_y_per_video:
        vals = list(net_y_per_video.values())
        print(f"  Net Y range: [{min(vals):.3f}, {max(vals):.3f}], median={np.median(vals):.3f}")

    # --- Camera calibration with net constraint --------------------------------
    print("\n=== CAMERA CALIBRATION (with net constraint) ===")
    cameras: dict[str, CameraModel] = {}
    cam_failures: list[str] = []

    for i, (vid_id, vcal) in enumerate(sorted(videos.items()), 1):
        cam: CameraModel | None = None

        # Try net-constrained calibration first.
        if vid_id in net_y_per_video:
            cam = calibrate_camera_with_net(
                vcal.image_corners, COURT_CORNERS,
                vcal.width, vcal.height,
                net_y_image=net_y_per_video[vid_id],
            )

        # Fallback: 4-corner-only calibration.
        if cam is None or not cam.is_valid:
            cam = calibrate_camera(
                vcal.image_corners, COURT_CORNERS, vcal.width, vcal.height,
            )

        if cam is not None and cam.is_valid:
            cameras[vid_id] = cam
            if verbose:
                print(
                    f"  [{i}/{len(videos)}] {vid_id[:12]}... "
                    f"cam_height={cam.camera_position[2]:.1f}m "
                    f"f={cam.focal_length_px:.0f}px "
                    f"reproj={cam.reprojection_error:.2f}px"
                )
        else:
            cam_failures.append(vid_id)
            err = cam.reprojection_error if cam else float("inf")
            print(f"  [{i}/{len(videos)}] {vid_id[:12]}... FAILED (reproj={err:.1f}px)")

    cam_success_rate = len(cameras) / len(videos) if videos else 0
    print(f"\nCamera success: {len(cameras)}/{len(videos)} ({cam_success_rate:.0%})")

    if cameras:
        heights = [c.camera_position[2] for c in cameras.values()]
        focals = [c.focal_length_px for c in cameras.values()]
        print(f"  Heights: min={min(heights):.1f} median={np.median(heights):.1f} max={max(heights):.1f} m")
        print(f"  Focals:  min={min(focals):.0f} median={np.median(focals):.0f} max={max(focals):.0f} px")

    # --- Trajectory fitting --------------------------------------------------
    print("\n=== TRAJECTORY FITTING ===")
    metrics = ArcMetrics()
    n_rallies_processed = 0
    n_rallies_skipped = 0
    per_video_stats: dict[str, dict[str, Any]] = {}

    for vid_idx, (vid_id, vid_rallies) in enumerate(sorted(rallies_by_video.items()), 1):
        if vid_id not in cameras:
            n_rallies_skipped += len(vid_rallies)
            continue

        camera = cameras[vid_id]
        vcal = videos[vid_id]
        vid_arcs = 0
        vid_valid = 0
        vid_serve_speeds: list[float] = []

        # Build a calibrator for contact detection.
        calibrator = CourtCalibrator()
        calibrator.calibrate(vcal.image_corners)

        for rally in vid_rallies:
            ball_positions = _parse_ball_positions(rally.ball_positions_json)
            player_positions = _parse_player_positions(rally.positions_json)

            if not ball_positions:
                n_rallies_skipped += 1
                continue

            # Run contact detection.
            contact_seq = _build_contact_sequence(
                ball_positions, player_positions, calibrator,
            )

            # Parse existing actions for z0 priors.
            actions = _parse_actions(rally.actions_json)

            # Fit 3D trajectory with geometric constraints.
            traj = fit_rally(
                camera=camera,
                contact_sequence=contact_seq,
                classified_actions=actions if actions else None,
                fps=rally.fps,
                rally_id=rally.rally_id,
                video_id=vid_id,
                net_height=2.24,
                joint=joint,
            )

            for idx, arc in enumerate(traj.arcs):
                is_serve = idx == 0
                _collect_arc_metrics(arc, is_serve, metrics)
                vid_arcs += 1
                if arc.is_valid:
                    vid_valid += 1
                if is_serve:
                    vid_serve_speeds.append(arc.speed_at_start)

                if verbose:
                    status = "✓" if arc.is_valid else "✗"
                    print(
                        f"    {rally.rally_id[:10]}.. arc {idx}: "
                        f"{status} speed={arc.speed_at_start:.1f}m/s "
                        f"h0={arc.initial_position[2]:.1f}m "
                        f"peak={arc.peak_height:.1f}m "
                        f"rmse={arc.reprojection_rmse:.1f}px "
                        f"obs={arc.num_observations}"
                    )

            n_rallies_processed += 1

        per_video_stats[vid_id] = {
            "arcs": vid_arcs,
            "valid": vid_valid,
            "serve_speeds": vid_serve_speeds,
        }

        pct = f"{vid_valid}/{vid_arcs}" if vid_arcs else "0/0"
        speeds_str = ""
        if vid_serve_speeds:
            speeds_str = f", serves: {np.mean(vid_serve_speeds):.1f}±{np.std(vid_serve_speeds):.1f} m/s"
        print(f"  [{vid_idx}/{len(rallies_by_video)}] {vid_id[:12]}... arcs {pct}{speeds_str}")

    print(f"\nProcessed {n_rallies_processed} rallies, skipped {n_rallies_skipped}")

    # --- Report --------------------------------------------------------------
    report = _compute_report(metrics, cam_success_rate, len(cameras), len(videos))
    _print_report(report)

    # Save to file.
    ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_dir = Path("outputs/ball_3d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{ts}.json"
    with out_path.open("w") as f:
        json.dump(_serialise_report(report, per_video_stats), f, indent=2)
    print(f"\nResults saved to {out_path}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")

    return report


def _compute_report(
    m: ArcMetrics,
    cam_success_rate: float,
    n_cam_ok: int,
    n_cam_total: int,
) -> dict[str, Any]:
    """Compute ship-gate metrics."""
    report: dict[str, Any] = {
        "camera_success": {
            "value": cam_success_rate,
            "n_ok": n_cam_ok,
            "n_total": n_cam_total,
            "gate": GATE_CAM_SUCCESS,
            "pass": cam_success_rate >= GATE_CAM_SUCCESS,
        },
        "total_arcs": m.total_arcs,
        "valid_arcs": m.valid_arcs,
    }

    # Serve speed in [10, 35] m/s.
    if m.serve_speeds:
        in_range = sum(1 for s in m.serve_speeds if 10 <= s <= 35)
        frac = in_range / len(m.serve_speeds)
        report["serve_speed"] = {
            "value": frac,
            "n_ok": in_range,
            "n_total": len(m.serve_speeds),
            "gate": GATE_SERVE_SPEED,
            "pass": frac >= GATE_SERVE_SPEED,
            "mean": float(np.mean(m.serve_speeds)),
            "std": float(np.std(m.serve_speeds)),
            "median": float(np.median(m.serve_speeds)),
            "min": float(np.min(m.serve_speeds)),
            "max": float(np.max(m.serve_speeds)),
        }

    # Net crossing in [2.24, 5.0] m.
    if m.net_crossings:
        in_range = sum(1 for h in m.net_crossings if 2.24 <= h <= 5.0)
        frac = in_range / len(m.net_crossings)
        report["net_crossing"] = {
            "value": frac,
            "n_ok": in_range,
            "n_total": len(m.net_crossings),
            "gate": GATE_NET_CROSSING,
            "pass": frac >= GATE_NET_CROSSING,
            "mean": float(np.mean(m.net_crossings)),
            "median": float(np.median(m.net_crossings)),
        }

    # Gravity residual |g_fitted/9.81 - 1| < 0.30.
    if m.gravity_residuals:
        in_range = sum(1 for r in m.gravity_residuals if abs(r) < 0.30)
        frac = in_range / len(m.gravity_residuals)
        report["gravity_residual"] = {
            "value": frac,
            "n_ok": in_range,
            "n_total": len(m.gravity_residuals),
            "gate": GATE_GRAVITY,
            "pass": frac >= GATE_GRAVITY,
            "mean_abs": float(np.mean(np.abs(m.gravity_residuals))),
        }

    # Contact height in [0, 4] m.
    if m.contact_heights:
        in_range = sum(1 for h in m.contact_heights if 0 <= h <= 4)
        frac = in_range / len(m.contact_heights)
        report["contact_height"] = {
            "value": frac,
            "n_ok": in_range,
            "n_total": len(m.contact_heights),
            "gate": GATE_CONTACT_HEIGHT,
            "pass": frac >= GATE_CONTACT_HEIGHT,
            "mean": float(np.mean(m.contact_heights)),
            "median": float(np.median(m.contact_heights)),
        }

    # Reprojection RMSE median < 5 px.
    if m.reproj_rmses:
        median_rmse = float(np.median(m.reproj_rmses))
        report["reproj_rmse"] = {
            "value": median_rmse,
            "gate": GATE_REPROJ_RMSE,
            "pass": median_rmse < GATE_REPROJ_RMSE,
            "mean": float(np.mean(m.reproj_rmses)),
            "p90": float(np.percentile(m.reproj_rmses, 90)),
        }

    # Overall verdict.
    gates = ["camera_success", "serve_speed", "net_crossing", "gravity_residual",
             "contact_height", "reproj_rmse"]
    all_pass = all(
        report.get(g, {}).get("pass", False) for g in gates if g in report
    )
    report["verdict"] = "GO" if all_pass else "NO-GO"

    return report


def _print_report(report: dict[str, Any]) -> None:
    """Pretty-print the go/no-go report."""
    print("\n" + "=" * 50)
    print("         GO / NO-GO SUMMARY")
    print("=" * 50)

    def _line(name: str, key: str) -> None:
        if key not in report:
            print(f"  {name:<30s} N/A (no data)")
            return
        entry = report[key]
        passed = entry["pass"]
        marker = "PASS ✓" if passed else "FAIL ✗"

        if "n_ok" in entry:
            val_str = f"{entry['n_ok']}/{entry['n_total']} ({entry['value']:.0%})"
        else:
            val_str = f"{entry['value']:.1f}"

        gate_str = f"gate: {entry['gate']}"
        print(f"  {name:<30s} {val_str:<20s} [{marker}] ({gate_str})")

    _line("Camera calibration", "camera_success")
    _line("Serve speed [10,35] m/s", "serve_speed")
    _line("Net crossing [2.24,5.0] m", "net_crossing")
    _line("Gravity residual <0.30", "gravity_residual")
    _line("Contact height [0,4] m", "contact_height")
    _line("Reproj RMSE median <5px", "reproj_rmse")

    print()
    print(f"  Total arcs: {report['total_arcs']}, valid: {report['valid_arcs']}")

    if "serve_speed" in report:
        ss = report["serve_speed"]
        print(f"  Serve speed: mean={ss['mean']:.1f} median={ss['median']:.1f} "
              f"std={ss['std']:.1f} range=[{ss['min']:.1f}, {ss['max']:.1f}] m/s")

    verdict = report["verdict"]
    print()
    print(f"  VERDICT: {verdict} {'✓' if verdict == 'GO' else '✗'}")
    print("=" * 50)


def _serialise_report(
    report: dict[str, Any],
    per_video: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Make report JSON-serialisable."""
    out = dict(report)
    out["per_video"] = {}
    for vid_id, stats in per_video.items():
        out["per_video"][vid_id] = {
            "arcs": stats["arcs"],
            "valid": stats["valid"],
            "serve_speeds": stats["serve_speeds"],
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3D ball trajectory estimation")
    parser.add_argument("--video-id", help="Evaluate a single video")
    parser.add_argument("--verbose", "-v", action="store_true", help="Per-arc detail")
    parser.add_argument("--no-joint", action="store_true",
                        help="Disable multi-arc joint fitting")
    args = parser.parse_args()

    run_evaluation(
        video_id=args.video_id,
        verbose=args.verbose,
        joint=not args.no_joint,
    )


if __name__ == "__main__":
    main()
