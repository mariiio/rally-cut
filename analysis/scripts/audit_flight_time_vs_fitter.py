"""Flight-time speed anchor — numerical falsifier for the ball 3D fitter.

For every rally with serve + receive action GT, computes an independent
estimate of average horizontal ball speed from:

    flight_time  = (receive_frame - serve_frame) / fps
    distance_xy  = ||receive_hand_court_xy - serve_hand_court_xy||
    v_avg        = distance_xy / flight_time

Then applies a drag correction to estimate release speed:

    v_release_expected = 1.25 * v_avg   # beach volleyball, 10-25 m/s flight

And compares to the fitter's ``FittedArc.speed_at_start`` on the serve arc.

Reports:
    - pct_physical_bound_pass: fraction with release_fitter >= 0.9 * v_avg
    - median/p70/p90 of |rel_err| where rel_err = (fitter - expected) / expected
    - pct_within_20pct: fraction with |rel_err| <= 0.20
    - per-rally CSV + JSON
    - histogram PNG comparing fitter vs flight-time-derived speeds

Usage
-----
    cd analysis
    uv run python scripts/audit_flight_time_vs_fitter.py
    uv run python scripts/audit_flight_time_vs_fitter.py --audit-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib.pyplot as plt
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
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition

COURT_CORNERS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (COURT_WIDTH, 0.0),
    (COURT_WIDTH, COURT_LENGTH),
    (0.0, COURT_LENGTH),
]

OUTPUT_DIR = Path("outputs/ball_3d_rig")
AUDIT_FILE = OUTPUT_DIR / "audit_rallies.json"

# Drag correction factor for a beach volleyball serve at 12-20 m/s.
# Derived from typical CD ≈ 0.4 → end_speed/start_speed ≈ 0.80 over a 12m flight.
# Acknowledges ±10% uncertainty vs serve type (float vs jump vs jump-float).
DRAG_CORRECTION = 1.25

# Physical-bound slack for frame quantisation + pose noise.
PHYS_BOUND_SLACK = 0.9  # release >= 0.9 * v_avg  →  allow ~10% underestimate

# Tier 1-C quality threshold.
REL_ERR_PASS = 0.20


# COCO keypoint indices.
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

# Minimum keypoint confidence to trust a wrist.
MIN_WRIST_CONF = 0.3


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class RallyRow:
    rally_id: str
    video_id: str
    fps: float
    ball_positions: list[BallPosition]
    player_positions: list[PlayerPosition]
    action_gt: list[dict]
    video_cal: tuple[list[tuple[float, float]], int, int]


def _parse_ball(bp_json: Any) -> list[BallPosition]:
    if not bp_json:
        return []
    positions = bp_json.get("positions", []) if isinstance(bp_json, dict) else bp_json
    return [
        BallPosition(
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            confidence=p.get("confidence", 0.5),
        )
        for p in positions
    ]


def _parse_players(pos_json: Any) -> list[PlayerPosition]:
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
            keypoints=p.get("keypoints"),
        ))
    return result


def load_rallies() -> list[RallyRow]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                r.id, r.video_id, pt.fps,
                pt.ball_positions_json, pt.positions_json, pt.action_ground_truth_json,
                v.court_calibration_json, v.width, v.height
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE pt.action_ground_truth_json IS NOT NULL
              AND pt.ball_positions_json IS NOT NULL
              AND v.court_calibration_json IS NOT NULL
        """)
        rows = cur.fetchall()
    out: list[RallyRow] = []
    for row in rows:
        cal = row[6]
        if not isinstance(cal, list) or len(cal) != 4:
            continue
        corners = [(c["x"], c["y"]) for c in cal]
        out.append(RallyRow(
            rally_id=str(row[0]),
            video_id=str(row[1]),
            fps=float(row[2] or 30.0),
            ball_positions=_parse_ball(row[3]),
            player_positions=_parse_players(row[4]),
            action_gt=row[5] or [],
            video_cal=(corners, int(row[7] or 1920), int(row[8] or 1080)),
        ))
    return out


# ---------------------------------------------------------------------------
# Hand position extraction
# ---------------------------------------------------------------------------


def player_feet_xy_at_contact(
    rally: RallyRow,
    frame: int,
    track_id: int,
) -> tuple[float, float] | None:
    """Return the normalised image (x, y) of the player's feet at a contact.

    The feet are on the ground plane (Z=0), so back-projecting through the
    court homography gives an exact court-plane xy. Using the wrist instead
    would back-project an above-ground point through a ground-plane homography,
    which produces meter-scale errors as the ray approaches horizontal (and
    can diverge entirely when the keypoint is near the image horizon).
    """
    best_dt = 999
    best_pos: PlayerPosition | None = None
    for pos in rally.player_positions:
        if pos.track_id != track_id:
            continue
        dt = abs(pos.frame_number - frame)
        if dt < best_dt:
            best_dt = dt
            best_pos = pos
    if best_pos is None or best_dt > 5:
        return None

    # Feet = bbox bottom-centre (normalised coords: center x,y + height/2 down).
    feet_x = float(best_pos.x)
    feet_y = float(best_pos.y + best_pos.height / 2.0)
    # Clamp inside frame.
    feet_y = min(max(feet_y, 0.0), 1.0)
    return feet_x, feet_y


def court_xy_from_image(
    calibrator: CourtCalibrator, image_xy: tuple[float, float]
) -> tuple[float, float] | None:
    try:
        cx, cy = calibrator.image_to_court(image_xy, 1, 1)
        return float(cx), float(cy)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Camera + fit
# ---------------------------------------------------------------------------


def build_camera(rally: RallyRow) -> tuple[CameraModel | None, CourtCalibrator]:
    calibrator = CourtCalibrator()
    calibrator.calibrate(rally.video_cal[0])
    corners, w, h = rally.video_cal

    # Estimate net-y from contact detector.
    cs = detect_contacts(
        ball_positions=rally.ball_positions,
        player_positions=rally.player_positions,
        config=ContactDetectionConfig(),
        court_calibrator=calibrator,
    )
    net_y = cs.net_y if 0.1 < cs.net_y < 0.9 else None

    cam: CameraModel | None = None
    if net_y is not None:
        cam = calibrate_camera_with_net(corners, COURT_CORNERS, w, h, net_y_image=net_y)
    if cam is None or not cam.is_valid:
        cam = calibrate_camera(corners, COURT_CORNERS, w, h)

    return cam, calibrator


def fitter_serve_speed(rally: RallyRow, camera: CameraModel) -> float | None:
    calibrator = CourtCalibrator()
    calibrator.calibrate(rally.video_cal[0])
    cs = detect_contacts(
        ball_positions=rally.ball_positions,
        player_positions=rally.player_positions,
        config=ContactDetectionConfig(),
        court_calibrator=calibrator,
    )
    traj = fit_rally(
        camera=camera,
        contact_sequence=cs,
        classified_actions=None,
        fps=rally.fps,
        rally_id=rally.rally_id,
        video_id=rally.video_id,
        net_height=2.24,
        joint=True,
    )
    if not traj.arcs:
        return None
    return float(traj.arcs[0].speed_at_start)


# ---------------------------------------------------------------------------
# Anchor computation
# ---------------------------------------------------------------------------


def compute_anchor_for_rally(rally: RallyRow) -> dict[str, Any] | None:
    serve_labels = [l for l in rally.action_gt if l.get("action") == "serve"]
    receive_labels = [l for l in rally.action_gt if l.get("action") == "receive"]
    if not serve_labels or not receive_labels:
        return None

    # Pair serve with the first receive that follows it.
    serve = serve_labels[0]
    receive = next(
        (r for r in receive_labels if r["frame"] > serve["frame"]), None
    )
    if receive is None:
        return None

    flight_frames = receive["frame"] - serve["frame"]
    flight_seconds = flight_frames / rally.fps
    if not (0.2 <= flight_seconds <= 3.0):
        return None

    # Court-plane hand positions at serve and receive.
    serve_track = serve.get("playerTrackId", -1)
    receive_track = receive.get("playerTrackId", -1)

    serve_img = player_feet_xy_at_contact(rally, serve["frame"], serve_track)
    receive_img = player_feet_xy_at_contact(rally, receive["frame"], receive_track)
    if serve_img is None or receive_img is None:
        return None

    calibrator = CourtCalibrator()
    calibrator.calibrate(rally.video_cal[0])
    serve_xy = court_xy_from_image(calibrator, serve_img)
    receive_xy = court_xy_from_image(calibrator, receive_img)
    if serve_xy is None or receive_xy is None:
        return None

    dx = receive_xy[0] - serve_xy[0]
    dy = receive_xy[1] - serve_xy[1]
    distance = math.sqrt(dx * dx + dy * dy)
    # Skip unphysical outliers (hand detection fallbacks in the wrong court half).
    if distance < 2.0 or distance > 25.0:
        return {
            "rally_id": rally.rally_id,
            "video_id": rally.video_id,
            "skipped_reason": f"distance {distance:.1f}m out of [2, 25]",
        }

    v_avg = distance / flight_seconds
    v_expected = DRAG_CORRECTION * v_avg

    # Camera + fitter call.
    cam, _ = build_camera(rally)
    if cam is None:
        return {
            "rally_id": rally.rally_id,
            "video_id": rally.video_id,
            "skipped_reason": "camera calibration failed",
        }
    v_fitter = fitter_serve_speed(rally, cam)
    if v_fitter is None:
        return {
            "rally_id": rally.rally_id,
            "video_id": rally.video_id,
            "skipped_reason": "fitter produced no arcs",
        }

    phys_bound_pass = v_fitter >= PHYS_BOUND_SLACK * v_avg
    rel_err = (v_fitter - v_expected) / v_expected
    within_20pct = abs(rel_err) <= REL_ERR_PASS

    return {
        "rally_id": rally.rally_id,
        "video_id": rally.video_id,
        "fps": rally.fps,
        "serve_frame": int(serve["frame"]),
        "receive_frame": int(receive["frame"]),
        "flight_frames": int(flight_frames),
        "flight_seconds": round(flight_seconds, 3),
        "serve_court_x": round(serve_xy[0], 3),
        "serve_court_y": round(serve_xy[1], 3),
        "receive_court_x": round(receive_xy[0], 3),
        "receive_court_y": round(receive_xy[1], 3),
        "distance_m": round(distance, 3),
        "v_avg_mps": round(v_avg, 3),
        "v_release_expected_mps": round(v_expected, 3),
        "v_release_fitter_mps": round(v_fitter, 3),
        "camera_height_m": round(float(cam.camera_position[2]), 3),
        "phys_bound_pass": phys_bound_pass,
        "rel_err": round(rel_err, 4),
        "within_20pct": within_20pct,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    kept = [r for r in results if "rel_err" in r]
    skipped = [r for r in results if "skipped_reason" in r]

    rel_errs = np.array([r["rel_err"] for r in kept])
    v_avg = np.array([r["v_avg_mps"] for r in kept])
    v_expected = np.array([r["v_release_expected_mps"] for r in kept])
    v_fitter = np.array([r["v_release_fitter_mps"] for r in kept])
    phys_pass = sum(1 for r in kept if r["phys_bound_pass"])
    within_20 = sum(1 for r in kept if r["within_20pct"])

    return {
        "n_rallies_total": len(results),
        "n_rallies_kept": len(kept),
        "n_rallies_skipped": len(skipped),
        "skipped_reasons": {
            reason: sum(1 for r in skipped if r.get("skipped_reason") == reason)
            for reason in {r["skipped_reason"] for r in skipped}
        },
        "v_avg_mps": {
            "mean": float(np.mean(v_avg)),
            "median": float(np.median(v_avg)),
            "min": float(np.min(v_avg)),
            "max": float(np.max(v_avg)),
        },
        "v_release_expected_mps": {
            "mean": float(np.mean(v_expected)),
            "median": float(np.median(v_expected)),
        },
        "v_release_fitter_mps": {
            "mean": float(np.mean(v_fitter)),
            "median": float(np.median(v_fitter)),
            "std": float(np.std(v_fitter)),
        },
        "phys_bound_pass_rate": phys_pass / len(kept) if kept else 0.0,
        "phys_bound_pass_n": f"{phys_pass}/{len(kept)}",
        "rel_err_median_abs": float(np.median(np.abs(rel_errs))),
        "rel_err_p70_abs": float(np.percentile(np.abs(rel_errs), 70)),
        "rel_err_p90_abs": float(np.percentile(np.abs(rel_errs), 90)),
        "within_20pct_rate": within_20 / len(kept) if kept else 0.0,
        "within_20pct_n": f"{within_20}/{len(kept)}",
    }


def write_histogram(results: list[dict[str, Any]], out_path: Path) -> None:
    kept = [r for r in results if "rel_err" in r]
    if not kept:
        return
    v_expected = np.array([r["v_release_expected_mps"] for r in kept])
    v_fitter = np.array([r["v_release_fitter_mps"] for r in kept])
    rel_errs = np.array([r["rel_err"] for r in kept])
    heights = np.array([r["camera_height_m"] for r in kept])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: overlayed histograms of expected vs fitter speed.
    axes[0, 0].hist(v_expected, bins=30, range=(0, 40), alpha=0.6, color="#0077ff", label="expected (1.25 × flight_avg)")
    axes[0, 0].hist(v_fitter, bins=30, range=(0, 40), alpha=0.6, color="#ff4455", label="fitter release")
    axes[0, 0].axvline(12, linestyle="--", color="gray", linewidth=0.8, label="amateur serve min (~12 m/s)")
    axes[0, 0].axvline(25, linestyle="--", color="gray", linewidth=0.8)
    axes[0, 0].set_xlabel("serve release speed (m/s)")
    axes[0, 0].set_ylabel("rallies")
    axes[0, 0].set_title("Fitter vs flight-time-derived release speed")
    axes[0, 0].legend(fontsize=8)

    # Panel 2: scatter fitter vs expected.
    axes[0, 1].scatter(v_expected, v_fitter, c=heights, cmap="viridis", s=30, alpha=0.8)
    axes[0, 1].plot([0, 40], [0, 40], "k--", linewidth=0.8, label="y = x (ideal)")
    axes[0, 1].plot([0, 40], [0, 32], color="orange", linewidth=0.8, alpha=0.7, label="y = 0.8x (-20%)")
    axes[0, 1].plot([0, 40], [0, 48], color="orange", linewidth=0.8, alpha=0.7, label="y = 1.2x (+20%)")
    axes[0, 1].set_xlabel("expected release m/s (1.25 × v_avg)")
    axes[0, 1].set_ylabel("fitter release m/s")
    axes[0, 1].set_title("Fitter vs expected (coloured by camera height)")
    axes[0, 1].set_xlim(0, 40)
    axes[0, 1].set_ylim(0, 45)
    axes[0, 1].legend(fontsize=8, loc="upper left")
    plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label="camera height (m)")

    # Panel 3: rel_err histogram.
    axes[1, 0].hist(rel_errs, bins=40, range=(-1, 1), color="#9a3f94", alpha=0.8)
    axes[1, 0].axvline(-0.2, linestyle="--", color="green", linewidth=1.0, label="±20% band")
    axes[1, 0].axvline(0.2, linestyle="--", color="green", linewidth=1.0)
    axes[1, 0].axvline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("relative error (fitter − expected) / expected")
    axes[1, 0].set_ylabel("rallies")
    axes[1, 0].set_title("Relative error distribution")
    axes[1, 0].legend(fontsize=8)

    # Panel 4: rel_err by camera height tier.
    tiers = []
    for r in kept:
        h = r["camera_height_m"]
        tiers.append("low" if h < 1.3 else "mid" if h < 1.7 else "high")
    colours_map = {"low": "#d62728", "mid": "#2ca02c", "high": "#1f77b4"}
    for tier in ("low", "mid", "high"):
        errs = [r["rel_err"] for r, t in zip(kept, tiers) if t == tier]
        if errs:
            axes[1, 1].hist(errs, bins=25, range=(-1, 1), alpha=0.6,
                            color=colours_map[tier], label=f"{tier} (n={len(errs)})")
    axes[1, 1].axvline(-0.2, linestyle="--", color="green", linewidth=1.0)
    axes[1, 1].axvline(0.2, linestyle="--", color="green", linewidth=1.0)
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("relative error")
    axes[1, 1].set_title("By camera-height tier")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Restrict to the Phase C audit set (20 rallies). Default: all serve+receive GT rallies."
    )
    args = parser.parse_args()

    all_rallies = load_rallies()
    print(f"Loaded {len(all_rallies)} rallies with serve+receive action GT + calibration")

    if args.audit_only:
        audit_info = json.loads(AUDIT_FILE.read_text())
        ids = {r["rally_id"] for r in audit_info["audit_rallies"]}
        all_rallies = [r for r in all_rallies if r.rally_id in ids]
        print(f"  Restricted to {len(all_rallies)} audit rallies")

    results: list[dict[str, Any]] = []
    for i, rally in enumerate(all_rallies, 1):
        r = compute_anchor_for_rally(rally)
        if r is None:
            continue
        results.append(r)
        if "rel_err" in r:
            status = "pass" if r["phys_bound_pass"] else "FAIL"
            print(
                f"  [{i}/{len(all_rallies)}] {rally.rally_id[:10]} "
                f"flight={r['flight_seconds']}s "
                f"dist={r['distance_m']:.1f}m "
                f"v_avg={r['v_avg_mps']:.1f} "
                f"v_exp={r['v_release_expected_mps']:.1f} "
                f"v_fit={r['v_release_fitter_mps']:.1f} "
                f"err={r['rel_err']:+.2f} "
                f"[{status}]"
            )

    summary = summarize(results)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # Decision lines.
    print("\n=== DECISION GATES ===")
    print(
        f"  Physical bound (release ≥ 0.9 × v_avg) : "
        f"{summary['phys_bound_pass_rate']:.0%} ({summary['phys_bound_pass_n']})  "
        f"gate 95%  → {'PASS' if summary['phys_bound_pass_rate'] >= 0.95 else 'FAIL'}"
    )
    print(
        f"  Tier 1-C speed quality (|rel_err| ≤ 20%): "
        f"{summary['within_20pct_rate']:.0%} ({summary['within_20pct_n']})  "
        f"gate 70%  → {'PASS' if summary['within_20pct_rate'] >= 0.70 else 'FAIL'}"
    )
    print(
        f"  Median |rel_err|: {summary['rel_err_median_abs']:.1%}  "
        f"gate ≤20%  → {'PASS' if summary['rel_err_median_abs'] <= 0.20 else 'FAIL'}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "flight_time_anchor.json"
    json_path.write_text(json.dumps({
        "summary": summary,
        "per_rally": results,
    }, indent=2, default=float))
    print(f"\nWrote {json_path}")

    csv_path = OUTPUT_DIR / "flight_time_anchor.csv"
    kept = [r for r in results if "rel_err" in r]
    if kept:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
            writer.writeheader()
            writer.writerows(kept)
        print(f"Wrote {csv_path}")

    hist_path = OUTPUT_DIR / "flight_time_anchor.png"
    write_histogram(results, hist_path)
    print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
