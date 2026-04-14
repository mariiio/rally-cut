"""Phase B probe P4: ray / ground-plane intersection feasibility.

For every WASB ball detection across the audit set, cast the camera ray
through the detection's image point and find where it intersects the
ground plane (z=0). Classify each detection by:

    valid_exists   — ray intersects z=0 at all (not parallel / not behind)
    valid_infront  — intersection is in front of the camera (ray parameter > 0)
    valid_inbounds — intersection is within a generous court box
                     (|x| < COURT_WIDTH + 5, y ∈ [-5, COURT_LENGTH + 10])

The plan-level gate is: ≥80% of ball detections produce usable ground-ray
intersections. If this fails, any "ray discretization from field up to
camera" framework (Yandex 2025 soccer pattern) needs a different state
parameterization before we can plumb in other measurement channels.

Why this matters
----------------
The plan's Tier 2 research-framework candidate is a multi-hypothesis
Bayesian filter that discretizes candidate ball positions along each ray.
For that to be tractable, most rays must have a well-defined, bounded
sampling interval — either between the ground plane and the camera, or
between two physical anchors. If rays are near-parallel to the ground
(1-2m cameras), a large fraction may fail this precondition, and the
framework has to be redesigned from scratch.

Usage
-----
    cd analysis
    uv run python scripts/probe_ray_ground_intersection.py
"""

from __future__ import annotations

import json
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
    image_ray,
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

# In-bounds generosity (metres from court edges).
BOUNDS_X_MARGIN = 5.0
BOUNDS_Y_MARGIN_NEAR = 5.0
BOUNDS_Y_MARGIN_FAR = 10.0

OUTPUT_DIR = Path("outputs/ball_3d_rig")
AUDIT_FILE = OUTPUT_DIR / "audit_rallies.json"

# Gate.
P4_PASS_RATE = 0.80


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class RallyBundle:
    rally_id: str
    video_id: str
    tier: str
    camera_height_m_meta: float
    ball_positions: list[BallPosition]
    player_positions: list[PlayerPosition]
    video_cal: tuple[list[tuple[float, float]], int, int]


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


def load_bundles(audit_info: dict) -> list[RallyBundle]:
    rally_ids = [r["rally_id"] for r in audit_info["audit_rallies"]]
    meta_by_id = {r["rally_id"]: r for r in audit_info["audit_rallies"]}

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.video_id, pt.ball_positions_json, pt.positions_json,
                   v.court_calibration_json, v.width, v.height
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE r.id = ANY(%s)
        """, (rally_ids,))
        rows = cur.fetchall()

    bundles: list[RallyBundle] = []
    for row in rows:
        rid = str(row[0])
        meta = meta_by_id[rid]
        cal = row[4]
        if not isinstance(cal, list) or len(cal) != 4:
            continue
        corners = [(c["x"], c["y"]) for c in cal]
        bundles.append(RallyBundle(
            rally_id=rid,
            video_id=str(row[1]),
            tier=meta["tier"],
            camera_height_m_meta=meta.get("camera_height_m", 0.0),
            ball_positions=_parse_ball(row[2]),
            player_positions=_parse_players(row[3]),
            video_cal=(corners, int(row[5] or 1920), int(row[6] or 1080)),
        ))
    return bundles


def build_camera(bundle: RallyBundle) -> CameraModel | None:
    calibrator = CourtCalibrator()
    calibrator.calibrate(bundle.video_cal[0])
    cs = detect_contacts(
        ball_positions=bundle.ball_positions,
        player_positions=bundle.player_positions,
        config=ContactDetectionConfig(),
        court_calibrator=calibrator,
    )
    net_y = cs.net_y if 0.1 < cs.net_y < 0.9 else None
    corners, w, h = bundle.video_cal
    cam = None
    if net_y is not None:
        cam = calibrate_camera_with_net(corners, COURT_CORNERS, w, h, net_y_image=net_y)
    if cam is None or not cam.is_valid:
        cam = calibrate_camera(corners, COURT_CORNERS, w, h)
    return cam


# ---------------------------------------------------------------------------
# Geometry check
# ---------------------------------------------------------------------------


def classify_detection(
    camera: CameraModel, ball: BallPosition,
) -> tuple[str, tuple[float, float, float] | None]:
    """Return ('exists'|'infront'|'inbounds'|'parallel'|'behind'|'outbounds', intersection).

    Status progresses: parallel/behind/outbounds/infront/inbounds. 'inbounds'
    is the fully-usable case.
    """
    origin, direction = image_ray(camera, (ball.x, ball.y))
    if abs(direction[2]) < 1e-8:
        return "parallel", None
    t = (0.0 - origin[2]) / direction[2]
    if t <= 0:
        return "behind", None
    hit = origin + t * direction
    hx, hy, hz = float(hit[0]), float(hit[1]), float(hit[2])
    in_x = -BOUNDS_X_MARGIN <= hx <= COURT_WIDTH + BOUNDS_X_MARGIN
    in_y = -BOUNDS_Y_MARGIN_NEAR <= hy <= COURT_LENGTH + BOUNDS_Y_MARGIN_FAR
    if in_x and in_y:
        return "inbounds", (hx, hy, hz)
    return "outbounds", (hx, hy, hz)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not AUDIT_FILE.exists():
        print(f"ERROR: audit file not found at {AUDIT_FILE}")
        sys.exit(1)
    audit_info = json.loads(AUDIT_FILE.read_text())
    bundles = load_bundles(audit_info)
    print(f"Loaded {len(bundles)} audit rallies")

    # Per-rally + tier stats.
    status_counts: dict[str, int] = {
        "parallel": 0, "behind": 0, "outbounds": 0, "inbounds": 0,
    }
    tier_stats: dict[str, dict[str, int]] = {
        "low": {"inbounds": 0, "total": 0},
        "mid": {"inbounds": 0, "total": 0},
        "high": {"inbounds": 0, "total": 0},
    }
    per_rally: list[dict[str, Any]] = []
    # Also track where the out-of-bounds hits end up (for histogram).
    bounds_violations_x: list[float] = []
    bounds_violations_y: list[float] = []

    for bundle in bundles:
        cam = build_camera(bundle)
        if cam is None:
            print(f"  [skip] {bundle.rally_id[:10]} — camera calibration failed")
            continue

        n_total = 0
        n_inbounds = 0
        local_status: dict[str, int] = {
            "parallel": 0, "behind": 0, "outbounds": 0, "inbounds": 0,
        }
        for bp in bundle.ball_positions:
            status, hit = classify_detection(cam, bp)
            n_total += 1
            local_status[status] += 1
            status_counts[status] += 1
            if status == "inbounds":
                n_inbounds += 1
            if status == "outbounds" and hit is not None:
                bounds_violations_x.append(hit[0])
                bounds_violations_y.append(hit[1])

        tier_stats[bundle.tier]["total"] += n_total
        tier_stats[bundle.tier]["inbounds"] += n_inbounds

        rate = n_inbounds / n_total if n_total else 0.0
        per_rally.append({
            "rally_id": bundle.rally_id,
            "tier": bundle.tier,
            "camera_height_m": float(cam.camera_position[2]),
            "n_total": n_total,
            "n_inbounds": n_inbounds,
            "inbounds_rate": rate,
            "status": local_status,
        })
        print(
            f"  [{bundle.tier:4s}] {bundle.rally_id[:10]} "
            f"cam={cam.camera_position[2]:.2f}m  "
            f"inbounds={n_inbounds}/{n_total} ({rate * 100:.1f}%)  "
            f"parallel={local_status['parallel']} behind={local_status['behind']} "
            f"outbounds={local_status['outbounds']}"
        )

    total = sum(status_counts.values())
    inbounds_rate = status_counts["inbounds"] / total if total else 0.0

    print()
    print("=" * 60)
    print(f"TOTAL detections: {total}")
    print(f"  parallel   : {status_counts['parallel']:>5}  "
          f"({status_counts['parallel'] / total * 100:5.1f}%)")
    print(f"  behind cam : {status_counts['behind']:>5}  "
          f"({status_counts['behind'] / total * 100:5.1f}%)")
    print(f"  out of box : {status_counts['outbounds']:>5}  "
          f"({status_counts['outbounds'] / total * 100:5.1f}%)")
    print(f"  in box     : {status_counts['inbounds']:>5}  "
          f"({status_counts['inbounds'] / total * 100:5.1f}%)")
    print()
    print(f"Gate P4: inbounds ≥{P4_PASS_RATE:.0%}  "
          f"→ {inbounds_rate:.1%}  {'PASS' if inbounds_rate >= P4_PASS_RATE else 'FAIL'}")
    print()
    print("Per tier:")
    for tier in ("low", "mid", "high"):
        ts = tier_stats[tier]
        if ts["total"]:
            rate = ts["inbounds"] / ts["total"]
            print(f"  {tier:4s}: {ts['inbounds']}/{ts['total']} ({rate * 100:.1f}%)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot histogram of out-of-bounds hit locations (diagnostic).
    if bounds_violations_x:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(bounds_violations_y, bins=50, color="#d62728", alpha=0.8)
        axes[0].axvspan(-BOUNDS_Y_MARGIN_NEAR, 0, alpha=0.15, color="green", label="near margin")
        axes[0].axvspan(COURT_LENGTH, COURT_LENGTH + BOUNDS_Y_MARGIN_FAR, alpha=0.15, color="green", label="far margin")
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].axvline(COURT_LENGTH, color="black", linewidth=0.8)
        axes[0].set_xlabel("court Y at ray-ground intersection (m)")
        axes[0].set_ylabel("detections")
        axes[0].set_title("Out-of-bounds hits by Y")
        axes[0].legend(fontsize=8)

        axes[1].hist(bounds_violations_x, bins=50, color="#1f77b4", alpha=0.8)
        axes[1].axvspan(-BOUNDS_X_MARGIN, 0, alpha=0.15, color="green")
        axes[1].axvspan(COURT_WIDTH, COURT_WIDTH + BOUNDS_X_MARGIN, alpha=0.15, color="green")
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].axvline(COURT_WIDTH, color="black", linewidth=0.8)
        axes[1].set_xlabel("court X at ray-ground intersection (m)")
        axes[1].set_title("Out-of-bounds hits by X")

        plt.tight_layout()
        hist_path = OUTPUT_DIR / "probe_ray_ground_outbounds.png"
        plt.savefig(hist_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {hist_path}")

    out = OUTPUT_DIR / "probe_ray_ground.json"
    out.write_text(json.dumps({
        "gate_pass_rate": P4_PASS_RATE,
        "inbounds_rate": inbounds_rate,
        "pass": inbounds_rate >= P4_PASS_RATE,
        "status_counts": status_counts,
        "tier_stats": tier_stats,
        "per_rally": per_rally,
        "n_out_of_bounds_sample": len(bounds_violations_x),
    }, indent=2, default=float))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
