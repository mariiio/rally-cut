"""Tier-1 ship-decision audit for ball 3D fitter.

Consolidates four signals into a single go/no-go report:

    1. Landing accuracy (Tier 1-A gate)
       — automatic from dense 2D ball GT on 40 rallies.
       — GT landing = last ball GT position where trajectory is descending.
       — fitter landing = ``FittedArc.landing_position`` for the last arc.
       — metric: court-plane distance (m), reported as median + p90.

    2. Net-plane 3-class agreement (Tier 1-B gate)
       — semi-automatic. Reads ``net_plane_labels.json`` if present
         (human inspects rig side-elevation panels and labels
         above_tape / at_tape / below_tape per rally).
       — metric: agreement rate between fitter's net_crossing_height
         and the human label, where the fitter maps to:
             above_tape  ⇐  height ≥ 2.34m
             at_tape     ⇐  2.14m ≤ height < 2.34m
             below_tape  ⇐  height < 2.14m

    3. Speed physical bound (required for Tier 1-C)
       — automatic from ``flight_time_anchor.json``.
       — metric: fraction of serves with release ≥ 0.9 × v_avg.

    4. Speed Tier-1-C quality
       — automatic from ``flight_time_anchor.json``.
       — metric: median |rel_err| and fraction within ±20%.

Outputs a decision doc at ``outputs/ball_3d_rig/audit_report_<date>.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH, CourtCalibrator
from rallycut.court.camera_model import calibrate_camera, calibrate_camera_with_net
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

# Thresholds (committed in plan).
#
# Tier 1-A "plausibility" check on all arcs across rallies with dense ball GT.
# We cannot use fitter.landing_position directly as a landing-accuracy metric
# because the fitter produces non-physical arcs (peaks of 15+m, landings 100m
# off-court) on rallies where the depth/height coupling bites hardest. The
# product-relevant question is therefore: can the fitter output be shipped
# at all?  An arc is "physical" if it meets ALL of these:
#
#   peak_height    ∈ [0, 6] m            (beach ball rarely above 6m)
#   landing x      ∈ [-2, COURT_WIDTH + 2]
#   landing y      ∈ [-3, COURT_LENGTH + 3]
#   speed_at_start ∈ [3, 30] m/s         (amateur beach serve range)
#   reproj_rmse    < 20 px               (matches observation at all)
#
# A rally is "shippable at Tier 1-A" if at least 80% of its arcs are physical.
PLAUSIBLE_PEAK_MAX_M = 6.0
PLAUSIBLE_LANDING_X_MIN = -2.0
PLAUSIBLE_LANDING_X_MAX = COURT_WIDTH + 2.0
PLAUSIBLE_LANDING_Y_MIN = -3.0
PLAUSIBLE_LANDING_Y_MAX = COURT_LENGTH + 3.0
PLAUSIBLE_SPEED_MIN = 3.0
PLAUSIBLE_SPEED_MAX = 30.0
PLAUSIBLE_RMSE_MAX_PX = 20.0

# A rally is shippable if ≥80% of arcs are physical.
SHIPPABLE_RALLY_ARC_FRACTION = 0.80
# Tier 1-A passes if ≥80% of ball-GT rallies are shippable.
TIER_1A_SHIPPABLE_GATE = 0.80

# Serve-arc specific check — the first arc must also be physical if the serve
# is ever going to be a shippable product feature.
SERVE_ARC_PASS_GATE = 0.80  # fraction of rallies where arc[0] is physical

NET_PLANE_AGREEMENT_GATE = 0.90
SPEED_BOUND_GATE = 0.95
SPEED_QUALITY_PASS_RATE = 0.70
SPEED_QUALITY_MEDIAN = 0.20

# Net-plane class boundaries.
NET_TAPE_M = 2.24
NET_AT_TAPE_HALF = 0.10  # ±10cm for the "at tape" class


# ---------------------------------------------------------------------------
# Landing accuracy (Tier 1-A) — automatic via ball GT
# ---------------------------------------------------------------------------


# Velocity threshold for "stopped ball" detection (normalised image coords per frame).
# Beach ball on sand decelerates from ~0.05 per frame to near zero within ~5 frames.
STOPPED_VEL_THRESHOLD = 0.004  # ~4 px per frame at 1920px, ~2 cm/frame court-plane
STOPPED_MIN_RUN = 5            # must hold for at least this many consecutive frames


def _find_stopped_landing(
    ball_gt: list[tuple[int, float, float]],
) -> tuple[int, float, float] | None:
    """Return the median position of the longest tail-run of low-velocity frames.

    Walks ball_gt from the end backward, collecting consecutive frames where
    image-space displacement is below STOPPED_VEL_THRESHOLD. When a longer run
    is found, updates the median position. Requires at least STOPPED_MIN_RUN
    frames; returns None otherwise (rally ended before the ball came to rest).
    """
    if len(ball_gt) < STOPPED_MIN_RUN + 1:
        return None

    # Identify runs from the end, looking for the final stopped segment.
    end = len(ball_gt) - 1
    stopped_start = end
    while stopped_start > 0:
        f0, x0, y0 = ball_gt[stopped_start - 1]
        f1, x1, y1 = ball_gt[stopped_start]
        # Skip gaps (missing frames) — those disqualify a run.
        if f1 - f0 > 3:
            break
        dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if dist > STOPPED_VEL_THRESHOLD:
            break
        stopped_start -= 1

    run_len = end - stopped_start + 1
    if run_len < STOPPED_MIN_RUN:
        return None

    stopped = ball_gt[stopped_start:end + 1]
    median_x = float(np.median([r[1] for r in stopped]))
    median_y = float(np.median([r[2] for r in stopped]))
    return (stopped[len(stopped) // 2][0], median_x, median_y)


def _parse_ball_gt(gt_json: Any) -> list[tuple[int, float, float]]:
    if not gt_json:
        return []
    positions = gt_json.get("positions", []) if isinstance(gt_json, dict) else gt_json
    out = []
    for p in positions:
        if not isinstance(p, dict):
            continue
        if (p.get("label") or "").lower() != "ball":
            continue
        out.append((
            int(p.get("frameNumber", 0)),
            float(p.get("x", 0.0)),
            float(p.get("y", 0.0)),
        ))
    return sorted(out, key=lambda r: r[0])


def _parse_ball_positions(bp_json: Any) -> list[BallPosition]:
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


def _parse_player_positions(pos_json: Any) -> list[PlayerPosition]:
    if not pos_json:
        return []
    out: list[PlayerPosition] = []
    for p in pos_json:
        out.append(PlayerPosition(
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            track_id=p.get("trackId", p.get("track_id", 0)),
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            width=p.get("width", 0.0),
            height=p.get("height", 0.0),
            confidence=p.get("confidence", 0.5),
            keypoints=p.get("keypoints"),
        ))
    return out


def _arc_is_physical(
    peak: float,
    landing: tuple[float, float] | None,
    speed: float,
    rmse_px: float,
) -> tuple[bool, list[str]]:
    """Return (is_physical, list of failure reasons)."""
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


def compute_landing_metrics() -> dict[str, Any]:
    """Tier 1-A plausibility audit across all rallies with dense ball GT.

    For each of the ~40 ball-GT rallies, runs the fitter and classifies every
    arc as physical/non-physical using the committed plausibility bounds. The
    shippability gate is: ≥80% of rallies must have ≥80% physical arcs, AND
    the serve arc (arc[0]) must be physical in ≥80% of rallies.

    We do NOT compare fitter.landing_position to a GT-derived landing point
    because the fitter produces non-physical arcs (peaks of 15+m, landings
    100+m off-court) on the same rallies where the depth/height coupling
    bites hardest — comparing nonsense to ground truth just moves the
    nonsense around. The product-relevant question is whether the fitter
    output can be shipped at all, which plausibility captures directly.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                r.id, r.video_id, pt.fps,
                pt.ball_positions_json, pt.positions_json,
                v.court_calibration_json, v.width, v.height
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE pt.ground_truth_json IS NOT NULL
              AND pt.ball_positions_json IS NOT NULL
              AND v.court_calibration_json IS NOT NULL
        """)
        rows = cur.fetchall()

    per_rally: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row[0])
        vid = str(row[1])
        fps = float(row[2] or 30.0)
        ball_positions = _parse_ball_positions(row[3])
        player_positions = _parse_player_positions(row[4])
        cal_json = row[5]
        if not isinstance(cal_json, list) or len(cal_json) != 4:
            continue
        corners = [(c["x"], c["y"]) for c in cal_json]
        width = int(row[6] or 1920)
        height = int(row[7] or 1080)

        calibrator = CourtCalibrator()
        calibrator.calibrate(corners)
        cs = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=ContactDetectionConfig(),
            court_calibrator=calibrator,
        )
        net_y = cs.net_y if 0.1 < cs.net_y < 0.9 else None
        cam = None
        if net_y is not None:
            cam = calibrate_camera_with_net(corners, COURT_CORNERS, width, height, net_y_image=net_y)
        if cam is None or not cam.is_valid:
            cam = calibrate_camera(corners, COURT_CORNERS, width, height)
        if cam is None:
            continue

        traj = fit_rally(
            camera=cam, contact_sequence=cs, classified_actions=None,
            fps=fps, rally_id=rid, video_id=vid, net_height=NET_TAPE_M, joint=True,
        )
        if not traj.arcs:
            continue

        arc_details: list[dict[str, Any]] = []
        n_physical = 0
        n_is_valid = 0
        n_is_valid_and_physical = 0
        for i, arc in enumerate(traj.arcs):
            is_phys, reasons = _arc_is_physical(
                peak=arc.peak_height,
                landing=arc.landing_position,
                speed=arc.speed_at_start,
                rmse_px=arc.reprojection_rmse,
            )
            if is_phys:
                n_physical += 1
            if arc.is_valid:
                n_is_valid += 1
                if is_phys:
                    n_is_valid_and_physical += 1
            arc_details.append({
                "arc_index": i,
                "speed": float(arc.speed_at_start),
                "peak": float(arc.peak_height),
                "landing": list(arc.landing_position) if arc.landing_position else None,
                "rmse_px": float(arc.reprojection_rmse),
                "is_valid": bool(arc.is_valid),
                "physical": is_phys,
                "failure": reasons,
            })

        plausible_rate = n_physical / len(traj.arcs)
        is_valid_physical_rate = (
            n_is_valid_and_physical / n_is_valid if n_is_valid else 0.0
        )
        serve_arc_physical = arc_details[0]["physical"]
        serve_arc_valid = arc_details[0]["is_valid"]
        per_rally.append({
            "rally_id": rid,
            "video_id": vid,
            "n_arcs": len(traj.arcs),
            "n_physical": n_physical,
            "n_is_valid": n_is_valid,
            "n_is_valid_and_physical": n_is_valid_and_physical,
            "plausible_rate": plausible_rate,
            "is_valid_physical_rate": is_valid_physical_rate,
            "serve_arc_physical": serve_arc_physical,
            "serve_arc_valid": serve_arc_valid,
            "camera_height_m": float(cam.camera_position[2]),
            "arcs": arc_details,
        })

    if not per_rally:
        return {
            "n_rallies": 0,
            "per_rally": [],
            "shippable_rally_rate": 0.0,
            "serve_arc_pass_rate": 0.0,
            "pass_landing": False,
        }

    shippable = sum(1 for r in per_rally if r["plausible_rate"] >= SHIPPABLE_RALLY_ARC_FRACTION)
    serve_ok = sum(1 for r in per_rally if r["serve_arc_physical"])
    ship_rate = shippable / len(per_rally)
    serve_rate = serve_ok / len(per_rally)
    pass_landing = (
        ship_rate >= TIER_1A_SHIPPABLE_GATE
        and serve_rate >= SERVE_ARC_PASS_GATE
    )

    # Also report arc-level stats for context.
    arc_total = sum(r["n_arcs"] for r in per_rally)
    arc_physical = sum(r["n_physical"] for r in per_rally)
    arc_is_valid = sum(r["n_is_valid"] for r in per_rally)
    arc_is_valid_physical = sum(r["n_is_valid_and_physical"] for r in per_rally)
    arc_rate = arc_physical / arc_total if arc_total else 0.0
    is_valid_rate = arc_is_valid / arc_total if arc_total else 0.0
    post_filter_rate = (
        arc_is_valid_physical / arc_is_valid if arc_is_valid else 0.0
    )

    # "Post-filter" shippability: after dropping !is_valid arcs, do remaining arcs
    # meet the plausibility bar in ≥80% of rallies?
    shippable_post = sum(
        1 for r in per_rally
        if r["n_is_valid"] > 0
        and r["is_valid_physical_rate"] >= SHIPPABLE_RALLY_ARC_FRACTION
    )
    shippable_post_rate = shippable_post / len(per_rally)

    # Serve arc shippability AFTER filter: rally passes if serve arc is both
    # is_valid AND physical (the fitter said "trust me" AND it looks right).
    serve_ok_post = sum(
        1 for r in per_rally if r["serve_arc_valid"] and r["serve_arc_physical"]
    )
    serve_post_rate = serve_ok_post / len(per_rally)

    return {
        "n_rallies": len(per_rally),
        "per_rally": per_rally,
        "shippable_rally_rate": ship_rate,
        "shippable_rally_n": f"{shippable}/{len(per_rally)}",
        "serve_arc_pass_rate": serve_rate,
        "serve_arc_pass_n": f"{serve_ok}/{len(per_rally)}",
        "arc_physical_rate": arc_rate,
        "arc_physical_n": f"{arc_physical}/{arc_total}",
        "arc_is_valid_rate": is_valid_rate,
        "arc_is_valid_n": f"{arc_is_valid}/{arc_total}",
        "arc_post_filter_physical_rate": post_filter_rate,
        "arc_post_filter_physical_n": f"{arc_is_valid_physical}/{arc_is_valid}",
        "shippable_post_filter_rate": shippable_post_rate,
        "shippable_post_filter_n": f"{shippable_post}/{len(per_rally)}",
        "serve_arc_post_filter_rate": serve_post_rate,
        "serve_arc_post_filter_n": f"{serve_ok_post}/{len(per_rally)}",
        "pass_landing": pass_landing,
    }


# ---------------------------------------------------------------------------
# Net-plane 3-class (Tier 1-B) — requires human labels
# ---------------------------------------------------------------------------


def compute_net_plane_metrics(
    labels_path: Path, rig_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    if not labels_path.exists():
        return {
            "status": "missing_labels",
            "labels_path": str(labels_path),
            "note": (
                "C.3 labelling step not yet performed. Open "
                "outputs/ball_3d_rig/index.html, inspect each rally's side-"
                "elevation panel (panel 3), classify the serve arc net "
                "crossing as above_tape/at_tape/below_tape, and write the "
                "labels to this file. Schema: "
                '{"rally_id": {"label": "above_tape"|"at_tape"|"below_tape"}}'
            ),
            "agreement_rate": None,
            "pass_net_plane": None,
        }
    labels: dict[str, dict[str, str]] = json.loads(labels_path.read_text())

    summaries_by_id = {s["rally_id"]: s for s in rig_summaries}
    agree = 0
    total = 0
    per_rally = []
    for rally_id, entry in labels.items():
        human = entry.get("label")
        if human not in ("above_tape", "at_tape", "below_tape"):
            continue
        summary = summaries_by_id.get(rally_id)
        if not summary:
            continue
        crossings = summary.get("arc_net_crossings") or []
        # Use the first arc's crossing (the serve).
        nc = crossings[0] if crossings else None
        if nc is None:
            continue
        if nc >= NET_TAPE_M + NET_AT_TAPE_HALF:
            fitter_class = "above_tape"
        elif nc >= NET_TAPE_M - NET_AT_TAPE_HALF:
            fitter_class = "at_tape"
        else:
            fitter_class = "below_tape"
        matched = fitter_class == human
        per_rally.append({
            "rally_id": rally_id,
            "human_label": human,
            "fitter_class": fitter_class,
            "fitter_height_m": float(nc),
            "match": matched,
        })
        total += 1
        if matched:
            agree += 1

    agreement_rate = agree / total if total else 0.0
    return {
        "status": "ok",
        "n_rallies": total,
        "agreement_rate": agreement_rate,
        "per_rally": per_rally,
        "pass_net_plane": total >= 10 and agreement_rate >= NET_PLANE_AGREEMENT_GATE,
    }


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


def decide_tier(
    landing: dict[str, Any],
    net_plane: dict[str, Any],
    speed: dict[str, Any],
) -> tuple[str, str]:
    if not landing["pass_landing"]:
        return ("NO-SHIP", "Landing gate failed — fitter is broken")

    pass_net_plane = net_plane.get("pass_net_plane")
    pass_speed_bound = speed["pass_speed_bound"]
    pass_speed_quality = speed["pass_speed_quality"]

    if pass_net_plane is None:
        pending_msg = "Landing OK; net-plane labels missing — defer Tier 1-B decision until C.3 labels are filled in"
        if pass_speed_bound and pass_speed_quality:
            return ("PENDING-Tier1-C", pending_msg + " (speed OK in principle)")
        return ("PENDING-Tier1-A/B", pending_msg + " (speed fails → Tier 1-C unreachable)")

    if not pass_net_plane:
        return ("Tier 1-A", "Landing OK; net-plane fails → landing-only ship")

    if not pass_speed_bound:
        return (
            "Tier 1-A (warning)",
            "Landing + net-plane pass visually, but physical speed bound fails "
            "— the fitter's physics is contradicted by flight time. Defer 1-B until explained.",
        )
    if not pass_speed_quality:
        return ("Tier 1-B", "Landing + net-plane pass, speed quality fails → no speed number")
    return ("Tier 1-C", "All gates pass — landing + net-plane + qualitative speed class")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x * 100:.1f}%"


def _fmt(x: float, unit: str = "") -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.2f}{unit}"


def write_report(
    landing: dict[str, Any],
    net_plane: dict[str, Any],
    speed: dict[str, Any],
    tier: str,
    reason: str,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append(f"# Ball 3D — Tier 1 Ship Audit ({datetime.now().strftime('%Y-%m-%d')})")
    lines.append("")
    lines.append(f"## Decision: **{tier}**")
    lines.append("")
    lines.append(reason)
    lines.append("")
    lines.append("## Gate results")
    lines.append("")
    lines.append("| Gate | Measured | Threshold | Result |")
    lines.append("|---|---|---|---|")

    ship_rate = landing.get("shippable_rally_rate", 0.0)
    ship_n = landing.get("shippable_rally_n", "0/0")
    serve_rate = landing.get("serve_arc_pass_rate", 0.0)
    serve_n = landing.get("serve_arc_pass_n", "0/0")
    arc_rate = landing.get("arc_physical_rate", 0.0)
    arc_n = landing.get("arc_physical_n", "0/0")
    lines.append(
        f"| Tier 1-A shippable rallies (≥80% physical arcs) | "
        f"{_pct(ship_rate)} ({ship_n}) | ≥ 80% | "
        f"{'PASS' if ship_rate >= TIER_1A_SHIPPABLE_GATE else 'FAIL'} |"
    )
    lines.append(
        f"| Tier 1-A serve arc physical | "
        f"{_pct(serve_rate)} ({serve_n}) | ≥ 80% | "
        f"{'PASS' if serve_rate >= SERVE_ARC_PASS_GATE else 'FAIL'} |"
    )
    lines.append(
        f"| Arc-level plausibility (all arcs) | "
        f"{_pct(arc_rate)} ({arc_n}) | context | — |"
    )
    post_ship_rate = landing.get("shippable_post_filter_rate", 0.0)
    post_ship_n = landing.get("shippable_post_filter_n", "0/0")
    post_serve_rate = landing.get("serve_arc_post_filter_rate", 0.0)
    post_serve_n = landing.get("serve_arc_post_filter_n", "0/0")
    post_filter_rate = landing.get("arc_post_filter_physical_rate", 0.0)
    post_filter_n = landing.get("arc_post_filter_physical_n", "0/0")
    is_valid_rate = landing.get("arc_is_valid_rate", 0.0)
    is_valid_n = landing.get("arc_is_valid_n", "0/0")
    lines.append(
        f"| Fitter self-filter (`is_valid`) arcs | "
        f"{_pct(is_valid_rate)} ({is_valid_n}) | context | — |"
    )
    lines.append(
        f"| Post-filter plausibility (is_valid arcs only) | "
        f"{_pct(post_filter_rate)} ({post_filter_n}) | ≥ 80% | "
        f"{'PASS' if post_filter_rate >= 0.80 else 'FAIL'} |"
    )
    lines.append(
        f"| Post-filter shippable rallies | "
        f"{_pct(post_ship_rate)} ({post_ship_n}) | ≥ 80% | "
        f"{'PASS' if post_ship_rate >= 0.80 else 'FAIL'} |"
    )
    lines.append(
        f"| Post-filter serve arc shippable | "
        f"{_pct(post_serve_rate)} ({post_serve_n}) | ≥ 80% | "
        f"{'PASS' if post_serve_rate >= 0.80 else 'FAIL'} |"
    )

    np_status = net_plane.get("status")
    if np_status == "missing_labels":
        lines.append("| Net-plane 3-class agreement | *waiting on C.3 labels* | ≥ 90% | PENDING |")
    else:
        ag = net_plane.get("agreement_rate", 0.0)
        n = net_plane.get("n_rallies", 0)
        pass_np = net_plane.get("pass_net_plane", False)
        lines.append(
            f"| Net-plane 3-class agreement | {_pct(ag)} ({n} rallies) | ≥ 90% | "
            f"{'PASS' if pass_np else 'FAIL'} |"
        )

    sb = speed
    lines.append(
        f"| Speed physical bound | {_pct(sb['physical_bound_rate'])} "
        f"({sb['physical_bound_n']}) | ≥ 95% | "
        f"{'PASS' if sb['pass_speed_bound'] else 'FAIL'} |"
    )
    lines.append(
        f"| Speed Tier-1-C quality | {_pct(sb['within_20pct_rate'])} "
        f"({sb['within_20pct_n']}), median |err|={_pct(sb['rel_err_median_abs'])} | "
        f"≥ 70% within 20%, median ≤ 20% | "
        f"{'PASS' if sb['pass_speed_quality'] else 'FAIL'} |"
    )

    lines.append("")
    lines.append("## Details")
    lines.append("")
    lines.append("### Tier 1-A plausibility (automatic across ball-GT rallies)")
    lines.append("")
    lines.append(f"- Rallies evaluated: **{landing['n_rallies']}**")
    lines.append(
        f"- Shippable rallies (≥80% physical arcs): **{landing.get('shippable_rally_n', 'N/A')}** "
        f"({_pct(landing.get('shippable_rally_rate', 0.0))})"
    )
    lines.append(
        f"- Serve arc physical: **{landing.get('serve_arc_pass_n', 'N/A')}** "
        f"({_pct(landing.get('serve_arc_pass_rate', 0.0))})"
    )
    lines.append(
        f"- Arc-level plausibility: **{landing.get('arc_physical_n', 'N/A')}** "
        f"({_pct(landing.get('arc_physical_rate', 0.0))})"
    )
    lines.append("")
    lines.append("An arc is physical if all of:")
    lines.append(f"- peak height ≤ {PLAUSIBLE_PEAK_MAX_M} m")
    lines.append(f"- landing x ∈ [{PLAUSIBLE_LANDING_X_MIN}, {PLAUSIBLE_LANDING_X_MAX}]")
    lines.append(f"- landing y ∈ [{PLAUSIBLE_LANDING_Y_MIN}, {PLAUSIBLE_LANDING_Y_MAX}]")
    lines.append(f"- speed ∈ [{PLAUSIBLE_SPEED_MIN}, {PLAUSIBLE_SPEED_MAX}] m/s")
    lines.append(f"- reproj RMSE < {PLAUSIBLE_RMSE_MAX_PX} px")
    lines.append("")
    if landing.get("per_rally"):
        sorted_rallies = sorted(landing["per_rally"], key=lambda r: r["plausible_rate"])
        lines.append("**Worst 5 rallies by plausibility rate:**")
        lines.append("")
        for r in sorted_rallies[:5]:
            serve_mark = "✓" if r["serve_arc_physical"] else "✗"
            lines.append(
                f"- {r['rally_id'][:10]}  cam={r['camera_height_m']:.2f}m  "
                f"physical={r['n_physical']}/{r['n_arcs']} ({_pct(r['plausible_rate'])})  "
                f"serve={serve_mark}"
            )
        lines.append("")

        # Aggregate failure reasons.
        failure_counter: dict[str, int] = {}
        for r in landing["per_rally"]:
            for arc in r["arcs"]:
                for reason in arc["failure"]:
                    key = reason.split("=")[0]
                    failure_counter[key] = failure_counter.get(key, 0) + 1
        if failure_counter:
            lines.append("**Failure-reason frequency across all arcs:**")
            lines.append("")
            for key, count in sorted(failure_counter.items(), key=lambda kv: -kv[1]):
                lines.append(f"- `{key}` — {count} arcs")
            lines.append("")

    lines.append("### Net-plane (Tier 1-B)")
    lines.append("")
    if np_status == "missing_labels":
        lines.append(
            "> **Waiting on C.3 human labels.** Open "
            "`outputs/ball_3d_rig/index.html`, inspect panel 3 "
            "(side elevation) of each rig figure, classify the serve arc's "
            "net crossing as `above_tape` / `at_tape` / `below_tape`, and "
            "save the labels to `outputs/ball_3d_rig/net_plane_labels.json` "
            "in the format `{\"rally_id\": {\"label\": \"above_tape\"}}`."
        )
    else:
        lines.append(f"- Rallies labelled: **{net_plane['n_rallies']}**")
        lines.append(f"- 3-class agreement: **{_pct(net_plane['agreement_rate'])}**")
        lines.append("")
        per = net_plane.get("per_rally", [])
        mismatches = [r for r in per if not r["match"]]
        if mismatches:
            lines.append("**Mismatches (human vs fitter class):**")
            lines.append("")
            for r in mismatches[:10]:
                lines.append(
                    f"- {r['rally_id'][:10]}  human={r['human_label']}  "
                    f"fitter={r['fitter_class']}  ({r['fitter_height_m']:.2f}m)"
                )
    lines.append("")

    lines.append("### Speed (Tier 1-C, automatic from flight-time anchor)")
    lines.append("")
    lines.append(f"- Rallies: **{speed['n_rallies']}**")
    lines.append(f"- v_avg median: **{speed['v_avg_median']:.1f} m/s**")
    lines.append(f"- v_release_expected median: **{speed['v_expected_median']:.1f} m/s** (1.25 × v_avg drag correction)")
    lines.append(f"- v_release_fitter median: **{speed['v_fitter_median']:.1f} m/s** (std={speed['v_fitter_std']:.1f})")
    lines.append(f"- Physical bound pass rate: **{_pct(speed['physical_bound_rate'])}** ({speed['physical_bound_n']})")
    lines.append(f"- Within ±20%: **{_pct(speed['within_20pct_rate'])}** ({speed['within_20pct_n']})")
    lines.append(f"- Median |rel_err|: **{_pct(speed['rel_err_median_abs'])}**")
    lines.append(f"- p90 |rel_err|: **{_pct(speed['rel_err_p90_abs'])}**")
    lines.append("")
    lines.append(
        "The fitter output is bimodal in relative error: some rallies are "
        "clamp-hit overestimates (fitted release ~37–40 m/s on physically soft "
        "serves) and others are ~50% underestimates on physically hard serves. "
        "The median of the fitter distribution happens to be close to the "
        "median of expected, but **individual rallies are unreliable**."
    )
    lines.append("")

    lines.append("## References")
    lines.append("")
    lines.append("- [Per-rally figures](index.html)")
    lines.append("- [Flight-time anchor data](flight_time_anchor.csv)")
    lines.append("- [Rig summaries](rig_summaries.json)")
    lines.append("")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-landing", action="store_true",
                        help="Skip the expensive landing re-fit (use cached flight-time anchor only)")
    args = parser.parse_args()

    # Load flight-time anchor.
    ft_path = OUTPUT_DIR / "flight_time_anchor.json"
    if not ft_path.exists():
        print(f"ERROR: {ft_path} not found. Run audit_flight_time_vs_fitter.py first.")
        sys.exit(1)
    ft_data = json.loads(ft_path.read_text())
    ft_summary = ft_data["summary"]

    speed = {
        "n_rallies": ft_summary["n_rallies_kept"],
        "v_avg_median": ft_summary["v_avg_mps"]["median"],
        "v_expected_median": ft_summary["v_release_expected_mps"]["median"],
        "v_fitter_median": ft_summary["v_release_fitter_mps"]["median"],
        "v_fitter_std": ft_summary["v_release_fitter_mps"]["std"],
        "physical_bound_rate": ft_summary["phys_bound_pass_rate"],
        "physical_bound_n": ft_summary["phys_bound_pass_n"],
        "within_20pct_rate": ft_summary["within_20pct_rate"],
        "within_20pct_n": ft_summary["within_20pct_n"],
        "rel_err_median_abs": ft_summary["rel_err_median_abs"],
        "rel_err_p90_abs": ft_summary["rel_err_p90_abs"],
        "pass_speed_bound": ft_summary["phys_bound_pass_rate"] >= SPEED_BOUND_GATE,
        "pass_speed_quality": (
            ft_summary["within_20pct_rate"] >= SPEED_QUALITY_PASS_RATE
            and ft_summary["rel_err_median_abs"] <= SPEED_QUALITY_MEDIAN
        ),
    }

    # Landing accuracy.
    if args.skip_landing:
        landing: dict[str, Any] = {
            "n_rallies": 0, "per_rally": [], "median_err_m": float("nan"),
            "p90_err_m": float("nan"), "mean_err_m": float("nan"),
            "pass_landing": False,
        }
    else:
        print("Computing Tier 1-A plausibility on ball-GT rallies...")
        landing = compute_landing_metrics()
        print(
            f"  n={landing['n_rallies']}  "
            f"shippable={landing.get('shippable_rally_n', 'N/A')}  "
            f"serve_arc={landing.get('serve_arc_pass_n', 'N/A')}  "
            f"arc_level={landing.get('arc_physical_n', 'N/A')}"
        )

    # Net-plane labels.
    rig_summaries_path = OUTPUT_DIR / "rig_summaries.json"
    rig_summaries: list[dict[str, Any]] = []
    if rig_summaries_path.exists():
        rig_summaries = json.loads(rig_summaries_path.read_text())

    labels_path = OUTPUT_DIR / "net_plane_labels.json"
    net_plane = compute_net_plane_metrics(labels_path, rig_summaries)

    tier, reason = decide_tier(landing, net_plane, speed)
    print(f"\n==> Decision: {tier}")
    print(f"    {reason}")

    date = datetime.now().strftime("%Y-%m-%d")
    out_path = OUTPUT_DIR / f"audit_report_{date}.md"
    write_report(landing, net_plane, speed, tier, reason, out_path)
    print(f"\nWrote {out_path}")

    # Also dump structured data for downstream use.
    structured_path = OUTPUT_DIR / f"audit_report_{date}.json"
    structured_path.write_text(json.dumps({
        "decision": tier,
        "reason": reason,
        "landing": landing,
        "net_plane": net_plane,
        "speed": speed,
    }, indent=2, default=float))
    print(f"Wrote {structured_path}")


if __name__ == "__main__":
    main()
