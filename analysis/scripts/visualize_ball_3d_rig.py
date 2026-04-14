"""4-panel verification rig for ball 3D trajectory reconstruction.

For each audit rally in ``outputs/ball_3d_rig/audit_rallies.json``, runs the
current fitter and renders a diagnostic figure with:

    1. Reprojection overlay — normalised image space; observed (WASB) vs
       reprojected predicted 3D ball position, colour-gradiented by time.
       Optionally a single mid-rally video frame as background if available.
    2. Top-down court view — metric xy coordinates, arc trajectories, known
       contact points from action GT, fitter-predicted landing, ball GT
       landing (when dense ball GT is available).
    3. Side elevation — y vs z with the net drawn at Y=8m, tape band at
       2.24m ±5cm, arcs colour-coded. This is the most sensitive panel for
       the depth/height coupling — a 40% speed underestimate looks visibly
       "flat" here.
    4. Per-frame reprojection residuals — px scale, grouped by arc. Tells
       us whether the fit is "certain but wrong" or "correctly uncertain."

Writes one PNG per rally to ``outputs/ball_3d_rig/<rally_id>.png`` plus
``index.html`` linking them. The index is the entry point for the human
audit of Phase C.

Usage
-----
    cd analysis
    uv run python scripts/visualize_ball_3d_rig.py
    uv run python scripts/visualize_ball_3d_rig.py --rally <rally-id>
    uv run python scripts/visualize_ball_3d_rig.py --no-video-frame
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH, CourtCalibrator
from rallycut.court.camera_model import (
    CameraModel,
    calibrate_camera,
    calibrate_camera_with_net,
    project_3d_to_image,
)
from rallycut.court.trajectory_3d import (
    GRAVITY,
    FittedArc,
    TrajectoryResult,
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

NET_Y = 8.0
NET_TAPE_M = 2.24
NET_TAPE_BAND = 0.05  # ±5 cm visual tape band

OUTPUT_DIR = Path("outputs/ball_3d_rig")
AUDIT_FILE = OUTPUT_DIR / "audit_rallies.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class RallyBundle:
    rally_id: str
    video_id: str
    fps: float
    ball_positions: list[BallPosition]
    ball_gt_positions: list[tuple[int, float, float]]  # (frame, x, y) normalised
    player_positions: list[PlayerPosition]
    action_gt: list[dict]
    video_cal: tuple[list[tuple[float, float]], int, int]
    net_y_image: float | None
    camera_height_m: float
    tier: str
    has_dense_ball_gt: bool
    video_s3_key: str | None
    video_content_hash: str | None


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
        for p in (positions or [])
    ]


def _parse_player_positions(pos_json: Any) -> list[PlayerPosition]:
    if not pos_json:
        return []
    return [
        PlayerPosition(
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            track_id=p.get("trackId", p.get("track_id", 0)),
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            width=p.get("width", 0.0),
            height=p.get("height", 0.0),
            confidence=p.get("confidence", 0.5),
        )
        for p in pos_json
    ]


def _extract_ball_gt(gt_json: Any) -> list[tuple[int, float, float]]:
    if not gt_json:
        return []
    positions = gt_json.get("positions", []) if isinstance(gt_json, dict) else gt_json
    out: list[tuple[int, float, float]] = []
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


def load_rally_bundles(audit_info: dict) -> list[RallyBundle]:
    """Load all DB data for the audit rallies."""
    rally_ids = [r["rally_id"] for r in audit_info["audit_rallies"]]
    rally_meta = {r["rally_id"]: r for r in audit_info["audit_rallies"]}

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                r.id, r.video_id, pt.fps,
                pt.ball_positions_json, pt.positions_json,
                pt.action_ground_truth_json, pt.ground_truth_json,
                v.court_calibration_json, v.width, v.height,
                v.s3_key, v.content_hash
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE r.id = ANY(%s)
        """, (rally_ids,))
        rows = cur.fetchall()

    bundles: list[RallyBundle] = []
    for row in rows:
        rid = str(row[0])
        meta = rally_meta[rid]
        cal_json = row[7]
        if not isinstance(cal_json, list) or len(cal_json) != 4:
            continue
        corners = [(c["x"], c["y"]) for c in cal_json]
        bundles.append(RallyBundle(
            rally_id=rid,
            video_id=str(row[1]),
            fps=float(row[2] or 30.0),
            ball_positions=_parse_ball_positions(row[3]),
            ball_gt_positions=_extract_ball_gt(row[6]),
            player_positions=_parse_player_positions(row[4]),
            action_gt=row[5] or [],
            video_cal=(corners, int(row[8] or 1920), int(row[9] or 1080)),
            net_y_image=None,  # filled in below via contact detector
            camera_height_m=meta.get("camera_height_m", 0.0),
            tier=meta["tier"],
            has_dense_ball_gt=meta.get("has_dense_ball_gt", False),
            video_s3_key=row[10],
            video_content_hash=row[11],
        ))

    return bundles


def estimate_net_y(bundle: RallyBundle) -> float | None:
    """Estimate the net image-Y for one rally via contact detector."""
    if len(bundle.ball_positions) < 20:
        return None
    calibrator = CourtCalibrator()
    calibrator.calibrate(bundle.video_cal[0])
    cs = detect_contacts(
        ball_positions=bundle.ball_positions,
        player_positions=bundle.player_positions,
        config=ContactDetectionConfig(),
        court_calibrator=calibrator,
    )
    if 0.1 < cs.net_y < 0.9:
        return float(cs.net_y)
    return None


def build_camera(bundle: RallyBundle, net_y: float | None) -> CameraModel | None:
    corners, w, h = bundle.video_cal
    cam: CameraModel | None = None
    if net_y is not None:
        cam = calibrate_camera_with_net(
            corners, COURT_CORNERS, w, h, net_y_image=net_y,
        )
    if cam is None or not cam.is_valid:
        cam = calibrate_camera(corners, COURT_CORNERS, w, h)
    return cam


def fit_bundle(bundle: RallyBundle, camera: CameraModel) -> TrajectoryResult:
    calibrator = CourtCalibrator()
    calibrator.calibrate(bundle.video_cal[0])
    cs = detect_contacts(
        ball_positions=bundle.ball_positions,
        player_positions=bundle.player_positions,
        config=ContactDetectionConfig(),
        court_calibrator=calibrator,
    )
    return fit_rally(
        camera=camera,
        contact_sequence=cs,
        classified_actions=None,
        fps=bundle.fps,
        rally_id=bundle.rally_id,
        video_id=bundle.video_id,
        net_height=NET_TAPE_M,
        joint=True,
    )


# ---------------------------------------------------------------------------
# Trajectory math
# ---------------------------------------------------------------------------


def arc_time_samples(arc: FittedArc, fps: float, n: int = 80) -> np.ndarray:
    duration = (arc.end_frame - arc.start_frame) / fps
    return np.linspace(0.0, max(duration, 0.05), n)


def eval_arc(arc: FittedArc, t: np.ndarray) -> np.ndarray:
    """Evaluate the 3D parabola at time offsets t (seconds from arc start)."""
    pos0 = arc.initial_position
    vel0 = arc.initial_velocity
    x = pos0[0] + vel0[0] * t
    y = pos0[1] + vel0[1] * t
    z = pos0[2] + vel0[2] * t - 0.5 * GRAVITY * t**2
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Video frame loading (optional)
# ---------------------------------------------------------------------------


def load_mid_rally_frame(
    bundle: RallyBundle, mid_frame: int,
) -> np.ndarray | None:
    try:
        import cv2

        from rallycut.evaluation.video_resolver import VideoResolver
    except Exception:  # noqa: BLE001
        return None
    if not bundle.video_s3_key or not bundle.video_content_hash:
        return None
    try:
        resolver = VideoResolver()
        if not resolver.is_cached(bundle.video_content_hash):
            return None  # don't download on the first pass; use cached only
        path = resolver.get_cached_path(bundle.video_content_hash)
        if path is None:
            return None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _time_colours(n: int) -> np.ndarray:
    return plt.cm.viridis(np.linspace(0, 1, max(n, 2)))


def render_panel1_reprojection(
    ax: plt.Axes,
    bundle: RallyBundle,
    camera: CameraModel,
    result: TrajectoryResult,
    frame_img: np.ndarray | None,
) -> None:
    """Panel 1: observed ball vs reprojected predicted ball in image space."""
    ax.set_title("1. Reprojection overlay (observed vs predicted)", fontsize=10)
    ax.set_aspect("equal")

    w, h = camera.image_size
    if frame_img is not None:
        ax.imshow(frame_img, extent=(0, 1, 1, 0), zorder=0, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
    else:
        # Schematic image rectangle.
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#444"))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(1.05, -0.05)

    # Observed ball positions (WASB).
    bp_by_frame: dict[int, BallPosition] = {b.frame_number: b for b in bundle.ball_positions}

    # Reprojected predicted 3D positions from each fitted arc.
    for arc in result.arcs:
        t = arc_time_samples(arc, bundle.fps, n=60)
        pts3d = eval_arc(arc, t)
        uv = np.array([project_3d_to_image(camera, p) for p in pts3d])
        ax.plot(uv[:, 0], uv[:, 1], "-", color="#0077ff", linewidth=1.5, alpha=0.8)

    # Observed ball detections colour-gradiented by time.
    obs_frames = sorted(bp_by_frame.keys())
    if obs_frames:
        colours = _time_colours(len(obs_frames))
        xs = [bp_by_frame[f].x for f in obs_frames]
        ys = [bp_by_frame[f].y for f in obs_frames]
        ax.scatter(xs, ys, c=colours, s=14, edgecolors="white", linewidths=0.3, zorder=5)

    # Predicted ball at each observed frame (to compare directly).
    for arc in result.arcs:
        for frame in range(arc.start_frame, arc.end_frame + 1):
            if frame not in bp_by_frame:
                continue
            t = (frame - arc.start_frame) / bundle.fps
            pt3d = eval_arc(arc, np.array([t]))[0]
            u, v = project_3d_to_image(camera, pt3d)
            ax.plot([bp_by_frame[frame].x, u], [bp_by_frame[frame].y, v],
                    "-", color="#ff4455", linewidth=0.5, alpha=0.4)
            ax.scatter(u, v, marker="x", color="#ff4455", s=20, zorder=6)

    ax.legend(
        handles=[
            mpatches.Patch(color="#0077ff", label="fitted 3D → reprojected"),
            mpatches.Patch(color="#ff4455", label="predicted at obs frames (x)"),
            mpatches.Patch(color="#440154", label="observed WASB (coloured by time)"),
        ],
        loc="upper left",
        fontsize=7,
        framealpha=0.8,
    )
    ax.set_xlabel("u (normalised)")
    ax.set_ylabel("v (normalised)")


def render_panel2_topdown(
    ax: plt.Axes,
    bundle: RallyBundle,
    result: TrajectoryResult,
) -> None:
    """Panel 2: top-down court view — metric XY."""
    ax.set_title("2. Top-down court (X, Y metric)", fontsize=10)
    ax.set_aspect("equal")

    # Court rectangle.
    ax.add_patch(mpatches.Rectangle(
        (0, 0), COURT_WIDTH, COURT_LENGTH,
        fill=False, edgecolor="black", linewidth=1.5,
    ))
    # Net line.
    ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], color="red", linewidth=1.5, alpha=0.6, label="net")

    # Fitter trajectories.
    colours = _time_colours(max(len(result.arcs), 2))
    for i, arc in enumerate(result.arcs):
        t = arc_time_samples(arc, bundle.fps, n=60)
        pts3d = eval_arc(arc, t)
        ax.plot(pts3d[:, 0], pts3d[:, 1], "-", color=colours[i], linewidth=1.5, alpha=0.8)
        # Landing markers.
        if arc.landing_position is not None:
            lx, ly = arc.landing_position
            ax.scatter([lx], [ly], marker="v", color=colours[i], s=60,
                       edgecolors="black", linewidths=0.5, zorder=5)

    # Contact frames from action GT plotted in court space if possible.
    # (Using known court-plane positions from action_gt ballX/ballY, which are
    # in normalised image coords — need calibrator back-project.)
    calibrator = CourtCalibrator()
    calibrator.calibrate(bundle.video_cal[0])
    for lab in bundle.action_gt:
        bx, by = lab.get("ballX"), lab.get("ballY")
        if bx is None or by is None:
            continue
        try:
            cx, cy = calibrator.image_to_court((bx, by), 1, 1)
        except Exception:  # noqa: BLE001
            continue
        action = lab.get("action", "?")
        marker = {"serve": "^", "receive": "D", "set": "s", "attack": "P"}.get(action, "o")
        ax.scatter([cx], [cy], marker=marker, s=60, color="orange",
                   edgecolors="black", linewidths=0.8, zorder=6, label=f"{action}")

    ax.set_xlim(-1, COURT_WIDTH + 1)
    ax.set_ylim(-1, COURT_LENGTH + 1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m) ← camera side")

    # Dedup legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    uniq: list = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq.append((h, l))
    if uniq:
        ax.legend([h for h, _ in uniq], [l for _, l in uniq],
                  loc="upper right", fontsize=7, framealpha=0.8)


def render_panel3_side(
    ax: plt.Axes,
    bundle: RallyBundle,
    result: TrajectoryResult,
) -> None:
    """Panel 3: side elevation — Y vs Z with net tape visible."""
    ax.set_title("3. Side elevation (Y, Z metric)", fontsize=10)

    # Net post at Y=8, tape band.
    ax.axvline(NET_Y, color="red", linewidth=2, alpha=0.4, label="net plane")
    ax.fill_betweenx(
        [NET_TAPE_M - NET_TAPE_BAND, NET_TAPE_M + NET_TAPE_BAND],
        NET_Y - 0.15, NET_Y + 0.15,
        color="white", edgecolor="red", linewidth=1.0, alpha=0.9,
        label="tape (2.24m)",
    )
    # Court extent.
    ax.plot([0, COURT_LENGTH], [0, 0], color="black", linewidth=1.0, alpha=0.5)

    colours = _time_colours(max(len(result.arcs), 2))
    for i, arc in enumerate(result.arcs):
        t = arc_time_samples(arc, bundle.fps, n=80)
        pts3d = eval_arc(arc, t)
        ax.plot(pts3d[:, 1], pts3d[:, 2], "-", color=colours[i], linewidth=1.5, alpha=0.8)
        # Peak height annotation.
        if arc.speed_at_start > 0:
            ax.annotate(
                f"{arc.speed_at_start:.1f}m/s",
                xy=(pts3d[np.argmax(pts3d[:, 2]), 1], pts3d[:, 2].max()),
                xytext=(4, 4), textcoords="offset points",
                fontsize=6, color=colours[i],
            )

    ax.set_xlim(-1, COURT_LENGTH + 1)
    ax.set_ylim(-0.5, 8)  # 8m ceiling — typical serve peak is 4-6m
    ax.set_xlabel("Y (m) ← camera side")
    ax.set_ylabel("Z (m, height)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)


def render_panel4_residuals(
    ax: plt.Axes,
    bundle: RallyBundle,
    camera: CameraModel,
    result: TrajectoryResult,
) -> None:
    """Panel 4: per-frame reprojection residuals in pixels."""
    ax.set_title("4. Reprojection residual (px per obs frame)", fontsize=10)

    bp_by_frame: dict[int, BallPosition] = {b.frame_number: b for b in bundle.ball_positions}
    w, h = camera.image_size
    max_side = max(w, h)

    colours = _time_colours(max(len(result.arcs), 2))
    all_frames: list[int] = []
    all_errs: list[float] = []
    all_colours: list[tuple[float, float, float, float]] = []

    for i, arc in enumerate(result.arcs):
        for frame in range(arc.start_frame, arc.end_frame + 1):
            if frame not in bp_by_frame:
                continue
            t = (frame - arc.start_frame) / bundle.fps
            pt3d = eval_arc(arc, np.array([t]))[0]
            u, v = project_3d_to_image(camera, pt3d)
            obs = bp_by_frame[frame]
            err_norm = math.sqrt((u - obs.x) ** 2 + (v - obs.y) ** 2)
            err_px = err_norm * max_side
            all_frames.append(frame)
            all_errs.append(err_px)
            all_colours.append(colours[i])

    if all_frames:
        ax.bar(all_frames, all_errs, color=all_colours, alpha=0.8, width=1.2)
        median = float(np.median(all_errs))
        p90 = float(np.percentile(all_errs, 90))
        ax.axhline(5.0, color="green", linestyle="--", linewidth=0.8, label="5px (ship gate)")
        ax.axhline(median, color="orange", linestyle=":", linewidth=0.8, label=f"median={median:.1f}px")
        ax.axhline(p90, color="red", linestyle=":", linewidth=0.8, label=f"p90={p90:.1f}px")
        # Log scale when residuals span a large range, so 5px gate stays visible.
        if max(all_errs) > 50:
            ax.set_yscale("log")
            ax.set_ylim(0.5, max(all_errs) * 1.2)

    ax.set_xlabel("frame")
    ax.set_ylabel("reproj error (px)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.3)


def render_rally_figure(
    bundle: RallyBundle,
    camera: CameraModel,
    result: TrajectoryResult,
    frame_img: np.ndarray | None,
    out_path: Path,
) -> dict[str, Any]:
    """Render the 4-panel figure for one rally. Returns summary stats."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    # Pre-compute rig-level reproj stats for the title.
    bp_by_frame: dict[int, BallPosition] = {b.frame_number: b for b in bundle.ball_positions}
    _max_side = max(camera.image_size)
    all_errs_px: list[float] = []
    for arc in result.arcs:
        for frame in range(arc.start_frame, arc.end_frame + 1):
            if frame not in bp_by_frame:
                continue
            t = (frame - arc.start_frame) / bundle.fps
            pt3d = eval_arc(arc, np.array([t]))[0]
            u, v = project_3d_to_image(camera, pt3d)
            obs = bp_by_frame[frame]
            err = math.sqrt((u - obs.x) ** 2 + (v - obs.y) ** 2) * _max_side
            all_errs_px.append(err)
    median_reproj = float(np.median(all_errs_px)) if all_errs_px else float("nan")
    p90_reproj = float(np.percentile(all_errs_px, 90)) if all_errs_px else float("nan")

    serve_str = ""
    if result.arcs and result.arcs[0].speed_at_start:
        serve_str = f"  serve={result.arcs[0].speed_at_start:.1f}m/s"

    fig.suptitle(
        f"{bundle.rally_id[:10]}  [{bundle.tier}, cam={bundle.camera_height_m:.2f}m]  "
        f"vid={bundle.video_id[:10]}  fps={bundle.fps:.0f}  arcs={len(result.arcs)}  "
        f"obs={len(bundle.ball_positions)}{serve_str}  "
        f"reproj med={median_reproj:.0f}/p90={p90_reproj:.0f}px  "
        f"ball_gt={'yes' if bundle.has_dense_ball_gt else 'no'}",
        fontsize=10,
    )

    render_panel1_reprojection(axes[0, 0], bundle, camera, result, frame_img)
    render_panel2_topdown(axes[0, 1], bundle, result)
    render_panel3_side(axes[1, 0], bundle, result)
    render_panel4_residuals(axes[1, 1], bundle, camera, result)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Compute summary.
    serve_speed = result.arcs[0].speed_at_start if result.arcs else None
    summary = {
        "rally_id": bundle.rally_id,
        "video_id": bundle.video_id,
        "tier": bundle.tier,
        "camera_height_m": bundle.camera_height_m,
        "n_arcs": len(result.arcs),
        "n_ball_obs": len(bundle.ball_positions),
        "serve_speed_mps": serve_speed,
        "arc_peak_heights": [a.peak_height for a in result.arcs],
        "arc_net_crossings": [a.net_crossing_height for a in result.arcs],
        "arc_speeds": [a.speed_at_start for a in result.arcs],
        "arc_landings": [a.landing_position for a in result.arcs],
        "arc_reproj_rmse_px": [a.reprojection_rmse for a in result.arcs],
        "reproj_median_px": median_reproj,
        "reproj_p90_px": p90_reproj,
        "has_dense_ball_gt": bundle.has_dense_ball_gt,
    }
    return summary


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


def write_index(summaries: list[dict[str, Any]], out_path: Path) -> None:
    """Write an HTML index linking all rally figures."""
    sorted_summaries = sorted(
        summaries,
        key=lambda s: (s["tier"], s.get("camera_height_m", 0), s["rally_id"]),
    )
    rows: list[str] = []
    for s in sorted_summaries:
        serve_str = f"{s['serve_speed_mps']:.1f} m/s" if s.get("serve_speed_mps") else "—"
        peaks = s.get("arc_peak_heights", [])
        peaks_str = ", ".join(f"{p:.1f}" for p in peaks[:4])
        rows.append(f"""
    <tr>
      <td><a href="{s['rally_id']}.png">{s['rally_id'][:10]}</a></td>
      <td>{s['tier']}</td>
      <td>{s.get('camera_height_m', 0):.2f} m</td>
      <td>{s['n_arcs']}</td>
      <td>{s['n_ball_obs']}</td>
      <td>{serve_str}</td>
      <td>{peaks_str}</td>
      <td>{'✓' if s['has_dense_ball_gt'] else '—'}</td>
      <td><img src="{s['rally_id']}.png" width="220"/></td>
    </tr>""")

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Ball 3D Rig — audit set</title>
<style>
body {{ font-family: -apple-system, Helvetica, sans-serif; max-width: 1400px; margin: 2em auto; padding: 0 1em; }}
h1 {{ font-size: 1.4em; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f0; }}
tr:hover {{ background: #f9f9f9; }}
img {{ display: block; border: 1px solid #ddd; }}
</style></head>
<body>
<h1>Ball 3D Verification Rig — Phase C audit set</h1>
<p>{len(summaries)} audit rallies. Click a rally ID to open the full-size figure.</p>
<table>
  <thead>
    <tr>
      <th>rally</th>
      <th>tier</th>
      <th>cam h</th>
      <th># arcs</th>
      <th># obs</th>
      <th>serve v</th>
      <th>peak heights (m)</th>
      <th>ball GT</th>
      <th>preview</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}
  </tbody>
</table>
</body></html>"""

    out_path.write_text(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rally", help="Process a single rally id")
    parser.add_argument("--no-video-frame", action="store_true",
                        help="Skip loading a background video frame")
    parser.add_argument("--audit-file", default=str(AUDIT_FILE))
    args = parser.parse_args()

    audit_path = Path(args.audit_file)
    if not audit_path.exists():
        print(f"ERROR: audit file not found at {audit_path}")
        print("Run scripts/select_ball_3d_audit_rallies.py first.")
        sys.exit(1)

    audit_info = json.loads(audit_path.read_text())
    bundles = load_rally_bundles(audit_info)
    if args.rally:
        bundles = [b for b in bundles if b.rally_id == args.rally]
        if not bundles:
            print(f"ERROR: rally {args.rally} not found in audit set")
            sys.exit(1)

    print(f"Loaded {len(bundles)} audit rallies")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for i, bundle in enumerate(bundles, 1):
        net_y = estimate_net_y(bundle)
        camera = build_camera(bundle, net_y)
        if camera is None:
            print(f"  [{i}/{len(bundles)}] {bundle.rally_id[:10]} SKIPPED — no camera model")
            continue
        result = fit_bundle(bundle, camera)

        frame_img: np.ndarray | None = None
        if not args.no_video_frame and bundle.ball_positions:
            mid = bundle.ball_positions[len(bundle.ball_positions) // 2].frame_number
            frame_img = load_mid_rally_frame(bundle, mid)

        out_path = OUTPUT_DIR / f"{bundle.rally_id}.png"
        summary = render_rally_figure(bundle, camera, result, frame_img, out_path)
        summary["camera_height_m"] = float(camera.camera_position[2])
        summaries.append(summary)

        serve = f"{summary['serve_speed_mps']:.1f}m/s" if summary.get("serve_speed_mps") else "—"
        print(
            f"  [{i}/{len(bundles)}] {bundle.rally_id[:10]} "
            f"[{bundle.tier}, cam={camera.camera_position[2]:.2f}m] "
            f"arcs={len(result.arcs)} obs={len(bundle.ball_positions)} serve={serve}"
        )

    index_path = OUTPUT_DIR / "index.html"
    write_index(summaries, index_path)
    print(f"\nWrote {len(summaries)} figures and {index_path}")

    # Dump summaries as JSON for the audit report to consume.
    summary_path = OUTPUT_DIR / "rig_summaries.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=float))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
