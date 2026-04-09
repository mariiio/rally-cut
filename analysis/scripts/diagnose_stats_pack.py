"""Combined pre-work diagnostic for W3 stats quality pack #1.

Runs the production pipeline once per rally and evaluates four candidate
stats in a single pass (the pipeline is the expensive step, not the
per-stat math):

    1. attack direction   — cut / cross / line, horizontal homography
    2. attack type        — shot / power, pixel-space only
    3. set zones          — origin → destination (1–5), horizontal homography
    4. serve speed v2     — pixel velocity × per-video horizontal scale factor
                            (sidesteps the above-court ball-height blow-up)

Each stat has its own ship-gate block at the end of the report. No
production mutations; dumps a JSON to
``analysis/outputs/stats_pack_diagnostic.json``.

Usage
-----
    cd analysis
    uv run python scripts/diagnose_stats_pack.py
    uv run python scripts/diagnose_stats_pack.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_action_detection import (  # noqa: E402
    _build_player_positions,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
)
from production_eval import (  # noqa: E402
    PipelineContext,
    _build_calibrators,
    _parse_ball,
    _parse_positions,
)

from rallycut.tracking.action_classifier import classify_rally_actions  # noqa: E402
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import detect_contacts  # noqa: E402
from rallycut.tracking.match_tracker import verify_team_assignments  # noqa: E402
from rallycut.tracking.sequence_action_runtime import (  # noqa: E402
    apply_sequence_override,
    get_sequence_probs,
)

console = Console()

# Physical bounds for serve speed sanity.
MIN_MPS = 5.0
MAX_MPS = 30.0

# Serve type thresholds (to be tuned from the diagnostic distribution).
JUMP_MPS = 18.0
STANDING_MPS = 10.0
JUMP_RISE = 0.05
FLOAT_RISE = 0.03

# Attack direction bins (degrees, measured relative to the attacker→opposite-net
# direction, where 0° = straight across, +90° = toward right sideline).
CUT_DEG = 15.0  # |angle| < CUT_DEG → line; narrowed from 25° after iteration 2.
CROSS_VS_CUT_DEG = 40.0  # |angle| >= CROSS_VS_CUT_DEG → cut (sharp angle)

# Attack type thresholds (pixel-space). Set from observed distributions.
POWER_PIX_VEL = 0.025  # normalized units / frame
SHOT_PIX_VEL = 0.012
POWER_ARC_MAX = 0.020
SHOT_ARC_MIN = 0.035


def _local_ground_speed_mps(
    calibrator: Any,
    p0: BallPosition,
    p1: BallPosition,
    fps: float,
) -> float | None:
    """Ground-plane speed via the homography's local Jacobian at the ball.

    Evaluates the image→court projection at the midpoint between p0 and p1
    plus two epsilon neighbors. This gives a local 2×2 Jacobian mapping
    normalized image deltas → court-plane deltas in meters. We then
    transform (p1 - p0) through that Jacobian and take the hypotenuse.

    This is still wrong in absolute terms for balls above the court plane
    (the Jacobian describes the ground-ray intersection, not the ball's
    ground shadow), but it is internally consistent across the image and
    avoids the axis-asymmetry + vanishing-point pathologies of a single
    global scale factor.
    """
    mx = (p0.x + p1.x) * 0.5
    my = (p0.y + p1.y) * 0.5
    eps = 0.005  # normalized image units
    try:
        c0 = calibrator.image_to_court((mx, my), 1, 1)
        cu = calibrator.image_to_court((mx + eps, my), 1, 1)
        cv = calibrator.image_to_court((mx, my + eps), 1, 1)
    except Exception:  # noqa: BLE001
        return None
    # Jacobian columns: ∂court/∂u and ∂court/∂v, per unit normalized image.
    jxu = (cu[0] - c0[0]) / eps
    jyu = (cu[1] - c0[1]) / eps
    jxv = (cv[0] - c0[0]) / eps
    jyv = (cv[1] - c0[1]) / eps
    du = p1.x - p0.x
    dv = p1.y - p0.y
    dx_court = jxu * du + jxv * dv
    dy_court = jyu * du + jyv * dv
    dist_m = math.hypot(dx_court, dy_court)
    dframes = abs(p1.frame_number - p0.frame_number)
    if dframes == 0:
        return None
    dt = dframes / fps
    return dist_m / dt


def _video_scale_factor(calibrator: Any) -> float | None:
    """Per-video meters-per-normalized-x at the court.

    Uses ``HomographyResult.image_corners`` / ``court_corners`` already
    stored on the calibrator. Returns None if not calibrated or degenerate.
    """
    hom = getattr(calibrator, "_homography", None)
    if hom is None:
        return None
    img = hom.image_corners
    crt = hom.court_corners
    if not img or not crt or len(img) < 4 or len(crt) < 4:
        return None
    # Average the top-edge (img[0]→img[1]) and bottom-edge (img[3]→img[2])
    # pixel lengths, against their known court distances.
    scales: list[float] = []
    for a, b in ((0, 1), (3, 2)):
        dx_img = math.hypot(img[b][0] - img[a][0], img[b][1] - img[a][1])
        dx_crt = math.hypot(crt[b][0] - crt[a][0], crt[b][1] - crt[a][1])
        if dx_img > 1e-6:
            scales.append(dx_crt / dx_img)
    if not scales:
        return None
    return sum(scales) / len(scales)


def _ball_at(
    ball_positions: list[BallPosition], target: int, radius: int = 3
) -> BallPosition | None:
    best: BallPosition | None = None
    best_d = radius + 1
    for bp in ball_positions:
        d = abs(bp.frame_number - target)
        if d <= radius and d < best_d:
            best = bp
            best_d = d
    return best


def _ball_court_xy(
    calibrator: Any, bp: BallPosition
) -> tuple[float, float] | None:
    try:
        return calibrator.image_to_court((bp.x, bp.y), 1, 1)
    except Exception:  # noqa: BLE001
        return None


def _run_rally(
    rally: Any, match_teams: dict[int, int] | None, calibrator: Any, ctx: PipelineContext
) -> tuple[list[dict], list[BallPosition]]:
    ball_positions = _parse_ball(rally.ball_positions_json or [])
    player_positions = _build_player_positions(
        rally.positions_json or [], rally_id=rally.rally_id, inject_pose=True
    )
    teams = dict(match_teams) if match_teams else None
    if teams is not None and not ctx.skip_verify_teams:
        teams = verify_team_assignments(teams, player_positions)
    sequence_probs = get_sequence_probs(
        ball_positions=ball_positions,
        player_positions=player_positions,
        court_split_y=rally.court_split_y,
        frame_count=rally.frame_count,
        team_assignments=teams,
        calibrator=calibrator,
    )
    contact_sequence = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=None,
        use_classifier=True,
        frame_count=rally.frame_count or None,
        team_assignments=teams,
        court_calibrator=calibrator,
        sequence_probs=sequence_probs,
    )
    rally_actions = classify_rally_actions(
        contact_sequence,
        rally.rally_id,
        team_assignments=teams,
        match_team_assignments=teams,
        calibrator=calibrator,
    )
    if sequence_probs is not None:
        apply_sequence_override(rally_actions, sequence_probs)
    pred_dicts = [a.to_dict() for a in rally_actions.actions if not a.is_synthetic]
    return pred_dicts, ball_positions


# ------------------------------ per-stat analyzers ------------------------------

def _analyze_serve_speed(
    pred_actions: list[dict],
    ball_positions: list[BallPosition],
    calibrator: Any,
    fps: float,
) -> dict[str, Any] | None:
    serve = next((a for a in pred_actions if a.get("action") == "serve"), None)
    if serve is None:
        return None
    sf = int(serve.get("frame", -1))
    # Post-contact window only: avoid the ball-at-toss-peak near-stationary
    # frames that make a symmetric window underestimate speed.
    pre = _ball_at(ball_positions, sf + 1, radius=2)
    post = _ball_at(ball_positions, sf + 5, radius=3)
    if pre is None or post is None:
        return {"mps": None, "type": "missing_ball"}
    dframes = abs(post.frame_number - pre.frame_number)
    if dframes == 0:
        return {"mps": None, "type": "zero_dt"}
    if calibrator is None or not getattr(calibrator, "is_calibrated", False):
        return {"mps": None, "type": "no_scale"}
    mps = _local_ground_speed_mps(calibrator, pre, post, fps)
    if mps is None:
        return {"mps": None, "type": "no_projection"}
    # Pixel velocity retained only for type classification.
    pix_v = math.hypot(post.x - pre.x, post.y - pre.y) * fps / dframes
    # Peak rise (image y-up) in the 6 frames post-contact.
    sbp = _ball_at(ball_positions, sf, radius=2)
    peak_rise = 0.0
    if sbp is not None:
        for bp in ball_positions:
            if 0 <= (bp.frame_number - sf) <= 6:
                rise = sbp.y - bp.y
                if rise > peak_rise:
                    peak_rise = rise
    if mps >= JUMP_MPS and peak_rise > JUMP_RISE:
        stype = "jump"
    elif mps < STANDING_MPS and peak_rise < FLOAT_RISE:
        stype = "standing"
    elif peak_rise < FLOAT_RISE:
        stype = "float"
    else:
        stype = "unknown"
    return {"mps": mps, "peak_rise": peak_rise, "type": stype, "pix_v": pix_v}


def _analyze_attack_direction(
    pred_actions: list[dict],
    ball_positions: list[BallPosition],
    calibrator: Any,
) -> list[dict[str, Any]]:
    """Direction from ball LANDING point, not fixed-frame sample.

    Walks forward from the attack contact and takes the first ball sample
    that is either (a) past the net (y crosses the court midline y=8m) or
    (b) ≥20 frames in the future. This ensures dx accumulates over the full
    flight, so cross-court attacks don't get binned as line just because
    the 6-frame window was too short.
    """
    out: list[dict[str, Any]] = []
    attacks = [a for a in pred_actions if a.get("action") == "attack"]
    for a in attacks:
        af = int(a.get("frame", -1))
        contact_bp = _ball_at(ball_positions, af, radius=2)
        if contact_bp is None:
            out.append({"frame": af, "dir": "missing_ball"})
            continue
        pc = _ball_court_xy(calibrator, contact_bp)
        if pc is None:
            out.append({"frame": af, "dir": "no_projection"})
            continue
        contact_y_court = pc[1]
        net_y = 8.0  # COURT_LENGTH / 2 — beach volleyball net midline
        land_bp: BallPosition | None = None
        for bp in ball_positions:
            dt = bp.frame_number - af
            if dt <= 0:
                continue
            if dt > 45:  # cap at ~1.5s; avoid runaway
                break
            p = _ball_court_xy(calibrator, bp)
            if p is None:
                continue
            # Crossed net? i.e. y now on the opposite side of net_y vs contact.
            crossed = (
                (contact_y_court < net_y and p[1] >= net_y)
                or (contact_y_court > net_y and p[1] <= net_y)
            )
            if crossed or dt >= 20:
                land_bp = bp
                break
        if land_bp is None:
            out.append({"frame": af, "dir": "no_landing"})
            continue
        pf = _ball_court_xy(calibrator, land_bp)
        if pf is None:
            out.append({"frame": af, "dir": "no_projection"})
            continue
        dx = pf[0] - pc[0]
        dy = pf[1] - pc[1]
        # Horizontal angle in court space; cross-court = large |dx|/|dy|.
        if abs(dy) < 1e-6:
            out.append({"frame": af, "dir": "no_depth"})
            continue
        # angle from the straight-across-net direction (|dy|-axis).
        # atan2(dx, |dy|): 0 = straight across, +/- toward sidelines.
        angle_deg = math.degrees(math.atan2(dx, abs(dy)))
        abs_a = abs(angle_deg)
        if abs_a < CUT_DEG:
            d = "line"
        elif abs_a < CROSS_VS_CUT_DEG:
            d = "cross"
        else:
            d = "cut"
        out.append({"frame": af, "dir": d, "angle_deg": angle_deg, "dx": dx, "dy": dy})
    return out


def _analyze_attack_type(
    pred_actions: list[dict],
    ball_positions: list[BallPosition],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in pred_actions:
        if a.get("action") != "attack":
            continue
        af = int(a.get("frame", -1))
        pre = _ball_at(ball_positions, af - 1, radius=2)
        post = _ball_at(ball_positions, af + 2, radius=3)
        if pre is None or post is None:
            out.append({"frame": af, "type": "missing_ball"})
            continue
        dframes = abs(post.frame_number - pre.frame_number) or 1
        pix_v = math.hypot(post.x - pre.x, post.y - pre.y) / dframes
        # Arc ratio: max vertical rise (ball y smaller = higher) in the 6 frames
        # post-contact divided by horizontal displacement over same window.
        sbp = _ball_at(ball_positions, af, radius=2)
        peak_rise = 0.0
        horiz = 1e-6
        if sbp is not None:
            for bp in ball_positions:
                dt = bp.frame_number - af
                if 0 <= dt <= 6:
                    rise = sbp.y - bp.y
                    if rise > peak_rise:
                        peak_rise = rise
                    horiz = max(horiz, abs(bp.x - sbp.x))
        arc_ratio = peak_rise / horiz
        if pix_v >= POWER_PIX_VEL and arc_ratio <= POWER_ARC_MAX:
            t = "power"
        elif pix_v <= SHOT_PIX_VEL or arc_ratio >= SHOT_ARC_MIN:
            t = "shot"
        else:
            t = "unknown"
        out.append({
            "frame": af, "type": t, "pix_v": pix_v, "arc_ratio": arc_ratio,
            "peak_rise": peak_rise,
        })
    return out


def _analyze_set_zones(
    pred_actions: list[dict],
    ball_positions: list[BallPosition],
    positions_raw: list[dict],
    calibrator: Any,
) -> list[dict[str, Any]]:
    """Origin = SETTER court-x at set contact. Dest = ball court-x at
    next attack contact. Zones 1–5 span the 8m court width left to right.
    """
    out: list[dict[str, Any]] = []
    actions_sorted = sorted(pred_actions, key=lambda a: a.get("frame", 0))

    # Build a quick lookup: (frame, track_id) → (x, y) for setter position.
    pos_by_key: dict[tuple[int, int], tuple[float, float]] = {}
    for pp in positions_raw:
        try:
            pos_by_key[(int(pp["frameNumber"]), int(pp["trackId"]))] = (
                float(pp["x"]), float(pp["y"])
            )
        except (KeyError, TypeError, ValueError):
            continue

    def _zone(x_m: float) -> int:
        z = int(x_m / 1.6) + 1
        return max(1, min(5, z))

    def _setter_xy(sf: int, tid: int) -> tuple[float, float] | None:
        # Nearest-frame lookup ±3 frames (pose cache can skip frames).
        best: tuple[float, float] | None = None
        best_d = 4
        for d in range(0, 4):
            for delta in (-d, d):
                xy = pos_by_key.get((sf + delta, tid))
                if xy is not None and abs(delta) < best_d:
                    best = xy
                    best_d = abs(delta)
                    if delta == 0:
                        return best
        return best

    for i, a in enumerate(actions_sorted):
        if a.get("action") != "set":
            continue
        sf = int(a.get("frame", -1))
        setter_tid = int(a.get("playerTrackId", -1))
        next_attack: dict | None = None
        for b in actions_sorted[i + 1 :]:
            if b.get("action") == "attack":
                next_attack = b
                break
        if next_attack is None:
            out.append({"frame": sf, "origin": None, "dest": None, "why": "no_next_attack"})
            continue

        # Origin = setter court position at set frame.
        origin_zone: int | None = None
        if setter_tid >= 0:
            setter_img = _setter_xy(sf, setter_tid)
            if setter_img is not None and calibrator is not None:
                try:
                    px, py = calibrator.image_to_court(setter_img, 1, 1)
                    origin_zone = _zone(px)
                except Exception:  # noqa: BLE001
                    pass

        # Dest = ball court-x at next attack contact.
        dest_zone: int | None = None
        atk_bp = _ball_at(ball_positions, int(next_attack.get("frame", -1)), radius=3)
        if atk_bp is not None and calibrator is not None:
            pb = _ball_court_xy(calibrator, atk_bp)
            if pb is not None:
                dest_zone = _zone(pb[0])

        out.append({
            "frame": sf,
            "origin": origin_zone,
            "dest": dest_zone,
            "why": None if (origin_zone and dest_zone) else "partial",
        })
    return out


# ------------------------------ main loop ------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    if args.limit:
        rallies = rallies[: args.limit]
    console.print(f"  {len(rallies)} rallies")

    rally_pos_lookup: dict[str, list[Any]] = {
        r.rally_id: _parse_positions(r.positions_json) for r in rallies if r.positions_json
    }
    video_ids = {r.video_id for r in rallies}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    calibrators = _build_calibrators(video_ids)
    ctx = PipelineContext()

    n_calibrated = sum(1 for cal in calibrators.values() if getattr(cal, "is_calibrated", False))
    console.print(f"  calibrated videos: {n_calibrated}/{len(calibrators)}")

    # Accumulators per stat.
    # --- serve speed ---
    serve_pred = 0
    serve_mps: list[float] = []
    serve_oob = 0
    serve_types: Counter[str] = Counter()
    per_video_mps: dict[str, list[float]] = {}
    # --- attack direction ---
    atk_dir_total = 0
    atk_dir_counts: Counter[str] = Counter()
    atk_dir_samples: list[dict[str, Any]] = []
    # --- attack type ---
    atk_type_total = 0
    atk_type_counts: Counter[str] = Counter()
    atk_pix_velocities: list[float] = []
    atk_arc_ratios: list[float] = []
    # --- set zones ---
    set_total = 0
    set_covered = 0
    set_transitions: Counter[tuple[int, int]] = Counter()

    n_rejected = 0
    per_rally: list[dict] = []

    for idx, rally in enumerate(rallies, start=1):
        if not rally.ball_positions_json or not rally.positions_json or not rally.frame_count:
            n_rejected += 1
            continue
        calibrator = calibrators.get(rally.video_id)
        try:
            pred_actions, ball_positions = _run_rally(
                rally, team_map.get(rally.rally_id), calibrator, ctx
            )
        except Exception as exc:  # noqa: BLE001
            n_rejected += 1
            console.print(f"  [red][{idx}/{len(rallies)}][/red] {rally.rally_id}: {type(exc).__name__}: {exc}")
            continue

        # --- serve speed ---
        sres = _analyze_serve_speed(pred_actions, ball_positions, calibrator, rally.fps)
        if sres is not None:
            serve_pred += 1
            mps = sres.get("mps")
            if isinstance(mps, (int, float)):
                serve_mps.append(float(mps))
                per_video_mps.setdefault(rally.video_id, []).append(float(mps))
                if mps < MIN_MPS or mps > MAX_MPS:
                    serve_oob += 1
                serve_types[sres["type"]] += 1
            else:
                serve_types[sres["type"]] += 1

        # --- attack direction ---
        for r in _analyze_attack_direction(pred_actions, ball_positions, calibrator):
            atk_dir_total += 1
            atk_dir_counts[r["dir"]] += 1
            if "angle_deg" in r and len(atk_dir_samples) < 25:
                atk_dir_samples.append({
                    "rally_id": rally.rally_id,
                    "frame": r["frame"],
                    "dir": r["dir"],
                    "angle_deg": round(r["angle_deg"], 1),
                    "dx": round(r.get("dx", 0.0), 2),
                    "dy": round(r.get("dy", 0.0), 2),
                })

        # --- attack type ---
        for r in _analyze_attack_type(pred_actions, ball_positions):
            atk_type_total += 1
            atk_type_counts[r["type"]] += 1
            pv = r.get("pix_v")
            ar = r.get("arc_ratio")
            if isinstance(pv, (int, float)):
                atk_pix_velocities.append(float(pv))
            if isinstance(ar, (int, float)) and math.isfinite(ar):
                atk_arc_ratios.append(float(ar))

        # --- set zones ---
        for r in _analyze_set_zones(
            pred_actions, ball_positions, rally.positions_json or [], calibrator
        ):
            set_total += 1
            if r.get("origin") is not None and r.get("dest") is not None:
                set_covered += 1
                set_transitions[(int(r["origin"]), int(r["dest"]))] += 1

        if idx % 20 == 0 or idx == len(rallies):
            console.print(f"  [{idx}/{len(rallies)}] processed")

    # ------------------------------ reports ------------------------------
    console.print()
    console.print(f"rejections: {n_rejected}")

    # ---- Stat 4: serve speed ----
    console.print()
    console.print("[bold]Stat 4 — serve speed v2 (per-video scale factor)[/bold]")
    console.print(f"  pred serves:              {serve_pred}")
    console.print(f"  mps computed:             {len(serve_mps)}")
    if serve_pred:
        cov = 100.0 * len(serve_mps) / serve_pred
        console.print(f"  coverage vs pred serves:  {cov:.1f}%")
    console.print(f"  out-of-bounds [{MIN_MPS},{MAX_MPS}]: {serve_oob}")
    if serve_mps:
        s_sorted = sorted(serve_mps)
        console.print(f"  min/med/max: {min(serve_mps):.2f} / {statistics.median(serve_mps):.2f} / {max(serve_mps):.2f}")
        console.print(f"  mean±sd: {statistics.mean(serve_mps):.2f} ± {statistics.stdev(serve_mps) if len(serve_mps) >= 2 else 0.0:.2f}")
        if len(per_video_mps) >= 2:
            cross = statistics.stdev(serve_mps)
            within = [statistics.stdev(v) for v in per_video_mps.values() if len(v) >= 2]
            avg_w = statistics.mean(within) if within else 0.0
            ratio = (avg_w / cross) if cross else float("nan")
            console.print(f"  within/cross stdev ratio: {ratio:.2f} (gate < 0.70)")
    tbl_s = Table(title="serve type distribution")
    tbl_s.add_column("type"); tbl_s.add_column("count", justify="right")
    for k, v in serve_types.most_common():
        tbl_s.add_row(k, str(v))
    console.print(tbl_s)

    # ---- Stat 1: attack direction ----
    console.print()
    console.print("[bold]Stat 1 — attack direction[/bold]")
    console.print(f"  total attacks analyzed: {atk_dir_total}")
    tbl_d = Table(title="attack direction distribution")
    tbl_d.add_column("class"); tbl_d.add_column("count", justify="right")
    for k, v in atk_dir_counts.most_common():
        tbl_d.add_row(k, str(v))
    console.print(tbl_d)
    if atk_dir_samples:
        console.print("  sample attack angles (for manual sanity check):")
        for s in atk_dir_samples:
            console.print(
                f"    {s['rally_id'][:8]} f={s['frame']:4d} "
                f"angle={s['angle_deg']:+6.1f}° dx={s['dx']:+5.2f} dy={s['dy']:+5.2f} → {s['dir']}"
            )

    # ---- Stat 2: attack type ----
    console.print()
    console.print("[bold]Stat 2 — attack type[/bold]")
    console.print(f"  total attacks analyzed: {atk_type_total}")
    tbl_t = Table(title="attack type distribution")
    tbl_t.add_column("class"); tbl_t.add_column("count", justify="right")
    for k, v in atk_type_counts.most_common():
        tbl_t.add_row(k, str(v))
    console.print(tbl_t)
    if atk_pix_velocities:
        pv_sorted = sorted(atk_pix_velocities)
        console.print(
            f"  pix_v quartiles: "
            f"{pv_sorted[len(pv_sorted)//4]:.4f} / "
            f"{pv_sorted[len(pv_sorted)//2]:.4f} / "
            f"{pv_sorted[3*len(pv_sorted)//4]:.4f}"
        )
    if atk_arc_ratios:
        ar_sorted = sorted(atk_arc_ratios)
        console.print(
            f"  arc_ratio quartiles: "
            f"{ar_sorted[len(ar_sorted)//4]:.4f} / "
            f"{ar_sorted[len(ar_sorted)//2]:.4f} / "
            f"{ar_sorted[3*len(ar_sorted)//4]:.4f}"
        )

    # ---- Stat 3: set zones ----
    console.print()
    console.print("[bold]Stat 3 — set zones[/bold]")
    console.print(f"  total sets seen:       {set_total}")
    console.print(f"  sets with origin+dest: {set_covered}")
    if set_total:
        console.print(f"  coverage:              {100.0 * set_covered / set_total:.1f}%")
    if set_transitions:
        tbl_z = Table(title="top set transitions (origin → dest)")
        tbl_z.add_column("origin", justify="right")
        tbl_z.add_column("dest", justify="right")
        tbl_z.add_column("count", justify="right")
        top = sorted(set_transitions.items(), key=lambda kv: -kv[1])[:15]
        for (o, d), c in top:
            tbl_z.add_row(str(o), str(d), str(c))
        console.print(tbl_z)

    # ---- gate evaluation ----
    console.print()
    console.print("[bold]Gate evaluation[/bold]")
    gates: dict[str, bool] = {}
    gates["serve_speed coverage≥90%"] = bool(serve_pred and len(serve_mps) / max(1, serve_pred) >= 0.90)
    gates["serve_speed no OOB"] = serve_oob == 0
    if serve_mps and len(per_video_mps) >= 2 and statistics.stdev(serve_mps) > 0:
        within = [statistics.stdev(v) for v in per_video_mps.values() if len(v) >= 2]
        ratio = statistics.mean(within) / statistics.stdev(serve_mps) if within else 1.0
        gates["serve_speed within<0.70×cross"] = ratio < 0.70
    else:
        gates["serve_speed within<0.70×cross"] = False
    gates["serve_speed ≥2 types"] = sum(1 for k in serve_types if k in {"jump", "float", "standing"}) >= 2

    gates["attack_direction all 3 classes"] = all(
        atk_dir_counts.get(k, 0) > 0 for k in ("line", "cross", "cut")
    )
    gates["attack_direction coverage≥90%"] = (
        atk_dir_total > 0
        and sum(atk_dir_counts.get(k, 0) for k in ("line", "cross", "cut")) / atk_dir_total
        >= 0.90
    )

    gates["attack_type both classes"] = (
        atk_type_counts.get("power", 0) > 0 and atk_type_counts.get("shot", 0) > 0
    )
    power_frac = (
        atk_type_counts.get("power", 0)
        / max(1, atk_type_counts.get("power", 0) + atk_type_counts.get("shot", 0))
    )
    gates["attack_type power 60-90%"] = 0.60 <= power_frac <= 0.90
    gates["attack_type unknown ≤15%"] = (
        atk_type_counts.get("unknown", 0) / max(1, atk_type_total) <= 0.15
    )

    gates["set_zones coverage≥90%"] = (
        set_total > 0 and set_covered / set_total >= 0.90
    )

    for name, ok in gates.items():
        mark = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {mark} {name}")

    out = {
        "n_rallies": len(rallies),
        "rejections": n_rejected,
        "serve_speed": {
            "pred_serves": serve_pred,
            "mps_n": len(serve_mps),
            "out_of_bounds": serve_oob,
            "types": dict(serve_types),
            "min": min(serve_mps) if serve_mps else None,
            "median": statistics.median(serve_mps) if serve_mps else None,
            "max": max(serve_mps) if serve_mps else None,
            "mean": statistics.mean(serve_mps) if serve_mps else None,
            "stdev": statistics.stdev(serve_mps) if len(serve_mps) >= 2 else None,
        },
        "attack_direction": {
            "total": atk_dir_total,
            "counts": dict(atk_dir_counts),
        },
        "attack_type": {
            "total": atk_type_total,
            "counts": dict(atk_type_counts),
            "power_fraction_among_classified": power_frac,
        },
        "set_zones": {
            "total": set_total,
            "covered": set_covered,
            "top_transitions": {
                f"{o}->{d}": c
                for (o, d), c in sorted(set_transitions.items(), key=lambda kv: -kv[1])[:15]
            },
        },
        "gates": gates,
    }
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stats_pack_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    console.print(f"[green]wrote[/green] {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
