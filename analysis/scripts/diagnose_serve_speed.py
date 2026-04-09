"""Pre-work diagnostic for W3 serve speed + serve type.

Question: does computing serve speed as court-space Euclidean displacement
of the ball across 5 frames around the serve contact, via the existing
CourtCalibrator homography, yield physically plausible m/s values with
≥90% coverage across the 62-video action-GT eval set?

Produces a per-rally table of (rally_id, serve_frame, pixel_velocity,
court_mps, serve_type_guess) and aggregate coverage/bounds/spread stats.
No production mutations. Writes JSON to
``analysis/outputs/serve_speed_diagnostic.json``.

Usage
-----
    cd analysis
    uv run python scripts/diagnose_serve_speed.py
    uv run python scripts/diagnose_serve_speed.py --limit 50
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

from rallycut.tracking.action_classifier import (  # noqa: E402
    ActionType,
    classify_rally_actions,
)
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import detect_contacts  # noqa: E402
from rallycut.tracking.match_tracker import verify_team_assignments  # noqa: E402
from rallycut.tracking.sequence_action_runtime import (  # noqa: E402
    apply_sequence_override,
    get_sequence_probs,
)

console = Console()

# Sampling window around the serve contact frame (frames).
PRE = 2
POST = 2
# Physical bounds for a beach volleyball serve (meters per second).
MIN_MPS = 5.0
MAX_MPS = 35.0


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


def _project(calibrator: Any, x: float, y: float) -> tuple[float, float] | None:
    try:
        return calibrator.image_to_court((x, y), 1, 1)
    except Exception:  # noqa: BLE001
        return None


def _mps_between(
    calibrator: Any,
    a: BallPosition,
    b: BallPosition,
    fps: float,
) -> float | None:
    pa = _project(calibrator, a.x, a.y)
    pb = _project(calibrator, b.x, b.y)
    if pa is None or pb is None:
        return None
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    dist_m = math.hypot(dx, dy)
    dframes = abs(b.frame_number - a.frame_number)
    if dframes == 0:
        return None
    dt = dframes / fps
    return dist_m / dt


def _classify_type(
    post_dy_image: float, peak_vertical_rise_image: float, mps: float | None
) -> str:
    """Very rough initial heuristic — the diagnostic surfaces the
    distribution, not a shipping classifier. `dy` uses image-space
    normalized coords where +y is down.
    """
    if mps is None:
        return "unknown"
    if mps >= 18.0 and peak_vertical_rise_image > 0.05:
        return "jump"
    if mps >= 10.0 and peak_vertical_rise_image < 0.03:
        return "float"
    if mps < 10.0:
        return "standing"
    return "unknown"


def _run_rally_for_serve(
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

    n_with_calibration = 0
    n_pred_serve = 0
    n_coverage = 0  # got a non-null mps
    n_out_of_bounds = 0
    n_rejected = 0

    mps_values: list[float] = []
    per_video_mps: dict[str, list[float]] = {}
    type_counts: Counter[str] = Counter()
    per_rally_rows: list[dict] = []

    for idx, rally in enumerate(rallies, start=1):
        if not rally.ball_positions_json or not rally.positions_json or not rally.frame_count:
            n_rejected += 1
            continue
        calibrator = calibrators.get(rally.video_id)
        has_cal = calibrator is not None and getattr(calibrator, "is_calibrated", False)
        if has_cal:
            n_with_calibration += 1

        try:
            pred_actions, ball_positions = _run_rally_for_serve(
                rally, team_map.get(rally.rally_id), calibrator, ctx
            )
        except Exception as exc:  # noqa: BLE001
            n_rejected += 1
            console.print(f"  [red][{idx}/{len(rallies)}][/red] {rally.rally_id}: {type(exc).__name__}: {exc}")
            continue

        serve_action = next((a for a in pred_actions if a.get("action") == "serve"), None)
        if serve_action is None:
            per_rally_rows.append(
                {"rally_id": rally.rally_id, "serve_frame": None, "mps": None,
                 "type": "no_serve"}
            )
            continue
        n_pred_serve += 1

        if not has_cal:
            per_rally_rows.append(
                {"rally_id": rally.rally_id, "serve_frame": serve_action.get("frame"),
                 "mps": None, "type": "no_calibration"}
            )
            continue

        serve_frame = int(serve_action.get("frame", -1))
        pre_bp = _ball_at(ball_positions, serve_frame - PRE, radius=3)
        post_bp = _ball_at(ball_positions, serve_frame + POST, radius=3)
        if pre_bp is None or post_bp is None:
            per_rally_rows.append(
                {"rally_id": rally.rally_id, "serve_frame": serve_frame, "mps": None,
                 "type": "missing_ball"}
            )
            continue

        mps = _mps_between(calibrator, pre_bp, post_bp, rally.fps)
        if mps is None:
            per_rally_rows.append(
                {"rally_id": rally.rally_id, "serve_frame": serve_frame, "mps": None,
                 "type": "projection_fail"}
            )
            continue

        n_coverage += 1
        if mps < MIN_MPS or mps > MAX_MPS:
            n_out_of_bounds += 1

        # Image-space vertical rise for type heuristic: peak rise above
        # serve-frame ball_y in the next 6 frames. In normalized image coords
        # small y = up, so "rise" = max(serve_y - y) in that window.
        serve_bp = _ball_at(ball_positions, serve_frame, radius=2)
        peak_rise = 0.0
        post_dy = 0.0
        if serve_bp is not None:
            for bp in ball_positions:
                if 0 <= (bp.frame_number - serve_frame) <= 6:
                    rise = serve_bp.y - bp.y
                    if rise > peak_rise:
                        peak_rise = rise
                    if (bp.frame_number - serve_frame) == 3:
                        post_dy = bp.y - serve_bp.y

        stype = _classify_type(post_dy, peak_rise, mps)
        type_counts[stype] += 1

        mps_values.append(mps)
        per_video_mps.setdefault(rally.video_id, []).append(mps)
        per_rally_rows.append(
            {
                "rally_id": rally.rally_id,
                "video_id": rally.video_id,
                "serve_frame": serve_frame,
                "mps": round(mps, 2),
                "peak_rise": round(peak_rise, 4),
                "type": stype,
            }
        )

        if idx % 20 == 0 or idx == len(rallies):
            console.print(f"  [{idx}/{len(rallies)}] processed")

    # ---------------- Report ----------------
    console.print()
    console.print("[bold]Coverage[/bold]")
    console.print(f"  rallies loaded:             {len(rallies)}")
    console.print(f"  rallies with calibration:   {n_with_calibration}")
    console.print(f"  rallies with a pred serve:  {n_pred_serve}")
    console.print(f"  rallies with mps computed:  {n_coverage}")
    if n_pred_serve:
        cov = 100.0 * n_coverage / n_pred_serve
        console.print(f"  coverage vs pred_serve:     {cov:.1f}%")
    console.print(f"  out-of-bounds (<{MIN_MPS} or >{MAX_MPS}): {n_out_of_bounds}")
    console.print(f"  rejections: {n_rejected}")

    if mps_values:
        console.print()
        console.print("[bold]Speed distribution (m/s)[/bold]")
        mps_sorted = sorted(mps_values)
        console.print(f"  n       = {len(mps_values)}")
        console.print(f"  min     = {min(mps_values):.2f}")
        console.print(f"  p25     = {mps_sorted[len(mps_sorted)//4]:.2f}")
        console.print(f"  median  = {statistics.median(mps_values):.2f}")
        console.print(f"  p75     = {mps_sorted[3*len(mps_sorted)//4]:.2f}")
        console.print(f"  max     = {max(mps_values):.2f}")
        console.print(f"  mean    = {statistics.mean(mps_values):.2f}")
        if len(mps_values) >= 2:
            console.print(f"  stdev   = {statistics.stdev(mps_values):.2f}")

    # Per-video spread vs cross-video spread.
    if per_video_mps:
        cross_std = statistics.stdev(mps_values) if len(mps_values) >= 2 else 0.0
        within_stds = [
            statistics.stdev(v) for v in per_video_mps.values() if len(v) >= 2
        ]
        avg_within = statistics.mean(within_stds) if within_stds else 0.0
        ratio = (avg_within / cross_std) if cross_std else float("nan")
        console.print()
        console.print("[bold]Per-video consistency[/bold]")
        console.print(f"  cross-video stdev    : {cross_std:.2f}")
        console.print(
            f"  mean within-video std: {avg_within:.2f} "
            f"(across {len(within_stds)} videos with ≥2 serves)"
        )
        console.print(f"  ratio (within/cross) : {ratio:.2f}  (gate: < 0.70)")

    if type_counts:
        tbl = Table(title="serve_type distribution")
        tbl.add_column("type")
        tbl.add_column("count", justify="right")
        for k, v in type_counts.most_common():
            tbl.add_row(k, str(v))
        console.print(tbl)

    console.print()
    console.print("[bold]Gate[/bold]")
    cov_ok = n_pred_serve > 0 and (n_coverage / n_pred_serve) >= 0.90
    bounds_ok = n_out_of_bounds == 0
    ratio_ok = (
        not per_video_mps
        or cross_std == 0
        or (avg_within / cross_std) < 0.70
    )
    diverse = len(type_counts) >= 2
    for label, ok in [
        ("coverage ≥ 90%",          cov_ok),
        ("no out-of-bounds",        bounds_ok),
        ("per-video < 0.70 × cross", ratio_ok),
        ("≥2 serve types seen",     diverse),
    ]:
        mark = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {mark} {label}")

    out = {
        "n_rallies": len(rallies),
        "n_with_calibration": n_with_calibration,
        "n_pred_serve": n_pred_serve,
        "n_coverage": n_coverage,
        "n_out_of_bounds": n_out_of_bounds,
        "n_rejected": n_rejected,
        "mps_stats": {
            "n": len(mps_values),
            "min": min(mps_values) if mps_values else None,
            "median": statistics.median(mps_values) if mps_values else None,
            "max": max(mps_values) if mps_values else None,
            "mean": statistics.mean(mps_values) if mps_values else None,
            "stdev": statistics.stdev(mps_values) if len(mps_values) >= 2 else None,
        },
        "type_counts": dict(type_counts),
        "per_rally": per_rally_rows,
    }
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "serve_speed_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    console.print(f"[green]wrote[/green] {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
