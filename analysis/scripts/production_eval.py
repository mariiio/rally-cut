"""Canonical production-mirroring eval dashboard.

This script is the **one honest measurement instrument** for the RallyCut
`track-players --actions` action-classification path. It mirrors production
exactly by reusing the same shared helpers production imports — it does NOT
re-implement any pipeline stage. See
`memory/production_truth_2026_04.md` for the full production spec.

What it does
------------
For each of the ~339 GT rallies in the DB, it loads pre-computed tracked
ball/player JSON and runs stages 9–14 of `track_player.py` end-to-end:

    1. `verify_team_assignments`         (match_tracker.py)
    2. `get_sequence_probs`              (sequence_action_runtime.py)   [MS-TCN++]
    3. pose injection from pose_cache    (pre-stored — equivalent to the
                                          production two-pass enrichment's
                                          end state)
    4. `detect_contacts`                 (contact_detector.py)
    5. `classify_rally_actions`          (action_classifier.py)
    6. `apply_sequence_override`         (sequence_action_runtime.py)

Then it computes honest aggregate metrics with rerun variance bars and
reports any rallies production couldn't process in a `production_rejected`
list (we never silently skip).

Usage
-----
    cd analysis
    uv run python scripts/production_eval.py                     # full run, N=3 reruns
    uv run python scripts/production_eval.py --reruns 5
    uv run python scripts/production_eval.py --parity-check <rid>  # per-contact diff vs DB
    uv run python scripts/production_eval.py --ablate mstcn_override

Ablation registry
-----------------
See `ABLATIONS` below. To add a new ablation in <10 lines:

    1. Add a `PipelineContext` field (e.g. `skip_pose_enrich: bool = False`).
    2. Gate the relevant stage in `_run_rally()` on the field.
    3. Add `"pose_enrich": lambda ctx: setattr(ctx, "skip_pose_enrich", True)`
       to `ABLATIONS`.

Output JSON schema
------------------
Written to `analysis/outputs/production_eval/run_<YYYY-MM-DD-HHMMSS>.json`:

    {
      "run_id":             "2026-04-07-142310",
      "git_sha":            "abc1234",
      "n_reruns":           3,
      "ablations_active":   ["mstcn_override"],
      "weights": {
        "<name>": {"path": "...", "size_bytes": N, "sha256_prefix": "..."}
      },
      "n_rallies_loaded":   339,
      "n_rallies_evaluated": 331,
      "production_rejected": [
        {"rally_id": "...", "reason": "missing ball_positions_json"},
        {"rally_id": "...", "reason": "RuntimeError: ..."}
      ],
      "metrics": {
        "contact_f1":                  {"mean": 0.817, "std": 0.003, "values": [...]},
        "contact_recall":              {...},
        "contact_precision":           {...},
        "action_accuracy":             {...},
        "court_side_accuracy":         {...},
        "player_attribution_accuracy": {...},
        "serve_id_accuracy":           {...},
        "serve_attr_accuracy":         {...},
        "action_accuracy_per_class":   {"serve": {"mean": ..., "std": ...}, ...}
      },
      "variance_warning": null | "metric X std dev 1.3pp > 1pp — consider --reruns 5"
    }

Phase 3 (component contribution audit) consumes this schema mechanically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

# Ensure sibling scripts (eval_action_detection.py) are importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_action_detection import (  # type: ignore[import-not-found]  # noqa: E402
    MatchResult,
    RallyData,
    _build_player_positions,
    _load_match_team_assignments,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)

from rallycut.tracking.action_classifier import classify_rally_actions  # noqa: E402
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    detect_contacts,
)
from rallycut.tracking.match_tracker import verify_team_assignments  # noqa: E402
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking.sequence_action_runtime import (  # noqa: E402
    apply_sequence_override,
    get_sequence_probs,
)

console = Console()


# --------------------------------------------------------------------------- #
# Weights inventory — loud abort if MS-TCN++ weights missing (per spec)       #
# --------------------------------------------------------------------------- #

WEIGHTS_PATHS: dict[str, Path] = {
    "contact_classifier": Path("weights/contact_classifier/contact_classifier.pkl"),
    "pose_attribution":   Path("weights/pose_attribution/pose_attribution.joblib"),
    "action_classifier":  Path("weights/action_classifier/action_classifier.pkl"),
    "server_classifier":  Path("weights/server_classifier/server_classifier.pkl"),
    "sequence_action":    Path("weights/sequence_action/ms_tcn_production.pt"),
}


def _weights_inventory(analysis_root: Path) -> dict[str, dict[str, Any]]:
    """Stat every production weight file; fail loudly if MS-TCN++ is missing.

    Returns a dict keyed by weight-group name. Each entry has `path`,
    `size_bytes`, and `sha256_prefix` (first 16 hex chars — enough to prove
    which file was loaded, cheap to compute).
    """
    inventory: dict[str, dict[str, Any]] = {}
    for name, rel in WEIGHTS_PATHS.items():
        abs_path = (analysis_root / rel).resolve()
        if not abs_path.exists():
            if name == "sequence_action":
                raise RuntimeError(
                    f"MS-TCN++ weights missing at {abs_path}. "
                    "Production path loads these unconditionally and the spec "
                    "forbids silent fallback. Run "
                    "`uv run rallycut train pull-weights` or retrain before "
                    "running this eval."
                )
            console.print(
                f"[yellow]warning:[/yellow] weight file missing: {abs_path} "
                f"(production would auto-load this; downstream stage may degrade)"
            )
            inventory[name] = {"path": str(abs_path), "size_bytes": 0, "sha256_prefix": ""}
            continue
        size = abs_path.stat().st_size
        h = hashlib.sha256()
        with abs_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        inventory[name] = {
            "path": str(abs_path),
            "size_bytes": size,
            "sha256_prefix": h.hexdigest()[:16],
        }
    return inventory


# --------------------------------------------------------------------------- #
# Pipeline context + ablation registry                                        #
# --------------------------------------------------------------------------- #


@dataclass
class PipelineContext:
    """Runtime toggles consumed by `_run_rally`. Ablations mutate this."""

    skip_sequence_override: bool = False
    # Phase 3 will add: skip_pose_enrich, skip_verify_teams, skip_adaptive_dedup, ...


# Each ablation is a one-line mutator. Extension pattern is documented in the
# module docstring. Keep these under 10 lines apiece.
ABLATIONS: dict[str, Callable[[PipelineContext], None]] = {
    "mstcn_override": lambda ctx: setattr(ctx, "skip_sequence_override", True),
}


# --------------------------------------------------------------------------- #
# Court calibrators (match production enrichment)                             #
# --------------------------------------------------------------------------- #


def _build_calibrators(video_ids: set[str]) -> dict[str, Any]:
    """Build per-video `CourtCalibrator` from DB-stored court keypoints.

    Mirrors how `track_player.py` passes a real `CourtCalibrator` into
    `get_sequence_probs`, `detect_contacts`, and `classify_rally_actions`.
    Videos without calibration yield no entry (helpers treat `None` as
    "no homography enrichment", matching production's uncalibrated path).
    """
    from rallycut.court.calibration import CourtCalibrator  # noqa: PLC0415
    from rallycut.evaluation.tracking.db import load_court_calibration  # noqa: PLC0415

    out: dict[str, Any] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if not corners or len(corners) != 4:
            continue
        cal = CourtCalibrator()
        cal.calibrate([(c["x"], c["y"]) for c in corners])
        if cal.is_calibrated:
            out[vid] = cal
    return out


# --------------------------------------------------------------------------- #
# Per-rally production-mirrored pipeline                                      #
# --------------------------------------------------------------------------- #


@dataclass
class RallyOutcome:
    """Result of running the production path on one rally."""

    rally_id: str
    matches: list[MatchResult] = field(default_factory=list)
    unmatched_preds: list[dict] = field(default_factory=list)
    pred_actions: list[dict] = field(default_factory=list)  # kept for parity check


def _parse_positions(raw: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp.get("width", 0.05),
            height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
            keypoints=pp.get("keypoints"),
        )
        for pp in raw
    ]


def _parse_ball(raw: list[dict]) -> list[BallPosition]:
    return [
        BallPosition(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in raw
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]


def _run_rally(
    rally: RallyData,
    match_teams: dict[int, int] | None,
    calibrator: Any,
    ctx: PipelineContext,
) -> list[dict]:
    """Mirror `track_player.py:1011–1093` stages 9–14. Returns prediction dicts.

    Any exception is raised to the caller, which records it in
    `production_rejected`. The caller also handles the "missing inputs"
    rejection gate before we get here.
    """
    ball_positions = _parse_ball(rally.ball_positions_json or [])
    # `inject_pose=True` reads `pose_cache` for this rally — equivalent end
    # state to production's two-pass `enrich_positions_with_pose`, which
    # mutates keypoints into `player_positions`. The cache is what the
    # production pipeline wrote during the track that populated positions_json.
    player_positions: list[PlayerPosition] = _build_player_positions(
        rally.positions_json or [], rally_id=rally.rally_id, inject_pose=True,
    )

    # Stage 9 — verify team assignments (per-rally bulk flip).
    teams: dict[int, int] | None = dict(match_teams) if match_teams else None
    if teams is not None:
        teams = verify_team_assignments(teams, player_positions)

    # Stage 10 — MS-TCN++ per-frame action probabilities. Calibrator is
    # forwarded so trajectory features get the homography-enriched signal
    # that production uses when court calibration is available
    # (track_player.py passes the verified CourtCalibrator).
    sequence_probs: np.ndarray | None = get_sequence_probs(
        ball_positions=ball_positions,
        player_positions=player_positions,
        court_split_y=rally.court_split_y,
        frame_count=rally.frame_count,
        team_assignments=teams,
        calibrator=calibrator,
    )

    # Stages 11+12 — pose-informed contact detection. Pose keypoints are
    # already injected via pose_cache above; production's default
    # ContactDetectionConfig is used implicitly (no config= kwarg) so the
    # eval picks up any new defaults automatically — mirrors
    # track_player.py:978 which also omits `config=`.
    contact_sequence = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        frame_count=rally.frame_count or None,
        team_assignments=teams,
        court_calibrator=calibrator,
        sequence_probs=sequence_probs,
    )

    # Stage 13 — action classification (state machine + GBM + repair + etc.).
    rally_actions = classify_rally_actions(
        contact_sequence,
        rally.rally_id,
        team_assignments=teams,
        match_team_assignments=teams,
        calibrator=calibrator,
    )

    # Stage 14 — MS-TCN++ hybrid override (serves exempt).
    if sequence_probs is not None and not ctx.skip_sequence_override:
        apply_sequence_override(rally_actions, sequence_probs)

    return [a.to_dict() for a in rally_actions.actions]


# --------------------------------------------------------------------------- #
# Eval loop                                                                   #
# --------------------------------------------------------------------------- #


def _tolerance_frames(fps: float, tolerance_ms: int = 167) -> int:
    return max(1, round(fps * tolerance_ms / 1000))


def _run_once(
    rallies: list[RallyData],
    team_map: dict[str, dict[int, int]],
    calibrators: dict[str, Any],
    ctx: PipelineContext,
    *,
    print_progress: bool = True,
) -> tuple[list[MatchResult], list[dict], list[dict[str, str]]]:
    """Run the full 339-rally loop once. Returns (matches, unmatched_preds, rejections)."""
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []
    rejections: list[dict[str, str]] = []

    for idx, rally in enumerate(rallies, start=1):
        # Rejection gate — surface rather than silently skip.
        if not rally.ball_positions_json:
            rejections.append({"rally_id": rally.rally_id, "reason": "missing ball_positions_json"})
            continue
        if not rally.positions_json:
            rejections.append({"rally_id": rally.rally_id, "reason": "missing positions_json"})
            continue
        if not rally.frame_count or rally.frame_count < 10:
            rejections.append({
                "rally_id": rally.rally_id,
                "reason": f"frame_count={rally.frame_count}",
            })
            continue

        try:
            pred_actions = _run_rally(
                rally,
                team_map.get(rally.rally_id),
                calibrators.get(rally.video_id),
                ctx,
            )
        except Exception as exc:  # noqa: BLE001 — we want to surface any prod failure
            rejections.append({
                "rally_id": rally.rally_id,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            if print_progress:
                console.print(
                    f"  [red][{idx}/{len(rallies)}][/red] {rally.rally_id}: "
                    f"REJECTED — {type(exc).__name__}: {exc}"
                )
            continue

        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        avail_tids = {pp["trackId"] for pp in rally.positions_json}
        match_teams = team_map.get(rally.rally_id)

        matches, unmatched = match_contacts(
            rally.gt_labels,
            real_pred,
            tolerance=_tolerance_frames(rally.fps),
            available_track_ids=avail_tids,
            team_assignments=match_teams,
        )
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

        if print_progress and (idx % 20 == 0 or idx == len(rallies)):
            console.print(f"  [{idx}/{len(rallies)}] processed")

    return all_matches, all_unmatched, rejections


def _serve_metrics(matches: list[MatchResult]) -> tuple[float, int, float, int]:
    """Compute (serve_id_accuracy, serve_id_n, serve_attr_accuracy, serve_attr_n).

    - serve_id:   of all GT serves, what fraction have a predicted contact
                  classified as "serve" at that frame?
    - serve_attr: of all matched GT serves that have an evaluable track_id,
                  what fraction got the server's player_track_id right?
    """
    gt_serves = [m for m in matches if m.gt_action == "serve"]
    id_n = len(gt_serves)
    id_correct = sum(1 for m in gt_serves if m.pred_action == "serve")

    attr_pool = [m for m in gt_serves if m.pred_frame is not None and m.player_evaluable]
    attr_n = len(attr_pool)
    attr_correct = sum(1 for m in attr_pool if m.player_correct)

    id_acc = id_correct / max(1, id_n)
    attr_acc = attr_correct / max(1, attr_n)
    return id_acc, id_n, attr_acc, attr_n


def _flatten_run(matches: list[MatchResult], unmatched: list[dict]) -> dict[str, float]:
    """Collapse a single run into scalar metrics consumed by the aggregator."""
    m = compute_metrics(matches, unmatched)
    serve_id, _, serve_attr, _ = _serve_metrics(matches)
    out: dict[str, float] = {
        "contact_f1":                  float(m["f1"]),
        "contact_recall":              float(m["recall"]),
        "contact_precision":           float(m["precision"]),
        "action_accuracy":             float(m["action_accuracy"]),
        "court_side_accuracy":         float(m["court_side_accuracy"]),
        "player_attribution_accuracy": float(m["player_evaluable_accuracy"]),
        "serve_id_accuracy":           float(serve_id),
        "serve_attr_accuracy":         float(serve_attr),
    }
    # Per-class F1 goes in with a prefix so the aggregator treats each as
    # its own metric for variance purposes.
    for cls, stats in m["per_class"].items():
        out[f"per_class::{cls}::f1"] = float(stats["f1"])
    return out


def _aggregate_runs(run_scalars: list[dict[str, float]]) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Compute mean/std/values and a variance warning if any std dev > 1pp."""
    if not run_scalars:
        return {}, None
    keys = run_scalars[0].keys()
    agg: dict[str, dict[str, Any]] = {}
    warn: str | None = None
    for k in keys:
        vals = [r[k] for r in run_scalars]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=0))
        agg[k] = {"mean": mean, "std": std, "values": vals}
        if std > 0.01 and warn is None:
            warn = (
                f"metric {k!r} std dev {std * 100:.2f}pp > 1pp — "
                "consider increasing --reruns"
            )
    return agg, warn


# --------------------------------------------------------------------------- #
# Parity check                                                                #
# --------------------------------------------------------------------------- #


def _parity_check(rally_id: str) -> int:
    """Run the dashboard on one rally and diff per-contact against DB-stored production output.

    Comparison is against `rally.contacts_json` / `rally.actions_json` which
    were written by a prior `track-players --actions` run. Recency caveat: if
    those DB rows are stale, divergence may reflect pipeline progress rather
    than a dashboard bug.
    """
    rallies = load_rallies_with_action_gt(rally_id=rally_id)
    if not rallies:
        console.print(f"[red]rally {rally_id} not found or has no action GT[/red]")
        return 1
    rally = rallies[0]

    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    if rally.positions_json:
        rally_pos_lookup[rally.rally_id] = _parse_positions(rally.positions_json)
    team_map = _load_match_team_assignments({rally.video_id}, rally_positions=rally_pos_lookup)

    calibrators = _build_calibrators({rally.video_id})
    ctx = PipelineContext()
    pred_actions = _run_rally(
        rally,
        team_map.get(rally.rally_id),
        calibrators.get(rally.video_id),
        ctx,
    )

    # DB-stored production output (from the last track-players run that
    # populated this row). May be stale.
    stored_actions: list[dict] = []
    if rally.actions_json and isinstance(rally.actions_json, dict):
        stored_actions = list(rally.actions_json.get("actions", []) or [])
    if not stored_actions:
        console.print(
            "[yellow]no actions_json stored for this rally — cannot diff. "
            "Pick a rally whose production run populated actions_json.[/yellow]"
        )
        return 2

    # Index stored by frame for a cheap nearest-match per GT contact.
    stored_by_frame = {int(a["frame"]): a for a in stored_actions if "frame" in a}
    diffs = 0
    total = 0
    table = Table(title=f"parity diff for {rally_id}", show_lines=False)
    table.add_column("gt_frame", justify="right")
    table.add_column("gt_action")
    table.add_column("dash (action, tid, side)")
    table.add_column("stored (action, tid, side)")
    table.add_column("match?")

    pred_by_frame: dict[int, dict] = {}
    for a in pred_actions:
        if a.get("isSynthetic"):
            continue
        pred_by_frame[int(a["frame"])] = a

    for gt in rally.gt_labels:
        total += 1
        # Nearest-frame within ±5 for both.
        dash = _nearest(pred_by_frame, gt.frame, 5)
        stored = _nearest(stored_by_frame, gt.frame, 5)
        if dash is None or stored is None:
            table.add_row(
                str(gt.frame), gt.action,
                _fmt(dash), _fmt(stored),
                "[yellow]missing[/yellow]",
            )
            diffs += 1
            continue
        same = (
            dash.get("action") == stored.get("action")
            and dash.get("playerTrackId") == stored.get("playerTrackId")
            and dash.get("courtSide") == stored.get("courtSide")
        )
        table.add_row(
            str(gt.frame), gt.action,
            _fmt(dash), _fmt(stored),
            "[green]ok[/green]" if same else "[red]DIFF[/red]",
        )
        if not same:
            diffs += 1

    console.print(table)
    console.print(
        f"parity: {total - diffs}/{total} GT contacts match DB-stored production output"
    )
    if diffs:
        console.print(
            "[yellow]Diffs present. May be (a) dashboard bug — fix call order/args, "
            "(b) stale DB row — pick a more recent rally, or (c) non-determinism "
            "in MS-TCN / contact GBM. If (a), stop and investigate before running "
            "full eval.[/yellow]"
        )
    return 0 if diffs == 0 else 3


def _nearest(by_frame: dict[int, dict], target: int, tol: int) -> dict | None:
    best: dict | None = None
    best_d = tol + 1
    for f, a in by_frame.items():
        d = abs(f - target)
        if d < best_d:
            best = a
            best_d = d
    return best


def _fmt(a: dict | None) -> str:
    if a is None:
        return "—"
    return f"{a.get('action')}, tid={a.get('playerTrackId')}, {a.get('courtSide')}"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _print_summary(
    agg: dict[str, dict[str, Any]],
    n_loaded: int,
    n_eval: int,
    rejections: list[dict[str, str]],
    warning: str | None,
) -> None:
    headline_keys = [
        "contact_f1", "contact_recall", "contact_precision",
        "action_accuracy", "court_side_accuracy",
        "player_attribution_accuracy",
        "serve_id_accuracy", "serve_attr_accuracy",
    ]
    table = Table(title="production_eval — aggregate (mean ± std across reruns)")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("± std", justify="right")
    for k in headline_keys:
        if k not in agg:
            continue
        m = agg[k]
        table.add_row(k, f"{m['mean']:.1%}", f"{m['std'] * 100:.2f}pp")
    console.print(table)

    per_class = {k: v for k, v in agg.items() if k.startswith("per_class::")}
    if per_class:
        pc = Table(title="per-class F1 (mean ± std)")
        pc.add_column("class")
        pc.add_column("f1", justify="right")
        pc.add_column("± std", justify="right")
        for k, v in per_class.items():
            cls = k.split("::")[1]
            pc.add_row(cls, f"{v['mean']:.1%}", f"{v['std'] * 100:.2f}pp")
        console.print(pc)

    console.print(
        f"[bold]rallies:[/bold] loaded={n_loaded}  evaluated={n_eval}  "
        f"rejected={len(rejections)}"
    )
    if rejections:
        console.print("[yellow]production_rejected (first 10):[/yellow]")
        for r in rejections[:10]:
            console.print(f"  - {r['rally_id']}: {r['reason']}")
    if warning:
        console.print(f"[yellow]variance warning:[/yellow] {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--reruns", type=int, default=3,
                        help="Number of full-loop reruns for variance (default: 3)")
    parser.add_argument("--ablate", action="append", default=[],
                        choices=sorted(ABLATIONS.keys()),
                        help="Disable a production component (repeatable).")
    parser.add_argument("--parity-check", type=str, default=None, metavar="RALLY_ID",
                        help="Diff one rally's dashboard output vs DB-stored production output and exit.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rallies (debug).")
    args = parser.parse_args()

    analysis_root = Path(__file__).resolve().parent.parent
    weights = _weights_inventory(analysis_root)

    if args.parity_check:
        return _parity_check(args.parity_check)

    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    if args.limit:
        rallies = rallies[: args.limit]
    n_loaded = len(rallies)
    console.print(f"  {n_loaded} rallies")

    # Build verified match team assignments exactly as production does
    # (pass `rally_positions=...` so `verify_team_assignments` runs).
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = _parse_positions(r.positions_json)
    video_ids = {r.video_id for r in rallies}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)

    # Production passes a real `CourtCalibrator` into the three enrichment
    # stages when calibration is available. Build them once up front.
    calibrators = _build_calibrators(video_ids)
    console.print(f"  court calibration available for {len(calibrators)}/{len(video_ids)} videos")

    # Build PipelineContext + apply ablations.
    ctx = PipelineContext()
    for name in args.ablate:
        ABLATIONS[name](ctx)
    if args.ablate:
        console.print(f"[yellow]ablations active:[/yellow] {', '.join(args.ablate)}")

    # Run N times.
    run_scalars: list[dict[str, float]] = []
    last_rejections: list[dict[str, str]] = []
    n_eval = 0
    for run_i in range(args.reruns):
        console.print(f"[bold]Run {run_i + 1}/{args.reruns}[/bold]")
        t0 = time.time()
        matches, unmatched, rejections = _run_once(rallies, team_map, calibrators, ctx)
        dt = time.time() - t0
        console.print(f"  run {run_i + 1} done in {dt:.1f}s  "
                      f"(matches={len(matches)}, rejections={len(rejections)})")
        run_scalars.append(_flatten_run(matches, unmatched))
        last_rejections = rejections
        n_eval = n_loaded - len(rejections)

    agg, warning = _aggregate_runs(run_scalars)
    _print_summary(agg, n_loaded, n_eval, last_rejections, warning)

    # Write JSON.
    out_dir = analysis_root / "outputs" / "production_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_path = out_dir / f"run_{stamp}.json"

    # Regroup per_class:: keys under a nested dict for a cleaner schema.
    metrics_out: dict[str, Any] = {}
    per_class_out: dict[str, dict[str, Any]] = {}
    for k, v in agg.items():
        if k.startswith("per_class::"):
            _, cls, stat = k.split("::")
            per_class_out.setdefault(cls, {})[stat] = v
        else:
            metrics_out[k] = v
    if per_class_out:
        metrics_out["action_accuracy_per_class"] = per_class_out

    payload: dict[str, Any] = {
        "run_id": stamp,
        "git_sha": _git_sha(),
        "n_reruns": args.reruns,
        "ablations_active": list(args.ablate),
        "weights": weights,
        "n_rallies_loaded": n_loaded,
        "n_rallies_evaluated": n_eval,
        "production_rejected": last_rejections,
        "metrics": metrics_out,
        "variance_warning": warning,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]wrote[/green] {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
