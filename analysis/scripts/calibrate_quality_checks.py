"""Calibrate quality-check thresholds against the 63-video GT.

For each quality check + metric-key + threshold-sweep, measures:

    lift = P(pipeline_failure | check fires) / P(pipeline_failure | !fires)

A check is considered useful if its best lift >= 3.0 AND n_fires >= 5.

Pipeline failure is defined per video as ANY of:
    - action_acc  < 0.75  (from production eval pipeline on action-GT rallies)
    - score_acc   < 0.75  (from the same run)
    - HOTA        < 0.60  (from tracking eval on player-tracking-GT rallies)

If a metric is not available for a video (e.g., no action GT), that metric is
excluded from the failure decision for that video but the other metrics still
apply. If NO metric is available for a video (rare), it is skipped entirely.

Usage
-----
    cd analysis
    uv run python scripts/calibrate_quality_checks.py --limit 2   # smoke test
    uv run python scripts/calibrate_quality_checks.py             # full run

Report written to analysis/reports/quality_calibration_<YYYY-MM-DD>.json
and also printed to stdout.

Implementation notes
--------------------
- Quality checks are run by calling _load_video_inputs (YOLO + court detection)
  which requires the video file; we download via get_video_path from S3/MinIO.
- Pipeline metrics are computed by re-running the production-mirrored stages
  (verify_team_assignments, detect_contacts, classify_rally_actions, etc.)
  via helpers imported from eval_action_detection.py and production_eval.py.
  This is the same code path as production_eval.py --limit, just partitioned
  per video.
- HOTA is computed using load_labeled_rallies + evaluate_rally, averaged over
  the video's tracking-GT rallies. Videos without tracking GT get HOTA=None.
- The full run over 63 videos is ~30-60 min. Use --limit 2 for smoke test.

Fallback
--------
If weights are missing (contact classifier, MS-TCN++, etc.), action_acc and
score_acc will be None for all videos and the harness will report "no data
for action_acc / score_acc — only HOTA sweep available". This is expected on
a fresh checkout without weight files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

# Ensure the analysis/ scripts dir is importable (for eval_action_detection etc.)
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# Threshold sweeps — drop CLIP row (no open-clip in runtime image for A1)
# ──────────────────────────────────────────────────────────────────────────────

# Post-2026-04-15: camera_too_far, crowded_scene, shaky_camera, video_rotated,
# too_dark, and overexposed were dropped after the calibration in
# analysis/reports/quality_calibration_2026-04-14.json recommended "drop" for
# every one (zero or anti-predictive lift) AND a validation sweep against
# ~/Desktop/rallies/Negative/* + two positives confirmed camera_too_far as a
# false positive on normal footage. Only wrong_angle_or_not_volleyball retains
# empirical support. The historical sweep report is kept for audit.
SWEEPS: list[tuple[str, str, list[float], str]] = [
    ("wrong_angle_or_not_volleyball", "courtConfidence",   [0.3,  0.5,  0.6,  0.75      ], "<"),
]

# Failure thresholds for pipeline metrics
ACTION_ACC_FAIL  = 0.75
SCORE_ACC_FAIL   = 0.75
HOTA_FAIL        = 0.60

# Minimum number of check firings needed for a lift estimate to be meaningful
MIN_FIRES_FOR_LIFT = 5

# Minimum lift to recommend "ship" over "drop"
MIN_LIFT_TO_SHIP = 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Per-video result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoResult:
    video_id: str
    video_path: str | None = None
    # Quality check raw metrics (key -> value)
    quality_metrics: dict[str, float] = field(default_factory=dict)
    # Pipeline metrics (None = not available for this video)
    action_acc: float | None = None
    score_acc: float | None = None
    hota: float | None = None
    # Failure flag (derived; True if any available metric is below threshold)
    is_failure: bool = False
    # Error info
    quality_error: str | None = None
    pipeline_error: str | None = None
    hota_error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — enumerate GT videos
# ──────────────────────────────────────────────────────────────────────────────

def load_gt_video_ids() -> list[str]:
    """Return unique video_ids that have at least one action-GT rally."""
    from eval_action_detection import load_rallies_with_action_gt

    rallies = load_rallies_with_action_gt()
    seen: list[str] = []
    for r in rallies:
        if r.video_id not in seen:
            seen.append(r.video_id)
    return seen


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — quality checks for one video
# ──────────────────────────────────────────────────────────────────────────────

def run_quality_checks(video_path: str) -> dict[str, float]:
    """Run all preflight checks against a local video file.

    Returns a flat dict of metric_key -> value. Only keys that the checks
    actually emit are present (e.g., tiltDeg is absent when court confidence
    is too low to compute tilt).
    """
    from rallycut.quality.camera_geometry import check_camera_geometry
    from rallycut.quality.metadata import check_metadata
    from rallycut.quality.runner import _load_video_inputs

    meta, corners = _load_video_inputs(video_path, sample_seconds=60)

    all_metrics: dict[str, float] = {}
    for check_result in [
        check_metadata(meta),
        check_camera_geometry(corners),
    ]:
        all_metrics.update(check_result.metrics)

    return all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — pipeline metrics (action_acc, score_acc) for one video
# ──────────────────────────────────────────────────────────────────────────────

def compute_pipeline_metrics_for_video(
    video_id: str,
    rallies_for_video: list[Any],
    team_map: dict[str, dict[int, int]],
    calibrators: dict[str, Any],
    t2p_by_rally: dict[str, dict[int, int]],
    formation_flip_by_rally: dict[str, bool],
    camera_heights: dict[str, float],
    team_templates_by_video: dict[str, tuple[Any, Any]],
) -> tuple[float | None, float | None]:
    """Return (action_acc, score_acc) for a single video's rallies.

    Returns (None, None) if the video has no GT rallies or pipeline fails.
    """
    # Import pipeline helpers from production_eval (they're all pure functions
    # once weights are loaded; no heavyweight initialisation at import time).
    from production_eval import (
        PipelineContext,
        _run_rally,
        _load_match_team_assignments,
        _parse_positions,
    )
    from eval_action_detection import compute_metrics, match_contacts, RallyData
    from rallycut.evaluation.score_ground_truth import compute_score_metrics

    if not rallies_for_video:
        return None, None

    ctx = PipelineContext()
    all_matches = []
    all_unmatched: list[dict] = []
    pred_by_video: dict[str, list[tuple[int, Any]]] = {}
    gt_lookup: dict[str, tuple[str | None, str | None]] = {}

    def _tolerance_frames(fps: float) -> int:
        return max(1, int(round(fps * 0.5)))

    for rally in rallies_for_video:
        if not rally.ball_positions_json or not rally.positions_json:
            continue
        if not rally.frame_count or rally.frame_count < 10:
            continue
        try:
            rally_t2p = t2p_by_rally.get(rally.rally_id) or None
            semantic_flip = formation_flip_by_rally.get(rally.rally_id, False)
            cam_h = camera_heights.get(video_id, 0.0)
            pred_actions, rally_actions_obj = _run_rally(
                rally,
                team_map.get(rally.rally_id),
                calibrators.get(video_id),
                ctx,
                track_to_player=rally_t2p,
                formation_semantic_flip=semantic_flip,
                camera_height=cam_h,
            )
        except Exception:
            continue

        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        raw_avail_tids = {pp["trackId"] for pp in rally.positions_json}
        if rally_t2p:
            avail_tids = {rally_t2p.get(tid, tid) for tid in raw_avail_tids}
        else:
            avail_tids = raw_avail_tids

        matches, unmatched = match_contacts(
            rally.gt_labels,
            real_pred,
            tolerance=_tolerance_frames(rally.fps),
            available_track_ids=avail_tids,
            team_assignments=team_map.get(rally.rally_id),
            track_id_map=rally_t2p,
        )
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

        pred_by_video.setdefault(video_id, []).append(
            (rally.start_ms, rally_actions_obj)
        )
        gt_lookup[rally.rally_id] = (rally.gt_serving_team, rally.gt_point_winner)

    if not all_matches and not all_unmatched:
        return None, None

    m = compute_metrics(all_matches, all_unmatched)
    action_acc: float | None = float(m["action_accuracy"]) if "action_accuracy" in m else None

    score_m = compute_score_metrics(pred_by_video, gt_lookup)
    score_acc: float | None = (
        float(score_m.score_accuracy) if score_m.n_rallies_scored > 0 else None
    )

    return action_acc, score_acc


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — HOTA for one video (tracking GT, separate from action GT)
# ──────────────────────────────────────────────────────────────────────────────

def compute_hota_for_video(video_id: str) -> float | None:
    """Return mean HOTA over the video's tracking-GT rallies, or None."""
    from rallycut.evaluation.tracking.db import load_labeled_rallies
    from rallycut.evaluation.tracking.metrics import evaluate_rally

    rallies = load_labeled_rallies(video_id=video_id)
    if not rallies:
        return None

    hotas: list[float] = []
    for rally in rallies:
        if rally.predictions is None:
            continue
        try:
            result = evaluate_rally(
                rally.rally_id,
                rally.ground_truth,
                rally.predictions,
                video_width=rally.video_width,
                video_height=rally.video_height,
            )
            if result.hota_metrics is not None:
                hotas.append(float(result.hota_metrics.hota))
        except Exception:
            continue

    return float(sum(hotas) / len(hotas)) if hotas else None


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — sweep calibration
# ──────────────────────────────────────────────────────────────────────────────

def _check_fires(metric_val: float, threshold: float, direction: str) -> bool:
    if direction == "<":
        return metric_val < threshold
    return metric_val > threshold


def calibrate_sweeps(
    results: list[VideoResult],
) -> list[dict[str, Any]]:
    """For each (check_id, metric_key, threshold, direction) compute lift table."""
    calibration_rows: list[dict[str, Any]] = []

    for check_id, metric_key, thresholds, direction in SWEEPS:
        threshold_rows: list[dict[str, Any]] = []
        best_lift = 0.0
        best_threshold: float | None = None
        best_n_fires = 0

        for thresh in thresholds:
            fires: list[bool] = []       # is_failure for videos where metric is available and check fires
            not_fires: list[bool] = []   # is_failure for videos where metric is available and check doesn't fire

            for vr in results:
                if metric_key not in vr.quality_metrics:
                    continue  # metric not available for this video, skip
                val = vr.quality_metrics[metric_key]
                fired = _check_fires(val, thresh, direction)
                if fired:
                    fires.append(vr.is_failure)
                else:
                    not_fires.append(vr.is_failure)

            n_fires = len(fires)
            n_not_fires = len(not_fires)

            p_fail_fires = sum(fires) / n_fires if n_fires > 0 else 0.0
            p_fail_not_fires = sum(not_fires) / n_not_fires if n_not_fires > 0 else 0.0
            lift = p_fail_fires / max(p_fail_not_fires, 1e-6)

            row = {
                "threshold": thresh,
                "n_fires": n_fires,
                "n_not_fires": n_not_fires,
                "p_fail_given_fires": round(p_fail_fires, 4),
                "p_fail_given_not_fires": round(p_fail_not_fires, 4),
                "lift": round(lift, 3),
            }
            threshold_rows.append(row)

            if n_fires >= MIN_FIRES_FOR_LIFT and lift > best_lift:
                best_lift = lift
                best_threshold = thresh
                best_n_fires = n_fires

        recommendation = "ship" if best_lift >= MIN_LIFT_TO_SHIP else "drop"
        calibration_rows.append({
            "check_id":          check_id,
            "metric_key":        metric_key,
            "direction":         direction,
            "best_threshold":    best_threshold,
            "best_lift":         round(best_lift, 3),
            "best_n_fires":      best_n_fires,
            "recommendation":    recommendation,
            "threshold_sweep":   threshold_rows,
        })

    return calibration_rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N videos (smoke test). Default: all.",
    )
    args = parser.parse_args()

    print("[calibrate_quality_checks] Loading GT video IDs...")
    video_ids = load_gt_video_ids()
    if args.limit:
        video_ids = video_ids[: args.limit]
    n_videos = len(video_ids)
    print(f"  {n_videos} videos to process (limit={args.limit})")

    # ── Pre-load shared data (team maps, calibrators, etc.) ───────────────────
    print("[calibrate_quality_checks] Loading shared GT data (team maps, calibrators)...")
    try:
        from eval_action_detection import (
            load_rallies_with_action_gt,
            _load_match_team_assignments,
            _load_track_to_player_maps,
        )
        from production_eval import (
            _build_calibrators,
            _build_camera_heights,
            _load_formation_semantic_flips_from_gt,
            _load_team_templates_by_video,
            _parse_positions,
        )
        from rallycut.tracking.player_tracker import PlayerPosition

        all_rallies = load_rallies_with_action_gt()
        # Filter to our video subset
        video_id_set = set(video_ids)
        all_rallies = [r for r in all_rallies if r.video_id in video_id_set]

        # Group rallies by video for fast per-video lookup
        rallies_by_video: dict[str, list[Any]] = {}
        for r in all_rallies:
            rallies_by_video.setdefault(r.video_id, []).append(r)

        # Build position lookup for verify_team_assignments
        rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
        for r in all_rallies:
            if r.positions_json:
                rally_pos_lookup[r.rally_id] = _parse_positions(r.positions_json)

        team_map = _load_match_team_assignments(video_id_set, rally_positions=rally_pos_lookup)
        t2p_by_rally = _load_track_to_player_maps(video_id_set)
        formation_flip_by_rally = _load_formation_semantic_flips_from_gt(video_id_set)
        team_templates_by_video = _load_team_templates_by_video(video_id_set)
        calibrators = _build_calibrators(video_id_set)
        camera_heights = _build_camera_heights(video_id_set, calibrators)
        pipeline_available = True
        print(f"  {len(all_rallies)} action-GT rallies across {len(rallies_by_video)} videos")
    except Exception as exc:
        print(f"  WARNING: Could not load pipeline helpers: {exc}")
        print("  action_acc / score_acc will be None for all videos.")
        pipeline_available = False
        rallies_by_video = {}
        team_map = {}
        t2p_by_rally = {}
        formation_flip_by_rally = {}
        team_templates_by_video = {}
        calibrators = {}
        camera_heights = {}

    # ── Sanity check: validate first video end-to-end ─────────────────────────
    if n_videos > 0:
        first_vid = video_ids[0]
        print(f"\n[calibrate_quality_checks] Sanity-checking first video: {first_vid[:8]}...")
        try:
            from rallycut.evaluation.tracking.db import get_video_path
            vpath = get_video_path(first_vid)
            if vpath is None:
                print(f"  WARNING: video file not locally resolvable for {first_vid[:8]} — will skip quality checks for unreachable videos")
            else:
                print(f"  Video path: {vpath}")
                _ = run_quality_checks(str(vpath))
                print("  Quality checks: OK")
        except Exception as exc:
            print(f"  WARNING: Sanity check failed: {exc}")
            print("  Continuing — errors will be recorded per-video.")

    # ── Per-video loop ─────────────────────────────────────────────────────────
    print(f"\n[calibrate_quality_checks] Processing {n_videos} videos...\n")
    results: list[VideoResult] = []

    for idx, video_id in enumerate(video_ids, start=1):
        t_start = time.monotonic()
        vr = VideoResult(video_id=video_id)

        # ── Quality checks ────────────────────────────────────────────────
        try:
            from rallycut.evaluation.tracking.db import get_video_path
            vpath = get_video_path(video_id)
            if vpath is not None:
                vr.video_path = str(vpath)
                vr.quality_metrics = run_quality_checks(str(vpath))
            else:
                vr.quality_error = "video file not locally resolvable"
        except Exception as exc:
            vr.quality_error = f"{type(exc).__name__}: {exc}"

        # ── Pipeline metrics (action_acc, score_acc) ──────────────────────
        if pipeline_available and video_id in rallies_by_video:
            try:
                action_acc, score_acc = compute_pipeline_metrics_for_video(
                    video_id,
                    rallies_by_video.get(video_id, []),
                    team_map,
                    calibrators,
                    t2p_by_rally,
                    formation_flip_by_rally,
                    camera_heights,
                    team_templates_by_video,
                )
                vr.action_acc = action_acc
                vr.score_acc = score_acc
            except Exception as exc:
                vr.pipeline_error = f"{type(exc).__name__}: {exc}"
                if args.limit:  # print full traceback in smoke-test mode
                    traceback.print_exc()

        # ── HOTA ─────────────────────────────────────────────────────────
        try:
            vr.hota = compute_hota_for_video(video_id)
        except Exception as exc:
            vr.hota_error = f"{type(exc).__name__}: {exc}"

        # ── Failure flag ──────────────────────────────────────────────────
        fail_signals: list[bool] = []
        if vr.action_acc is not None:
            fail_signals.append(vr.action_acc < ACTION_ACC_FAIL)
        if vr.score_acc is not None:
            fail_signals.append(vr.score_acc < SCORE_ACC_FAIL)
        if vr.hota is not None:
            fail_signals.append(vr.hota < HOTA_FAIL)

        if fail_signals:
            vr.is_failure = any(fail_signals)

        results.append(vr)
        elapsed = time.monotonic() - t_start

        # Progress line
        aa_str = f"action_acc={vr.action_acc:.2f}" if vr.action_acc is not None else "action_acc=N/A"
        sc_str = f"score_acc={vr.score_acc:.2f}" if vr.score_acc is not None else "score_acc=N/A"
        ht_str = f"HOTA={vr.hota:.2f}" if vr.hota is not None else "HOTA=N/A"
        err = ""
        if vr.quality_error:
            err += f" [qerr:{vr.quality_error[:40]}]"
        if vr.pipeline_error:
            err += f" [perr:{vr.pipeline_error[:40]}]"
        if vr.hota_error:
            err += f" [herr:{vr.hota_error[:40]}]"
        print(
            f"[{idx}/{n_videos}] {video_id[:8]}: {aa_str} {sc_str} {ht_str} "
            f"fail={vr.is_failure} ({elapsed:.1f}s){err}"
        )

    # ── Calibrate sweeps ───────────────────────────────────────────────────────
    print("\n[calibrate_quality_checks] Computing lift sweeps...")
    calibration_rows = calibrate_sweeps(results)

    # ── Print results table ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("QUALITY CHECK CALIBRATION RESULTS")
    print("=" * 80)
    print(f"{'check_id':<40} {'metric':<22} {'best_thresh':>12} {'best_lift':>10} {'n_fires':>8} {'rec':>6}")
    print("-" * 80)
    for row in calibration_rows:
        print(
            f"{row['check_id']:<40} {row['metric_key']:<22} "
            f"{str(row['best_threshold']):>12} {row['best_lift']:>10.3f} "
            f"{row['best_n_fires']:>8} {row['recommendation']:>6}"
        )
    print("=" * 80)

    # Coverage summary
    n_with_qm = sum(1 for vr in results if vr.quality_metrics)
    n_with_aa = sum(1 for vr in results if vr.action_acc is not None)
    n_with_sc = sum(1 for vr in results if vr.score_acc is not None)
    n_with_ht = sum(1 for vr in results if vr.hota is not None)
    n_failures = sum(1 for vr in results if vr.is_failure)
    print(f"\nCoverage: quality_metrics={n_with_qm}/{n_videos}, "
          f"action_acc={n_with_aa}/{n_videos}, "
          f"score_acc={n_with_sc}/{n_videos}, "
          f"hota={n_with_ht}/{n_videos}")
    print(f"Failures: {n_failures}/{n_videos} videos marked as pipeline failure")

    # ── Write JSON report ──────────────────────────────────────────────────────
    today = date.today().isoformat()
    report_path = Path(__file__).resolve().parent.parent / "reports" / f"quality_calibration_{today}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "date": today,
        "n_videos": n_videos,
        "failure_thresholds": {
            "action_acc": ACTION_ACC_FAIL,
            "score_acc": SCORE_ACC_FAIL,
            "hota": HOTA_FAIL,
        },
        "coverage": {
            "quality_metrics": n_with_qm,
            "action_acc": n_with_aa,
            "score_acc": n_with_sc,
            "hota": n_with_ht,
        },
        "n_failures": n_failures,
        "calibration": calibration_rows,
        "per_video": [
            {
                "video_id":       vr.video_id,
                "action_acc":     vr.action_acc,
                "score_acc":      vr.score_acc,
                "hota":           vr.hota,
                "is_failure":     vr.is_failure,
                "quality_metrics": vr.quality_metrics,
                "quality_error":  vr.quality_error,
                "pipeline_error": vr.pipeline_error,
                "hota_error":     vr.hota_error,
            }
            for vr in results
        ],
    }

    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
