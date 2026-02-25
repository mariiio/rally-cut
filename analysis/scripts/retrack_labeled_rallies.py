"""Re-run player tracking on labeled rallies with the current pipeline.

Downloads video files, runs full tracking (YOLO + BoT-SORT + post-processing
with court identity resolution), evaluates against GT, and compares with
the stored DB baseline.

Usage:
    uv run python scripts/retrack_labeled_rallies.py
    uv run python scripts/retrack_labeled_rallies.py --rally <rally-id>
    uv run python scripts/retrack_labeled_rallies.py --stride 2  # faster (skip frames)
    uv run python scripts/retrack_labeled_rallies.py --dry-run   # just check video availability
    uv run python scripts/retrack_labeled_rallies.py --save      # save new predictions to DB
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import (
    TrackingEvaluationRally,
    get_video_path,
    load_labeled_rallies,
    save_predictions,
)
from rallycut.evaluation.tracking.metrics import (
    TrackingEvaluationResult,
    evaluate_rally,
)
from rallycut.tracking.player_tracker import PlayerTracker, PlayerTrackingResult

logger = logging.getLogger(__name__)


def _create_calibrator(
    corners_json: list[dict[str, float]] | None,
) -> CourtCalibrator | None:
    """Create a CourtCalibrator from DB corner JSON."""
    if not corners_json or len(corners_json) != 4:
        return None
    calibrator = CourtCalibrator()
    image_corners = [(c["x"], c["y"]) for c in corners_json]
    calibrator.calibrate(image_corners)
    return calibrator


def _format_metrics(result: TrackingEvaluationResult) -> dict[str, float | int]:
    """Extract key metrics."""
    agg = result.aggregate
    hota = result.hota_metrics
    return {
        "HOTA": hota.hota * 100 if hota else 0.0,
        "AssA": hota.assa * 100 if hota else 0.0,
        "MOTA": agg.mota * 100,
        "F1": agg.f1 * 100,
        "IDsw": agg.num_id_switches,
    }


def _adjust_frame_numbers(
    result: PlayerTrackingResult,
    start_ms: int,
    video_path: str,
) -> None:
    """Adjust frame numbers from absolute video frames to rally-relative."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps:
        logger.warning("Could not read FPS for %s, defaulting to 30.0", video_path)
        video_fps = 30.0
    cap.release()

    start_frame = int((start_ms / 1000.0) * video_fps)

    for pos in result.positions:
        pos.frame_number = pos.frame_number - start_frame

    for pos in result.raw_positions:
        pos.frame_number = pos.frame_number - start_frame

    if result.ball_positions:
        for bp in result.ball_positions:
            bp.frame_number = bp.frame_number - start_frame


def _retrack_rally(
    rally: TrackingEvaluationRally,
    stride: int = 1,
    team_aware: bool = False,
    enable_ball: bool = True,
) -> PlayerTrackingResult | None:
    """Re-run tracking for a single rally."""
    # Get video file
    video_path = get_video_path(rally.video_id)
    if video_path is None:
        logger.error(f"Could not get video for {rally.video_id}")
        return None

    # Create calibrator
    calibrator = _create_calibrator(rally.court_calibration_json)

    # Build team-aware config if requested
    ta_config = None
    if team_aware:
        from rallycut.tracking.team_aware_tracker import TeamAwareConfig

        ta_config = TeamAwareConfig(enabled=True)

    # Run ball tracking first (matches production pipeline)
    ball_positions = None
    if enable_ball:
        from rallycut.tracking.ball_tracker import create_ball_tracker

        ball_tracker = create_ball_tracker()
        ball_result = ball_tracker.track_video(
            video_path, start_ms=rally.start_ms, end_ms=rally.end_ms,
        )
        ball_positions = ball_result.positions
        logger.info(
            f"Ball tracking: {len(ball_positions)} positions"
        )

    # Create tracker
    tracker = PlayerTracker()

    # Run tracking with full pipeline
    result = tracker.track_video(
        video_path=video_path,
        start_ms=rally.start_ms,
        end_ms=rally.end_ms,
        stride=stride,
        filter_enabled=True,
        court_calibrator=calibrator,
        team_aware_config=ta_config,
        ball_positions=ball_positions,
    )

    # Convert absolute video frame numbers to rally-relative (0-indexed)
    # track_video() uses absolute frame indices; GT expects 0-indexed
    _adjust_frame_numbers(result, rally.start_ms, str(video_path))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-track labeled rallies with current pipeline"
    )
    parser.add_argument("--rally", help="Specific rally ID (full UUID)")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Frame stride (1=all, 2=every other frame for 60fps videos)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Just check video availability")
    parser.add_argument(
        "--save", action="store_true",
        help="Save new predictions to DB (overwrites stored predictions)",
    )
    parser.add_argument(
        "--team-aware", action="store_true",
        help="Enable team-aware BoT-SORT penalty (requires calibration)",
    )
    parser.add_argument(
        "--no-ball", action="store_true",
        help="Skip ball tracking (faster, but diverges from production pipeline)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    # Suppress noisy loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("rallycut.tracking.ball_filter").setLevel(logging.WARNING)

    # Load labeled rallies
    rallies = load_labeled_rallies(rally_id=args.rally)
    if not rallies:
        print("No labeled rallies found")
        sys.exit(1)

    print(f"Loaded {len(rallies)} labeled rally(s)")

    # Check video availability
    if args.dry_run:
        print("\nDry run: checking video availability...")
        for rally in rallies:
            video_path = get_video_path(rally.video_id)
            calib = "Yes" if rally.court_calibration_json else "No"
            duration = (rally.end_ms - rally.start_ms) / 1000
            if video_path:
                print(f"  {rally.rally_id[:12]} video={rally.video_id[:8]} "
                      f"calib={calib} duration={duration:.1f}s OK: {video_path}")
            else:
                print(f"  {rally.rally_id[:12]} video={rally.video_id[:8]} "
                      f"calib={calib} duration={duration:.1f}s MISSING")
        return

    # Track results
    header = (
        f"{'Rally':<14} "
        f"{'Base HOTA':>9} {'New HOTA':>9} {'Δ HOTA':>7} "
        f"{'Base IDsw':>9} {'New IDsw':>9} {'Δ IDsw':>7} "
        f"{'Base F1':>8} {'New F1':>8} "
        f"{'Time':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    total_base_hota = 0.0
    total_new_hota = 0.0
    total_base_idsw = 0
    total_new_idsw = 0
    processed = 0

    for rally in rallies:
        rally_short = rally.rally_id[:12]

        # Baseline (stored predictions)
        if rally.predictions is None:
            print(f"{rally_short:<14} (no stored predictions)")
            continue

        baseline_result = evaluate_rally(
            rally.rally_id,
            rally.ground_truth,
            rally.predictions,
            video_width=rally.video_width,
            video_height=rally.video_height,
        )
        bm = _format_metrics(baseline_result)

        # Re-track
        t0 = time.time()
        new_predictions = _retrack_rally(
            rally,
            stride=args.stride,
            team_aware=args.team_aware,
            enable_ball=not args.no_ball,
        )
        elapsed = time.time() - t0

        if new_predictions is None:
            print(f"{rally_short:<14} {bm['HOTA']:>8.1f}% {'FAIL':>9} "
                  f"{'':>7} {bm['IDsw']:>9} {'':>9} {'':>7} "
                  f"{bm['F1']:>7.1f}% {'':>8} {elapsed:>5.1f}s")
            continue

        new_result = evaluate_rally(
            rally.rally_id,
            rally.ground_truth,
            new_predictions,
            video_width=rally.video_width,
            video_height=rally.video_height,
        )
        nm = _format_metrics(new_result)

        delta_hota = nm["HOTA"] - bm["HOTA"]
        delta_idsw = nm["IDsw"] - bm["IDsw"]

        saved = ""
        if args.save:
            save_predictions(rally.rally_id, new_predictions, elapsed * 1000)
            saved = " [SAVED]"

        print(
            f"{rally_short:<14} "
            f"{bm['HOTA']:>8.1f}% {nm['HOTA']:>8.1f}% {delta_hota:>+6.1f}% "
            f"{bm['IDsw']:>9} {nm['IDsw']:>9} {delta_idsw:>+7d} "
            f"{bm['F1']:>7.1f}% {nm['F1']:>7.1f}% "
            f"{elapsed:>5.1f}s{saved}"
        )

        base_hota = baseline_result.hota_metrics.hota if baseline_result.hota_metrics else 0.0
        new_hota = new_result.hota_metrics.hota if new_result.hota_metrics else 0.0
        total_base_hota += base_hota
        total_new_hota += new_hota
        total_base_idsw += baseline_result.aggregate.num_id_switches
        total_new_idsw += new_result.aggregate.num_id_switches
        processed += 1

    if processed > 0:
        print(f"\n{'='*80}")
        print(f"Summary ({processed} rallies re-tracked)")
        print(f"  Baseline: HOTA={total_base_hota/processed*100:.1f}%, IDsw={total_base_idsw}")
        print(f"  New:      HOTA={total_new_hota/processed*100:.1f}%, IDsw={total_new_idsw}")
        print(f"  Delta:    HOTA={(total_new_hota-total_base_hota)/processed*100:+.1f}%, "
              f"IDsw={total_new_idsw-total_base_idsw:+d}")


if __name__ == "__main__":
    main()
