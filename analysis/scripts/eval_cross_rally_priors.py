"""Evaluate cross-rally priors impact on global identity.

For each video that has GT-labeled rallies:
1. Run match-players to get accumulated profiles
2. Re-track each labeled rally WITHOUT priors (baseline)
3. Re-track each labeled rally WITH priors
4. Compare IDsw and HOTA

Usage:
    uv run python scripts/eval_cross_rally_priors.py
    uv run python scripts/eval_cross_rally_priors.py --rally <rally-id>
    uv run python scripts/eval_cross_rally_priors.py --video <video-id>
    uv run python scripts/eval_cross_rally_priors.py --stride 2   # faster for 60fps
    uv run python scripts/eval_cross_rally_priors.py -v            # verbose logging
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import (
    TrackingEvaluationRally,
    get_video_path,
    load_labeled_rallies,
    load_rallies_for_video,
)
from rallycut.evaluation.tracking.metrics import (
    TrackingEvaluationResult,
    evaluate_rally,
)
from rallycut.tracking.match_tracker import (
    build_cross_rally_priors,
    extract_rally_appearances,
    match_players_across_rallies,
)
from rallycut.tracking.player_tracker import PlayerTracker, PlayerTrackingResult

logger = logging.getLogger(__name__)


def _create_calibrator(
    corners_json: list[dict[str, float]] | None,
) -> CourtCalibrator | None:
    if not corners_json or len(corners_json) != 4:
        return None
    calibrator = CourtCalibrator()
    image_corners = [(c["x"], c["y"]) for c in corners_json]
    calibrator.calibrate(image_corners)
    return calibrator


def _format_metrics(result: TrackingEvaluationResult) -> dict[str, float | int]:
    agg = result.aggregate
    hota = result.hota_metrics
    return {
        "HOTA": hota.hota * 100 if hota else 0.0,
        "AssA": hota.assa * 100 if hota else 0.0,
        "F1": agg.f1 * 100,
        "IDsw": agg.num_id_switches,
    }


def _adjust_frame_numbers(
    result: PlayerTrackingResult,
    start_ms: int,
    video_path: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    start_frame = int((start_ms / 1000.0) * video_fps)

    for pos in result.positions:
        pos.frame_number = pos.frame_number - start_frame
    for pos in result.raw_positions:
        pos.frame_number = pos.frame_number - start_frame
    if result.ball_positions:
        for bp in result.ball_positions:
            bp.frame_number = bp.frame_number - start_frame


def _track_rally(
    rally: TrackingEvaluationRally,
    stride: int = 1,
    cross_rally_priors: list | None = None,
) -> PlayerTrackingResult | None:
    """Track a single rally, optionally with cross-rally priors."""
    video_path = get_video_path(rally.video_id)
    if video_path is None:
        logger.error(f"Could not get video for {rally.video_id}")
        return None

    calibrator = _create_calibrator(rally.court_calibration_json)

    # Ball tracking (matches production pipeline)
    from rallycut.tracking.ball_tracker import create_ball_tracker

    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(
        video_path, start_ms=rally.start_ms, end_ms=rally.end_ms,
    )

    # Player tracking
    tracker = PlayerTracker()
    result = tracker.track_video(
        video_path=video_path,
        start_ms=rally.start_ms,
        end_ms=rally.end_ms,
        stride=stride,
        filter_enabled=True,
        court_calibrator=calibrator,
        ball_positions=ball_result.positions,
        cross_rally_priors=cross_rally_priors,
    )

    _adjust_frame_numbers(result, rally.start_ms, str(video_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-rally priors impact on global identity"
    )
    parser.add_argument("--rally", help="Specific rally ID")
    parser.add_argument("--video", help="Specific video ID")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Frame stride (1=all, 2=every other for 60fps)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("rallycut.tracking.ball_filter").setLevel(logging.WARNING)
    logging.getLogger("rallycut.tracking.player_filter").setLevel(logging.WARNING)
    logging.getLogger("rallycut.tracking.player_tracker").setLevel(logging.WARNING)
    if not args.verbose:
        logging.getLogger("rallycut.tracking.global_identity").setLevel(logging.WARNING)
        logging.getLogger("rallycut.tracking.court_identity").setLevel(logging.WARNING)
        logging.getLogger("rallycut.tracking.color_repair").setLevel(logging.WARNING)

    # Load labeled rallies
    rallies = load_labeled_rallies(
        rally_id=args.rally,
        video_id=args.video,
    )
    if not rallies:
        print("No labeled rallies found")
        sys.exit(1)

    # Group by video
    rallies_by_video: dict[str, list[TrackingEvaluationRally]] = defaultdict(list)
    for rally in rallies:
        rallies_by_video[rally.video_id].append(rally)

    print(f"Found {len(rallies)} labeled rally(s) across {len(rallies_by_video)} video(s)")

    # Step 1: Run match-players for each video to get profiles
    print("\n--- Step 1: Running match-players per video ---")
    video_priors: dict[str, list | None] = {}

    for video_id, video_rallies in rallies_by_video.items():
        video_path = get_video_path(video_id)
        if video_path is None:
            print(f"  {video_id[:8]}: SKIP (video not available)")
            video_priors[video_id] = None
            continue

        # Load ALL tracked rallies for this video (not just GT-labeled ones)
        all_rallies = load_rallies_for_video(video_id)
        if len(all_rallies) < 2:
            print(f"  {video_id[:8]}: SKIP ({len(all_rallies)} rally, need ≥2 for priors)")
            video_priors[video_id] = None
            continue

        print(f"  {video_id[:8]}: matching {len(all_rallies)} rallies...", end=" ", flush=True)
        t0 = time.time()
        match_result = match_players_across_rallies(
            video_path=video_path,
            rallies=all_rallies,
            num_samples=12,
        )
        elapsed = time.time() - t0

        # Build priors from profiles
        profiles_data = {
            str(pid): profile.to_dict()
            for pid, profile in match_result.player_profiles.items()
            if profile.rally_count > 0
        }
        priors = build_cross_rally_priors(profiles_data)

        if priors:
            rally_counts = [p.rally_count for p in priors]
            print(
                f"{len(priors)} priors "
                f"(rally_counts: {rally_counts}, "
                f"confidence: {[f'{p.confidence:.2f}' for p in priors]}) "
                f"[{elapsed:.1f}s]"
            )
        else:
            print(f"no usable priors [{elapsed:.1f}s]")

        video_priors[video_id] = priors

    # Step 2: Track each rally with and without priors
    print("\n--- Step 2: Tracking with vs without priors ---")

    header = (
        f"{'Rally':<14} {'Video':<10} "
        f"{'No-Prior':>9} {'Prior':>9} {'Δ HOTA':>7} "
        f"{'NP IDsw':>7} {'P IDsw':>7} {'Δ IDsw':>7} "
        f"{'Priors':>6} {'Time':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    total_np_hota = 0.0
    total_p_hota = 0.0
    total_np_idsw = 0
    total_p_idsw = 0
    processed = 0
    skipped_no_priors = 0

    for video_id, video_rallies in rallies_by_video.items():
        priors = video_priors.get(video_id)

        for rally in video_rallies:
            rally_short = rally.rally_id[:12]
            video_short = video_id[:8]

            if rally.predictions is None:
                print(f"{rally_short:<14} {video_short:<10} (no stored predictions)")
                continue

            t0 = time.time()

            # Track WITHOUT priors
            result_no_prior = _track_rally(rally, stride=args.stride, cross_rally_priors=None)
            if result_no_prior is None:
                print(f"{rally_short:<14} {video_short:<10} FAIL (tracking error)")
                continue

            eval_no_prior = evaluate_rally(
                rally.rally_id, rally.ground_truth, result_no_prior,
                video_width=rally.video_width, video_height=rally.video_height,
            )
            np_m = _format_metrics(eval_no_prior)

            # Track WITH priors (if available)
            if priors:
                result_prior = _track_rally(rally, stride=args.stride, cross_rally_priors=priors)
                if result_prior is None:
                    print(f"{rally_short:<14} {video_short:<10} FAIL (prior tracking error)")
                    continue

                eval_prior = evaluate_rally(
                    rally.rally_id, rally.ground_truth, result_prior,
                    video_width=rally.video_width, video_height=rally.video_height,
                )
                p_m = _format_metrics(eval_prior)
            else:
                p_m = dict(np_m)  # Same result when no priors
                skipped_no_priors += 1

            elapsed = time.time() - t0

            delta_hota = p_m["HOTA"] - np_m["HOTA"]
            delta_idsw = p_m["IDsw"] - np_m["IDsw"]
            n_priors = len(priors) if priors else 0

            print(
                f"{rally_short:<14} {video_short:<10} "
                f"{np_m['HOTA']:>8.1f}% {p_m['HOTA']:>8.1f}% {delta_hota:>+6.1f}% "
                f"{np_m['IDsw']:>7} {p_m['IDsw']:>7} {delta_idsw:>+7d} "
                f"{n_priors:>6} {elapsed:>5.0f}s"
            )

            np_hota = eval_no_prior.hota_metrics.hota if eval_no_prior.hota_metrics else 0.0
            p_hota = eval_prior.hota_metrics.hota if priors and eval_prior.hota_metrics else np_hota
            total_np_hota += np_hota
            total_p_hota += p_hota
            total_np_idsw += eval_no_prior.aggregate.num_id_switches
            total_p_idsw += (
                eval_prior.aggregate.num_id_switches
                if priors
                else eval_no_prior.aggregate.num_id_switches
            )
            processed += 1

    # Summary
    if processed > 0:
        print(f"\n{'='*80}")
        print(f"Summary ({processed} rallies, {skipped_no_priors} without priors)")
        print(f"  No priors: HOTA={total_np_hota/processed*100:.1f}%, IDsw={total_np_idsw}")
        print(f"  W/ priors: HOTA={total_p_hota/processed*100:.1f}%, IDsw={total_p_idsw}")
        print(
            f"  Delta:     HOTA={(total_p_hota-total_np_hota)/processed*100:+.1f}%, "
            f"IDsw={total_p_idsw-total_np_idsw:+d}"
        )


if __name__ == "__main__":
    main()
