"""Cache raw (unfiltered) ball positions for all GT rallies.

Downloads videos, runs VballNet inference without Kalman filtering,
and saves raw positions to the ball grid search cache.

Usage:
    uv run python scripts/cache_raw_ball_positions.py
    uv run python scripts/cache_raw_ball_positions.py --video 07fedbd4-...
"""

from __future__ import annotations

import argparse
import logging
import sys

from rallycut.evaluation.tracking.ball_grid_search import BallRawCache, CachedBallData
from rallycut.evaluation.tracking.db import (
    get_video_path,
    load_labeled_rallies,
)
from rallycut.tracking.ball_tracker import BallPosition, BallTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache raw ball positions for GT rallies")
    parser.add_argument("--video", "-v", help="Filter to specific video ID")
    parser.add_argument("--model", default="v2", help="Ball model (default: v2)")
    parser.add_argument("--clear", action="store_true", help="Clear cache before running")
    args = parser.parse_args()

    cache = BallRawCache()

    if args.clear:
        import shutil

        shutil.rmtree(cache.cache_dir, ignore_errors=True)
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared ball cache")

    # Load GT rallies
    rallies = load_labeled_rallies(video_id=args.video, ball_gt_only=True)
    if not rallies:
        logger.error("No rallies found with ball GT")
        sys.exit(1)

    logger.info(f"Found {len(rallies)} rallies with ball GT")

    # Group rallies by video to avoid downloading the same video multiple times
    rallies_by_video: dict[str, list] = {}
    for rally in rallies:
        rallies_by_video.setdefault(rally.video_id, []).append(rally)

    tracker = BallTracker(model=args.model)
    cached_count = 0
    skipped_count = 0

    for video_id, video_rallies in rallies_by_video.items():
        # Check if all rallies for this video are already cached
        uncached = [r for r in video_rallies if not cache.has(r.rally_id)]
        if not uncached:
            logger.info(f"Video {video_id[:8]}...: all {len(video_rallies)} rallies cached, skipping")
            skipped_count += len(video_rallies)
            continue

        # Download video
        logger.info(f"Video {video_id[:8]}...: downloading...")
        video_path = get_video_path(video_id)
        if video_path is None:
            logger.error(f"  Could not download video {video_id}")
            continue

        logger.info(f"  Video path: {video_path}")

        # Run ball tracking for each rally (without filtering)
        for rally in uncached:
            if cache.has(rally.rally_id):
                skipped_count += 1
                continue

            logger.info(
                f"  Rally {rally.rally_id[:8]}...: "
                f"tracking {rally.start_ms}ms-{rally.end_ms}ms..."
            )

            result = tracker.track_video(
                video_path,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
                enable_filtering=False,  # Raw, unfiltered positions
            )

            # Convert absolute frame numbers to rally-relative (0-indexed)
            # BallTracker uses absolute frame numbers, but GT is rally-relative
            # Use the actual first frame from the result as offset
            if result.positions:
                first_frame = min(p.frame_number for p in result.positions)
            else:
                first_frame = 0
            raw_positions = [
                BallPosition(
                    frame_number=p.frame_number - first_frame,
                    x=p.x,
                    y=p.y,
                    confidence=p.confidence,
                )
                for p in result.positions
            ]

            logger.info(
                f"    Got {len(raw_positions)} raw detections "
                f"(detection rate: {result.detection_rate * 100:.1f}%, "
                f"frames 0-{raw_positions[-1].frame_number if raw_positions else 0}, "
                f"offset={first_frame})"
            )

            # Cache raw positions
            cached_data = CachedBallData(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                raw_ball_positions=raw_positions,
                video_fps=rally.video_fps,
                frame_count=result.frame_count,
                video_width=rally.video_width,
                video_height=rally.video_height,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
            )
            cache.put(cached_data)
            cached_count += 1

    stats = cache.stats()
    logger.info(f"\nDone! Cached: {cached_count}, Skipped: {skipped_count}")
    logger.info(f"Cache: {stats['count']} rallies, {stats['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
