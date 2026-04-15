"""Populate BallRawCache for GT rallies missing raw WASB output.

For every ball-GT rally whose BallRawCache entry is absent, downloads the
source video via VideoResolver, runs WASBBallTracker.track_video with
enable_filtering=False + preserve_raw=True to capture raw detections, and
writes a CachedBallData record. Idempotent — skips rallies that already
have a cache entry.

Usage:
    uv run python scripts/populate_ball_raw_cache.py           # fill all missing
    uv run python scripts/populate_ball_raw_cache.py --limit 3 # sanity run
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from rallycut.evaluation.tracking.ball_grid_search import (
    BallRawCache,
    CachedBallData,
)
from rallycut.evaluation.tracking.db import (
    get_rally_info,
    load_labeled_rallies,
)
from rallycut.evaluation.video_resolver import VideoResolver
from rallycut.tracking.wasb_model import WASBBallTracker

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    rallies = load_labeled_rallies()
    ball_gt_rallies = [
        r for r in rallies
        if any(p.label == "ball" for p in r.ground_truth.positions)
    ]
    cache = BallRawCache()
    missing = [r for r in ball_gt_rallies if not cache.has(r.rally_id)]
    total_missing = len(missing)
    if args.limit is not None:
        missing = missing[: args.limit]

    logger.info(
        f"Ball-GT rallies: {len(ball_gt_rallies)} | "
        f"already cached: {len(ball_gt_rallies) - total_missing} | "
        f"total missing: {total_missing} | "
        f"this run: {len(missing)}"
    )
    if not missing:
        logger.info("Nothing to do.")
        return 0

    resolver = VideoResolver()
    tracker = WASBBallTracker()

    t_start = time.time()
    for i, rally in enumerate(missing, start=1):
        logger.info(
            f"[{i}/{len(missing)}] rally={rally.rally_id[:8]} "
            f"video={rally.video_id[:8]} "
            f"ball_gt={sum(1 for p in rally.ground_truth.positions if p.label == 'ball')}"
        )
        info = get_rally_info(rally.rally_id)
        if info is None:
            logger.warning("  → skipping: no RallyVideoInfo")
            continue
        try:
            video_path: Path = resolver.resolve(info.s3_key, info.content_hash)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  → skipping: video resolve failed: {e}")
            continue
        t0 = time.time()
        try:
            result = tracker.track_video(
                video_path,
                start_ms=info.start_ms,
                end_ms=info.end_ms,
                enable_filtering=False,
                preserve_raw=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"  → tracker failed: {e}")
            continue
        raw = result.raw_positions if result.raw_positions is not None else result.positions
        if not raw:
            logger.warning("  → empty raw output, skipping cache put")
            continue
        cached = CachedBallData(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            raw_ball_positions=raw,
            video_fps=rally.video_fps,
            frame_count=result.frame_count,
            video_width=rally.video_width,
            video_height=rally.video_height,
            start_ms=info.start_ms,
            end_ms=info.end_ms,
        )
        cache.put(cached)
        logger.info(
            f"  → cached {len(raw)} raw positions ({time.time() - t0:.1f}s)"
        )
    logger.info(
        f"Done. {len(missing)} rally(s) processed in {time.time() - t_start:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
