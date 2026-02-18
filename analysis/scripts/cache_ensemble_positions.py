"""Cache merged WASB+VballNet ensemble raw positions.

Downloads videos, runs both WASB and VballNet inference (unfiltered),
merges with WASB-primary strategy, applies WASB frame offset correction,
and stores merged positions in BallRawCache.

By default caches only ball-GT rallies. Use --all-tracked to cache
all rallies with player tracks (for pseudo-label training).

Source tagging via motion_energy field:
  - motion_energy >= 1.0 = WASB source
  - motion_energy < 1.0 = VballNet source

Usage:
    cd analysis
    uv run python scripts/cache_ensemble_positions.py
    uv run python scripts/cache_ensemble_positions.py --all-tracked
    uv run python scripts/cache_ensemble_positions.py --device mps
    uv run python scripts/cache_ensemble_positions.py --clear
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from eval_wasb import run_wasb_inference

from rallycut.evaluation.tracking.ball_grid_search import BallRawCache, CachedBallData
from rallycut.evaluation.tracking.ball_metrics import find_optimal_frame_offset
from rallycut.evaluation.tracking.db import (
    TrackedRally,
    get_video_path,
    load_all_tracked_rallies,
    load_labeled_rallies,
)
from rallycut.tracking.ball_filter import BallFilterConfig, BallTemporalFilter
from rallycut.tracking.ball_tracker import BallPosition, BallTracker
from rallycut.tracking.wasb_model import load_wasb_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ENSEMBLE_CACHE_DIR = Path.home() / ".cache" / "rallycut" / "ensemble_grid_search"


def merge_wasb_primary(
    wasb_positions: list[BallPosition],
    vballnet_positions: list[BallPosition],
) -> list[BallPosition]:
    """Merge WASB and VballNet: WASB takes priority, VballNet fills gaps.

    Source tagged via motion_energy: >= 1.0 = WASB, < 1.0 = VballNet.
    """
    wasb_by_frame: dict[int, BallPosition] = {}
    for p in wasb_positions:
        if p.confidence > 0:
            existing = wasb_by_frame.get(p.frame_number)
            if existing is None or p.confidence > existing.confidence:
                wasb_by_frame[p.frame_number] = p

    vnet_by_frame: dict[int, BallPosition] = {}
    for p in vballnet_positions:
        if p.confidence > 0:
            existing = vnet_by_frame.get(p.frame_number)
            if existing is None or p.confidence > existing.confidence:
                vnet_by_frame[p.frame_number] = p

    merged: dict[int, BallPosition] = {}
    all_frames = set(wasb_by_frame.keys()) | set(vnet_by_frame.keys())

    for frame in all_frames:
        wasb_det = wasb_by_frame.get(frame)
        vnet_det = vnet_by_frame.get(frame)

        if wasb_det is not None:
            merged[frame] = BallPosition(
                frame_number=frame,
                x=wasb_det.x,
                y=wasb_det.y,
                confidence=wasb_det.confidence,
                motion_energy=1.0,
            )
        elif vnet_det is not None:
            merged[frame] = BallPosition(
                frame_number=frame,
                x=vnet_det.x,
                y=vnet_det.y,
                confidence=vnet_det.confidence,
                motion_energy=min(vnet_det.motion_energy, 0.99),
            )

    return [merged[f] for f in sorted(merged.keys())]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache ensemble (WASB+VballNet) raw positions for GT rallies"
    )
    parser.add_argument("--device", default="mps", help="Device for WASB (cpu, cuda, mps)")
    parser.add_argument(
        "--wasb-threshold", type=float, default=0.3, help="WASB heatmap threshold"
    )
    parser.add_argument("--vballnet-model", default="v2", help="VballNet model (default: v2)")
    parser.add_argument("--video", "-v", help="Filter to specific video ID")
    parser.add_argument(
        "--all-tracked",
        action="store_true",
        help="Cache all rallies with player tracks (not just ball GT rallies)",
    )
    parser.add_argument("--clear", action="store_true", help="Clear cache before running")
    args = parser.parse_args()

    cache = BallRawCache(cache_dir=ENSEMBLE_CACHE_DIR)

    if args.clear:
        import shutil

        shutil.rmtree(cache.cache_dir, ignore_errors=True)
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared ensemble cache")

    # Load WASB model
    try:
        wasb_model = load_wasb_model(device=args.device)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"WASB threshold: {args.wasb_threshold}, Device: {args.device}")

    # Load rallies — either all tracked or just ball GT
    if args.all_tracked:
        all_tracked = load_all_tracked_rallies(video_id=args.video)
        if not all_tracked:
            logger.error("No tracked rallies found")
            sys.exit(1)
        logger.info(f"Found {len(all_tracked)} tracked rallies (--all-tracked)")

        # Build set of rally IDs that have ball GT (for offset detection)
        gt_rallies = load_labeled_rallies(video_id=args.video)
        gt_rally_map = {r.rally_id: r for r in gt_rallies}
        logger.info(f"  ({len(gt_rally_map)} have ball GT for offset detection)")

        # Convert to uniform interface: list of TrackedRally with optional GT
        rallies_with_gt: list[tuple[TrackedRally, bool]] = []
        for r in all_tracked:
            has_gt = r.rally_id in gt_rally_map
            rallies_with_gt.append((r, has_gt))
    else:
        gt_rallies = load_labeled_rallies(video_id=args.video)
        if not gt_rallies:
            logger.error("No rallies found with ball GT")
            sys.exit(1)
        logger.info(f"Found {len(gt_rallies)} rallies with ball GT")
        gt_rally_map = {r.rally_id: r for r in gt_rallies}
        rallies_with_gt = [
            (
                TrackedRally(
                    rally_id=r.rally_id,
                    video_id=r.video_id,
                    start_ms=r.start_ms,
                    end_ms=r.end_ms,
                    video_fps=r.video_fps,
                    video_width=r.video_width,
                    video_height=r.video_height,
                ),
                True,
            )
            for r in gt_rallies
        ]

    # Also load VballNet raw cache for fallback positions
    vnet_raw_cache = BallRawCache()

    # Group rallies by video
    rallies_by_video: dict[str, list[tuple[TrackedRally, bool]]] = {}
    for rally, has_gt in rallies_with_gt:
        rallies_by_video.setdefault(rally.video_id, []).append((rally, has_gt))

    vnet_tracker = BallTracker(model=args.vballnet_model)
    cached_count = 0
    skipped_count = 0

    for video_id, video_rallies in rallies_by_video.items():
        uncached = [(r, gt) for r, gt in video_rallies if not cache.has(r.rally_id)]
        if not uncached:
            logger.info(
                f"Video {video_id[:8]}...: all {len(video_rallies)} rallies cached, skipping"
            )
            skipped_count += len(video_rallies)
            continue

        # Download video
        logger.info(f"Video {video_id[:8]}...: downloading...")
        video_path = get_video_path(video_id)
        if video_path is None:
            logger.error(f"  Could not download video {video_id}")
            continue

        for rally, has_gt in uncached:
            if cache.has(rally.rally_id):
                skipped_count += 1
                continue

            logger.info(
                f"  Rally {rally.rally_id[:8]}...: "
                f"tracking {rally.start_ms}ms-{rally.end_ms}ms..."
                f"{'' if has_gt else ' (no GT)'}"
            )

            # --- WASB inference ---
            wasb_raw = run_wasb_inference(
                wasb_model,
                video_path,
                rally.start_ms,
                rally.end_ms,
                device=args.device,
                threshold=args.wasb_threshold,
            )

            # Find optimal WASB frame offset (only if GT available)
            if has_gt and rally.rally_id in gt_rally_map:
                gt_rally = gt_rally_map[rally.rally_id]
                wasb_offset, _ = find_optimal_frame_offset(
                    gt_rally.ground_truth.positions,
                    wasb_raw,
                    rally.video_width,
                    rally.video_height,
                )
            else:
                wasb_offset = 0

            # Apply offset to WASB
            if wasb_offset > 0:
                wasb_shifted = [
                    BallPosition(
                        frame_number=p.frame_number - wasb_offset,
                        x=p.x,
                        y=p.y,
                        confidence=p.confidence,
                        motion_energy=p.motion_energy,
                    )
                    for p in wasb_raw
                ]
                logger.info(f"    WASB: {len(wasb_raw)} detections, offset={wasb_offset}")
            else:
                wasb_shifted = wasb_raw
                logger.info(f"    WASB: {len(wasb_raw)} detections, no offset")

            # --- VballNet inference ---
            # Try to use cached raw VballNet positions first
            vnet_cached = vnet_raw_cache.get(rally.rally_id)
            if vnet_cached is not None:
                vnet_raw = vnet_cached.raw_ball_positions
                logger.info(f"    VballNet: {len(vnet_raw)} positions (from cache)")
            else:
                vnet_result = vnet_tracker.track_video(
                    video_path,
                    start_ms=rally.start_ms,
                    end_ms=rally.end_ms,
                    enable_filtering=False,
                )
                # Convert to rally-relative frame numbers
                if vnet_result.positions:
                    first_frame = min(p.frame_number for p in vnet_result.positions)
                else:
                    first_frame = 0
                vnet_raw = [
                    BallPosition(
                        frame_number=p.frame_number - first_frame,
                        x=p.x,
                        y=p.y,
                        confidence=p.confidence,
                        motion_energy=p.motion_energy,
                    )
                    for p in vnet_result.positions
                ]
                logger.info(f"    VballNet: {len(vnet_raw)} positions (fresh inference)")

            # --- Pre-filter VballNet: remove stationary false positives ---
            # VballNet detects players as ball with low motion energy. These would
            # become fallback positions where WASB has no detection. Motion energy
            # filter zeroes their confidence so they're excluded from the merge.
            vnet_prefilter = BallFilterConfig(
                enable_motion_energy_filter=True,
                enable_segment_pruning=False,
                enable_oscillation_pruning=False,
                enable_exit_ghost_removal=False,
                enable_outlier_removal=False,
                enable_blip_removal=False,
                enable_interpolation=False,
            )
            vnet_filtered = BallTemporalFilter(vnet_prefilter).filter_batch(vnet_raw)
            me_removed = sum(1 for p in vnet_raw if p.confidence > 0) - sum(
                1 for p in vnet_filtered if p.confidence > 0
            )
            if me_removed > 0:
                logger.info(f"    VballNet pre-filter: removed {me_removed} low-energy FPs")

            # --- Merge ---
            merged = merge_wasb_primary(wasb_shifted, vnet_filtered)
            wasb_count = sum(1 for p in merged if p.motion_energy >= 1.0)
            vnet_count = sum(1 for p in merged if p.motion_energy < 1.0)
            logger.info(
                f"    Merged: {wasb_count} WASB + {vnet_count} VballNet = {len(merged)} total"
            )

            # Cache merged positions
            cached_data = CachedBallData(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                raw_ball_positions=merged,
                video_fps=rally.video_fps,
                frame_count=len(merged),
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
