"""Diagnostic script for ball tracking issues.

Loads cached raw positions for affected rallies and runs the pipeline
stage-by-stage, showing what's removed at each stage for specific timestamps.
"""

from __future__ import annotations

import logging

from rallycut.evaluation.tracking.ball_grid_search import BallRawCache
from rallycut.evaluation.tracking.ball_metrics import (
    evaluate_ball_tracking,
    find_optimal_frame_offset,
)
from rallycut.evaluation.tracking.db import load_labeled_rallies
from rallycut.tracking.ball_filter import BallFilterConfig, BallTemporalFilter
from rallycut.tracking.ball_tracker import BallPosition

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Key timestamps to inspect (rally_id prefix -> frame numbers at ~30fps)
# Mapping: video_id -> rally_id:
#   a5866029 -> bd77efd1 (trajectory glitch at ~0:18.5)
#   920ba69d -> 0af554b5 (premature termination at ~0:13.4)
#   07fedbd4 -> 1bfcbc4f (false start at ~0:17 / ~0:17.9)
#   c6e4c876 -> not in GT videos (no cache)
KEY_TIMESTAMPS: dict[str, list[int]] = {
    "bd77efd1": [555],  # video a5866029 ~0:18.5 - trajectory glitch
    "0af554b5": [402],  # video 920ba69d ~0:13.4 - premature termination
    "1bfcbc4f": [510, 537],  # video 07fedbd4 ~0:17 / ~0:17.9 - false start
}


def find_positions_near_frame(
    positions: list[BallPosition], frame: int, window: int = 10
) -> list[BallPosition]:
    """Find positions within window frames of target frame."""
    return [p for p in positions if abs(p.frame_number - frame) <= window]


def run_pipeline_stages(
    raw_positions: list[BallPosition],
    config: BallFilterConfig | None = None,
) -> dict[str, list[BallPosition]]:
    """Run pipeline stage-by-stage, returning positions after each stage."""
    if config is None:
        config = BallFilterConfig()

    filt = BallTemporalFilter(config)
    stages: dict[str, list[BallPosition]] = {}

    # Stage 0: Raw input
    sorted_pos = sorted(raw_positions, key=lambda p: p.frame_number)
    stages["0_raw"] = list(sorted_pos)

    # Stage 1: Detect exit ghost ranges (on raw data)
    ghost_ranges: list[tuple[int, int]] = []
    if config.enable_exit_ghost_removal:
        ghost_ranges = filt._detect_exit_ghost_ranges(sorted_pos)
        if ghost_ranges:
            logger.info(f"  Ghost ranges detected: {ghost_ranges}")

    # Stage 2: Segment pruning (with ghost awareness)
    if config.enable_segment_pruning:
        pruned = filt._prune_segments(sorted_pos, ghost_ranges=ghost_ranges)
    else:
        pruned = list(sorted_pos)
    stages["1_segment_pruned"] = pruned

    # Stage 3: Apply exit ghost removal
    if ghost_ranges:
        ghost_frames: set[int] = set()
        for start, end in ghost_ranges:
            for p in pruned:
                if start <= p.frame_number <= end:
                    ghost_frames.add(p.frame_number)
        after_ghost = [p for p in pruned if p.frame_number not in ghost_frames]
    else:
        after_ghost = list(pruned)
    stages["2_ghost_removed"] = after_ghost

    # Stage 4: Oscillation pruning
    if config.enable_oscillation_pruning:
        after_osc = filt._prune_oscillating(after_ghost)
    else:
        after_osc = list(after_ghost)
    stages["3_oscillation_pruned"] = after_osc

    # Stage 5: Outlier removal
    if config.enable_outlier_removal:
        after_outlier = filt._remove_outliers(after_osc)
    else:
        after_outlier = list(after_osc)
    stages["4_outlier_removed"] = after_outlier

    # Stage 6: Blip removal
    if config.enable_blip_removal:
        after_blip = filt._remove_trajectory_blips(after_outlier)
    else:
        after_blip = list(after_outlier)
    stages["5_blip_removed"] = after_blip

    # Stage 7: Re-prune (if outliers/blips were removed)
    outlier_count = len(after_osc) - len(after_outlier)
    blip_count = len(after_outlier) - len(after_blip)
    if outlier_count > 0 or blip_count > 0:
        after_reprune = list(after_blip)
        if config.enable_oscillation_pruning:
            after_reprune = filt._prune_oscillating(after_reprune)
        if config.enable_segment_pruning:
            after_reprune = filt._prune_segments(after_reprune)
    else:
        after_reprune = list(after_blip)
    stages["6_repruned"] = after_reprune

    # Stage 8: Interpolation
    if config.enable_interpolation:
        after_interp = filt._interpolate_missing(after_reprune)
    else:
        after_interp = list(after_reprune)
    stages["7_interpolated"] = after_interp

    return stages


def diagnose_rally(
    rally_id: str,
    raw_positions: list[BallPosition],
    gt_positions: list | None = None,
    video_width: int = 1920,
    video_height: int = 1080,
    video_fps: float = 30.0,
) -> None:
    """Run full diagnostics on a single rally."""
    prefix = rally_id[:8]
    key_frames = KEY_TIMESTAMPS.get(prefix, [])

    print(f"\n{'='*70}")
    print(f"Rally: {rally_id}")
    print(f"Raw positions: {len(raw_positions)}")
    if raw_positions:
        frames = sorted(p.frame_number for p in raw_positions)
        print(f"Frame range: {frames[0]} - {frames[-1]}")
    if key_frames:
        print(f"Key frames to inspect: {key_frames}")
    print(f"{'='*70}")

    # Run pipeline stages
    stages = run_pipeline_stages(raw_positions)

    for stage_name, positions in stages.items():
        print(f"\n--- {stage_name}: {len(positions)} positions ---")
        if positions:
            frames = sorted(p.frame_number for p in positions)
            print(f"  Frame range: {frames[0]} - {frames[-1]}")

        # Show what was removed vs previous stage
        stage_keys = list(stages.keys())
        stage_idx = stage_keys.index(stage_name)
        if stage_idx > 0:
            prev = stages[stage_keys[stage_idx - 1]]
            prev_frames = {p.frame_number for p in prev}
            curr_frames = {p.frame_number for p in positions}
            removed = prev_frames - curr_frames
            if removed:
                sorted_removed = sorted(removed)
                # Group consecutive frames into ranges
                ranges: list[str] = []
                start = sorted_removed[0]
                end = start
                for f in sorted_removed[1:]:
                    if f == end + 1:
                        end = f
                    else:
                        ranges.append(
                            f"{start}-{end}" if start != end else str(start)
                        )
                        start = end = f
                ranges.append(f"{start}-{end}" if start != end else str(start))
                print(f"  Removed {len(removed)} frames: {', '.join(ranges[:20])}")
                if len(ranges) > 20:
                    print(f"  ... and {len(ranges) - 20} more ranges")

        # Show key frame positions
        for kf in key_frames:
            nearby = find_positions_near_frame(positions, kf, window=5)
            if nearby:
                closest = min(nearby, key=lambda p: abs(p.frame_number - kf))
                print(
                    f"  Frame ~{kf}: "
                    f"f={closest.frame_number} "
                    f"pos=({closest.x:.3f}, {closest.y:.3f}) "
                    f"conf={closest.confidence:.2f}"
                )
            else:
                print(f"  Frame ~{kf}: NOT PRESENT")

    # If GT available, compute metrics at each stage
    if gt_positions:
        print(f"\n--- Metrics per stage (vs GT) ---")
        for stage_name, positions in stages.items():
            if not positions:
                print(f"  {stage_name}: no positions")
                continue
            best_offset, _ = find_optimal_frame_offset(
                gt_positions, positions, video_width, video_height
            )
            metrics = evaluate_ball_tracking(
                gt_positions,
                positions,
                video_width,
                video_height,
                video_fps,
            )
            print(
                f"  {stage_name}: "
                f"det={metrics.detection_rate:.1%} "
                f"match={metrics.match_rate:.1%} "
                f"err={metrics.mean_error_px:.1f}px "
                f"(offset={best_offset})"
            )


def main() -> None:
    cache = BallRawCache()
    cached_ids = cache.list_cached()
    print(f"Cached rallies: {len(cached_ids)}")

    # Load GT data for rallies that have it
    try:
        gt_rallies = load_labeled_rallies()
        gt_by_rally = {r.rally_id: r for r in gt_rallies}
        print(f"GT rallies loaded: {len(gt_rallies)}")
    except Exception as e:
        print(f"Could not load GT rallies: {e}")
        gt_by_rally = {}

    # Process each affected rally
    target_prefixes = list(KEY_TIMESTAMPS.keys())

    for cached_id in cached_ids:
        prefix = cached_id[:8]
        if prefix not in target_prefixes:
            continue

        cached_data = cache.get(cached_id)
        if not cached_data:
            print(f"Failed to load cache for {cached_id}")
            continue

        gt_pos = None
        video_width = cached_data.video_width
        video_height = cached_data.video_height
        video_fps = cached_data.video_fps

        if cached_id in gt_by_rally:
            rally = gt_by_rally[cached_id]
            gt_pos = [
                p
                for p in rally.ground_truth.positions
                if p.label == "ball"
            ]

        diagnose_rally(
            cached_id,
            cached_data.raw_ball_positions,
            gt_pos,
            video_width,
            video_height,
            video_fps,
        )

    # Also check rallies that aren't cached but have key prefixes
    for prefix in target_prefixes:
        found = any(cid[:8] == prefix for cid in cached_ids)
        if not found:
            # Try to find in GT rallies
            for rally in gt_by_rally.values():
                if rally.rally_id[:8] == prefix:
                    print(f"\nRally {rally.rally_id} not in cache, using DB predictions")
                    if rally.predictions and rally.predictions.ball_positions:
                        gt_pos = [
                            p
                            for p in rally.ground_truth.positions
                            if p.label == "ball"
                        ]
                        diagnose_rally(
                            rally.rally_id,
                            rally.predictions.ball_positions,
                            gt_pos,
                            rally.video_width,
                            rally.video_height,
                            rally.video_fps,
                        )
                    break
            else:
                print(f"\nRally with prefix {prefix} not found in cache or DB")


if __name__ == "__main__":
    main()
