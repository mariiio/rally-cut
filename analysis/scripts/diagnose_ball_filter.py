"""Diagnose ball filter pipeline stage by stage for a rally.

Shows how many detections survive each filter stage to identify which
filter is killing real ball positions.

Usage:
    uv run python scripts/diagnose_ball_filter.py <video_path> [--start-ms N] [--end-ms N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.tracking.ball_filter import (
    BallFilterConfig,
    BallTemporalFilter,
    get_wasb_filter_config,
)
from rallycut.tracking.ball_tracker import BallPosition, create_ball_tracker


def diagnose(video_path: str, start_ms: int | None, end_ms: int | None) -> None:
    print(f"Video: {video_path}")
    print(f"Range: {start_ms}ms - {end_ms}ms")

    # Run WASB inference (no filtering)
    print("\n1. Running WASB inference (raw, no filter)...")
    tracker = create_ball_tracker()
    raw_result = tracker.track_video(
        video_path, start_ms=start_ms, end_ms=end_ms,
        enable_filtering=False,
    )
    raw = raw_result.positions
    confident_raw = [p for p in raw if p.confidence > 0]
    print(f"   Raw: {len(raw)} total, {len(confident_raw)} confident ({len(confident_raw)/len(raw)*100:.1f}%)")

    # Now run each filter stage individually
    config = get_wasb_filter_config()
    print(f"\n2. Filter config: segment_pruning={config.enable_segment_pruning}, "
          f"exit_ghost={config.enable_exit_ghost_removal}, "
          f"outlier={config.enable_outlier_removal}, "
          f"oscillation={config.enable_oscillation_pruning}, "
          f"blip={config.enable_blip_removal}, "
          f"interp={config.enable_interpolation}")

    # Stage by stage
    positions = list(raw)

    # Stage 0: motion energy
    if config.enable_motion_energy_filter:
        f = BallTemporalFilter(BallFilterConfig(
            enable_motion_energy_filter=True,
            enable_segment_pruning=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        ))
        after = f.filter_batch(positions)
        c = sum(1 for p in after if p.confidence > 0)
        print(f"\n   After motion energy: {c} confident")
    else:
        print("\n   Motion energy: DISABLED")

    # Stage 1: segment pruning only
    f1 = BallTemporalFilter(BallFilterConfig(
        enable_motion_energy_filter=False,
        enable_segment_pruning=True,
        segment_jump_threshold=config.segment_jump_threshold,
        min_segment_frames=config.min_segment_frames,
        min_output_confidence=config.min_output_confidence,
        enable_exit_ghost_removal=False,
        enable_oscillation_pruning=False,
        enable_outlier_removal=False,
        enable_blip_removal=False,
        enable_interpolation=False,
    ))
    after_prune = f1.filter_batch(list(raw))
    c_prune = sum(1 for p in after_prune if p.confidence > 0)
    print(f"   After segment pruning only: {c_prune} confident (lost {len(confident_raw) - c_prune})")

    # Show segments
    import numpy as np
    confident = [p for p in raw if p.confidence > 0]
    segments: list[list[BallPosition]] = [[confident[0]]] if confident else []
    for i in range(1, len(confident)):
        prev = confident[i-1]
        curr = confident[i]
        dist = float(np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2))
        gap = curr.frame_number - prev.frame_number
        if dist > config.segment_jump_threshold or gap > 15:
            segments.append([curr])
        else:
            segments[-1].append(curr)

    print(f"\n   Raw segments ({len(segments)}):")
    for i, seg in enumerate(segments):
        cx = sum(p.x for p in seg) / len(seg)
        cy = sum(p.y for p in seg) / len(seg)
        print(f"     Seg {i}: frames {seg[0].frame_number}-{seg[-1].frame_number} "
              f"({len(seg)} dets), centroid=({cx:.3f}, {cy:.3f})")

    # Stage 2: exit ghost detection (detect ranges manually on raw data)
    ghost_filter = BallTemporalFilter(BallFilterConfig(
        enable_motion_energy_filter=False,
        enable_segment_pruning=False,
        enable_exit_ghost_removal=True,
        exit_edge_zone=config.exit_edge_zone,
        exit_approach_frames=config.exit_approach_frames,
        exit_min_approach_speed=config.exit_min_approach_speed,
        exit_max_ghost_frames=config.exit_max_ghost_frames,
        enable_oscillation_pruning=False,
        enable_outlier_removal=False,
        enable_blip_removal=False,
        enable_interpolation=False,
    ))
    ghost_ranges = ghost_filter._detect_exit_ghost_ranges(raw)
    if ghost_ranges:
        print("\n   Exit ghost ranges detected:")
        for start, end in ghost_ranges:
            print(f"     Frames {start}-{end}")
        ghost_frames = set()
        for s, e in ghost_ranges:
            for p in raw:
                if s <= p.frame_number <= e:
                    ghost_frames.add(p.frame_number)
        print(f"   Total ghost frames: {len(ghost_frames)}")
    else:
        print("\n   Exit ghost: no ghost ranges detected")

    # Stage 3: outlier removal only
    f3 = BallTemporalFilter(BallFilterConfig(
        enable_motion_energy_filter=False,
        enable_segment_pruning=False,
        enable_exit_ghost_removal=False,
        enable_oscillation_pruning=False,
        enable_outlier_removal=True,
        enable_blip_removal=False,
        enable_interpolation=False,
    ))
    after_outlier = f3.filter_batch(list(raw))
    c_outlier = sum(1 for p in after_outlier if p.confidence > 0)
    print(f"\n   After outlier removal only: {c_outlier} confident (lost {len(confident_raw) - c_outlier})")

    # Full pipeline
    print("\n3. Full WASB filter pipeline:")
    f_full = BallTemporalFilter(config)
    after_full = f_full.filter_batch(list(raw))
    c_full = sum(1 for p in after_full if p.confidence > 0)
    print(f"   After full pipeline: {c_full} confident (lost {len(confident_raw) - c_full})")

    # Show what survived
    survived = [p for p in after_full if p.confidence > 0]
    if survived:
        print("\n   Surviving detections:")
        for p in survived[:20]:
            print(f"     frame={p.frame_number:4d}  x={p.x:.3f}  y={p.y:.3f}  conf={p.confidence:.3f}")
        if len(survived) > 20:
            print(f"     ... ({len(survived) - 20} more)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    args = parser.parse_args()
    diagnose(args.video, args.start_ms, args.end_ms)


if __name__ == "__main__":
    main()
