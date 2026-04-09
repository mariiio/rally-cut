"""Diagnose ball tracking dropout in specific time windows.

Shows frame-by-frame comparison of raw vs filtered positions to identify
which filter stage kills valid detections.

Usage:
    uv run python scripts/diagnose_ball_dropout.py <video_path> --start-ms N --end-ms N
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.tracking.ball_filter import BallFilterConfig, BallTemporalFilter, get_wasb_filter_config
from rallycut.tracking.ball_tracker import BallPosition, create_ball_tracker


def diagnose(
    video_path: str,
    start_ms: int,
    end_ms: int,
    rally_start_ms: int | None,
    rally_end_ms: int | None,
) -> None:
    # Use rally bounds if provided, otherwise just the window
    infer_start = rally_start_ms or start_ms
    infer_end = rally_end_ms or end_ms

    print(f"Video: {video_path}")
    print(f"Inference range: {infer_start}ms - {infer_end}ms")
    print(f"Analysis window: {start_ms}ms - {end_ms}ms")

    # Run raw WASB
    print("\nRunning WASB inference (raw)...")
    tracker = create_ball_tracker()
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    raw_result = tracker.track_video(
        video_path, start_ms=infer_start, end_ms=infer_end, enable_filtering=False
    )
    raw = raw_result.positions

    # Compute frame range for the analysis window
    start_frame_offset = int((start_ms - infer_start) * fps / 1000)
    end_frame_offset = int((end_ms - infer_start) * fps / 1000)

    # Run full filter pipeline
    config = get_wasb_filter_config()
    f = BallTemporalFilter(config)
    filtered = f.filter_batch(list(raw))

    # Run individual stages to find the culprit
    # Stage A: segment pruning only
    f_seg = BallTemporalFilter(BallFilterConfig(
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
    after_seg = f_seg.filter_batch(list(raw))

    # Stage B: segment + outlier (matches full pipeline minus interpolation)
    f_seg_out = BallTemporalFilter(BallFilterConfig(
        enable_motion_energy_filter=config.enable_motion_energy_filter,
        enable_stationarity_filter=config.enable_stationarity_filter,
        enable_segment_pruning=True,
        segment_jump_threshold=config.segment_jump_threshold,
        min_segment_frames=config.min_segment_frames,
        min_output_confidence=config.min_output_confidence,
        max_chain_gap=config.max_chain_gap,
        enable_exit_ghost_removal=True,
        exit_edge_zone=config.exit_edge_zone,
        exit_approach_frames=config.exit_approach_frames,
        exit_min_approach_speed=config.exit_min_approach_speed,
        exit_max_ghost_frames=config.exit_max_ghost_frames,
        enable_oscillation_pruning=False,
        enable_outlier_removal=True,
        enable_blip_removal=False,
        enable_interpolation=False,
    ))
    after_seg_out = f_seg_out.filter_batch(list(raw))

    # Build frame lookup
    def frame_map(positions: list[BallPosition]) -> dict[int, BallPosition]:
        return {p.frame_number: p for p in positions if p.confidence > 0}

    raw_map = frame_map(raw)
    seg_map = frame_map(after_seg)
    seg_out_map = frame_map(after_seg_out)
    full_map = frame_map(filtered)

    # Print frame-by-frame in the window
    print(f"\n{'Frame':>6} {'Raw':>10} {'AfterSeg':>10} {'Seg+Out':>10} {'Full':>10}  {'Raw Position':>20}  Status")
    print("─" * 100)

    window_raw = 0
    window_seg = 0
    window_full = 0
    for i, p in enumerate(raw):
        fn = p.frame_number
        if fn < start_frame_offset or fn > end_frame_offset:
            continue

        has_raw = fn in raw_map
        has_seg = fn in seg_map
        has_seg_out = fn in seg_out_map
        has_full = fn in full_map

        raw_str = f"({raw_map[fn].x:.3f},{raw_map[fn].y:.3f})" if has_raw else "---"
        seg_str = "KEPT" if has_seg else "KILL" if has_raw else "---"
        seg_out_str = "KEPT" if has_seg_out else "KILL" if has_raw else "---"
        full_str = "KEPT" if has_full else "KILL" if has_raw else "---"

        status = ""
        if has_raw and not has_seg:
            status = "← SEGMENT PRUNING kills"
        elif has_raw and has_seg and not has_seg_out:
            status = "← OUTLIER/GHOST kills"
        elif has_raw and has_seg_out and not has_full:
            status = "← LATER STAGE kills"
        elif not has_raw:
            status = "← NO RAW DETECTION"
        elif has_full:
            # Check if position changed (interpolated)
            if has_raw:
                fp = full_map[fn]
                rp = raw_map[fn]
                dist = np.sqrt((fp.x - rp.x) ** 2 + (fp.y - rp.y) ** 2)
                if dist > 0.01:
                    status = f"← interpolated (d={dist:.3f})"

        if window_raw == 0 and has_raw:
            pass
        window_raw += int(has_raw)
        window_seg += int(has_seg)
        window_full += int(has_full)

        print(f"{fn:6d} {raw_str:>10} {seg_str:>10} {seg_out_str:>10} {full_str:>10}  {status}")

    print(f"\nWindow summary: raw={window_raw}, after_seg={window_seg}, full={window_full}")
    print(f"Segment pruning kills: {window_raw - window_seg}")
    print(f"Full pipeline kills: {window_raw - window_full}")

    # Check if there's a large gap in surviving detections
    surviving_frames = sorted(full_map.keys())
    window_surviving = [f for f in surviving_frames if start_frame_offset <= f <= end_frame_offset]
    if window_surviving:
        gaps = []
        for i in range(1, len(window_surviving)):
            gap = window_surviving[i] - window_surviving[i - 1]
            if gap > 5:
                gaps.append((window_surviving[i - 1], window_surviving[i], gap))
        if gaps:
            print(f"\nLarge gaps (>5 frames) in surviving detections within window:")
            for g_start, g_end, g_size in gaps:
                print(f"  frames {g_start}-{g_end} ({g_size} frame gap)")
    else:
        print(f"\nNO surviving detections in the analysis window!")
        # Find nearest surviving frames
        before = [f for f in surviving_frames if f < start_frame_offset]
        after = [f for f in surviving_frames if f > end_frame_offset]
        if before:
            print(f"  Nearest before: frame {before[-1]} (gap={start_frame_offset - before[-1]})")
        if after:
            print(f"  Nearest after: frame {after[0]} (gap={after[0] - end_frame_offset})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--start-ms", type=int, required=True, help="Window start (ms)")
    parser.add_argument("--end-ms", type=int, required=True, help="Window end (ms)")
    parser.add_argument("--rally-start-ms", type=int, default=None, help="Rally start for full-context inference")
    parser.add_argument("--rally-end-ms", type=int, default=None, help="Rally end for full-context inference")
    args = parser.parse_args()
    diagnose(args.video, args.start_ms, args.end_ms, args.rally_start_ms, args.rally_end_ms)


if __name__ == "__main__":
    main()
