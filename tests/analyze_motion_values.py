#!/usr/bin/env python3
"""Analyze raw motion detection values to understand the distribution.

This helps diagnose why motion detection isn't filtering effectively.
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from rallycut.core.config import get_config, reset_config
from rallycut.core.video import Video


def parse_time(time_str: str) -> float:
    """Parse time string like '1:30' or '0:12.5' to seconds."""
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    return float(time_str)


def analyze_motion_distribution(video_path: Path, gt_rallies: list) -> dict:
    """Analyze raw motion values and compare rally vs non-rally periods."""
    config = get_config()

    analysis_size = config.motion.analysis_size
    stride = config.two_pass.motion_stride

    motion_values = []  # (frame_idx, motion_ratio, is_rally)

    with Video(video_path) as video:
        fps = video.info.fps
        total_frames = video.info.frame_count

        prev_gray = None

        for frame_idx, frame in video.iter_frames(end_frame=total_frames, step=stride):
            # Same processing as MotionDetector
            small = cv2.resize(frame, analysis_size, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            motion_ratio = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
                motion_ratio = np.count_nonzero(thresh) / thresh.size

            prev_gray = gray

            # Check if this frame is in a rally
            time_sec = frame_idx / fps
            is_rally = any(r['start'] <= time_sec <= r['end'] for r in gt_rallies)

            motion_values.append((frame_idx, motion_ratio, is_rally))

    return {
        'fps': fps,
        'total_frames': total_frames,
        'motion_values': motion_values,
    }


def print_analysis(video_name: str, data: dict, thresholds: dict) -> None:
    """Print motion value analysis."""
    fps = data['fps']
    motion_values = data['motion_values']

    rally_motions = [m for _, m, is_rally in motion_values if is_rally and m > 0]
    dead_motions = [m for _, m, is_rally in motion_values if not is_rally and m > 0]

    print(f"\n{'='*70}")
    print(f"Video: {video_name}")
    print(f"{'='*70}")

    # Distribution stats
    all_motions = [m for _, m, _ in motion_values if m > 0]

    print(f"\n--- Motion Value Distribution ---")
    print(f"Total samples: {len(motion_values)}")
    if all_motions:
        print(f"Overall: min={min(all_motions):.4f}, max={max(all_motions):.4f}, "
              f"mean={np.mean(all_motions):.4f}, median={np.median(all_motions):.4f}")

    if rally_motions:
        print(f"Rally frames: min={min(rally_motions):.4f}, max={max(rally_motions):.4f}, "
              f"mean={np.mean(rally_motions):.4f}, median={np.median(rally_motions):.4f}")

    if dead_motions:
        print(f"Dead time: min={min(dead_motions):.4f}, max={max(dead_motions):.4f}, "
              f"mean={np.mean(dead_motions):.4f}, median={np.median(dead_motions):.4f}")

    # Current threshold analysis
    print(f"\n--- Current Thresholds ---")
    print(f"high_threshold: {thresholds['high']}")
    print(f"low_threshold: {thresholds['low']}")

    above_high = sum(1 for _, m, _ in motion_values if m >= thresholds['high'])
    above_low = sum(1 for _, m, _ in motion_values if m >= thresholds['low'])

    print(f"Frames above high_threshold: {above_high}/{len(motion_values)} ({above_high/len(motion_values)*100:.1f}%)")
    print(f"Frames above low_threshold: {above_low}/{len(motion_values)} ({above_low/len(motion_values)*100:.1f}%)")

    # Threshold sweep to find optimal values
    print(f"\n--- Threshold Sweep ---")
    print(f"{'Threshold':>10} | {'Rally Recall':>12} | {'Dead Filtered':>12} | {'Filter Ratio':>12}")
    print("-" * 55)

    for thresh in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        # How many rally frames would we catch?
        rally_caught = sum(1 for _, m, is_rally in motion_values if is_rally and m >= thresh)
        rally_total = sum(1 for _, _, is_rally in motion_values if is_rally)
        rally_recall = rally_caught / rally_total if rally_total else 0

        # How many dead time frames would be filtered?
        dead_filtered = sum(1 for _, m, is_rally in motion_values if not is_rally and m < thresh)
        dead_total = sum(1 for _, _, is_rally in motion_values if not is_rally)
        dead_filter_rate = dead_filtered / dead_total if dead_total else 0

        # Overall filter ratio
        total_filtered = sum(1 for _, m, _ in motion_values if m < thresh)
        filter_ratio = total_filtered / len(motion_values) if motion_values else 0

        print(f"{thresh:>10.3f} | {rally_recall*100:>11.1f}% | {dead_filter_rate*100:>11.1f}% | {filter_ratio*100:>11.1f}%")

    # Show sample values at different time points
    print(f"\n--- Sample Motion Values by Time ---")
    print(f"{'Time':>8} | {'Motion':>8} | {'Is Rally':>10} | {'Above Thresh':>12}")
    print("-" * 50)

    # Sample every 10 seconds
    for _, m, is_rally in motion_values[::int(10 * fps / 32)]:  # ~every 10 sec given stride 32
        time_sec = _ / fps
        above = "YES" if m >= thresholds['low'] else "no"
        rally_str = "RALLY" if is_rally else "dead"
        print(f"{time_sec:>7.1f}s | {m:>8.4f} | {rally_str:>10} | {above:>12}")


def main():
    reset_config()
    config = get_config()

    fixtures_dir = Path(__file__).parent / "fixtures"
    gt_path = fixtures_dir / "ground_truth.json"

    with open(gt_path) as f:
        ground_truth = json.load(f)

    thresholds = {
        'high': config.two_pass.motion_high_threshold,
        'low': config.two_pass.motion_low_threshold,
    }

    for video_name, video_data in ground_truth.items():
        video_path = fixtures_dir / video_name
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue

        # Convert rally times
        rallies = []
        for rally in video_data['rallies']:
            rallies.append({
                'start': parse_time(rally['start']),
                'end': parse_time(rally['end']),
            })

        print(f"\nAnalyzing {video_name}...")
        data = analyze_motion_distribution(video_path, rallies)
        print_analysis(video_name, data, thresholds)


if __name__ == "__main__":
    main()
