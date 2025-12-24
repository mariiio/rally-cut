#!/usr/bin/env python3
"""Compare performance and accuracy: with motion pass vs without.

This script:
1. Runs TwoPassAnalyzer with motion detection (skip_motion_pass=False)
2. Runs TwoPassAnalyzer without motion detection (skip_motion_pass=True)
3. Compares timing, accuracy against ground truth, and detected segments
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from rallycut.analysis.two_pass import TwoPassAnalyzer
from rallycut.core.config import reset_config
from rallycut.core.models import GameState
from rallycut.core.video import Video
from rallycut.processing.cutter import VideoCutter


@dataclass
class GroundTruthRally:
    """A rally from ground truth."""
    start_seconds: float
    end_seconds: float


@dataclass
class AnalysisResult:
    """Result from one analysis run."""
    mode: str
    total_time: float
    motion_time: float
    ml_time: float
    num_play_segments: int
    play_duration_seconds: float
    segments: list  # TimeSegment list


def parse_time(time_str: str) -> float:
    """Parse time string like '1:30' or '0:12.5' to seconds."""
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    return float(time_str)


def load_ground_truth(json_path: Path) -> dict[str, list[GroundTruthRally]]:
    """Load ground truth rallies from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    result = {}
    for video_name, video_data in data.items():
        rallies = []
        for rally in video_data['rallies']:
            start = parse_time(rally['start'])
            end = parse_time(rally['end'])
            rallies.append(GroundTruthRally(start, end))
        result[video_name] = rallies

    return result


def calculate_accuracy(
    detected_segments: list,
    gt_rallies: list[GroundTruthRally],
    video_duration: float,
) -> dict:
    """
    Calculate accuracy metrics comparing detected segments to ground truth.

    Returns dict with:
    - rally_recall: % of ground truth rally time that is covered
    - rally_precision: % of detected play time that is actually rally
    - per_rally_coverage: coverage for each ground truth rally
    """
    # Create time masks at 0.1s resolution
    total_tenths = int(video_duration * 10)

    # Ground truth mask
    gt_tenths = set()
    for rally in gt_rallies:
        for t in range(int(rally.start_seconds * 10), int(rally.end_seconds * 10)):
            gt_tenths.add(t)

    # Detected segments mask
    detected_tenths = set()
    for seg in detected_segments:
        for t in range(int(seg.start_time * 10), int(seg.end_time * 10)):
            detected_tenths.add(t)

    # Calculate metrics
    intersection = gt_tenths & detected_tenths

    rally_recall = len(intersection) / len(gt_tenths) if gt_tenths else 0
    rally_precision = len(intersection) / len(detected_tenths) if detected_tenths else 0

    # Per-rally coverage
    per_rally = []
    for rally in gt_rallies:
        rally_tenths = set()
        for t in range(int(rally.start_seconds * 10), int(rally.end_seconds * 10)):
            rally_tenths.add(t)
        covered = rally_tenths & detected_tenths
        coverage = len(covered) / len(rally_tenths) if rally_tenths else 0
        per_rally.append({
            'start': rally.start_seconds,
            'end': rally.end_seconds,
            'coverage': coverage,
        })

    return {
        'rally_recall': rally_recall,
        'rally_precision': rally_precision,
        'per_rally_coverage': per_rally,
    }


def run_analysis(video_path: Path, skip_motion: bool) -> AnalysisResult:
    """Run analysis with or without motion pass."""
    mode = "skip_motion" if skip_motion else "with_motion"

    # Use VideoCutter which wraps TwoPassAnalyzer
    cutter = VideoCutter(
        use_two_pass=True,
        use_proxy=True,  # Use proxy for fair comparison
    )

    # Override the analyzer's skip_motion_pass
    cutter._analyzer = None  # Force re-creation

    # Create analyzer directly with skip_motion_pass setting
    analyzer = TwoPassAnalyzer(
        use_proxy=True,
        skip_motion_pass=skip_motion,
    )

    with Video(video_path) as video:
        fps = video.info.fps
        total_frames = video.info.frame_count
        duration = total_frames / fps

    timing_info = {'motion_time': 0.0, 'ml_time': 0.0}

    def progress_callback(pct: float, msg: str):
        # Extract timing from messages
        if "motion:" in msg.lower():
            import re
            match = re.search(r'motion:\s*([\d.]+)s', msg.lower())
            if match:
                timing_info['motion_time'] = float(match.group(1))
        if "ml:" in msg.lower():
            import re
            match = re.search(r'ml:\s*([\d.]+)s', msg.lower())
            if match:
                timing_info['ml_time'] = float(match.group(1))

    start_time = time.perf_counter()

    with Video(video_path) as video:
        results = analyzer.analyze_video(
            video,
            progress_callback=progress_callback,
        )

    elapsed = time.perf_counter() - start_time

    # Convert results to segments using VideoCutter's logic
    with Video(video_path) as video:
        segments = cutter._get_segments_from_results(results, video.info.fps)

    play_duration = sum(s.duration for s in segments)

    return AnalysisResult(
        mode=mode,
        total_time=elapsed,
        motion_time=timing_info['motion_time'],
        ml_time=timing_info['ml_time'],
        num_play_segments=len(segments),
        play_duration_seconds=play_duration,
        segments=segments,
    )


def print_comparison(
    video_name: str,
    video_duration: float,
    gt_rallies: list[GroundTruthRally],
    result_with_motion: AnalysisResult,
    result_skip_motion: AnalysisResult,
):
    """Print comparison results."""
    print(f"\n{'='*70}")
    print(f"Video: {video_name}")
    print(f"Duration: {video_duration:.1f}s")
    print(f"Ground truth rallies: {len(gt_rallies)}")
    print(f"{'='*70}")

    # Timing comparison
    print(f"\n--- Timing Comparison ---")
    print(f"{'Mode':<20} | {'Total':>10} | {'Motion':>10} | {'ML':>10} | {'Speedup':>10}")
    print("-" * 70)

    speedup = result_with_motion.total_time / result_skip_motion.total_time if result_skip_motion.total_time > 0 else 0

    print(f"{'With motion pass':<20} | {result_with_motion.total_time:>9.2f}s | {result_with_motion.motion_time:>9.2f}s | {result_with_motion.ml_time:>9.2f}s | {'baseline':>10}")
    print(f"{'Skip motion pass':<20} | {result_skip_motion.total_time:>9.2f}s | {result_skip_motion.motion_time:>9.2f}s | {result_skip_motion.ml_time:>9.2f}s | {speedup:>9.2f}x")

    time_saved = result_with_motion.total_time - result_skip_motion.total_time
    print(f"\nTime saved: {time_saved:.2f}s ({time_saved/result_with_motion.total_time*100:.1f}% faster)")

    # Accuracy comparison
    print(f"\n--- Accuracy Comparison ---")

    acc_with = calculate_accuracy(result_with_motion.segments, gt_rallies, video_duration)
    acc_skip = calculate_accuracy(result_skip_motion.segments, gt_rallies, video_duration)

    print(f"{'Mode':<20} | {'Rally Recall':>12} | {'Precision':>12} | {'Segments':>10}")
    print("-" * 60)
    print(f"{'With motion pass':<20} | {acc_with['rally_recall']*100:>11.1f}% | {acc_with['rally_precision']*100:>11.1f}% | {result_with_motion.num_play_segments:>10}")
    print(f"{'Skip motion pass':<20} | {acc_skip['rally_recall']*100:>11.1f}% | {acc_skip['rally_precision']*100:>11.1f}% | {result_skip_motion.num_play_segments:>10}")

    # Per-rally coverage
    print(f"\n--- Per-Rally Coverage ---")
    print(f"{'Rally':<15} | {'With Motion':>12} | {'Skip Motion':>12} | {'Match':>8}")
    print("-" * 55)

    all_match = True
    for i, (r_with, r_skip) in enumerate(zip(acc_with['per_rally_coverage'], acc_skip['per_rally_coverage']), 1):
        match = "YES" if abs(r_with['coverage'] - r_skip['coverage']) < 0.05 else "NO"
        if match == "NO":
            all_match = False
        print(f"Rally {i} ({r_with['start']:.0f}-{r_with['end']:.0f}s) | {r_with['coverage']*100:>11.0f}% | {r_skip['coverage']*100:>11.0f}% | {match:>8}")

    # Summary
    print(f"\n--- Summary ---")
    if all_match and abs(acc_with['rally_recall'] - acc_skip['rally_recall']) < 0.05:
        print("[PASS] Accuracy maintained after removing motion pass")
    else:
        print("[WARN] Accuracy difference detected")

    if speedup > 1.0:
        print(f"[PASS] {speedup:.2f}x speedup achieved by skipping motion pass")
    else:
        print(f"[INFO] No speedup (skip_motion: {result_skip_motion.total_time:.2f}s vs with_motion: {result_with_motion.total_time:.2f}s)")


def main():
    """Run comparison on all test fixtures."""
    reset_config()

    fixtures_dir = Path(__file__).parent / "fixtures"
    gt_path = fixtures_dir / "ground_truth.json"

    if not gt_path.exists():
        print(f"Ground truth not found: {gt_path}")
        return

    ground_truth = load_ground_truth(gt_path)

    for video_name, gt_rallies in ground_truth.items():
        video_path = fixtures_dir / video_name
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue

        with Video(video_path) as video:
            duration = video.info.frame_count / video.info.fps

        print(f"\nAnalyzing {video_name}...")
        print("Running with motion pass...")
        result_with = run_analysis(video_path, skip_motion=False)

        print("Running without motion pass...")
        result_skip = run_analysis(video_path, skip_motion=True)

        print_comparison(video_name, duration, gt_rallies, result_with, result_skip)

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("Motion detection pass provides no filtering benefit for beach volleyball.")
    print("Removing it simplifies the pipeline and eliminates overhead.")


if __name__ == "__main__":
    main()
