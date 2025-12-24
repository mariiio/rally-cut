#!/usr/bin/env python3
"""Profile first pass (motion detection) and compare against ground truth.

Measures:
1. Speed of motion detection
2. Coverage: What % of ground truth rally frames are covered by motion regions?
3. Precision: What % of motion region frames are actually rally frames?
4. Filter ratio: How much of the video is filtered out (not sent to ML)?
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from rallycut.analysis.motion_detector import MotionDetector
from rallycut.analysis.two_pass import TwoPassAnalyzer
from rallycut.core.config import get_config, reset_config
from rallycut.core.models import GameState
from rallycut.core.video import Video


@dataclass
class GroundTruthRally:
    """A rally from ground truth."""
    start_seconds: float
    end_seconds: float


@dataclass
class MotionRegion:
    """A detected motion region."""
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float


@dataclass
class EvaluationResult:
    """Evaluation metrics for a single video."""
    video_name: str
    video_duration_seconds: float
    fps: float

    # Timing
    motion_detection_time: float

    # Ground truth
    num_gt_rallies: int
    gt_rally_seconds: float

    # Motion detection results
    num_motion_regions: int
    motion_region_seconds: float

    # Coverage metrics
    rally_coverage: float  # What % of rally frames are in motion regions
    motion_precision: float  # What % of motion frames are rally frames
    filter_ratio: float  # What % of video is filtered out

    # Per-rally coverage
    per_rally_coverage: list[tuple[float, float, float]]  # (start, end, coverage %)


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


def run_motion_detection(video_path: Path, fps: float) -> tuple[list[MotionRegion], float]:
    """Run motion detection and return regions with timing."""
    config = get_config()

    # Create motion detector with two_pass thresholds (what TwoPassAnalyzer uses)
    detector = MotionDetector(
        high_motion_threshold=config.two_pass.motion_high_threshold,
        low_motion_threshold=config.two_pass.motion_low_threshold,
    )

    with Video(video_path) as video:
        start_time = time.perf_counter()
        results = detector.analyze_video(video, stride=config.two_pass.motion_stride)
        elapsed = time.perf_counter() - start_time

    # Convert results to motion regions (merge consecutive PLAY segments)
    regions = []
    current_start = None
    current_end = None

    for result in results:
        if result.state == GameState.PLAY:
            if current_start is None:
                current_start = result.start_frame
            current_end = result.end_frame
        else:
            if current_start is not None:
                regions.append(MotionRegion(
                    start_frame=current_start,
                    end_frame=current_end,
                    start_seconds=current_start / fps,
                    end_seconds=current_end / fps,
                ))
                current_start = None
                current_end = None

    if current_start is not None:
        regions.append(MotionRegion(
            start_frame=current_start,
            end_frame=current_end,
            start_seconds=current_start / fps,
            end_seconds=current_end / fps,
        ))

    return regions, elapsed


def run_two_pass_motion(video_path: Path, fps: float) -> tuple[list[MotionRegion], float]:
    """Run motion detection via TwoPassAnalyzer to get actual regions with padding."""
    config = get_config()

    analyzer = TwoPassAnalyzer()

    with Video(video_path) as video:
        total_frames = video.info.frame_count
        fps = video.info.fps

        start_time = time.perf_counter()
        raw_regions = analyzer._detect_motion_regions(
            video, fps, total_frames, progress_callback=None
        )
        elapsed = time.perf_counter() - start_time

    # Convert to our MotionRegion format
    regions = [
        MotionRegion(
            start_frame=r.start_frame,
            end_frame=r.end_frame,
            start_seconds=r.start_frame / fps,
            end_seconds=r.end_frame / fps,
        )
        for r in raw_regions
    ]

    return regions, elapsed


def calculate_coverage(
    gt_rallies: list[GroundTruthRally],
    motion_regions: list[MotionRegion],
    fps: float,
    total_frames: int,
) -> tuple[float, float, float, list[tuple[float, float, float]]]:
    """
    Calculate coverage metrics.

    Returns:
        (rally_coverage, motion_precision, filter_ratio, per_rally_coverage)
    """
    # Create frame-level masks
    total_duration = total_frames / fps

    # Ground truth mask (which seconds are rallies)
    gt_seconds = set()
    for rally in gt_rallies:
        for s in range(int(rally.start_seconds * 10), int(rally.end_seconds * 10)):
            gt_seconds.add(s / 10)  # 0.1 second resolution

    # Motion region mask
    motion_seconds = set()
    for region in motion_regions:
        for s in range(int(region.start_seconds * 10), int(region.end_seconds * 10)):
            motion_seconds.add(s / 10)

    # Calculate metrics
    intersection = gt_seconds & motion_seconds

    rally_coverage = len(intersection) / len(gt_seconds) if gt_seconds else 0
    motion_precision = len(intersection) / len(motion_seconds) if motion_seconds else 0

    # Filter ratio: how much is NOT in motion regions
    total_tenths = int(total_duration * 10)
    filter_ratio = 1.0 - (len(motion_seconds) / total_tenths) if total_tenths else 0

    # Per-rally coverage
    per_rally = []
    for rally in gt_rallies:
        rally_tenths = set()
        for s in range(int(rally.start_seconds * 10), int(rally.end_seconds * 10)):
            rally_tenths.add(s / 10)

        covered = rally_tenths & motion_seconds
        coverage = len(covered) / len(rally_tenths) if rally_tenths else 0
        per_rally.append((rally.start_seconds, rally.end_seconds, coverage))

    return rally_coverage, motion_precision, filter_ratio, per_rally


def evaluate_video(
    video_path: Path,
    gt_rallies: list[GroundTruthRally],
    use_two_pass: bool = True,
) -> EvaluationResult:
    """Evaluate motion detection on a single video."""
    with Video(video_path) as video:
        fps = video.info.fps
        total_frames = video.info.frame_count
        duration = total_frames / fps

    # Run motion detection
    if use_two_pass:
        motion_regions, elapsed = run_two_pass_motion(video_path, fps)
    else:
        motion_regions, elapsed = run_motion_detection(video_path, fps)

    # Calculate ground truth duration
    gt_rally_seconds = sum(r.end_seconds - r.start_seconds for r in gt_rallies)

    # Calculate motion region duration
    motion_seconds = sum(r.end_seconds - r.start_seconds for r in motion_regions)

    # Calculate coverage
    rally_coverage, motion_precision, filter_ratio, per_rally = calculate_coverage(
        gt_rallies, motion_regions, fps, total_frames
    )

    return EvaluationResult(
        video_name=video_path.name,
        video_duration_seconds=duration,
        fps=fps,
        motion_detection_time=elapsed,
        num_gt_rallies=len(gt_rallies),
        gt_rally_seconds=gt_rally_seconds,
        num_motion_regions=len(motion_regions),
        motion_region_seconds=motion_seconds,
        rally_coverage=rally_coverage,
        motion_precision=motion_precision,
        filter_ratio=filter_ratio,
        per_rally_coverage=per_rally,
    )


def print_results(result: EvaluationResult) -> None:
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Video: {result.video_name}")
    print(f"{'='*60}")
    print(f"Duration: {result.video_duration_seconds:.1f}s @ {result.fps:.1f} fps")
    print(f"Motion detection time: {result.motion_detection_time:.3f}s")
    print(f"Processing speed: {result.video_duration_seconds / result.motion_detection_time:.1f}x realtime")

    print(f"\n--- Ground Truth ---")
    print(f"Rallies: {result.num_gt_rallies}")
    print(f"Total rally time: {result.gt_rally_seconds:.1f}s ({result.gt_rally_seconds/result.video_duration_seconds*100:.1f}% of video)")

    print(f"\n--- Motion Detection ---")
    print(f"Regions detected: {result.num_motion_regions}")
    print(f"Total motion time: {result.motion_region_seconds:.1f}s ({result.motion_region_seconds/result.video_duration_seconds*100:.1f}% of video)")

    print(f"\n--- Filter Effectiveness ---")
    print(f"Rally coverage: {result.rally_coverage*100:.1f}% (higher is better - don't miss rallies)")
    print(f"Motion precision: {result.motion_precision*100:.1f}% (higher is better - motion = actual rallies)")
    print(f"Filter ratio: {result.filter_ratio*100:.1f}% (video filtered out, not sent to ML)")

    print(f"\n--- Per-Rally Coverage ---")
    for i, (start, end, coverage) in enumerate(result.per_rally_coverage, 1):
        status = "OK" if coverage >= 1.0 else "PARTIAL" if coverage >= 0.8 else "MISSED"
        print(f"  Rally {i} ({start:.1f}s - {end:.1f}s): {coverage*100:.0f}% [{status}]")


def main():
    """Run profiling on all test fixtures."""
    reset_config()  # Ensure clean config

    fixtures_dir = Path(__file__).parent / "fixtures"
    gt_path = fixtures_dir / "ground_truth.json"

    if not gt_path.exists():
        print(f"Ground truth not found: {gt_path}")
        return

    ground_truth = load_ground_truth(gt_path)

    results = []
    for video_name, gt_rallies in ground_truth.items():
        video_path = fixtures_dir / video_name
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            continue

        print(f"\nProcessing {video_name}...")
        result = evaluate_video(video_path, gt_rallies, use_two_pass=True)
        results.append(result)
        print_results(result)

    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        avg_coverage = sum(r.rally_coverage for r in results) / len(results)
        avg_precision = sum(r.motion_precision for r in results) / len(results)
        avg_filter = sum(r.filter_ratio for r in results) / len(results)

        print(f"Average rally coverage: {avg_coverage*100:.1f}%")
        print(f"Average motion precision: {avg_precision*100:.1f}%")
        print(f"Average filter ratio: {avg_filter*100:.1f}%")

        # Check if filtering is effective
        if avg_coverage >= 0.95:
            print("\n[PASS] Motion detection covers 95%+ of rallies")
        else:
            print(f"\n[WARN] Motion detection only covers {avg_coverage*100:.1f}% of rallies")

        if avg_filter >= 0.30:
            print(f"[PASS] Filtering out {avg_filter*100:.1f}% of video (saves ML compute)")
        else:
            print(f"[WARN] Only filtering {avg_filter*100:.1f}% - limited benefit for ML skipping")


if __name__ == "__main__":
    main()
