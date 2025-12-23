#!/usr/bin/env python3
"""Evaluate rally detectors against ground truth."""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.core.models import GameState
from rallycut.core.video import Video
from rallycut.processing.cutter import VideoCutter


# Ground truth rallies for match_first_2min.mp4
GROUND_TRUTH = [
    (12, 19),   # Rally 1: 0:12-0:19
    (32, 46),   # Rally 2: 0:32-0:46
    (63, 70),   # Rally 3: 1:03-1:10
    (87, 111),  # Rally 4: 1:27-1:51
]


def time_to_seconds(minutes: int, seconds: int) -> int:
    return minutes * 60 + seconds


def evaluate_detector(
    detector_name: str,
    results: list,
    video: Video,
    ground_truth: list[tuple[int, int]],
) -> dict:
    """
    Evaluate detector results against ground truth.

    Args:
        detector_name: Name of the detector
        results: List of GameStateResult from detector
        video: Video object
        ground_truth: List of (start_seconds, end_seconds) tuples

    Returns:
        Dict with precision, recall, F1, IoU metrics
    """
    fps = video.info.fps

    # Convert results to segments using VideoCutter logic
    cutter = VideoCutter(min_play_duration=2.0, min_gap_seconds=1.5)
    segments = cutter._get_segments_from_results(results, fps)

    # Convert segments to time ranges
    detected_ranges = []
    for seg in segments:
        if seg.state in (GameState.PLAY, GameState.SERVICE):
            start_sec = seg.start_frame / fps
            end_sec = seg.end_frame / fps
            detected_ranges.append((start_sec, end_sec))

    print(f"\n=== {detector_name} Results ===")
    print(f"Detected {len(detected_ranges)} segments:")
    for i, (start, end) in enumerate(detected_ranges):
        print(f"  {i+1}. {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")

    # Calculate metrics
    true_positives = 0
    false_negatives = 0

    print(f"\nGround truth comparison:")
    for i, (gt_start, gt_end) in enumerate(ground_truth):
        # Check if this ground truth rally is detected
        detected = False
        best_iou = 0
        best_match = None

        for det_start, det_end in detected_ranges:
            # Calculate IoU
            intersection_start = max(gt_start, det_start)
            intersection_end = min(gt_end, det_end)
            intersection = max(0, intersection_end - intersection_start)

            union_start = min(gt_start, det_start)
            union_end = max(gt_end, det_end)
            union = union_end - union_start

            iou = intersection / union if union > 0 else 0

            # Consider detected if >30% overlap with ground truth
            if intersection > 0.3 * (gt_end - gt_start):
                detected = True
                if iou > best_iou:
                    best_iou = iou
                    best_match = (det_start, det_end)

        if detected:
            true_positives += 1
            status = f"FOUND (IoU: {best_iou:.2f}, detected: {best_match[0]:.1f}-{best_match[1]:.1f}s)"
        else:
            false_negatives += 1
            status = "MISSED"

        print(f"  Rally {i+1} ({gt_start}s-{gt_end}s): {status}")

    # False positives: detected segments not matching any ground truth
    false_positives = 0
    for det_start, det_end in detected_ranges:
        matches_gt = False
        for gt_start, gt_end in ground_truth:
            intersection_start = max(gt_start, det_start)
            intersection_end = min(gt_end, det_end)
            intersection = max(0, intersection_end - intersection_start)
            if intersection > 0.3 * (det_end - det_start):
                matches_gt = True
                break
        if not matches_gt:
            false_positives += 1

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nMetrics:")
    print(f"  True Positives: {true_positives}/{len(ground_truth)}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detected_segments": len(detected_ranges),
    }


def test_motion_detector(video_path: Path):
    """Test the motion detector."""
    from rallycut.analysis.motion_detector import MotionDetector

    video = Video(video_path)
    detector = MotionDetector()

    print(f"\nTesting MotionDetector on {video_path.name}")

    start_time = time.time()
    results = detector.analyze_video(video)
    elapsed = time.time() - start_time

    print(f"\nProcessing time: {elapsed:.2f}s")
    print(f"Speed: {video.info.duration / elapsed:.1f}x real-time")

    metrics = evaluate_detector("MotionDetector", results, video, GROUND_TRUTH)
    metrics["processing_time"] = elapsed
    metrics["speed_factor"] = video.info.duration / elapsed

    return metrics


def test_videomae_detector(video_path: Path):
    """Test VideoMAE ML detector as baseline."""
    from rallycut.analysis.game_state import GameStateAnalyzer

    video = Video(video_path)
    analyzer = GameStateAnalyzer()

    print(f"\nTesting VideoMAE (ML baseline) on {video_path.name}")

    start_time = time.time()
    results = analyzer.analyze_video(video, stride=16)  # stride 16 for speed
    elapsed = time.time() - start_time

    print(f"\nProcessing time: {elapsed:.2f}s")
    print(f"Speed: {video.info.duration / elapsed:.1f}x real-time")

    metrics = evaluate_detector("VideoMAE", results, video, GROUND_TRUTH)
    metrics["processing_time"] = elapsed
    metrics["speed_factor"] = video.info.duration / elapsed

    return metrics


def main():
    # Default test video path
    video_path = Path.home() / "Desktop" / "match_first_2min.mp4"

    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        print("Usage: python scripts/evaluate_detector.py [video_path]")
        sys.exit(1)

    print("=" * 60)
    print("Rally Detection Evaluation")
    print("=" * 60)

    # Test motion detector
    motion_metrics = test_motion_detector(video_path)

    # Test VideoMAE (ML baseline)
    videomae_metrics = test_videomae_detector(video_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Detector':<20} {'Recall':<10} {'Precision':<10} {'Speed':<10}")
    print("-" * 60)
    print(
        f"{'VideoMAE (baseline)':<20} {videomae_metrics['recall']:<10.0%} "
        f"{videomae_metrics['precision']:<10.0%} {videomae_metrics['speed_factor']:<10.1f}x"
    )
    print(
        f"{'MotionDetector':<20} {motion_metrics['recall']:<10.0%} "
        f"{motion_metrics['precision']:<10.0%} {motion_metrics['speed_factor']:<10.1f}x"
    )


if __name__ == "__main__":
    main()
