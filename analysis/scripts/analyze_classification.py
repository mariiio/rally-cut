#!/usr/bin/env python3
"""Analyze ML classification pipeline stages.

Compares raw ML predictions, smoothed predictions, and final segments
to understand where classification errors occur.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.analysis.game_state import GameStateAnalyzer
from rallycut.core.models import GameStateResult
from rallycut.core.video import Video


def load_ground_truth(path: Path) -> dict[str, list[dict[str, float | str]]]:
    """Load ground truth segments."""
    with open(path) as f:
        data = json.load(f)
    return {name: info.get("segments", []) for name, info in data.items()}


def results_to_timeline(
    results: list[GameStateResult],
    fps: float,
    duration: float,
    resolution: float = 1.0,
) -> list[tuple[float, str, float]]:
    """Convert results to (time, state, confidence) timeline."""
    timeline = []
    t = 0.0

    while t < duration:
        frame_idx = int(t * fps)
        state = "no_play"
        conf = 0.0

        for r in results:
            if r.start_frame is not None and r.end_frame is not None:
                if r.start_frame <= frame_idx <= r.end_frame:
                    state = r.state.value
                    conf = r.confidence
                    break

        timeline.append((t, state, conf))
        t += resolution

    return timeline


def print_comparison(
    time: float,
    gt_label: str,
    raw_state: str,
    raw_conf: float,
    smooth_state: str,
    smooth_conf: float,
) -> str:
    """Format a single row of comparison."""
    gt_short = {"service": "SRV", "play": "PLY", "no_play": "NOP"}.get(gt_label, "???")
    raw_short = {"service": "SRV", "play": "PLY", "no_play": "NOP"}.get(raw_state, "???")
    smooth_short = {"service": "SRV", "play": "PLY", "no_play": "NOP"}.get(smooth_state, "???")

    # Check correctness (2-class: service+play = active)
    gt_active = gt_label in ("service", "play")
    raw_active = raw_state in ("service", "play")
    smooth_active = smooth_state in ("service", "play")

    raw_ok = "✓" if raw_active == gt_active else "✗"
    smooth_ok = "✓" if smooth_active == gt_active else "✗"

    return f"{time:6.1f}s | {gt_short} | {raw_short} {raw_conf:.2f} {raw_ok} | {smooth_short} {smooth_conf:.2f} {smooth_ok}"


def get_gt_label(time: float, segments: list[dict[str, float | str]]) -> str:
    """Get ground truth label at time."""
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if start <= time < end:
            return str(seg["label"])
    return "no_play"


def analyze_video(video_path: Path, gt_segments: list[dict[str, float | str]]) -> dict[str, float]:
    """Analyze a single video and return metrics."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {video_path.name}")
    print("=" * 70)

    # Load video
    video = Video(video_path)
    fps = video.info.fps
    duration = min(video.info.duration, 120.0)

    print(f"FPS: {fps:.1f}, Duration: {duration:.1f}s")

    # Get raw and smoothed predictions
    analyzer = GameStateAnalyzer(enable_temporal_smoothing=True)
    smoothed, raw = analyzer.analyze_video(video, return_raw=True, limit_seconds=duration)

    # Ensure we have lists
    assert isinstance(smoothed, list)
    assert isinstance(raw, list)

    print(f"Raw predictions: {len(raw)}, Smoothed: {len(smoothed)}")

    # Convert to timelines
    raw_timeline = results_to_timeline(raw, fps, duration, resolution=1.0)
    smooth_timeline = results_to_timeline(smoothed, fps, duration, resolution=1.0)

    # Print header
    print(f"\n{'Time':>6} | GT  | Raw        | Smoothed")
    print("-" * 50)

    # Compare at each second
    raw_correct = 0
    smooth_correct = 0
    total = 0

    # Track state distribution
    raw_counts = {"service": 0, "play": 0, "no_play": 0}
    smooth_counts = {"service": 0, "play": 0, "no_play": 0}
    gt_counts = {"service": 0, "play": 0, "no_play": 0}

    for i, (t, raw_state, raw_conf) in enumerate(raw_timeline):
        smooth_state, smooth_conf = smooth_timeline[i][1], smooth_timeline[i][2]
        gt_label = get_gt_label(t, gt_segments)

        # Count states
        raw_counts[raw_state] = raw_counts.get(raw_state, 0) + 1
        smooth_counts[smooth_state] = smooth_counts.get(smooth_state, 0) + 1
        gt_counts[gt_label] = gt_counts.get(gt_label, 0) + 1

        # Check correctness (2-class)
        gt_active = gt_label in ("service", "play")
        raw_active = raw_state in ("service", "play")
        smooth_active = smooth_state in ("service", "play")

        if raw_active == gt_active:
            raw_correct += 1
        if smooth_active == gt_active:
            smooth_correct += 1
        total += 1

        # Print row (only if there's a mismatch or at boundaries)
        raw_mismatch = raw_active != gt_active
        smooth_mismatch = smooth_active != gt_active
        is_boundary = i == 0 or get_gt_label(t - 1, gt_segments) != gt_label

        if raw_mismatch or smooth_mismatch or is_boundary:
            line = print_comparison(t, gt_label, raw_state, raw_conf, smooth_state, smooth_conf)
            print(line)

    # Summary stats
    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)

    raw_acc = raw_correct / total * 100
    smooth_acc = smooth_correct / total * 100

    print("\nAccuracy (2-class):")
    print(f"  Raw ML:     {raw_acc:5.1f}% ({raw_correct}/{total})")
    print(f"  Smoothed:   {smooth_acc:5.1f}% ({smooth_correct}/{total})")
    print(f"  Delta:      {smooth_acc - raw_acc:+5.1f}%")

    print("\nState Distribution (seconds):")
    print(f"  {'':12} {'GT':>6} {'Raw':>6} {'Smooth':>6}")
    for state in ["service", "play", "no_play"]:
        gt = gt_counts.get(state, 0)
        raw = raw_counts.get(state, 0)
        smooth = smooth_counts.get(state, 0)
        print(f"  {state:12} {gt:>6} {raw:>6} {smooth:>6}")

    # Active vs no_play comparison
    gt_active = gt_counts.get("service", 0) + gt_counts.get("play", 0)
    raw_active = raw_counts.get("service", 0) + raw_counts.get("play", 0)
    smooth_active = smooth_counts.get("service", 0) + smooth_counts.get("play", 0)

    print(f"\n  {'active':12} {gt_active:>6} {raw_active:>6} {smooth_active:>6}")
    print(f"  {'no_play':12} {gt_counts.get('no_play', 0):>6} {raw_counts.get('no_play', 0):>6} {smooth_counts.get('no_play', 0):>6}")

    # Detection ratio
    if gt_active > 0:
        raw_detection_ratio = raw_active / gt_active * 100
        smooth_detection_ratio = smooth_active / gt_active * 100
        print("\nActive Detection Ratio (predicted/actual):")
        print(f"  Raw:        {raw_detection_ratio:5.1f}%")
        print(f"  Smoothed:   {smooth_detection_ratio:5.1f}%")

    return {
        "raw_accuracy": raw_acc,
        "smooth_accuracy": smooth_acc,
        "raw_active_ratio": raw_active / gt_active * 100 if gt_active > 0 else 0,
        "smooth_active_ratio": smooth_active / gt_active * 100 if gt_active > 0 else 0,
    }


def main() -> int:
    project_root = Path(__file__).parent.parent
    gt_path = project_root / "tests/fixtures/ground_truth.json"
    video_dir = project_root / "tests/fixtures"

    ground_truth = load_ground_truth(gt_path)

    all_metrics = []

    for video_name, segments in ground_truth.items():
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"Skipping {video_name}: not found")
            continue

        metrics = analyze_video(video_path, segments)
        all_metrics.append(metrics)

    # Overall summary
    if len(all_metrics) > 1:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        avg_raw_acc = sum(m["raw_accuracy"] for m in all_metrics) / len(all_metrics)
        avg_smooth_acc = sum(m["smooth_accuracy"] for m in all_metrics) / len(all_metrics)
        avg_raw_ratio = sum(m["raw_active_ratio"] for m in all_metrics) / len(all_metrics)
        avg_smooth_ratio = sum(m["smooth_active_ratio"] for m in all_metrics) / len(all_metrics)

        print("\nAverage 2-class Accuracy:")
        print(f"  Raw ML:     {avg_raw_acc:5.1f}%")
        print(f"  Smoothed:   {avg_smooth_acc:5.1f}%")

        print("\nAverage Active Detection Ratio:")
        print(f"  Raw ML:     {avg_raw_ratio:5.1f}%")
        print(f"  Smoothed:   {avg_smooth_ratio:5.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
