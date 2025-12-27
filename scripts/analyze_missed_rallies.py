#!/usr/bin/env python3
"""Analyze ML predictions for missed rallies and test parameter impacts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.analysis.game_state import GameStateAnalyzer
from rallycut.core.config import get_config, reset_config
from rallycut.core.models import GameState
from rallycut.core.video import Video
from rallycut.processing.cutter import VideoCutter


def load_ground_truth() -> dict:
    """Load ground truth data."""
    gt_path = Path(__file__).parent.parent / "tests/fixtures/ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def analyze_predictions_in_window(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    stride: int = 48,
) -> dict:
    """Analyze ML predictions within a specific time window."""
    analyzer = GameStateAnalyzer(enable_temporal_smoothing=False)

    with Video(video_path) as video:
        fps = video.info.fps
        results = analyzer.analyze_video(
            video,
            stride=stride,
            limit_seconds=end_sec + 10,  # Some buffer
        )

    # Filter results within the window
    window_results = []
    for r in results:
        if r.start_frame is None or r.end_frame is None:
            continue
        result_start = r.start_frame / fps
        result_end = r.end_frame / fps

        # Check if result overlaps with window
        if result_end >= start_sec and result_start <= end_sec:
            window_results.append({
                "start_time": result_start,
                "end_time": result_end,
                "state": r.state.value,
                "confidence": r.confidence,
                "play_prob": r.play_confidence,
                "service_prob": r.service_confidence,
                "no_play_prob": r.no_play_confidence,
            })

    # Count states
    play_count = sum(1 for r in window_results if r["state"] == "play")
    service_count = sum(1 for r in window_results if r["state"] == "service")
    no_play_count = sum(1 for r in window_results if r["state"] == "no_play")

    return {
        "window": f"{start_sec:.1f}s - {end_sec:.1f}s",
        "total_predictions": len(window_results),
        "play_count": play_count,
        "service_count": service_count,
        "no_play_count": no_play_count,
        "active_rate": (play_count + service_count) / len(window_results) if window_results else 0,
        "predictions": window_results,
    }


def test_stride_impact(video_path: Path, ground_truth_rallies: list) -> dict:
    """Test how different stride values affect detection."""
    strides = [16, 24, 32, 48, 64]
    results = {}

    for stride in strides:
        reset_config()
        # Run detection with specific stride
        cutter = VideoCutter(
            stride=stride,
            min_segment_start_seconds=0.0,  # Disable early filter for this test
        )
        segments = cutter.analyze_only(video_path)

        # Check which rallies were detected
        detected_rallies = []
        for rally_idx, (rally_start, rally_end) in enumerate(ground_truth_rallies):
            for seg in segments:
                # Check if segment overlaps with rally
                overlap_start = max(rally_start, seg.start_time)
                overlap_end = min(rally_end, seg.end_time)
                if overlap_end > overlap_start:
                    detected_rallies.append(rally_idx + 1)
                    break

        results[stride] = {
            "detected_segments": len(segments),
            "detected_rallies": detected_rallies,
            "recall": len(detected_rallies) / len(ground_truth_rallies) if ground_truth_rallies else 0,
            "segments": [(s.start_time, s.end_time) for s in segments],
        }

    return results


def analyze_min_segment_start_impact(video_path: Path, ground_truth_rallies: list) -> dict:
    """Test how MIN_SEGMENT_START_SECONDS affects detection."""
    thresholds = [0.0, 5.0, 10.0, 15.0]
    results = {}

    for threshold in thresholds:
        reset_config()
        cutter = VideoCutter(min_segment_start_seconds=threshold)
        segments = cutter.analyze_only(video_path)

        detected_rallies = []
        for rally_idx, (rally_start, rally_end) in enumerate(ground_truth_rallies):
            for seg in segments:
                overlap_start = max(rally_start, seg.start_time)
                overlap_end = min(rally_end, seg.end_time)
                if overlap_end > overlap_start:
                    detected_rallies.append(rally_idx + 1)
                    break

        results[threshold] = {
            "detected_segments": len(segments),
            "detected_rallies": detected_rallies,
            "recall": len(detected_rallies) / len(ground_truth_rallies) if ground_truth_rallies else 0,
        }

    return results


def main() -> int:
    project_root = Path(__file__).parent.parent
    fixtures_dir = project_root / "tests/fixtures"
    ground_truth = load_ground_truth()

    # Videos to analyze
    videos = [
        ("match-2-first-2min.MOV", [(10.1, 22.0), (40.1, 48.0), (72.35, 79.0), (96.7, 106.0)]),
        ("match-3-first-2min.MOV", [(1.6, 13.5), (34.3, 44.0), (63.3, 69.5), (85.0, 95.0)]),
    ]

    print("=" * 100)
    print("ANALYSIS OF MISSED RALLIES AND PARAMETER IMPACTS")
    print("=" * 100)

    for video_name, rallies in videos:
        video_path = fixtures_dir / video_name
        if not video_path.exists():
            print(f"\nSkipping {video_name} - not found")
            continue

        print(f"\n{'='*100}")
        print(f"VIDEO: {video_name}")
        print(f"{'='*100}")

        # 1. Analyze ML predictions for first rally (the missed one)
        first_rally_start, first_rally_end = rallies[0]
        print(f"\n--- ML PREDICTIONS FOR FIRST RALLY ({first_rally_start:.1f}s - {first_rally_end:.1f}s) ---")

        pred_analysis = analyze_predictions_in_window(
            video_path, first_rally_start - 5, first_rally_end + 5, stride=48
        )

        print(f"Predictions in window: {pred_analysis['total_predictions']}")
        print(f"  PLAY: {pred_analysis['play_count']}")
        print(f"  SERVICE: {pred_analysis['service_count']}")
        print(f"  NO_PLAY: {pred_analysis['no_play_count']}")
        print(f"  Active rate: {pred_analysis['active_rate']:.1%}")

        if pred_analysis['predictions']:
            print("\nPrediction details:")
            for p in pred_analysis['predictions']:
                state_emoji = "🎾" if p['state'] in ('play', 'service') else "⏸️"
                print(f"  {state_emoji} {p['start_time']:.1f}s-{p['end_time']:.1f}s: {p['state']} "
                      f"(conf={p['confidence']:.2f}, play={p['play_prob']:.2f}, "
                      f"service={p['service_prob']:.2f}, no_play={p['no_play_prob']:.2f})")

        # 2. Test MIN_SEGMENT_START_SECONDS impact
        print(f"\n--- MIN_SEGMENT_START_SECONDS IMPACT ---")
        start_impact = analyze_min_segment_start_impact(video_path, rallies)

        print(f"{'Threshold':<15} | {'Segments':<10} | {'Rallies Detected':<20} | {'Recall':<10}")
        print("-" * 65)
        for threshold, data in start_impact.items():
            rallies_str = ", ".join(map(str, data['detected_rallies'])) or "None"
            print(f"{threshold:<15.1f} | {data['detected_segments']:<10} | {rallies_str:<20} | {data['recall']:.0%}")

        # 3. Test stride impact
        print(f"\n--- STRIDE IMPACT (with min_segment_start=0) ---")
        stride_impact = test_stride_impact(video_path, rallies)

        print(f"{'Stride':<10} | {'Segments':<10} | {'Rallies Detected':<20} | {'Recall':<10}")
        print("-" * 60)
        for stride, data in stride_impact.items():
            rallies_str = ", ".join(map(str, data['detected_rallies'])) or "None"
            print(f"{stride:<10} | {data['detected_segments']:<10} | {rallies_str:<20} | {data['recall']:.0%}")

    # Summary and recommendations
    print("\n" + "=" * 100)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 100)

    print("""
KEY FINDINGS:

1. MIN_SEGMENT_START_SECONDS = 10.0 filters early rallies
   - Match 2 Rally 1 starts at 10.1s (borderline, may be filtered)
   - Match 3 Rally 1 starts at 1.6s (definitely filtered!)
   - RECOMMENDATION: Reduce to 5.0 or 0.0 for better early-rally detection

2. Stride impacts prediction density
   - stride=48 at 30fps = sample every 1.6 seconds
   - stride=32 at 30fps = sample every ~1.07 seconds
   - Lower stride = more predictions = better coverage but slower

3. ML Model behavior
   - Check if early video sections have different characteristics
   - Model may need anchor PLAY predictions for heuristics to work
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
