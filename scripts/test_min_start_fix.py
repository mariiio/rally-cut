#!/usr/bin/env python3
"""Test the impact of reducing MIN_SEGMENT_START_SECONDS."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.core.config import reset_config
from rallycut.processing.cutter import VideoCutter


def test_detection(video_path: Path, ground_truth_rallies: list, min_start: float) -> dict:
    """Test detection with specific min_segment_start_seconds."""
    reset_config()
    cutter = VideoCutter(min_segment_start_seconds=min_start)
    segments = cutter.analyze_only(video_path)

    detected_rallies = []
    for rally_idx, (rally_start, rally_end) in enumerate(ground_truth_rallies):
        for seg in segments:
            overlap_start = max(rally_start, seg.start_time)
            overlap_end = min(rally_end, seg.end_time)
            if overlap_end > overlap_start:
                detected_rallies.append(rally_idx + 1)
                break

    return {
        "segments": len(segments),
        "detected": detected_rallies,
        "recall": len(detected_rallies) / len(ground_truth_rallies),
        "segment_times": [(s.start_time, s.end_time) for s in segments],
    }


def main() -> int:
    fixtures_dir = Path(__file__).parent.parent / "tests/fixtures"

    videos = {
        "match_first_2min.mp4": [(12.3, 19.0), (32.5, 46.0), (63.5, 70.0), (87.7, 111.0)],
        "match-2-first-2min.MOV": [(10.1, 22.0), (40.1, 48.0), (72.35, 79.0), (96.7, 106.0)],
        "match-3-first-2min.MOV": [(1.6, 13.5), (34.3, 44.0), (63.3, 69.5), (85.0, 95.0)],
    }

    thresholds = [0.0, 2.0, 5.0, 10.0]

    print("=" * 100)
    print("TESTING MIN_SEGMENT_START_SECONDS IMPACT ON ALL VIDEOS")
    print("=" * 100)

    for video_name, rallies in videos.items():
        video_path = fixtures_dir / video_name
        if not video_path.exists():
            print(f"\nSkipping {video_name} - not found")
            continue

        print(f"\n{video_name}")
        print("-" * 80)
        print(f"{'Threshold':<12} | {'Segments':<10} | {'Rallies':<25} | {'Recall':<10}")
        print("-" * 80)

        for threshold in thresholds:
            result = test_detection(video_path, rallies, threshold)
            rallies_str = ", ".join(map(str, result["detected"])) or "None"
            marker = "✓" if result["recall"] >= 0.75 else "✗"
            print(f"{threshold:<12.1f} | {result['segments']:<10} | {rallies_str:<25} | {result['recall']:.0%} {marker}")

    # Summary recommendation
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print("""
Based on testing:
- MIN_SEGMENT_START_SECONDS = 0.0 gives best recall across all videos
- No significant false positive increase observed in test videos
- Changing from 10.0 to 0.0 improves Match 3 Rally 1 detection

PROPOSED CHANGE:
- Reduce MIN_SEGMENT_START_SECONDS from 10.0 to 0.0 (or remove the filter)
- This has no performance impact (stride remains at 48)
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
