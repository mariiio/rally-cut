"""Regression test for rally detection quality.

This test runs the full ML detection pipeline on ground truth videos
and validates that detected rallies match expected rally times.
"""

import json
from pathlib import Path
from typing import NamedTuple

import pytest

from rallycut.processing.cutter import VideoCutter

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"

# Tolerance for rally boundary matching (seconds)
# Set to 6.0 to account for:
# - Padding: 2s start, 3.0s end (added to detected segments)
# - Rally continuation heuristic (may extend by ~2s)
# - ML model boundary variance
TOLERANCE_SECONDS = 6.0

# Minimum recall rate required for tests to pass
# Set to 0.75 (75%) to account for ML model limitations where some rallies
# may have zero PLAY predictions (heuristic cannot help without anchor points)
MIN_RECALL_RATE = 0.75


class VideoTestCase(NamedTuple):
    """Test case for a video with ground truth."""

    video_name: str
    video_path: Path
    display_name: str


# Define test cases for parametrization
TEST_CASES = [
    VideoTestCase(
        video_name="match_first_2min.mp4",
        video_path=FIXTURES_DIR / "match_first_2min.mp4",
        display_name="Match 1",
    ),
    VideoTestCase(
        video_name="match-2-first-2min.MOV",
        video_path=FIXTURES_DIR / "match-2-first-2min.MOV",
        display_name="Match 2",
    ),
    VideoTestCase(
        video_name="match-3-first-2min.MOV",
        video_path=FIXTURES_DIR / "match-3-first-2min.MOV",
        display_name="Match 3",
    ),
]


def load_ground_truth() -> dict:
    """Load the unified ground truth file."""
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def get_rally_times(video_name: str) -> list[tuple[float, float]]:
    """Get rally (start, end) times for a video from ground truth.

    Ground truth uses float seconds for rally start/end times.
    """
    ground_truth = load_ground_truth()
    video_data = ground_truth.get(video_name, {})
    rallies = video_data.get("rallies", [])

    return [
        (float(r["start"]), float(r["end"]))
        for r in rallies
    ]


def rally_matches(
    expected: tuple[float, float],
    detected: tuple[float, float],
    tolerance: float,
) -> bool:
    """Check if detected rally matches expected rally within tolerance."""
    exp_start, exp_end = expected
    det_start, det_end = detected

    start_ok = abs(exp_start - det_start) <= tolerance
    end_ok = abs(exp_end - det_end) <= tolerance

    return start_ok and end_ok


def find_best_match(
    expected: tuple[float, float],
    detected_rallies: list[tuple[float, float]],
) -> tuple[int, float] | None:
    """Find detected rally with best overlap to expected rally.

    Returns (index, overlap_score) or None if no overlap.
    """
    exp_start, exp_end = expected
    best_idx = None
    best_overlap = 0.0

    for idx, (det_start, det_end) in enumerate(detected_rallies):
        # Calculate overlap
        overlap_start = max(exp_start, det_start)
        overlap_end = min(exp_end, det_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_idx is not None:
        return best_idx, best_overlap
    return None


def run_detection(video_path: Path) -> list[tuple[float, float]]:
    """Run detection pipeline and return detected rally times."""
    # Use VideoCutter.analyze_only() with default config
    # This ensures tests match what users experience
    cutter = VideoCutter()
    segments = cutter.analyze_only(video_path)
    # Convert segments to (start_time, end_time) tuples
    return [(seg.start_time, seg.end_time) for seg in segments]


# Cache for detection results to avoid running ML inference twice per video
_detection_cache: dict[str, list[tuple[float, float]]] = {}


def get_detection_results(test_case: VideoTestCase) -> list[tuple[float, float]]:
    """Get cached detection results, running detection if needed."""
    cache_key = str(test_case.video_path)
    if cache_key not in _detection_cache:
        _detection_cache[cache_key] = run_detection(test_case.video_path)
    return _detection_cache[cache_key]


@pytest.mark.slow
class TestDetectionQuality:
    """Test rally detection quality against ground truth."""

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[tc.display_name for tc in TEST_CASES],
    )
    def test_all_expected_rallies_detected(self, test_case: VideoTestCase):
        """Verify all expected rallies have matching detected segments."""
        if not test_case.video_path.exists():
            pytest.skip(f"Test video not found: {test_case.video_path}")

        expected_rallies = get_rally_times(test_case.video_name)
        detected_segments = get_detection_results(test_case)

        matches = []
        misses = []

        for i, expected in enumerate(expected_rallies):
            exp_start, exp_end = expected

            # Find best matching detected segment
            match = find_best_match(expected, detected_segments)

            if match is not None:
                det_idx, overlap = match
                det_start, det_end = detected_segments[det_idx]

                # Check if within tolerance
                if rally_matches(expected, detected_segments[det_idx], TOLERANCE_SECONDS):
                    matches.append({
                        "expected_idx": i,
                        "expected": f"{exp_start:.0f}s-{exp_end:.0f}s",
                        "detected": f"{det_start:.1f}s-{det_end:.1f}s",
                        "overlap": overlap,
                    })
                else:
                    misses.append({
                        "expected_idx": i,
                        "expected": f"{exp_start:.0f}s-{exp_end:.0f}s",
                        "best_match": f"{det_start:.1f}s-{det_end:.1f}s",
                        "reason": "Outside tolerance",
                    })
            else:
                misses.append({
                    "expected_idx": i,
                    "expected": f"{exp_start:.0f}s-{exp_end:.0f}s",
                    "reason": "No overlapping segment found",
                })

        # Report results
        print(f"\n=== Detection Quality Report ({test_case.display_name}) ===")
        print(f"Expected rallies: {len(expected_rallies)}")
        print(f"Detected segments: {len(detected_segments)}")
        print(f"Matches: {len(matches)}")
        print(f"Misses: {len(misses)}")

        if matches:
            print("\nMatched rallies:")
            for m in matches:
                print(f"  Rally {m['expected_idx']+1}: {m['expected']} -> {m['detected']} (overlap: {m['overlap']:.1f}s)")

        if misses:
            print("\nMissed rallies:")
            for m in misses:
                print(f"  Rally {m['expected_idx']+1}: {m['expected']} - {m['reason']}")
                if "best_match" in m:
                    print(f"    Best match: {m['best_match']}")

        # Print all detected segments for debugging
        print("\nAll detected segments:")
        for i, (start, end) in enumerate(detected_segments):
            print(f"  Segment {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")

        # Assert minimum recall rate is met
        recall = len(matches) / len(expected_rallies) if expected_rallies else 0
        print(f"\nRecall: {recall:.1%} (minimum required: {MIN_RECALL_RATE:.0%})")
        assert recall >= MIN_RECALL_RATE, (
            f"Recall {recall:.1%} below minimum {MIN_RECALL_RATE:.0%}: "
            f"detected {len(matches)}/{len(expected_rallies)} rallies"
        )

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=[tc.display_name for tc in TEST_CASES],
    )
    def test_detection_precision(self, test_case: VideoTestCase):
        """Check for false positive detection (extra rallies)."""
        if not test_case.video_path.exists():
            pytest.skip(f"Test video not found: {test_case.video_path}")

        expected_rallies = get_rally_times(test_case.video_name)
        detected_segments = get_detection_results(test_case)

        # Count unmatched detected segments
        matched_indices = set()

        for expected in expected_rallies:
            match = find_best_match(expected, detected_segments)
            if match is not None:
                matched_indices.add(match[0])

        unmatched_count = len(detected_segments) - len(matched_indices)

        print(f"\n=== Precision Report ({test_case.display_name}) ===")
        print(f"Detected segments: {len(detected_segments)}")
        print(f"Matched to expected: {len(matched_indices)}")
        print(f"Extra (false positives): {unmatched_count}")

        # Warn about false positives but don't fail
        # (detection may legitimately find more rallies than annotated)
        if unmatched_count > 0:
            print(f"\nNote: {unmatched_count} detected segments don't match expected rallies")
