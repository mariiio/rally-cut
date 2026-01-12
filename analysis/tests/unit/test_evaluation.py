"""Unit tests for evaluation metrics and matching."""

from __future__ import annotations

import pytest

from rallycut.evaluation.matching import MatchingResult, RallyMatch, compute_iou, match_rallies
from rallycut.evaluation.metrics import (
    BoundaryMetrics,
    RallyMetrics,
    aggregate_metrics,
    compute_metrics,
)
from rallycut.evaluation.param_grid import (
    AVAILABLE_GRIDS,
    DEFAULT_PARAMS,
    generate_param_combinations,
    get_grid,
    grid_size,
)


class TestIoU:
    """Tests for IoU computation."""

    def test_perfect_overlap(self) -> None:
        """Identical intervals should have IoU = 1.0."""
        assert compute_iou(0, 10, 0, 10) == 1.0

    def test_no_overlap(self) -> None:
        """Non-overlapping intervals should have IoU = 0.0."""
        assert compute_iou(0, 10, 20, 30) == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap should calculate correctly."""
        # Overlap: 5-10 (5s), Union: 0-15 (15s)
        iou = compute_iou(0, 10, 5, 15)
        assert abs(iou - 5 / 15) < 0.001

    def test_contained_interval(self) -> None:
        """Smaller interval contained in larger."""
        # GT: 0-20, Pred: 5-15
        # Intersection: 5-15 (10s), Union: 0-20 (20s)
        iou = compute_iou(0, 20, 5, 15)
        assert abs(iou - 10 / 20) < 0.001

    def test_adjacent_intervals(self) -> None:
        """Adjacent intervals (no gap, no overlap) should have IoU = 0."""
        # They touch at 10, but no actual overlap
        iou = compute_iou(0, 10, 10, 20)
        assert iou == 0.0

    def test_zero_duration(self) -> None:
        """Zero-duration intervals should handle gracefully."""
        iou = compute_iou(5, 5, 5, 5)
        assert iou == 0.0  # Both have zero duration


class TestMatching:
    """Tests for rally matching algorithm."""

    def test_perfect_matches(self) -> None:
        """All predictions match perfectly."""
        gt = [(0, 10), (20, 30)]
        pred = [(0, 10), (20, 30)]
        result = match_rallies(gt, pred, iou_threshold=0.5)

        assert len(result.matches) == 2
        assert len(result.unmatched_ground_truth) == 0
        assert len(result.unmatched_predictions) == 0

    def test_missed_rally(self) -> None:
        """Missing one rally in predictions."""
        gt = [(0, 10), (20, 30)]
        pred = [(0, 10)]  # Missing second rally
        result = match_rallies(gt, pred, iou_threshold=0.5)

        assert len(result.matches) == 1
        assert len(result.unmatched_ground_truth) == 1
        assert len(result.unmatched_predictions) == 0
        assert result.unmatched_ground_truth[0] == 1  # Second GT missed

    def test_false_positive(self) -> None:
        """Extra detection not in ground truth."""
        gt = [(0, 10)]
        pred = [(0, 10), (50, 60)]  # Extra detection
        result = match_rallies(gt, pred, iou_threshold=0.5)

        assert len(result.matches) == 1
        assert len(result.unmatched_ground_truth) == 0
        assert len(result.unmatched_predictions) == 1
        assert result.unmatched_predictions[0] == 1  # Second pred is FP

    def test_threshold_filtering(self) -> None:
        """Low IoU matches should be filtered."""
        gt = [(0, 10)]
        pred = [(8, 18)]  # Overlap: 2s, Union: 18s, IoU = 0.11
        result = match_rallies(gt, pred, iou_threshold=0.5)

        assert len(result.matches) == 0  # Below threshold
        assert len(result.unmatched_ground_truth) == 1
        assert len(result.unmatched_predictions) == 1

    def test_greedy_matching(self) -> None:
        """Greedy matching picks best match for each GT."""
        gt = [(0, 10)]
        pred = [(2, 12), (0, 10)]  # Second is better match
        result = match_rallies(gt, pred, iou_threshold=0.5)

        assert len(result.matches) == 1
        assert result.matches[0].predicted_idx == 1  # Picked the perfect match
        assert result.matches[0].iou == 1.0

    def test_boundary_errors(self) -> None:
        """Boundary errors should be calculated correctly."""
        gt = [(10, 20)]
        pred = [(12, 18)]  # Late start (+2s), early end (-2s)
        result = match_rallies(gt, pred, iou_threshold=0.3)

        assert len(result.matches) == 1
        match = result.matches[0]
        assert match.start_error_ms == 2000  # 2s late
        assert match.end_error_ms == -2000  # 2s early

    def test_empty_inputs(self) -> None:
        """Handle empty inputs gracefully."""
        result = match_rallies([], [], iou_threshold=0.5)
        assert len(result.matches) == 0
        assert len(result.unmatched_ground_truth) == 0
        assert len(result.unmatched_predictions) == 0

        result = match_rallies([(0, 10)], [], iou_threshold=0.5)
        assert len(result.unmatched_ground_truth) == 1

        result = match_rallies([], [(0, 10)], iou_threshold=0.5)
        assert len(result.unmatched_predictions) == 1


class TestMetrics:
    """Tests for metrics calculation."""

    def test_precision_calculation(self) -> None:
        """Precision = TP / (TP + FP)."""
        m = RallyMetrics(true_positives=8, false_positives=2, false_negatives=0)
        assert m.precision == 0.8

    def test_recall_calculation(self) -> None:
        """Recall = TP / (TP + FN)."""
        m = RallyMetrics(true_positives=8, false_positives=0, false_negatives=2)
        assert m.recall == 0.8

    def test_f1_calculation(self) -> None:
        """F1 = 2 * P * R / (P + R)."""
        m = RallyMetrics(true_positives=8, false_positives=2, false_negatives=2)
        # Precision = 0.8, Recall = 0.8
        # F1 = 2 * 0.8 * 0.8 / 1.6 = 0.8
        assert abs(m.f1 - 0.8) < 0.001

    def test_perfect_metrics(self) -> None:
        """Perfect predictions should give 1.0 everywhere."""
        m = RallyMetrics(true_positives=10, false_positives=0, false_negatives=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_zero_metrics(self) -> None:
        """No true positives should give 0.0 everywhere."""
        m = RallyMetrics(true_positives=0, false_positives=5, false_negatives=5)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_boundary_metrics(self) -> None:
        """Boundary metrics calculation."""
        bm = BoundaryMetrics(
            start_errors_ms=[100, -100, 200],
            end_errors_ms=[50, -50, 100],
        )

        # Mean should be (100 - 100 + 200) / 3 = 66.67
        assert bm.mean_start_error_ms is not None
        assert abs(bm.mean_start_error_ms - 66.67) < 0.1

        # Mean absolute should be (100 + 100 + 200) / 3 = 133.33
        assert bm.mean_abs_start_error_ms is not None
        assert abs(bm.mean_abs_start_error_ms - 133.33) < 0.1

        # Median of [100, -100, 200] = 100
        assert bm.median_start_error_ms == 100

    def test_empty_boundary_metrics(self) -> None:
        """Empty boundary metrics should return None."""
        bm = BoundaryMetrics()
        assert bm.mean_start_error_ms is None
        assert bm.mean_abs_start_error_ms is None
        assert bm.median_start_error_ms is None


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_basic(self) -> None:
        """Basic compute_metrics test."""
        gt = [(0, 10), (20, 30)]
        pred = [(0, 10), (21, 31)]  # Second is close but shifted

        # Manually create matching result
        matching = MatchingResult(
            matches=[
                RallyMatch(
                    ground_truth_idx=0,
                    predicted_idx=0,
                    iou=1.0,
                    start_error_ms=0,
                    end_error_ms=0,
                ),
                RallyMatch(
                    ground_truth_idx=1,
                    predicted_idx=1,
                    iou=0.8,  # Some overlap
                    start_error_ms=1000,
                    end_error_ms=1000,
                ),
            ],
            unmatched_ground_truth=[],
            unmatched_predictions=[],
        )

        result = compute_metrics(
            ground_truth=gt,
            predictions=pred,
            matching_result=matching,
            video_id="test-video",
            video_filename="test.mp4",
            iou_threshold=0.5,
        )

        assert result.ground_truth_count == 2
        assert result.prediction_count == 2
        assert result.rally_metrics.precision == 1.0
        assert result.rally_metrics.recall == 1.0
        assert result.rally_metrics.f1 == 1.0


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""

    def test_aggregate_multiple_videos(self) -> None:
        """Aggregate metrics from multiple videos."""
        results = [
            compute_metrics(
                ground_truth=[(0, 10)],
                predictions=[(0, 10)],
                matching_result=MatchingResult(
                    matches=[RallyMatch(0, 0, 1.0, 0, 0)],
                    unmatched_ground_truth=[],
                    unmatched_predictions=[],
                ),
                video_id="v1",
                video_filename="v1.mp4",
                iou_threshold=0.5,
            ),
            compute_metrics(
                ground_truth=[(0, 10), (20, 30)],
                predictions=[(0, 10)],  # Missed one
                matching_result=MatchingResult(
                    matches=[RallyMatch(0, 0, 1.0, 0, 0)],
                    unmatched_ground_truth=[1],
                    unmatched_predictions=[],
                ),
                video_id="v2",
                video_filename="v2.mp4",
                iou_threshold=0.5,
            ),
        ]

        agg = aggregate_metrics(results)

        assert agg.video_count == 2
        assert agg.total_ground_truth == 3
        assert agg.total_predictions == 2
        assert agg.rally_metrics.true_positives == 2
        assert agg.rally_metrics.false_negatives == 1
        # Recall = 2/3 = 0.667
        assert abs(agg.rally_metrics.recall - 2 / 3) < 0.01


class TestParamGrid:
    """Tests for parameter grid functionality."""

    def test_available_grids(self) -> None:
        """All expected grids should be available."""
        expected = {"quick", "full", "beach", "strict", "relaxed"}
        assert set(AVAILABLE_GRIDS.keys()) == expected

    def test_get_grid(self) -> None:
        """get_grid should return the correct grid."""
        grid = get_grid("quick")
        assert "min_gap_seconds" in grid
        assert "rally_continuation_seconds" in grid

    def test_get_grid_invalid(self) -> None:
        """get_grid should raise for unknown grid."""
        with pytest.raises(ValueError, match="Unknown grid"):
            get_grid("nonexistent")

    def test_grid_size(self) -> None:
        """grid_size should calculate correctly."""
        grid = {"a": [1, 2, 3], "b": [4, 5]}
        assert grid_size(grid) == 6

    def test_generate_combinations(self) -> None:
        """generate_param_combinations should produce correct count."""
        grid = {"min_gap_seconds": [3.0, 5.0], "rally_continuation_seconds": [1.0, 2.0]}
        combos = generate_param_combinations(grid)

        assert len(combos) == 4

        # Check all combinations are unique
        combo_tuples = [
            (c.min_gap_seconds, c.rally_continuation_seconds) for c in combos
        ]
        assert len(set(combo_tuples)) == 4

    def test_generate_combinations_preserves_defaults(self) -> None:
        """Non-overridden params should keep default values."""
        grid = {"min_gap_seconds": [7.0]}
        combos = generate_param_combinations(grid)

        assert len(combos) == 1
        assert combos[0].min_gap_seconds == 7.0
        # Other params should be defaults
        assert combos[0].rally_continuation_seconds == DEFAULT_PARAMS.rally_continuation_seconds
        assert combos[0].min_play_duration == DEFAULT_PARAMS.min_play_duration
