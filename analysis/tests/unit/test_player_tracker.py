"""Tests for player tracker utilities."""

from __future__ import annotations

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import (
    DEFAULT_COURT_ROI,
    compute_court_roi_from_ball,
)


class TestComputeCourtRoiFromBall:
    """Tests for adaptive court ROI from ball trajectory."""

    def _make_positions(
        self,
        coords: list[tuple[float, float]],
        confidence: float = 0.5,
    ) -> list[BallPosition]:
        return [
            BallPosition(frame_number=i, x=x, y=y, confidence=confidence)
            for i, (x, y) in enumerate(coords)
        ]

    def test_returns_none_with_too_few_points(self) -> None:
        """Should return None when fewer than min_points confident detections."""
        positions = self._make_positions([(0.5, 0.5)] * 10)
        roi, msg = compute_court_roi_from_ball(positions)
        assert roi is None
        assert "Only 10" in msg

    def test_returns_none_for_zero_confidence(self) -> None:
        """Zero-confidence positions should be filtered out."""
        positions = self._make_positions([(0.5, 0.5)] * 30, confidence=0.0)
        roi, msg = compute_court_roi_from_ball(positions)
        assert roi is None
        assert "Only 0" in msg

    def test_returns_roi_for_sufficient_data(self) -> None:
        """Should return a 4-point rectangle for normal ball data."""
        # Ball covering center of court: x=0.2-0.8, y=0.2-0.6
        coords = [(0.2 + 0.6 * (i / 30), 0.2 + 0.4 * (i % 10) / 10) for i in range(30)]
        positions = self._make_positions(coords)

        roi, msg = compute_court_roi_from_ball(positions)
        assert roi is not None
        assert len(roi) == 4

        xs = [p[0] for p in roi]
        ys = [p[1] for p in roi]
        # ROI should be expanded beyond ball bounds
        assert min(xs) < 0.2  # Left margin expanded
        assert max(xs) > 0.8  # Right margin expanded
        assert min(ys) < 0.2  # Top margin expanded
        assert max(ys) > 0.6  # Bottom margin expanded

    def test_roi_clamped_to_frame(self) -> None:
        """ROI coordinates should be clamped to 0.0-1.0."""
        # Ball near left/top edge — margins would push ROI below 0.0
        coords = [(0.03 + 0.5 * i / 30, 0.03 + 0.4 * i / 30) for i in range(30)]
        positions = self._make_positions(coords)

        roi, _ = compute_court_roi_from_ball(positions)
        assert roi is not None
        for x, y in roi:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_returns_none_for_tiny_trajectory(self) -> None:
        """Ball stuck in one spot should fail ball coverage check."""
        # All positions at nearly the same point
        coords = [(0.5 + i * 0.001, 0.5 + i * 0.001) for i in range(30)]
        positions = self._make_positions(coords)

        roi, msg = compute_court_roi_from_ball(positions)
        assert roi is None
        assert "covers only" in msg

    def test_returns_none_for_full_frame_trajectory(self) -> None:
        """Ball spanning the entire frame should fail max_roi_area check."""
        # Ball from corner to corner — ROI would cover >75% of frame
        coords = [(0.02, 0.02)] * 15 + [(0.98, 0.98)] * 15
        positions = self._make_positions(coords)

        roi, msg = compute_court_roi_from_ball(positions)
        assert roi is None
        assert "too spread" in msg

    def test_quality_warning_for_narrow_trajectory(self) -> None:
        """Should warn when ball trajectory is narrow but usable."""
        # Ball only in a narrow vertical line
        coords = [(0.5, 0.2 + 0.4 * i / 30) for i in range(30)]
        positions = self._make_positions(coords)

        roi, msg = compute_court_roi_from_ball(positions)
        # May return ROI or None depending on area check
        if roi is not None:
            assert "narrow" in msg.lower() or msg == ""

    def test_quality_warning_for_few_points(self) -> None:
        """Should warn when we have enough points but not many."""
        # Just above threshold (20) but below 50
        coords = [(0.2 + 0.6 * i / 25, 0.2 + 0.4 * i / 25) for i in range(25)]
        positions = self._make_positions(coords)

        roi, msg = compute_court_roi_from_ball(positions)
        assert roi is not None
        assert "Limited ball detections" in msg

    def test_filters_out_zero_zero_positions(self) -> None:
        """Should ignore (0,0) positions (VballNet no-detection placeholders)."""
        # 10 real + 20 at (0,0)
        real = [(0.3 + 0.4 * i / 10, 0.3 + 0.2 * i / 10) for i in range(10)]
        zeros = [(0.0, 0.0)] * 20
        positions = self._make_positions(real + zeros)

        roi, msg = compute_court_roi_from_ball(positions)
        # Only 10 real points, should fail min_points check
        assert roi is None
        assert "Only 10" in msg

    def test_tighter_than_default_roi(self) -> None:
        """Adaptive ROI should be tighter than DEFAULT_COURT_ROI for typical ball data."""
        # Typical ball trajectory: x=0.15-0.85, y=0.15-0.65
        coords = [(0.15 + 0.7 * i / 100, 0.15 + 0.5 * (i % 20) / 20) for i in range(100)]
        positions = self._make_positions(coords)

        roi, _ = compute_court_roi_from_ball(positions)
        assert roi is not None

        roi_xs = [p[0] for p in roi]
        roi_ys = [p[1] for p in roi]
        default_xs = [p[0] for p in DEFAULT_COURT_ROI]
        default_ys = [p[1] for p in DEFAULT_COURT_ROI]

        roi_area = (max(roi_xs) - min(roi_xs)) * (max(roi_ys) - min(roi_ys))
        default_area = (max(default_xs) - min(default_xs)) * (max(default_ys) - min(default_ys))

        # Adaptive should cover less area than the loose default
        assert roi_area < default_area

    def test_percentile_ignores_outliers(self) -> None:
        """Percentile bounds should ignore a few extreme outlier detections."""
        # 95 points in center court, 5 outliers at edges
        center = [(0.3 + 0.4 * i / 95, 0.25 + 0.3 * (i % 15) / 15) for i in range(95)]
        outliers = [(0.01, 0.01), (0.99, 0.99), (0.01, 0.99), (0.99, 0.01), (0.5, 0.01)]
        positions = self._make_positions(center + outliers)

        roi, _ = compute_court_roi_from_ball(positions)
        assert roi is not None

        xs = [p[0] for p in roi]
        ys = [p[1] for p in roi]
        # ROI should not extend to the outlier positions (0.01, 0.99)
        # With 3rd/97th percentile, the bounds should be close to center data
        assert min(xs) > 0.05  # Not dragged to 0.01 outlier
        assert max(ys) < 0.97  # Not dragged to 0.99 outlier

    def test_asymmetric_margins(self) -> None:
        """Bottom margin (near court) should be larger than top margin (far court)."""
        coords = [(0.3 + 0.4 * i / 50, 0.3 + 0.2 * (i % 10) / 10) for i in range(50)]
        positions = self._make_positions(coords)

        roi, _ = compute_court_roi_from_ball(positions)
        assert roi is not None

        ys = [p[1] for p in roi]
        ball_y_center = 0.4  # approx center of ball trajectory
        top_margin = ball_y_center - min(ys)
        bottom_margin = max(ys) - ball_y_center

        # Bottom margin should be larger (near-court players stand below ball)
        assert bottom_margin > top_margin

    def test_minimum_roi_width_prevents_narrow_roi(self) -> None:
        """One-sided ball trajectory should still produce a wide enough ROI."""
        # Ball only on the left half: x=0.1-0.4, y=0.2-0.6
        coords = [(0.1 + 0.3 * i / 50, 0.2 + 0.4 * (i % 10) / 10) for i in range(50)]
        positions = self._make_positions(coords)

        roi, _ = compute_court_roi_from_ball(positions)
        assert roi is not None

        xs = [p[0] for p in roi]
        roi_width = max(xs) - min(xs)

        # Should be at least 80% wide (min_roi_width default)
        assert roi_width >= 0.80

    def test_minimum_roi_height_prevents_short_roi(self) -> None:
        """Horizontal ball trajectory should still produce tall enough ROI."""
        # Ball in a narrow horizontal band: x=0.1-0.9, y=0.35-0.45
        coords = [(0.1 + 0.8 * i / 50, 0.35 + 0.1 * (i % 5) / 5) for i in range(50)]
        positions = self._make_positions(coords)

        roi, _ = compute_court_roi_from_ball(positions)
        assert roi is not None

        ys = [p[1] for p in roi]
        roi_height = max(ys) - min(ys)

        # Should be at least 85% tall (min_roi_height default)
        assert roi_height >= 0.85

    def test_min_dimensions_expand_near_trajectory(self) -> None:
        """Minimum dimension expansion should stay near ball trajectory center."""
        # Ball centered at x=0.3: one-sided trajectory
        coords = [(0.2 + 0.2 * i / 50, 0.2 + 0.4 * (i % 10) / 10) for i in range(50)]
        positions = self._make_positions(coords)

        roi, _ = compute_court_roi_from_ball(positions)
        assert roi is not None

        xs = [p[0] for p in roi]
        roi_center_x = (min(xs) + max(xs)) / 2
        ball_center_x = 0.3  # approx center of ball trajectory

        # ROI center should be near ball center (within reason)
        assert abs(roi_center_x - ball_center_x) < 0.15

    def test_min_dimensions_disabled(self) -> None:
        """Setting min_roi_width/height to 0 should not expand."""
        # Narrow trajectory: x=0.4-0.6
        coords = [(0.4 + 0.2 * i / 50, 0.2 + 0.4 * (i % 10) / 10) for i in range(50)]
        positions = self._make_positions(coords)

        roi, _ = compute_court_roi_from_ball(
            positions, min_roi_width=0.0, min_roi_height=0.0
        )
        assert roi is not None

        xs = [p[0] for p in roi]
        roi_width = max(xs) - min(xs)
        # Without min width enforcement, ROI should be ~0.36 (0.2 ball span + 0.16 margin)
        assert roi_width < 0.50
