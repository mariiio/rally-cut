"""Unit tests for the F1 stable-presence appearance filter (MATCHER_VERSION v9).

These tests exercise the module-level helper
``is_appearance_clean_position`` rather than the full
``extract_rally_appearances`` pipeline, so they don't require a real
video file. The closure inside ``extract_rally_appearances`` is a thin
curry over this helper.
"""
from __future__ import annotations

import pytest

from rallycut.tracking.match_tracker import (
    APPEARANCE_FILTER_MAX_OCCLUSION_IOU,
    APPEARANCE_FILTER_MIN_AREA_FRACTION,
    APPEARANCE_FILTER_MIN_ASPECT_RATIO,
    is_appearance_clean_position,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _pos(
    *,
    track_id: int = 1,
    frame: int = 0,
    x: float = 0.5,
    y: float = 0.5,
    width: float = 0.05,
    height: float = 0.15,
    confidence: float = 0.9,
) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame,
        track_id=track_id,
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=confidence,
    )


class TestAspectRatioFilter:
    def test_clean_bbox_accepted(self) -> None:
        p = _pos(width=0.05, height=0.15)  # aspect 3.0, well above 1.4
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, **ctx) is True

    def test_squashed_bbox_rejected(self) -> None:
        p = _pos(width=0.10, height=0.10)  # aspect 1.0
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, **ctx) is False

    def test_degenerate_bbox_rejected(self) -> None:
        p = _pos(width=0.0, height=0.15)
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, **ctx) is False


class TestConfidenceFilter:
    def test_low_confidence_rejected(self) -> None:
        p = _pos(confidence=0.3)  # below 0.5
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, **ctx) is False


class TestOcclusionFilter:
    def test_overlap_above_threshold_rejected(self) -> None:
        p1 = _pos(track_id=1, x=0.5, y=0.5, width=0.10, height=0.20)
        p2 = _pos(track_id=2, x=0.51, y=0.5, width=0.10, height=0.20)  # high IoU
        ctx = {
            "primary_by_frame": {0: [(1, p1), (2, p2)]},
            "median_area_by_track": {},
        }
        assert is_appearance_clean_position(1, p1, **ctx) is False

    def test_no_overlap_accepted(self) -> None:
        p1 = _pos(track_id=1, x=0.2, width=0.05, height=0.15)
        p2 = _pos(track_id=2, x=0.8, width=0.05, height=0.15)  # far apart
        ctx = {
            "primary_by_frame": {0: [(1, p1), (2, p2)]},
            "median_area_by_track": {},
        }
        assert is_appearance_clean_position(1, p1, **ctx) is True


class TestAreaFloor:
    """F1, v9 — bbox area must be at least 0.5x the track's median area."""

    def test_below_floor_rejected(self) -> None:
        # Track 1's median area is 0.05 * 0.15 = 0.0075.
        # The candidate has area 0.02 * 0.06 = 0.0012, which is well below
        # 0.5 * median = 0.00375.
        p = _pos(track_id=1, width=0.02, height=0.06)  # aspect 3.0, passes aspect gate
        ctx = {
            "primary_by_frame": {0: [(1, p)]},
            "median_area_by_track": {1: 0.05 * 0.15},
        }
        assert is_appearance_clean_position(1, p, **ctx) is False

    def test_at_median_accepted(self) -> None:
        p = _pos(track_id=1, width=0.05, height=0.15)  # at median
        ctx = {
            "primary_by_frame": {0: [(1, p)]},
            "median_area_by_track": {1: 0.05 * 0.15},
        }
        assert is_appearance_clean_position(1, p, **ctx) is True

    def test_just_above_floor_accepted(self) -> None:
        # Median 0.0075. Floor at 0.00375. Candidate at 0.005 (>= floor).
        median = 0.05 * 0.15
        floor = median * APPEARANCE_FILTER_MIN_AREA_FRACTION
        # Width 0.04, height 0.13 → area 0.0052 (above floor) and aspect 3.25 (above gate).
        p = _pos(track_id=1, width=0.04, height=0.13)
        ctx = {
            "primary_by_frame": {0: [(1, p)]},
            "median_area_by_track": {1: median},
        }
        assert p.width * p.height >= floor  # sanity
        assert is_appearance_clean_position(1, p, **ctx) is True

    def test_empty_median_skips_floor_check(self) -> None:
        # When the median is unavailable for the track, the area-floor
        # check is skipped (forward-compat: very-short tracks).
        p = _pos(width=0.02, height=0.06)
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, **ctx) is True


class _FakeCalibrator:
    """Minimal stand-in for CourtCalibrator that returns hand-crafted court points."""

    def __init__(
        self,
        *,
        in_court: bool,
        is_calibrated: bool = True,
        raise_on_project: bool = False,
    ) -> None:
        self._in_court = in_court
        self.is_calibrated = is_calibrated
        self._raise = raise_on_project

    def image_to_court(
        self,
        image_point: tuple[float, float],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        if self._raise:
            raise RuntimeError("not calibrated (simulated)")
        # Return a sentinel value that the membership check below ignores.
        return (0.0, 0.0)

    def is_point_in_court_with_margin(
        self,
        court_point: tuple[float, float],
        *,
        sideline_margin: float,
        baseline_margin: float,
    ) -> bool:
        return self._in_court


class TestCourtInteriorGate:
    """F1, v9 — bbox foot point must project inside court polygon when calibrator provided."""

    def test_no_calibrator_skips_check(self) -> None:
        p = _pos()
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, calibrator=None, **ctx) is True

    def test_uncalibrated_calibrator_skips_check(self) -> None:
        p = _pos()
        cal = _FakeCalibrator(in_court=False, is_calibrated=False)
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, calibrator=cal, **ctx) is True

    def test_outside_court_rejected(self) -> None:
        p = _pos()
        cal = _FakeCalibrator(in_court=False)
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, calibrator=cal, **ctx) is False

    def test_inside_court_accepted(self) -> None:
        p = _pos()
        cal = _FakeCalibrator(in_court=True)
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, calibrator=cal, **ctx) is True

    def test_homography_runtime_error_accepts_position(self) -> None:
        # If the calibrator raises during projection (e.g. uncalibrated
        # mid-rally state), the filter degrades open rather than dropping
        # all positions.
        p = _pos()
        cal = _FakeCalibrator(in_court=False, raise_on_project=True)
        ctx = {"primary_by_frame": {0: [(1, p)]}, "median_area_by_track": {}}
        assert is_appearance_clean_position(1, p, calibrator=cal, **ctx) is True


class TestConstants:
    def test_aspect_ratio_constant_matches_documented_default(self) -> None:
        # Sanity: the v9 documentation in MATCHER_VERSION history refers
        # to "aspect ratio >= 1.4" as the existing pre-v9 threshold.
        assert APPEARANCE_FILTER_MIN_ASPECT_RATIO == pytest.approx(1.4)
        assert APPEARANCE_FILTER_MAX_OCCLUSION_IOU == pytest.approx(0.3)
        assert APPEARANCE_FILTER_MIN_AREA_FRACTION == pytest.approx(0.5)
