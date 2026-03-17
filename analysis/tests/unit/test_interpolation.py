"""Tests for player gap interpolation."""

from __future__ import annotations

import pytest

from rallycut.tracking.player_filter import (
    PlayerFilterConfig,
    interpolate_player_gaps,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _pos(frame: int, track_id: int, x: float = 0.5, y: float = 0.5) -> PlayerPosition:
    """Create a PlayerPosition with sensible defaults."""
    return PlayerPosition(
        frame_number=frame,
        track_id=track_id,
        x=x,
        y=y,
        width=0.10,
        height=0.20,
        confidence=0.90,
    )


class TestInterpolatePlayerGaps:
    """Tests for interpolate_player_gaps."""

    def test_empty_positions(self) -> None:
        result, count = interpolate_player_gaps([], {1, 2})
        assert result == []
        assert count == 0

    def test_empty_primary_ids(self) -> None:
        positions = [_pos(0, 1), _pos(1, 1)]
        result, count = interpolate_player_gaps(positions, set())
        assert result == positions
        assert count == 0

    def test_no_gaps(self) -> None:
        """Consecutive frames produce no interpolation."""
        positions = [_pos(f, 1) for f in range(10)]
        result, count = interpolate_player_gaps(positions, {1})
        assert count == 0
        assert len(result) == 10

    def test_single_frame_gap(self) -> None:
        """A 2-frame gap (1 missing frame) is interpolated."""
        positions = [_pos(0, 1, x=0.2), _pos(2, 1, x=0.4)]
        result, count = interpolate_player_gaps(positions, {1})
        assert count == 1
        assert len(result) == 3
        interp = [p for p in result if p.frame_number == 1][0]
        assert interp.track_id == 1
        assert interp.confidence == 0.5
        assert abs(interp.x - 0.3) < 1e-6  # Midpoint

    def test_multi_frame_gap(self) -> None:
        """A 4-frame gap (3 missing frames) is filled."""
        positions = [_pos(0, 1, x=0.0, y=0.0), _pos(4, 1, x=0.4, y=0.8)]
        result, count = interpolate_player_gaps(positions, {1})
        assert count == 3
        # Check linear interpolation at t=0.5 (frame 2)
        f2 = [p for p in result if p.frame_number == 2][0]
        assert abs(f2.x - 0.2) < 1e-6
        assert abs(f2.y - 0.4) < 1e-6

    def test_gap_exceeds_max(self) -> None:
        """Gaps larger than max_interpolation_gap are not filled."""
        config = PlayerFilterConfig(max_interpolation_gap=5)
        positions = [_pos(0, 1), _pos(10, 1)]
        result, count = interpolate_player_gaps(positions, {1}, config)
        assert count == 0
        assert len(result) == 2

    def test_gap_at_max_boundary(self) -> None:
        """Gap exactly at max_interpolation_gap is filled."""
        config = PlayerFilterConfig(max_interpolation_gap=5)
        positions = [_pos(0, 1), _pos(5, 1)]
        result, count = interpolate_player_gaps(positions, {1}, config)
        assert count == 4  # frames 1,2,3,4

    def test_non_primary_tracks_ignored(self) -> None:
        """Only primary tracks are interpolated."""
        positions = [
            _pos(0, 1), _pos(5, 1),   # primary, has gap
            _pos(0, 2), _pos(5, 2),   # non-primary, has gap
        ]
        result, count = interpolate_player_gaps(positions, {1})
        assert count == 4  # Only track 1 interpolated
        interp_track_ids = {p.track_id for p in result if p.confidence == 0.5}
        assert interp_track_ids == {1}

    def test_multiple_tracks(self) -> None:
        """Multiple primary tracks with gaps are all interpolated."""
        positions = [
            _pos(0, 1), _pos(3, 1),
            _pos(0, 2), _pos(3, 2),
        ]
        result, count = interpolate_player_gaps(positions, {1, 2})
        assert count == 4  # 2 gaps per track
        interp_track_ids = {p.track_id for p in result if p.confidence == 0.5}
        assert interp_track_ids == {1, 2}

    def test_multiple_gaps_same_track(self) -> None:
        """Multiple gaps in one track are each filled."""
        positions = [_pos(0, 1), _pos(3, 1), _pos(6, 1)]
        result, count = interpolate_player_gaps(positions, {1})
        assert count == 4  # 2 + 2

    def test_interpolation_disabled(self) -> None:
        config = PlayerFilterConfig(enable_interpolation=False)
        positions = [_pos(0, 1), _pos(5, 1)]
        result, count = interpolate_player_gaps(positions, {1}, config)
        assert count == 0
        assert len(result) == 2

    def test_width_height_interpolated(self) -> None:
        """Bbox dimensions are also linearly interpolated."""
        p1 = PlayerPosition(
            frame_number=0, track_id=1, x=0.5, y=0.5,
            width=0.10, height=0.20, confidence=0.9,
        )
        p2 = PlayerPosition(
            frame_number=2, track_id=1, x=0.5, y=0.5,
            width=0.14, height=0.28, confidence=0.9,
        )
        result, count = interpolate_player_gaps([p1, p2], {1})
        assert count == 1
        interp = [p for p in result if p.frame_number == 1][0]
        assert abs(interp.width - 0.12) < 1e-6
        assert abs(interp.height - 0.24) < 1e-6

    def test_output_sorted(self) -> None:
        """Result is sorted by (frame_number, track_id)."""
        positions = [_pos(0, 2), _pos(3, 2), _pos(0, 1), _pos(3, 1)]
        result, _ = interpolate_player_gaps(positions, {1, 2})
        frames_and_ids = [(p.frame_number, p.track_id) for p in result]
        assert frames_and_ids == sorted(frames_and_ids)

    def test_single_frame_track_ignored(self) -> None:
        """A track with only one detection has nothing to interpolate."""
        positions = [_pos(5, 1)]
        result, count = interpolate_player_gaps(positions, {1})
        assert count == 0
        assert len(result) == 1

    def test_accepts_list_primary_ids(self) -> None:
        """primary_track_ids can be a list (from DB storage)."""
        positions = [_pos(0, 1), _pos(3, 1)]
        result, count = interpolate_player_gaps(positions, [1])
        assert count == 2

    def test_original_positions_preserved(self) -> None:
        """Original positions are not modified, only new ones added."""
        positions = [_pos(0, 1, x=0.2), _pos(3, 1, x=0.5)]
        result, count = interpolate_player_gaps(positions, {1})
        originals = [p for p in result if p.confidence == 0.9]
        assert len(originals) == 2
        assert originals[0].x == 0.2
        assert originals[1].x == 0.5
