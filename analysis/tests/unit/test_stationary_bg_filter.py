"""Tests for the stationary background filter in player_filter.py."""

from __future__ import annotations

from rallycut.tracking.player_filter import PlayerFilterConfig, remove_stationary_background_tracks
from rallycut.tracking.player_tracker import PlayerPosition


def _make_positions(
    track_id: int,
    n_frames: int,
    x: float,
    y: float,
    height: float = 0.20,
    x_jitter: float = 0.0,
    y_jitter: float = 0.0,
) -> list[PlayerPosition]:
    import random
    rng = random.Random(track_id * 1000 + n_frames)
    return [
        PlayerPosition(
            frame_number=f,
            track_id=track_id,
            x=x + rng.uniform(-x_jitter, x_jitter),
            y=y + rng.uniform(-y_jitter, y_jitter),
            width=0.04,
            height=height,
            confidence=0.9,
        )
        for f in range(n_frames)
    ]


def _moving_players(start_id: int = 10, n: int = 5) -> list[PlayerPosition]:
    """Generate 5 clearly-moving tracks so safety net (< max_players) never triggers."""
    positions: list[PlayerPosition] = []
    for i in range(n):
        positions += _make_positions(
            start_id + i, 100, x=0.20 + i * 0.12, y=0.30 + i * 0.08,
            height=0.20, x_jitter=0.03, y_jitter=0.03,
        )
    return positions


class TestStationaryBackgroundFilter:
    def test_sideline_person_removed(self) -> None:
        """Person-sized track at sideline (x=0.95) with very low spread → removed."""
        config = PlayerFilterConfig()
        sideline = _make_positions(1, 100, x=0.95, y=0.50, height=0.20, x_jitter=0.0005, y_jitter=0.0005)
        positions = sideline + _moving_players()
        result, removed = remove_stationary_background_tracks(positions, config, 100)
        assert 1 in removed, "sideline person-sized track should be removed"

    def test_court_center_person_spared(self) -> None:
        """Person-sized track at court center (x=0.50) with very low spread → spared.

        This is the a43fb033 scenario: player standing still on court waiting
        for serve. Should NOT be killed even if spread < person_spread threshold.
        """
        config = PlayerFilterConfig()
        still_player = _make_positions(1, 100, x=0.50, y=0.48, height=0.19, x_jitter=0.002, y_jitter=0.002)
        positions = still_player + _moving_players()
        result, removed = remove_stationary_background_tracks(positions, config, 100)
        assert 1 not in removed, "court-center person-sized track should be spared"

    def test_small_object_center_still_removed(self) -> None:
        """Small non-person object at court center → still removed."""
        config = PlayerFilterConfig()
        small_obj = _make_positions(1, 100, x=0.50, y=0.50, height=0.05, x_jitter=0.0005, y_jitter=0.0005)
        positions = small_obj + _moving_players()
        result, removed = remove_stationary_background_tracks(positions, config, 100)
        assert 1 in removed, "small non-person object should be removed even at court center"

    def test_near_sideline_person_removed(self) -> None:
        """Person at x=0.92 (near sideline) with zero spread → removed."""
        config = PlayerFilterConfig()
        near_side = _make_positions(1, 100, x=0.92, y=0.50, height=0.20, x_jitter=0.0005, y_jitter=0.0005)
        positions = near_side + _moving_players()
        result, removed = remove_stationary_background_tracks(positions, config, 100)
        assert 1 in removed, "near-sideline person should be removed"

    def test_left_sideline_person_removed(self) -> None:
        """Person at x=0.07 (left sideline) → removed."""
        config = PlayerFilterConfig()
        left_side = _make_positions(1, 100, x=0.07, y=0.50, height=0.20, x_jitter=0.0005, y_jitter=0.0005)
        positions = left_side + _moving_players()
        result, removed = remove_stationary_background_tracks(positions, config, 100)
        assert 1 in removed, "left-sideline person should be removed"

    def test_zero_spread_court_center_removed(self) -> None:
        """Person-sized at court center but with near-zero spread → removed.

        A true background object (sign, seated person) inside the court area
        with spread < court_interior_spread threshold should still be killed.
        """
        config = PlayerFilterConfig()
        bg_obj = _make_positions(1, 100, x=0.50, y=0.50, height=0.15, x_jitter=0.0001, y_jitter=0.0001)
        positions = bg_obj + _moving_players()
        result, removed = remove_stationary_background_tracks(positions, config, 100)
        assert 1 in removed, "zero-spread person-sized object at court center should be removed"

    def test_person_with_enough_spread_always_spared(self) -> None:
        """Person-sized track with spread >= person_spread (0.003) is spared everywhere."""
        config = PlayerFilterConfig()
        moving_enough = _make_positions(1, 100, x=0.95, y=0.50, height=0.20, x_jitter=0.008, y_jitter=0.008)
        positions = moving_enough + _moving_players()
        result, removed = remove_stationary_background_tracks(positions, config, 100)
        assert 1 not in removed, "person with spread >= threshold should be spared even at sideline"
