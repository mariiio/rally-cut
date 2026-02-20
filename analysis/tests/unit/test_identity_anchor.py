"""Tests for serve-based identity anchoring."""

from __future__ import annotations

from rallycut.tracking.identity_anchor import (
    ServiceOrderState,
    detect_serve_anchor,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _make_positions(
    track_id: int,
    frames: range,
    x: float = 0.5,
    y: float = 0.5,
) -> list[PlayerPosition]:
    return [
        PlayerPosition(f, track_id, x, y, 0.05, 0.15, 0.9)
        for f in frames
    ]


class TestServiceOrderState:
    def test_empty_prediction(self) -> None:
        state = ServiceOrderState()
        tid, conf = state.predict_server_track(0, [1, 2])
        assert tid == -1
        assert conf == 0.0

    def test_single_serve_no_prediction(self) -> None:
        state = ServiceOrderState()
        state.record_serve(0, 0, 1)
        tid, conf = state.predict_server_track(0, [1, 2])
        assert tid == -1  # Need at least 2 serves

    def test_alternation_prediction(self) -> None:
        state = ServiceOrderState()
        state.record_serve(0, 0, 1)  # Rally 0: track 1 served
        state.record_serve(2, 0, 2)  # Rally 2: track 2 served
        # Rally 4: should predict track 1 (same as second-to-last)
        tid, conf = state.predict_server_track(0, [1, 2])
        assert tid == 1
        assert conf > 0.5

    def test_same_server_twice_predicts_other(self) -> None:
        state = ServiceOrderState()
        state.record_serve(0, 0, 1)
        state.record_serve(2, 0, 1)  # Same player served twice
        # Should predict the OTHER player
        tid, conf = state.predict_server_track(0, [1, 2])
        assert tid == 2
        assert conf > 0.5


class TestDetectServeAnchor:
    def test_empty_positions_returns_none(self) -> None:
        result = detect_serve_anchor([], {})
        assert result is None

    def test_no_team_assignments_returns_none(self) -> None:
        positions = _make_positions(1, range(0, 30))
        result = detect_serve_anchor(positions, {})
        assert result is None

    def test_near_team_server_at_high_y(self) -> None:
        """Near-team player at high Y (baseline) should be detected as server."""
        positions = (
            # Track 1: near-team player at baseline (high Y)
            _make_positions(1, range(0, 30), x=0.5, y=0.85)
            # Track 2: near-team player at mid-court
            + _make_positions(2, range(0, 30), x=0.5, y=0.65)
            # Track 3: far-team player
            + _make_positions(3, range(0, 30), x=0.5, y=0.35)
            # Track 4: far-team player
            + _make_positions(4, range(0, 30), x=0.5, y=0.40)
        )
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        anchor = detect_serve_anchor(positions, team_assignments)
        assert anchor is not None
        assert anchor.server_track_id == 1  # Closest to near baseline
        assert anchor.server_team == 0
        assert anchor.confidence > 0.3

    def test_far_team_server_at_low_y(self) -> None:
        """Far-team player at low Y should be detected as server."""
        positions = (
            # Near-team players — at mid-court (not near baseline)
            _make_positions(1, range(0, 30), x=0.5, y=0.62)
            + _make_positions(2, range(0, 30), x=0.5, y=0.60)
            # Far-team: track 3 at baseline (very low Y)
            + _make_positions(3, range(0, 30), x=0.5, y=0.20)
            # Far-team: track 4 at mid-court
            + _make_positions(4, range(0, 30), x=0.5, y=0.40)
        )
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        anchor = detect_serve_anchor(positions, team_assignments)
        assert anchor is not None
        assert anchor.server_track_id == 3
        assert anchor.server_team == 1

    def test_teammate_identified(self) -> None:
        """Should identify the teammate (same team, not server)."""
        positions = (
            _make_positions(1, range(0, 30), x=0.5, y=0.85)
            + _make_positions(2, range(0, 30), x=0.5, y=0.65)
            + _make_positions(3, range(0, 30), x=0.5, y=0.35)
        )
        team_assignments = {1: 0, 2: 0, 3: 1}

        anchor = detect_serve_anchor(positions, team_assignments)
        assert anchor is not None
        assert anchor.server_track_id == 1
        assert anchor.teammate_track_id == 2

    def test_receivers_identified(self) -> None:
        """Should identify receivers (opposite team)."""
        positions = (
            _make_positions(1, range(0, 30), x=0.5, y=0.85)
            + _make_positions(2, range(0, 30), x=0.5, y=0.65)
            + _make_positions(3, range(0, 30), x=0.5, y=0.35)
            + _make_positions(4, range(0, 30), x=0.5, y=0.40)
        )
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        anchor = detect_serve_anchor(positions, team_assignments)
        assert anchor is not None
        assert set(anchor.receiver_track_ids) == {3, 4}
