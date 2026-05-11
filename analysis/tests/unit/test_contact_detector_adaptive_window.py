"""Unit tests for the v3.0 adaptive candidate-generation window.

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md
"""

from __future__ import annotations

from rallycut.tracking.contact_detector import _collect_best_per_track
from rallycut.tracking.player_tracker import PlayerPosition


def _pos(frame: int, tid: int, x: float = 0.5, y: float = 0.5) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame, track_id=tid,
        x=x, y=y, width=0.05, height=0.10, confidence=1.0,
    )


class TestCollectBestPerTrack:
    """Helper returns best (closest depth-corrected) entry per track in window."""

    def test_returns_all_tracks_within_window(self) -> None:
        """All primary tracks visible within the window are collected."""
        positions = [
            _pos(frame=100, tid=1, x=0.50, y=0.50),
            _pos(frame=100, tid=2, x=0.60, y=0.50),
            _pos(frame=100, tid=3, x=0.70, y=0.50),
            _pos(frame=100, tid=4, x=0.80, y=0.50),
        ]
        result = _collect_best_per_track(
            player_positions=positions, frame=100,
            search_frames=15, ball_x=0.5, ball_y=0.5,
            primary_track_ids=[1, 2, 3, 4],
            lower_bound_frame=85,
            upper_bound_frame=115,
            court_calibrator=None,
        )
        assert set(result.keys()) == {1, 2, 3, 4}

    def test_excludes_non_primary_tracks(self) -> None:
        """Non-primary track ids are filtered out."""
        positions = [
            _pos(frame=100, tid=1), _pos(frame=100, tid=99),
        ]
        result = _collect_best_per_track(
            player_positions=positions, frame=100,
            search_frames=15, ball_x=0.5, ball_y=0.5,
            primary_track_ids=[1, 2, 3, 4],
            lower_bound_frame=85,
            upper_bound_frame=115,
            court_calibrator=None,
        )
        assert set(result.keys()) == {1}

    def test_excludes_positions_outside_window(self) -> None:
        """Positions outside [lower_bound_frame, upper_bound_frame] excluded."""
        positions = [
            _pos(frame=84, tid=1),   # before lower bound
            _pos(frame=100, tid=2),  # in window
            _pos(frame=116, tid=3),  # after upper bound
        ]
        result = _collect_best_per_track(
            player_positions=positions, frame=100,
            search_frames=15, ball_x=0.5, ball_y=0.5,
            primary_track_ids=[1, 2, 3, 4],
            lower_bound_frame=85, upper_bound_frame=115,
            court_calibrator=None,
        )
        assert set(result.keys()) == {2}

    def test_keeps_best_per_track_when_multiple_frames(self) -> None:
        """For a single track with multiple positions, keep the one with smallest distance."""
        positions = [
            _pos(frame=98, tid=1, x=0.80, y=0.50),  # far from ball
            _pos(frame=100, tid=1, x=0.51, y=0.50),  # close to ball
            _pos(frame=102, tid=1, x=0.60, y=0.50),  # medium
        ]
        result = _collect_best_per_track(
            player_positions=positions, frame=100,
            search_frames=15, ball_x=0.5, ball_y=0.5,
            primary_track_ids=[1, 2, 3, 4],
            lower_bound_frame=85, upper_bound_frame=115,
            court_calibrator=None,
        )
        assert 1 in result
        # Best (closest to ball) entry kept; its img_dist should be the smallest.
        # The result dict's value is (rank_dist, img_dist, center_y).
        _rank_dist, img_dist, _y = result[1]
        # Closest position (frame=100, x=0.51) → img_dist ≈ |0.51 - 0.5| = 0.01 (after upper-quarter bbox shift adds a small offset).
        assert img_dist < 0.05
