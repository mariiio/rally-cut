"""Unit tests for the v3.0 adaptive candidate-generation window.

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md
"""

from __future__ import annotations

from unittest.mock import patch

from rallycut.tracking.contact_detector import (
    _collect_best_per_track,
    _find_nearest_player,
    _find_nearest_players,
)
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


class TestAdaptiveFallback:
    """Truth table for the adaptive forward-only fallback in _find_nearest_players."""

    def test_no_fallback_when_standard_window_full(self) -> None:
        """If standard ±15 returns all 4 primary tracks, no expansion occurs."""
        # All 4 primaries visible in the standard window.
        positions = [
            _pos(frame=100, tid=1, x=0.51),
            _pos(frame=100, tid=2, x=0.55),
            _pos(frame=100, tid=3, x=0.60),
            _pos(frame=100, tid=4, x=0.70),
            # A late-tracked candidate (tid=5) that WOULD be picked up by fallback.
            _pos(frame=150, tid=5, x=0.49),  # closest to ball, but should be excluded.
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4, 5],
            )
        tids = [tid for tid, _d, _y in result]
        # tid=5 was outside the standard window AND fallback didn't fire
        # (because the standard window already had 4 candidates).
        assert tids == [1, 2, 3, 4]

    def test_fallback_fires_when_underfull(self) -> None:
        """When standard ±15 returns fewer than max_candidates, expand forward."""
        # Only 3 tracks visible in ±15; tid=4 only visible later.
        positions = [
            _pos(frame=100, tid=1, x=0.55),
            _pos(frame=100, tid=2, x=0.60),
            _pos(frame=100, tid=3, x=0.70),
            _pos(frame=150, tid=4, x=0.51),  # 50 frames forward — within +60.
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4],
            )
        tids = [tid for tid, _d, _y in result]
        # tid=4 (closest by raw distance from frame 150) should now appear.
        # Pass 1 entries (tids 1, 2, 3) take precedence; tid=4 is added via fallback.
        # Final ranking is by distance — tid=4's bbox-corrected distance is smallest.
        assert 4 in tids
        assert set(tids) == {1, 2, 3, 4}

    def test_fallback_forward_only_excludes_backward_window(self) -> None:
        """Fallback expands FORWARD only — never looks earlier than the contact."""
        # Underfull standard window (only tid=1, 2 within ±15).
        # tid=3 is BEFORE contact (frame=70, gap=30 backward).
        # tid=4 is AFTER contact (frame=150, gap=50 forward) — within +60.
        positions = [
            _pos(frame=100, tid=1, x=0.55),
            _pos(frame=100, tid=2, x=0.60),
            _pos(frame=70, tid=3, x=0.51),   # backward — should be excluded
            _pos(frame=150, tid=4, x=0.52),  # forward — should be included
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4],
            )
        tids = [tid for tid, _d, _y in result]
        # tid=3 (backward) excluded; tid=4 (forward) included.
        assert 3 not in tids
        assert 4 in tids

    def test_env_flag_off_disables_fallback(self) -> None:
        """ADAPTIVE_PLAYER_SEARCH_WINDOW=0 restores pre-v3 behavior."""
        positions = [
            _pos(frame=100, tid=1, x=0.55),
            _pos(frame=100, tid=2, x=0.60),
            _pos(frame=150, tid=4, x=0.51),  # only seen in forward window
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "0"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4],
            )
        tids = [tid for tid, _d, _y in result]
        # Pre-v3 behavior: tid=4 NOT included (outside ±15 window).
        assert 4 not in tids

    def test_pass1_entries_take_precedence_over_pass2_for_same_track(self) -> None:
        """When the same track appears in both passes, Pass 1 (closer to contact) wins."""
        # tid=1 visible at frame=100 (Pass 1) AND frame=160 (Pass 2, +60).
        # Pass 1 entry should be retained.
        positions = [
            _pos(frame=100, tid=1, x=0.55, y=0.50),  # Pass 1 entry
            _pos(frame=100, tid=2, x=0.60),
            _pos(frame=160, tid=1, x=0.51, y=0.50),  # Pass 2 entry — closer to ball
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4],
            )
        # tid=1's distance should reflect the Pass 1 frame=100 position (x=0.55),
        # not the Pass 2 frame=160 position (x=0.51, which would be closer).
        tid1_entry = next((t for t in result if t[0] == 1), None)
        assert tid1_entry is not None
        _tid, img_dist, _y = tid1_entry
        # Distance from (x=0.55, y=0.50-0.10*0.25) to (0.5, 0.5) is ~0.057, not 0.012.
        assert img_dist > 0.04, "Pass 1 entry should be retained, not overwritten by Pass 2"


class TestFindNearestPlayerAdaptive:
    """The singular _find_nearest_player gets the same fallback behavior."""

    def test_singular_fallback_finds_late_tracked_player_when_underfull(self) -> None:
        """When ±15 has no candidates (or fewer than max_candidates if filtered),
        the singular form's fallback catches a late-tracked one."""
        positions = [
            # Only one primary visible in ±15 — underfull
            _pos(frame=100, tid=1, x=0.70),
            # tid=2 only visible later, but closer to ball
            _pos(frame=150, tid=2, x=0.51),
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            tid, dist, _center_y = _find_nearest_player(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15,
                primary_track_ids=[1, 2, 3, 4],
            )
        # Without fallback: tid=1 wins (it's the only candidate).
        # With fallback: tid=2 wins because its forward-window position is closer.
        assert tid == 2

    def test_singular_env_off_returns_only_pass1(self) -> None:
        """With env flag OFF, singular form returns only Pass 1 result."""
        positions = [
            _pos(frame=100, tid=1, x=0.70),
            _pos(frame=150, tid=2, x=0.51),  # forward-only, would be ignored
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "0"}):
            tid, _dist, _center_y = _find_nearest_player(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15,
                primary_track_ids=[1, 2, 3, 4],
            )
        # Pre-v3 behavior: only tid=1 (the sole candidate in ±15).
        assert tid == 1

    def test_singular_no_primary_track_filter_uses_max_candidates(self) -> None:
        """When primary_track_ids is None, the fallback uses max_candidates as
        the underfull threshold (consistent with _find_nearest_players)."""
        # Singular form doesn't take max_candidates explicitly — it just picks
        # the single nearest. The adaptive fallback should still fire if Pass 1
        # is empty.
        positions = [
            _pos(frame=150, tid=1, x=0.51),  # forward-only
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            tid, _dist, _y = _find_nearest_player(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15,
                primary_track_ids=None,
            )
        # Pass 1 empty → fallback fires → tid=1 found.
        assert tid == 1
