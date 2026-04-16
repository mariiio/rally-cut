"""Unit tests for the trajectory-continuity term in global-identity cost.

The trajectory cost penalises assignments where a candidate segment's start
position doesn't match the extrapolated trajectory of the profile's anchor
segment. It is intentionally active only for short temporal gaps — for large
gaps (> 15 frames), the player's position may have drifted and the penalty
should degrade to zero.
"""

from __future__ import annotations

from rallycut.tracking.global_identity import (
    PlayerProfile,
    TrackSegment,
    _compute_trajectory_cost,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _pp(frame: int, track_id: int, x: float, y: float) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame, track_id=track_id,
        x=x, y=y, width=0.05, height=0.15, confidence=0.8,
    )


def _seg(start_frame: int, track_id: int, positions: list[PlayerPosition]) -> TrackSegment:
    return TrackSegment(
        track_id=track_id,
        start_frame=start_frame,
        end_frame=start_frame + len(positions) - 1,
        team=0,
        positions=positions,
    )


class TestTrajectoryCostBasics:
    def test_returns_zero_when_profile_has_no_anchor_history(self) -> None:
        """Graceful fallback: no prior positions → no penalty."""
        seg = _seg(50, 10, [_pp(50, 10, 0.5, 0.5)])
        profile = PlayerProfile(player_id=1, team=0, centroid=(0.5, 0.5))
        # No anchor_end_frame / anchor_last_positions set.
        assert _compute_trajectory_cost(seg, profile) == 0.0

    def test_returns_zero_when_segment_has_no_positions(self) -> None:
        seg = _seg(50, 10, [])
        profile = PlayerProfile(
            player_id=1, team=0, centroid=(0.5, 0.5),
            anchor_end_frame=40,
            anchor_last_positions=[_pp(36, 1, 0.4, 0.5), _pp(40, 1, 0.5, 0.5)],
        )
        assert _compute_trajectory_cost(seg, profile) == 0.0


class TestTrajectoryCostLargeGap:
    def test_large_gap_returns_zero(self) -> None:
        """> 15-frame gap: trajectory doesn't apply; penalty must be 0."""
        seg = _seg(100, 10, [_pp(100, 10, 0.5, 0.5)])
        profile = PlayerProfile(
            player_id=1, team=0, centroid=(0.5, 0.5),
            anchor_end_frame=40,  # 60-frame gap
            anchor_last_positions=[_pp(36, 1, 0.0, 0.5), _pp(40, 1, 0.1, 0.5)],
        )
        assert _compute_trajectory_cost(seg, profile) == 0.0


class TestTrajectoryCostShortGap:
    def test_continuation_has_low_cost(self) -> None:
        """Segment starts right where extrapolation predicts → near-zero cost."""
        # Anchor ends at frame 50 at x=0.5, moving right at 0.01/frame.
        anchor_positions = [
            _pp(46, 1, 0.46, 0.5), _pp(47, 1, 0.47, 0.5),
            _pp(48, 1, 0.48, 0.5), _pp(49, 1, 0.49, 0.5),
            _pp(50, 1, 0.50, 0.5),
        ]
        # Segment starts at frame 55 at x=0.55 (consistent with continuation).
        seg = _seg(55, 10, [_pp(55, 10, 0.55, 0.5)])
        profile = PlayerProfile(
            player_id=1, team=0, centroid=(0.48, 0.5),
            anchor_end_frame=50,
            anchor_last_positions=anchor_positions,
        )
        cost = _compute_trajectory_cost(seg, profile)
        assert cost < 0.1

    def test_jump_has_high_cost(self) -> None:
        """Segment starts far from extrapolation → high cost."""
        # Anchor ends at frame 50 at x=0.1, moving slowly.
        anchor_positions = [
            _pp(46, 1, 0.06, 0.5), _pp(47, 1, 0.07, 0.5),
            _pp(48, 1, 0.08, 0.5), _pp(49, 1, 0.09, 0.5),
            _pp(50, 1, 0.10, 0.5),
        ]
        # Segment starts 2 frames later at x=0.8 (implausible jump across the court).
        seg = _seg(52, 10, [_pp(52, 10, 0.8, 0.5)])
        profile = PlayerProfile(
            player_id=1, team=0, centroid=(0.08, 0.5),
            anchor_end_frame=50,
            anchor_last_positions=anchor_positions,
        )
        cost = _compute_trajectory_cost(seg, profile)
        assert cost > 0.5

    def test_stationary_player_stays_stationary(self) -> None:
        """Zero velocity + same position → zero cost."""
        anchor_positions = [_pp(48, 1, 0.3, 0.3) for _ in range(3)]
        seg = _seg(50, 10, [_pp(50, 10, 0.3, 0.3)])
        profile = PlayerProfile(
            player_id=1, team=0, centroid=(0.3, 0.3),
            anchor_end_frame=48,
            anchor_last_positions=anchor_positions,
        )
        cost = _compute_trajectory_cost(seg, profile)
        assert cost < 0.05

    def test_negative_gap_returns_zero(self) -> None:
        """If the candidate segment starts BEFORE the anchor ended, trajectory
        doesn't apply (the segment can't be a continuation). Return 0."""
        anchor_positions = [_pp(50, 1, 0.5, 0.5)]
        seg = _seg(30, 10, [_pp(30, 10, 0.5, 0.5)])
        profile = PlayerProfile(
            player_id=1, team=0, centroid=(0.5, 0.5),
            anchor_end_frame=50,
            anchor_last_positions=anchor_positions,
        )
        assert _compute_trajectory_cost(seg, profile) == 0.0
