"""Unit tests for per-player and per-match statistics."""

from __future__ import annotations

from rallycut.statistics.match_stats import (
    PlayerStats,
    compute_match_stats,
    compute_player_movement,
    compute_position_heatmap,
)
from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    RallyActions,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _pp(frame: int, track_id: int, x: float, y: float) -> PlayerPosition:
    """Helper to create a PlayerPosition."""
    return PlayerPosition(
        frame_number=frame, track_id=track_id,
        x=x, y=y, width=0.05, height=0.15, confidence=0.9,
    )


def _action(
    action_type: ActionType,
    frame: int,
    player: int = 1,
    court_side: str = "near",
    velocity: float = 0.02,
) -> ClassifiedAction:
    """Helper to create a ClassifiedAction."""
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=velocity,
        player_track_id=player,
        court_side=court_side,
        confidence=0.9,
    )


class TestPlayerStats:
    """Tests for PlayerStats dataclass."""

    def test_total_actions(self) -> None:
        """total_actions sums all action counts."""
        ps = PlayerStats(track_id=1, serves=2, receives=3, sets=1, attacks=4, blocks=1, digs=2)
        assert ps.total_actions == 13

    def test_to_dict(self) -> None:
        """to_dict produces expected keys."""
        ps = PlayerStats(track_id=1, serves=2, attacks=3)
        d = ps.to_dict()
        assert d["trackId"] == 1
        assert d["serves"] == 2
        assert d["attacks"] == 3
        assert d["totalActions"] == 5


class TestComputePlayerMovement:
    """Tests for player movement distance computation."""

    def test_stationary_player(self) -> None:
        """Player at same position has zero movement."""
        positions = [_pp(i, 1, 0.5, 0.5) for i in range(10)]
        dist_px, dist_m, avg_speed = compute_player_movement(positions, 1)
        assert dist_px == 0.0
        assert avg_speed == 0.0

    def test_horizontal_movement(self) -> None:
        """Player moving horizontally accumulates distance."""
        positions = [_pp(i, 1, 0.1 + i * 0.01, 0.5) for i in range(10)]
        dist_px, dist_m, avg_speed = compute_player_movement(
            positions, 1, video_width=1920, video_height=1080,
        )
        # 9 moves of 0.01 * 1920 = 19.2px each = 172.8px total
        assert dist_px > 100

    def test_ignores_other_tracks(self) -> None:
        """Only computes for specified track_id."""
        positions = [
            _pp(0, 1, 0.1, 0.5),
            _pp(1, 1, 0.2, 0.5),
            _pp(0, 2, 0.3, 0.5),
            _pp(1, 2, 0.9, 0.5),  # Large movement for track 2
        ]
        dist_px, _, _ = compute_player_movement(positions, 1, video_width=1920, video_height=1080)
        # Only track 1: 0.1 * 1920 = 192px
        assert 150 < dist_px < 250

    def test_skips_large_frame_gaps(self) -> None:
        """Frame gaps > 5 are not counted."""
        positions = [
            _pp(0, 1, 0.1, 0.5),
            _pp(10, 1, 0.9, 0.5),  # Gap of 10 frames
        ]
        dist_px, _, _ = compute_player_movement(positions, 1)
        assert dist_px == 0.0

    def test_single_position(self) -> None:
        """Single position returns zero distance."""
        positions = [_pp(0, 1, 0.5, 0.5)]
        dist_px, _, _ = compute_player_movement(positions, 1)
        assert dist_px == 0.0


class TestComputePositionHeatmap:
    """Tests for position heatmap computation."""

    def test_empty_track(self) -> None:
        """Empty positions return zero grid."""
        heatmap = compute_position_heatmap([], track_id=1)
        assert len(heatmap) == 10
        assert all(all(v == 0.0 for v in row) for row in heatmap)

    def test_single_position_concentrated(self) -> None:
        """Single position creates single cell with 1.0."""
        positions = [_pp(0, 1, 0.5, 0.5)]
        heatmap = compute_position_heatmap(positions, track_id=1)
        total = sum(sum(row) for row in heatmap)
        assert abs(total - 1.0) < 0.001  # Normalized to 1.0

    def test_distributed_positions(self) -> None:
        """Multiple positions create distributed heatmap."""
        positions = [
            _pp(0, 1, 0.1, 0.1),
            _pp(1, 1, 0.9, 0.9),
        ]
        heatmap = compute_position_heatmap(positions, track_id=1)
        total = sum(sum(row) for row in heatmap)
        assert abs(total - 1.0) < 0.001
        # Two cells should each have 0.5
        assert heatmap[1][1] == 0.5  # (0.1 → grid[1][1])
        assert heatmap[9][9] == 0.5  # (0.9 → grid[9][9])

    def test_ignores_other_tracks(self) -> None:
        """Only uses specified track_id."""
        positions = [
            _pp(0, 1, 0.5, 0.5),
            _pp(0, 2, 0.1, 0.1),
        ]
        heatmap = compute_position_heatmap(positions, track_id=1)
        # Only one position counted
        non_zero = sum(1 for row in heatmap for v in row if v > 0)
        assert non_zero == 1


class TestComputeMatchStats:
    """Tests for comprehensive match statistics computation."""

    def _make_rally_actions(self) -> list[RallyActions]:
        """Create sample rally actions for testing."""
        rally1 = RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=1, court_side="near"),
                _action(ActionType.RECEIVE, 30, player=3, court_side="far"),
                _action(ActionType.SET, 45, player=4, court_side="far"),
                _action(ActionType.ATTACK, 55, player=3, court_side="far", velocity=0.04),
            ],
            rally_id="rally-1",
        )
        rally2 = RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=3, court_side="far"),
                _action(ActionType.RECEIVE, 25, player=2, court_side="near"),
                _action(ActionType.SET, 35, player=1, court_side="near"),
                _action(ActionType.ATTACK, 45, player=2, court_side="near", velocity=0.035),
                _action(ActionType.BLOCK, 50, player=4, court_side="far"),
            ],
            rally_id="rally-2",
        )
        return [rally1, rally2]

    def _make_player_positions(self) -> list[PlayerPosition]:
        """Create sample player positions."""
        positions = []
        for i in range(60):
            positions.append(_pp(i, 1, 0.4 + i * 0.001, 0.7))  # Near court
            positions.append(_pp(i, 2, 0.6 - i * 0.001, 0.75))  # Near court
            positions.append(_pp(i, 3, 0.4, 0.3))  # Far court (stationary)
            positions.append(_pp(i, 4, 0.6, 0.25))  # Far court
        return positions

    def test_player_action_counts(self) -> None:
        """Per-player action counts are correct."""
        actions = self._make_rally_actions()
        positions = self._make_player_positions()
        stats = compute_match_stats(actions, positions)

        player_map = {p.track_id: p for p in stats.player_stats}
        assert player_map[1].serves == 1
        assert player_map[1].sets == 1
        assert player_map[3].serves == 1
        assert player_map[3].receives == 1
        assert player_map[3].attacks == 1
        assert player_map[4].blocks == 1

    def test_rally_stats(self) -> None:
        """Per-rally stats are computed correctly."""
        actions = self._make_rally_actions()
        positions = self._make_player_positions()
        stats = compute_match_stats(actions, positions, video_fps=30.0)

        assert stats.total_rallies == 2
        assert stats.total_contacts == 8  # 4 + (5-1 block) = 8

        # Rally 2 has a block
        rally2 = stats.rally_stats[1]
        assert rally2.has_block is True

    def test_match_aggregates(self) -> None:
        """Match-level aggregates are computed."""
        actions = self._make_rally_actions()
        positions = self._make_player_positions()
        stats = compute_match_stats(actions, positions, video_fps=30.0)

        assert stats.avg_rally_duration_s > 0
        assert stats.longest_rally_duration_s >= stats.avg_rally_duration_s
        assert stats.avg_contacts_per_rally > 0

    def test_player_court_side(self) -> None:
        """Players assigned correct court side based on avg Y position."""
        actions = self._make_rally_actions()
        positions = self._make_player_positions()
        stats = compute_match_stats(actions, positions)

        player_map = {p.track_id: p for p in stats.player_stats}
        assert player_map[1].court_side == "near"  # y=0.7
        assert player_map[3].court_side == "far"   # y=0.3

    def test_empty_actions(self) -> None:
        """Empty actions produce empty stats."""
        stats = compute_match_stats([], [])
        assert stats.total_rallies == 0
        assert stats.total_contacts == 0
        assert len(stats.player_stats) == 0

    def test_to_dict(self) -> None:
        """MatchStats.to_dict produces expected structure."""
        actions = self._make_rally_actions()
        positions = self._make_player_positions()
        stats = compute_match_stats(actions, positions)
        d = stats.to_dict()

        assert "totalRallies" in d
        assert "playerStats" in d
        assert "rallyStats" in d
        assert len(d["playerStats"]) == len(stats.player_stats)


class TestExtendedExchange:
    """Tests for extended exchange detection (3+ contacts same side)."""

    def test_three_contacts_same_side(self) -> None:
        """3 consecutive contacts on same side = extended exchange."""
        actions = [RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=1, court_side="near"),
                _action(ActionType.RECEIVE, 30, player=3, court_side="far"),
                _action(ActionType.SET, 45, player=4, court_side="far"),
                _action(ActionType.ATTACK, 55, player=3, court_side="far"),
            ],
            rally_id="r1",
        )]
        stats = compute_match_stats(actions, [])
        assert stats.rally_stats[0].has_extended_exchange is True

    def test_two_contacts_same_side(self) -> None:
        """Only 2 consecutive contacts on same side = not extended."""
        actions = [RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=1, court_side="near"),
                _action(ActionType.RECEIVE, 30, player=3, court_side="far"),
                _action(ActionType.DIG, 45, player=1, court_side="near"),
            ],
            rally_id="r1",
        )]
        stats = compute_match_stats(actions, [])
        assert stats.rally_stats[0].has_extended_exchange is False
