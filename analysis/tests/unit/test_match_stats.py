"""Unit tests for per-player and per-match statistics."""

from __future__ import annotations

from rallycut.statistics.match_stats import (
    PlayerStats,
    TeamStats,
    compute_match_scores,
    compute_match_stats,
    compute_player_movement,
    compute_position_heatmap,
)
from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    RallyActions,
    _team_label,
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
    team: str = "unknown",
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
        team=team,
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


class TestTeamLabel:
    """Tests for _team_label helper mapping."""

    def test_near_court_is_team_a(self) -> None:
        """Team 0 (near court) maps to 'A'."""
        assert _team_label(1, {1: 0, 2: 1}) == "A"

    def test_far_court_is_team_b(self) -> None:
        """Team 1 (far court) maps to 'B'."""
        assert _team_label(2, {1: 0, 2: 1}) == "B"

    def test_missing_track_is_unknown(self) -> None:
        """Track not in assignments returns 'unknown'."""
        assert _team_label(99, {1: 0, 2: 1}) == "unknown"

    def test_negative_track_is_unknown(self) -> None:
        """Negative track ID (e.g. synthetic serve) returns 'unknown'."""
        assert _team_label(-1, {1: 0, 2: 1}) == "unknown"

    def test_none_assignments_is_unknown(self) -> None:
        """None team_assignments returns 'unknown'."""
        assert _team_label(1, None) == "unknown"

    def test_empty_assignments_is_unknown(self) -> None:
        """Empty team_assignments returns 'unknown'."""
        assert _team_label(1, {}) == "unknown"


class TestActionTeamAttribution:
    """Tests for team field on ClassifiedAction."""

    def test_action_team_from_team_assignments(self) -> None:
        """Actions get correct team when team_assignments provided."""
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}  # 1,2=near/A, 3,4=far/B
        rally = RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=1, team="A"),
                _action(ActionType.RECEIVE, 30, player=3, team="B"),
                _action(ActionType.SET, 45, player=4, team="B"),
                _action(ActionType.ATTACK, 55, player=3, team="B"),
            ],
            rally_id="r1",
            team_assignments=team_assignments,
        )
        assert rally.actions[0].team == "A"
        assert rally.actions[1].team == "B"
        assert rally.actions[2].team == "B"
        assert rally.actions[3].team == "B"

    def test_action_team_unknown_without_assignments(self) -> None:
        """Actions default to 'unknown' without team_assignments."""
        action = _action(ActionType.SERVE, 5, player=1)
        assert action.team == "unknown"

    def test_serving_team_property(self) -> None:
        """RallyActions.serving_team returns serve's team."""
        rally = RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=1, team="A"),
                _action(ActionType.RECEIVE, 30, player=3, team="B"),
            ],
            rally_id="r1",
        )
        assert rally.serving_team == "A"

    def test_serving_team_none_when_unknown(self) -> None:
        """serving_team returns None when serve team is unknown."""
        rally = RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=1, team="unknown"),
            ],
            rally_id="r1",
        )
        assert rally.serving_team is None

    def test_serving_team_none_when_no_serve(self) -> None:
        """serving_team returns None when no serve action exists."""
        rally = RallyActions(
            actions=[_action(ActionType.RECEIVE, 30, player=3)],
            rally_id="r1",
        )
        assert rally.serving_team is None

    def test_actions_by_team(self) -> None:
        """actions_by_team filters correctly."""
        rally = RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=1, team="A"),
                _action(ActionType.RECEIVE, 30, player=3, team="B"),
                _action(ActionType.SET, 45, player=4, team="B"),
            ],
            rally_id="r1",
        )
        assert len(rally.actions_by_team("A")) == 1
        assert len(rally.actions_by_team("B")) == 2
        assert len(rally.actions_by_team("unknown")) == 0

    def test_to_dict_includes_team(self) -> None:
        """ClassifiedAction.to_dict includes team field."""
        action = _action(ActionType.SERVE, 5, player=1, team="A")
        d = action.to_dict()
        assert d["team"] == "A"

    def test_rally_to_dict_includes_team_data(self) -> None:
        """RallyActions.to_dict includes teamAssignments and servingTeam."""
        rally = RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=1, team="A"),
            ],
            rally_id="r1",
            team_assignments={1: 0, 3: 1},
        )
        d = rally.to_dict()
        assert d["teamAssignments"] == {"1": "A", "3": "B"}
        assert d["servingTeam"] == "A"


class TestComputeMatchScores:
    """Tests for serve-based score computation."""

    def test_serve_change_means_new_server_won(self) -> None:
        """If serving team changes, new server won previous point."""
        rallies = [
            RallyActions(
                actions=[_action(ActionType.SERVE, 5, player=1, team="A")],
                rally_id="r1",
            ),
            RallyActions(
                actions=[_action(ActionType.SERVE, 5, player=3, team="B")],
                rally_id="r2",
            ),
            RallyActions(
                actions=[_action(ActionType.SERVE, 5, player=1, team="A")],
                rally_id="r3",
            ),
        ]
        scores = compute_match_scores(rallies)
        assert len(scores) == 3
        # Rally 1: A served, then B serves next → B won rally 1
        assert scores[0].point_winner == "B"
        assert scores[0].score_a == 0
        assert scores[0].score_b == 1
        # Rally 2: B served, then A serves next → A won rally 2
        assert scores[1].point_winner == "A"
        assert scores[1].score_a == 1
        assert scores[1].score_b == 1
        # Rally 3: last rally → winner unknown
        assert scores[2].point_winner == "unknown"
        assert scores[2].score_a == 1
        assert scores[2].score_b == 1

    def test_serve_stays_means_server_scored(self) -> None:
        """If same team serves again, they won the point (side-out failed)."""
        rallies = [
            RallyActions(
                actions=[_action(ActionType.SERVE, 5, player=1, team="A")],
                rally_id="r1",
            ),
            RallyActions(
                actions=[_action(ActionType.SERVE, 5, player=2, team="A")],
                rally_id="r2",
            ),
        ]
        scores = compute_match_scores(rallies)
        assert len(scores) == 2
        # Rally 1: A served, then A serves again → A scored
        assert scores[0].point_winner == "A"
        assert scores[0].score_a == 1
        assert scores[0].score_b == 0

    def test_missing_serve_skipped(self) -> None:
        """Rallies without serving_team are skipped in scoring."""
        rallies = [
            RallyActions(
                actions=[_action(ActionType.SERVE, 5, player=1, team="A")],
                rally_id="r1",
            ),
            RallyActions(
                actions=[_action(ActionType.RECEIVE, 30, player=3)],  # No serve
                rally_id="r2",
            ),
            RallyActions(
                actions=[_action(ActionType.SERVE, 5, player=3, team="B")],
                rally_id="r3",
            ),
        ]
        scores = compute_match_scores(rallies)
        # Only 2 rallies with known serving team
        assert len(scores) == 2
        assert scores[0].rally_id == "r1"
        assert scores[0].point_winner == "B"  # B serves next
        assert scores[1].rally_id == "r3"
        assert scores[1].point_winner == "unknown"  # Last rally

    def test_empty_rallies(self) -> None:
        """Empty rally list returns empty scores."""
        scores = compute_match_scores([])
        assert scores == []

    def test_single_rally(self) -> None:
        """Single rally has unknown winner (no next rally)."""
        rallies = [
            RallyActions(
                actions=[_action(ActionType.SERVE, 5, player=1, team="A")],
                rally_id="r1",
            ),
        ]
        scores = compute_match_scores(rallies)
        assert len(scores) == 1
        assert scores[0].point_winner == "unknown"
        assert scores[0].score_a == 0
        assert scores[0].score_b == 0


class TestTeamStatsAggregation:
    """Tests for per-team statistics aggregation."""

    def test_team_stats_from_team_assignments(self) -> None:
        """Team stats aggregate correctly from player stats with team assignments."""
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}
        rally = RallyActions(
            actions=[
                _action(ActionType.SERVE, 5, player=1, team="A"),
                _action(ActionType.RECEIVE, 30, player=3, team="B"),
                _action(ActionType.SET, 45, player=4, team="B"),
                _action(ActionType.ATTACK, 55, player=3, team="B"),
            ],
            rally_id="r1",
            team_assignments=team_assignments,
        )
        positions = [
            _pp(i, 1, 0.4, 0.7) for i in range(30)
        ] + [
            _pp(i, 2, 0.6, 0.75) for i in range(30)
        ] + [
            _pp(i, 3, 0.4, 0.3) for i in range(30)
        ] + [
            _pp(i, 4, 0.6, 0.25) for i in range(30)
        ]

        stats = compute_match_stats([rally], positions)

        # Check player team labels
        player_map = {p.track_id: p for p in stats.player_stats}
        assert player_map[1].team == "A"
        assert player_map[2].team == "A"
        assert player_map[3].team == "B"
        assert player_map[4].team == "B"

        # Check team stats
        assert len(stats.team_stats) == 2
        team_map = {t.team: t for t in stats.team_stats}
        team_a = team_map["A"]
        team_b = team_map["B"]

        assert team_a.serves == 1
        assert team_a.receives == 0
        assert team_b.receives == 1
        assert team_b.sets == 1
        assert team_b.attacks == 1
        assert sorted(team_a.player_ids) == [1, 2]
        assert sorted(team_b.player_ids) == [3, 4]

    def test_team_stats_to_dict(self) -> None:
        """TeamStats.to_dict produces expected keys."""
        ts = TeamStats(team="A", player_ids=[1, 2], serves=3, attacks=5)
        d = ts.to_dict()
        assert d["team"] == "A"
        assert d["playerIds"] == [1, 2]
        assert d["serves"] == 3
        assert d["attacks"] == 5
        assert d["totalActions"] == 8

    def test_match_stats_to_dict_includes_team_data(self) -> None:
        """MatchStats.to_dict includes teamStats when present."""
        team_assignments = {1: 0, 3: 1}
        rallies = [
            RallyActions(
                actions=[
                    _action(ActionType.SERVE, 5, player=1, team="A"),
                    _action(ActionType.RECEIVE, 30, player=3, team="B"),
                ],
                rally_id="r1",
                team_assignments=team_assignments,
            ),
            RallyActions(
                actions=[
                    _action(ActionType.SERVE, 5, player=3, team="B"),
                    _action(ActionType.RECEIVE, 30, player=1, team="A"),
                ],
                rally_id="r2",
                team_assignments=team_assignments,
            ),
        ]
        positions = [
            _pp(0, 1, 0.5, 0.7),
            _pp(0, 3, 0.5, 0.3),
        ]
        stats = compute_match_stats(rallies, positions)
        d = stats.to_dict()
        assert "teamStats" in d
        assert len(d["teamStats"]) == 2
        assert "scoreProgression" in d
