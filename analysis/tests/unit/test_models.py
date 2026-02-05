"""Tests for core domain models."""

from pathlib import Path

from rallycut.core.models import (
    Action,
    ActionCount,
    ActionType,
    GameState,
    GameStateResult,
    MatchStatistics,
    Rally,
    TimeSegment,
    VideoInfo,
)


class TestGameState:
    """Tests for GameState enum."""

    def test_game_state_values(self):
        """Test enum values."""
        assert GameState.SERVICE.value == "service"
        assert GameState.PLAY.value == "play"
        assert GameState.NO_PLAY.value == "no_play"

    def test_game_state_from_string(self):
        """Test creating GameState from string."""
        assert GameState("service") == GameState.SERVICE
        assert GameState("play") == GameState.PLAY
        assert GameState("no_play") == GameState.NO_PLAY


class TestActionType:
    """Tests for ActionType enum."""

    def test_action_type_values(self):
        """Test enum values."""
        assert ActionType.SERVE.value == "serve"
        assert ActionType.RECEPTION.value == "reception"
        assert ActionType.SET.value == "set"
        assert ActionType.ATTACK.value == "attack"
        assert ActionType.BLOCK.value == "block"
        assert ActionType.BALL.value == "ball"


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""

    def test_video_info_creation(self):
        """Test creating VideoInfo."""
        info = VideoInfo(
            path=Path("/test/video.mp4"),
            duration=120.5,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=3615,
            codec="h264",
        )

        assert info.path == Path("/test/video.mp4")
        assert info.duration == 120.5
        assert info.fps == 30.0
        assert info.width == 1920
        assert info.height == 1080
        assert info.frame_count == 3615
        assert info.codec == "h264"

    def test_video_info_resolution(self):
        """Test resolution property."""
        info = VideoInfo(
            path=Path("/test/video.mp4"),
            duration=120.5,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=3615,
            codec="h264",
        )

        assert info.resolution == (1920, 1080)


class TestGameStateResult:
    """Tests for GameStateResult dataclass."""

    def test_game_state_result_creation(self):
        """Test creating GameStateResult."""
        result = GameStateResult(
            state=GameState.PLAY,
            confidence=0.95,
            start_frame=100,
            end_frame=200,
        )

        assert result.state == GameState.PLAY
        assert result.confidence == 0.95
        assert result.start_frame == 100
        assert result.end_frame == 200

    def test_game_state_result_frame_range(self):
        """Test frame_range property."""
        result = GameStateResult(
            state=GameState.PLAY,
            confidence=0.95,
            start_frame=100,
            end_frame=200,
        )

        assert result.frame_range == (100, 200)


class TestTimeSegment:
    """Tests for TimeSegment dataclass."""

    def test_time_segment_creation(self):
        """Test creating TimeSegment."""
        segment = TimeSegment(
            start_frame=0,
            end_frame=300,
            start_time=0.0,
            end_time=10.0,
            state=GameState.PLAY,
        )

        assert segment.start_frame == 0
        assert segment.end_frame == 300
        assert segment.start_time == 0.0
        assert segment.end_time == 10.0
        assert segment.state == GameState.PLAY

    def test_time_segment_duration(self):
        """Test duration property."""
        segment = TimeSegment(
            start_frame=0,
            end_frame=300,
            start_time=5.0,
            end_time=15.0,
            state=GameState.PLAY,
        )

        assert segment.duration == 10.0

    def test_time_segment_frame_count(self):
        """Test frame_count property."""
        segment = TimeSegment(
            start_frame=0,
            end_frame=300,
            start_time=0.0,
            end_time=10.0,
            state=GameState.PLAY,
        )

        assert segment.frame_count == 301  # Inclusive


class TestAction:
    """Tests for Action dataclass."""

    def test_action_creation(self):
        """Test creating Action."""
        action = Action(
            action_type=ActionType.SERVE,
            frame_idx=150,
            timestamp=5.0,
            confidence=0.88,
        )

        assert action.action_type == ActionType.SERVE
        assert action.frame_idx == 150
        assert action.timestamp == 5.0
        assert action.confidence == 0.88
        assert action.bbox is None

    def test_action_position_without_bbox(self):
        """Test position property without bbox."""
        action = Action(
            action_type=ActionType.ATTACK,
            frame_idx=200,
            timestamp=6.67,
            confidence=0.92,
        )

        assert action.position is None


class TestRally:
    """Tests for Rally dataclass."""

    def test_rally_creation(self):
        """Test creating Rally."""
        rally = Rally(
            rally_id=1,
            start_frame=0,
            end_frame=300,
            start_time=0.0,
            end_time=10.0,
        )

        assert rally.rally_id == 1
        assert rally.start_frame == 0
        assert rally.end_frame == 300
        assert rally.start_time == 0.0
        assert rally.end_time == 10.0
        assert rally.actions == []

    def test_rally_duration(self):
        """Test duration property."""
        rally = Rally(
            rally_id=1,
            start_frame=0,
            end_frame=300,
            start_time=5.0,
            end_time=12.5,
        )

        assert rally.duration == 7.5

    def test_rally_with_actions(self):
        """Test Rally with actions."""
        actions = [
            Action(ActionType.SERVE, 10, 0.33, 0.9),
            Action(ActionType.RECEPTION, 50, 1.67, 0.85),
            Action(ActionType.SET, 80, 2.67, 0.8),
            Action(ActionType.ATTACK, 120, 4.0, 0.95),
        ]

        rally = Rally(
            rally_id=1,
            start_frame=0,
            end_frame=150,
            start_time=0.0,
            end_time=5.0,
            actions=actions,
        )

        assert len(rally.actions) == 4
        assert rally.actions[0].action_type == ActionType.SERVE

    def test_rally_action_sequence(self):
        """Test action_sequence property."""
        actions = [
            Action(ActionType.ATTACK, 120, 4.0, 0.95),
            Action(ActionType.SERVE, 10, 0.33, 0.9),
            Action(ActionType.SET, 80, 2.67, 0.8),
        ]

        rally = Rally(
            rally_id=1,
            start_frame=0,
            end_frame=150,
            start_time=0.0,
            end_time=5.0,
            actions=actions,
        )

        # Should be sorted by frame_idx
        sequence = rally.action_sequence
        assert sequence == [ActionType.SERVE, ActionType.SET, ActionType.ATTACK]


class TestActionCount:
    """Tests for ActionCount dataclass."""

    def test_action_count_creation(self):
        """Test creating ActionCount."""
        count = ActionCount(count=24)

        assert count.count == 24
        assert count.success_count is None
        assert count.success_rate is None

    def test_action_count_with_success(self):
        """Test ActionCount with success count."""
        count = ActionCount(
            count=24,
            success_count=22,
        )

        assert count.count == 24
        assert count.success_count == 22
        assert count.success_rate == 22 / 24

    def test_action_count_zero_count(self):
        """Test ActionCount with zero count doesn't divide by zero."""
        count = ActionCount(count=0, success_count=0)

        assert count.success_rate is None


class TestMatchStatistics:
    """Tests for MatchStatistics dataclass."""

    def test_match_statistics_creation(self):
        """Test creating MatchStatistics."""
        video_info = VideoInfo(
            path=Path("/test/video.mp4"),
            duration=3600.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=108000,
            codec="h264",
        )

        stats = MatchStatistics(
            video_info=video_info,
            total_rallies=50,
            total_duration=3600.0,
            play_duration=1800.0,
            dead_time_duration=1800.0,
            serves=ActionCount(48),
            receptions=ActionCount(45, success_count=40),
            sets=ActionCount(120),
            attacks=ActionCount(95, success_count=43),
            blocks=ActionCount(15, success_count=3),
            avg_rally_duration=36.0,
            longest_rally_duration=120.0,
            shortest_rally_duration=5.0,
            touches_per_rally=6.5,
        )

        assert stats.total_rallies == 50
        assert stats.play_duration == 1800.0
        assert stats.serves.count == 48
        assert stats.attacks.success_rate == 43 / 95

    def test_match_statistics_dead_time_percentage(self):
        """Test dead_time_percentage property."""
        video_info = VideoInfo(
            path=Path("/test/video.mp4"),
            duration=100.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=3000,
            codec="h264",
        )

        stats = MatchStatistics(
            video_info=video_info,
            total_rallies=10,
            total_duration=100.0,
            play_duration=40.0,
            dead_time_duration=60.0,
            serves=ActionCount(10),
            receptions=ActionCount(8),
            sets=ActionCount(20),
            attacks=ActionCount(15),
            blocks=ActionCount(3),
            avg_rally_duration=4.0,
            longest_rally_duration=10.0,
            shortest_rally_duration=2.0,
            touches_per_rally=5.0,
        )

        assert stats.dead_time_percentage == 60.0

    def test_match_statistics_zero_duration(self):
        """Test dead_time_percentage with zero duration."""
        video_info = VideoInfo(
            path=Path("/test/video.mp4"),
            duration=0.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=0,
            codec="h264",
        )

        stats = MatchStatistics(
            video_info=video_info,
            total_rallies=0,
            total_duration=0.0,
            play_duration=0.0,
            dead_time_duration=0.0,
        )

        assert stats.dead_time_percentage == 0.0
