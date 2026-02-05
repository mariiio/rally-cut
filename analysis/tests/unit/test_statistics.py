"""Tests for statistics aggregator."""

from pathlib import Path

import pytest

from rallycut.core.models import (
    Action,
    ActionType,
    GameState,
    Rally,
    TimeSegment,
    VideoInfo,
)
from rallycut.statistics.aggregator import StatisticsAggregator


class TestStatisticsAggregator:
    """Tests for StatisticsAggregator."""

    @pytest.fixture
    def video_info(self):
        """Create sample video info."""
        return VideoInfo(
            path=Path("/test/video.mp4"),
            duration=120.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=3600,
        )

    @pytest.fixture
    def sample_segments(self):
        """Create sample time segments."""
        return [
            TimeSegment(
                start_frame=300,
                end_frame=600,
                start_time=10.0,
                end_time=20.0,
                state=GameState.SERVICE,
            ),
            TimeSegment(
                start_frame=1500,
                end_frame=2100,
                start_time=50.0,
                end_time=70.0,
                state=GameState.PLAY,
            ),
        ]

    @pytest.fixture
    def sample_actions(self):
        """Create sample actions."""
        return [
            Action(ActionType.SERVE, 320, 10.67, 0.9),
            Action(ActionType.RECEPTION, 380, 12.67, 0.85),
            Action(ActionType.SET, 450, 15.0, 0.8),
            Action(ActionType.ATTACK, 520, 17.33, 0.95),
            Action(ActionType.SERVE, 1520, 50.67, 0.88),
            Action(ActionType.RECEPTION, 1600, 53.33, 0.82),
            Action(ActionType.SET, 1700, 56.67, 0.78),
            Action(ActionType.ATTACK, 1800, 60.0, 0.9),
            Action(ActionType.BLOCK, 1850, 61.67, 0.75),
        ]

    def test_aggregator_creation(self, video_info):
        """Test creating aggregator."""
        aggregator = StatisticsAggregator(video_info)
        assert aggregator.video_info == video_info

    def test_create_rallies_from_segments(self, video_info, sample_segments, sample_actions):
        """Test rally creation from segments."""
        aggregator = StatisticsAggregator(video_info)
        rallies = aggregator.create_rallies(sample_actions, sample_segments)

        # Should have 2 rallies (one for each segment)
        assert len(rallies) == 2

        # Verify rally properties
        for rally in rallies:
            assert isinstance(rally, Rally)
            assert rally.start_time >= 0
            assert rally.end_time >= rally.start_time

    def test_create_rallies_from_gaps(self, video_info, sample_actions):
        """Test rally creation based on time gaps."""
        aggregator = StatisticsAggregator(video_info)
        rallies = aggregator.create_rallies(sample_actions, rally_gap_seconds=5.0)

        # Should have 2 rallies (gap between 17.33 and 50.67 is > 5s)
        assert len(rallies) == 2

    def test_count_actions(self, video_info, sample_actions):
        """Test action counting."""
        aggregator = StatisticsAggregator(video_info)
        counts = aggregator.count_actions(sample_actions)

        assert counts[ActionType.SERVE].count == 2
        assert counts[ActionType.RECEPTION].count == 2
        assert counts[ActionType.SET].count == 2
        assert counts[ActionType.ATTACK].count == 2
        assert counts[ActionType.BLOCK].count == 1

    def test_compute_statistics(self, video_info, sample_segments, sample_actions):
        """Test statistics computation."""
        aggregator = StatisticsAggregator(video_info)
        stats = aggregator.compute_statistics(sample_actions, sample_segments)

        # Verify statistics structure
        assert stats.video_info == video_info
        assert stats.total_rallies == 2
        assert stats.play_duration > 0
        assert stats.dead_time_duration >= 0

        # Verify action counts
        assert stats.serves.count == 2
        assert stats.receptions.count == 2
        assert stats.sets.count == 2
        assert stats.attacks.count == 2
        assert stats.blocks.count == 1

    def test_empty_actions(self, video_info, sample_segments):
        """Test with no actions."""
        aggregator = StatisticsAggregator(video_info)
        stats = aggregator.compute_statistics([], sample_segments)

        # No rallies when no actions (even with segments)
        assert stats.total_rallies == 0
        assert stats.serves.count == 0
        assert stats.attacks.count == 0
        # But play_duration is still calculated from segments
        assert stats.play_duration == 30.0  # 10s + 20s from segments

    def test_empty_segments(self, video_info, sample_actions):
        """Test with no segments (uses gap-based rallies)."""
        aggregator = StatisticsAggregator(video_info)
        stats = aggregator.compute_statistics(sample_actions, None)

        # Should create rallies from gaps
        assert stats.total_rallies >= 1

    def test_empty_all(self, video_info):
        """Test with no actions and no segments."""
        aggregator = StatisticsAggregator(video_info)
        stats = aggregator.compute_statistics([], None)

        assert stats.total_rallies == 0
        assert stats.play_duration == 0

    def test_rally_duration_calculation(self, video_info, sample_segments, sample_actions):
        """Test rally duration is calculated correctly."""
        aggregator = StatisticsAggregator(video_info)
        rallies = aggregator.create_rallies(sample_actions, sample_segments)

        for rally in rallies:
            expected_duration = rally.end_time - rally.start_time
            assert abs(rally.duration - expected_duration) < 0.01

    def test_to_dict(self, video_info, sample_segments, sample_actions):
        """Test conversion to dictionary."""
        aggregator = StatisticsAggregator(video_info)
        stats = aggregator.compute_statistics(sample_actions, sample_segments)
        result = aggregator.to_dict(stats)

        assert "video" in result
        assert "summary" in result
        assert "actions" in result
        assert "rallies" in result
        assert result["actions"]["serves"] == 2


class TestActionAssignmentToRallies:
    """Tests for correct action-to-rally assignment."""

    @pytest.fixture
    def video_info(self):
        return VideoInfo(
            path=Path("/test/video.mp4"),
            duration=30.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=900,
        )

    def test_actions_assigned_to_correct_rally(self, video_info):
        """Test that actions are assigned to the rally containing their frame."""
        segments = [
            TimeSegment(0, 100, 0.0, 3.33, GameState.PLAY),
            TimeSegment(150, 250, 5.0, 8.33, GameState.PLAY),
        ]

        actions = [
            Action(ActionType.SERVE, 50, 1.67, 0.9),  # Rally 1
            Action(ActionType.ATTACK, 80, 2.67, 0.85),  # Rally 1
            Action(ActionType.SERVE, 180, 6.0, 0.88),  # Rally 2
        ]

        aggregator = StatisticsAggregator(video_info)
        rallies = aggregator.create_rallies(actions, segments)

        # Verify actions in first rally
        assert len(rallies[0].actions) == 2

        # Verify action in second rally
        assert len(rallies[1].actions) == 1

    def test_actions_outside_rallies_not_assigned(self, video_info):
        """Test that actions outside rally boundaries aren't assigned."""
        segments = [
            TimeSegment(100, 200, 3.33, 6.67, GameState.PLAY),
        ]

        actions = [
            Action(ActionType.SERVE, 50, 1.67, 0.9),  # Before rally
            Action(ActionType.ATTACK, 150, 5.0, 0.85),  # In rally
            Action(ActionType.SERVE, 250, 8.33, 0.88),  # After rally
        ]

        aggregator = StatisticsAggregator(video_info)
        rallies = aggregator.create_rallies(actions, segments)

        # Only middle action should be in rally
        assert len(rallies) == 1
        assert len(rallies[0].actions) == 1
        assert rallies[0].actions[0].frame_idx == 150
