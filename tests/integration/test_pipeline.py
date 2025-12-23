"""Integration tests for the analysis pipeline."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile

from rallycut.core.models import (
    Action,
    ActionType,
    BallPosition,
    GameState,
    GameStateResult,
    TimeSegment,
    VideoInfo,
)


class TestPipelineIntegration:
    """Integration tests for the full analysis pipeline."""

    @pytest.fixture
    def mock_video_info(self):
        """Create mock video info."""
        return VideoInfo(
            path=Path("/test/video.mp4"),
            duration=60.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=1800,
        )

    @pytest.fixture
    def mock_game_state_results(self):
        """Create mock game state results."""
        return [
            GameStateResult(GameState.NO_PLAY, 0.9, start_frame=0, end_frame=149),
            GameStateResult(GameState.SERVICE, 0.85, start_frame=150, end_frame=299),
            GameStateResult(GameState.PLAY, 0.95, start_frame=300, end_frame=599),
            GameStateResult(GameState.NO_PLAY, 0.88, start_frame=600, end_frame=749),
            GameStateResult(GameState.PLAY, 0.92, start_frame=750, end_frame=1199),
            GameStateResult(GameState.NO_PLAY, 0.9, start_frame=1200, end_frame=1800),
        ]

    @pytest.fixture
    def mock_actions(self):
        """Create mock detected actions."""
        return [
            Action(ActionType.SERVE, 160, 5.33, 0.9),
            Action(ActionType.RECEPTION, 200, 6.67, 0.85),
            Action(ActionType.SET, 250, 8.33, 0.8),
            Action(ActionType.ATTACK, 350, 11.67, 0.95),
            Action(ActionType.SERVE, 760, 25.33, 0.88),
            Action(ActionType.RECEPTION, 820, 27.33, 0.82),
            Action(ActionType.ATTACK, 950, 31.67, 0.9),
        ]

    @pytest.fixture
    def mock_segments(self):
        """Create mock segments for highlights."""
        return [
            TimeSegment(150, 600, 5.0, 20.0, GameState.PLAY),
            TimeSegment(750, 1200, 25.0, 40.0, GameState.PLAY),
        ]

    @pytest.fixture
    def mock_ball_positions(self):
        """Create mock ball tracking data."""
        return [
            BallPosition(x=500, y=300, confidence=0.9, frame_idx=i)
            for i in range(30)
        ]

    def test_statistics_flow(
        self,
        mock_video_info,
        mock_game_state_results,
        mock_actions,
    ):
        """Test statistics generation flow."""
        from rallycut.statistics.aggregator import StatisticsAggregator
        from rallycut.analysis.game_state import GameStateAnalyzer

        # Get segments from game state results
        analyzer = GameStateAnalyzer(device="cpu")
        segments = analyzer.get_segments(mock_game_state_results, mock_video_info.fps)

        # Aggregate statistics
        aggregator = StatisticsAggregator(mock_video_info)
        rallies = aggregator.create_rallies(mock_actions, segments)
        stats = aggregator.compute_statistics(mock_actions, segments)

        # Verify statistics
        assert stats.video_info == mock_video_info
        assert stats.total_rallies >= 0
        assert stats.serves.count == 2
        assert stats.attacks.count == 2

    def test_statistics_to_dict(
        self,
        mock_video_info,
        mock_actions,
        mock_segments,
    ):
        """Test statistics export to dictionary."""
        from rallycut.statistics.aggregator import StatisticsAggregator

        aggregator = StatisticsAggregator(mock_video_info)
        stats = aggregator.compute_statistics(mock_actions, mock_segments)
        result = aggregator.to_dict(stats)

        # Verify dict structure
        assert "video" in result
        assert "summary" in result
        assert "actions" in result
        assert "rallies" in result
        assert result["actions"]["serves"] == 2

    def test_highlight_scoring_flow(self, mock_segments):
        """Test highlight scoring flow."""
        from rallycut.processing.highlights import HighlightScorer

        scorer = HighlightScorer(min_duration=1.0)
        top_highlights = scorer.get_top_highlights(mock_segments, count=5)

        # Verify scoring
        assert len(top_highlights) == 2
        assert all(h.score > 0 for h in top_highlights)
        assert all(h.rank > 0 for h in top_highlights)

    def test_trajectory_processing_flow(self, mock_ball_positions):
        """Test trajectory processing flow."""
        from rallycut.tracking.trajectory import TrajectoryProcessor

        processor = TrajectoryProcessor(
            smooth_sigma=1.5,
            max_gap_frames=10,
        )

        # Process trajectory
        segments = processor.process(
            mock_ball_positions,
            interpolate=True,
            smooth=True,
        )

        # Should return list of trajectory segments
        assert len(segments) >= 1
        # Each segment has positions
        assert len(segments[0].positions) > 0


class TestCLICommands:
    """Tests for CLI command structure."""

    def test_cli_app_exists(self):
        """Test that CLI app is properly defined."""
        from rallycut.cli.main import app

        assert app is not None

    def test_commands_registered(self):
        """Test that main commands are registered."""
        from rallycut.cli.main import app

        # Get registered command names (from sub-apps)
        command_names = [cmd.name for cmd in app.registered_commands]

        # Check that key commands exist
        expected_commands = ["cut", "stats", "overlay", "highlights"]
        for cmd in expected_commands:
            assert cmd in command_names, f"Command '{cmd}' not found in CLI"


class TestErrorHandling:
    """Tests for error handling across the pipeline."""

    def test_missing_video_file(self):
        """Test handling of missing video file."""
        from rallycut.core.video import Video

        with pytest.raises(FileNotFoundError):
            Video(Path("/nonexistent/video.mp4"))

    def test_empty_segment_list_for_highlights(self):
        """Test handling empty segment list for highlights."""
        from rallycut.processing.highlights import HighlightScorer

        scorer = HighlightScorer()
        result = scorer.get_top_highlights([], count=10)

        # Should return empty list, not raise
        assert result == []

    def test_highlight_scorer_with_short_segments(self):
        """Test that short segments are filtered."""
        from rallycut.processing.highlights import HighlightScorer

        segments = [
            TimeSegment(0, 30, 0.0, 1.0, GameState.PLAY),  # 1s - too short
            TimeSegment(60, 180, 2.0, 6.0, GameState.PLAY),  # 4s - long enough
        ]

        scorer = HighlightScorer(min_duration=3.0)
        result = scorer.get_top_highlights(segments, count=10)

        # Only the longer segment should be included
        assert len(result) == 1
        assert result[0].duration == 4.0


class TestCutterIntegration:
    """Tests for video cutter functionality."""

    @pytest.fixture
    def mock_results(self):
        """Create mock game state results for cutter."""
        fps = 30.0
        return [
            # Frame 0-149: NO_PLAY
            GameStateResult(GameState.NO_PLAY, 0.9, start_frame=0, end_frame=149),
            # Frame 150-599: PLAY (15s rally)
            GameStateResult(GameState.PLAY, 0.95, start_frame=150, end_frame=599),
            # Frame 600-749: NO_PLAY
            GameStateResult(GameState.NO_PLAY, 0.88, start_frame=600, end_frame=749),
            # Frame 750-1199: PLAY (15s rally)
            GameStateResult(GameState.PLAY, 0.92, start_frame=750, end_frame=1199),
        ], fps

    def test_segment_extraction(self, mock_results):
        """Test that cutter extracts correct segments."""
        from rallycut.processing.cutter import VideoCutter

        results, fps = mock_results

        cutter = VideoCutter(
            padding_seconds=1.0,
            min_play_duration=2.0,
            stride=16,
        )

        segments = cutter._get_segments_from_results(results, fps)

        # Should have 2 play segments
        assert len(segments) == 2

        # Check first segment has padding
        assert segments[0].start_frame < 150  # Padded before
        assert segments[0].end_frame > 599  # Padded after

    def test_segment_merging(self):
        """Test that overlapping segments are merged."""
        from rallycut.processing.cutter import VideoCutter
        from rallycut.core.models import GameStateResult

        # Two play segments close together (would overlap with padding)
        fps = 30.0
        results = [
            GameStateResult(GameState.PLAY, 0.95, start_frame=0, end_frame=100),
            GameStateResult(GameState.NO_PLAY, 0.88, start_frame=100, end_frame=120),  # Short gap
            GameStateResult(GameState.PLAY, 0.92, start_frame=120, end_frame=250),
        ]

        cutter = VideoCutter(
            padding_seconds=1.0,
            min_play_duration=1.0,
        )

        segments = cutter._get_segments_from_results(results, fps)

        # Should be merged into 1 segment due to short gap and padding
        assert len(segments) == 1

    def test_cut_stats(self):
        """Test cut statistics calculation."""
        from rallycut.processing.cutter import VideoCutter

        segments = [
            TimeSegment(0, 300, 0.0, 10.0, GameState.PLAY),
            TimeSegment(600, 900, 20.0, 30.0, GameState.PLAY),
        ]

        cutter = VideoCutter()
        stats = cutter.get_cut_stats(60.0, segments)

        assert stats["original_duration"] == 60.0
        assert stats["kept_duration"] == 20.0
        assert stats["removed_duration"] == 40.0
        assert stats["segment_count"] == 2
