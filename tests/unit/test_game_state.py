"""Tests for game state classifier."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from rallycut.core.models import GameState, GameStateResult, TimeSegment


class TestGameStateAnalyzer:
    """Tests for GameStateAnalyzer."""

    @pytest.fixture
    def mock_classifier(self):
        """Create mock classifier."""
        classifier = Mock()
        classifier.classify_segment.return_value = (GameState.PLAY, 0.95)
        return classifier

    @pytest.fixture
    def sample_results(self):
        """Create sample game state results."""
        return [
            GameStateResult(frame_idx=0, state=GameState.NO_PLAY, confidence=0.9),
            GameStateResult(frame_idx=16, state=GameState.NO_PLAY, confidence=0.85),
            GameStateResult(frame_idx=32, state=GameState.SERVICE, confidence=0.8),
            GameStateResult(frame_idx=48, state=GameState.PLAY, confidence=0.95),
            GameStateResult(frame_idx=64, state=GameState.PLAY, confidence=0.92),
            GameStateResult(frame_idx=80, state=GameState.PLAY, confidence=0.88),
            GameStateResult(frame_idx=96, state=GameState.NO_PLAY, confidence=0.9),
        ]

    def test_get_segments(self, sample_results):
        """Test segment creation from results."""
        from rallycut.analysis.game_state import GameStateAnalyzer

        analyzer = GameStateAnalyzer(device="cpu")
        fps = 30.0

        segments = analyzer.get_segments(sample_results, fps)

        # Should have 3 segments: NO_PLAY, SERVICE+PLAY, NO_PLAY
        assert len(segments) >= 1

        # Verify segment properties
        for segment in segments:
            assert isinstance(segment, TimeSegment)
            assert segment.start_frame >= 0
            assert segment.end_frame >= segment.start_frame
            assert segment.state in GameState

    def test_get_play_segments(self, sample_results):
        """Test filtering for play segments."""
        from rallycut.analysis.game_state import GameStateAnalyzer

        analyzer = GameStateAnalyzer(device="cpu")
        fps = 30.0

        all_segments = analyzer.get_segments(sample_results, fps)
        play_segments = analyzer.get_play_segments(
            all_segments,
            padding_seconds=0.5,
            min_duration_seconds=0.1,
            fps=fps,
        )

        # All returned segments should be play-related
        for segment in play_segments:
            assert segment.state in (GameState.SERVICE, GameState.PLAY)

    def test_get_play_segments_with_min_duration(self, sample_results):
        """Test min duration filtering."""
        from rallycut.analysis.game_state import GameStateAnalyzer

        analyzer = GameStateAnalyzer(device="cpu")
        fps = 30.0

        all_segments = analyzer.get_segments(sample_results, fps)

        # Very high min duration should filter everything
        play_segments = analyzer.get_play_segments(
            all_segments,
            padding_seconds=0,
            min_duration_seconds=1000.0,  # Impossibly long
            fps=fps,
        )

        assert len(play_segments) == 0

    def test_empty_results(self):
        """Test handling of empty results."""
        from rallycut.analysis.game_state import GameStateAnalyzer

        analyzer = GameStateAnalyzer(device="cpu")
        fps = 30.0

        segments = analyzer.get_segments([], fps)
        assert segments == []

    def test_single_result(self):
        """Test handling of single result."""
        from rallycut.analysis.game_state import GameStateAnalyzer

        analyzer = GameStateAnalyzer(device="cpu")
        fps = 30.0

        results = [
            GameStateResult(frame_idx=0, state=GameState.PLAY, confidence=0.9)
        ]

        segments = analyzer.get_segments(results, fps)
        assert len(segments) == 1
        assert segments[0].state == GameState.PLAY


class TestGameStateClassifierMocked:
    """Tests for classifier with mocked model."""

    @patch("rallycut.analysis.game_state.GameStateAnalyzer._get_classifier")
    def test_analyze_video_progress(self, mock_get_classifier):
        """Test progress callback during analysis."""
        from rallycut.analysis.game_state import GameStateAnalyzer
        from rallycut.core.models import VideoInfo
        from pathlib import Path

        # Mock classifier with batch method
        mock_classifier = Mock()
        # classify_segments_batch returns list of (state, confidence) tuples
        mock_classifier.classify_segments_batch.return_value = [
            (GameState.PLAY, 0.9)
        ]
        mock_get_classifier.return_value = mock_classifier

        # Mock video
        mock_video = Mock()
        mock_video.info = VideoInfo(
            path=Path("/test.mp4"),
            duration=1.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=30,
        )
        mock_video.read_frames.return_value = [
            np.zeros((224, 224, 3), dtype=np.uint8)
            for _ in range(16)
        ]

        # Track progress
        progress_calls = []

        def progress_callback(pct, msg):
            progress_calls.append((pct, msg))

        analyzer = GameStateAnalyzer(device="cpu")
        results = analyzer.analyze_video(mock_video, progress_callback=progress_callback)

        # Verify progress was reported
        assert len(progress_calls) > 0
        # Final progress should be close to 1.0
        if progress_calls:
            assert progress_calls[-1][0] <= 1.0
        # Should have at least one result
        assert len(results) >= 1


class TestSegmentMerging:
    """Tests for segment merging logic."""

    def test_merge_adjacent_same_state(self):
        """Test that adjacent segments of same state are merged."""
        from rallycut.analysis.game_state import GameStateAnalyzer

        analyzer = GameStateAnalyzer(device="cpu")

        results = [
            GameStateResult(frame_idx=0, state=GameState.PLAY, confidence=0.9),
            GameStateResult(frame_idx=16, state=GameState.PLAY, confidence=0.85),
            GameStateResult(frame_idx=32, state=GameState.PLAY, confidence=0.88),
        ]

        segments = analyzer.get_segments(results, fps=30.0)

        # Should be merged into single segment
        assert len(segments) == 1
        assert segments[0].state == GameState.PLAY

    def test_no_merge_different_states(self):
        """Test that different states create separate segments."""
        from rallycut.analysis.game_state import GameStateAnalyzer

        analyzer = GameStateAnalyzer(device="cpu")

        results = [
            GameStateResult(frame_idx=0, state=GameState.NO_PLAY, confidence=0.9),
            GameStateResult(frame_idx=16, state=GameState.PLAY, confidence=0.95),
            GameStateResult(frame_idx=32, state=GameState.NO_PLAY, confidence=0.88),
        ]

        segments = analyzer.get_segments(results, fps=30.0)

        # Should have 3 separate segments
        assert len(segments) == 3
        assert segments[0].state == GameState.NO_PLAY
        assert segments[1].state == GameState.PLAY
        assert segments[2].state == GameState.NO_PLAY
