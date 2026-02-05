"""Tests for game state classifier and segment merging."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

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
            GameStateResult(start_frame=0, end_frame=15, state=GameState.NO_PLAY, confidence=0.9),
            GameStateResult(start_frame=16, end_frame=31, state=GameState.NO_PLAY, confidence=0.85),
            GameStateResult(start_frame=32, end_frame=47, state=GameState.SERVICE, confidence=0.8),
            GameStateResult(start_frame=48, end_frame=63, state=GameState.PLAY, confidence=0.95),
            GameStateResult(start_frame=64, end_frame=79, state=GameState.PLAY, confidence=0.92),
            GameStateResult(start_frame=80, end_frame=95, state=GameState.PLAY, confidence=0.88),
            GameStateResult(start_frame=96, end_frame=111, state=GameState.NO_PLAY, confidence=0.9),
        ]


class TestGameStateClassifierMocked:
    """Tests for classifier with mocked model."""

    @patch("rallycut.analysis.game_state.GameStateAnalyzer._get_classifier")
    def test_analyze_video_progress(self, mock_get_classifier):
        """Test progress callback during analysis."""
        from pathlib import Path

        from rallycut.analysis.game_state import GameStateAnalyzer
        from rallycut.core.models import VideoInfo

        # Mock classifier with batch method
        # Returns: (state, confidence, no_play_prob, play_prob, service_prob)
        mock_classifier = Mock()
        mock_classifier.classify_segments_batch.return_value = [
            (GameState.PLAY, 0.9, 0.05, 0.9, 0.05)
        ]
        mock_get_classifier.return_value = mock_classifier

        # Mock video with iter_frames
        mock_video = Mock()
        mock_video.info = VideoInfo(
            path=Path("/test.mp4"),
            duration=1.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=30,
        )

        # Mock iter_frames to yield frames
        frames = [(i, np.zeros((224, 224, 3), dtype=np.uint8)) for i in range(30)]
        mock_video.iter_frames.return_value = iter(frames)

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
    """Tests for segment merging logic in VideoCutter."""

    def test_merge_adjacent_same_state(self):
        """Test that adjacent segments of same state are merged."""
        from rallycut.processing.cutter import VideoCutter

        # Use low min_play_duration to not filter out short segments
        cutter = VideoCutter(min_play_duration=0.1, padding_seconds=0, rally_continuation_seconds=0)

        results = [
            GameStateResult(start_frame=0, end_frame=29, state=GameState.PLAY, confidence=0.9),
            GameStateResult(start_frame=30, end_frame=59, state=GameState.PLAY, confidence=0.85),
            GameStateResult(start_frame=60, end_frame=89, state=GameState.PLAY, confidence=0.88),
        ]

        segments, _ = cutter._get_segments_from_results(results, fps=30.0)

        # Should have exactly 1 merged PLAY segment (after filtering)
        assert len(segments) == 1
        assert segments[0].state == GameState.PLAY

    def test_no_merge_different_states(self):
        """Test that different states create separate segments."""
        from rallycut.processing.cutter import VideoCutter

        cutter = VideoCutter(min_play_duration=0.1, min_gap_seconds=0.1, rally_continuation_seconds=0)

        # Need at least 2 PLAY windows to pass MIN_ACTIVE_WINDOWS filter
        results = [
            GameStateResult(start_frame=0, end_frame=29, state=GameState.NO_PLAY, confidence=0.9),
            GameStateResult(start_frame=30, end_frame=59, state=GameState.PLAY, confidence=0.95),
            GameStateResult(start_frame=60, end_frame=89, state=GameState.PLAY, confidence=0.92),
            GameStateResult(start_frame=90, end_frame=120, state=GameState.NO_PLAY, confidence=0.88),
        ]

        segments, _ = cutter._get_segments_from_results(results, fps=30.0)

        # Should have 1 play segment (NO_PLAY segments are filtered out)
        assert len(segments) == 1
        assert segments[0].state == GameState.PLAY

    def test_empty_results(self):
        """Test handling of empty results."""
        from rallycut.processing.cutter import VideoCutter

        cutter = VideoCutter()
        segments, suggested = cutter._get_segments_from_results([], fps=30.0)
        assert segments == []
        assert suggested == []

    def test_single_play_result(self):
        """Test handling of two adjacent play results (minimum to pass filter)."""
        from rallycut.processing.cutter import VideoCutter

        cutter = VideoCutter(min_play_duration=0.1, rally_continuation_seconds=0)

        # Need at least 2 PLAY windows to pass MIN_ACTIVE_WINDOWS filter
        results = [
            GameStateResult(start_frame=0, end_frame=30, state=GameState.PLAY, confidence=0.9),
            GameStateResult(start_frame=31, end_frame=60, state=GameState.PLAY, confidence=0.85),
        ]

        segments, _ = cutter._get_segments_from_results(results, fps=30.0)
        assert len(segments) == 1
        assert segments[0].state == GameState.PLAY

    def test_gap_merging(self):
        """Test that short NO_PLAY gaps are merged into PLAY segments."""
        from rallycut.processing.cutter import VideoCutter

        # min_gap_seconds=1.5 means gaps shorter than 45 frames (at 30fps) are merged
        cutter = VideoCutter(min_play_duration=0.1, min_gap_seconds=1.5, rally_continuation_seconds=0)

        results = [
            GameStateResult(start_frame=0, end_frame=59, state=GameState.PLAY, confidence=0.9),
            # Short gap (30 frames = 1 second < 1.5 second threshold)
            GameStateResult(start_frame=60, end_frame=89, state=GameState.NO_PLAY, confidence=0.8),
            GameStateResult(start_frame=90, end_frame=149, state=GameState.PLAY, confidence=0.85),
        ]

        segments, _ = cutter._get_segments_from_results(results, fps=30.0)

        # Should be merged into single segment (gap was short)
        assert len(segments) == 1

    def test_long_gap_not_merged(self):
        """Test that long NO_PLAY gaps are not merged."""
        from rallycut.processing.cutter import VideoCutter

        # Use very long gap to ensure it doesn't get merged by any heuristics
        # min_gap_seconds=1.0 means gaps longer than 30 frames are not merged
        cutter = VideoCutter(
            min_play_duration=0.1,
            min_gap_seconds=1.0,
            padding_seconds=0,
            rally_continuation_seconds=0,
        )

        # Need at least 2 PLAY windows per segment to pass MIN_ACTIVE_WINDOWS filter
        # Use a very long gap (300 frames = 10 seconds) to ensure no merging
        results = [
            GameStateResult(start_frame=0, end_frame=29, state=GameState.PLAY, confidence=0.9),
            GameStateResult(start_frame=30, end_frame=59, state=GameState.PLAY, confidence=0.88),
            # Very long gap (300 frames = 10 seconds >> threshold)
            GameStateResult(start_frame=60, end_frame=359, state=GameState.NO_PLAY, confidence=0.8),
            GameStateResult(start_frame=360, end_frame=389, state=GameState.PLAY, confidence=0.85),
            GameStateResult(start_frame=390, end_frame=419, state=GameState.PLAY, confidence=0.82),
        ]

        segments, _ = cutter._get_segments_from_results(results, fps=30.0)

        # Should be 2 separate segments (gap was too long)
        assert len(segments) == 2

    def test_min_duration_filter(self):
        """Test min duration filtering."""
        from rallycut.processing.cutter import VideoCutter

        # 2 seconds = 60 frames min duration, no padding
        cutter = VideoCutter(min_play_duration=2.0, padding_seconds=0, min_gap_seconds=5.0, rally_continuation_seconds=0)

        # Need at least 2 PLAY windows per segment to pass MIN_ACTIVE_WINDOWS filter
        results = [
            # Short segment with 2 windows (40 frames total < 60 frames = 2 seconds) - filtered
            GameStateResult(start_frame=0, end_frame=19, state=GameState.PLAY, confidence=0.9),
            GameStateResult(start_frame=20, end_frame=39, state=GameState.PLAY, confidence=0.88),
            # Long gap to prevent merging
            GameStateResult(start_frame=40, end_frame=299, state=GameState.NO_PLAY, confidence=0.8),
            # Long segment with 4 windows (120 frames = 4 seconds) - should pass
            GameStateResult(start_frame=300, end_frame=329, state=GameState.PLAY, confidence=0.85),
            GameStateResult(start_frame=330, end_frame=359, state=GameState.PLAY, confidence=0.82),
            GameStateResult(start_frame=360, end_frame=389, state=GameState.PLAY, confidence=0.80),
            GameStateResult(start_frame=390, end_frame=419, state=GameState.PLAY, confidence=0.78),
        ]

        segments, suggested = cutter._get_segments_from_results(results, fps=30.0)

        # Both segments should remain since both have 2+ windows
        # The first segment (40 frames = 1.33s) is below min_play_duration but
        # min_play_duration filter only applies to multi-window segments
        # Since both have 2+ windows and are multi-window, min_duration applies
        assert len(segments) == 1
        assert segments[0].start_frame >= 300  # The longer segment


class TestCutterSegmentStats:
    """Tests for cut statistics."""

    def test_cut_stats(self):
        """Test cut statistics calculation."""
        from rallycut.processing.cutter import VideoCutter

        cutter = VideoCutter()

        segments = [
            TimeSegment(
                start_frame=0, end_frame=300,
                start_time=0.0, end_time=10.0,
                state=GameState.PLAY
            ),
            TimeSegment(
                start_frame=600, end_frame=900,
                start_time=20.0, end_time=30.0,
                state=GameState.PLAY
            ),
        ]

        stats = cutter.get_cut_stats(original_duration=60.0, segments=segments)

        assert stats["original_duration"] == 60.0
        assert stats["kept_duration"] == 20.0  # 10 + 10 seconds
        assert stats["removed_duration"] == 40.0
        assert stats["segment_count"] == 2
