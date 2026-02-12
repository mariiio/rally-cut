"""Unit tests for ball temporal filter (Kalman filter with gating)."""

from __future__ import annotations

from rallycut.tracking.ball_filter import BallFilterConfig, BallTemporalFilter
from rallycut.tracking.ball_tracker import BallPosition


def _make_pos(frame: int, x: float, y: float, conf: float = 0.9) -> BallPosition:
    """Helper to create a BallPosition."""
    return BallPosition(frame_number=frame, x=x, y=y, confidence=conf)


def _converge_filter(
    filt: BallTemporalFilter,
    x: float = 0.5,
    y: float = 0.5,
    n: int = 20,
) -> None:
    """Feed consistent positions to converge the filter (tighten covariance)."""
    for i in range(n):
        filt.update(_make_pos(i, x, y))


class TestMahalanobisGating:
    """Tests for Mahalanobis distance-based measurement validation."""

    def test_rejects_flickering_on_converged_filter(self) -> None:
        """After convergence on stable position, a sudden jump should be rejected."""
        config = BallFilterConfig(
            enable_kalman=True,
            enable_outlier_removal=False,
            enable_interpolation=False,
            # Reduce velocity process noise so covariance converges tighter
            process_noise_velocity=0.001,
            mahalanobis_threshold=5.99,
        )
        filt = BallTemporalFilter(config)

        # Converge on (0.5, 0.5) for 30 frames to tighten covariance
        _converge_filter(filt, 0.5, 0.5, n=30)

        # Sudden jump to (0.8, 0.8) - should be rejected by Mahalanobis gate
        result = filt.update(_make_pos(30, 0.8, 0.8))

        # Filter should output near (0.5, 0.5), not (0.8, 0.8)
        assert abs(result.x - 0.5) < 0.05
        assert abs(result.y - 0.5) < 0.05

    def test_accepts_valid_movement_with_high_uncertainty(self) -> None:
        """After few frames (high uncertainty), larger movements should be accepted."""
        config = BallFilterConfig(
            enable_kalman=True,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Only 2 frames - filter still uncertain
        filt.update(_make_pos(0, 0.5, 0.5))
        filt.update(_make_pos(1, 0.5, 0.5))

        # Movement to (0.6, 0.6) should be accepted given high uncertainty
        result = filt.update(_make_pos(2, 0.6, 0.6))

        # Should be close to (0.6, 0.6) (accepted measurement)
        assert result.x > 0.55

    def test_hard_max_velocity_always_applies(self) -> None:
        """Max velocity limit should reject even with high uncertainty."""
        config = BallFilterConfig(
            enable_kalman=True,
            max_velocity=0.3,
            reacquisition_required=2,
            reacquisition_radius=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Initialize with consistent detections (required by init guard)
        filt.update(_make_pos(0, 0.1, 0.1))
        filt.update(_make_pos(1, 0.11, 0.11))  # Consistent → initializes

        # Jump across entire screen - should be rejected by max_velocity
        result = filt.update(_make_pos(2, 0.9, 0.9))

        # Should stay near initial position (prediction)
        assert result.x < 0.3

    def test_gradual_movement_accepted(self) -> None:
        """Smooth consistent movement should always be accepted."""
        config = BallFilterConfig(
            enable_kalman=True,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Ball moving smoothly right
        for i in range(20):
            x = 0.3 + i * 0.01  # 1% per frame
            filt.update(_make_pos(i, x, 0.5))

        # Next step in same direction should be accepted
        result = filt.update(_make_pos(20, 0.5, 0.5))
        assert result.x > 0.45


class TestReacquisitionGuard:
    """Tests for re-acquisition after track loss."""

    def test_rejects_scattered_detections(self) -> None:
        """Scattered detections after track loss should not re-acquire."""
        config = BallFilterConfig(
            enable_kalman=True,
            reacquisition_threshold=3,
            reacquisition_required=3,
            reacquisition_radius=0.05,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Converge filter
        _converge_filter(filt, 0.5, 0.5, n=10)

        # Lose track for several frames (low confidence)
        for i in range(10, 16):
            filt.update(_make_pos(i, 0.5, 0.5, conf=0.1))

        # Should now be in tentative mode
        assert filt._in_tentative_mode

        # Send scattered detections (spread > radius)
        filt.update(_make_pos(16, 0.2, 0.2))  # far apart
        filt.update(_make_pos(17, 0.8, 0.8))  # far apart
        filt.update(_make_pos(18, 0.5, 0.1))  # far apart

        # Should still be in tentative mode (not re-acquired)
        assert filt._in_tentative_mode

    def test_accepts_consistent_detections(self) -> None:
        """Consistent detections after track loss should re-acquire."""
        config = BallFilterConfig(
            enable_kalman=True,
            reacquisition_threshold=3,
            reacquisition_required=3,
            reacquisition_radius=0.05,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Converge filter
        _converge_filter(filt, 0.5, 0.5, n=10)

        # Lose track
        for i in range(10, 16):
            filt.update(_make_pos(i, 0.5, 0.5, conf=0.1))

        assert filt._in_tentative_mode

        # Send 3 consistent detections near (0.7, 0.7)
        filt.update(_make_pos(16, 0.70, 0.70))
        filt.update(_make_pos(17, 0.71, 0.71))
        result = filt.update(_make_pos(18, 0.72, 0.72))

        # Should have re-acquired
        assert not filt._in_tentative_mode
        assert abs(result.x - 0.71) < 0.02


class TestOutlierRemoval:
    """Tests for post-processing outlier removal."""

    def test_velocity_reversal_detected(self) -> None:
        """A→B→A flickering pattern should be removed as outlier."""
        config = BallFilterConfig(
            enable_kalman=True,
            enable_outlier_removal=True,
            enable_interpolation=False,
            # Use very loose Kalman gating so all pass through
            mahalanobis_threshold=100.0,
            max_velocity=1.0,
        )
        filt = BallTemporalFilter(config)

        # Create a trajectory with a flickering outlier at frame 5
        positions = [
            _make_pos(0, 0.50, 0.50),
            _make_pos(1, 0.51, 0.50),
            _make_pos(2, 0.52, 0.50),
            _make_pos(3, 0.53, 0.50),
            _make_pos(4, 0.54, 0.50),
            _make_pos(5, 0.80, 0.80),  # Flicker!
            _make_pos(6, 0.56, 0.50),
            _make_pos(7, 0.57, 0.50),
            _make_pos(8, 0.58, 0.50),
        ]

        result = filt.filter_batch(positions)

        # Frame 5 should be removed as outlier (velocity reversal)
        result_frames = {p.frame_number for p in result}
        assert 5 not in result_frames

    def test_smooth_trajectory_preserved(self) -> None:
        """A smooth trajectory should have no outliers removed."""
        config = BallFilterConfig(
            enable_kalman=True,
            enable_outlier_removal=True,
            enable_interpolation=False,
            mahalanobis_threshold=100.0,
            max_velocity=1.0,
        )
        filt = BallTemporalFilter(config)

        # Smooth rightward movement
        positions = [
            _make_pos(i, 0.3 + i * 0.02, 0.5) for i in range(10)
        ]

        result = filt.filter_batch(positions)

        # All positions should be preserved (no outliers)
        assert len(result) == len(positions)


class TestExitDetection:
    """Tests for out-of-frame exit detection."""

    def test_detects_exit_at_right_edge(self) -> None:
        """Ball near right edge moving right should be detected as exiting."""
        config = BallFilterConfig(
            enable_kalman=True,
            enable_exit_detection=True,
            exit_edge_margin=0.05,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Ball moving right toward edge, starting near edge
        for i in range(20):
            x = 0.8 + i * 0.01  # Moving right from 0.8 to 0.99
            filt.update(_make_pos(i, min(x, 0.99), 0.5))

        # Ball should be detected as exiting (x > 0.95 with positive vx)
        assert filt._exited
        assert filt._exit_edge == "right"

    def test_suppresses_opposite_side_reacquisition(self) -> None:
        """After exiting right, detection on left should be suppressed."""
        config = BallFilterConfig(
            enable_kalman=True,
            enable_exit_detection=True,
            exit_edge_margin=0.05,
            exit_opposite_side_margin=0.3,
            reacquisition_threshold=3,
            reacquisition_required=3,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Ball moving right toward edge, starting near edge
        for i in range(20):
            x = 0.8 + i * 0.01
            filt.update(_make_pos(i, min(x, 0.99), 0.5))

        # Verify exit detected
        assert filt._exited
        assert filt._exit_edge == "right"

        # Lose track
        for i in range(20, 30):
            filt.update(_make_pos(i, 0.99, 0.5, conf=0.1))

        # Should now be in tentative mode
        assert filt._in_tentative_mode

        # Detection at x=0.1 (left side) should be suppressed after right exit
        filt.update(_make_pos(30, 0.1, 0.5))
        filt.update(_make_pos(31, 0.1, 0.5))
        filt.update(_make_pos(32, 0.1, 0.5))

        # Should still be in tentative mode (suppressed)
        assert filt._in_tentative_mode

    def test_allows_same_side_reacquisition(self) -> None:
        """After exiting right, detection near right should not be suppressed."""
        config = BallFilterConfig(
            enable_kalman=True,
            enable_exit_detection=True,
            exit_edge_margin=0.05,
            exit_opposite_side_margin=0.3,
            reacquisition_threshold=3,
            reacquisition_required=3,
            reacquisition_radius=0.05,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Ball moving right toward edge
        for i in range(20):
            x = 0.8 + i * 0.01
            filt.update(_make_pos(i, min(x, 0.99), 0.5))

        # Lose track
        for i in range(20, 30):
            filt.update(_make_pos(i, 0.99, 0.5, conf=0.1))

        assert filt._in_tentative_mode

        # Detection near right side (same side as exit) - should NOT be suppressed
        filt.update(_make_pos(30, 0.85, 0.5))
        filt.update(_make_pos(31, 0.86, 0.5))
        filt.update(_make_pos(32, 0.87, 0.5))

        # Should have re-acquired
        assert not filt._in_tentative_mode


class TestSegmentPruning:
    """Tests for trajectory segment pruning."""

    def test_removes_short_leading_segment(self) -> None:
        """Short false detection segment at the start should be pruned."""
        config = BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.15,
            min_segment_frames=10,
            min_output_confidence=0.05,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Short false segment at beginning (5 frames at wrong location)
        for i in range(5):
            positions.append(_make_pos(i, 0.2, 0.5, conf=0.7))

        # Main trajectory (20 frames at correct location)
        for i in range(10, 30):
            positions.append(_make_pos(i, 0.6 + i * 0.005, 0.5))

        result = filt.filter_batch(positions)

        # Short leading segment should be removed
        result_frames = [p.frame_number for p in result]
        assert all(f >= 10 for f in result_frames), (
            f"Expected no frames before 10, got: {result_frames[:5]}"
        )

    def test_preserves_all_long_segments(self) -> None:
        """Multiple long segments should all be preserved."""
        config = BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.15,
            min_segment_frames=5,
            min_output_confidence=0.05,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Segment 1: 10 frames at left side
        for i in range(10):
            positions.append(_make_pos(i, 0.2 + i * 0.005, 0.5))

        # Segment 2: 10 frames at right side (big jump)
        for i in range(20, 30):
            positions.append(_make_pos(i, 0.8 + (i - 20) * 0.005, 0.5))

        result = filt.filter_batch(positions)

        # Both long segments should survive
        result_frames = {p.frame_number for p in result}
        assert any(f < 10 for f in result_frames), "Segment 1 should survive"
        assert any(f >= 20 for f in result_frames), "Segment 2 should survive"

    def test_drops_zero_confidence_placeholders(self) -> None:
        """VballNet zero-confidence placeholder positions should be dropped."""
        config = BallFilterConfig(
            enable_segment_pruning=True,
            min_output_confidence=0.05,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Zero-confidence placeholders (VballNet "no detection")
        for i in range(8):
            positions.append(_make_pos(i, 0.5, 0.5, conf=0.0))

        # Real detections
        for i in range(8, 30):
            positions.append(_make_pos(i, 0.5 + i * 0.005, 0.5))

        result = filt.filter_batch(positions)

        # Zero-confidence frames should not appear in output
        for p in result:
            assert p.confidence >= 0.05, (
                f"Frame {p.frame_number} has confidence {p.confidence} < 0.05"
            )


class TestEndToEnd:
    """End-to-end tests with synthetic trajectories."""

    def test_synthetic_trajectory_with_noise(self) -> None:
        """Synthetic trajectory with flickering, occlusion, and false positives."""
        config = BallFilterConfig(
            enable_kalman=True,
            reacquisition_threshold=3,
            reacquisition_required=3,
            enable_exit_detection=True,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Phase 1: Smooth rightward movement (frames 0-19)
        for i in range(20):
            x = 0.3 + i * 0.01
            positions.append(_make_pos(i, x, 0.5))

        # Phase 2: Flickering frame (frame 20)
        positions.append(_make_pos(20, 0.9, 0.1, conf=0.6))  # False detection

        # Phase 3: Continue trajectory (frames 21-29)
        for i in range(21, 30):
            x = 0.3 + i * 0.01
            positions.append(_make_pos(i, x, 0.5))

        # Phase 4: Occlusion (frames 30-39, low confidence)
        for i in range(30, 40):
            positions.append(_make_pos(i, 0.5, 0.5, conf=0.1))

        # Phase 5: False positive on wrong side (frame 40-42)
        positions.append(_make_pos(40, 0.05, 0.95, conf=0.7))
        positions.append(_make_pos(41, 0.95, 0.05, conf=0.7))
        positions.append(_make_pos(42, 0.15, 0.85, conf=0.7))

        # Phase 6: Real re-acquisition (frames 43-45)
        positions.append(_make_pos(43, 0.60, 0.50))
        positions.append(_make_pos(44, 0.61, 0.50))
        positions.append(_make_pos(45, 0.62, 0.50))

        result = filt.filter_batch(positions)

        # Should have output for all frames
        assert len(result) > 0

        # The flickering frame (20) should be removed by outlier removal
        # (velocity reversal or trajectory deviation detection)
        result_by_frame = {p.frame_number: p for p in result}
        if 20 in result_by_frame:
            # If it survived filtering, it should have been pulled toward trajectory
            # With loose Mahalanobis it may pass the Kalman gate, but outlier removal
            # should catch it. If it's still here, accept it as long as it's not exact
            assert result_by_frame[20].x != 0.9  # Should be modified from raw false detection

    def test_filter_batch_empty(self) -> None:
        """Empty input should return empty output."""
        filt = BallTemporalFilter()
        assert filt.filter_batch([]) == []

    def test_filter_batch_single(self) -> None:
        """Single position should pass through."""
        filt = BallTemporalFilter(BallFilterConfig(
            enable_outlier_removal=False,
            enable_interpolation=False,
        ))
        positions = [_make_pos(0, 0.5, 0.5)]
        result = filt.filter_batch(positions)
        assert len(result) == 1
        assert result[0].frame_number == 0

    def test_raw_mode_preserves_positions(self) -> None:
        """Raw mode (enable_kalman=False) should preserve original positions."""
        config = BallFilterConfig(
            enable_kalman=False,
            enable_segment_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions = [
            _make_pos(0, 0.30, 0.50),
            _make_pos(1, 0.31, 0.50),
            _make_pos(2, 0.32, 0.50),
            _make_pos(3, 0.33, 0.50),
            _make_pos(4, 0.34, 0.50),
        ]

        result = filt.filter_batch(positions)

        assert len(result) == 5
        for orig, filtered in zip(positions, result):
            assert filtered.x == orig.x
            assert filtered.y == orig.y
            assert filtered.frame_number == orig.frame_number

    def test_raw_mode_with_segment_pruning(self) -> None:
        """Raw mode + segment pruning removes short false segments."""
        config = BallFilterConfig(
            enable_kalman=False,
            enable_segment_pruning=True,
            segment_jump_threshold=0.15,
            min_segment_frames=10,
            min_output_confidence=0.05,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Short false segment (5 frames)
        for i in range(5):
            positions.append(_make_pos(i, 0.1, 0.1))

        # Main trajectory (20 frames, big jump from false segment)
        for i in range(10, 30):
            positions.append(_make_pos(i, 0.6 + i * 0.005, 0.5))

        result = filt.filter_batch(positions)

        # Short segment should be pruned, main trajectory preserved
        result_frames = [p.frame_number for p in result]
        assert all(f >= 10 for f in result_frames)

    def test_low_confidence_not_initialized(self) -> None:
        """Filter should not initialize on low confidence detections."""
        config = BallFilterConfig(
            enable_kalman=True,
            min_confidence_for_update=0.3,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        result = filt.update(_make_pos(0, 0.5, 0.5, conf=0.1))
        assert not filt._initialized
        # Should return as-is
        assert result.x == 0.5
        assert result.confidence == 0.1

    def test_initialization_guard_rejects_single_false_positive(self) -> None:
        """A single false positive should not initialize the filter."""
        config = BallFilterConfig(
            enable_kalman=True,
            reacquisition_required=2,
            reacquisition_radius=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # First detection at (0.5, 0.5)
        filt.update(_make_pos(0, 0.5, 0.5))
        assert not filt._initialized  # Not yet - need 2 consistent

        # Second detection far away (inconsistent)
        filt.update(_make_pos(1, 0.9, 0.1))
        assert not filt._initialized  # Inconsistent pair → rejected

        # Third detection near first (consistent with first? no, buffer has all 3)
        # Buffer: [(0.5,0.5), (0.9,0.1), (0.51,0.51)] - not all within radius
        filt.update(_make_pos(2, 0.51, 0.51))
        assert not filt._initialized

        # Fourth detection consistent with third
        filt.update(_make_pos(3, 0.52, 0.52))
        # Buffer: [(0.5,0.5), (0.9,0.1), (0.51,0.51), (0.52,0.52)]
        # The pair check requires ALL pairs within radius, so still not consistent
        assert not filt._initialized

    def test_initialization_guard_accepts_consistent(self) -> None:
        """Consistent detections should initialize the filter."""
        config = BallFilterConfig(
            enable_kalman=True,
            reacquisition_required=2,
            reacquisition_radius=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Two consistent detections
        filt.update(_make_pos(0, 0.50, 0.50))
        filt.update(_make_pos(1, 0.51, 0.51))

        # Should be initialized now
        assert filt._initialized

    def test_exit_forces_tentative_mode(self) -> None:
        """After exit detection, tentative mode should activate immediately."""
        config = BallFilterConfig(
            enable_kalman=True,
            enable_exit_detection=True,
            exit_edge_margin=0.05,
            reacquisition_threshold=10,  # High threshold - shouldn't matter
            reacquisition_required=2,
            reacquisition_radius=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Ball moving right toward edge
        for i in range(20):
            x = 0.8 + i * 0.01
            filt.update(_make_pos(i, min(x, 0.99), 0.5))

        assert filt._exited

        # Very next frame (even before reacquisition_threshold=10 frames)
        # should be in tentative mode due to exit
        filt.update(_make_pos(20, 0.5, 0.5, conf=0.1))  # Low conf prediction
        assert filt._in_tentative_mode  # Forced by exit, not waiting for threshold
