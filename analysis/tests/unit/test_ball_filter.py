"""Unit tests for ball temporal filter."""

from __future__ import annotations

import numpy as np

from rallycut.tracking.ball_filter import BallFilterConfig, BallTemporalFilter
from rallycut.tracking.ball_tracker import BallPosition


def _make_pos(
    frame: int,
    x: float,
    y: float,
    conf: float = 0.9,
    motion_energy: float = 0.0,
) -> BallPosition:
    """Helper to create a BallPosition."""
    return BallPosition(
        frame_number=frame, x=x, y=y, confidence=conf, motion_energy=motion_energy,
    )


class TestOutlierRemoval:
    """Tests for post-processing outlier removal."""

    def test_velocity_reversal_detected(self) -> None:
        """A→B→A flickering pattern should be removed as outlier."""
        config = BallFilterConfig(
            enable_outlier_removal=True,
            enable_segment_pruning=False,
            enable_interpolation=False,
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
            enable_outlier_removal=True,
            enable_segment_pruning=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Smooth rightward movement
        positions = [
            _make_pos(i, 0.3 + i * 0.02, 0.5) for i in range(10)
        ]

        result = filt.filter_batch(positions)

        # All positions should be preserved (no outliers)
        assert len(result) == len(positions)


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
        """Multiple long segments should all be preserved as anchors."""
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

        # Segment 2: 10 frames at right side (far away but long enough)
        for i in range(20, 30):
            positions.append(_make_pos(i, 0.8 + (i - 20) * 0.005, 0.5))

        result = filt.filter_batch(positions)

        # Both long segments should survive as anchors
        result_frames = {p.frame_number for p in result}
        assert any(f < 10 for f in result_frames), "Segment 1 should survive"
        assert any(f >= 20 for f in result_frames), "Segment 2 should survive"

    def test_drops_zero_confidence_placeholders(self) -> None:
        """Zero-confidence placeholder positions should be dropped."""
        config = BallFilterConfig(
            enable_segment_pruning=True,
            min_output_confidence=0.05,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Zero-confidence placeholders (no detection)
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


    def test_recovers_short_segments_near_anchor(self) -> None:
        """Short trajectory fragments near anchors should be kept, not pruned.

        Simulates the detector interleaving single-frame false positives within a
        real trajectory: real(20f) → false(1f) → real(3f) → false(3f) → real(2f).
        The real fragments are near the anchor, false positives are far away.
        """
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

        # Anchor segment: 20 frames of real trajectory at x≈0.5, y≈0.3
        for i in range(20):
            positions.append(_make_pos(i, 0.50 + i * 0.002, 0.30))

        # Single-frame false positive (jumps to player position)
        positions.append(_make_pos(20, 0.90, 0.70, conf=0.8))

        # Short real fragment: 3 frames close to anchor end
        for i in range(21, 24):
            positions.append(_make_pos(i, 0.54 + (i - 21) * 0.002, 0.30))

        # Multi-frame false cluster (3 frames at player position)
        for i in range(24, 27):
            positions.append(_make_pos(i, 0.88, 0.72, conf=0.7))

        # Short real fragment: 2 frames close to anchor
        for i in range(27, 29):
            positions.append(_make_pos(i, 0.55, 0.30))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Anchor frames (0-19) must survive
        assert all(f in result_frames for f in range(20)), "Anchor should survive"

        # Real fragments near anchor should be recovered
        assert 21 in result_frames, "Frame 21 (near anchor) should be recovered"
        assert 22 in result_frames, "Frame 22 (near anchor) should be recovered"
        assert 27 in result_frames, "Frame 27 (near anchor) should be recovered"

        # False positive frames should NOT be in output
        assert 20 not in result_frames, "Frame 20 (false positive) should be removed"
        assert 24 not in result_frames, "Frame 24 (false cluster) should be removed"
        assert 25 not in result_frames, "Frame 25 (false cluster) should be removed"

    def test_no_recovery_across_large_temporal_gap(self) -> None:
        """Short segment spatially near anchor but after large gap should be pruned.

        Reproduces the bd77efd1 bug: after a 58-frame gap (ball exits frame),
        the detector restarts at a player position that happens to be spatially
        close to the last anchor endpoint. Without temporal gating, the
        proximity recovery would keep this false segment.
        """
        config = BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.15,
            min_segment_frames=10,
            min_output_confidence=0.05,
            max_interpolation_gap=10,  # recovery gate = 3 * 10 = 30 frames
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Anchor: 20 frames ending at (0.40, 0.37)
        for i in range(20):
            positions.append(_make_pos(i, 0.30 + i * 0.005, 0.30 + i * 0.004))

        # Large gap: 50 frames (ball exits frame, no detections)

        # Short segment at player position, spatially near anchor endpoint
        # (0.38, 0.35) is within proximity of anchor end (0.395, 0.376)
        # but temporally far (50 frames gap > 30 frame threshold)
        # Must be < min_segment_frames (10) so it's not an anchor itself
        for i in range(70, 78):
            positions.append(
                _make_pos(i, 0.38 + (i - 70) * 0.001, 0.35 + (i - 70) * 0.001)
            )

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Anchor should survive
        assert all(f in result_frames for f in range(20))
        # Short segment should be pruned (large temporal gap despite proximity)
        assert not any(f >= 70 for f in result_frames), (
            f"Segment across large gap should be pruned, but found: "
            f"{sorted(f for f in result_frames if f >= 70)}"
        )

    def test_recovery_within_temporal_gap(self) -> None:
        """Short segment spatially and temporally near anchor should be recovered."""
        config = BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.15,
            min_segment_frames=10,
            min_output_confidence=0.05,
            max_interpolation_gap=10,  # recovery gate = 3 * 10 = 30 frames
            enable_outlier_removal=False,
            enable_interpolation=False,
            enable_stationarity_filter=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Anchor: 20 frames ending at (0.395, 0.376)
        for i in range(20):
            positions.append(_make_pos(i, 0.30 + i * 0.005, 0.30 + i * 0.004))

        # Small gap: 5 frames (within recovery gate of 30)

        # Short segment spatially near anchor endpoint
        for i in range(25, 33):
            positions.append(
                _make_pos(i, 0.38 + (i - 25) * 0.001, 0.35 + (i - 25) * 0.001)
            )

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Both should survive (small temporal gap + spatial proximity)
        assert all(f in result_frames for f in range(20))
        assert any(f >= 25 for f in result_frames), (
            "Short segment within temporal gap should be recovered"
        )

    def test_short_segment_far_from_anchor_still_pruned(self) -> None:
        """Short segments far from any anchor should still be pruned."""
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

        # Short false segment at beginning (5 frames far from anchor)
        for i in range(5):
            positions.append(_make_pos(i, 0.2, 0.8, conf=0.7))

        # Main anchor trajectory (20 frames)
        for i in range(10, 30):
            positions.append(_make_pos(i, 0.6 + i * 0.005, 0.3))

        # Short false segment at end (5 frames far from anchor)
        for i in range(35, 40):
            positions.append(_make_pos(i, 0.1, 0.9, conf=0.6))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Only anchor frames should survive
        assert all(f >= 10 and f < 30 for f in result_frames), (
            f"Only anchor frames expected, got: {sorted(result_frames)}"
        )


class TestOscillationPruning:
    """Tests for oscillation (A→B→A→B) pruning."""

    def test_sustained_oscillation_trimmed(self) -> None:
        """A→B→A→B tail after real trajectory should be trimmed."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Real trajectory: smooth rightward movement (30 frames)
        for i in range(30):
            positions.append(_make_pos(i, 0.3 + i * 0.01, 0.5))

        # Oscillating tail: alternating between two player positions (20 frames)
        for i in range(30, 50):
            if i % 2 == 0:
                positions.append(_make_pos(i, 0.7, 0.3))  # Player A
            else:
                positions.append(_make_pos(i, 0.7, 0.6))  # Player B

        result = filt.filter_batch(positions)

        # Real trajectory should survive, oscillating tail should be trimmed
        result_frames = {p.frame_number for p in result}
        # At least some real frames kept
        assert any(f < 30 for f in result_frames)
        # Most oscillating frames should be trimmed
        oscillating_kept = sum(1 for f in result_frames if f >= 30)
        assert oscillating_kept < 10, (
            f"Expected most oscillating frames trimmed, but {oscillating_kept} survived"
        )

    def test_cluster_oscillation_trimmed(self) -> None:
        """Cluster-based oscillation (real detector pattern) should be trimmed.

        Real pattern from rally 0d84f858: positions stay near player B
        (x≈0.875) for 2-5 frames, jump to player A (x≈0.82) for 1-2 frames,
        then back. Within-cluster displacement is tiny (<0.01), but the
        cluster transition rate is high.
        """
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Real trajectory: smooth rightward movement (30 frames)
        for i in range(30):
            positions.append(_make_pos(i, 0.3 + i * 0.01, 0.5))

        # Cluster-based oscillating tail (20 frames):
        # Positions stay at B for 2-3 frames, jump to A for 1-2 frames, repeat
        # Pattern: BBBAAABBBBBAABBBB (mimics real detector output)
        cluster_pattern = [
            # B cluster (x=0.875, y=0.48)
            (0.875, 0.48), (0.876, 0.481), (0.874, 0.479),
            # A cluster (x=0.82, y=0.50)
            (0.827, 0.50), (0.825, 0.501),
            # B cluster
            (0.875, 0.48), (0.876, 0.481), (0.874, 0.479), (0.875, 0.48), (0.876, 0.481),
            # A cluster
            (0.825, 0.50), (0.827, 0.501),
            # B cluster
            (0.875, 0.48), (0.876, 0.481), (0.874, 0.479), (0.875, 0.48),
            # A cluster
            (0.825, 0.50), (0.827, 0.501),
            # B cluster
            (0.875, 0.48), (0.876, 0.481),
        ]
        for i, (x, y) in enumerate(cluster_pattern):
            positions.append(_make_pos(30 + i, x, y))

        result = filt.filter_batch(positions)

        # Real trajectory should survive, cluster oscillation should be trimmed
        result_frames = {p.frame_number for p in result}
        assert any(f < 30 for f in result_frames)
        oscillating_kept = sum(1 for f in result_frames if f >= 30)
        assert oscillating_kept < 10, (
            f"Expected cluster oscillation trimmed, but {oscillating_kept} survived"
        )

    def test_brief_direction_change_preserved(self) -> None:
        """A 2-frame direction change (bounce/hit) should NOT be trimmed."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Smooth trajectory with a brief bounce at frame 15
        for i in range(30):
            x = 0.3 + i * 0.01
            y = 0.5
            if i == 15:
                y = 0.4  # Brief deflection (1 frame)
            positions.append(_make_pos(i, x, y))

        result = filt.filter_batch(positions)

        # All 30 frames should be preserved (no sustained oscillation)
        assert len(result) == 30

    def test_ball_bounce_with_brief_confusion_preserved(self) -> None:
        """Ball bounce + 1-frame detector confusion should NOT be trimmed.

        Real pattern from rally 1bfcbc4f: ball descends near net (y≈0.25),
        bounces to player area (y≈0.42), detector briefly reads net area for
        1 frame, then continues at player area. Creates 3 transitions in a
        window but the clusters are NOT compact (ball is moving through space,
        not locked onto fixed positions).
        """
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Ball descending near net (x≈0.55, y decreasing from 0.37 to 0.25)
        for i in range(20):
            y = 0.37 - i * 0.006
            positions.append(_make_pos(i, 0.55, y))

        # Ball bounces to player area
        positions.append(_make_pos(20, 0.55, 0.42))  # Bounce to court
        positions.append(_make_pos(21, 0.55, 0.27))  # Detector confusion (1 frame)
        positions.append(_make_pos(22, 0.56, 0.39))  # Continues at player area
        positions.append(_make_pos(23, 0.56, 0.39))

        # Ball hit and moves left across court
        for i in range(24, 40):
            x = 0.55 - (i - 24) * 0.015
            positions.append(_make_pos(i, x, 0.30))

        result = filt.filter_batch(positions)

        # All 40 frames should be preserved — this is a real trajectory
        assert len(result) == 40, (
            f"Expected 40 frames preserved, but only {len(result)} survived"
        )

    def test_smooth_trajectory_untouched(self) -> None:
        """A smooth arc trajectory should have all frames kept."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Parabolic arc (serve trajectory)
        positions: list[BallPosition] = []
        for i in range(40):
            t = i / 39.0
            x = 0.2 + 0.6 * t
            y = 0.3 + 0.4 * t * (1 - t)  # Parabola
            positions.append(_make_pos(i, x, y))

        result = filt.filter_batch(positions)
        assert len(result) == 40

    def test_stationary_ball_with_noise_untouched(self) -> None:
        """Sub-threshold jitter around a stationary position should not trigger."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,  # 3% threshold
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Stationary ball with tiny jitter (< 0.01 = 1% screen)
        positions: list[BallPosition] = []
        rng = np.random.default_rng(42)
        for i in range(30):
            x = 0.5 + rng.uniform(-0.005, 0.005)
            y = 0.5 + rng.uniform(-0.005, 0.005)
            positions.append(_make_pos(i, x, y))

        result = filt.filter_batch(positions)
        assert len(result) == 30

    def test_oscillation_disabled_by_config(self) -> None:
        """When enable_oscillation_pruning=False, oscillation should be kept."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=False,  # Disabled
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Real trajectory + oscillating tail
        for i in range(20):
            positions.append(_make_pos(i, 0.3 + i * 0.01, 0.5))
        for i in range(20, 40):
            if i % 2 == 0:
                positions.append(_make_pos(i, 0.7, 0.3))
            else:
                positions.append(_make_pos(i, 0.7, 0.6))

        result = filt.filter_batch(positions)

        # All 40 frames should be preserved (pruning disabled)
        assert len(result) == 40

    def test_oscillation_in_separate_segments(self) -> None:
        """Only the oscillating segment should be trimmed, not the clean one."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Segment 1: clean trajectory (frames 0-19)
        for i in range(20):
            positions.append(_make_pos(i, 0.3 + i * 0.01, 0.5))

        # Gap of >5 frames (segment break)
        # Segment 2: oscillating (frames 30-49)
        for i in range(30, 50):
            if i % 2 == 0:
                positions.append(_make_pos(i, 0.7, 0.3))
            else:
                positions.append(_make_pos(i, 0.7, 0.6))

        result = filt.filter_batch(positions)

        result_frames = {p.frame_number for p in result}
        # Segment 1 fully preserved
        assert all(f in result_frames for f in range(20))
        # Segment 2 mostly trimmed
        oscillating_kept = sum(1 for f in result_frames if f >= 30)
        assert oscillating_kept < 5, (
            f"Expected most of segment 2 trimmed, but {oscillating_kept} survived"
        )

    def test_hovering_segment_after_gap_trimmed(self) -> None:
        """Segment hovering near one position after a large gap should be dropped.

        Models the pattern where ball exits frame and the model locks
        onto a single player position, producing many frames within ~3% of screen.
        """
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,
            max_interpolation_gap=10,
            segment_jump_threshold=0.20,  # hover_radius = 0.20/4 = 0.05
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Real trajectory: 30 frames of ball moving across court
        for i in range(30):
            positions.append(_make_pos(i, 0.3 + i * 0.01, 0.5))

        # Gap of 40 frames (ball exits frame, no detections)

        # Hovering segment: 20 frames of player lock-on (all within 0.03 of centroid)
        # This is after a gap >> max_interpolation_gap=10
        for i in range(70, 90):
            x = 0.875 + (i - 70) * 0.001  # tiny jitter, spread ~0.02
            y = 0.48 + (i - 70) * 0.0005
            positions.append(_make_pos(i, x, y))

        result = filt.filter_batch(positions)

        # Real trajectory should survive
        result_frames = {p.frame_number for p in result}
        assert any(f < 30 for f in result_frames), "Real trajectory should survive"

        # Hovering segment should be dropped entirely
        hovering_kept = sum(1 for f in result_frames if f >= 70)
        assert hovering_kept == 0, (
            f"Expected hovering segment dropped, but {hovering_kept} frames survived"
        )

    def test_hovering_without_gap_preserved(self) -> None:
        """Compact segment without a preceding gap should NOT be flagged as hovering."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,
            max_interpolation_gap=10,
            segment_jump_threshold=0.20,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Segment 1: 20 frames of trajectory
        for i in range(20):
            positions.append(_make_pos(i, 0.3 + i * 0.01, 0.5))

        # Small gap (only 8 frames, below max_interpolation_gap=10)

        # Segment 2: compact but with small gap (not hovering false positive)
        for i in range(28, 48):
            x = 0.6 + (i - 28) * 0.001
            y = 0.5 + (i - 28) * 0.0005
            positions.append(_make_pos(i, x, y))

        result = filt.filter_batch(positions)

        # Both segments should survive (gap too small to trigger hovering)
        result_frames = {p.frame_number for p in result}
        assert any(f < 20 for f in result_frames), "Segment 1 should survive"
        assert any(f >= 28 for f in result_frames), "Segment 2 should survive (no large gap)"

    def test_end_to_end_raw_pipeline_with_oscillation(self) -> None:
        """Full raw pipeline: leading segment pruned + oscillating tail trimmed."""
        config = BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.15,
            min_segment_frames=10,
            min_output_confidence=0.05,
            enable_oscillation_pruning=True,
            min_oscillation_frames=12,
            oscillation_reversal_rate=0.25,
            oscillation_min_displacement=0.03,
            enable_outlier_removal=True,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Short false segment at start (5 frames, should be segment-pruned)
        for i in range(5):
            positions.append(_make_pos(i, 0.1, 0.1, conf=0.7))

        # Main trajectory (frames 10-39, 30 frames)
        for i in range(10, 40):
            positions.append(_make_pos(i, 0.3 + (i - 10) * 0.01, 0.5))

        # Oscillating tail (frames 40-59, 20 frames)
        for i in range(40, 60):
            if i % 2 == 0:
                positions.append(_make_pos(i, 0.7, 0.3))
            else:
                positions.append(_make_pos(i, 0.7, 0.6))

        result = filt.filter_batch(positions)

        result_frames = {p.frame_number for p in result}

        # Leading false segment should be pruned
        assert not any(f < 10 for f in result_frames), "Leading segment should be pruned"

        # Main trajectory should survive
        main_kept = sum(1 for f in result_frames if 10 <= f < 40)
        assert main_kept >= 20, f"Expected most main trajectory kept, got {main_kept}"

        # Oscillating tail should be mostly trimmed
        oscillating_kept = sum(1 for f in result_frames if f >= 40)
        assert oscillating_kept < 10, (
            f"Expected oscillating tail trimmed, but {oscillating_kept} survived"
        )


class TestExitGhostRemoval:
    """Tests for exit ghost removal (physics-impossible reversals near edges)."""

    def test_top_exit_ghost_removed(self) -> None:
        """Ball approaches top edge, reverses — ghosts should be removed."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=True,
            exit_edge_zone=0.10,
            exit_approach_frames=3,
            exit_min_approach_speed=0.008,
            max_interpolation_gap=10,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Ball moving across court, then approaching top edge
        for i in range(20):
            positions.append(_make_pos(i, 0.5, 0.3 - i * 0.01))

        # Ball near top edge, approaching fast (y decreasing toward 0)
        # y goes from 0.08 → 0.06 → 0.04 → 0.02
        positions.append(_make_pos(20, 0.5, 0.08))
        positions.append(_make_pos(21, 0.5, 0.06))
        positions.append(_make_pos(22, 0.5, 0.04))
        positions.append(_make_pos(23, 0.5, 0.02))  # Last real frame near top

        # REVERSAL: ball "comes back" — these are ghosts
        positions.append(_make_pos(24, 0.5, 0.10))  # Impossible reversal
        positions.append(_make_pos(25, 0.5, 0.15))
        positions.append(_make_pos(26, 0.5, 0.20))
        positions.append(_make_pos(27, 0.5, 0.30))
        positions.append(_make_pos(28, 0.5, 0.40))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Real frames should be kept
        assert all(f in result_frames for f in range(24))
        # Ghost frames should be removed
        assert 24 not in result_frames, "Frame 24 (reversal) should be removed"
        assert 25 not in result_frames, "Frame 25 (ghost) should be removed"
        assert 28 not in result_frames, "Frame 28 (ghost) should be removed"

    def test_bottom_exit_ghost_removed(self) -> None:
        """Ball approaches bottom edge, reverses — ghosts should be removed."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=True,
            exit_edge_zone=0.10,
            exit_approach_frames=3,
            exit_min_approach_speed=0.008,
            max_interpolation_gap=10,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Ball approaching bottom edge (y increasing toward 1.0)
        for i in range(10):
            positions.append(_make_pos(i, 0.5, 0.5 + i * 0.04))

        # Near bottom edge
        positions.append(_make_pos(10, 0.5, 0.92))
        positions.append(_make_pos(11, 0.5, 0.94))
        positions.append(_make_pos(12, 0.5, 0.96))

        # Ghost reversal
        positions.append(_make_pos(13, 0.5, 0.85))  # Reversal
        positions.append(_make_pos(14, 0.5, 0.75))
        positions.append(_make_pos(15, 0.5, 0.65))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Ghosts removed
        assert 13 not in result_frames
        assert 14 not in result_frames
        assert 15 not in result_frames
        # Real frames kept
        assert all(f in result_frames for f in range(13))

    def test_approach_from_outside_edge_zone(self) -> None:
        """Approach starting outside edge zone but reaching edge should trigger.

        Reproduces production bug: ball approaches top edge with frames at
        y=0.108, 0.085, 0.062. The first frame (y=0.108) is outside the 10%
        edge zone, but the last frame (y=0.062) is inside. Only the last
        approach frame needs to be in the edge zone.
        """
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=True,
            exit_edge_zone=0.10,
            exit_approach_frames=3,
            exit_min_approach_speed=0.008,
            max_interpolation_gap=10,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Ball tracking approaching top (real production values)
        for i in range(10):
            positions.append(_make_pos(i, 0.39, 0.35 - i * 0.02))

        # Approach: first frame outside edge zone, last inside
        positions.append(_make_pos(10, 0.38, 0.108))  # Outside 10% zone
        positions.append(_make_pos(11, 0.38, 0.085))  # Inside
        positions.append(_make_pos(12, 0.37, 0.062))  # Inside (exit point)

        # Ghost reversal after gap (ball was out of frame f=13-17)
        positions.append(_make_pos(18, 0.29, 0.367))  # Ghost at player
        positions.append(_make_pos(19, 0.29, 0.370))
        positions.append(_make_pos(20, 0.28, 0.400))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Real frames kept
        assert all(f in result_frames for f in range(13))
        # Ghost frames removed
        assert 18 not in result_frames, "Ghost at player position should be removed"
        assert 19 not in result_frames
        assert 20 not in result_frames

    def test_no_false_positive_on_bounce(self) -> None:
        """Ball reversal in center of screen (not near edge) should be kept."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=True,
            exit_edge_zone=0.10,
            exit_approach_frames=3,
            exit_min_approach_speed=0.008,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Ball moving upward in center of court (y=0.5→0.3), then bounces back
        for i in range(10):
            positions.append(_make_pos(i, 0.5, 0.5 - i * 0.02))

        # Reversal at y=0.30 (center, not near edge)
        positions.append(_make_pos(10, 0.5, 0.32))
        positions.append(_make_pos(11, 0.5, 0.35))
        positions.append(_make_pos(12, 0.5, 0.40))

        result = filt.filter_batch(positions)

        # All frames should be kept — reversal is in center, not near edge
        assert len(result) == 13

    def test_no_false_positive_slow_approach(self) -> None:
        """Ball near edge but approach speed too low should be kept."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=True,
            exit_edge_zone=0.10,
            exit_approach_frames=3,
            exit_min_approach_speed=0.008,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Ball drifting very slowly near top edge (speed < 0.008)
        positions.append(_make_pos(0, 0.5, 0.09))
        positions.append(_make_pos(1, 0.5, 0.085))  # dy = -0.005 (too slow)
        positions.append(_make_pos(2, 0.5, 0.080))  # dy = -0.005
        positions.append(_make_pos(3, 0.5, 0.075))  # dy = -0.005

        # Reversal
        positions.append(_make_pos(4, 0.5, 0.12))

        result = filt.filter_batch(positions)

        # All kept — approach was too slow to confirm exit
        assert len(result) == 5

    def test_ghost_terminated_at_gap(self) -> None:
        """Ghost marking should stop at a frame gap > max_interpolation_gap."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=True,
            exit_edge_zone=0.10,
            exit_approach_frames=3,
            exit_min_approach_speed=0.008,
            max_interpolation_gap=10,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Ball approaching top edge
        positions.append(_make_pos(0, 0.5, 0.08))
        positions.append(_make_pos(1, 0.5, 0.06))
        positions.append(_make_pos(2, 0.5, 0.04))
        positions.append(_make_pos(3, 0.5, 0.02))

        # Ghost reversal
        positions.append(_make_pos(4, 0.5, 0.10))
        positions.append(_make_pos(5, 0.5, 0.15))

        # Gap of 20 frames (> max_interpolation_gap=10) — new detections
        positions.append(_make_pos(25, 0.5, 0.50))
        positions.append(_make_pos(26, 0.5, 0.51))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Ghosts removed
        assert 4 not in result_frames
        assert 5 not in result_frames
        # Post-gap positions kept
        assert 25 in result_frames
        assert 26 in result_frames


    def test_ghost_terminated_at_edge_reentry(self) -> None:
        """Ghost marking should stop when position returns to exit edge zone."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=True,
            exit_edge_zone=0.10,
            exit_approach_frames=3,
            exit_min_approach_speed=0.008,
            max_interpolation_gap=10,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Ball approaching top edge
        positions.append(_make_pos(0, 0.5, 0.08))
        positions.append(_make_pos(1, 0.5, 0.06))
        positions.append(_make_pos(2, 0.5, 0.04))
        positions.append(_make_pos(3, 0.5, 0.02))

        # Ghost reversal (ball exits, detector locks on player)
        positions.append(_make_pos(4, 0.5, 0.30))  # Ghost
        positions.append(_make_pos(5, 0.5, 0.40))  # Ghost
        positions.append(_make_pos(6, 0.5, 0.45))  # Ghost

        # Ball re-enters from same edge (returns to edge zone)
        positions.append(_make_pos(7, 0.5, 0.05))  # Back in top edge zone
        positions.append(_make_pos(8, 0.5, 0.10))  # Moving away from edge
        positions.append(_make_pos(9, 0.5, 0.15))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Ghost frames removed
        assert 4 not in result_frames
        assert 5 not in result_frames
        assert 6 not in result_frames
        # Re-entry frames kept
        assert 7 in result_frames
        assert 8 in result_frames
        assert 9 in result_frames


    def test_ghost_anchor_does_not_rescue_nearby_false_positives(self) -> None:
        """Short segments near a ghost anchor should NOT be recovered.

        Reproduces the real bug: ghost segment [145-161] becomes an anchor,
        causing segment pruning to recover nearby false positives [135-136]
        at player positions. With ghost-aware pruning, the ghost anchor is
        excluded, so the false positive fragment gets pruned.
        """
        config = BallFilterConfig(
            enable_segment_pruning=True,
            enable_exit_ghost_removal=True,
            exit_edge_zone=0.10,
            exit_approach_frames=3,
            exit_min_approach_speed=0.008,
            max_interpolation_gap=10,
            min_segment_frames=15,
            segment_jump_threshold=0.20,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Main real trajectory: 20 frames at center-top of court
        for i in range(20):
            positions.append(_make_pos(i, 0.50, 0.20 + i * 0.002))

        # Ball approaches top edge (y decreasing toward 0)
        positions.append(_make_pos(20, 0.50, 0.08))
        positions.append(_make_pos(21, 0.50, 0.06))
        positions.append(_make_pos(22, 0.50, 0.04))
        positions.append(_make_pos(23, 0.50, 0.02))

        # Short gap (ball briefly gone)

        # False positive detections at player position (y=0.37)
        # These are close to the ghost segment below (within proximity)
        positions.append(_make_pos(26, 0.30, 0.37, conf=0.60))
        positions.append(_make_pos(27, 0.30, 0.37, conf=0.65))

        # Ghost detections drifting to player (reversal from top edge)
        # This forms a long segment that would normally be an anchor (≥15 frames)
        for i in range(20):
            positions.append(
                _make_pos(30 + i, 0.30, 0.12 + i * 0.015, conf=0.70)
            )

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Main trajectory should be kept
        assert all(f in result_frames for f in range(20))

        # The false positives at player position should be pruned
        # (ghost anchor excluded, so they have no anchor to recover from)
        assert 26 not in result_frames, (
            "False positive at player position should be pruned "
            "(ghost anchor should not rescue it)"
        )
        assert 27 not in result_frames

        # Ghost detections should be removed
        for f in range(30, 50):
            assert f not in result_frames, f"Ghost frame {f} should be removed"


    def test_partial_ghost_overlap_preserves_non_ghost_anchor(self) -> None:
        """Segment partially overlapping ghost range keeps anchor status.

        Reproduces the 0d84f858 bug: a segment has ghost overlap at the
        start, but the non-ghost portion is long enough (≥ min_segment_frames)
        to be an anchor on its own. The segment should keep anchor status
        so the non-ghost portion survives pruning.
        """
        config = BallFilterConfig(
            enable_segment_pruning=True,
            enable_exit_ghost_removal=True,
            exit_edge_zone=0.10,
            exit_approach_frames=3,
            exit_min_approach_speed=0.008,
            max_interpolation_gap=10,
            min_segment_frames=15,
            segment_jump_threshold=0.20,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # First anchor: 20 frames, ball approaching top edge
        for i in range(20):
            positions.append(
                _make_pos(i, 0.10, 0.30 - i * 0.014, conf=0.80)
            )
        # Last 3 frames approach top edge: y=0.072, 0.058, 0.044
        # (all approaching with speed > 0.008)

        # Ghost reversal: ball "comes back" from top edge
        # Frame 20 at y=0.15 is the reversal point (reversed from y≈0.04)
        positions.append(_make_pos(20, 0.10, 0.15, conf=0.70))

        # Frames 21-30: ghost portion drifting to player position
        for i in range(21, 31):
            positions.append(
                _make_pos(i, 0.40, 0.15 + (i - 21) * 0.01, conf=0.74)
            )

        # Gap of 15 frames (> max_interpolation_gap=10) → ghost range terminates

        # Frames 46-90: real detections (ball back in play, 45 frames)
        # These form a single segment with the ghost portion because
        # segment splitting uses position jumps, not frame gaps alone.
        # But they are NOT in the ghost range (terminated at gap).
        for i in range(46, 91):
            positions.append(
                _make_pos(i, 0.40, 0.25 + (i - 46) * 0.005, conf=0.76)
            )

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # First anchor should survive
        assert any(f < 20 for f in result_frames), "First anchor should survive"

        # Non-ghost portion should survive — the segment [ghost + real]
        # keeps anchor status because non-ghost frames ≥ min_segment_frames
        non_ghost_kept = sum(1 for f in range(46, 91) if f in result_frames)
        assert non_ghost_kept > 0, (
            "Non-ghost portion of partially-overlapping segment should survive"
        )


class TestTrajectoryBlipRemoval:
    """Tests for multi-frame trajectory blip removal."""

    def test_blip_at_player_position_removed(self) -> None:
        """3-frame blip at player position mid-trajectory should be removed."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=True,
            blip_context_min_frames=5,
            blip_max_deviation=0.15,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Real trajectory moving right at y≈0.16
        for i in range(15):
            positions.append(_make_pos(i, 0.35 + i * 0.02, 0.16))

        # Blip: 3 frames at player position (y=0.31, far from trajectory)
        positions.append(_make_pos(15, 0.33, 0.31))
        positions.append(_make_pos(16, 0.33, 0.31))
        positions.append(_make_pos(17, 0.34, 0.31))

        # Trajectory continues
        for i in range(18, 30):
            positions.append(_make_pos(i, 0.55 + (i - 18) * 0.02, 0.20))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Blip frames should be removed
        assert 15 not in result_frames, "Blip frame 15 should be removed"
        assert 16 not in result_frames, "Blip frame 16 should be removed"
        assert 17 not in result_frames, "Blip frame 17 should be removed"
        # Real trajectory preserved
        assert all(f in result_frames for f in range(15))
        assert all(f in result_frames for f in range(18, 30))

    def test_single_frame_deviation_preserved(self) -> None:
        """Single-frame deviation (real bounce) should NOT be removed."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=True,
            blip_context_min_frames=5,
            blip_max_deviation=0.15,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Smooth trajectory with single-frame direction change at f=15
        for i in range(15):
            positions.append(_make_pos(i, 0.3 + i * 0.01, 0.5))
        positions.append(_make_pos(15, 0.45, 0.7))  # Single-frame jump
        for i in range(16, 30):
            positions.append(_make_pos(i, 0.46 + (i - 16) * 0.01, 0.5))

        result = filt.filter_batch(positions)

        # All preserved — single frame deviation is not a blip
        assert len(result) == 30

    def test_blip_removal_disabled_by_config(self) -> None:
        """When enable_blip_removal=False, blips should be kept."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []
        for i in range(15):
            positions.append(_make_pos(i, 0.35 + i * 0.02, 0.16))
        # Blip
        positions.append(_make_pos(15, 0.33, 0.31))
        positions.append(_make_pos(16, 0.33, 0.31))
        for i in range(17, 30):
            positions.append(_make_pos(i, 0.55 + (i - 17) * 0.02, 0.20))

        result = filt.filter_batch(positions)
        assert len(result) == 30  # All preserved


class TestEndToEnd:
    """End-to-end tests with synthetic trajectories."""

    def test_synthetic_trajectory_with_noise(self) -> None:
        """Synthetic trajectory with flickering and false positives."""
        config = BallFilterConfig(
            enable_outlier_removal=True,
            enable_segment_pruning=True,
            segment_jump_threshold=0.20,
            min_segment_frames=5,
            min_output_confidence=0.05,
            enable_interpolation=True,
            max_interpolation_gap=10,
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

        result = filt.filter_batch(positions)

        # Should have output
        assert len(result) > 0

        # The flickering frame (20) should be removed by outlier removal
        # (interpolation may fill it with an interpolated position)
        result_by_frame = {p.frame_number: p for p in result}
        if 20 in result_by_frame:
            # Interpolated position should not be the original false detection
            assert result_by_frame[20].x != 0.9
            assert result_by_frame[20].y != 0.1

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
        """Raw mode should preserve original positions."""
        config = BallFilterConfig(
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

class TestBlipRemovalSpreadScaling:
    """Tests for blip removal with spread scaling for longer excursions."""

    def test_long_blip_with_transitional_spread_removed(self) -> None:
        """8-frame blip with transitional spread >5% but <11% should be removed.

        Longer blips have transitional frames as the tracker moves to/from
        the wrong position, creating more spread than the base 5% threshold.
        The scaled threshold for 8 frames = 0.05 + 0.01*(8-2) = 0.11.

        All frames must individually deviate >15% from the interpolated trajectory
        to pass Phase 1, while having internal spread between 5% and 11%.
        """
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=True,
            blip_context_min_frames=5,
            blip_max_deviation=0.15,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Real trajectory moving right at y≈0.16
        for i in range(15):
            positions.append(_make_pos(i, 0.35 + i * 0.02, 0.16))

        # 8-frame blip at player position y≈0.55-0.62
        # All frames deviate >25% from trajectory (which is at y≈0.16-0.20)
        # Internal spread ~0.08 (>5% base threshold but <11% scaled threshold)
        # Position is extreme enough that even Phase 1 (which includes suspect
        # frames as context) flags the edge frames as suspects.
        positions.append(_make_pos(15, 0.62, 0.55))
        positions.append(_make_pos(16, 0.63, 0.57))
        positions.append(_make_pos(17, 0.64, 0.60))
        positions.append(_make_pos(18, 0.64, 0.61))
        positions.append(_make_pos(19, 0.65, 0.60))
        positions.append(_make_pos(20, 0.64, 0.62))
        positions.append(_make_pos(21, 0.65, 0.61))
        positions.append(_make_pos(22, 0.66, 0.63))

        # Trajectory continues at y≈0.20
        for i in range(23, 38):
            positions.append(_make_pos(i, 0.65 + (i - 23) * 0.02, 0.20))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Blip frames (15-22) should be removed with scaled spread threshold
        blip_kept = [f for f in range(15, 23) if f in result_frames]
        assert len(blip_kept) == 0, (
            f"Blip frames should be removed with scaled spread, "
            f"but {blip_kept} survived"
        )
        # Real trajectory preserved
        assert all(f in result_frames for f in range(15))
        assert all(f in result_frames for f in range(23, 38))

    def test_short_blip_still_requires_tight_spread(self) -> None:
        """2-frame blip should still use base 5% spread threshold."""
        config = BallFilterConfig(
            enable_segment_pruning=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=True,
            blip_context_min_frames=5,
            blip_max_deviation=0.15,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Real trajectory
        for i in range(15):
            positions.append(_make_pos(i, 0.35 + i * 0.02, 0.16))

        # 2-frame blip at player position (compact, spread < 5%)
        positions.append(_make_pos(15, 0.33, 0.31))
        positions.append(_make_pos(16, 0.34, 0.31))

        # Trajectory continues
        for i in range(17, 30):
            positions.append(_make_pos(i, 0.55 + (i - 17) * 0.02, 0.20))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # 2-frame blip should still be removed (spread < base 5%)
        assert 15 not in result_frames
        assert 16 not in result_frames


class TestFalseStartAnchorRemoval:
    """Tests for false start/tail anchor removal in segment pruning."""

    def test_false_start_anchor_removed(self) -> None:
        """Short initial anchor with jump to longer anchor should be removed.

        Simulates detector warmup: 20-frame false segment at wrong position,
        then the real 100-frame trajectory starts. The false segment is long
        enough to be an anchor (>15 frames), but much shorter than the main
        trajectory and spatially disconnected.
        """
        config = BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.20,
            min_segment_frames=15,
            min_output_confidence=0.05,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # False start: 20 frames at wrong position (qualifies as anchor)
        for i in range(20):
            positions.append(_make_pos(i, 0.15 + i * 0.002, 0.70))

        # Large jump (>20% of screen)

        # Main trajectory: 100 frames (much longer anchor)
        for i in range(25, 125):
            positions.append(_make_pos(i, 0.50 + (i - 25) * 0.003, 0.30))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # False start frames should be removed
        assert not any(f < 20 for f in result_frames), (
            f"False start anchor should be removed, but found: "
            f"{sorted(f for f in result_frames if f < 20)}"
        )
        # Main trajectory preserved
        assert all(f in result_frames for f in range(25, 125))

    def test_false_tail_anchor_removed(self) -> None:
        """Short trailing anchor with jump from longer anchor should be removed."""
        config = BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.20,
            min_segment_frames=15,
            min_output_confidence=0.05,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Main trajectory: 100 frames
        for i in range(100):
            positions.append(_make_pos(i, 0.30 + i * 0.003, 0.30))

        # Large jump

        # False tail: 20 frames at wrong position (qualifies as anchor)
        for i in range(110, 130):
            positions.append(_make_pos(i, 0.15, 0.75 + (i - 110) * 0.002))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # False tail frames should be removed
        assert not any(f >= 110 for f in result_frames), (
            f"False tail anchor should be removed, but found: "
            f"{sorted(f for f in result_frames if f >= 110)}"
        )
        # Main trajectory preserved
        assert all(f in result_frames for f in range(100))

    def test_single_anchor_not_removed(self) -> None:
        """If there's only one anchor, it should NOT be removed."""
        config = BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.20,
            min_segment_frames=15,
            min_output_confidence=0.05,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # Only one anchor segment (20 frames)
        for i in range(20):
            positions.append(_make_pos(i, 0.50 + i * 0.005, 0.30))

        # Short non-anchor segment (5 frames, will be pruned)
        for i in range(25, 30):
            positions.append(_make_pos(i, 0.10, 0.80))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Single anchor should survive
        assert all(f in result_frames for f in range(20))

    def test_similar_length_anchors_both_kept(self) -> None:
        """Two anchors of similar length should both be kept (no false start)."""
        config = BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.20,
            min_segment_frames=15,
            min_output_confidence=0.05,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions: list[BallPosition] = []

        # First anchor: 50 frames
        for i in range(50):
            positions.append(_make_pos(i, 0.20 + i * 0.003, 0.30))

        # Large jump

        # Second anchor: 60 frames (similar length, not 3x longer)
        for i in range(60, 120):
            positions.append(_make_pos(i, 0.70 + (i - 60) * 0.003, 0.70))

        result = filt.filter_batch(positions)
        result_frames = {p.frame_number for p in result}

        # Both anchors should survive (50 is NOT < 60/3 = 20)
        assert any(f < 50 for f in result_frames), "First anchor should survive"
        assert any(f >= 60 for f in result_frames), "Second anchor should survive"


class TestSpatialOutlierAnchorRemoval:
    """Tests for Step 3d: spatial outlier anchor removal."""

    def _pruning_config(self) -> BallFilterConfig:
        return BallFilterConfig(
            enable_segment_pruning=True,
            segment_jump_threshold=0.20,
            min_segment_frames=15,
            min_output_confidence=0.05,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )

    def test_pigeon_segment_removed(self) -> None:
        """A pigeon drifting horizontally at top of frame should be removed.

        Pigeon moves left-to-right in X but stays at constant Y ~0.10.
        Its centroid is far enough from the real ball anchors, and its
        min-extent (Y) is tiny — no trajectory continuation protection.
        """
        filt = BallTemporalFilter(self._pruning_config())
        positions: list[BallPosition] = []

        # Pigeon: 30 frames drifting right at top (large X-extent, tiny Y-extent)
        for i in range(30):
            positions.append(_make_pos(i, 0.20 + i * 0.015, 0.10 + i * 0.001))

        # Gap
        for i in range(30, 35):
            positions.append(_make_pos(i, 0.0, 0.0, conf=0.0))

        # Real ball anchor 1: 50 frames at center
        for i in range(35, 85):
            positions.append(_make_pos(i, 0.45 + i * 0.002, 0.40 + i * 0.001))

        # Gap
        for i in range(85, 90):
            positions.append(_make_pos(i, 0.0, 0.0, conf=0.0))

        # Real ball anchor 2: 50 frames at center-right
        for i in range(90, 140):
            positions.append(_make_pos(i, 0.50 + i * 0.001, 0.35 - i * 0.001))

        result = filt.filter_batch(positions)
        confident = [p for p in result if p.confidence > 0]

        # Pigeon should be removed — no confident detections in frames 0-29
        pigeon_frames = [p for p in confident if p.frame_number < 30]
        assert len(pigeon_frames) == 0, (
            f"Pigeon segment should be removed, got {len(pigeon_frames)} frames"
        )
        # Real ball should survive
        ball_frames = [p for p in confident if p.frame_number >= 35]
        assert len(ball_frames) > 0, "Real ball segments should survive"

    def test_player_hand_compact_segment_removed(self) -> None:
        """A compact player-hand segment (low spread in both X and Y) removed."""
        filt = BallTemporalFilter(self._pruning_config())
        positions: list[BallPosition] = []

        # Real ball anchor 1: 50 frames
        for i in range(50):
            positions.append(_make_pos(i, 0.40 + i * 0.002, 0.20 + i * 0.001))

        # Gap
        for i in range(50, 55):
            positions.append(_make_pos(i, 0.0, 0.0, conf=0.0))

        # Player hand: 20 frames at far-right, compact cluster
        for i in range(55, 75):
            positions.append(_make_pos(i, 0.80 + i * 0.001, 0.45 + i * 0.001))

        # Gap
        for i in range(75, 80):
            positions.append(_make_pos(i, 0.0, 0.0, conf=0.0))

        # Real ball anchor 2: 50 frames
        for i in range(80, 130):
            positions.append(_make_pos(i, 0.42 + i * 0.001, 0.22 - i * 0.001))

        result = filt.filter_batch(positions)
        confident = [p for p in result if p.confidence > 0]

        # Player hand should be removed
        hand_frames = [p for p in confident if 55 <= p.frame_number < 75]
        assert len(hand_frames) == 0, (
            f"Player hand segment should be removed, got {len(hand_frames)} frames"
        )

    def test_ball_traversal_kept_via_trajectory_continuation(self) -> None:
        """A ball arcing across the court should be preserved.

        The ball traverses from top-left to bottom-right (large extent in
        both X and Y). Its centroid is far from other anchors, but its
        start endpoint connects to another anchor's end, and it has high
        2D spread. This is a trajectory continuation, not a different object.
        """
        filt = BallTemporalFilter(self._pruning_config())
        positions: list[BallPosition] = []

        # Ball anchor 1: 40 frames at center-top
        for i in range(40):
            positions.append(_make_pos(i, 0.40 + i * 0.002, 0.15 + i * 0.001))

        # Ball anchor 2: 30 frames continuing at center
        for i in range(40, 70):
            j = i - 40
            positions.append(_make_pos(i, 0.48 + j * 0.001, 0.19 - j * 0.002))

        # Small gap where ball exits/re-enters
        for i in range(70, 75):
            positions.append(_make_pos(i, 0.0, 0.0, conf=0.0))

        # Ball anchor 3: arcs from top-left to bottom-right (large 2D spread)
        # Start near anchor 2's end position (~0.51, 0.13) → endpoint connected
        for i in range(75, 115):
            j = i - 75
            # Arc from (0.35, 0.05) to (0.90, 0.55) — 0.55 X-extent, 0.50 Y-extent
            positions.append(_make_pos(
                i, 0.35 + j * 0.014, 0.05 + j * 0.013,
            ))

        result = filt.filter_batch(positions)
        confident = [p for p in result if p.confidence > 0]

        # All three anchors should survive — including the traversal
        traversal_frames = [p for p in confident if p.frame_number >= 75]
        assert len(traversal_frames) >= 30, (
            f"Ball traversal should be kept, got {len(traversal_frames)} frames"
        )

    def test_drifting_segment_not_protected_despite_endpoint_proximity(self) -> None:
        """A pigeon whose endpoint happens to be near a ball anchor is NOT protected.

        Endpoint connectivity alone is not enough — the segment must also
        have large 2D extent (min of X and Y extent > threshold).
        """
        filt = BallTemporalFilter(self._pruning_config())
        positions: list[BallPosition] = []

        # Pigeon: 25 frames drifting horizontally, ends near (0.45, 0.12)
        for i in range(25):
            positions.append(_make_pos(i, 0.20 + i * 0.010, 0.11 + i * 0.0005))

        # Gap
        for i in range(25, 30):
            positions.append(_make_pos(i, 0.0, 0.0, conf=0.0))

        # Real ball: starts near (0.49, 0.38) — >0.20 jump from pigeon to split
        # segments, but pigeon endpoint is within 0.20 of ball start (0.26 dist).
        for i in range(30, 80):
            positions.append(_make_pos(i, 0.43 + i * 0.002, 0.35 + i * 0.001))

        # Gap
        for i in range(80, 85):
            positions.append(_make_pos(i, 0.0, 0.0, conf=0.0))

        # Real ball anchor 2 (same general area as anchor 1)
        for i in range(85, 135):
            positions.append(_make_pos(i, 0.50 + i * 0.001, 0.40 - i * 0.001))

        result = filt.filter_batch(positions)
        confident = [p for p in result if p.confidence > 0]

        # Pigeon should be removed despite endpoint proximity
        pigeon_frames = [p for p in confident if p.frame_number < 25]
        assert len(pigeon_frames) == 0, (
            f"Pigeon with endpoint proximity should still be removed, "
            f"got {len(pigeon_frames)} frames"
        )

    def test_fewer_than_three_anchors_skips_outlier_removal(self) -> None:
        """With only 2 anchors, no spatial outlier removal is attempted."""
        filt = BallTemporalFilter(self._pruning_config())
        positions: list[BallPosition] = []

        # Anchor 1: 20 frames at top-left
        for i in range(20):
            positions.append(_make_pos(i, 0.10 + i * 0.001, 0.10))

        # Gap
        for i in range(20, 25):
            positions.append(_make_pos(i, 0.0, 0.0, conf=0.0))

        # Anchor 2: 20 frames at bottom-right (very far from anchor 1)
        for i in range(25, 45):
            positions.append(_make_pos(i, 0.80 + i * 0.001, 0.80))

        result = filt.filter_batch(positions)
        confident = [p for p in result if p.confidence > 0]

        # Both anchors should survive — only 2, so no outlier removal
        anchor1 = [p for p in confident if p.frame_number < 20]
        anchor2 = [p for p in confident if p.frame_number >= 25]
        assert len(anchor1) > 0, "First anchor should survive with only 2 anchors"
        assert len(anchor2) > 0, "Second anchor should survive with only 2 anchors"


class TestRemoveStationaryRuns:
    """Tests for stationarity detection (player lock-on removal)."""

    def test_stationary_wasb_run_zeroed(self) -> None:
        """20 frames at a fixed position should be zeroed (WASB player lock-on)."""
        config = BallFilterConfig(
            enable_stationarity_filter=True,
            stationarity_min_frames=12,
            stationarity_max_spread=0.005,
            enable_segment_pruning=False,
            enable_motion_energy_filter=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # 20 frames at nearly fixed position (WASB locked onto player)
        positions = []
        for i in range(20):
            # Tiny jitter within 0.1% — well under 0.5% threshold
            positions.append(
                _make_pos(i, 0.40 + i * 0.0001, 0.35 + i * 0.00005,
                          motion_energy=1.0)
            )

        result = filt.filter_batch(positions)

        # All positions should be zeroed
        confident = [p for p in result if p.confidence > 0]
        assert len(confident) == 0, (
            f"Expected all stationary positions zeroed, "
            f"but {len(confident)} still confident"
        )

    def test_moving_trajectory_preserved(self) -> None:
        """Trajectory with >2% spread per frame should be preserved."""
        config = BallFilterConfig(
            enable_stationarity_filter=True,
            stationarity_min_frames=12,
            stationarity_max_spread=0.005,
            enable_segment_pruning=False,
            enable_motion_energy_filter=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Real ball movement: ~2% per frame
        positions = []
        for i in range(20):
            positions.append(
                _make_pos(i, 0.30 + i * 0.02, 0.40, motion_energy=1.0)
            )

        result = filt.filter_batch(positions)

        # All positions should be preserved
        confident = [p for p in result if p.confidence > 0]
        assert len(confident) == 20

    def test_short_stationary_run_preserved(self) -> None:
        """10 frames at fixed position (below 12-frame threshold) should be preserved."""
        config = BallFilterConfig(
            enable_stationarity_filter=True,
            stationarity_min_frames=12,
            stationarity_max_spread=0.005,
            enable_segment_pruning=False,
            enable_motion_energy_filter=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # Only 10 stationary frames — below threshold of 12
        positions = []
        for i in range(10):
            positions.append(
                _make_pos(i, 0.40, 0.35, motion_energy=1.0)
            )

        result = filt.filter_batch(positions)

        # All should be preserved (too short to flag)
        confident = [p for p in result if p.confidence > 0]
        assert len(confident) == 10

    def test_stationarity_filter_disabled(self) -> None:
        """When disabled, stationary runs should be preserved."""
        config = BallFilterConfig(
            enable_stationarity_filter=False,
            enable_segment_pruning=False,
            enable_motion_energy_filter=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        # 15 stationary frames
        positions = []
        for i in range(15):
            positions.append(
                _make_pos(i, 0.40, 0.35, motion_energy=1.0)
            )

        result = filt.filter_batch(positions)

        # All preserved when filter is disabled
        confident = [p for p in result if p.confidence > 0]
        assert len(confident) == 15

    def test_large_gap_splits_windows(self) -> None:
        """Large gap prevents sliding window from spanning both groups."""
        config = BallFilterConfig(
            enable_stationarity_filter=True,
            stationarity_min_frames=12,
            stationarity_max_spread=0.005,
            enable_segment_pruning=False,
            enable_motion_energy_filter=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions = []
        # First group: 8 stationary frames (frames 0-7)
        for i in range(8):
            positions.append(
                _make_pos(i, 0.40, 0.35, motion_energy=1.0)
            )
        # Large gap of 40 frames — windows spanning this gap exceed
        # the temporal contiguity limit (min_frames * 3 = 36)
        # Second group: 8 stationary frames (frames 48-55)
        for i in range(48, 56):
            positions.append(
                _make_pos(i, 0.40, 0.35, motion_energy=1.0)
            )

        result = filt.filter_batch(positions)

        # Each group has 8 < 12 frames; windows spanning the gap are rejected
        # by temporal contiguity check, so all preserved
        confident = [p for p in result if p.confidence > 0]
        assert len(confident) == 16

    def test_interleaved_stationary_blocks_detected(self) -> None:
        """Stationary blocks interleaved with moving positions are detected.

        WASB can intermittently snap back to a fixed player position between
        real detections. The sliding window catches each stationary block
        independently, even within one contiguous trajectory.
        """
        config = BallFilterConfig(
            enable_stationarity_filter=True,
            stationarity_min_frames=12,
            stationarity_max_spread=0.005,
            enable_segment_pruning=False,
            enable_motion_energy_filter=False,
            enable_exit_ghost_removal=False,
            enable_oscillation_pruning=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        filt = BallTemporalFilter(config)

        positions = []
        # 15 frames locked at (0.10, 0.33) — stationary block
        for i in range(15):
            positions.append(
                _make_pos(i, 0.10, 0.33, motion_energy=1.0)
            )
        # 20 frames of real moving trajectory
        for i in range(15, 35):
            positions.append(
                _make_pos(i, 0.30 + i * 0.01, 0.40, motion_energy=1.0)
            )
        # 15 frames back at the same locked position
        for i in range(35, 50):
            positions.append(
                _make_pos(i, 0.10, 0.33, motion_energy=1.0)
            )

        result = filt.filter_batch(positions)
        confident = [p for p in result if p.confidence > 0]

        # The 20 moving frames should survive
        moving_survived = [
            p for p in confident
            if 15 <= p.frame_number < 35 and p.x > 0.20
        ]
        assert len(moving_survived) == 20, (
            f"Expected 20 moving frames, got {len(moving_survived)}"
        )

        # Both stationary blocks should be zeroed
        stationary_survived = [
            p for p in confident
            if abs(p.x - 0.10) < 0.02 and abs(p.y - 0.33) < 0.02
        ]
        assert len(stationary_survived) == 0, (
            f"Expected 0 stationary frames, got {len(stationary_survived)}"
        )
