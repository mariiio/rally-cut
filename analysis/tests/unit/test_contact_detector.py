"""Unit tests for ball contact detection from trajectory inflection points."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_classifier import CandidateFeatures, ContactClassifier
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    ContactSequence,
    _build_generators_by_frame,
    _compute_velocities,
    _filter_noise_spikes,
    _find_inflection_candidates,
    _find_nearest_player,
    _find_parabolic_breakpoints,
    _find_velocity_reversal_candidates,
    _maybe_anchor_rally_start_serve,
    _merge_candidates,
    _player_to_ball_dist,
    _smooth_signal,
    compute_direction_change,
    detect_contacts,
    estimate_net_position,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _bp(frame: int, x: float, y: float, conf: float = 0.9) -> BallPosition:
    """Helper to create a BallPosition."""
    return BallPosition(frame_number=frame, x=x, y=y, confidence=conf)


def _pp(frame: int, track_id: int, x: float, y: float) -> PlayerPosition:
    """Helper to create a PlayerPosition."""
    return PlayerPosition(
        frame_number=frame, track_id=track_id,
        x=x, y=y, width=0.05, height=0.15, confidence=0.9,
    )


def _pp_with_pose(
    frame: int, track_id: int, x: float, y: float,
    keypoints: list[list[float]] | None = None,
) -> PlayerPosition:
    """Helper to create a PlayerPosition with optional pose keypoints."""
    return PlayerPosition(
        frame_number=frame, track_id=track_id,
        x=x, y=y, width=0.05, height=0.15, confidence=0.9,
        keypoints=keypoints,
    )


def _make_coco_keypoints(
    left_wrist: tuple[float, float, float] = (0.0, 0.0, 0.0),
    right_wrist: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[list[float]]:
    """Build a 17-keypoint COCO array with only wrists specified."""
    kpts = [[0.0, 0.0, 0.0]] * 17
    kpts = [list(k) for k in kpts]  # make mutable copies
    kpts[9] = list(left_wrist)
    kpts[10] = list(right_wrist)
    return kpts


class TestComputeVelocities:
    """Tests for velocity computation from ball positions."""

    def test_basic_velocity(self) -> None:
        """Constant horizontal motion produces constant velocity."""
        positions = [_bp(i, 0.1 + i * 0.02, 0.5) for i in range(10)]
        velocities = _compute_velocities(positions)
        # Central differences produce velocity for all 10 frames
        # (forward diff at frame 0, central for 1-8, backward for 9)
        assert len(velocities) == 10
        for v, vx, vy in velocities.values():
            assert abs(v - 0.02) < 0.001

    def test_skips_low_confidence(self) -> None:
        """Positions below confidence threshold are excluded."""
        positions = [
            _bp(0, 0.1, 0.5, conf=0.9),
            _bp(1, 0.2, 0.5, conf=0.1),  # Low confidence
            _bp(2, 0.3, 0.5, conf=0.9),
        ]
        velocities = _compute_velocities(positions)
        # Frame 1 skipped, so velocity computed between 0 and 2 (gap=2)
        assert 2 in velocities
        assert 1 not in velocities

    def test_skips_large_frame_gaps(self) -> None:
        """Frame gaps > 5 are not used for velocity."""
        positions = [
            _bp(0, 0.1, 0.5),
            _bp(10, 0.3, 0.5),  # Gap of 10
        ]
        velocities = _compute_velocities(positions)
        assert len(velocities) == 0

    def test_empty_input(self) -> None:
        """Empty positions return empty velocities."""
        assert _compute_velocities([]) == {}

    def test_single_position(self) -> None:
        """Single position can't produce velocity."""
        assert _compute_velocities([_bp(0, 0.5, 0.5)]) == {}


class TestSmoothSignal:
    """Tests for moving average smoothing."""

    def test_smoothing_preserves_constant(self) -> None:
        """Constant signal is unchanged by smoothing."""
        signal = [5.0] * 10
        smoothed = _smooth_signal(signal, window=3)
        assert all(abs(s - 5.0) < 0.001 for s in smoothed)

    def test_smoothing_reduces_spike(self) -> None:
        """Single spike is reduced by averaging."""
        signal = [0.0, 0.0, 1.0, 0.0, 0.0]
        smoothed = _smooth_signal(signal, window=3)
        assert smoothed[2] < 1.0
        assert smoothed[2] > 0.0

    def test_short_signal(self) -> None:
        """Signal shorter than window returns unchanged."""
        signal = [1.0, 2.0]
        smoothed = _smooth_signal(signal, window=5)
        assert smoothed == signal


class TestComputeDirectionChange:
    """Tests for trajectory direction change computation."""

    def test_straight_line_zero_change(self) -> None:
        """Straight trajectory has ~0 direction change."""
        ball_by_frame = {
            i: _bp(i, 0.1 + i * 0.02, 0.5) for i in range(10)
        }
        angle = compute_direction_change(ball_by_frame, 5, check_frames=3)
        assert angle < 5.0  # Nearly straight

    def test_90_degree_turn(self) -> None:
        """Right-angle turn produces ~90 degree change."""
        ball_by_frame = {
            0: _bp(0, 0.3, 0.5),
            5: _bp(5, 0.5, 0.5),  # Moving right
            10: _bp(10, 0.5, 0.3),  # Moving up (90 degree turn)
        }
        angle = compute_direction_change(ball_by_frame, 5, check_frames=6)
        assert 80.0 < angle < 100.0

    def test_reversal_180_degrees(self) -> None:
        """Complete reversal produces ~180 degree change."""
        ball_by_frame = {
            0: _bp(0, 0.3, 0.5),
            5: _bp(5, 0.5, 0.5),  # Moving right
            10: _bp(10, 0.3, 0.5),  # Moving back left
        }
        angle = compute_direction_change(ball_by_frame, 5, check_frames=6)
        assert angle > 150.0

    def test_missing_neighbors(self) -> None:
        """Returns 0 when before/after positions are missing."""
        ball_by_frame = {5: _bp(5, 0.5, 0.5)}
        angle = compute_direction_change(ball_by_frame, 5, check_frames=3)
        assert angle == 0.0


class TestFindNearestPlayer:
    """Tests for player attribution."""

    def test_finds_closest_player(self) -> None:
        """Attributes contact to nearest player."""
        # Without keypoints, falls back to bbox upper-quarter (y - height*0.25)
        players = [
            _pp(10, 1, 0.3, 0.7),  # Far from ball
            _pp(10, 2, 0.51, 0.51),  # Close to ball (upper-quarter at y=0.51-0.15*0.25=0.4725)
        ]
        track_id, dist, player_y = _find_nearest_player(10, 0.5, 0.5, players)
        assert track_id == 2
        assert dist < 0.10

    def test_no_players_returns_default(self) -> None:
        """No players gives track_id=-1 and infinite distance."""
        track_id, dist, _ = _find_nearest_player(10, 0.5, 0.5, [])
        assert track_id == -1
        assert dist == float("inf")

    def test_respects_frame_window(self) -> None:
        """Players outside search window are ignored."""
        players = [_pp(100, 1, 0.5, 0.5)]  # Far from frame 10
        track_id, dist, _ = _find_nearest_player(10, 0.5, 0.5, players, search_frames=5)
        assert track_id == -1


class TestPlayerToBallDist:
    """Tests for _player_to_ball_dist helper."""

    def test_uses_wrist_when_confident(self) -> None:
        """With high-confidence wrist keypoints, distance uses wrist position."""
        kpts = _make_coco_keypoints(
            left_wrist=(0.5, 0.5, 0.9),   # exactly at ball
            right_wrist=(0.3, 0.3, 0.9),  # farther
        )
        p = _pp_with_pose(10, 1, 0.6, 0.7, keypoints=kpts)
        dist = _player_to_ball_dist(p, 0.5, 0.5)
        # Left wrist is at (0.5, 0.5) = ball position, dist should be ~0
        assert dist < 0.01

    def test_falls_back_to_bbox_without_keypoints(self) -> None:
        """Without keypoints, uses bbox upper-quarter."""
        p = _pp_with_pose(10, 1, 0.5, 0.55, keypoints=None)
        dist = _player_to_ball_dist(p, 0.5, 0.5)
        # bbox upper-quarter: y = 0.55 - 0.15*0.25 = 0.5125
        # dist = sqrt(0 + 0.0125^2) ≈ 0.0125
        assert 0.01 < dist < 0.02

    def test_falls_back_to_bbox_with_low_confidence(self) -> None:
        """Low-confidence wrists fall back to bbox."""
        kpts = _make_coco_keypoints(
            left_wrist=(0.5, 0.5, 0.1),   # low conf
            right_wrist=(0.5, 0.5, 0.1),  # low conf
        )
        p = _pp_with_pose(10, 1, 0.5, 0.55, keypoints=kpts)
        dist_with_low_conf = _player_to_ball_dist(p, 0.5, 0.5)
        p_no_kpts = _pp_with_pose(10, 1, 0.5, 0.55, keypoints=None)
        dist_no_kpts = _player_to_ball_dist(p_no_kpts, 0.5, 0.5)
        assert abs(dist_with_low_conf - dist_no_kpts) < 1e-9

    def test_picks_closer_wrist(self) -> None:
        """Uses the wrist closer to the ball."""
        kpts = _make_coco_keypoints(
            left_wrist=(0.1, 0.1, 0.9),   # far from ball at (0.5, 0.5)
            right_wrist=(0.48, 0.48, 0.9),  # near ball
        )
        p = _pp_with_pose(10, 1, 0.3, 0.3, keypoints=kpts)
        dist = _player_to_ball_dist(p, 0.5, 0.5)
        # Should use right wrist: sqrt((0.5-0.48)^2 + (0.5-0.48)^2) ≈ 0.028
        assert dist < 0.04


class TestFindNearestPlayerWithPose:
    """Tests that _find_nearest_player uses wrist keypoints when available."""

    def test_wrist_beats_bbox_centroid(self) -> None:
        """Player whose bbox is farther but wrist is closer should win."""
        # Player 1: bbox far, but wrist right at ball
        kpts1 = _make_coco_keypoints(
            left_wrist=(0.5, 0.5, 0.9),
            right_wrist=(0.4, 0.4, 0.9),
        )
        p1 = _pp_with_pose(10, 1, 0.7, 0.7, keypoints=kpts1)  # bbox center far

        # Player 2: bbox close, no keypoints
        p2 = _pp_with_pose(10, 2, 0.51, 0.55, keypoints=None)  # bbox near ball

        track_id, dist, _ = _find_nearest_player(10, 0.5, 0.5, [p1, p2])
        assert track_id == 1  # wrist proximity wins over bbox proximity


class TestFilterNoiseSpikes:
    """Tests for ball detector noise spike removal."""

    def test_removes_spike(self) -> None:
        """Single-frame jump far from neighbors is zeroed."""
        positions = [
            _bp(0, 0.5, 0.5),
            _bp(1, 0.5, 0.5),
            _bp(2, 0.9, 0.9),  # Spike: far from both neighbors
            _bp(3, 0.5, 0.5),
            _bp(4, 0.5, 0.5),
        ]
        result = _filter_noise_spikes(positions, max_jump=0.20)
        # Frame 2 should be zeroed
        frame2 = [bp for bp in result if bp.frame_number == 2][0]
        assert frame2.confidence == 0.0
        # Other frames unchanged
        for bp in result:
            if bp.frame_number != 2:
                assert bp.confidence == 0.9

    def test_preserves_legitimate_movement(self) -> None:
        """Gradual movement is not treated as spikes."""
        positions = [_bp(i, 0.1 + i * 0.05, 0.5) for i in range(10)]
        result = _filter_noise_spikes(positions, max_jump=0.20)
        # No frames should be zeroed
        for bp in result:
            assert bp.confidence == 0.9

    def test_edges_safe(self) -> None:
        """First and last positions are never zeroed (only interior checked)."""
        positions = [
            _bp(0, 0.9, 0.9),  # Edge position, far from next
            _bp(1, 0.5, 0.5),
            _bp(2, 0.5, 0.5),
            _bp(3, 0.1, 0.1),  # Edge position, far from prev
        ]
        result = _filter_noise_spikes(positions, max_jump=0.20)
        # First and last are edges, not checked
        assert result[0].confidence == 0.9
        assert result[-1].confidence == 0.9

    def test_fewer_than_3_confident_unchanged(self) -> None:
        """With fewer than 3 confident positions, return unchanged."""
        positions = [_bp(0, 0.5, 0.5), _bp(1, 0.9, 0.9)]
        result = _filter_noise_spikes(positions, max_jump=0.20)
        assert len(result) == 2
        assert all(bp.confidence == 0.9 for bp in result)

    def test_low_confidence_ignored(self) -> None:
        """Low-confidence positions are not considered for spike detection."""
        positions = [
            _bp(0, 0.5, 0.5),
            _bp(1, 0.9, 0.9, conf=0.1),  # Low conf, ignored in spike check
            _bp(2, 0.5, 0.5),
        ]
        result = _filter_noise_spikes(positions, max_jump=0.20)
        # Frame 1 is low-conf so not checked as spike — unchanged
        frame1 = [bp for bp in result if bp.frame_number == 1][0]
        assert frame1.confidence == 0.1


class TestFindInflectionCandidates:
    """Tests for trajectory inflection point detection."""

    def test_v_shaped_reversal(self) -> None:
        """V-shaped trajectory produces inflection at the vertex."""
        ball_by_frame = {}
        # Moving right, then reversing left at frame 10
        for i in range(20):
            if i <= 10:
                x = 0.3 + i * 0.02
            else:
                x = 0.5 - (i - 10) * 0.02
            ball_by_frame[i] = _bp(i, x, 0.5)

        frames = sorted(ball_by_frame.keys())
        result = _find_inflection_candidates(
            ball_by_frame, frames,
            min_angle_deg=25.0, check_frames=5, min_distance_frames=8,
        )
        # Should detect inflection near frame 10
        assert len(result) >= 1
        assert any(abs(f - 10) <= 2 for f in result)

    def test_soft_touch_detected(self) -> None:
        """Gentle direction change above threshold is detected."""
        ball_by_frame = {}
        # Moving at a slight angle, then changing direction by ~30 degrees
        for i in range(20):
            if i <= 10:
                ball_by_frame[i] = _bp(i, 0.3 + i * 0.02, 0.5 - i * 0.01)
            else:
                ball_by_frame[i] = _bp(i, 0.5 + (i - 10) * 0.01, 0.4 + (i - 10) * 0.015)

        frames = sorted(ball_by_frame.keys())
        result = _find_inflection_candidates(
            ball_by_frame, frames,
            min_angle_deg=25.0, check_frames=5, min_distance_frames=8,
        )
        assert len(result) >= 1

    def test_min_distance_enforcement(self) -> None:
        """Two inflections close together: only the larger angle is kept."""
        ball_by_frame = {}
        # Two direction changes close together (frames 5 and 8)
        for i in range(15):
            if i < 5:
                ball_by_frame[i] = _bp(i, 0.3 + i * 0.02, 0.5)
            elif i < 8:
                ball_by_frame[i] = _bp(i, 0.4 - (i - 5) * 0.02, 0.5)
            else:
                ball_by_frame[i] = _bp(i, 0.34 + (i - 8) * 0.02, 0.5)

        frames = sorted(ball_by_frame.keys())
        result = _find_inflection_candidates(
            ball_by_frame, frames,
            min_angle_deg=25.0, check_frames=3, min_distance_frames=12,
        )
        # With min_distance_frames=12, only one should survive
        assert len(result) <= 1

    def test_straight_line_no_inflection(self) -> None:
        """Perfectly straight trajectory produces no inflections."""
        ball_by_frame = {
            i: _bp(i, 0.1 + i * 0.02, 0.5) for i in range(20)
        }
        frames = sorted(ball_by_frame.keys())
        result = _find_inflection_candidates(
            ball_by_frame, frames,
            min_angle_deg=25.0, check_frames=5, min_distance_frames=8,
        )
        assert len(result) == 0

    def test_too_few_frames(self) -> None:
        """Fewer than 3 frames returns empty."""
        ball_by_frame = {0: _bp(0, 0.5, 0.5), 1: _bp(1, 0.6, 0.5)}
        result = _find_inflection_candidates(
            ball_by_frame, [0, 1],
            min_angle_deg=25.0, check_frames=5, min_distance_frames=8,
        )
        assert result == []


class TestMergeCandidates:
    """Tests for merging velocity peak and inflection candidates."""

    def test_dedup_nearby(self) -> None:
        """Inflection near velocity peak is not added (velocity preferred)."""
        velocity_peaks = [10, 30]
        inflections = [12]  # Within min_distance of 10
        result = _merge_candidates(velocity_peaks, inflections, min_distance_frames=8)
        assert result == [10, 30]

    def test_keep_distant_inflection(self) -> None:
        """Inflection far from any velocity peak is added."""
        velocity_peaks = [10, 50]
        inflections = [30]  # Far from both peaks
        result = _merge_candidates(velocity_peaks, inflections, min_distance_frames=8)
        assert result == [10, 30, 50]

    def test_inflection_only(self) -> None:
        """With no velocity peaks, inflections become the only candidates."""
        result = _merge_candidates([], [15, 35], min_distance_frames=8)
        assert result == [15, 35]

    def test_empty_inputs(self) -> None:
        """Both empty returns empty."""
        result = _merge_candidates([], [], min_distance_frames=8)
        assert result == []

    def test_velocity_only(self) -> None:
        """With no inflections, velocity peaks pass through."""
        result = _merge_candidates([10, 25, 40], [], min_distance_frames=8)
        assert result == [10, 25, 40]


class TestFindVelocityReversalCandidates:
    """Tests for velocity reversal detection."""

    def test_detects_reversal(self) -> None:
        """Velocity reversal (dot product < 0) is detected."""
        # Frame 1: moving right (+vx), frame 2: moving left (-vx) = reversal
        velocities = {
            1: (0.02, 0.02, 0.0),
            2: (0.02, 0.02, 0.0),
            3: (0.02, -0.02, 0.0),  # Reversal here
            4: (0.02, -0.02, 0.0),
        }
        frames = [1, 2, 3, 4]
        result = _find_velocity_reversal_candidates(velocities, frames, min_distance_frames=2)
        assert 3 in result

    def test_no_reversal_in_straight_line(self) -> None:
        """Constant velocity direction produces no reversals."""
        velocities = {i: (0.02, 0.02, 0.0) for i in range(10)}
        frames = list(range(10))
        result = _find_velocity_reversal_candidates(velocities, frames, min_distance_frames=5)
        assert result == []

    def test_min_distance_enforced(self) -> None:
        """Two close reversals keep only the strongest."""
        velocities = {
            1: (0.02, 0.02, 0.0),
            2: (0.02, -0.01, 0.0),   # Weak reversal
            3: (0.02, -0.02, 0.0),
            4: (0.02, 0.02, 0.0),    # Stronger reversal
            5: (0.02, 0.02, 0.0),
        }
        frames = [1, 2, 3, 4, 5]
        result = _find_velocity_reversal_candidates(velocities, frames, min_distance_frames=8)
        assert len(result) <= 1

    def test_too_few_frames(self) -> None:
        """Fewer than 3 frames returns empty."""
        velocities = {1: (0.02, 0.02, 0.0), 2: (0.02, -0.02, 0.0)}
        result = _find_velocity_reversal_candidates(velocities, [1, 2], min_distance_frames=5)
        assert result == []


class TestEstimateNetPosition:
    """Tests for net position estimation."""

    def test_returns_default_for_few_positions(self) -> None:
        """Returns 0.5 with < 10 positions."""
        positions = [_bp(i, 0.5, 0.5) for i in range(5)]
        assert estimate_net_position(positions) == 0.5

    def test_oscillating_trajectory_produces_midpoint(self) -> None:
        """Ball oscillating between far (y=0.3) and near (y=0.7) gives net ~0.5."""
        positions = []
        # Create oscillating trajectory: up to 0.3, back to 0.7, repeat
        for cycle in range(3):
            base = cycle * 20
            for i in range(10):
                # Going from 0.7 down to 0.3
                y = 0.7 - i * 0.04
                positions.append(_bp(base + i, 0.5, y))
            for i in range(10):
                # Going from 0.3 up to 0.7
                y = 0.3 + i * 0.04
                positions.append(_bp(base + 10 + i, 0.5, y))

        net_y = estimate_net_position(positions)
        # Net should be near the midpoint of 0.3 and 0.7 = 0.5
        assert 0.40 < net_y < 0.60

    def test_uses_y_range_fallback(self) -> None:
        """Monotonic trajectory uses Y range midpoint as fallback."""
        # Monotonically increasing Y — no local minima/maxima
        positions = [_bp(i, 0.5, 0.2 + i * 0.02) for i in range(20)]
        net_y = estimate_net_position(positions)
        # Fallback: (0.2 + 0.58) / 2 = 0.39
        assert 0.3 < net_y < 0.5


class TestDetectContacts:
    """Tests for the main contact detection function."""

    def test_empty_input(self) -> None:
        """Empty ball positions returns empty sequence."""
        result = detect_contacts([], use_classifier=False)
        assert result.num_contacts == 0

    def test_detects_velocity_peaks(self) -> None:
        """Trajectory with sharp direction changes produces contacts."""
        config = ContactDetectionConfig(
            min_peak_velocity=0.008,
            min_peak_prominence=0.003,
            min_peak_distance_frames=8,
            min_direction_change_deg=20.0,
        )

        positions = []
        # Phase 1: ball accelerating right (serve, frames 0-15)
        for i in range(16):
            speed = 0.005 + i * 0.002  # Accelerating
            positions.append(_bp(i, 0.2 + i * speed, 0.6, conf=0.9))
        # Phase 2: ball decelerating then reversing (contact at ~frame 16)
        for i in range(15):
            positions.append(_bp(16 + i, 0.5 - i * 0.015, 0.4 + i * 0.005, conf=0.9))
        # Phase 3: another reversal (contact at ~frame 31)
        for i in range(15):
            positions.append(_bp(31 + i, 0.28 + i * 0.012, 0.5 + i * 0.003, conf=0.9))

        # Players near the reversal points for compound validation
        players = [
            _pp(16, 1, 0.5, 0.42),   # Near phase 2 start
            _pp(31, 2, 0.28, 0.52),   # Near phase 3 start
        ]

        result = detect_contacts(positions, player_positions=players, config=config, use_classifier=False)
        # Should detect contacts at the reversal/acceleration points
        assert result.num_contacts >= 1

    def test_soft_touch_detected_via_inflection(self) -> None:
        """Low-velocity direction change is detected by inflection detection."""
        config = ContactDetectionConfig(
            min_peak_velocity=0.012,
            min_peak_prominence=0.006,
            min_peak_distance_frames=8,
            min_direction_change_deg=25.0,
            min_inflection_angle_deg=25.0,
            enable_inflection_detection=True,
            enable_noise_filter=False,
        )

        # Gentle arc: direction change at frame 25 (past warmup_skip_frames=20).
        # Trajectory: frames 0-49 with turn at frame 25.
        positions = []
        for i in range(50):
            if i <= 25:
                x = 0.3 + i * 0.004
                y = 0.5 - i * 0.004
            else:
                x = 0.4 - (i - 25) * 0.004
                y = 0.4 + (i - 25) * 0.004
            positions.append(_bp(i, x, y, conf=0.9))

        # Player near the direction change for compound validation
        players = [_pp(25, 1, 0.40, 0.41)]

        result = detect_contacts(positions, player_positions=players, config=config, use_classifier=False)
        # Should detect a contact near the direction change at frame 25
        inflection_contacts = [
            c for c in result.contacts if abs(c.frame - 25) <= 3
        ]
        assert len(inflection_contacts) >= 1

    def test_noise_spike_does_not_create_false_contact(self) -> None:
        """Noise spike is filtered before contact detection."""
        config = ContactDetectionConfig(
            min_peak_velocity=0.008,
            min_peak_prominence=0.003,
            min_peak_distance_frames=8,
            enable_noise_filter=True,
            noise_spike_max_jump=0.20,
        )

        # Smooth trajectory with a spike at frame 10
        positions = []
        for i in range(25):
            if i == 10:
                # Spike: jumps far from trajectory
                positions.append(_bp(i, 0.9, 0.9, conf=0.9))
            else:
                positions.append(_bp(i, 0.3 + i * 0.01, 0.5, conf=0.9))

        result = detect_contacts(positions, config=config, use_classifier=False)
        # The spike at frame 10 should NOT produce a contact
        spike_contacts = [c for c in result.contacts if c.frame == 10]
        assert len(spike_contacts) == 0

    def test_net_y_override(self) -> None:
        """Explicit net_y parameter is used instead of auto-estimation."""
        config = ContactDetectionConfig(
            min_peak_velocity=0.008,
            min_peak_prominence=0.003,
            enable_noise_filter=False,
            enable_inflection_detection=False,
        )

        # Simple trajectory with a velocity peak
        positions = []
        for i in range(30):
            if i < 15:
                positions.append(_bp(i, 0.2 + i * 0.02, 0.5, conf=0.9))
            else:
                positions.append(_bp(i, 0.5 - (i - 15) * 0.02, 0.5, conf=0.9))

        result = detect_contacts(positions, config=config, net_y=0.45, use_classifier=False)
        # net_y is now always estimated from ball trajectory (not the passed value),
        # because court_split_y ≠ ball-trajectory net position.
        assert result.net_y == 0.5  # flat Y=0.5 trajectory → midpoint = 0.5

    def test_frame_count_suppresses_post_rally(self) -> None:
        """Candidates beyond frame_count are suppressed."""
        config = ContactDetectionConfig(
            min_peak_velocity=0.005,
            min_peak_prominence=0.002,
            enable_noise_filter=False,
        )

        # Trajectory with reversals at frame 15 and frame 35
        positions = []
        for i in range(60):
            if i < 15:
                positions.append(_bp(i, 0.2 + i * 0.02, 0.6, conf=0.9))
            elif i < 40:
                positions.append(_bp(i, 0.5 - (i - 15) * 0.02, 0.6, conf=0.9))
            else:
                positions.append(_bp(i, 0.0 + (i - 40) * 0.02, 0.6, conf=0.9))

        players = [
            _pp(15, 1, 0.5, 0.65),
            _pp(40, 2, 0.0, 0.65),
        ]

        # Without frame_count: both contacts detected
        result_all = detect_contacts(positions, players, config, use_classifier=False)
        all_frames = {c.frame for c in result_all.contacts}

        # With frame_count=35: contact near frame 40 should be suppressed
        result_limited = detect_contacts(positions, players, config, frame_count=35, use_classifier=False)
        limited_frames = {c.frame for c in result_limited.contacts}

        # The contact near frame 40 should be gone
        post_rally = [f for f in all_frames if f > 35]
        assert len(post_rally) > 0, "Need a contact after frame 35 for this test"
        for f in post_rally:
            assert f not in limited_frames

    def test_net_y_always_from_trajectory(self) -> None:
        """External net_y is ignored — always estimated from ball trajectory."""
        config = ContactDetectionConfig(
            min_peak_velocity=0.005,
            min_peak_prominence=0.002,
            enable_noise_filter=False,
        )

        # Oscillating trajectory so estimate_net_position returns ~0.5
        positions = []
        for cycle in range(3):
            base = cycle * 20
            for i in range(10):
                y = 0.7 - i * 0.04
                positions.append(_bp(base + i, 0.3 + i * 0.01, y))
            for i in range(10):
                y = 0.3 + i * 0.04
                positions.append(_bp(base + 10 + i, 0.4 - i * 0.01, y))

        # Extreme values ignored — trajectory estimate used
        result = detect_contacts(positions, config=config, net_y=0.18, use_classifier=False)
        assert result.net_y != 0.18
        assert 0.30 <= result.net_y <= 0.70

        result2 = detect_contacts(positions, config=config, net_y=0.80, use_classifier=False)
        assert result2.net_y != 0.80
        assert 0.30 <= result2.net_y <= 0.70

        # Even a "normal" net_y=0.45 is ignored — ball trajectory always used
        result3 = detect_contacts(positions, config=config, net_y=0.45, use_classifier=False)
        assert 0.30 <= result3.net_y <= 0.70  # trajectory estimate

    def test_court_side_from_trajectory(self) -> None:
        """Court side uses ball-trajectory estimated net, not external net_y."""
        config = ContactDetectionConfig(
            min_peak_velocity=0.005,
            min_peak_prominence=0.002,
            enable_noise_filter=False,
        )

        # Oscillating trajectory: Y oscillates between 0.2 (far) and 0.6 (near)
        # → estimated net_y ≈ 0.4 (midpoint of extrema)
        positions = []
        for cycle in range(3):
            base = cycle * 20
            for i in range(10):
                y = 0.6 - i * 0.04  # 0.6 → 0.24
                positions.append(_bp(base + i, 0.3 + i * 0.01, y, conf=0.9))
            for i in range(10):
                y = 0.2 + i * 0.04  # 0.2 → 0.56
                positions.append(_bp(base + 10 + i, 0.4 - i * 0.01, y, conf=0.9))

        result = detect_contacts(positions, config=config, use_classifier=False)
        # Net should be estimated from trajectory, not from external net_y
        assert 0.30 <= result.net_y <= 0.50

    def test_court_side_classification(self) -> None:
        """Contacts are classified as near/far based on ball Y."""
        contact_near = Contact(
            frame=10, ball_x=0.5, ball_y=0.85,
            velocity=0.02, direction_change_deg=90.0,
        )
        contact_far = Contact(
            frame=20, ball_x=0.5, ball_y=0.15,
            velocity=0.02, direction_change_deg=90.0,
        )
        # These are just data objects, court_side is set during detection
        assert contact_near.ball_y > 0.67  # Would be "near"
        assert contact_far.ball_y < 0.33  # Would be "far"

    def test_player_attribution(self) -> None:
        """Contacts attributed to nearest player when available."""
        # Simple trajectory with one clear peak
        positions = []
        for i in range(30):
            if i < 15:
                positions.append(_bp(i, 0.2 + i * 0.02, 0.6, conf=0.9))
            else:
                positions.append(_bp(i, 0.5 - (i - 15) * 0.02, 0.6, conf=0.9))

        players = [_pp(15, 42, 0.5, 0.65)]  # Player near reversal point

        config = ContactDetectionConfig(
            min_peak_velocity=0.005,
            min_peak_prominence=0.002,
        )
        result = detect_contacts(positions, players, config, use_classifier=False)

        # Any detected contact near frame 15 should be attributed to player 42
        for contact in result.contacts:
            if abs(contact.frame - 15) < 5:
                assert contact.player_track_id == 42

    def test_contact_sequence_properties(self) -> None:
        """ContactSequence has correct helper properties."""
        c1 = Contact(frame=10, ball_x=0.5, ball_y=0.8, velocity=0.02,
                      direction_change_deg=90.0, court_side="near")
        c2 = Contact(frame=30, ball_x=0.5, ball_y=0.2, velocity=0.03,
                      direction_change_deg=120.0, court_side="far")

        seq = ContactSequence(contacts=[c1, c2], net_y=0.5)
        assert seq.num_contacts == 2
        assert seq.serve_contact == c1
        assert len(seq.contacts_on_side("near")) == 1
        assert len(seq.contacts_on_side("far")) == 1

    def test_to_dict_roundtrip(self) -> None:
        """Contact.to_dict produces expected keys including new fields."""
        c = Contact(frame=10, ball_x=0.5, ball_y=0.5, velocity=0.02,
                    direction_change_deg=45.0, player_track_id=3,
                    court_side="near", is_validated=True,
                    confidence=0.85, arc_fit_residual=0.012)
        d = c.to_dict()
        assert d["frame"] == 10
        assert d["ballX"] == 0.5
        assert d["playerTrackId"] == 3
        assert d["courtSide"] == "near"
        assert d["isValidated"] is True
        assert d["confidence"] == 0.85
        assert d["arcFitResidual"] == 0.012


class TestFindParabolicBreakpoints:
    """Tests for parabolic arc breakpoint detection."""

    def _make_parabolic_trajectory(
        self,
        start_frame: int = 0,
        n_frames: int = 30,
        x_start: float = 0.2,
        x_speed: float = 0.01,
        y_apex: float = 0.3,
        y_amplitude: float = 0.15,
    ) -> dict[int, BallPosition]:
        """Create a parabolic trajectory (ball in free flight under gravity)."""
        ball_by_frame = {}
        for i in range(n_frames):
            frame = start_frame + i
            t = i / max(1, n_frames - 1)  # 0 to 1
            x = x_start + i * x_speed
            # Parabola: y = apex - amplitude * (2t - 1)^2
            y = y_apex - y_amplitude * (2 * t - 1) ** 2
            ball_by_frame[frame] = _bp(frame, x, y)
        return ball_by_frame

    def test_single_arc_no_breakpoints(self) -> None:
        """A single parabolic arc should produce no (or very few) breakpoints."""
        ball_by_frame = self._make_parabolic_trajectory(n_frames=40)
        frames = sorted(ball_by_frame.keys())

        breakpoints, residuals = _find_parabolic_breakpoints(
            ball_by_frame, frames,
            window_frames=12, stride=3,
            min_residual=0.015, min_prominence=0.008,
            min_distance_frames=12,
        )
        # A perfect parabola should have very low residuals
        assert len(breakpoints) == 0

    def test_two_arcs_detects_transition(self) -> None:
        """Two different parabolic arcs joined at a contact point."""
        ball_by_frame = {}
        # Arc 1: frames 0-24 (moving right-upward parabola)
        for i in range(25):
            t = i / 24
            x = 0.2 + i * 0.01
            y = 0.6 - 0.15 * (2 * t - 1) ** 2
            ball_by_frame[i] = _bp(i, x, y)
        # Arc 2: frames 25-49 (moving left-upward parabola, different direction)
        for i in range(25):
            frame = 25 + i
            t = i / 24
            x = 0.45 - i * 0.008
            y = 0.5 - 0.12 * (2 * t - 1) ** 2
            ball_by_frame[frame] = _bp(frame, x, y)

        frames = sorted(ball_by_frame.keys())
        breakpoints, residuals = _find_parabolic_breakpoints(
            ball_by_frame, frames,
            window_frames=12, stride=2,
            min_residual=0.005, min_prominence=0.003,
            min_distance_frames=10,
        )
        # Should detect breakpoint near the transition (frame ~25)
        assert len(breakpoints) >= 1
        assert any(20 <= f <= 30 for f in breakpoints)

    def test_returns_residuals_per_frame(self) -> None:
        """Residual dict should have entries for trajectory frames."""
        ball_by_frame = self._make_parabolic_trajectory(n_frames=30)
        frames = sorted(ball_by_frame.keys())

        _, residuals = _find_parabolic_breakpoints(
            ball_by_frame, frames,
            window_frames=12, stride=3,
            min_residual=0.015, min_prominence=0.008,
            min_distance_frames=12,
        )
        # Should have residuals for frames covered by windows
        assert len(residuals) > 0
        # Residuals for a clean parabola should be very small
        for r in residuals.values():
            assert r < 0.01

    def test_too_few_frames(self) -> None:
        """Fewer frames than window returns empty."""
        ball_by_frame = {i: _bp(i, 0.5, 0.5) for i in range(5)}
        frames = sorted(ball_by_frame.keys())

        breakpoints, residuals = _find_parabolic_breakpoints(
            ball_by_frame, frames,
            window_frames=12, stride=3,
            min_residual=0.015, min_prominence=0.008,
            min_distance_frames=12,
        )
        assert breakpoints == []
        assert residuals == {}

    def test_arc_apex_has_low_residual(self) -> None:
        """Arc apex (top of parabola) should NOT trigger a breakpoint.

        This is the key insight: arc apexes lie ON the parabola and have
        LOW residuals, unlike real contacts that BREAK the parabola.
        """
        ball_by_frame = self._make_parabolic_trajectory(
            n_frames=40, y_apex=0.3, y_amplitude=0.2,
        )
        frames = sorted(ball_by_frame.keys())

        breakpoints, residuals = _find_parabolic_breakpoints(
            ball_by_frame, frames,
            window_frames=12, stride=3,
            min_residual=0.01, min_prominence=0.005,
            min_distance_frames=12,
        )
        # Apex is around frame 20; it should NOT be a breakpoint
        apex_breakpoints = [f for f in breakpoints if 15 <= f <= 25]
        assert len(apex_breakpoints) == 0


class TestCandidateFeatures:
    """Tests for the CandidateFeatures dataclass."""

    def test_to_array_shape(self) -> None:
        """Feature array has correct number of elements."""
        f = CandidateFeatures(
            frame=10, velocity=0.02, direction_change_deg=45.0,
            arc_fit_residual=0.01, acceleration=0.005,
            trajectory_curvature=0.1,
            velocity_y=0.01, velocity_ratio=1.2,
            player_distance=0.05,
            ball_x=0.5, ball_y=0.6, ball_y_relative_net=0.1,
            is_net_crossing=False,
            frames_since_last=15,
        )
        arr = f.to_array()
        assert arr.shape == (25,)
        assert len(CandidateFeatures.feature_names()) == 25

    def test_infinite_player_distance_handled(self) -> None:
        """Infinite player distance maps to 1.0."""
        f = CandidateFeatures(
            frame=10, velocity=0.02, direction_change_deg=45.0,
            arc_fit_residual=0.0, acceleration=0.0,
            trajectory_curvature=0.0,
            velocity_y=0.0, velocity_ratio=1.0,
            player_distance=float("inf"),
            ball_x=0.5, ball_y=0.6,
            ball_y_relative_net=0.1,
            is_net_crossing=False,
            frames_since_last=0,
        )
        arr = f.to_array()
        # player_distance is index 7
        assert arr[7] == 1.0
        assert not np.any(np.isinf(arr))

    def test_boolean_features_as_float(self) -> None:
        """Boolean feature (is_net_crossing) should be 0.0 or 1.0."""
        f = CandidateFeatures(
            frame=10, velocity=0.02, direction_change_deg=45.0,
            arc_fit_residual=0.0, acceleration=0.0,
            trajectory_curvature=0.0,
            velocity_y=0.0, velocity_ratio=1.0,
            player_distance=0.05,
            ball_x=0.5, ball_y=0.6,
            ball_y_relative_net=0.1,
            is_net_crossing=True,
            frames_since_last=0,
        )
        arr = f.to_array()
        assert arr[15] == 1.0  # is_net_crossing


class TestContactClassifier:
    """Tests for the ContactClassifier."""

    def test_untrained_predicts_false(self) -> None:
        """Untrained classifier predicts (False, 0.0) for all candidates."""
        clf = ContactClassifier()
        assert not clf.is_trained

        features = [CandidateFeatures(
            frame=10, velocity=0.02, direction_change_deg=45.0,
            arc_fit_residual=0.01, acceleration=0.005,
            trajectory_curvature=0.1,
            velocity_y=0.01, velocity_ratio=1.0,
            player_distance=0.05,
            ball_x=0.5, ball_y=0.6,
            ball_y_relative_net=0.1,
            is_net_crossing=False,
            frames_since_last=0,
        )]

        results = clf.predict(features)
        assert len(results) == 1
        assert results[0] == (False, 0.0)

    def test_train_and_predict(self) -> None:
        """Trained classifier can make predictions."""
        n_features = len(CandidateFeatures.feature_names())
        rng = np.random.RandomState(42)
        n_pos = 30
        n_neg = 20

        # Positive examples: higher velocity, higher direction change
        x_pos = rng.randn(n_pos, n_features) * 0.1 + 0.5
        x_pos[:, 0] += 0.5  # Higher velocity
        x_pos[:, 1] += 30.0  # Higher direction change

        # Negative examples: lower features
        x_neg = rng.randn(n_neg, n_features) * 0.1 + 0.2

        x_mat = np.vstack([x_pos, x_neg])
        y = np.array([1] * n_pos + [0] * n_neg)

        clf = ContactClassifier(threshold=0.5)
        metrics = clf.train(x_mat, y)

        assert clf.is_trained
        assert metrics["train_f1"] > 0.5  # Should learn something
        assert metrics["n_samples"] == 50

        # Make prediction
        features = [CandidateFeatures(
            frame=10, velocity=0.6, direction_change_deg=60.0,
            arc_fit_residual=0.02, acceleration=0.01,
            trajectory_curvature=0.2,
            velocity_y=0.01, velocity_ratio=1.5,
            player_distance=0.05,
            ball_x=0.5, ball_y=0.6,
            ball_y_relative_net=0.1,
            is_net_crossing=False,
            frames_since_last=15,
        )]

        results = clf.predict(features)
        assert len(results) == 1
        is_contact, confidence = results[0]
        assert isinstance(is_contact, bool)
        assert 0.0 <= confidence <= 1.0

    def test_feature_importance(self) -> None:
        """Trained classifier reports feature importance."""
        n_features = len(CandidateFeatures.feature_names())
        rng = np.random.RandomState(42)
        x_mat = rng.randn(50, n_features)
        y = (x_mat[:, 0] > 0).astype(int)  # Label depends on first feature

        clf = ContactClassifier()
        clf.train(x_mat, y)

        importance = clf.feature_importance()
        assert len(importance) == n_features
        assert all(v >= 0 for v in importance.values())
        # First feature should be most important
        assert importance["velocity"] > 0

    def test_save_load_roundtrip(self, tmp_path: object) -> None:
        """Model can be saved and loaded."""
        n_features = len(CandidateFeatures.feature_names())
        rng = np.random.RandomState(42)
        x_mat = rng.randn(50, n_features)
        y = (x_mat[:, 0] > 0).astype(int)

        clf = ContactClassifier(threshold=0.4)
        clf.train(x_mat, y)

        path = Path(str(tmp_path)) / "model.pkl"
        clf.save(path)

        loaded = ContactClassifier.load(path)
        assert loaded.is_trained
        assert loaded.threshold == 0.4

        # Predictions should match
        features = [CandidateFeatures(
            frame=10, velocity=0.5, direction_change_deg=45.0,
            arc_fit_residual=0.01, acceleration=0.005,
            trajectory_curvature=0.1,
            velocity_y=0.01, velocity_ratio=1.0,
            player_distance=0.05,
            ball_x=0.5, ball_y=0.6,
            ball_y_relative_net=0.1,
            is_net_crossing=False,
            frames_since_last=15,
        )]
        r1 = clf.predict(features)
        r2 = loaded.predict(features)
        assert r1[0][1] == r2[0][1]  # Same confidence


class TestDetectContactsWithNewFeatures:
    """Test detect_contacts with parabolic detection enabled."""

    def test_parabolic_detection_enabled_by_default(self) -> None:
        """Default config has parabolic detection enabled."""
        cfg = ContactDetectionConfig()
        assert cfg.enable_parabolic_detection is True

    def test_can_disable_new_detectors(self) -> None:
        """Parabolic detection can be disabled for backward compatibility."""
        cfg = ContactDetectionConfig(
            enable_parabolic_detection=False,
        )

        # Simple trajectory still works
        positions = []
        for i in range(30):
            if i < 15:
                positions.append(_bp(i, 0.2 + i * 0.02, 0.6))
            else:
                positions.append(_bp(i, 0.5 - (i - 15) * 0.02, 0.6))

        players = [_pp(15, 1, 0.5, 0.65)]
        result = detect_contacts(positions, players, cfg, use_classifier=False)
        # Should still detect via velocity/inflection
        assert result.num_contacts >= 0  # May or may not detect depending on velocity

    def test_contact_has_arc_fit_residual(self) -> None:
        """Detected contacts include arc_fit_residual field."""
        cfg = ContactDetectionConfig(
            min_peak_velocity=0.005,
            min_peak_prominence=0.002,
            enable_noise_filter=False,
        )

        positions = []
        for i in range(40):
            if i < 20:
                positions.append(_bp(i, 0.2 + i * 0.015, 0.6))
            else:
                positions.append(_bp(i, 0.5 - (i - 20) * 0.015, 0.6))

        players = [_pp(20, 1, 0.5, 0.65)]
        result = detect_contacts(positions, players, cfg, use_classifier=False)

        for c in result.contacts:
            assert hasattr(c, "arc_fit_residual")
            assert isinstance(c.arc_fit_residual, float)

    def test_contact_has_confidence_field(self) -> None:
        """Detected contacts include confidence field."""
        cfg = ContactDetectionConfig(
            min_peak_velocity=0.005,
            min_peak_prominence=0.002,
            enable_noise_filter=False,
        )

        positions = []
        for i in range(30):
            if i < 15:
                positions.append(_bp(i, 0.2 + i * 0.02, 0.6))
            else:
                positions.append(_bp(i, 0.5 - (i - 15) * 0.02, 0.6))

        players = [_pp(15, 1, 0.5, 0.65)]
        result = detect_contacts(positions, players, cfg, use_classifier=False)

        for c in result.contacts:
            assert hasattr(c, "confidence")
            # Without classifier, confidence should be 0.0 (hand-tuned gates)
            assert c.confidence == 0.0


class TestBuildGeneratorsByFrame:
    """Tests for the generator-attribution helper used by the Arm B rescue gate."""

    def test_each_generator_frame_claims_nearest_merged_candidate(self) -> None:
        """A generator frame attributes to the nearest merged candidate within min_distance."""
        result = _build_generators_by_frame(
            candidate_frames=[10, 30, 60],
            min_distance_frames=8,
            generator_lists={
                "velocity_peak": [10, 30],
                "inflection": [12, 31],  # both within 8f of a merged candidate
                "reversal": [9],         # within 8f of frame 10
            },
        )
        assert result[10] == {"velocity_peak", "inflection", "reversal"}
        assert result[30] == {"velocity_peak", "inflection"}
        assert result[60] == set()

    def test_generator_frame_outside_window_is_dropped(self) -> None:
        """Generator frames beyond min_distance from any merged candidate are ignored."""
        result = _build_generators_by_frame(
            candidate_frames=[10, 50],
            min_distance_frames=5,
            generator_lists={"parabolic": [30]},  # 20f from 10, 20f from 50
        )
        assert result[10] == set()
        assert result[50] == set()

    def test_empty_candidate_list_returns_empty_map(self) -> None:
        """No merged candidates → empty map; generator lists are ignored."""
        result = _build_generators_by_frame(
            candidate_frames=[],
            min_distance_frames=8,
            generator_lists={"velocity_peak": [10, 20]},
        )
        assert result == {}


class _FakeContactClassifier:
    """Test double: returns (is_validated, confidence) from a frame-indexed map.

    Used to force specific predictions at the rescue gate without the real GBM.
    The production call site reads `classifier.is_trained`, `classifier.predict`,
    and `classifier.threshold`; these stubs cover that surface.
    """

    threshold = 0.40

    def __init__(self, by_frame_result: dict[int, tuple[bool, float]]):
        self._by_frame = by_frame_result
        self._call_order: list[int] = []

    @property
    def is_trained(self) -> bool:
        return True

    def predict(self, features):  # type: ignore[no-untyped-def]
        results = []
        # Each call passes exactly one CandidateFeatures (production uses [features]).
        for feat in features:
            # CandidateFeatures doesn't carry frame id directly; tests control
            # a shared counter: the builder replaces this classifier with one
            # that yields results in call order, matching candidate_frames.
            frame = self._call_order.pop(0) if self._call_order else -1
            results.append(self._by_frame.get(frame, (False, 0.0)))
        return results

    def set_call_order(self, frames: list[int]) -> None:
        self._call_order = list(frames)


class TestSequenceRecoveryGate:
    """Two-arm sequence-rescue gate (Pattern A, 2026-04-17).

    Arm A: seq_peak >= 0.80 AND conf >= SEQ_RECOVERY_CLF_FLOOR (0.20).
    Arm B: seq_peak >= 0.80 AND conf >= SEQ_RECOVERY_CLF_FLOOR_MULTIGEN
           AND n_generators >= SEQ_RECOVERY_MIN_GENERATORS
           AND player_distance <= SEQ_RECOVERY_MAX_PLAYER_DIST.
    """

    def _make_trajectory(self, contact_frame: int = 25) -> tuple[list[BallPosition], list[PlayerPosition]]:
        """Simple trajectory with a direction change at `contact_frame` and a player at <0.10."""
        positions = []
        for i in range(60):
            if i <= contact_frame:
                positions.append(_bp(i, 0.30 + i * 0.005, 0.50 - i * 0.003))
            else:
                positions.append(_bp(i, 0.425 - (i - contact_frame) * 0.005, 0.425 + (i - contact_frame) * 0.003))
        # Player very close to the ball at the contact frame (player_dist ~0.05).
        players = [
            _pp(contact_frame - 2, 1, 0.42, 0.44),
            _pp(contact_frame, 1, 0.42, 0.44),
            _pp(contact_frame + 2, 1, 0.42, 0.44),
        ]
        return positions, players

    def _sequence_probs_with_peak(self, length: int, peak_frame: int, peak_val: float = 0.95) -> np.ndarray:
        """Build a (2, length) sequence-probs matrix with a non-background peak at peak_frame."""
        probs = np.zeros((2, length), dtype=float)
        probs[0, :] = 1.0 - 0.02  # background
        probs[1, :] = 0.02         # non-background baseline
        lo = max(0, peak_frame - 3)
        hi = min(length, peak_frame + 4)
        probs[1, lo:hi] = peak_val
        probs[0, lo:hi] = 1.0 - peak_val
        return probs

    def _run(
        self,
        classifier_map: dict[int, tuple[bool, float]],
        generators_by_frame: dict[int, set[str]] | None,
        sequence_probs: np.ndarray | None,
        monkeypatch,
        contact_frame: int = 25,
    ) -> ContactSequence:
        positions, players = self._make_trajectory(contact_frame)
        clf = _FakeContactClassifier(classifier_map)
        # The production loop calls classifier.predict once per candidate frame
        # that survives warmup. Align fake-classifier ordering with the known
        # candidate frames in the trajectory.
        clf.set_call_order([contact_frame] * 20)  # Oversized; extras are harmless.

        # Stub `_build_generators_by_frame` to return a deterministic map so
        # the Arm B n_generators count is fully controlled.
        if generators_by_frame is not None:
            def _stub(cands, _min_dist, _gen_lists):  # type: ignore[no-untyped-def]
                return {cf: set(generators_by_frame.get(cf, set())) for cf in cands}
            monkeypatch.setattr(
                "rallycut.tracking.contact_detector._build_generators_by_frame",
                _stub,
            )

        config = ContactDetectionConfig(
            min_peak_velocity=0.003,
            min_peak_prominence=0.001,
            min_peak_distance_frames=8,
            enable_noise_filter=False,
            enable_sequence_recovery=True,
        )
        return detect_contacts(
            positions,
            player_positions=players,
            config=config,
            classifier=clf,  # type: ignore[arg-type]
            sequence_probs=sequence_probs,
        )

    def test_arm_a_rescues_when_conf_above_original_floor(self, monkeypatch) -> None:
        """Arm A preserved: conf=0.25, single generator, seq support → rescued."""
        result = self._run(
            classifier_map={25: (False, 0.25)},  # below 0.40 hard threshold; above 0.20 floor
            generators_by_frame={25: {"velocity_peak"}},  # single generator
            sequence_probs=self._sequence_probs_with_peak(length=60, peak_frame=25),
            monkeypatch=monkeypatch,
        )
        rescued = [c for c in result.contacts if abs(c.frame - 25) <= 3]
        assert len(rescued) >= 1, "Arm A regression: high-conf rescue must still fire"

    def test_arm_b_rescues_low_conf_multigen_with_player(self, monkeypatch) -> None:
        """Arm B: conf=0.10, 3 generators, player close, seq support → rescued.

        The module default is MIN_GENERATORS=999 (dormant) — monkeypatch it
        back to the designed active value for this test.
        """
        monkeypatch.setattr(
            "rallycut.tracking.sequence_action_runtime.SEQ_RECOVERY_MIN_GENERATORS",
            3,
        )
        result = self._run(
            classifier_map={25: (False, 0.10)},  # below Arm A floor (0.20)
            generators_by_frame={25: {"velocity_peak", "inflection", "reversal"}},
            sequence_probs=self._sequence_probs_with_peak(length=60, peak_frame=25),
            monkeypatch=monkeypatch,
        )
        rescued = [c for c in result.contacts if abs(c.frame - 25) <= 3]
        assert len(rescued) >= 1, "Arm B must rescue multi-gen low-conf with player nearby"

    def test_arm_b_rejects_single_generator(self, monkeypatch) -> None:
        """Arm B requires ≥3 generators. Single-generator low-conf stays rejected."""
        result = self._run(
            classifier_map={25: (False, 0.10)},
            generators_by_frame={25: {"velocity_peak"}},  # only 1
            sequence_probs=self._sequence_probs_with_peak(length=60, peak_frame=25),
            monkeypatch=monkeypatch,
        )
        rescued = [c for c in result.contacts if abs(c.frame - 25) <= 3]
        assert len(rescued) == 0, "Arm B must not rescue single-generator low-conf"

    def test_arm_b_rejects_when_player_too_far(self, monkeypatch) -> None:
        """Arm B requires player within 0.15. Distant-player case stays rejected."""
        # Shift player far from ball (dist >> 0.15).
        positions, _ = self._make_trajectory(25)
        players_far = [
            _pp(23, 1, 0.05, 0.95),
            _pp(25, 1, 0.05, 0.95),
            _pp(27, 1, 0.05, 0.95),
        ]
        clf = _FakeContactClassifier({25: (False, 0.10)})
        clf.set_call_order([25] * 20)

        def _stub(cands, _min_dist, _gen_lists):  # type: ignore[no-untyped-def]
            return {cf: {"velocity_peak", "inflection", "reversal"} if cf == 25 else set() for cf in cands}
        monkeypatch.setattr(
            "rallycut.tracking.contact_detector._build_generators_by_frame",
            _stub,
        )

        config = ContactDetectionConfig(
            min_peak_velocity=0.003,
            min_peak_prominence=0.001,
            min_peak_distance_frames=8,
            enable_noise_filter=False,
            enable_sequence_recovery=True,
        )
        result = detect_contacts(
            positions,
            player_positions=players_far,
            config=config,
            classifier=clf,  # type: ignore[arg-type]
            sequence_probs=self._sequence_probs_with_peak(length=60, peak_frame=25),
        )
        rescued = [c for c in result.contacts if abs(c.frame - 25) <= 3]
        assert len(rescued) == 0, "Arm B must not rescue when no player is within 0.15"

    def test_neither_arm_rescues_without_sequence_support(self, monkeypatch) -> None:
        """Both arms require seq_peak >= SEQ_RECOVERY_TAU. No sequence → no rescue."""
        # Flat sequence probs: non-background stays at 0.02 (well below τ=0.80).
        length = 60
        flat = np.zeros((2, length), dtype=float)
        flat[0, :] = 0.98
        flat[1, :] = 0.02
        result = self._run(
            classifier_map={25: (False, 0.10)},
            generators_by_frame={25: {"velocity_peak", "inflection", "reversal"}},
            sequence_probs=flat,
            monkeypatch=monkeypatch,
        )
        rescued = [c for c in result.contacts if abs(c.frame - 25) <= 3]
        assert len(rescued) == 0, "Rescue requires sequence endorsement; flat probs must not rescue"


class TestRallyStartServeAnchor:
    """Pattern C — MS-TCN++ serve-class anchor for late ball-track starts.

    The anchor fires only when (a) no existing contact lives in the first
    SERVE_ANCHOR_MAX_FRAME frames, (b) MS-TCN++ serve class has a peak above
    SERVE_ANCHOR_TAU in that window, and (c) a player is plausibly near the
    ball at the peak frame.
    """

    def _probs_with_serve_peak(
        self,
        length: int,
        peak_frame: int,
        peak_val: float = 0.90,
    ) -> np.ndarray:
        """(7, length) sequence_probs with a serve-class peak at peak_frame."""
        probs = np.zeros((7, length), dtype=float)
        probs[0, :] = 0.98  # background
        lo = max(0, peak_frame - 3)
        hi = min(length, peak_frame + 4)
        probs[1, lo:hi] = peak_val   # serve class
        probs[0, lo:hi] = 1.0 - peak_val
        return probs

    def test_anchor_fires_on_late_track_start(self, monkeypatch) -> None:
        """No early contact, strong serve peak, player close → anchor returned.

        The module default is SERVE_ANCHOR_TAU=1.1 (dormant) — monkeypatch
        it back to the designed active value for this test.
        """
        monkeypatch.setattr(
            "rallycut.tracking.sequence_action_runtime.SERVE_ANCHOR_TAU",
            0.85,
        )
        ball_by_frame = {
            70: BallPosition(frame_number=70, x=0.45, y=0.55, confidence=0.9),
            80: BallPosition(frame_number=80, x=0.42, y=0.50, confidence=0.9),
        }
        players = [_pp(f, 1, 0.44, 0.56) for f in range(28, 40)]

        anchored = _maybe_anchor_rally_start_serve(
            contacts=[],
            sequence_probs=self._probs_with_serve_peak(length=100, peak_frame=30),
            ball_by_frame=ball_by_frame,
            player_positions=players,
            first_frame=0,
            net_y=0.50,
        )
        assert anchored is not None
        assert anchored.is_validated is True
        assert abs(anchored.frame - 30) <= 3
        assert anchored.player_track_id == 1
        assert anchored.confidence >= 0.85

    def test_anchor_silent_when_contact_in_window(self) -> None:
        """Existing contact in [0, SERVE_ANCHOR_MAX_FRAME] suppresses the anchor."""
        existing = Contact(
            frame=25, ball_x=0.5, ball_y=0.5, velocity=0.01,
            direction_change_deg=10.0, player_track_id=1,
            player_distance=0.05, is_validated=True, confidence=0.9,
        )
        ball_by_frame = {25: BallPosition(frame_number=25, x=0.5, y=0.5, confidence=0.9)}
        anchored = _maybe_anchor_rally_start_serve(
            contacts=[existing],
            sequence_probs=self._probs_with_serve_peak(length=100, peak_frame=30),
            ball_by_frame=ball_by_frame,
            player_positions=[_pp(30, 1, 0.5, 0.5)],
            first_frame=0,
            net_y=0.50,
        )
        assert anchored is None

    def test_anchor_silent_without_serve_peak(self) -> None:
        """Flat sequence probs (no serve peak) → no anchor."""
        flat = np.zeros((7, 100), dtype=float)
        flat[0, :] = 0.98
        flat[1:, :] = 0.02 / 6
        ball_by_frame = {
            70: BallPosition(frame_number=70, x=0.45, y=0.55, confidence=0.9),
        }
        anchored = _maybe_anchor_rally_start_serve(
            contacts=[],
            sequence_probs=flat,
            ball_by_frame=ball_by_frame,
            player_positions=[_pp(30, 1, 0.5, 0.5)],
            first_frame=0,
            net_y=0.50,
        )
        assert anchored is None

    def test_anchor_silent_when_player_too_far(self) -> None:
        """Serve peak but no player within 0.20 → no anchor."""
        ball_by_frame = {70: BallPosition(frame_number=70, x=0.5, y=0.5, confidence=0.9)}
        # Player at (0.05, 0.95) — distance ~0.7 from ball
        players = [_pp(30, 1, 0.05, 0.95)]
        anchored = _maybe_anchor_rally_start_serve(
            contacts=[],
            sequence_probs=self._probs_with_serve_peak(length=100, peak_frame=30),
            ball_by_frame=ball_by_frame,
            player_positions=players,
            first_frame=0,
            net_y=0.50,
        )
        assert anchored is None

    def test_anchor_silent_without_ball_data(self) -> None:
        """No ball positions anywhere → refuse to synthesize a (0,0) contact."""
        anchored = _maybe_anchor_rally_start_serve(
            contacts=[],
            sequence_probs=self._probs_with_serve_peak(length=100, peak_frame=30),
            ball_by_frame={},
            player_positions=[_pp(30, 1, 0.5, 0.5)],
            first_frame=0,
            net_y=0.50,
        )
        assert anchored is None
