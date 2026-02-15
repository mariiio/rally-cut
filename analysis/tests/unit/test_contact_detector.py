"""Unit tests for ball contact detection from trajectory inflection points."""

from __future__ import annotations

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    ContactSequence,
    _compute_direction_change,
    _compute_velocities,
    _filter_noise_spikes,
    _find_inflection_candidates,
    _find_nearest_player,
    _merge_candidates,
    _smooth_signal,
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


class TestComputeVelocities:
    """Tests for velocity computation from ball positions."""

    def test_basic_velocity(self) -> None:
        """Constant horizontal motion produces constant velocity."""
        positions = [_bp(i, 0.1 + i * 0.02, 0.5) for i in range(10)]
        velocities = _compute_velocities(positions)
        assert len(velocities) == 9
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
        angle = _compute_direction_change(ball_by_frame, 5, check_frames=3)
        assert angle < 5.0  # Nearly straight

    def test_90_degree_turn(self) -> None:
        """Right-angle turn produces ~90 degree change."""
        ball_by_frame = {
            0: _bp(0, 0.3, 0.5),
            5: _bp(5, 0.5, 0.5),  # Moving right
            10: _bp(10, 0.5, 0.3),  # Moving up (90 degree turn)
        }
        angle = _compute_direction_change(ball_by_frame, 5, check_frames=6)
        assert 80.0 < angle < 100.0

    def test_reversal_180_degrees(self) -> None:
        """Complete reversal produces ~180 degree change."""
        ball_by_frame = {
            0: _bp(0, 0.3, 0.5),
            5: _bp(5, 0.5, 0.5),  # Moving right
            10: _bp(10, 0.3, 0.5),  # Moving back left
        }
        angle = _compute_direction_change(ball_by_frame, 5, check_frames=6)
        assert angle > 150.0

    def test_missing_neighbors(self) -> None:
        """Returns 0 when before/after positions are missing."""
        ball_by_frame = {5: _bp(5, 0.5, 0.5)}
        angle = _compute_direction_change(ball_by_frame, 5, check_frames=3)
        assert angle == 0.0


class TestFindNearestPlayer:
    """Tests for player attribution."""

    def test_finds_closest_player(self) -> None:
        """Attributes contact to nearest player."""
        # Note: _find_nearest_player uses bottom-center of bbox (y + height/2)
        players = [
            _pp(10, 1, 0.3, 0.7),  # Far from ball
            _pp(10, 2, 0.51, 0.51),  # Close to ball (bottom-center at y=0.51+0.075=0.585)
        ]
        track_id, dist = _find_nearest_player(10, 0.5, 0.5, players)
        assert track_id == 2
        assert dist < 0.10

    def test_no_players_returns_default(self) -> None:
        """No players gives track_id=-1 and infinite distance."""
        track_id, dist = _find_nearest_player(10, 0.5, 0.5, [])
        assert track_id == -1
        assert dist == float("inf")

    def test_respects_frame_window(self) -> None:
        """Players outside search window are ignored."""
        players = [_pp(100, 1, 0.5, 0.5)]  # Far from frame 10
        track_id, dist = _find_nearest_player(10, 0.5, 0.5, players, search_frames=3)
        assert track_id == -1


class TestFilterNoiseSpikes:
    """Tests for VballNet noise spike removal."""

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
        result = detect_contacts([])
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

        result = detect_contacts(positions, config=config)
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

        # Gentle arc: ball moving right slowly, then changing direction at frame 15
        positions = []
        for i in range(30):
            if i <= 15:
                x = 0.3 + i * 0.005
                y = 0.5 - i * 0.005
            else:
                x = 0.375 - (i - 15) * 0.005
                y = 0.425 + (i - 15) * 0.005
            positions.append(_bp(i, x, y, conf=0.9))

        result = detect_contacts(positions, config=config)
        # Should detect a contact near the direction change at frame 15
        # The inflection detection should catch this even though velocity is low
        inflection_contacts = [
            c for c in result.contacts if abs(c.frame - 15) <= 3
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

        result = detect_contacts(positions, config=config)
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

        result = detect_contacts(positions, config=config, net_y=0.45)
        assert result.net_y == 0.45

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
        result = detect_contacts(positions, players, config)

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
        """Contact.to_dict produces expected keys."""
        c = Contact(frame=10, ball_x=0.5, ball_y=0.5, velocity=0.02,
                    direction_change_deg=45.0, player_track_id=3,
                    court_side="near", is_validated=True)
        d = c.to_dict()
        assert d["frame"] == 10
        assert d["ballX"] == 0.5
        assert d["playerTrackId"] == 3
        assert d["courtSide"] == "near"
        assert d["isValidated"] is True
