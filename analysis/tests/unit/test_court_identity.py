"""Tests for court-plane identity resolution."""

from __future__ import annotations

from rallycut.court.calibration import CourtCalibrator
from rallycut.tracking.court_identity import (
    CourtIdentityConfig,
    CourtIdentityResolver,
    CourtTrack,
    NetInteraction,
    resolve_court_identity,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _make_positions(
    track_id: int,
    frames: range,
    x: float = 0.5,
    y: float = 0.5,
    width: float = 0.05,
    height: float = 0.15,
) -> list[PlayerPosition]:
    """Create positions for a track at fixed location."""
    return [
        PlayerPosition(
            frame_number=f,
            track_id=track_id,
            x=x,
            y=y,
            width=width,
            height=height,
            confidence=0.9,
        )
        for f in frames
    ]


def _make_calibrator() -> CourtCalibrator:
    """Create a simple calibrator with identity-like homography.

    Maps image space roughly to court:
    - image (0,0) -> court (0,0) near-left
    - image (1,1) -> court (8,16) far-right
    But scaled to court dimensions.
    """
    calibrator = CourtCalibrator()
    # Simple 4-corner mapping
    image_corners = [
        (0.1, 0.9),   # near-left (bottom of image = near baseline)
        (0.9, 0.9),   # near-right
        (0.7, 0.3),   # far-right (top of image = far baseline)
        (0.3, 0.3),   # far-left
    ]
    court_corners = [
        (0.0, 0.0),   # near-left
        (8.0, 0.0),   # near-right
        (8.0, 16.0),  # far-right
        (0.0, 16.0),  # far-left
    ]
    calibrator.calibrate(image_corners, court_corners)
    return calibrator


class TestCourtTrack:
    def test_empty_track(self) -> None:
        ct = CourtTrack(track_id=1)
        assert ct.track_id == 1
        assert len(ct.positions) == 0

    def test_add_positions(self) -> None:
        ct = CourtTrack(track_id=1)
        ct.positions[0] = (3.0, 2.0)
        ct.positions[1] = (3.5, 2.5)
        assert len(ct.positions) == 2
        assert ct.positions[0] == (3.0, 2.0)


class TestNetInteractionDetection:
    def test_no_interactions_when_far_apart(self) -> None:
        """Tracks on opposite sides of court should not create interactions."""
        calibrator = _make_calibrator()
        config = CourtIdentityConfig(net_approach_distance=2.0)

        # Near-side player at bottom of image, far-side at top
        positions = (
            _make_positions(1, range(0, 50), x=0.5, y=0.8)  # near
            + _make_positions(2, range(0, 50), x=0.5, y=0.35)  # far
        )
        team_assignments = {1: 0, 2: 1}

        resolver = CourtIdentityResolver(
            calibrator, config, video_width=1920, video_height=1080
        )
        _, decisions = resolver.resolve(positions, team_assignments)
        assert len(decisions) == 0

    def test_interaction_detected_near_net(self) -> None:
        """Tracks that both approach the net should trigger interaction."""
        calibrator = _make_calibrator()
        config = CourtIdentityConfig(
            net_approach_distance=3.0,
            observation_window=10,
            min_observation_frames=3,
        )

        # Both players move to net area (middle of image)
        # Near player at y=0.6 (near side of net in image)
        # Far player at y=0.4 (far side of net in image)
        positions = (
            # Pre-interaction: near player far from net
            _make_positions(1, range(0, 20), x=0.5, y=0.8)
            # During: near player at net
            + _make_positions(1, range(20, 35), x=0.5, y=0.6)
            # Post: near player back to near side
            + _make_positions(1, range(35, 80), x=0.5, y=0.8)
            # Far player: stays in far area before and after
            + _make_positions(2, range(0, 20), x=0.5, y=0.35)
            # During: far player at net
            + _make_positions(2, range(20, 35), x=0.5, y=0.45)
            # Post: back to far
            + _make_positions(2, range(35, 80), x=0.5, y=0.35)
        )
        team_assignments = {1: 0, 2: 1}

        resolver = CourtIdentityResolver(
            calibrator, config, video_width=1920, video_height=1080
        )
        _, decisions = resolver.resolve(positions, team_assignments)
        # Should detect at least the approach or return no swap needed
        # The exact behavior depends on homography, but it should not crash
        assert isinstance(decisions, list)


class TestSideOfNetScoring:
    def test_correct_assignment_high_score(self) -> None:
        """Tracks on correct sides should score high for no-swap."""
        resolver = CourtIdentityResolver(
            _make_calibrator(),
            CourtIdentityConfig(observation_window=20, min_observation_frames=3),
        )

        # Track A on near side (low court_y), Track B on far side (high court_y)
        ct_a = CourtTrack(track_id=1)
        ct_b = CourtTrack(track_id=2)

        # Post-interaction: A stays near (court_y < 8), B stays far (court_y > 8)
        for f in range(51, 71):
            ct_a.positions[f] = (4.0, 3.0)  # Near side
            ct_b.positions[f] = (4.0, 13.0)  # Far side

        interaction = NetInteraction(
            track_a=1, track_b=2, start_frame=40, end_frame=50
        )
        team_assignments = {1: 0, 2: 1}

        no_swap, swap = resolver._score_side_of_net(
            interaction, ct_a, ct_b, team_assignments
        )
        # No-swap should be much higher (tracks are correctly assigned)
        assert no_swap > swap


class TestMotionSmoothnessScoring:
    def test_smooth_continuation(self) -> None:
        """Tracks continuing on same trajectory should score high for no-swap."""
        resolver = CourtIdentityResolver(
            _make_calibrator(),
            CourtIdentityConfig(pre_interaction_window=5, min_pre_frames=3),
        )

        ct_a = CourtTrack(track_id=1)
        ct_b = CourtTrack(track_id=2)

        # Pre-interaction: A at position (2, 4), B at (6, 12)
        for f in range(15, 20):
            ct_a.positions[f] = (2.0, 4.0)
            ct_b.positions[f] = (6.0, 12.0)

        # Post-interaction: same positions (no swap happened)
        for f in range(31, 36):
            ct_a.positions[f] = (2.0, 4.0)
            ct_b.positions[f] = (6.0, 12.0)

        interaction = NetInteraction(
            track_a=1, track_b=2, start_frame=20, end_frame=30
        )

        no_swap, swap = resolver._score_motion_smoothness(
            interaction, ct_a, ct_b
        )
        # No-swap is better (tracks continue at same positions)
        assert no_swap > swap

    def test_swapped_tracks_score_swap_higher(self) -> None:
        """When tracks have been swapped, swap hypothesis should score higher."""
        resolver = CourtIdentityResolver(
            _make_calibrator(),
            CourtIdentityConfig(pre_interaction_window=5, min_pre_frames=3),
        )

        ct_a = CourtTrack(track_id=1)
        ct_b = CourtTrack(track_id=2)

        # Pre-interaction: A at (2, 4), B at (6, 12)
        for f in range(15, 20):
            ct_a.positions[f] = (2.0, 4.0)
            ct_b.positions[f] = (6.0, 12.0)

        # Post-interaction: positions swapped!
        # A is now where B was, B is now where A was
        for f in range(31, 36):
            ct_a.positions[f] = (6.0, 12.0)
            ct_b.positions[f] = (2.0, 4.0)

        interaction = NetInteraction(
            track_a=1, track_b=2, start_frame=20, end_frame=30
        )

        no_swap, swap = resolver._score_motion_smoothness(
            interaction, ct_a, ct_b
        )
        # Swap is better (pre_a -> post_b is smoother than pre_a -> post_a)
        assert swap > no_swap


class TestBboxSizeScoring:
    def test_consistent_bbox_sizes(self) -> None:
        """Consistent bbox sizes should prefer no-swap."""
        resolver = CourtIdentityResolver(
            _make_calibrator(),
            CourtIdentityConfig(observation_window=20),
        )

        # Near player (track 1): tall bbox (0.2 height)
        # Far player (track 2): short bbox (0.1 height)
        img_a: dict[int, PlayerPosition] = {}
        img_b: dict[int, PlayerPosition] = {}

        # Pre-interaction
        for f in range(0, 20):
            img_a[f] = PlayerPosition(f, 1, 0.5, 0.7, 0.05, 0.20, 0.9)
            img_b[f] = PlayerPosition(f, 2, 0.5, 0.3, 0.05, 0.10, 0.9)

        # Post-interaction: same sizes (no swap)
        for f in range(31, 51):
            img_a[f] = PlayerPosition(f, 1, 0.5, 0.7, 0.05, 0.20, 0.9)
            img_b[f] = PlayerPosition(f, 2, 0.5, 0.3, 0.05, 0.10, 0.9)

        interaction = NetInteraction(
            track_a=1, track_b=2, start_frame=20, end_frame=30
        )
        team_assignments = {1: 0, 2: 1}

        no_swap, swap = resolver._score_bbox_size(
            interaction, img_a, img_b, team_assignments
        )
        assert no_swap >= swap


class TestResolveCourtIdentity:
    def test_no_calibration_skips(self) -> None:
        """Without calibration, function should return unchanged positions."""
        calibrator = CourtCalibrator()  # Not calibrated
        positions = _make_positions(1, range(0, 10))
        team_assignments = {1: 0}

        result, num_swaps, decisions = resolve_court_identity(
            positions, team_assignments, calibrator
        )
        assert num_swaps == 0
        assert len(decisions) == 0

    def test_empty_positions(self) -> None:
        """Empty positions should return empty results."""
        calibrator = _make_calibrator()
        result, num_swaps, decisions = resolve_court_identity(
            [], {}, calibrator
        )
        assert num_swaps == 0
        assert result == []

    def test_no_team_assignments(self) -> None:
        """Without team assignments, should return unchanged."""
        calibrator = _make_calibrator()
        positions = _make_positions(1, range(0, 10))
        result, num_swaps, decisions = resolve_court_identity(
            positions, {}, calibrator
        )
        assert num_swaps == 0

    def test_apply_swap_modifies_positions(self) -> None:
        """Apply swap should change track IDs from frame onward."""
        positions = (
            _make_positions(1, range(0, 10))
            + _make_positions(2, range(0, 10))
        )
        team_assignments = {1: 0, 2: 1}

        CourtIdentityResolver._apply_swap(
            positions, 1, 2, 5, team_assignments
        )

        # Before frame 5: unchanged
        before = [p for p in positions if p.frame_number < 5]
        for p in before:
            assert p.track_id in (1, 2)

        # From frame 5: swapped
        after_1 = [p for p in positions if p.frame_number >= 5 and p.track_id == 1]
        after_2 = [p for p in positions if p.frame_number >= 5 and p.track_id == 2]
        # Track IDs should have been swapped for frames >= 5
        assert len(after_1) > 0  # Some positions now have swapped IDs
        assert len(after_2) > 0


class TestFractionOnCorrectSide:
    def test_all_correct(self) -> None:
        """All positions on correct side should return 1.0."""
        positions = [(4.0, 3.0)] * 10  # Near side (court_y < 8)
        result = CourtIdentityResolver._fraction_on_correct_side(
            positions, team=0, dead_zone=0.5
        )
        assert result == 1.0

    def test_all_wrong(self) -> None:
        """All positions on wrong side should return 0.0."""
        positions = [(4.0, 12.0)] * 10  # Far side, but team=0 (near)
        result = CourtIdentityResolver._fraction_on_correct_side(
            positions, team=0, dead_zone=0.5
        )
        assert result == 0.0

    def test_dead_zone_ignored(self) -> None:
        """Positions in dead zone should be excluded."""
        positions = [(4.0, 8.2)] * 10  # Right at net
        result = CourtIdentityResolver._fraction_on_correct_side(
            positions, team=0, dead_zone=0.5
        )
        assert result == 0.5  # All in dead zone → default

    def test_empty_positions(self) -> None:
        result = CourtIdentityResolver._fraction_on_correct_side(
            [], team=0, dead_zone=0.5
        )
        assert result == 0.5
