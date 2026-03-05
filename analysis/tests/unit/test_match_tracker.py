"""Unit tests for cross-rally player matching."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from rallycut.tracking.match_tracker import (
    MatchPlayerTracker,
    RallyTrackingResult,
    _compute_track_positions,
    _dist,
)
from rallycut.tracking.player_features import (
    HS_BINS,
    HS_RANGES,
    PlayerAppearanceFeatures,
    TrackAppearanceStats,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _make_positions(
    track_ids: list[int],
    y_values: list[float],
    num_frames: int = 30,
) -> list[PlayerPosition]:
    """Create player positions for given tracks at given Y positions."""
    positions = []
    for frame in range(num_frames):
        for tid, y in zip(track_ids, y_values):
            positions.append(
                PlayerPosition(
                    frame_number=frame,
                    track_id=tid,
                    x=0.5,
                    y=y,
                    width=0.05,
                    height=0.15,
                    confidence=0.9,
                )
            )
    return positions


def _make_histogram(dominant_hue: float, dominant_sat: float) -> np.ndarray:
    """Create a synthetic HS histogram with a peak at the given H/S values.

    Args:
        dominant_hue: Hue value (0-180, OpenCV scale).
        dominant_sat: Saturation value (0-255).

    Returns:
        L1-normalized float32 histogram matching HS_BINS shape.
    """
    # Create a small synthetic HSV image (20x20 = 400 pixels, above MIN_HIST_PIXELS)
    h = np.full((20, 20), dominant_hue, dtype=np.uint8)
    s = np.full((20, 20), int(dominant_sat), dtype=np.uint8)
    v = np.full((20, 20), 180, dtype=np.uint8)
    hsv = np.stack([h, s, v], axis=-1)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HS_BINS), HS_RANGES)
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def _make_stats(
    track_id: int,
    skin_hsv: tuple[float, float, float] = (20.0, 150.0, 180.0),
    height: float = 0.15,
    num_features: int = 5,
    upper_hue: float = 20.0,
    upper_sat: float = 100.0,
    lower_hue: float = 100.0,
    lower_sat: float = 200.0,
) -> TrackAppearanceStats:
    """Create TrackAppearanceStats with known feature values."""
    upper_hist = _make_histogram(upper_hue, upper_sat)
    lower_hist = _make_histogram(lower_hue, lower_sat)
    stats = TrackAppearanceStats(track_id=track_id)
    for i in range(num_features):
        f = PlayerAppearanceFeatures(
            track_id=track_id,
            frame_number=i,
            skin_tone_hsv=skin_hsv,
            skin_pixel_count=100,
            upper_body_hist=upper_hist.copy(),
            lower_body_hist=lower_hist.copy(),
            bbox_height=height,
            bbox_aspect_ratio=0.33,
        )
        stats.features.append(f)
    stats.compute_averages()
    return stats


class TestHungarianAssignment:
    """Test global Hungarian assignment with side penalty."""

    def test_first_rally_deterministic_assignment(self) -> None:
        """First rally assigns deterministically by Y within each team."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 11, 20, 21], [0.7, 0.8, 0.3, 0.4])
        stats = {
            10: _make_stats(10),
            11: _make_stats(11),
            20: _make_stats(20),
            21: _make_stats(21),
        }

        result = tracker.process_rally(
            track_stats=stats,
            player_positions=positions,
            court_split_y=0.5,
        )

        assert result.rally_index == 0
        assert result.assignment_confidence == 0.5  # First rally = 0.5
        # Near tracks should map to players 1-2, far to players 3-4
        near_players = {result.track_to_player[10], result.track_to_player[11]}
        far_players = {result.track_to_player[20], result.track_to_player[21]}
        assert near_players == {1, 2}
        assert far_players == {3, 4}
        # Deterministic: sorted by Y within each side
        # Track 10 (y=0.7) < Track 11 (y=0.8) → P1, P2
        assert result.track_to_player[10] == 1
        assert result.track_to_player[11] == 2

    def test_hungarian_matches_similar_appearances(self) -> None:
        """Hungarian should match tracks to the most similar profiles."""
        tracker = MatchPlayerTracker()

        # Rally 1: establish profiles with distinct clothing colors + skin + height
        stats1 = {
            10: _make_stats(10, skin_hsv=(15.0, 180.0, 120.0), height=0.18,
                            lower_hue=0.0, lower_sat=200.0, upper_hue=110.0, upper_sat=180.0),
            11: _make_stats(11, skin_hsv=(25.0, 100.0, 200.0), height=0.12,
                            lower_hue=60.0, lower_sat=200.0, upper_hue=30.0, upper_sat=180.0),
            20: _make_stats(20, skin_hsv=(10.0, 160.0, 150.0), height=0.10,
                            lower_hue=15.0, lower_sat=220.0, upper_hue=0.0, upper_sat=30.0),
            21: _make_stats(21, skin_hsv=(30.0, 120.0, 170.0), height=0.09,
                            lower_hue=140.0, lower_sat=150.0, upper_hue=0.0, upper_sat=50.0),
        }
        positions1 = _make_positions([10, 11, 20, 21], [0.7, 0.8, 0.3, 0.4])
        result1 = tracker.process_rally(
            track_stats=stats1, player_positions=positions1, court_split_y=0.5
        )

        # Record which player got which track in rally 1
        p_for_t10 = result1.track_to_player[10]
        p_for_t11 = result1.track_to_player[11]
        p_for_t20 = result1.track_to_player[20]
        p_for_t21 = result1.track_to_player[21]

        # Rally 2: new track IDs but SAME appearances
        stats2 = {
            30: _make_stats(30, skin_hsv=(15.0, 180.0, 120.0), height=0.18,
                            lower_hue=0.0, lower_sat=200.0, upper_hue=110.0, upper_sat=180.0),
            31: _make_stats(31, skin_hsv=(25.0, 100.0, 200.0), height=0.12,
                            lower_hue=60.0, lower_sat=200.0, upper_hue=30.0, upper_sat=180.0),
            40: _make_stats(40, skin_hsv=(10.0, 160.0, 150.0), height=0.10,
                            lower_hue=15.0, lower_sat=220.0, upper_hue=0.0, upper_sat=30.0),
            41: _make_stats(41, skin_hsv=(30.0, 120.0, 170.0), height=0.09,
                            lower_hue=140.0, lower_sat=150.0, upper_hue=0.0, upper_sat=50.0),
        }
        positions2 = _make_positions([30, 31, 40, 41], [0.7, 0.8, 0.3, 0.4])
        result2 = tracker.process_rally(
            track_stats=stats2, player_positions=positions2, court_split_y=0.5
        )

        # Same appearance → same player ID
        assert result2.track_to_player[30] == p_for_t10
        assert result2.track_to_player[31] == p_for_t11
        assert result2.track_to_player[40] == p_for_t20
        assert result2.track_to_player[41] == p_for_t21

    def test_fewer_tracks_than_players(self) -> None:
        """1 track per side should still assign correctly."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 20], [0.7, 0.3])
        stats = {
            10: _make_stats(10),
            20: _make_stats(20),
        }

        result = tracker.process_rally(
            track_stats=stats, player_positions=positions, court_split_y=0.5
        )

        assert len(result.track_to_player) == 2
        assert result.track_to_player[10] in {1, 2}
        assert result.track_to_player[20] in {3, 4}

    def test_more_tracks_than_four(self) -> None:
        """More than 4 tracks should take top 4 by frame count globally."""
        tracker = MatchPlayerTracker()
        positions = _make_positions(
            [10, 11, 12, 20, 21], [0.7, 0.75, 0.8, 0.3, 0.35]
        )
        stats = {
            10: _make_stats(10, num_features=10),
            11: _make_stats(11, num_features=8),
            12: _make_stats(12, num_features=2),
            20: _make_stats(20, num_features=10),
            21: _make_stats(21, num_features=5),
        }

        result = tracker.process_rally(
            track_stats=stats, player_positions=positions, court_split_y=0.5
        )

        # Should assign 4 tracks (top 4 globally by features)
        assert len(result.track_to_player) == 4
        # Track 12 should be dropped (fewest features globally)
        assert 12 not in result.track_to_player

    def test_empty_tracks(self) -> None:
        """Empty track lists should produce empty assignments."""
        tracker = MatchPlayerTracker()
        result = tracker.process_rally(
            track_stats={}, player_positions=[], court_split_y=0.5
        )

        assert result.track_to_player == {}
        assert result.assignment_confidence == 0.0

    def test_compressed_y_range_consistent_assignment(self) -> None:
        """Tracks at similar Y positions (mochi scenario) get consistent IDs.

        Simulates the mochi video where all players are at y=0.49-0.63
        with court_split_y=0.637. The global Hungarian assignment should
        give the same physical player the same ID across rallies.
        """
        # Distinct appearances for each of 4 "physical players"
        player_appearances: list[dict[str, Any]] = [
            # Player A: dark skin, tall, red shorts
            dict(skin_hsv=(15.0, 180.0, 120.0), height=0.18,
                 lower_hue=0.0, lower_sat=200.0, upper_hue=110.0, upper_sat=180.0),
            # Player B: light skin, short, green shorts
            dict(skin_hsv=(25.0, 100.0, 200.0), height=0.12,
                 lower_hue=60.0, lower_sat=200.0, upper_hue=30.0, upper_sat=180.0),
            # Player C: reddish skin, orange shorts
            dict(skin_hsv=(10.0, 160.0, 150.0), height=0.10,
                 lower_hue=15.0, lower_sat=220.0, upper_hue=0.0, upper_sat=30.0),
            # Player D: yellowish skin, purple shorts
            dict(skin_hsv=(30.0, 120.0, 170.0), height=0.09,
                 lower_hue=140.0, lower_sat=150.0, upper_hue=0.0, upper_sat=50.0),
        ]

        tracker = MatchPlayerTracker()
        # All Y values compressed: 0.49-0.63, all below court_split_y=0.637
        # Median split should still separate them
        rally_assignments: list[dict[int, int]] = []

        for rally_idx in range(5):
            # Each rally gets different track IDs but same physical players
            base_tid = (rally_idx + 1) * 100
            # Vary Y positions slightly between rallies (simulating movement)
            y_offsets = [0.005 * (rally_idx % 3), -0.003 * (rally_idx % 2),
                         0.002 * ((rally_idx + 1) % 3), -0.004 * (rally_idx % 2)]
            y_values = [0.60 + y_offsets[0], 0.63 + y_offsets[1],
                        0.49 + y_offsets[2], 0.52 + y_offsets[3]]

            stats = {}
            for j, app in enumerate(player_appearances):
                tid = base_tid + j
                stats[tid] = _make_stats(tid, **app)

            tids = [base_tid + j for j in range(4)]
            positions = _make_positions(tids, y_values)

            result = tracker.process_rally(
                track_stats=stats,
                player_positions=positions,
                court_split_y=0.637,  # Above all tracks → all on one side
            )

            # Map physical player index to assigned player ID
            assignment = {}
            for j in range(4):
                tid = base_tid + j
                assignment[j] = result.track_to_player[tid]
            rally_assignments.append(assignment)

        # After first rally, same physical player should get same ID
        for phys_idx in range(4):
            ids = [a[phys_idx] for a in rally_assignments[1:]]
            assert len(set(ids)) == 1, (
                f"Physical player {phys_idx} got inconsistent IDs: {ids}"
            )

        # All 4 player IDs should be assigned (1-4)
        all_ids = set(rally_assignments[1].values())
        assert all_ids == {1, 2, 3, 4}


class TestSideSwitchDetection:
    """Test side switch detection from global assignment."""

    def test_no_switch_on_first_rallies(self) -> None:
        """Side switch detection is disabled — always returns False."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 11, 20, 21], [0.7, 0.8, 0.3, 0.4])
        stats = {
            10: _make_stats(10, skin_hsv=(15.0, 180.0, 120.0),
                            lower_hue=0.0, lower_sat=200.0),
            11: _make_stats(11, skin_hsv=(25.0, 100.0, 200.0),
                            lower_hue=60.0, lower_sat=200.0),
            20: _make_stats(20, skin_hsv=(10.0, 160.0, 150.0),
                            lower_hue=120.0, lower_sat=200.0),
            21: _make_stats(21, skin_hsv=(30.0, 120.0, 170.0),
                            lower_hue=140.0, lower_sat=200.0),
        }

        # Rally 1: no switch possible
        r1 = tracker.process_rally(
            track_stats=stats, player_positions=positions, court_split_y=0.5
        )
        assert not r1.side_switch_detected

        # Rally 2: also always False (detection disabled)
        stats2 = {
            30: _make_stats(30, skin_hsv=(15.0, 180.0, 120.0),
                            lower_hue=0.0, lower_sat=200.0),
            31: _make_stats(31, skin_hsv=(25.0, 100.0, 200.0),
                            lower_hue=60.0, lower_sat=200.0),
            40: _make_stats(40, skin_hsv=(10.0, 160.0, 150.0),
                            lower_hue=120.0, lower_sat=200.0),
            41: _make_stats(41, skin_hsv=(30.0, 120.0, 170.0),
                            lower_hue=140.0, lower_sat=200.0),
        }
        positions2 = _make_positions([30, 31, 40, 41], [0.7, 0.8, 0.3, 0.4])
        r2 = tracker.process_rally(
            track_stats=stats2, player_positions=positions2, court_split_y=0.5
        )
        assert not r2.side_switch_detected

    def test_switch_detection_disabled(self) -> None:
        """Side switch detection is disabled (all approaches produced 0 TP).

        Appearance-based detection doesn't work for beach volleyball because
        both teams wear similar clothing and court-side effects dominate.
        """
        tracker = MatchPlayerTracker()

        near_skin = (15.0, 180.0, 100.0)
        far_skin = (30.0, 80.0, 220.0)

        # Build profiles over 3 rallies
        for i in range(3):
            tids_near = [100 + i * 10, 101 + i * 10]
            tids_far = [200 + i * 10, 201 + i * 10]
            stats = {
                tids_near[0]: _make_stats(tids_near[0], skin_hsv=near_skin, height=0.18),
                tids_near[1]: _make_stats(tids_near[1], skin_hsv=near_skin, height=0.17),
                tids_far[0]: _make_stats(tids_far[0], skin_hsv=far_skin, height=0.10),
                tids_far[1]: _make_stats(tids_far[1], skin_hsv=far_skin, height=0.09),
            }
            positions = _make_positions(
                tids_near + tids_far, [0.7, 0.75, 0.3, 0.35]
            )
            tracker.process_rally(
                track_stats=stats, player_positions=positions, court_split_y=0.5
            )

        # Rally 4: even with swapped appearances, detection returns False
        stats_swapped = {
            500: _make_stats(500, skin_hsv=far_skin, height=0.10),
            501: _make_stats(501, skin_hsv=far_skin, height=0.09),
            600: _make_stats(600, skin_hsv=near_skin, height=0.18),
            601: _make_stats(601, skin_hsv=near_skin, height=0.17),
        }
        positions_swapped = _make_positions([500, 501, 600, 601], [0.7, 0.75, 0.3, 0.35])
        result = tracker.process_rally(
            track_stats=stats_swapped,
            player_positions=positions_swapped,
            court_split_y=0.5,
        )

        assert not result.side_switch_detected

    def test_no_switch_when_consistent(self) -> None:
        """No switch when appearances stay on the same side."""
        tracker = MatchPlayerTracker()

        skin_a = (15.0, 180.0, 100.0)
        skin_b = (30.0, 80.0, 220.0)

        # 4 rallies, all consistent
        for i in range(4):
            tids_near = [100 + i * 10, 101 + i * 10]
            tids_far = [200 + i * 10, 201 + i * 10]
            stats = {
                tids_near[0]: _make_stats(tids_near[0], skin_hsv=skin_a, height=0.18,
                                          lower_hue=0.0, lower_sat=220.0),
                tids_near[1]: _make_stats(tids_near[1], skin_hsv=skin_a, height=0.17,
                                          lower_hue=0.0, lower_sat=220.0),
                tids_far[0]: _make_stats(tids_far[0], skin_hsv=skin_b, height=0.10,
                                         lower_hue=120.0, lower_sat=200.0),
                tids_far[1]: _make_stats(tids_far[1], skin_hsv=skin_b, height=0.09,
                                         lower_hue=120.0, lower_sat=200.0),
            }
            positions = _make_positions(
                tids_near + tids_far, [0.7, 0.75, 0.3, 0.35]
            )
            result = tracker.process_rally(
                track_stats=stats, player_positions=positions, court_split_y=0.5
            )

        # None of the later rallies should detect a switch
        assert not result.side_switch_detected


class TestClassifyTrackSides:
    """Test _classify_track_sides soft labeling."""

    def test_good_split(self) -> None:
        """When court_split_y separates tracks, use it."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 11, 20, 21], [0.7, 0.8, 0.3, 0.4])
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21]}

        track_avg_y, track_sides = tracker._classify_track_sides(
            stats, positions, court_split_y=0.5
        )

        assert track_sides[10] == 0  # near (y=0.7 > 0.5)
        assert track_sides[11] == 0  # near (y=0.8 > 0.5)
        assert track_sides[20] == 1  # far (y=0.3 <= 0.5)
        assert track_sides[21] == 1  # far (y=0.4 <= 0.5)

    def test_all_one_side_fallback(self) -> None:
        """When all tracks are on one side, use median split."""
        tracker = MatchPlayerTracker()
        # All tracks below court_split_y=0.637 (mochi scenario)
        positions = _make_positions([10, 11, 20, 21], [0.60, 0.63, 0.49, 0.52])
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21]}

        track_avg_y, track_sides = tracker._classify_track_sides(
            stats, positions, court_split_y=0.637
        )

        # Should split at median: 2 near, 2 far
        near_count = sum(1 for s in track_sides.values() if s == 0)
        far_count = sum(1 for s in track_sides.values() if s == 1)
        assert near_count == 2
        assert far_count == 2

    def test_no_court_split_fallback(self) -> None:
        """Without court_split_y, use median split."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 11, 20, 21], [0.7, 0.8, 0.3, 0.4])
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21]}

        track_avg_y, track_sides = tracker._classify_track_sides(
            stats, positions, court_split_y=None
        )

        near_count = sum(1 for s in track_sides.values() if s == 0)
        far_count = sum(1 for s in track_sides.values() if s == 1)
        assert near_count == 2
        assert far_count == 2

    def test_team_assignments_override_court_split(self) -> None:
        """team_assignments should take priority over court_split_y."""
        tracker = MatchPlayerTracker()
        # All tracks compressed below court_split_y (mochi scenario)
        positions = _make_positions([10, 11, 20, 21], [0.60, 0.63, 0.49, 0.52])
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21]}

        # Without team_assignments, court_split_y=0.637 fails → median split
        _, sides_no_ta = tracker._classify_track_sides(
            stats, positions, court_split_y=0.637
        )

        # With team_assignments, explicit mapping takes priority
        team_assignments = {10: 0, 11: 0, 20: 1, 21: 1}
        _, sides_with_ta = tracker._classify_track_sides(
            stats, positions, court_split_y=0.637,
            team_assignments=team_assignments,
        )

        assert sides_with_ta[10] == 0  # near
        assert sides_with_ta[11] == 0  # near
        assert sides_with_ta[20] == 1  # far
        assert sides_with_ta[21] == 1  # far

    def test_team_assignments_partial_coverage_fallback(self) -> None:
        """When team_assignments covers <75% of tracks, fall through."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 11, 20, 21], [0.7, 0.8, 0.3, 0.4])
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21]}

        # Only 2/4 tracks covered = 50% < 75% threshold
        team_assignments = {10: 0, 11: 0}
        _, sides = tracker._classify_track_sides(
            stats, positions, court_split_y=0.5,
            team_assignments=team_assignments,
        )

        # Should fall through to court_split_y (which works here)
        assert sides[10] == 0  # near (y=0.7 > 0.5)
        assert sides[20] == 1  # far (y=0.3 <= 0.5)

    def test_team_assignments_with_uncovered_track(self) -> None:
        """Uncovered tracks get fallback assignment when team_assignments used."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 11, 20, 21, 30], [0.7, 0.8, 0.3, 0.4, 0.5])
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21, 30]}

        # 4/5 tracks covered = 80% >= 75% threshold
        team_assignments = {10: 0, 11: 0, 20: 1, 21: 1}
        _, sides = tracker._classify_track_sides(
            stats, positions, court_split_y=0.5,
            team_assignments=team_assignments,
        )

        assert sides[10] == 0
        assert sides[11] == 0
        assert sides[20] == 1
        assert sides[21] == 1
        # Track 30 (y=0.5) not in team_assignments, falls back to court_split_y
        assert sides[30] == 1  # y=0.5 <= court_split_y=0.5 → far

    def test_team_assignments_fixes_compressed_y(self) -> None:
        """team_assignments correctly classifies compressed-Y video tracks.

        Simulates mochi: court_split_y=0.637 but all players at y=0.49-0.63.
        With team_assignments from bbox-size clustering, teams are correct
        even though Y positions overlap.
        """
        tracker = MatchPlayerTracker()
        # Near players have larger bbox (lower in frame), far players smaller
        # But Y values are compressed — court_split_y puts all on one side
        positions = _make_positions([1, 2, 3, 4], [0.60, 0.63, 0.49, 0.52])
        stats = {t: _make_stats(t) for t in [1, 2, 3, 4]}

        # Bbox-size clustering correctly identified teams despite Y compression
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}
        result = tracker.process_rally(
            track_stats=stats,
            player_positions=positions,
            court_split_y=0.637,
            team_assignments=team_assignments,
        )

        # Near team tracks (1,2) should map to players 1-2,
        # far team tracks (3,4) should map to players 3-4
        near_pids = {result.track_to_player[1], result.track_to_player[2]}
        far_pids = {result.track_to_player[3], result.track_to_player[4]}
        assert near_pids == {1, 2}
        assert far_pids == {3, 4}


def _make_positions_xy(
    track_ids: list[int],
    xy_values: list[tuple[float, float]],
    num_frames: int = 60,
) -> list[PlayerPosition]:
    """Create player positions with specified (x, y) per track."""
    positions = []
    for frame in range(num_frames):
        for tid, (x, y) in zip(track_ids, xy_values):
            positions.append(
                PlayerPosition(
                    frame_number=frame,
                    track_id=tid,
                    x=x,
                    y=y,
                    width=0.05,
                    height=0.15,
                    confidence=0.9,
                )
            )
    return positions


class TestWithinTeamRefinement:
    """Test position-continuity-based within-team assignment refinement."""

    def _make_team_stats(
        self, track_ids: list[int]
    ) -> dict[int, TrackAppearanceStats]:
        """Make stats where all tracks have identical appearance (team-level only)."""
        return {
            tid: _make_stats(
                tid,
                skin_hsv=(20.0, 150.0, 180.0),
                height=0.15,
                lower_hue=100.0,
                lower_sat=200.0,
                upper_hue=50.0,
                upper_sat=150.0,
            )
            for tid in track_ids
        }

    def test_position_continuity_swaps_within_team(self) -> None:
        """Rally 2 has same players at consistent positions but global
        assignment gets within-team order wrong → refinement swaps to correct."""
        tracker = MatchPlayerTracker()

        # Rally 1: establish positions. Near team at distinct x positions.
        # Track 10 at x=0.3, Track 11 at x=0.7 (near side)
        # Track 20 at x=0.3, Track 21 at x=0.7 (far side)
        tids1 = [10, 11, 20, 21]
        positions1 = _make_positions_xy(
            tids1, [(0.3, 0.7), (0.7, 0.8), (0.3, 0.3), (0.7, 0.4)]
        )
        stats1 = self._make_team_stats(tids1)
        r1 = tracker.process_rally(
            track_stats=stats1, player_positions=positions1, court_split_y=0.5
        )

        # Record rally 1 assignments
        p_for_left_near = r1.track_to_player[10]  # x=0.3, near
        p_for_right_near = r1.track_to_player[11]  # x=0.7, near

        # Rally 2: same physical players at same positions, new track IDs.
        # Track 30 at x=0.3, Track 31 at x=0.7 — same positions as rally 1.
        tids2 = [30, 31, 40, 41]
        positions2 = _make_positions_xy(
            tids2, [(0.3, 0.7), (0.7, 0.8), (0.3, 0.3), (0.7, 0.4)]
        )
        stats2 = self._make_team_stats(tids2)
        r2 = tracker.process_rally(
            track_stats=stats2, player_positions=positions2, court_split_y=0.5
        )

        # With identical appearances, global Hungarian is random within-team.
        # Position continuity should ensure the left player keeps the same ID.
        assert r2.track_to_player[30] == p_for_left_near
        assert r2.track_to_player[31] == p_for_right_near

    def test_no_swap_when_positions_already_correct(self) -> None:
        """Position continuity confirms the global assignment → no swap."""
        tracker = MatchPlayerTracker()

        # Use distinct appearances so global Hungarian gets it right
        tids1 = [10, 11, 20, 21]
        stats1 = {
            10: _make_stats(10, skin_hsv=(15.0, 180.0, 120.0), height=0.18,
                            lower_hue=0.0, lower_sat=200.0),
            11: _make_stats(11, skin_hsv=(25.0, 100.0, 200.0), height=0.12,
                            lower_hue=60.0, lower_sat=200.0),
            20: _make_stats(20, skin_hsv=(10.0, 160.0, 150.0), height=0.10,
                            lower_hue=120.0, lower_sat=200.0),
            21: _make_stats(21, skin_hsv=(30.0, 120.0, 170.0), height=0.09,
                            lower_hue=140.0, lower_sat=200.0),
        }
        positions1 = _make_positions_xy(
            tids1, [(0.3, 0.7), (0.7, 0.8), (0.3, 0.3), (0.7, 0.4)]
        )
        r1 = tracker.process_rally(
            track_stats=stats1, player_positions=positions1, court_split_y=0.5
        )

        # Rally 2: same appearances, same positions
        tids2 = [30, 31, 40, 41]
        stats2 = {
            30: _make_stats(30, skin_hsv=(15.0, 180.0, 120.0), height=0.18,
                            lower_hue=0.0, lower_sat=200.0),
            31: _make_stats(31, skin_hsv=(25.0, 100.0, 200.0), height=0.12,
                            lower_hue=60.0, lower_sat=200.0),
            40: _make_stats(40, skin_hsv=(10.0, 160.0, 150.0), height=0.10,
                            lower_hue=120.0, lower_sat=200.0),
            41: _make_stats(41, skin_hsv=(30.0, 120.0, 170.0), height=0.09,
                            lower_hue=140.0, lower_sat=200.0),
        }
        positions2 = _make_positions_xy(
            tids2, [(0.3, 0.7), (0.7, 0.8), (0.3, 0.3), (0.7, 0.4)]
        )
        r2 = tracker.process_rally(
            track_stats=stats2, player_positions=positions2, court_split_y=0.5
        )

        # Global assignment correct + position continuity confirms = same result
        assert r2.track_to_player[30] == r1.track_to_player[10]
        assert r2.track_to_player[31] == r1.track_to_player[11]
        assert r2.track_to_player[40] == r1.track_to_player[20]
        assert r2.track_to_player[41] == r1.track_to_player[21]

    def test_no_swap_below_threshold(self) -> None:
        """When both permutations have similar cost → no swap (threshold)."""
        tracker = MatchPlayerTracker()

        # Rally 1: both near players at almost the same position
        tids1 = [10, 11, 20, 21]
        positions1 = _make_positions_xy(
            tids1, [(0.50, 0.7), (0.51, 0.8), (0.3, 0.3), (0.7, 0.4)]
        )
        stats1 = self._make_team_stats(tids1)
        r1 = tracker.process_rally(
            track_stats=stats1, player_positions=positions1, court_split_y=0.5
        )
        p1 = r1.track_to_player[10]
        p2 = r1.track_to_player[11]

        # Rally 2: positions slightly shuffled but still very close
        tids2 = [30, 31, 40, 41]
        positions2 = _make_positions_xy(
            tids2, [(0.51, 0.7), (0.50, 0.8), (0.3, 0.3), (0.7, 0.4)]
        )
        stats2 = self._make_team_stats(tids2)
        r2 = tracker.process_rally(
            track_stats=stats2, player_positions=positions2, court_split_y=0.5
        )

        # Costs should be so similar that the 20% threshold prevents swapping.
        # Whatever global Hungarian assigned, refinement should NOT flip it.
        assigned_pids = {r2.track_to_player[30], r2.track_to_player[31]}
        assert assigned_pids == {p1, p2}

    def test_graceful_degradation_no_previous_positions(self) -> None:
        """First rally: no previous positions → falls back to global assignment."""
        tracker = MatchPlayerTracker()

        tids = [10, 11, 20, 21]
        positions = _make_positions_xy(
            tids, [(0.3, 0.7), (0.7, 0.8), (0.3, 0.3), (0.7, 0.4)]
        )
        stats = self._make_team_stats(tids)
        result = tracker.process_rally(
            track_stats=stats, player_positions=positions, court_split_y=0.5
        )

        # Should still assign all 4 players without error
        assert len(result.track_to_player) == 4
        assert set(result.track_to_player.values()) == {1, 2, 3, 4}

    def test_last_positions_populated_across_rallies(self) -> None:
        """Player last positions are stored for position continuity."""
        tracker = MatchPlayerTracker()

        near_skin = (15.0, 180.0, 100.0)
        far_skin = (30.0, 80.0, 220.0)

        # Build profiles over 3 rallies
        for i in range(3):
            tids_near = [100 + i * 10, 101 + i * 10]
            tids_far = [200 + i * 10, 201 + i * 10]
            stats = {
                tids_near[0]: _make_stats(tids_near[0], skin_hsv=near_skin, height=0.18),
                tids_near[1]: _make_stats(tids_near[1], skin_hsv=near_skin, height=0.17),
                tids_far[0]: _make_stats(tids_far[0], skin_hsv=far_skin, height=0.10),
                tids_far[1]: _make_stats(tids_far[1], skin_hsv=far_skin, height=0.09),
            }
            positions = _make_positions(
                tids_near + tids_far, [0.7, 0.75, 0.3, 0.35]
            )
            tracker.process_rally(
                track_stats=stats, player_positions=positions, court_split_y=0.5
            )

        # After 3 rallies, last positions should be populated
        assert len(tracker.state.player_last_positions) > 0

        # Rally 4: new tracks, positions still tracked
        stats_4 = {
            500: _make_stats(500, skin_hsv=far_skin, height=0.10),
            501: _make_stats(501, skin_hsv=far_skin, height=0.09),
            600: _make_stats(600, skin_hsv=near_skin, height=0.18),
            601: _make_stats(601, skin_hsv=near_skin, height=0.17),
        }
        positions_4 = _make_positions([500, 501, 600, 601], [0.7, 0.75, 0.3, 0.35])
        tracker.process_rally(
            track_stats=stats_4, player_positions=positions_4, court_split_y=0.5
        )

        # Verify positions are stored for continuity.
        assert len(tracker.state.player_last_positions) > 0

    def test_position_continuity_across_multiple_rallies(self) -> None:
        """5 rallies with consistent positions → always correct assignment."""
        tracker = MatchPlayerTracker()

        # 4 players with distinct spatial positions but identical appearances
        # (simulating same-team identical clothing)
        player_xy = [
            (0.25, 0.70),  # near-left
            (0.75, 0.80),  # near-right
            (0.25, 0.30),  # far-left
            (0.75, 0.40),  # far-right
        ]

        rally_assignments: list[dict[int, int]] = []
        for rally_idx in range(5):
            base = (rally_idx + 1) * 100
            tids = [base + j for j in range(4)]
            positions = _make_positions_xy(tids, player_xy)
            stats = self._make_team_stats(tids)

            result = tracker.process_rally(
                track_stats=stats, player_positions=positions, court_split_y=0.5
            )
            rally_assignments.append(result.track_to_player.copy())

        # Each physical player position should get the same ID across all rallies
        for rally_idx in range(1, 5):
            base_prev = rally_idx * 100
            base_curr = (rally_idx + 1) * 100
            prev = rally_assignments[rally_idx - 1]
            curr = rally_assignments[rally_idx]
            for j in range(4):
                assert prev[base_prev + j] == curr[base_curr + j], (
                    f"Rally {rally_idx}: player at position {j} got different ID "
                    f"({prev[base_prev + j]} vs {curr[base_curr + j]})"
                )


class TestHelpers:
    """Test module-level helper functions."""

    def test_compute_track_positions_from_start(self) -> None:
        """Avg of first N frames."""
        positions = _make_positions_xy(
            [1, 2], [(0.2, 0.7), (0.8, 0.3)], num_frames=60
        )
        result = _compute_track_positions(positions, [1, 2], window=10, from_start=True)
        assert abs(result[1][0] - 0.2) < 0.01
        assert abs(result[1][1] - 0.7) < 0.01

    def test_compute_track_positions_from_end(self) -> None:
        """Avg of last N frames."""
        positions = _make_positions_xy(
            [1], [(0.5, 0.5)], num_frames=60
        )
        result = _compute_track_positions(positions, [1], window=10, from_start=False)
        assert abs(result[1][0] - 0.5) < 0.01

    def test_dist(self) -> None:
        """Euclidean distance."""
        assert abs(_dist((0.0, 0.0), (3.0, 4.0)) - 5.0) < 1e-6
        assert abs(_dist((1.0, 1.0), (1.0, 1.0))) < 1e-6


class TestConfidence:
    """Test assignment confidence scoring."""

    def test_first_rally_confidence(self) -> None:
        """First rally should have 0.5 confidence."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 20], [0.7, 0.3])
        stats = {10: _make_stats(10), 20: _make_stats(20)}

        result = tracker.process_rally(
            track_stats=stats, player_positions=positions, court_split_y=0.5
        )
        assert result.assignment_confidence == 0.5

    def test_identical_features_high_confidence(self) -> None:
        """Matching identical features should give high confidence."""
        tracker = MatchPlayerTracker()
        skin = (20.0, 150.0, 180.0)

        # Rally 1
        stats1 = {
            10: _make_stats(10, skin_hsv=skin, height=0.15),
            20: _make_stats(20, skin_hsv=skin, height=0.10),
        }
        positions1 = _make_positions([10, 20], [0.7, 0.3])
        tracker.process_rally(
            track_stats=stats1, player_positions=positions1, court_split_y=0.5
        )

        # Rally 2 with identical appearances
        stats2 = {
            30: _make_stats(30, skin_hsv=skin, height=0.15),
            40: _make_stats(40, skin_hsv=skin, height=0.10),
        }
        positions2 = _make_positions([30, 40], [0.7, 0.3])
        result = tracker.process_rally(
            track_stats=stats2, player_positions=positions2, court_split_y=0.5
        )

        assert result.assignment_confidence > 0.5

    def test_empty_tracks_zero_confidence(self) -> None:
        """No tracks should give 0 confidence."""
        tracker = MatchPlayerTracker()
        result = tracker.process_rally(
            track_stats={}, player_positions=[], court_split_y=0.5
        )
        assert result.assignment_confidence == 0.0


class TestCourtSplitFallbackWithTeamAssignments:
    """Test court_split_y failure with partial team_assignments fallback."""

    def test_court_split_failure_uses_team_assignments(self) -> None:
        """When court_split_y puts all on one side, team_assignments rescues."""
        tracker = MatchPlayerTracker()
        # All tracks below court_split_y=0.637 (compressed Y)
        positions = _make_positions([10, 11, 20, 21], [0.60, 0.63, 0.49, 0.52])
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21]}

        # court_split_y fails but team_assignments knows the answer
        team_assignments = {10: 0, 11: 0, 20: 1, 21: 1}
        _, sides = tracker._classify_track_sides(
            stats, positions, court_split_y=0.637,
            team_assignments=team_assignments,
        )

        assert sides[10] == 0  # near
        assert sides[11] == 0  # near
        assert sides[20] == 1  # far
        assert sides[21] == 1  # far

    def test_court_split_failure_partial_team_assignments(self) -> None:
        """Even partial team_assignments (2/4) can rescue court_split failure."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 11, 20, 21], [0.60, 0.63, 0.49, 0.52])
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21]}

        # Only 2 tracks covered, but they're on different teams → valid split
        team_assignments = {10: 0, 20: 1}
        _, sides = tracker._classify_track_sides(
            stats, positions, court_split_y=0.637,
            team_assignments=team_assignments,
        )

        # Should use team_assignments (two different teams seen)
        assert sides[10] == 0
        assert sides[20] == 1

    def test_court_split_failure_same_team_assignments_falls_through(self) -> None:
        """team_assignments all same team → still falls to median split."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 11, 20, 21], [0.60, 0.63, 0.49, 0.52])
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21]}

        # All covered tracks on same team → no split possible
        team_assignments = {10: 0, 11: 0}
        _, sides = tracker._classify_track_sides(
            stats, positions, court_split_y=0.637,
            team_assignments=team_assignments,
        )

        # Falls through to median split: 2 near, 2 far
        near_count = sum(1 for s in sides.values() if s == 0)
        far_count = sum(1 for s in sides.values() if s == 1)
        assert near_count == 2
        assert far_count == 2


class TestStoredRallyData:
    """Test that rally data is stored during processing."""

    def test_stored_rally_data_populated(self) -> None:
        """stored_rally_data should be populated after processing rallies."""
        tracker = MatchPlayerTracker()
        for i in range(3):
            base = (i + 1) * 100
            tids = [base, base + 1, base + 2, base + 3]
            positions = _make_positions(tids, [0.7, 0.8, 0.3, 0.4])
            stats = {t: _make_stats(t) for t in tids}
            tracker.process_rally(
                track_stats=stats, player_positions=positions, court_split_y=0.5
            )

        assert len(tracker.stored_rally_data) == 3
        for data in tracker.stored_rally_data:
            assert len(data.track_stats) > 0
            assert len(data.track_court_sides) > 0
            assert len(data.top_tracks) > 0


class TestRefineAssignments:
    """Test Pass 2 refinement corrects early-rally errors."""

    def test_refinement_corrects_early_rallies(self) -> None:
        """Pass 2 should correct early-rally assignments using richer profiles.

        Scenario: 4 physically distinct players. Rally 1 establishes profiles
        with deterministic Y-sort. Rallies 2-5 use identical appearances but
        richer profiles accumulate. Pass 2 re-scores Rally 2 (which had only
        1-rally profiles) with the final 5-rally profiles.
        """
        # 4 distinct player appearances
        appearances: list[dict[str, Any]] = [
            dict(skin_hsv=(15.0, 180.0, 120.0), height=0.18,
                 lower_hue=0.0, lower_sat=200.0, upper_hue=110.0, upper_sat=180.0),
            dict(skin_hsv=(25.0, 100.0, 200.0), height=0.12,
                 lower_hue=60.0, lower_sat=200.0, upper_hue=30.0, upper_sat=180.0),
            dict(skin_hsv=(10.0, 160.0, 150.0), height=0.10,
                 lower_hue=15.0, lower_sat=220.0, upper_hue=0.0, upper_sat=30.0),
            dict(skin_hsv=(30.0, 120.0, 170.0), height=0.09,
                 lower_hue=140.0, lower_sat=150.0, upper_hue=0.0, upper_sat=50.0),
        ]

        tracker = MatchPlayerTracker()
        results: list[RallyTrackingResult] = []

        for rally_idx in range(5):
            base = (rally_idx + 1) * 100
            tids = [base + j for j in range(4)]
            stats = {
                tids[j]: _make_stats(tids[j], **appearances[j])
                for j in range(4)
            }
            positions = _make_positions(tids, [0.7, 0.8, 0.3, 0.4])
            result = tracker.process_rally(
                track_stats=stats, player_positions=positions, court_split_y=0.5
            )
            results.append(result)

        # Refine
        refined = tracker.refine_assignments(results)

        assert len(refined) == 5

        # All rallies should have valid assignments
        for r in refined:
            assert len(r.track_to_player) == 4
            assert set(r.track_to_player.values()) == {1, 2, 3, 4}

    def test_refinement_preserves_first_rally(self) -> None:
        """Pass 2 keeps first rally unchanged (deterministic Y-sort)."""
        tracker = MatchPlayerTracker()
        results: list[RallyTrackingResult] = []

        for i in range(3):
            base = (i + 1) * 100
            tids = [base, base + 1, base + 2, base + 3]
            stats = {t: _make_stats(t) for t in tids}
            positions = _make_positions(tids, [0.7, 0.8, 0.3, 0.4])
            result = tracker.process_rally(
                track_stats=stats, player_positions=positions, court_split_y=0.5
            )
            results.append(result)

        refined = tracker.refine_assignments(results)

        # First rally should be identical
        assert refined[0].track_to_player == results[0].track_to_player

    def test_refinement_single_rally_noop(self) -> None:
        """Pass 2 with single rally returns it unchanged."""
        tracker = MatchPlayerTracker()
        stats = {t: _make_stats(t) for t in [10, 11, 20, 21]}
        positions = _make_positions([10, 11, 20, 21], [0.7, 0.8, 0.3, 0.4])
        result = tracker.process_rally(
            track_stats=stats, player_positions=positions, court_split_y=0.5
        )

        refined = tracker.refine_assignments([result])
        assert len(refined) == 1
        assert refined[0].track_to_player == result.track_to_player
