"""Unit tests for cross-rally player matching."""

from __future__ import annotations

from rallycut.tracking.match_tracker import MatchPlayerTracker
from rallycut.tracking.player_features import (
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


def _make_stats(
    track_id: int,
    skin_hsv: tuple[float, float, float] = (20.0, 150.0, 180.0),
    jersey_hsv: tuple[float, float, float] | None = None,
    height: float = 0.15,
    num_features: int = 5,
) -> TrackAppearanceStats:
    """Create TrackAppearanceStats with known feature values."""
    stats = TrackAppearanceStats(track_id=track_id)
    for i in range(num_features):
        f = PlayerAppearanceFeatures(
            track_id=track_id,
            frame_number=i,
            skin_tone_hsv=skin_hsv,
            skin_pixel_count=100,
            jersey_color_hsv=jersey_hsv,
            jersey_pixel_count=50 if jersey_hsv else 0,
            bbox_height=height,
            bbox_aspect_ratio=0.33,
        )
        stats.features.append(f)
    stats.compute_averages()
    return stats


class TestHungarianAssignment:
    """Test _assign_tracks_to_players with Hungarian algorithm."""

    def test_first_rally_arbitrary_assignment(self) -> None:
        """First rally should assign arbitrarily (near→P1,P2; far→P3,P4)."""
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

    def test_hungarian_matches_similar_appearances(self) -> None:
        """Hungarian should match tracks to the most similar profiles."""
        tracker = MatchPlayerTracker()

        # Rally 1: establish profiles
        # Track 10 (near): dark skin, tall
        # Track 11 (near): light skin, short
        # Track 20 (far): reddish skin
        # Track 21 (far): yellowish skin
        stats1 = {
            10: _make_stats(10, skin_hsv=(15.0, 180.0, 120.0), height=0.18),
            11: _make_stats(11, skin_hsv=(25.0, 100.0, 200.0), height=0.12),
            20: _make_stats(20, skin_hsv=(10.0, 160.0, 150.0), height=0.10),
            21: _make_stats(21, skin_hsv=(30.0, 120.0, 170.0), height=0.09),
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
        # Track 30 should match profile of track 10's player (dark skin, tall)
        # Track 31 should match profile of track 11's player (light skin, short)
        stats2 = {
            30: _make_stats(30, skin_hsv=(15.0, 180.0, 120.0), height=0.18),
            31: _make_stats(31, skin_hsv=(25.0, 100.0, 200.0), height=0.12),
            40: _make_stats(40, skin_hsv=(10.0, 160.0, 150.0), height=0.10),
            41: _make_stats(41, skin_hsv=(30.0, 120.0, 170.0), height=0.09),
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

    def test_more_tracks_than_two_per_side(self) -> None:
        """More than 2 tracks per side should take top 2 by frame count."""
        tracker = MatchPlayerTracker()
        # Track 10 has more features, track 12 has fewer
        positions = _make_positions([10, 11, 12, 20, 21], [0.7, 0.75, 0.8, 0.3, 0.35])
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

        # Should assign 4 tracks (top 2 per side)
        assert len(result.track_to_player) == 4
        # Track 12 should be dropped (fewest features on near side)
        assert 12 not in result.track_to_player

    def test_empty_tracks(self) -> None:
        """Empty track lists should produce empty assignments."""
        tracker = MatchPlayerTracker()
        result = tracker.process_rally(
            track_stats={}, player_positions=[], court_split_y=0.5
        )

        assert result.track_to_player == {}
        assert result.assignment_confidence == 0.0


class TestSideSwitchDetection:
    """Test _detect_side_switch appearance-based detection."""

    def test_no_switch_on_first_rallies(self) -> None:
        """Side switch should not be detected in first 2 rallies."""
        tracker = MatchPlayerTracker()
        positions = _make_positions([10, 11, 20, 21], [0.7, 0.8, 0.3, 0.4])
        stats = {
            10: _make_stats(10, skin_hsv=(15.0, 180.0, 120.0)),
            11: _make_stats(11, skin_hsv=(25.0, 100.0, 200.0)),
            20: _make_stats(20, skin_hsv=(10.0, 160.0, 150.0)),
            21: _make_stats(21, skin_hsv=(30.0, 120.0, 170.0)),
        }

        # Rally 1: no switch possible
        r1 = tracker.process_rally(
            track_stats=stats, player_positions=positions, court_split_y=0.5
        )
        assert not r1.side_switch_detected

        # Rally 2: still too early (need >=3 rallies for stable profiles)
        stats2 = {
            30: _make_stats(30, skin_hsv=(15.0, 180.0, 120.0)),
            31: _make_stats(31, skin_hsv=(25.0, 100.0, 200.0)),
            40: _make_stats(40, skin_hsv=(10.0, 160.0, 150.0)),
            41: _make_stats(41, skin_hsv=(30.0, 120.0, 170.0)),
        }
        positions2 = _make_positions([30, 31, 40, 41], [0.7, 0.8, 0.3, 0.4])
        r2 = tracker.process_rally(
            track_stats=stats2, player_positions=positions2, court_split_y=0.5
        )
        assert not r2.side_switch_detected

    def test_switch_detected_when_appearances_swap(self) -> None:
        """Switch should be detected when near/far appearances clearly swap."""
        tracker = MatchPlayerTracker()

        # Distinct appearances: near team has dark skin, far has light
        near_skin = (15.0, 180.0, 100.0)
        far_skin = (30.0, 80.0, 220.0)

        # Build profiles over 3 rallies with consistent appearances
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

        # Rally 4: appearances SWAPPED (far team now on near side)
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

        assert result.side_switch_detected

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
                tids_near[0]: _make_stats(tids_near[0], skin_hsv=skin_a, height=0.18),
                tids_near[1]: _make_stats(tids_near[1], skin_hsv=skin_a, height=0.17),
                tids_far[0]: _make_stats(tids_far[0], skin_hsv=skin_b, height=0.10),
                tids_far[1]: _make_stats(tids_far[1], skin_hsv=skin_b, height=0.09),
            }
            positions = _make_positions(
                tids_near + tids_far, [0.7, 0.75, 0.3, 0.35]
            )
            result = tracker.process_rally(
                track_stats=stats, player_positions=positions, court_split_y=0.5
            )

        # None of the later rallies should detect a switch
        assert not result.side_switch_detected


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
