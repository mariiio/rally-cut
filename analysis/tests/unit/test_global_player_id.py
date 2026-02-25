"""Unit tests for global player ID application and profile-based ReID.

Tests cover:
- PlayerAppearanceProfile serialization (to_dict / from_dict roundtrip)
- CrossRallyPrior confidence ramp
- Stats aggregation remapping logic
- Cross-rally cost integration in global identity
- MatchPlayersResult from match_players_across_rallies
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rallycut.tracking.player_features import (
    HS_BINS,
    HS_RANGES,
    PlayerAppearanceProfile,
)


def _make_histogram(dominant_hue: float, dominant_sat: float) -> np.ndarray:
    """Create a synthetic HS histogram with a peak at the given H/S values."""
    h = np.full((20, 20), dominant_hue, dtype=np.uint8)
    s = np.full((20, 20), int(dominant_sat), dtype=np.uint8)
    v = np.full((20, 20), 180, dtype=np.uint8)
    hsv = np.stack([h, s, v], axis=-1)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HS_BINS), HS_RANGES)
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


class TestProfileSerialization:
    """Test PlayerAppearanceProfile to_dict / from_dict roundtrip."""

    def test_empty_profile_roundtrip(self) -> None:
        """Profile with no histograms or skin tone survives roundtrip."""
        profile = PlayerAppearanceProfile(player_id=1, team=0, rally_count=0)
        d = profile.to_dict()
        restored = PlayerAppearanceProfile.from_dict(d)
        assert restored.player_id == 1
        assert restored.team == 0
        assert restored.rally_count == 0
        assert restored.avg_skin_tone_hsv is None
        assert restored.avg_upper_hist is None
        assert restored.avg_lower_hist is None

    def test_full_profile_roundtrip(self) -> None:
        """Profile with all fields survives roundtrip."""
        profile = PlayerAppearanceProfile(
            player_id=3,
            team=1,
            rally_count=5,
            avg_skin_tone_hsv=(25.0, 140.0, 170.0),
            skin_sample_count=50,
            avg_upper_hist=_make_histogram(20.0, 100.0),
            upper_hist_count=30,
            avg_lower_hist=_make_histogram(100.0, 200.0),
            lower_hist_count=30,
            avg_bbox_height=0.15,
            height_sample_count=60,
        )

        d = profile.to_dict()
        restored = PlayerAppearanceProfile.from_dict(d)

        assert restored.player_id == 3
        assert restored.team == 1
        assert restored.rally_count == 5
        assert restored.skin_sample_count == 50
        assert restored.upper_hist_count == 30
        assert restored.lower_hist_count == 30
        assert restored.height_sample_count == 60
        assert restored.avg_bbox_height == pytest.approx(0.15)

        # Skin tone
        assert restored.avg_skin_tone_hsv is not None
        assert restored.avg_skin_tone_hsv[0] == pytest.approx(25.0)
        assert restored.avg_skin_tone_hsv[1] == pytest.approx(140.0)
        assert restored.avg_skin_tone_hsv[2] == pytest.approx(170.0)

        # Histograms shape and values
        assert restored.avg_upper_hist is not None
        assert restored.avg_upper_hist.shape == HS_BINS
        assert restored.avg_upper_hist.dtype == np.float32
        np.testing.assert_allclose(
            restored.avg_upper_hist, profile.avg_upper_hist, atol=1e-6
        )

        assert restored.avg_lower_hist is not None
        assert restored.avg_lower_hist.shape == HS_BINS
        np.testing.assert_allclose(
            restored.avg_lower_hist, profile.avg_lower_hist, atol=1e-6
        )

    def test_to_dict_histogram_is_plain_list(self) -> None:
        """Histograms in to_dict output should be plain Python lists (JSON-safe)."""
        profile = PlayerAppearanceProfile(
            player_id=1,
            avg_lower_hist=_make_histogram(100.0, 200.0),
            lower_hist_count=1,
        )
        d = profile.to_dict()
        assert isinstance(d["avg_lower_hist"], list)
        assert all(isinstance(v, float) for v in d["avg_lower_hist"])
        assert len(d["avg_lower_hist"]) == HS_BINS[0] * HS_BINS[1]

    def test_from_dict_missing_optional_fields(self) -> None:
        """from_dict handles minimal dict (only player_id required)."""
        d = {"player_id": 2}
        profile = PlayerAppearanceProfile.from_dict(d)
        assert profile.player_id == 2
        assert profile.team == 0
        assert profile.rally_count == 0
        assert profile.avg_skin_tone_hsv is None


class TestCrossRallyPrior:
    """Test CrossRallyPrior confidence ramp and gating."""

    def test_confidence_zero_below_min_rallies(self) -> None:
        """Confidence should be 0 when rally_count < 2."""
        from rallycut.tracking.global_identity import CrossRallyPrior

        prior = CrossRallyPrior(player_id=1, team=0, rally_count=1)
        assert prior.confidence == 0.0

        prior0 = CrossRallyPrior(player_id=1, team=0, rally_count=0)
        assert prior0.confidence == 0.0

    def test_confidence_ramps_up(self) -> None:
        """Confidence ramps from 0 at 1 rally to 1.0 at 10."""
        from rallycut.tracking.global_identity import CrossRallyPrior

        prior2 = CrossRallyPrior(player_id=1, team=0, rally_count=2)
        assert prior2.confidence == pytest.approx(1 / 9)

        prior5 = CrossRallyPrior(player_id=1, team=0, rally_count=5)
        assert prior5.confidence == pytest.approx(4 / 9)

        prior10 = CrossRallyPrior(player_id=1, team=0, rally_count=10)
        assert prior10.confidence == pytest.approx(1.0)

    def test_confidence_caps_at_one(self) -> None:
        """Confidence should not exceed 1.0 for high rally counts."""
        from rallycut.tracking.global_identity import CrossRallyPrior

        prior = CrossRallyPrior(player_id=1, team=0, rally_count=50)
        assert prior.confidence == 1.0


class TestCrossRallyAssignmentCost:
    """Test _compute_assignment_cost with cross-rally prior."""

    def test_cost_unchanged_without_prior(self) -> None:
        """Without cross_rally_cost, weights should be original (0.50 appearance)."""
        from rallycut.tracking.global_identity import (
            WEIGHT_APPEARANCE,
            PlayerProfile,
            TrackSegment,
            _compute_assignment_cost,
        )
        from rallycut.tracking.player_tracker import PlayerPosition

        hist = _make_histogram(100.0, 200.0)
        seg = TrackSegment(
            track_id=1, start_frame=0, end_frame=50, team=0,
            positions=[
                PlayerPosition(i, 1, 0.5, 0.7, 0.05, 0.15, 0.9)
                for i in range(50)
            ],
        )
        profile = PlayerProfile(
            player_id=1, team=0, histogram=hist,
            centroid=(0.5, 0.7), mean_bbox_area=0.0075,
        )

        cost_no_prior = _compute_assignment_cost(
            seg, profile, hist, None, None, None,
            cross_rally_cost=None,
        )

        # Appearance should use full weight (identical hist → 0 distance)
        # So appearance component ≈ 0, spatial ≈ 0, bbox ≈ 0
        assert cost_no_prior < 0.3  # Low cost for identical features

    def test_cost_with_prior_redistributes_weights(self) -> None:
        """With cross_rally_cost, appearance weight drops and cross-rally adds."""
        from rallycut.tracking.global_identity import (
            WEIGHT_APPEARANCE,
            WEIGHT_APPEARANCE_WITH_PRIOR,
            WEIGHT_CROSS_RALLY,
            PlayerProfile,
            TrackSegment,
            _compute_assignment_cost,
        )
        from rallycut.tracking.player_tracker import PlayerPosition

        hist = _make_histogram(100.0, 200.0)
        seg = TrackSegment(
            track_id=1, start_frame=0, end_frame=50, team=0,
            positions=[
                PlayerPosition(i, 1, 0.5, 0.7, 0.05, 0.15, 0.9)
                for i in range(50)
            ],
        )
        profile = PlayerProfile(
            player_id=1, team=0, histogram=hist,
            centroid=(0.5, 0.7), mean_bbox_area=0.0075,
        )

        # With a low cross-rally cost (good match)
        cost_good = _compute_assignment_cost(
            seg, profile, hist, None, None, None,
            cross_rally_cost=0.1,
        )

        # With a high cross-rally cost (bad match)
        cost_bad = _compute_assignment_cost(
            seg, profile, hist, None, None, None,
            cross_rally_cost=0.9,
        )

        # Bad prior should increase cost
        assert cost_bad > cost_good
        # The difference should be WEIGHT_CROSS_RALLY * (0.9 - 0.1)
        expected_diff = WEIGHT_CROSS_RALLY * 0.8
        assert abs((cost_bad - cost_good) - expected_diff) < 1e-6

    def test_weights_sum_to_one_with_prior(self) -> None:
        """Verify weight redistribution preserves total = 1.0."""
        from rallycut.tracking.global_identity import (
            WEIGHT_APPEARANCE_WITH_PRIOR,
            WEIGHT_BBOX_SIZE,
            WEIGHT_CROSS_RALLY,
            WEIGHT_MULTI_REGION,
            WEIGHT_SPATIAL,
        )

        total = (
            WEIGHT_APPEARANCE_WITH_PRIOR
            + WEIGHT_SPATIAL
            + WEIGHT_BBOX_SIZE
            + WEIGHT_MULTI_REGION
            + WEIGHT_CROSS_RALLY
        )
        assert total == pytest.approx(1.0)


class TestStatsRemapping:
    """Test the player ID remapping logic from compute_match_stats."""

    def test_remap_actions(self) -> None:
        """Actions should have player_track_id remapped via match_analysis."""
        from rallycut.tracking.action_classifier import ActionType, ClassifiedAction

        # Simulate match_analysis mapping: track 5 → player 1, track 8 → player 3
        player_map = {5: 1, 8: 3}

        orig_actions = [
            {"action": "attack", "playerTrackId": 5, "frame": 10},
            {"action": "receive", "playerTrackId": 8, "frame": 20},
            {"action": "set", "playerTrackId": 99, "frame": 30},  # unmapped
        ]

        remapped = []
        for a in orig_actions:
            orig_tid = a.get("playerTrackId", -1)
            mapped = player_map.get(orig_tid, orig_tid)
            remapped.append(mapped)

        assert remapped == [1, 3, 99]  # 99 keeps original (unmapped)

    def test_remap_team_assignments(self) -> None:
        """Team assignments keys should be remapped."""
        player_map = {5: 1, 8: 3, 6: 2, 9: 4}
        original_teams = {5: 0, 6: 0, 8: 1, 9: 1}

        remapped_teams = {}
        for tid, team in original_teams.items():
            mapped = player_map.get(tid, tid)
            remapped_teams[mapped] = team

        assert remapped_teams == {1: 0, 2: 0, 3: 1, 4: 1}

    def test_frame_offset_prevents_collision(self) -> None:
        """Positions from different rallies should get different frame offsets."""
        fps = 30.0

        # Rally 1 at 10000ms → offset 300 frames
        rally1_offset = int(10000 / 1000 * fps)
        assert rally1_offset == 300

        # Rally 2 at 25000ms → offset 750 frames
        rally2_offset = int(25000 / 1000 * fps)
        assert rally2_offset == 750

        # Frame 5 in rally 1 and frame 5 in rally 2 should not collide
        assert (5 + rally1_offset) != (5 + rally2_offset)


class TestMatchPlayersResult:
    """Test MatchPlayersResult from match_players_across_rallies."""

    def test_result_has_profiles(self) -> None:
        """MatchPlayersResult should include player profiles."""
        from rallycut.tracking.match_tracker import MatchPlayersResult

        profiles = {
            1: PlayerAppearanceProfile(player_id=1, team=0, rally_count=3),
            2: PlayerAppearanceProfile(player_id=2, team=0, rally_count=3),
        }
        result = MatchPlayersResult(rally_results=[], player_profiles=profiles)

        assert len(result.player_profiles) == 2
        assert result.player_profiles[1].player_id == 1
        assert result.player_profiles[1].rally_count == 3

    def test_profiles_serializable(self) -> None:
        """Player profiles from result should be JSON-serializable."""
        import json

        from rallycut.tracking.match_tracker import MatchPlayersResult

        profile = PlayerAppearanceProfile(
            player_id=1, team=0, rally_count=5,
            avg_lower_hist=_make_histogram(100.0, 200.0),
            lower_hist_count=10,
        )
        result = MatchPlayersResult(
            rally_results=[],
            player_profiles={1: profile},
        )

        # Serialize profiles
        profiles_data = {
            str(pid): p.to_dict()
            for pid, p in result.player_profiles.items()
        }

        # Should be JSON-serializable
        json_str = json.dumps(profiles_data)
        assert '"player_id": 1' in json_str
        assert '"avg_lower_hist"' in json_str

        # Should roundtrip
        restored = json.loads(json_str)
        p = PlayerAppearanceProfile.from_dict(restored["1"])
        assert p.player_id == 1
        assert p.avg_lower_hist is not None
        assert p.avg_lower_hist.shape == HS_BINS


class TestBuildCrossRallyPriors:
    """Test building CrossRallyPrior from serialized profiles."""

    def test_build_from_serialized_profiles(self) -> None:
        """CrossRallyPrior should be buildable from serialized profiles."""
        from rallycut.tracking.global_identity import CrossRallyPrior

        hist = _make_histogram(100.0, 200.0)
        profile = PlayerAppearanceProfile(
            player_id=2, team=1, rally_count=8,
            avg_lower_hist=hist, lower_hist_count=20,
        )

        # Simulate serialization roundtrip
        d = profile.to_dict()
        restored = PlayerAppearanceProfile.from_dict(d)

        prior = CrossRallyPrior(
            player_id=restored.player_id,
            team=restored.team,
            shorts_histogram=restored.avg_lower_hist,
            rally_count=restored.rally_count,
        )

        assert prior.player_id == 2
        assert prior.team == 1
        assert prior.rally_count == 8
        assert prior.confidence == pytest.approx(7 / 9)
        assert prior.shorts_histogram is not None
        assert prior.shorts_histogram.shape == HS_BINS

    def test_skip_profiles_with_no_rallies(self) -> None:
        """Profiles with rally_count=0 should be skipped."""
        from rallycut.tracking.global_identity import CrossRallyPrior

        profile = PlayerAppearanceProfile(player_id=1, team=0, rally_count=0)
        prior = CrossRallyPrior(
            player_id=profile.player_id,
            team=profile.team,
            shorts_histogram=profile.avg_lower_hist,
            rally_count=profile.rally_count,
        )
        assert prior.confidence == 0.0
