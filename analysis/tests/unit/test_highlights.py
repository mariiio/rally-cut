"""Unit tests for highlight ranking and scoring."""

from __future__ import annotations

from rallycut.statistics.highlights import (
    HighlightFeatures,
    HighlightScore,
    HighlightScorer,
    HighlightScorerConfig,
    extract_rally_features,
)


def _features(
    rally_id: str = "r1",
    duration: float = 10.0,
    num_contacts: int = 5,
    has_block: bool = False,
    has_extended: bool = False,
    num_side_changes: int = 3,
    max_velocity: float = 0.03,
    audio_peak_rms: float = 0.0,
    score_diff: int | None = None,
    is_set_point: bool = False,
    is_match_point: bool = False,
) -> HighlightFeatures:
    """Helper to create HighlightFeatures."""
    return HighlightFeatures(
        rally_id=rally_id,
        duration_seconds=duration,
        num_contacts=num_contacts,
        has_block=has_block,
        has_extended_exchange=has_extended,
        num_side_changes=num_side_changes,
        max_velocity=max_velocity,
        audio_peak_rms=audio_peak_rms,
        score_diff=score_diff,
        is_set_point=is_set_point,
        is_match_point=is_match_point,
    )


class TestDurationScoring:
    """Tests for duration-based scoring component."""

    def test_below_minimum_is_zero(self) -> None:
        """Rally shorter than min_duration gets 0 score."""
        scorer = HighlightScorer()
        features = _features(duration=2.0)
        score = scorer._score_duration(features)
        assert score == 0.0

    def test_above_maximum_is_one(self) -> None:
        """Rally longer than max_duration gets 1.0 score."""
        scorer = HighlightScorer()
        features = _features(duration=25.0)
        score = scorer._score_duration(features)
        assert score == 1.0

    def test_midpoint_linear(self) -> None:
        """Midpoint between min and max gets ~0.5 score."""
        config = HighlightScorerConfig(min_duration_seconds=0.0, max_duration_seconds=20.0)
        scorer = HighlightScorer(config)
        features = _features(duration=10.0)
        score = scorer._score_duration(features)
        assert abs(score - 0.5) < 0.01


class TestActionScoring:
    """Tests for action drama scoring component."""

    def test_no_contacts_zero_score(self) -> None:
        """Rally with 0 contacts and no events gets 0 action score."""
        scorer = HighlightScorer()
        features = _features(
            num_contacts=0, has_block=False, max_velocity=0.0,
            num_side_changes=0,
        )
        score = scorer._score_actions(features)
        assert score == 0.0

    def test_block_bonus(self) -> None:
        """Rally with block gets bonus."""
        scorer = HighlightScorer()
        no_block = scorer._score_actions(_features(has_block=False))
        with_block = scorer._score_actions(_features(has_block=True))
        assert with_block > no_block

    def test_extended_exchange_bonus(self) -> None:
        """Rally with extended exchange gets bonus."""
        scorer = HighlightScorer()
        normal = scorer._score_actions(_features(has_extended=False))
        extended = scorer._score_actions(_features(has_extended=True))
        assert extended > normal

    def test_high_velocity_bonus(self) -> None:
        """Rally with high velocity attack gets bonus."""
        scorer = HighlightScorer()
        slow = scorer._score_actions(_features(max_velocity=0.01))
        fast = scorer._score_actions(_features(max_velocity=0.06))
        assert fast > slow

    def test_capped_at_one(self) -> None:
        """Action score is capped at 1.0."""
        scorer = HighlightScorer()
        features = _features(
            num_contacts=20, has_block=True,
            has_extended=True, num_side_changes=10,
            max_velocity=0.1,
        )
        score = scorer._score_actions(features)
        assert score <= 1.0


class TestAudioScoring:
    """Tests for audio excitement scoring component."""

    def test_no_audio_zero_score(self) -> None:
        """No audio data gives 0 score."""
        scorer = HighlightScorer()
        features = _features(audio_peak_rms=0.0)
        score = scorer._score_audio(features, audio_threshold=0.5)
        assert score == 0.0

    def test_above_threshold(self) -> None:
        """Audio above threshold gets score > 0.5."""
        scorer = HighlightScorer()
        features = _features(audio_peak_rms=0.8)
        score = scorer._score_audio(features, audio_threshold=0.5)
        assert score > 0.5

    def test_below_threshold(self) -> None:
        """Audio below threshold gets score < 0.5."""
        scorer = HighlightScorer()
        features = _features(audio_peak_rms=0.2)
        score = scorer._score_audio(features, audio_threshold=0.5)
        assert score < 0.5


class TestContextScoring:
    """Tests for score context scoring component."""

    def test_no_context_zero_score(self) -> None:
        """No score context gives 0."""
        scorer = HighlightScorer()
        features = _features()
        score = scorer._score_context(features)
        assert score == 0.0

    def test_match_point_bonus(self) -> None:
        """Match point gets highest bonus."""
        scorer = HighlightScorer()
        match_point = scorer._score_context(_features(is_match_point=True))
        set_point = scorer._score_context(_features(is_set_point=True))
        assert match_point > set_point

    def test_close_score_bonus(self) -> None:
        """Close score (diff <= 2) gets bonus."""
        scorer = HighlightScorer()
        close = scorer._score_context(_features(score_diff=1))
        far = scorer._score_context(_features(score_diff=10))
        assert close > far

    def test_capped_at_one(self) -> None:
        """Context score is capped at 1.0."""
        scorer = HighlightScorer()
        features = _features(is_match_point=True, score_diff=0)
        score = scorer._score_context(features)
        assert score <= 1.0


class TestScoreRallies:
    """Tests for overall rally scoring and ranking."""

    def test_sorted_by_total_score(self) -> None:
        """Rallies are sorted by total score descending."""
        scorer = HighlightScorer()
        features_list = [
            _features(rally_id="short", duration=4.0, num_contacts=2),
            _features(rally_id="long", duration=15.0, num_contacts=8, has_block=True),
            _features(rally_id="medium", duration=8.0, num_contacts=4),
        ]
        scores = scorer.score_rallies(features_list)

        assert len(scores) == 3
        assert scores[0].rally_id == "long"
        assert scores[0].total_score >= scores[1].total_score
        assert scores[1].total_score >= scores[2].total_score

    def test_empty_input(self) -> None:
        """Empty features list returns empty scores."""
        scorer = HighlightScorer()
        scores = scorer.score_rallies([])
        assert scores == []

    def test_get_top_k(self) -> None:
        """get_top_k returns first K elements."""
        scorer = HighlightScorer()
        features_list = [_features(rally_id=f"r{i}", duration=5 + i) for i in range(10)]
        scores = scorer.score_rallies(features_list)
        top3 = scorer.get_top_k(scores, k=3)
        assert len(top3) == 3
        assert top3[0].total_score >= top3[1].total_score

    def test_top_k_larger_than_list(self) -> None:
        """get_top_k with k > list length returns all."""
        scorer = HighlightScorer()
        features_list = [_features(rally_id="only")]
        scores = scorer.score_rallies(features_list)
        top5 = scorer.get_top_k(scores, k=5)
        assert len(top5) == 1


class TestHighlightScoreDataclass:
    """Tests for HighlightScore structure."""

    def test_to_dict(self) -> None:
        """to_dict produces expected structure."""
        score = HighlightScore(
            rally_id="test",
            total_score=0.75,
            duration_score=0.5,
            action_score=0.8,
            audio_score=0.3,
            context_score=0.0,
        )
        d = score.to_dict()
        assert d["rallyId"] == "test"
        assert d["totalScore"] == 0.75
        assert "durationScore" in d
        assert "actionScore" in d

    def test_to_dict_with_features(self) -> None:
        """to_dict includes features when present."""
        features = _features(rally_id="test")
        score = HighlightScore(
            rally_id="test",
            total_score=0.5,
            features=features,
        )
        d = score.to_dict()
        assert "features" in d
        assert d["features"]["rallyId"] == "test"


class TestHighlightFeaturesDataclass:
    """Tests for HighlightFeatures."""

    def test_to_dict(self) -> None:
        """to_dict produces expected keys."""
        features = _features(
            rally_id="r1",
            duration=10.0,
            score_diff=2,
            is_set_point=True,
        )
        d = features.to_dict()
        assert d["rallyId"] == "r1"
        assert d["durationSeconds"] == 10.0
        assert d["scoreDiff"] == 2
        assert d["isSetPoint"] is True

    def test_to_dict_optional_fields_omitted(self) -> None:
        """Optional fields omitted when not set."""
        features = _features()
        d = features.to_dict()
        assert "scoreDiff" not in d
        assert "isSetPoint" not in d
        assert "isMatchPoint" not in d


class TestExtractRallyFeatures:
    """Tests for feature extraction from rally stats."""

    def test_basic_extraction(self) -> None:
        """Extracts features from a rally stats object."""

        class MockRallyStats:
            rally_id = "test"
            duration_seconds = 12.5
            num_contacts = 6
            has_block = True
            has_extended_exchange = False
            max_rally_velocity = 0.04
            action_sequence = ["serve", "receive", "set", "attack", "dig", "attack"]

        features = extract_rally_features(
            MockRallyStats(),
            rally_id="test",
            start_ms=1000.0,
            end_ms=13500.0,
        )

        assert features.rally_id == "test"
        assert features.duration_seconds == 12.5
        assert features.num_contacts == 6
        assert features.has_block is True
        assert features.start_ms == 1000.0
        assert features.end_ms == 13500.0

    def test_side_change_counting(self) -> None:
        """Counts side changes from action sequence."""

        class MockStats:
            rally_id = "test"
            duration_seconds = 10.0
            num_contacts = 4
            has_block = False
            has_extended_exchange = False
            max_rally_velocity = 0.03
            action_sequence = ["serve", "receive", "set", "attack", "dig", "set", "attack"]

        features = extract_rally_features(MockStats())
        # receive after serve = 1, dig after attack = 1
        assert features.num_side_changes == 2

    def test_audio_features(self) -> None:
        """Audio RMS values produce audio features."""

        class MockStats:
            rally_id = "test"
            duration_seconds = 5.0
            num_contacts = 2
            has_block = False
            has_extended_exchange = False
            max_rally_velocity = 0.02
            action_sequence = []

        rms = [0.1, 0.2, 0.5, 0.3, 0.1]
        features = extract_rally_features(MockStats(), audio_rms=rms)

        assert features.audio_peak_rms == 0.5
        assert 0.2 < features.audio_mean_rms < 0.3
        assert features.audio_variance > 0
