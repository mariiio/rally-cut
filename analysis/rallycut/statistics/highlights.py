"""Highlight ranking and scoring for volleyball rallies.

Ranks rallies by excitement using multiple signals:
- Rally length (longer = more exciting)
- Action drama (blocks, extended exchanges, high velocity)
- Audio excitement (crowd noise energy)
- Score context (clutch moments — requires external score input)

Generates highlight clips by selecting top-K ranked rallies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HighlightFeatures:
    """Features extracted from a rally for highlight scoring."""

    rally_id: str
    # Duration features
    duration_seconds: float = 0.0
    num_contacts: int = 0
    # Action features
    has_block: bool = False
    has_extended_exchange: bool = False  # 3+ contacts on one side
    num_side_changes: int = 0  # Number of times ball crossed net
    max_velocity: float = 0.0  # Peak ball velocity
    # Audio features
    audio_peak_rms: float = 0.0  # Peak RMS energy during rally
    audio_mean_rms: float = 0.0  # Mean RMS energy
    audio_variance: float = 0.0  # RMS variance (excitement fluctuation)
    # Score context (optional, from external input)
    score_diff: int | None = None  # Point difference at rally start
    is_set_point: bool = False
    is_match_point: bool = False
    # Timing
    start_ms: float = 0.0
    end_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rallyId": self.rally_id,
            "durationSeconds": round(self.duration_seconds, 2),
            "numContacts": self.num_contacts,
            "hasBlock": self.has_block,
            "hasExtendedExchange": self.has_extended_exchange,
            "numSideChanges": self.num_side_changes,
            "maxVelocity": round(self.max_velocity, 4),
            "audioPeakRms": round(self.audio_peak_rms, 4),
            "audioMeanRms": round(self.audio_mean_rms, 4),
            "startMs": self.start_ms,
            "endMs": self.end_ms,
        }
        if self.score_diff is not None:
            result["scoreDiff"] = self.score_diff
        if self.is_set_point:
            result["isSetPoint"] = True
        if self.is_match_point:
            result["isMatchPoint"] = True
        return result


@dataclass
class HighlightScore:
    """Scored rally for highlight ranking."""

    rally_id: str
    total_score: float
    # Component scores (0-1 each, before weighting)
    duration_score: float = 0.0
    action_score: float = 0.0
    audio_score: float = 0.0
    context_score: float = 0.0
    # Features used for scoring
    features: HighlightFeatures | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rallyId": self.rally_id,
            "totalScore": round(self.total_score, 4),
            "durationScore": round(self.duration_score, 4),
            "actionScore": round(self.action_score, 4),
            "audioScore": round(self.audio_score, 4),
            "contextScore": round(self.context_score, 4),
        }
        if self.features:
            result["features"] = self.features.to_dict()
        return result


@dataclass
class HighlightScorerConfig:
    """Configuration for highlight scoring weights."""

    # Component weights (should sum to 1.0)
    weight_duration: float = 0.30
    weight_action: float = 0.35
    weight_audio: float = 0.20
    weight_context: float = 0.15

    # Duration scoring
    # Rallies longer than this get max duration score
    max_duration_seconds: float = 20.0
    # Rallies shorter than this get zero duration score
    min_duration_seconds: float = 3.0

    # Action scoring bonuses
    block_bonus: float = 0.25
    extended_exchange_bonus: float = 0.20
    side_change_bonus_per: float = 0.05  # Per additional side change beyond 2
    high_velocity_bonus: float = 0.15  # For max velocity > threshold
    high_velocity_threshold: float = 0.04  # Normalized velocity

    # Audio scoring
    audio_percentile_threshold: float = 0.75  # Top 25% audio = high excitement

    # Context scoring
    set_point_bonus: float = 0.50
    match_point_bonus: float = 0.80
    close_score_bonus: float = 0.30  # When score diff <= 2


class HighlightScorer:
    """Score and rank rallies for highlight generation.

    Produces a ranked list of rallies by excitement, using multiple
    signals that can be progressively enabled:

    Phase 1: Rally duration only (simple, always available)
    Phase 2: + action drama (requires action classification)
    Phase 3: + audio excitement (requires audio extraction)
    Phase 4: + score context (requires scoreboard OCR or manual input)
    """

    def __init__(self, config: HighlightScorerConfig | None = None):
        self.config = config or HighlightScorerConfig()

    def score_rallies(
        self,
        features_list: list[HighlightFeatures],
    ) -> list[HighlightScore]:
        """Score all rallies and return sorted by excitement.

        Args:
            features_list: Features for each rally.

        Returns:
            List of HighlightScore sorted by total_score descending.
        """
        if not features_list:
            return []

        # Compute audio normalization (relative to all rallies in match)
        audio_rms_values = [f.audio_peak_rms for f in features_list if f.audio_peak_rms > 0]
        audio_threshold = float(np.percentile(
            audio_rms_values, self.config.audio_percentile_threshold * 100
        )) if audio_rms_values else 0.0

        scores: list[HighlightScore] = []

        for features in features_list:
            duration_score = self._score_duration(features)
            action_score = self._score_actions(features)
            audio_score = self._score_audio(features, audio_threshold)
            context_score = self._score_context(features)

            total = (
                self.config.weight_duration * duration_score
                + self.config.weight_action * action_score
                + self.config.weight_audio * audio_score
                + self.config.weight_context * context_score
            )

            scores.append(HighlightScore(
                rally_id=features.rally_id,
                total_score=total,
                duration_score=duration_score,
                action_score=action_score,
                audio_score=audio_score,
                context_score=context_score,
                features=features,
            ))

        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)

        return scores

    def _score_duration(self, features: HighlightFeatures) -> float:
        """Score based on rally duration (longer = more exciting)."""
        cfg = self.config
        if features.duration_seconds <= cfg.min_duration_seconds:
            return 0.0
        if features.duration_seconds >= cfg.max_duration_seconds:
            return 1.0

        # Linear interpolation between min and max
        return (features.duration_seconds - cfg.min_duration_seconds) / (
            cfg.max_duration_seconds - cfg.min_duration_seconds
        )

    def _score_actions(self, features: HighlightFeatures) -> float:
        """Score based on action drama."""
        cfg = self.config
        score = 0.0

        # Base score from number of contacts (more contacts = more exciting)
        if features.num_contacts > 0:
            # Log scale: diminishing returns after ~8 contacts
            score += min(math.log2(features.num_contacts + 1) / 4.0, 0.5)

        # Bonuses
        if features.has_block:
            score += cfg.block_bonus
        if features.has_extended_exchange:
            score += cfg.extended_exchange_bonus

        # Side changes beyond the minimum 2 (serve + return)
        extra_side_changes = max(0, features.num_side_changes - 2)
        score += extra_side_changes * cfg.side_change_bonus_per

        # High velocity (powerful attacks)
        if features.max_velocity > cfg.high_velocity_threshold:
            score += cfg.high_velocity_bonus

        return min(score, 1.0)

    def _score_audio(
        self,
        features: HighlightFeatures,
        audio_threshold: float,
    ) -> float:
        """Score based on audio excitement (crowd noise)."""
        if features.audio_peak_rms <= 0 or audio_threshold <= 0:
            return 0.0  # No audio data

        # Score relative to match-level threshold
        if features.audio_peak_rms >= audio_threshold:
            # Above threshold: scale 0.5 to 1.0
            excess = features.audio_peak_rms / audio_threshold
            return min(0.5 + 0.5 * (excess - 1.0), 1.0)
        else:
            # Below threshold: scale 0 to 0.5
            return 0.5 * features.audio_peak_rms / audio_threshold

    def _score_context(self, features: HighlightFeatures) -> float:
        """Score based on score context (clutch moments)."""
        cfg = self.config
        score = 0.0

        if features.is_match_point:
            score += cfg.match_point_bonus
        elif features.is_set_point:
            score += cfg.set_point_bonus

        if features.score_diff is not None and abs(features.score_diff) <= 2:
            score += cfg.close_score_bonus

        return min(score, 1.0)

    def get_top_k(
        self,
        scores: list[HighlightScore],
        k: int = 10,
    ) -> list[HighlightScore]:
        """Get top-K highlights.

        Args:
            scores: Scored rallies (already sorted).
            k: Number of highlights to return.

        Returns:
            Top-K HighlightScore list.
        """
        return scores[:k]


def extract_rally_features(
    rally_stats: Any,  # RallyStats from match_stats
    rally_id: str = "",
    start_ms: float = 0.0,
    end_ms: float = 0.0,
    audio_rms: list[float] | None = None,
    score_diff: int | None = None,
    is_set_point: bool = False,
    is_match_point: bool = False,
) -> HighlightFeatures:
    """Extract highlight features from rally statistics.

    Args:
        rally_stats: RallyStats object from match_stats module.
        rally_id: Rally identifier.
        start_ms: Rally start time in milliseconds.
        end_ms: Rally end time in milliseconds.
        audio_rms: Per-frame audio RMS values during the rally.
        score_diff: Score difference at rally start.
        is_set_point: Whether this rally is a set point.
        is_match_point: Whether this rally is a match point.

    Returns:
        HighlightFeatures for this rally.
    """
    features = HighlightFeatures(
        rally_id=rally_id or getattr(rally_stats, "rally_id", ""),
        duration_seconds=getattr(rally_stats, "duration_seconds", 0.0),
        num_contacts=getattr(rally_stats, "num_contacts", 0),
        has_block=getattr(rally_stats, "has_block", False),
        has_extended_exchange=getattr(rally_stats, "has_extended_exchange", False),
        max_velocity=getattr(rally_stats, "max_rally_velocity", 0.0),
        start_ms=start_ms,
        end_ms=end_ms,
        score_diff=score_diff,
        is_set_point=is_set_point,
        is_match_point=is_match_point,
    )

    # Compute side changes from action sequence
    action_seq = getattr(rally_stats, "action_sequence", [])
    # Count how many times the implied court side changes
    # (each receive/dig after an attack implies a side change)
    side_changes = 0
    for i in range(1, len(action_seq)):
        if action_seq[i] in ("receive", "dig") and action_seq[i - 1] in ("attack", "serve"):
            side_changes += 1
    features.num_side_changes = side_changes

    # Audio features
    if audio_rms:
        rms_array = np.array(audio_rms, dtype=np.float64)
        features.audio_peak_rms = float(np.max(rms_array))
        features.audio_mean_rms = float(np.mean(rms_array))
        features.audio_variance = float(np.var(rms_array))

    return features


def extract_audio_rms(
    video_path: Path,
    start_ms: float,
    end_ms: float,
    hop_length: int = 512,
    sr: int = 22050,
) -> list[float]:
    """Extract per-frame audio RMS energy from a video segment.

    Args:
        video_path: Path to video file.
        start_ms: Segment start in milliseconds.
        end_ms: Segment end in milliseconds.
        hop_length: Audio analysis hop length.
        sr: Sample rate for audio loading.

    Returns:
        List of RMS values per audio frame.
    """
    try:
        import librosa  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("librosa not installed, audio features unavailable")
        return []

    try:
        # Load audio segment
        offset_s = start_ms / 1000.0
        duration_s = (end_ms - start_ms) / 1000.0

        y, _ = librosa.load(
            str(video_path),
            sr=sr,
            offset=offset_s,
            duration=duration_s,
            mono=True,
        )

        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        return list(rms.tolist())

    except Exception as e:
        logger.warning(f"Failed to extract audio: {e}")
        return []
