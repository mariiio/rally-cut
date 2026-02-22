"""Inference with trained TemporalMaxer model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from rallycut.temporal.temporal_maxer.model import TemporalMaxer, TemporalMaxerConfig

logger = logging.getLogger(__name__)


@dataclass
class TemporalMaxerResult:
    """Result from TemporalMaxer inference."""

    segments: list[tuple[float, float]] = field(default_factory=list)
    window_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    window_predictions: np.ndarray = field(default_factory=lambda: np.array([]))


class TemporalMaxerInference:
    """Run inference with a trained TemporalMaxer model."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Path) -> TemporalMaxer:
        """Load trained TemporalMaxer model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Reconstruct config from saved data
        config_dict = checkpoint.get("config", {})
        config = TemporalMaxerConfig(**config_dict)

        model = TemporalMaxer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def predict(
        self,
        features: np.ndarray,
        fps: float,
        stride: int,
        window_size: int = 16,
        min_segment_duration: float = 1.0,
        max_gap_duration: float = 3.0,
        max_segment_duration: float = 60.0,
        min_segment_confidence: float = 0.6,
    ) -> TemporalMaxerResult:
        """Run full-sequence inference and extract segments.

        Args:
            features: (num_windows, feature_dim) feature array.
            fps: Video frame rate.
            stride: Frame stride between windows.
            window_size: Frames per window.
            min_segment_duration: Minimum rally duration in seconds.
            max_gap_duration: Fill gaps shorter than this (seconds).
            max_segment_duration: Split segments longer than this (seconds).
            min_segment_confidence: Minimum average probability to keep a segment.
                0.0 disables confidence filtering (default).

        Returns:
            TemporalMaxerResult with segments and probabilities.
        """
        if len(features) == 0:
            return TemporalMaxerResult()

        # Run forward pass
        features_t = torch.from_numpy(features).float().T.unsqueeze(0).to(self.device)
        # features_t shape: (1, feature_dim, T)

        with torch.no_grad():
            logits = self.model(features_t)  # (1, 2, T)
            probs = torch.softmax(logits, dim=1)  # (1, 2, T)
            rally_probs = probs[0, 1].cpu().numpy()  # (T,) rally probability
            predictions = logits.argmax(dim=1)[0].cpu().numpy()  # (T,)

        # Post-process predictions into segments
        window_duration = stride / fps
        segments = self._predictions_to_segments(
            predictions,
            rally_probs,
            window_duration,
            window_size / fps,
            min_segment_duration,
            max_gap_duration,
            max_segment_duration,
            min_segment_confidence,
        )

        return TemporalMaxerResult(
            segments=segments,
            window_probs=rally_probs,
            window_predictions=predictions,
        )

    def _predictions_to_segments(
        self,
        predictions: np.ndarray,
        probs: np.ndarray,
        window_duration: float,
        window_length: float,
        min_duration: float,
        max_gap: float,
        max_duration: float,
        min_confidence: float = 0.0,
    ) -> list[tuple[float, float]]:
        """Convert window predictions to time segments with cleanup.

        Reuses cleanup logic patterns from deterministic_decoder.py:
        - Fill short gaps
        - Remove short segments
        - Confidence filter (remove low-confidence segments)
        - Anti-overmerge (split at lowest probability point)
        """
        n = len(predictions)
        if n == 0:
            return []

        # Extract raw segments
        raw_segments: list[tuple[float, float]] = []
        in_rally = False
        rally_start = 0.0

        for i in range(n):
            t = i * window_duration
            if predictions[i] == 1 and not in_rally:
                in_rally = True
                rally_start = t
            elif predictions[i] == 0 and in_rally:
                in_rally = False
                end_t = t - window_duration + window_length
                raw_segments.append((rally_start, end_t))

        if in_rally:
            end_t = (n - 1) * window_duration + window_length
            raw_segments.append((rally_start, end_t))

        if not raw_segments:
            return []

        # Fill short gaps
        filled: list[tuple[float, float]] = [raw_segments[0]]
        for start, end in raw_segments[1:]:
            prev_start, prev_end = filled[-1]
            gap = start - prev_end
            if gap <= max_gap:
                filled[-1] = (prev_start, end)
            else:
                filled.append((start, end))

        # Remove short segments
        filtered = [(s, e) for s, e in filled if e - s >= min_duration]

        # Confidence filter: remove segments with low average probability
        if min_confidence > 0:
            confident: list[tuple[float, float]] = []
            for start, end in filtered:
                start_idx = int(start / window_duration)
                end_idx = min(int(end / window_duration) + 1, len(probs))
                if end_idx <= start_idx:
                    end_idx = start_idx + 1
                seg_probs = probs[start_idx:min(end_idx, len(probs))]
                avg_prob = float(np.mean(seg_probs)) if len(seg_probs) > 0 else 0.0
                if avg_prob >= min_confidence:
                    confident.append((start, end))
            filtered = confident

        # Anti-overmerge: split segments > max_duration at lowest probability point
        final: list[tuple[float, float]] = []
        for start, end in filtered:
            if end - start <= max_duration:
                final.append((start, end))
            else:
                final.extend(
                    self._split_long_segment(start, end, probs, window_duration, max_duration)
                )

        return final

    def _split_long_segment(
        self,
        start: float,
        end: float,
        probs: np.ndarray,
        window_duration: float,
        max_duration: float,
    ) -> list[tuple[float, float]]:
        """Split a segment exceeding max_duration at its lowest probability point."""
        start_idx = int(start / window_duration)
        end_idx = min(int(end / window_duration), len(probs))

        if end_idx <= start_idx:
            return [(start, end)]

        seg_probs = probs[start_idx:end_idx]
        if len(seg_probs) < 2:
            return [(start, end)]

        min_idx = int(np.argmin(seg_probs))

        # Only split if the probability dip is meaningful
        if seg_probs[min_idx] > 0.6:
            return [(start, end)]  # No good split point

        # Ensure split makes progress (not at edges)
        if min_idx == 0:
            min_idx = 1
        elif min_idx >= len(seg_probs) - 1:
            min_idx = len(seg_probs) - 2

        split_time = (start_idx + min_idx) * window_duration

        # Recursively split if still too long
        result: list[tuple[float, float]] = []
        for s, e in [(start, split_time), (split_time, end)]:
            if e - s > 0.5:  # Minimum 0.5s after split
                if e - s > max_duration:
                    result.extend(self._split_long_segment(s, e, probs, window_duration, max_duration))
                else:
                    result.append((s, e))

        return result if result else [(start, end)]
