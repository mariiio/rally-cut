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
        valley_threshold: float = 0.5,
        min_valley_duration: float = 2.0,
        tta_shifts: int = 0,
        rescue_min_avg_prob: float = 0.50,
        rescue_min_windows: int = 6,
        rescue_max_duration: float = 10.0,
        ball_features: np.ndarray | None = None,
    ) -> TemporalMaxerResult:
        """Run full-sequence inference and extract segments.

        Args:
            features: (num_windows, feature_dim) visual feature array.
            fps: Video frame rate.
            stride: Frame stride between windows.
            window_size: Frames per window.
            min_segment_duration: Minimum rally duration in seconds.
            max_gap_duration: Fill gaps shorter than this (seconds).
            max_segment_duration: Split segments longer than this (seconds).
            min_segment_confidence: Minimum average probability to keep a segment.
                0.0 disables confidence filtering (default).
            valley_threshold: Split segments at sustained low-probability regions
                below this threshold. 0.0 disables valley splitting.
            min_valley_duration: Minimum duration (seconds) of a low-probability
                region to trigger a split.
            ball_features: (num_windows, ball_dim) raw ball features. Passed to
                model's ball_proj MLP when config.ball_feature_dim > 0.

        Returns:
            TemporalMaxerResult with segments and probabilities.
        """
        if len(features) == 0:
            return TemporalMaxerResult()

        expected_dim = self.model.config.feature_dim
        assert features.shape[1] == expected_dim, (
            f"Expected {expected_dim}-dim features, got {features.shape[1]}"
        )

        # Run forward pass (with optional TTA)
        features_t = torch.from_numpy(features).float().T.unsqueeze(0).to(self.device)
        # features_t shape: (1, feature_dim, T)

        # Prepare ball features tensor if provided
        ball_t: torch.Tensor | None = None
        if ball_features is not None and self.model.config.ball_feature_dim > 0:
            min_len = min(features_t.shape[2], len(ball_features))
            bf = ball_features[:min_len]
            ball_t = torch.from_numpy(bf).float().T.unsqueeze(0).to(self.device)
            features_t = features_t[:, :, :min_len]

        with torch.no_grad():
            if tta_shifts > 0:
                rally_probs = self._predict_with_tta(features_t, tta_shifts, ball_t)
            else:
                logits = self.model(features_t, ball_features=ball_t)  # (1, 2, T)
                probs = torch.softmax(logits, dim=1)  # (1, 2, T)
                rally_probs = probs[0, 1].cpu().numpy()  # (T,)
            predictions = (rally_probs > 0.5).astype(np.int64)

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
            valley_threshold=valley_threshold,
            min_valley_duration=min_valley_duration,
        )

        # Rescue pass: recover short rallies in gaps with weak but present signal
        if rescue_min_avg_prob > 0:
            rescued = self._rescue_short_rallies(
                segments,
                rally_probs,
                window_duration,
                rescue_min_avg_prob=rescue_min_avg_prob,
                rescue_min_windows=rescue_min_windows,
                rescue_min_duration=min_segment_duration,
                rescue_max_duration=rescue_max_duration,
            )
            if rescued:
                segments = sorted(segments + rescued)
                logger.debug("Rescued %d short rallies", len(rescued))

        return TemporalMaxerResult(
            segments=segments,
            window_probs=rally_probs,
            window_predictions=predictions,
        )

    def _predict_with_tta(
        self,
        features_t: torch.Tensor,
        num_shifts: int,
        ball_t: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Test-time augmentation via temporal shifts.

        Shifts features by ±1..num_shifts windows, runs inference on each,
        un-shifts the probabilities, and averages.
        """
        all_probs: list[np.ndarray] = []

        for shift in range(-num_shifts, num_shifts + 1):
            if shift == 0:
                shifted = features_t
                shifted_ball = ball_t
            else:
                shifted = torch.roll(features_t, shift, dims=2)
                # Zero out wrapped-around edges
                if shift > 0:
                    shifted[:, :, :shift] = 0
                else:
                    shifted[:, :, shift:] = 0
                shifted_ball = None
                if ball_t is not None:
                    shifted_ball = torch.roll(ball_t, shift, dims=2)
                    if shift > 0:
                        shifted_ball[:, :, :shift] = 0
                    else:
                        shifted_ball[:, :, shift:] = 0

            logits = self.model(shifted, ball_features=shifted_ball)
            probs = torch.softmax(logits, dim=1)
            rally_p = probs[0, 1].cpu().numpy()

            # Un-shift
            if shift != 0:
                rally_p = np.roll(rally_p, -shift)
                if shift > 0:
                    rally_p[-shift:] = 0
                else:
                    rally_p[: abs(shift)] = 0

            all_probs.append(rally_p)

        avg: np.ndarray = np.mean(all_probs, axis=0)
        return avg

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
        *,
        valley_threshold: float = 0.0,
        min_valley_duration: float = 2.0,
    ) -> list[tuple[float, float]]:
        """Convert window predictions to time segments with cleanup.

        Post-processing stages:
        - Fill short gaps
        - Remove short segments
        - Confidence filter (remove low-confidence segments)
        - Valley splitting (split at sustained low-probability regions)
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

        # Valley splitting: split at sustained low-probability regions
        if valley_threshold > 0 and min_valley_duration > 0:
            filtered = self._split_at_valleys(
                filtered, probs, window_duration,
                valley_threshold, min_valley_duration, min_duration,
            )

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

    def _rescue_short_rallies(
        self,
        segments: list[tuple[float, float]],
        probs: np.ndarray,
        window_duration: float,
        rescue_min_avg_prob: float,
        rescue_min_windows: int,
        rescue_min_duration: float,
        rescue_max_duration: float,
    ) -> list[tuple[float, float]]:
        """Rescue short rallies missed by the primary detection pass.

        Scans gaps between detected segments for contiguous windows where
        the average probability exceeds rescue_min_avg_prob. These are
        short rallies (aces, serves-to-net) where the model has weak but
        present signal that didn't reach the primary threshold.
        """
        total_time = len(probs) * window_duration
        rescued: list[tuple[float, float]] = []

        # Build gap regions (time ranges not covered by detected segments)
        gaps: list[tuple[float, float]] = []
        prev_end = 0.0
        for s, e in sorted(segments):
            if s > prev_end:
                gaps.append((prev_end, s))
            prev_end = max(prev_end, e)
        if prev_end < total_time:
            gaps.append((prev_end, total_time))

        for gap_s, gap_e in gaps:
            si = max(0, int(gap_s / window_duration))
            ei = min(len(probs), int(gap_e / window_duration) + 1)
            gap_probs = probs[si:ei]
            if len(gap_probs) == 0:
                continue

            # Per-window threshold matches avg threshold — each window in
            # the run must individually meet the bar.
            window_thresh = rescue_min_avg_prob

            in_run = False
            run_start = 0
            for i in range(len(gap_probs)):
                if gap_probs[i] >= window_thresh and not in_run:
                    run_start = i
                    in_run = True
                elif gap_probs[i] < window_thresh and in_run:
                    self._check_rescue_run(
                        gap_probs, run_start, i, si, window_duration,
                        rescue_min_avg_prob, rescue_min_windows,
                        rescue_min_duration, rescue_max_duration, rescued,
                    )
                    in_run = False
            if in_run:
                self._check_rescue_run(
                    gap_probs, run_start, len(gap_probs), si, window_duration,
                    rescue_min_avg_prob, rescue_min_windows,
                    rescue_min_duration, rescue_max_duration, rescued,
                )

        return rescued

    @staticmethod
    def _check_rescue_run(
        gap_probs: np.ndarray,
        run_start: int,
        run_end: int,
        gap_si: int,
        window_duration: float,
        min_avg_prob: float,
        min_windows: int,
        min_duration: float,
        max_duration: float,
        rescued: list[tuple[float, float]],
    ) -> None:
        """Check if a contiguous run qualifies as a rescued short rally."""
        run_len = run_end - run_start
        if run_len < min_windows:
            return
        run_probs = gap_probs[run_start:run_end]
        avg_prob = float(run_probs.mean())
        if avg_prob < min_avg_prob:
            return
        cand_s = (gap_si + run_start) * window_duration
        cand_e = (gap_si + run_end) * window_duration
        duration = cand_e - cand_s
        if min_duration <= duration <= max_duration:
            rescued.append((cand_s, cand_e))

    def _split_at_valleys(
        self,
        segments: list[tuple[float, float]],
        probs: np.ndarray,
        window_duration: float,
        valley_threshold: float,
        min_valley_duration: float,
        min_segment_duration: float,
    ) -> list[tuple[float, float]]:
        """Split segments at sustained low-probability valleys.

        For each segment, find contiguous runs of windows where prob < valley_threshold.
        If a run spans >= min_valley_duration seconds, remove that region and split
        the segment. Keep resulting sub-segments >= min_segment_duration.
        """
        min_valley_windows = max(1, int(min_valley_duration / window_duration))
        result: list[tuple[float, float]] = []

        for seg_start, seg_end in segments:
            start_idx = int(seg_start / window_duration)
            end_idx = min(int(seg_end / window_duration) + 1, len(probs))

            if end_idx <= start_idx:
                result.append((seg_start, seg_end))
                continue

            # Find contiguous valley runs within this segment
            valleys: list[tuple[int, int]] = []  # (start_idx, end_idx) of valleys
            valley_start: int | None = None

            for i in range(start_idx, end_idx):
                if probs[i] < valley_threshold:
                    if valley_start is None:
                        valley_start = i
                else:
                    if valley_start is not None:
                        valley_len = i - valley_start
                        if valley_len >= min_valley_windows:
                            valleys.append((valley_start, i))
                        valley_start = None

            # Close trailing valley
            if valley_start is not None:
                valley_len = end_idx - valley_start
                if valley_len >= min_valley_windows:
                    valleys.append((valley_start, end_idx))

            if not valleys:
                result.append((seg_start, seg_end))
                continue

            # Split segment around valleys
            sub_start = seg_start
            for v_start, v_end in valleys:
                sub_end = v_start * window_duration
                if sub_end - sub_start >= min_segment_duration:
                    result.append((sub_start, sub_end))
                sub_start = v_end * window_duration

            # Trailing sub-segment after last valley
            if seg_end - sub_start >= min_segment_duration:
                result.append((sub_start, seg_end))

        return result

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
