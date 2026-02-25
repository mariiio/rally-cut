"""Temporal inference utilities for rally detection.

Provides:
- Data classes for inference configuration and results
- Segment extraction from window predictions
- Anti-overmerge constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TemporalInferenceConfig:
    """Configuration for temporal model inference."""

    # Stride
    coarse_stride: int = 48

    # Anti-overmerge
    max_rally_duration_seconds: float = 60.0
    internal_no_rally_threshold: float = 0.4
    consecutive_no_rally_windows: int = 2
    forced_split_multiplier: float = 1.5


@dataclass
class RallySegment:
    """A detected rally segment with confidence."""

    start_time: float
    end_time: float
    confidence: float

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class TemporalInferenceResult:
    """Result of temporal model inference."""

    segments: list[RallySegment] = field(default_factory=list)
    window_probs: list[float] = field(default_factory=list)
    window_predictions: list[int] = field(default_factory=list)
    fps: float = 30.0
    stride: int = 48


def extract_segments_from_predictions(
    predictions: list[int],
    probs: list[float],
    fps: float,
    stride: int,
    window_size: int = 16,
) -> list[RallySegment]:
    """Extract rally segments from window predictions.

    Args:
        predictions: Per-window binary predictions (0/1).
        probs: Per-window rally probabilities.
        fps: Video frames per second.
        stride: Frame stride between windows.
        window_size: Frames per window.

    Returns:
        List of rally segments.
    """
    if not predictions:
        return []

    segments: list[RallySegment] = []
    window_duration = stride / fps  # Time per window step

    in_rally = False
    rally_start_idx = 0
    rally_probs: list[float] = []

    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        if pred == 1 and not in_rally:
            # Rally starts
            in_rally = True
            rally_start_idx = i
            rally_probs = [prob]
        elif pred == 1 and in_rally:
            # Rally continues
            rally_probs.append(prob)
        elif pred == 0 and in_rally:
            # Rally ends
            in_rally = False
            start_time = rally_start_idx * window_duration
            end_time = i * window_duration + (window_size / fps)
            avg_confidence = sum(rally_probs) / len(rally_probs) if rally_probs else 0.0
            segments.append(
                RallySegment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=avg_confidence,
                )
            )
            rally_probs = []

    # Handle rally at end of video
    if in_rally:
        start_time = rally_start_idx * window_duration
        end_time = len(predictions) * window_duration
        avg_confidence = sum(rally_probs) / len(rally_probs) if rally_probs else 0.0
        segments.append(
            RallySegment(
                start_time=start_time,
                end_time=end_time,
                confidence=avg_confidence,
            )
        )

    return segments


def apply_anti_overmerge_segments(
    segments: list[RallySegment],
    probs: list[float],
    fps: float,
    stride: int,
    config: TemporalInferenceConfig,
) -> list[RallySegment]:
    """Apply anti-overmerge constraints to detected segments.

    Splits segments that:
    1. Exceed maximum duration
    2. Have internal low-confidence regions

    Args:
        segments: List of rally segments.
        probs: Per-window rally probabilities.
        fps: Video frames per second.
        stride: Frame stride between windows.
        config: Inference configuration.

    Returns:
        List of segments with overmerges split.
    """
    window_duration = stride / fps
    result: list[RallySegment] = []

    for segment in segments:
        # Check if segment exceeds max duration
        if segment.duration <= config.max_rally_duration_seconds:
            result.append(segment)
            continue

        # Find split points based on internal low-confidence regions
        start_idx = int(segment.start_time / window_duration)
        end_idx = min(int(segment.end_time / window_duration), len(probs))

        # Look for consecutive low-confidence windows
        split_points: list[int] = []
        consecutive_low = 0

        for i in range(start_idx, end_idx):
            if i < len(probs) and probs[i] < config.internal_no_rally_threshold:
                consecutive_low += 1
                if consecutive_low >= config.consecutive_no_rally_windows:
                    # Mark the middle of the low-confidence region as split point
                    split_idx = i - config.consecutive_no_rally_windows // 2
                    split_points.append(split_idx)
                    consecutive_low = 0
            else:
                consecutive_low = 0

        # If no natural split points found, force split at regular intervals
        if (
            not split_points
            and segment.duration
            > config.max_rally_duration_seconds * config.forced_split_multiplier
        ):
            # Force split at max_duration intervals
            interval_windows = int(config.max_rally_duration_seconds / window_duration)
            for i in range(start_idx + interval_windows, end_idx, interval_windows):
                split_points.append(i)

        # Create split segments
        if split_points:
            prev_start = segment.start_time
            for split_idx in sorted(set(split_points)):
                split_time = split_idx * window_duration
                if split_time > prev_start + 1.0:  # Min 1 second segments
                    # Compute confidence for this sub-segment
                    sub_start_idx = int(prev_start / window_duration)
                    sub_end_idx = split_idx
                    sub_probs = [
                        probs[i] for i in range(sub_start_idx, sub_end_idx) if i < len(probs)
                    ]
                    sub_conf = sum(sub_probs) / len(sub_probs) if sub_probs else segment.confidence

                    result.append(
                        RallySegment(
                            start_time=prev_start,
                            end_time=split_time,
                            confidence=sub_conf,
                        )
                    )
                    prev_start = split_time

            # Add final sub-segment
            if segment.end_time > prev_start + 1.0:
                sub_start_idx = int(prev_start / window_duration)
                sub_probs = [probs[i] for i in range(sub_start_idx, end_idx) if i < len(probs)]
                sub_conf = sum(sub_probs) / len(sub_probs) if sub_probs else segment.confidence

                result.append(
                    RallySegment(
                        start_time=prev_start,
                        end_time=segment.end_time,
                        confidence=sub_conf,
                    )
                )
        else:
            result.append(segment)

    return result
