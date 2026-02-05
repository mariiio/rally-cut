"""Boundary refinement for rally detection using fine-stride features.

Refines segment boundaries detected at coarse stride (48) by running
binary head inference at fine stride (16) and finding transition points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class BoundaryRefinementConfig:
    """Configuration for boundary refinement.

    Note: Boundary refinement is disabled by default because the binary head
    model was trained on coarse-stride (48) features and doesn't accurately
    detect boundaries at fine stride (16). The baseline MAE of ~1.5s is
    acceptable for most use cases.

    Attributes:
        enabled: Whether boundary refinement is enabled.
        search_window_seconds: How far (in seconds) to search around each boundary.
        fine_stride: Frame stride for fine features (should be lower than coarse).
        smoothing_sigma: Gaussian smoothing sigma for probability smoothing.
        transition_threshold: Probability threshold for detecting transitions.
        min_segment_duration: Minimum segment duration after refinement (seconds).
    """

    enabled: bool = False  # Disabled by default - requires boundary-specific training
    search_window_seconds: float = 2.0
    fine_stride: int = 16
    smoothing_sigma: float = 1.0
    transition_threshold: float = 0.5
    min_segment_duration: float = 1.0


def _gaussian_smooth(probs: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to probability array.

    Args:
        probs: 1D array of probabilities.
        sigma: Gaussian kernel sigma (in window units).

    Returns:
        Smoothed probability array.
    """
    if sigma <= 0 or len(probs) < 3:
        return probs

    # Create Gaussian kernel
    kernel_size = int(sigma * 6) | 1  # Ensure odd kernel size
    kernel_size = max(3, min(kernel_size, len(probs)))
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Convolve with reflect padding
    padded = np.pad(probs, kernel_size // 2, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")

    return smoothed[: len(probs)]


def _find_transition_point(
    probs: np.ndarray,
    start_idx: int,
    end_idx: int,
    rising: bool,
    threshold: float,
) -> int | None:
    """Find transition point in probability array.

    Args:
        probs: Smoothed probability array.
        start_idx: Start of search window.
        end_idx: End of search window.
        rising: True for rising edge (rally start), False for falling edge (rally end).
        threshold: Probability threshold for transition.

    Returns:
        Index of transition point, or None if not found.
    """
    start_idx = max(0, start_idx)
    end_idx = min(len(probs), end_idx)

    if start_idx >= end_idx:
        return None

    window = probs[start_idx:end_idx]

    if rising:
        # Find first index where prob crosses threshold from below
        below = window < threshold
        above = window >= threshold
        # Look for transition from below to above
        for i in range(len(window) - 1):
            if below[i] and above[i + 1]:
                return start_idx + i + 1
        # Fallback: find first index above threshold
        if np.any(above):
            return int(start_idx + np.argmax(above))
    else:
        # Find first index where prob crosses threshold from above
        above = window >= threshold
        below = window < threshold
        # Look for transition from above to below
        for i in range(len(window) - 1):
            if above[i] and below[i + 1]:
                return start_idx + i
        # Fallback: find first index below threshold (don't extend)
        if np.any(below):
            return int(start_idx + np.argmax(below))

    return None


def run_binary_head_on_features(
    features: np.ndarray,
    model: nn.Module,
    device: str = "cpu",
) -> np.ndarray:
    """Run binary head inference on features to get probabilities.

    Args:
        features: Feature array of shape (num_windows, feature_dim).
        model: Trained BinaryHead model.
        device: Device for inference.

    Returns:
        Array of rally probabilities for each window.
    """
    features_t = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        logits = model(features_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs.squeeze()


def refine_boundaries(
    segments: list[tuple[float, float]],
    fine_features: np.ndarray,
    model: nn.Module,
    fps: float,
    config: BoundaryRefinementConfig,
    device: str = "cpu",
) -> list[tuple[float, float]]:
    """Refine segment boundaries using fine-stride binary head probabilities.

    For each segment boundary, searches a window around the coarse boundary
    to find the precise transition point using fine-stride features.

    Args:
        segments: List of (start_time, end_time) tuples from coarse detection.
        fine_features: Features extracted at fine stride (stride 16).
        model: Trained BinaryHead model.
        fps: Video frames per second.
        config: Boundary refinement configuration.
        device: Device for inference.

    Returns:
        List of refined (start_time, end_time) tuples.

    Raises:
        ValueError: If fps or fine_stride is non-positive.
    """
    # Validate parameters to prevent division by zero
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if config.fine_stride <= 0:
        raise ValueError(f"fine_stride must be positive, got {config.fine_stride}")

    if not segments or not config.enabled:
        return segments

    # Check for empty features
    if len(fine_features) == 0:
        return segments

    # Run binary head on fine features
    fine_probs = run_binary_head_on_features(fine_features, model, device)

    # Check for empty probabilities
    if len(fine_probs) == 0:
        return segments

    # Apply Gaussian smoothing
    smoothed_probs = _gaussian_smooth(fine_probs, config.smoothing_sigma)

    # Calculate window duration at fine stride
    window_duration = config.fine_stride / fps
    search_windows = int(config.search_window_seconds / window_duration)

    refined_segments: list[tuple[float, float]] = []

    for start_time, end_time in segments:
        # Convert times to fine-stride window indices
        coarse_start_idx = int(start_time / window_duration)
        coarse_end_idx = int(end_time / window_duration)

        # Refine start boundary (look for rising edge)
        refined_start_idx = _find_transition_point(
            smoothed_probs,
            coarse_start_idx - search_windows,
            coarse_start_idx + search_windows,
            rising=True,
            threshold=config.transition_threshold,
        )
        if refined_start_idx is None:
            refined_start_idx = coarse_start_idx

        # Refine end boundary (look for falling edge)
        refined_end_idx = _find_transition_point(
            smoothed_probs,
            coarse_end_idx - search_windows,
            coarse_end_idx + search_windows,
            rising=False,
            threshold=config.transition_threshold,
        )
        if refined_end_idx is None:
            refined_end_idx = coarse_end_idx

        # Convert back to times
        refined_start_time = max(0.0, refined_start_idx * window_duration)
        refined_end_time = refined_end_idx * window_duration

        # Ensure minimum segment duration
        if refined_end_time - refined_start_time >= config.min_segment_duration:
            refined_segments.append((refined_start_time, refined_end_time))
        elif end_time - start_time >= config.min_segment_duration:
            # Keep original if refinement made segment too short
            refined_segments.append((start_time, end_time))

    return refined_segments
