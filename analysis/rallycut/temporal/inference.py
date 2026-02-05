"""Temporal model inference for rally detection.

Provides functions for:
- Running temporal models on cached features
- Boundary refinement using fine-stride features
- Anti-overmerge constraints
- Integration with the existing rally detection pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    import torch.nn as nn

    from rallycut.temporal.deterministic_decoder import DecoderConfig, DecoderResult
    from rallycut.temporal.features import FeatureMetadata


@dataclass
class BoundaryRefinementConfig:
    """Configuration for boundary refinement.

    Start boundaries use earliest-threshold-crossing for precision.
    End boundaries use last-threshold-crossing for completeness.
    Extension is capped to prevent overmerge.

    Disabled by default as evaluation shows minimal improvement over
    coarse boundaries from the deterministic decoder.
    """

    enabled: bool = False
    search_window_seconds: float = 2.0  # Search window around boundary
    smoothing_sigma: float = 1.0  # Gaussian smoothing sigma
    start_threshold: float = 0.5  # Threshold for detecting rally (aligned with decoder)
    max_extension_seconds: float = 0.5  # Max allowed extension beyond coarse boundary


@dataclass
class TemporalInferenceConfig:
    """Configuration for temporal model inference."""

    # Model
    model_version: str = "v2"
    model_path: Path | None = None

    # Stride/Refinement
    fine_stride: int = 16  # Must match cached fine features
    coarse_stride: int = 48
    boundary_refinement: BoundaryRefinementConfig = field(
        default_factory=BoundaryRefinementConfig
    )

    # Anti-overmerge
    max_rally_duration_seconds: float = 60.0
    internal_no_rally_threshold: float = 0.4
    consecutive_no_rally_windows: int = 2
    forced_split_multiplier: float = 1.5

    # Device
    device: str = "cpu"


@dataclass
class RallySegment:
    """A detected rally segment with confidence."""

    start_time: float
    end_time: float
    confidence: float
    start_refined: bool = False
    end_refined: bool = False

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


def detect_model_version(model_path: Path) -> str | None:
    """Detect model version from checkpoint metadata.

    Args:
        model_path: Path to the model checkpoint.

    Returns:
        Model version string (v1, v2, v3) if found in metadata, None otherwise.
    """
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    version = state_dict.get("_model_version")
    return str(version) if version is not None else None


def load_temporal_model(
    model_path: Path,
    model_version: str | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Load a trained temporal model.

    Auto-detects model version from checkpoint metadata if not specified.
    Falls back to v1 if no version is specified or detected.

    Args:
        model_path: Path to the model checkpoint.
        model_version: Model version (v1, v2, v3). Auto-detected if None.
        device: Device to load model on.

    Returns:
        Loaded temporal model in eval mode.
    """
    from rallycut.temporal.models import get_temporal_model

    # Load weights first to check for metadata
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Auto-detect version from metadata if not specified
    if model_version is None:
        model_version = state_dict.pop("_model_version", None)
    else:
        # Remove metadata key if present to avoid load_state_dict error
        state_dict.pop("_model_version", None)

    # Default to v1 if no version detected (legacy models)
    if model_version is None:
        model_version = "v1"

    # Extract pos_weight for v1 models (stored as buffer)
    pos_weight = None
    if model_version == "v1" and "pos_weight" in state_dict:
        pw_tensor = state_dict.get("pos_weight")
        if pw_tensor is not None:
            pos_weight = float(pw_tensor.item())

    # Create model architecture with matching parameters
    model = get_temporal_model(model_version, feature_dim=768, pos_weight=pos_weight)

    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def run_inference(
    model: nn.Module,
    features: np.ndarray,
    device: str = "cpu",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> tuple[list[int], list[float]]:
    """Run inference on features with chunking for long videos.

    Args:
        model: Trained temporal model.
        features: Feature array of shape (num_windows, feature_dim).
        device: Device for inference.
        chunk_size: Max windows per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Tuple of (predictions, probabilities).
    """
    num_windows = len(features)

    if num_windows == 0:
        return [], []

    # Short video: process in one pass
    if num_windows <= chunk_size:
        features_t = torch.from_numpy(features).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # All models (v1/v2/v3) accept full features
            # v1 has internal projection (768 -> 2)
            output = model(features_t)

            preds = output["predictions"].squeeze(0).cpu().numpy()

            # Get probabilities if available
            if "probs" in output:
                probs = output["probs"].squeeze(0).cpu().numpy()
            elif "emissions" in output:
                emissions = output["emissions"].squeeze(0)
                probs = torch.softmax(emissions, dim=-1)[:, 1].cpu().numpy()
            else:
                probs = np.zeros(len(preds))

        return preds.tolist(), probs.tolist()

    # Long video: process in overlapping chunks
    all_probs = np.zeros(num_windows)
    all_counts = np.zeros(num_windows)
    all_preds = np.zeros(num_windows)

    for start in range(0, num_windows, chunk_size - chunk_overlap):
        end = min(start + chunk_size, num_windows)
        chunk_features = features[start:end]
        chunk_features_t = torch.from_numpy(chunk_features).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # All models accept full features
            output = model(chunk_features_t)

            chunk_preds = output["predictions"].squeeze(0).cpu().numpy()

            if "probs" in output:
                chunk_probs = output["probs"].squeeze(0).cpu().numpy()
            elif "emissions" in output:
                emissions = output["emissions"].squeeze(0)
                chunk_probs = torch.softmax(emissions, dim=-1)[:, 1].cpu().numpy()
            else:
                chunk_probs = np.zeros(len(chunk_preds))

        # Accumulate (overlapping regions get averaged)
        all_probs[start:end] += chunk_probs
        all_preds[start:end] += chunk_preds
        all_counts[start:end] += 1

    # Average overlapping regions
    all_probs = all_probs / all_counts
    avg_preds = all_preds / all_counts
    final_preds = (avg_preds > 0.5).astype(int)

    return final_preds.tolist(), all_probs.tolist()


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


def _refine_start_boundary(
    fine_probs: np.ndarray,
    coarse_boundary_time: float,
    fps: float,
    fine_stride: int,
    config: BoundaryRefinementConfig,
) -> tuple[float, bool]:
    """Refine start boundary using earliest-threshold-crossing.

    Searches for the first threshold crossing, allowing limited extension
    earlier than coarse boundary (capped by max_extension_seconds).

    Args:
        fine_probs: Array of fine-stride probabilities.
        coarse_boundary_time: Coarse boundary time in seconds.
        fps: Video frames per second.
        fine_stride: Fine stride in frames.
        config: Boundary refinement configuration.

    Returns:
        Tuple of (refined_time, was_refined).
    """
    window_duration = fine_stride / fps
    boundary_idx = int(coarse_boundary_time / window_duration)
    search_radius = int(config.search_window_seconds / window_duration)
    max_extension_windows = int(config.max_extension_seconds / window_duration)

    # Search window: limited extension before, full search after
    start_idx = max(0, boundary_idx - max_extension_windows)
    end_idx = min(len(fine_probs), boundary_idx + search_radius)

    segment = fine_probs[start_idx:end_idx]
    if len(segment) < 2:
        return coarse_boundary_time, False

    # Smooth to reduce noise
    smoothed = gaussian_filter1d(segment, sigma=config.smoothing_sigma)

    # Find earliest threshold crossing
    for i, prob in enumerate(smoothed):
        if prob >= config.start_threshold:
            refined_idx = start_idx + i
            refined_time = refined_idx * window_duration
            if refined_time != coarse_boundary_time:
                return refined_time, True
            return coarse_boundary_time, False

    return coarse_boundary_time, False


def _refine_end_boundary(
    fine_probs: np.ndarray,
    coarse_boundary_time: float,
    fps: float,
    fine_stride: int,
    config: BoundaryRefinementConfig,
) -> tuple[float, bool]:
    """Refine end boundary using last-threshold-crossing.

    Searches for the last threshold crossing, allowing limited extension
    later than coarse boundary (capped by max_extension_seconds).

    Args:
        fine_probs: Array of fine-stride probabilities.
        coarse_boundary_time: Coarse boundary time in seconds.
        fps: Video frames per second.
        fine_stride: Fine stride in frames.
        config: Boundary refinement configuration.

    Returns:
        Tuple of (refined_time, was_refined).
    """
    window_duration = fine_stride / fps
    boundary_idx = int(coarse_boundary_time / window_duration)
    search_radius = int(config.search_window_seconds / window_duration)
    max_extension_windows = int(config.max_extension_seconds / window_duration)

    # Search window: full search before, limited extension after
    start_idx = max(0, boundary_idx - search_radius)
    end_idx = min(len(fine_probs), boundary_idx + max_extension_windows + 1)

    segment = fine_probs[start_idx:end_idx]
    if len(segment) < 2:
        return coarse_boundary_time, False

    # Smooth to reduce noise
    smoothed = gaussian_filter1d(segment, sigma=config.smoothing_sigma)

    # Find last threshold crossing (search backwards)
    for i in range(len(smoothed) - 1, -1, -1):
        if smoothed[i] >= config.start_threshold:
            refined_idx = start_idx + i + 1  # +1 for end boundary
            refined_time = refined_idx * window_duration
            if refined_time != coarse_boundary_time:
                return refined_time, True
            return coarse_boundary_time, False

    return coarse_boundary_time, False


def refine_boundaries(
    segments: list[RallySegment],
    fine_probs: list[float],
    coarse_probs: list[float],
    fps: float,
    fine_stride: int,
    coarse_stride: int,
    config: BoundaryRefinementConfig | None = None,
) -> list[RallySegment]:
    """Refine segment boundaries using fine-stride probabilities.

    Start boundaries use earliest-threshold-crossing for precision.
    End boundaries use last-threshold-crossing for completeness.
    Both use Gaussian smoothing to reduce noise.

    Args:
        segments: List of segments detected at coarse stride.
        fine_probs: Probabilities at fine stride.
        coarse_probs: Probabilities at coarse stride (unused, kept for API compat).
        fps: Video frames per second.
        fine_stride: Fine stride in frames.
        coarse_stride: Coarse stride in frames (unused, kept for API compat).
        config: Boundary refinement configuration. Uses defaults if None.

    Returns:
        List of segments with refined boundaries.
    """
    if config is None:
        config = BoundaryRefinementConfig()

    fine_probs_arr = np.array(fine_probs)
    refined: list[RallySegment] = []

    for segment in segments:
        # Refine start boundary (earliest threshold crossing)
        refined_start, start_was_refined = _refine_start_boundary(
            fine_probs_arr,
            segment.start_time,
            fps,
            fine_stride,
            config,
        )

        # Refine end boundary (last threshold crossing)
        refined_end, end_was_refined = _refine_end_boundary(
            fine_probs_arr,
            segment.end_time,
            fps,
            fine_stride,
            config,
        )

        # Ensure valid segment (min 0.5 second)
        if refined_end > refined_start + 0.5:
            refined.append(
                RallySegment(
                    start_time=refined_start,
                    end_time=refined_end,
                    confidence=segment.confidence,
                    start_refined=start_was_refined,
                    end_refined=end_was_refined,
                )
            )
        else:
            # Keep original if refinement failed
            refined.append(segment)

    return refined


def load_binary_head_model(model_path: Path, device: str = "cpu") -> nn.Module:
    """Load trained binary head model.

    Args:
        model_path: Path to the model checkpoint.
        device: Device to load model on.

    Returns:
        Loaded BinaryHead model in eval mode.
    """
    from rallycut.temporal.binary_head import BinaryHead

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model = BinaryHead(feature_dim=768, hidden_dim=128, dropout=0.0)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_binary_head_decoder(
    features: np.ndarray,
    model: nn.Module,
    config: DecoderConfig,
    device: str = "cpu",
) -> DecoderResult:
    """Run binary head + deterministic decoder pipeline.

    Args:
        features: Feature array of shape (num_windows, feature_dim).
        model: Trained BinaryHead model.
        config: Decoder configuration.
        device: Device for inference.

    Returns:
        DecoderResult with segments and probabilities.
    """
    from rallycut.temporal.deterministic_decoder import decode

    features_t = torch.from_numpy(features).float().to(device)
    with torch.no_grad():
        logits = model(features_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    return decode(probs, config)


def run_temporal_inference(
    features: np.ndarray,
    metadata: FeatureMetadata,
    model: nn.Module,
    config: TemporalInferenceConfig,
    fine_features: np.ndarray | None = None,
) -> TemporalInferenceResult:
    """Run full temporal inference pipeline.

    Args:
        features: Coarse-stride features (num_windows, 768).
        metadata: Feature metadata.
        model: Trained temporal model.
        config: Inference configuration.
        fine_features: Optional fine-stride features for boundary refinement.

    Returns:
        Inference result with segments and probabilities.
    """

    fps = metadata.fps

    # Run coarse inference
    predictions, probs = run_inference(
        model,
        features,
        device=config.device,
    )

    # Extract segments
    segments = extract_segments_from_predictions(
        predictions,
        probs,
        fps,
        config.coarse_stride,
    )

    # Apply anti-overmerge
    segments = apply_anti_overmerge_segments(
        segments,
        probs,
        fps,
        config.coarse_stride,
        config,
    )

    # Boundary refinement (if enabled and fine features available)
    if config.boundary_refinement.enabled and fine_features is not None:
        # Run inference on fine features
        fine_preds, fine_probs = run_inference(
            model,
            fine_features,
            device=config.device,
        )

        segments = refine_boundaries(
            segments,
            fine_probs,
            probs,
            fps,
            config.fine_stride,
            config.coarse_stride,
            config.boundary_refinement,
        )

    return TemporalInferenceResult(
        segments=segments,
        window_probs=probs,
        window_predictions=predictions,
        fps=fps,
        stride=config.coarse_stride,
    )
