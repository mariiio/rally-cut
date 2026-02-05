"""Temporal model processor for rally detection.

Integrates temporal models into the main inference pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rallycut.core.models import TimeSegment
from rallycut.temporal.features import (
    FeatureCache,
    FeatureMetadata,
    extract_features_for_video,
)
from rallycut.temporal.inference import (
    BoundaryRefinementConfig,
    RallySegment,
    TemporalInferenceConfig,
    TemporalInferenceResult,
    load_temporal_model,
    run_temporal_inference,
)

if TYPE_CHECKING:
    import torch.nn as nn

    from lib.volleyball_ml.video_mae import GameStateClassifier
    from rallycut.core.video import Video


@dataclass
class TemporalProcessorConfig:
    """Configuration for temporal processor."""

    # Model
    model_path: Path | None = None
    model_version: str = "v2"

    # Feature extraction
    feature_cache_dir: Path | None = None
    fine_stride: int = 16  # Must match cached fine features
    coarse_stride: int = 48

    # Inference
    boundary_refinement: BoundaryRefinementConfig = field(
        default_factory=BoundaryRefinementConfig
    )
    max_rally_duration_seconds: float = 60.0

    # Device
    device: str = "cpu"


class TemporalProcessor:
    """Processes video using temporal models for rally detection.

    Provides an alternative to heuristic-based post-processing
    using learned temporal models (ConvCRF, BiLSTMCRF).
    """

    def __init__(self, config: TemporalProcessorConfig | None = None):
        """Initialize temporal processor.

        Args:
            config: Processor configuration. Uses defaults if None.
        """
        from rallycut.core.config import get_config

        self.config = config or TemporalProcessorConfig()

        # Set defaults from global config
        global_config = get_config()
        if self.config.feature_cache_dir is None:
            self.config.feature_cache_dir = global_config.feature_cache_dir
        if self.config.model_path is None:
            # Default to temporal model in weights directory
            self.config.model_path = (
                global_config.weights_dir / "temporal" / "best_temporal_model.pt"
            )

        self._model: nn.Module | None = None
        self._feature_cache: FeatureCache | None = None
        self._classifier: GameStateClassifier | None = None

    def _get_model(self) -> nn.Module:
        """Lazy load temporal model."""
        if self._model is None:
            if self.config.model_path is None or not self.config.model_path.exists():
                raise FileNotFoundError(
                    f"Temporal model not found at {self.config.model_path}. "
                    "Train one with: uv run rallycut train temporal"
                )
            self._model = load_temporal_model(
                self.config.model_path,
                model_version=self.config.model_version,
                device=self.config.device,
            )
        return self._model

    def _get_feature_cache(self) -> FeatureCache:
        """Lazy load feature cache.

        Returns:
            FeatureCache instance for storing/retrieving extracted features.
        """
        if self._feature_cache is None:
            self._feature_cache = FeatureCache(cache_dir=self.config.feature_cache_dir)
        return self._feature_cache

    def _get_classifier(self) -> GameStateClassifier:
        """Lazy load VideoMAE classifier for feature extraction.

        Returns:
            GameStateClassifier instance for extracting encoder features.
        """
        if self._classifier is None:
            from lib.volleyball_ml.video_mae import GameStateClassifier

            self._classifier = GameStateClassifier(device=self.config.device)
        return self._classifier

    def _create_inference_config(self) -> TemporalInferenceConfig:
        """Create inference configuration from processor config.

        Maps processor config fields to inference config format.

        Returns:
            TemporalInferenceConfig with settings for model inference.
        """
        return TemporalInferenceConfig(
            model_version=self.config.model_version,
            model_path=self.config.model_path,
            fine_stride=self.config.fine_stride,
            coarse_stride=self.config.coarse_stride,
            boundary_refinement=self.config.boundary_refinement,
            max_rally_duration_seconds=self.config.max_rally_duration_seconds,
            device=self.config.device,
        )

    def _run_inference(
        self,
        coarse_features: np.ndarray,
        metadata: FeatureMetadata,
        fine_features: np.ndarray | None,
    ) -> TemporalInferenceResult:
        """Run temporal model inference on features."""
        model = self._get_model()
        inference_config = self._create_inference_config()

        return run_temporal_inference(
            features=coarse_features,
            metadata=metadata,
            model=model,
            config=inference_config,
            fine_features=fine_features,
        )

    def _load_fine_features(self, cache: FeatureCache, content_hash: str) -> np.ndarray | None:
        """Load fine-stride features for boundary refinement if enabled."""
        if not self.config.boundary_refinement.enabled:
            return None
        cached_fine = cache.get(content_hash, self.config.fine_stride)
        if cached_fine is not None:
            return cached_fine[0]
        return None

    def process_video(
        self,
        video: Video,
        content_hash: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TemporalInferenceResult:
        """Process video using temporal model.

        Args:
            video: Video to process.
            content_hash: Video content hash for caching. Auto-computed if None.
            progress_callback: Progress callback (0-1 percentage, message).

        Returns:
            Inference result with rally segments.
        """
        cache = self._get_feature_cache()
        coarse_stride = self.config.coarse_stride

        # Compute content hash if not provided
        if content_hash is None:
            content_hash = video.compute_content_hash()

        # Check cache for coarse features
        cached_coarse = cache.get(content_hash, coarse_stride)
        if cached_coarse is not None:
            coarse_features, metadata = cached_coarse
        else:
            # Extract features at coarse stride
            if progress_callback:
                progress_callback(0.0, "Extracting features...")

            classifier = self._get_classifier()
            coarse_features, metadata = extract_features_for_video(
                video_path=video.path,
                classifier=classifier,
                stride=coarse_stride,
            )

            # Cache for future use
            cache.put(content_hash, coarse_stride, coarse_features, metadata)

            if progress_callback:
                progress_callback(0.5, "Features extracted")

        # Load fine features for boundary refinement
        fine_features = self._load_fine_features(cache, content_hash)

        # Run temporal model inference
        if progress_callback:
            progress_callback(0.6, "Running temporal model...")

        result = self._run_inference(coarse_features, metadata, fine_features)

        if progress_callback:
            progress_callback(1.0, "Temporal inference complete")

        return result

    def process_from_cache(
        self,
        content_hash: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TemporalInferenceResult:
        """Process video using cached features.

        Args:
            content_hash: Video content hash.
            progress_callback: Progress callback.

        Returns:
            Inference result with rally segments.

        Raises:
            ValueError: If no cached features found.
        """
        cache = self._get_feature_cache()

        # Load coarse features
        cached_coarse = cache.get(content_hash, self.config.coarse_stride)
        if cached_coarse is None:
            raise ValueError(
                f"No cached features found for {content_hash} at stride {self.config.coarse_stride}"
            )

        coarse_features, metadata = cached_coarse

        # Load fine features for boundary refinement
        fine_features = self._load_fine_features(cache, content_hash)

        # Run temporal model inference
        if progress_callback:
            progress_callback(0.0, "Running temporal model...")

        result = self._run_inference(coarse_features, metadata, fine_features)

        if progress_callback:
            progress_callback(1.0, "Temporal inference complete")

        return result


def rally_segments_to_time_segments(
    rally_segments: list[RallySegment],
    fps: float,
    padding_start: float = 0.5,
    padding_end: float = 1.0,
    total_frames: int | None = None,
) -> list[TimeSegment]:
    """Convert RallySegments to TimeSegments with padding.

    Args:
        rally_segments: Rally segments from temporal model.
        fps: Video frames per second.
        padding_start: Seconds of padding before rally start.
        padding_end: Seconds of padding after rally end.
        total_frames: Total video frames (for clamping).

    Returns:
        List of TimeSegments suitable for video cutting.
    """
    from rallycut.core.models import GameState

    segments: list[TimeSegment] = []

    for rally in rally_segments:
        # Apply padding
        start_time = max(0.0, rally.start_time - padding_start)
        end_time = rally.end_time + padding_end

        # Convert to frames
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Clamp to video bounds
        if total_frames is not None:
            start_frame = min(start_frame, total_frames - 1)
            end_frame = min(end_frame, total_frames - 1)

        if end_frame > start_frame:
            segments.append(
                TimeSegment(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_frame / fps,
                    end_time=end_frame / fps,
                    state=GameState.PLAY,
                )
            )

    # Merge overlapping segments
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda s: s.start_frame)
    merged = [sorted_segments[0]]

    for segment in sorted_segments[1:]:
        last = merged[-1]
        # Merge if segments overlap or are adjacent
        if segment.start_frame <= last.end_frame + 1:
            merged[-1] = TimeSegment(
                start_frame=last.start_frame,
                end_frame=max(last.end_frame, segment.end_frame),
                start_time=last.start_time,
                end_time=max(last.end_frame, segment.end_frame) / fps,
                state=last.state,
            )
        else:
            merged.append(segment)

    return merged
