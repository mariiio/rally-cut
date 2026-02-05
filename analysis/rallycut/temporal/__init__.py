"""Temporal modeling for rally detection.

This module provides:
- Feature extraction from VideoMAE encoder
- Binary head classifier with deterministic decoder (recommended, 80% F1)
- Boundary refinement using fine-stride features
- Feature caching for efficient training and inference
- Training pipeline for temporal models
- Inference with anti-overmerge constraints
- Integration with main video processing pipeline

**Deprecated**: Temporal models (v1/v2/v3) are deprecated in favor of
Binary Head + Deterministic Decoder. They remain available via
``--experimental-temporal`` flag but are no longer recommended.
"""

from rallycut.temporal.binary_head import (
    BinaryHead,
    BinaryHeadConfig,
    BinaryHeadWithSmoothing,
    SmoothingConfig,
    SmoothingResult,
)
from rallycut.temporal.boundary_refinement import (
    BoundaryRefinementConfig,
    refine_boundaries,
)
from rallycut.temporal.deterministic_decoder import (
    DecoderConfig,
    DecoderResult,
    GridSearchResult,
    compute_boundary_errors,
    compute_overmerge_rate,
    compute_segment_metrics,
    decode,
    grid_search,
)
from rallycut.temporal.features import (
    FeatureCache,
    extract_features_for_video,
    load_cached_features,
)
from rallycut.temporal.inference import (
    RallySegment,
    TemporalInferenceConfig,
    TemporalInferenceResult,
    apply_anti_overmerge_segments,
    load_binary_head_model,
    load_temporal_model,
    run_binary_head_decoder,
    run_inference,
    run_temporal_inference,
)
from rallycut.temporal.models import (
    TEMPORAL_MODELS,
    BiLSTMCRF,
    ConvCRF,
    LearnedSmoothing,
    get_temporal_model,
)
from rallycut.temporal.processor import (
    TemporalProcessor,
    TemporalProcessorConfig,
    rally_segments_to_time_segments,
)
from rallycut.temporal.training import (
    SequenceDataset,
    TemporalTrainingConfig,
    TemporalTrainingResult,
    prepare_training_data,
    train_temporal_model,
    video_level_split,
)

__all__ = [
    # Boundary refinement
    "BoundaryRefinementConfig",
    "refine_boundaries",
    # Binary head
    "BinaryHead",
    "BinaryHeadConfig",
    "BinaryHeadWithSmoothing",
    "SmoothingConfig",
    "SmoothingResult",
    # Deterministic decoder
    "DecoderConfig",
    "DecoderResult",
    "GridSearchResult",
    "compute_boundary_errors",
    "compute_overmerge_rate",
    "compute_segment_metrics",
    "decode",
    "grid_search",
    # Features
    "FeatureCache",
    "extract_features_for_video",
    "load_cached_features",
    # Models
    "TEMPORAL_MODELS",
    "LearnedSmoothing",
    "ConvCRF",
    "BiLSTMCRF",
    "get_temporal_model",
    # Training
    "SequenceDataset",
    "TemporalTrainingConfig",
    "TemporalTrainingResult",
    "prepare_training_data",
    "train_temporal_model",
    "video_level_split",
    # Inference
    "RallySegment",
    "TemporalInferenceConfig",
    "TemporalInferenceResult",
    "load_binary_head_model",
    "load_temporal_model",
    "run_binary_head_decoder",
    "run_inference",
    "run_temporal_inference",
    "apply_anti_overmerge_segments",
    # Processor (pipeline integration)
    "TemporalProcessor",
    "TemporalProcessorConfig",
    "rally_segments_to_time_segments",
]
