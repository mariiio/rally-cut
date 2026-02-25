"""Temporal modeling for rally detection.

This module provides:
- Feature extraction from VideoMAE encoder
- TemporalMaxer TAS model for rally detection (91.9% LOO F1 at IoU=0.4)
- Feature caching for efficient training and inference
"""

from rallycut.temporal.features import (
    FeatureCache,
    extract_features_for_video,
    generate_overlap_labels,
    load_cached_features,
    video_level_split,
)

__all__ = [
    "FeatureCache",
    "extract_features_for_video",
    "generate_overlap_labels",
    "load_cached_features",
    "video_level_split",
]
