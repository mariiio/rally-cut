"""Ball tracking module for automatic camera keyframe generation."""

from rallycut.tracking.ball_boundary import (
    BallBoundaryConfig,
    BoundaryRefinement,
    SegmentRefinement,
    find_rally_end_from_ball,
    find_rally_start_from_ball,
    refine_boundaries_parallel,
    refine_segment_boundaries,
)
from rallycut.tracking.ball_cache import BallFeatureCache, get_ball_cache
from rallycut.tracking.ball_features import SegmentBallFeatures, compute_ball_features
from rallycut.tracking.ball_tracker import BallPosition, BallTracker, BallTrackingResult
from rallycut.tracking.ball_validation import (
    BallValidationConfig,
    ValidationResult,
    should_validate_segment,
    validate_segment_with_ball,
    validate_segment_with_early_exit,
    validate_segments_batch,
)

__all__ = [
    # Ball tracker
    "BallTracker",
    "BallPosition",
    "BallTrackingResult",
    # Ball features
    "SegmentBallFeatures",
    "compute_ball_features",
    # Ball validation
    "BallValidationConfig",
    "ValidationResult",
    "should_validate_segment",
    "validate_segment_with_ball",
    "validate_segment_with_early_exit",
    "validate_segments_batch",
    # Ball boundary
    "BallBoundaryConfig",
    "BoundaryRefinement",
    "SegmentRefinement",
    "find_rally_start_from_ball",
    "find_rally_end_from_ball",
    "refine_segment_boundaries",
    "refine_boundaries_parallel",
    # Ball cache
    "BallFeatureCache",
    "get_ball_cache",
]
