"""Evaluation framework for rally detection quality assessment."""

from rallycut.evaluation.cached_analysis import (
    AnalysisCache,
    CachedAnalysis,
    PostProcessingParams,
    analyze_and_cache,
    apply_post_processing,
    apply_post_processing_custom,
)
from rallycut.evaluation.ground_truth import (
    EvaluationVideo,
    GroundTruthRally,
    load_evaluation_videos,
)
from rallycut.evaluation.matching import MatchingResult, RallyMatch, match_rallies
from rallycut.evaluation.metrics import (
    AggregateMetrics,
    BoundaryMetrics,
    RallyMetrics,
    VideoEvaluationResult,
    aggregate_metrics,
    compute_metrics,
)
from rallycut.evaluation.param_grid import (
    AVAILABLE_GRIDS,
    DEFAULT_PARAMS,
    generate_param_combinations,
    get_grid,
)
from rallycut.evaluation.tracking import (
    ErrorEvent,
    ErrorSummary,
    ErrorType,
    PerFrameMetrics,
    PerPlayerMetrics,
    TrackingEvaluationRally,
    TrackingEvaluationResult,
    analyze_errors,
    evaluate_rally,
    load_labeled_rallies,
    summarize_errors,
)
from rallycut.evaluation.tracking import (
    aggregate_results as aggregate_tracking_results,
)
from rallycut.evaluation.video_resolver import VideoResolver

__all__ = [
    # Ground truth
    "EvaluationVideo",
    "GroundTruthRally",
    "load_evaluation_videos",
    # Matching
    "MatchingResult",
    "RallyMatch",
    "match_rallies",
    # Metrics
    "AggregateMetrics",
    "BoundaryMetrics",
    "RallyMetrics",
    "VideoEvaluationResult",
    "aggregate_metrics",
    "compute_metrics",
    # Cached analysis
    "AnalysisCache",
    "CachedAnalysis",
    "PostProcessingParams",
    "analyze_and_cache",
    "apply_post_processing",
    "apply_post_processing_custom",
    # Parameter tuning
    "AVAILABLE_GRIDS",
    "DEFAULT_PARAMS",
    "generate_param_combinations",
    "get_grid",
    # Video resolver
    "VideoResolver",
    # Tracking evaluation
    "ErrorEvent",
    "ErrorSummary",
    "ErrorType",
    "PerFrameMetrics",
    "PerPlayerMetrics",
    "TrackingEvaluationRally",
    "TrackingEvaluationResult",
    "aggregate_tracking_results",
    "analyze_errors",
    "evaluate_rally",
    "load_labeled_rallies",
    "summarize_errors",
]
