"""Player and ball tracking evaluation framework."""

from rallycut.evaluation.tracking.ball_grid_search import (
    BallFilterConfigResult,
    BallFilterGridSearchResult,
    BallRawCache,
    CachedBallData,
    apply_ball_filter_config,
    ball_grid_search,
    evaluate_ball_config,
)
from rallycut.evaluation.tracking.ball_metrics import (
    BallFrameComparison,
    BallTrackingMetrics,
    aggregate_ball_metrics,
    evaluate_ball_tracking,
)
from rallycut.evaluation.tracking.ball_param_grid import (
    BALL_AVAILABLE_GRIDS,
    ball_grid_size,
    describe_ball_config_diff,
    generate_ball_filter_configs,
    get_ball_grid,
    get_default_ball_config,
)
from rallycut.evaluation.tracking.db import (
    TrackingEvaluationRally,
    load_labeled_rallies,
)
from rallycut.evaluation.tracking.error_analysis import (
    ErrorEvent,
    ErrorSummary,
    ErrorType,
    analyze_errors,
    summarize_errors,
)
from rallycut.evaluation.tracking.grid_search import (
    FilterConfigResult,
    FilterGridSearchResult,
    apply_filter_config,
    evaluate_config,
    grid_search,
)
from rallycut.evaluation.tracking.metrics import (
    PerFrameMetrics,
    PerPlayerMetrics,
    TrackingEvaluationResult,
    aggregate_results,
    evaluate_rally,
)
from rallycut.evaluation.tracking.param_grid import (
    AVAILABLE_GRIDS,
    describe_config_diff,
    generate_filter_configs,
    get_default_config,
    get_grid,
    grid_size,
)
from rallycut.evaluation.tracking.raw_cache import (
    CachedRallyData,
    RawPositionCache,
)

__all__ = [
    # Database loading
    "TrackingEvaluationRally",
    "load_labeled_rallies",
    # Ball metrics
    "BallFrameComparison",
    "BallTrackingMetrics",
    "aggregate_ball_metrics",
    "evaluate_ball_tracking",
    # Ball parameter grid
    "BALL_AVAILABLE_GRIDS",
    "ball_grid_size",
    "describe_ball_config_diff",
    "generate_ball_filter_configs",
    "get_ball_grid",
    "get_default_ball_config",
    # Ball grid search
    "BallFilterConfigResult",
    "BallFilterGridSearchResult",
    "BallRawCache",
    "CachedBallData",
    "apply_ball_filter_config",
    "ball_grid_search",
    "evaluate_ball_config",
    # Player metrics
    "PerFrameMetrics",
    "PerPlayerMetrics",
    "TrackingEvaluationResult",
    "aggregate_results",
    "evaluate_rally",
    # Error analysis
    "ErrorEvent",
    "ErrorSummary",
    "ErrorType",
    "analyze_errors",
    "summarize_errors",
    # Player parameter grid
    "AVAILABLE_GRIDS",
    "describe_config_diff",
    "generate_filter_configs",
    "get_default_config",
    "get_grid",
    "grid_size",
    # Raw position cache
    "CachedRallyData",
    "RawPositionCache",
    # Player grid search
    "FilterConfigResult",
    "FilterGridSearchResult",
    "apply_filter_config",
    "evaluate_config",
    "grid_search",
]
