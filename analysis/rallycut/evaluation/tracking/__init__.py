"""Player tracking evaluation framework."""

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
from rallycut.evaluation.tracking.metrics import (
    PerFrameMetrics,
    PerPlayerMetrics,
    TrackingEvaluationResult,
    aggregate_results,
    evaluate_rally,
)

__all__ = [
    # Database loading
    "TrackingEvaluationRally",
    "load_labeled_rallies",
    # Metrics
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
]
