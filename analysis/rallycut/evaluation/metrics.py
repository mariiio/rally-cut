"""Metric computation for rally detection evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median

from rallycut.evaluation.matching import MatchingResult


@dataclass
class RallyMetrics:
    """Rally-level detection metrics."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP). How many detections are correct."""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN). How many ground truth rallies are detected."""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 score: harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class BoundaryMetrics:
    """Boundary timing accuracy metrics for matched rallies."""

    start_errors_ms: list[int] = field(default_factory=list)
    end_errors_ms: list[int] = field(default_factory=list)

    @property
    def mean_start_error_ms(self) -> float | None:
        """Mean signed start error (positive = late start)."""
        return mean(self.start_errors_ms) if self.start_errors_ms else None

    @property
    def median_start_error_ms(self) -> float | None:
        """Median signed start error."""
        return median(self.start_errors_ms) if self.start_errors_ms else None

    @property
    def mean_end_error_ms(self) -> float | None:
        """Mean signed end error (positive = late end)."""
        return mean(self.end_errors_ms) if self.end_errors_ms else None

    @property
    def median_end_error_ms(self) -> float | None:
        """Median signed end error."""
        return median(self.end_errors_ms) if self.end_errors_ms else None

    @property
    def mean_abs_start_error_ms(self) -> float | None:
        """Mean absolute start error."""
        if not self.start_errors_ms:
            return None
        return mean(abs(e) for e in self.start_errors_ms)

    @property
    def mean_abs_end_error_ms(self) -> float | None:
        """Mean absolute end error."""
        if not self.end_errors_ms:
            return None
        return mean(abs(e) for e in self.end_errors_ms)


@dataclass
class OvermergeMetrics:
    """Metrics for detecting overmerged segments (too long rallies)."""

    overmerge_threshold_seconds: float = 60.0
    segment_durations_seconds: list[float] = field(default_factory=list)

    @property
    def overmerge_count(self) -> int:
        """Number of segments that exceed the threshold."""
        return sum(1 for d in self.segment_durations_seconds if d > self.overmerge_threshold_seconds)

    @property
    def overmerge_rate(self) -> float:
        """Fraction of segments that exceed the threshold."""
        if not self.segment_durations_seconds:
            return 0.0
        return self.overmerge_count / len(self.segment_durations_seconds)

    @property
    def max_segment_duration_seconds(self) -> float | None:
        """Maximum segment duration in seconds."""
        if not self.segment_durations_seconds:
            return None
        return max(self.segment_durations_seconds)


@dataclass
class VideoEvaluationResult:
    """Evaluation results for a single video."""

    video_id: str
    video_filename: str
    ground_truth_count: int
    prediction_count: int
    rally_metrics: RallyMetrics
    boundary_metrics: BoundaryMetrics
    iou_threshold: float
    processing_time_seconds: float | None = None
    video_duration_seconds: float | None = None
    overmerge_metrics: OvermergeMetrics = field(default_factory=OvermergeMetrics)

    @property
    def inference_time_per_minute(self) -> float | None:
        """Inference time in seconds per minute of video."""
        if self.processing_time_seconds is None or self.video_duration_seconds is None:
            return None
        if self.video_duration_seconds <= 0:
            return None
        video_minutes = self.video_duration_seconds / 60.0
        return self.processing_time_seconds / video_minutes


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all videos."""

    total_ground_truth: int
    total_predictions: int
    rally_metrics: RallyMetrics
    boundary_metrics: BoundaryMetrics
    overmerge_metrics: OvermergeMetrics
    per_video_results: list[VideoEvaluationResult]

    @property
    def video_count(self) -> int:
        """Number of videos evaluated."""
        return len(self.per_video_results)

    @property
    def total_processing_time(self) -> float | None:
        """Total processing time across all videos."""
        times = [
            r.processing_time_seconds
            for r in self.per_video_results
            if r.processing_time_seconds is not None
        ]
        return sum(times) if times else None

    @property
    def total_video_duration_seconds(self) -> float | None:
        """Total video duration across all videos."""
        durations = [
            r.video_duration_seconds
            for r in self.per_video_results
            if r.video_duration_seconds is not None
        ]
        return sum(durations) if durations else None

    @property
    def inference_time_per_minute(self) -> float | None:
        """Average inference time per minute of video."""
        proc_time = self.total_processing_time
        duration = self.total_video_duration_seconds
        if proc_time is None or duration is None or duration <= 0:
            return None
        return proc_time / (duration / 60.0)


def compute_metrics(
    ground_truth: list[tuple[float, float]],
    predictions: list[tuple[float, float]],
    matching_result: MatchingResult,
    video_id: str,
    video_filename: str,
    iou_threshold: float,
    processing_time: float | None = None,
    video_duration_seconds: float | None = None,
    overmerge_threshold_seconds: float = 60.0,
) -> VideoEvaluationResult:
    """Compute all metrics for a single video.

    Args:
        ground_truth: List of (start_seconds, end_seconds) for ground truth rallies.
        predictions: List of (start_seconds, end_seconds) for predicted rallies.
        matching_result: Result from match_rallies().
        video_id: Video identifier.
        video_filename: Video filename for display.
        iou_threshold: IoU threshold used for matching.
        processing_time: Optional processing time in seconds.
        video_duration_seconds: Optional video duration in seconds.
        overmerge_threshold_seconds: Threshold for overmerge detection (default 60s).

    Returns:
        VideoEvaluationResult with all computed metrics.
    """
    rally_metrics = RallyMetrics(
        true_positives=len(matching_result.matches),
        false_positives=len(matching_result.unmatched_predictions),
        false_negatives=len(matching_result.unmatched_ground_truth),
    )

    boundary_metrics = BoundaryMetrics(
        start_errors_ms=[m.start_error_ms for m in matching_result.matches],
        end_errors_ms=[m.end_error_ms for m in matching_result.matches],
    )

    # Compute segment durations for overmerge detection
    segment_durations = [end - start for start, end in predictions]
    overmerge_metrics = OvermergeMetrics(
        overmerge_threshold_seconds=overmerge_threshold_seconds,
        segment_durations_seconds=segment_durations,
    )

    return VideoEvaluationResult(
        video_id=video_id,
        video_filename=video_filename,
        ground_truth_count=len(ground_truth),
        prediction_count=len(predictions),
        rally_metrics=rally_metrics,
        boundary_metrics=boundary_metrics,
        iou_threshold=iou_threshold,
        processing_time_seconds=processing_time,
        video_duration_seconds=video_duration_seconds,
        overmerge_metrics=overmerge_metrics,
    )


def aggregate_metrics(results: list[VideoEvaluationResult]) -> AggregateMetrics:
    """Aggregate metrics across multiple videos.

    Args:
        results: List of per-video evaluation results.

    Returns:
        AggregateMetrics with combined statistics.
    """
    total_tp = sum(r.rally_metrics.true_positives for r in results)
    total_fp = sum(r.rally_metrics.false_positives for r in results)
    total_fn = sum(r.rally_metrics.false_negatives for r in results)

    all_start_errors: list[int] = []
    all_end_errors: list[int] = []
    all_segment_durations: list[float] = []
    overmerge_threshold = 60.0  # default

    for r in results:
        all_start_errors.extend(r.boundary_metrics.start_errors_ms)
        all_end_errors.extend(r.boundary_metrics.end_errors_ms)
        all_segment_durations.extend(r.overmerge_metrics.segment_durations_seconds)
        overmerge_threshold = r.overmerge_metrics.overmerge_threshold_seconds

    return AggregateMetrics(
        total_ground_truth=sum(r.ground_truth_count for r in results),
        total_predictions=sum(r.prediction_count for r in results),
        rally_metrics=RallyMetrics(
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
        ),
        boundary_metrics=BoundaryMetrics(
            start_errors_ms=all_start_errors,
            end_errors_ms=all_end_errors,
        ),
        overmerge_metrics=OvermergeMetrics(
            overmerge_threshold_seconds=overmerge_threshold,
            segment_durations_seconds=all_segment_durations,
        ),
        per_video_results=results,
    )
