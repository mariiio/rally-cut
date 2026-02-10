"""Ball tracking evaluation metrics.

Provides detailed metrics for evaluating ball tracking accuracy against
ground truth labels. Computes detection rate, position accuracy, and
error distribution statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from rallycut.labeling.ground_truth import GroundTruthPosition
from rallycut.tracking.ball_tracker import BallPosition


@dataclass
class BallFrameComparison:
    """Comparison result for a single frame."""

    frame_number: int
    gt_x: float
    gt_y: float
    pred_x: float | None
    pred_y: float | None
    pred_confidence: float | None
    error_norm: float | None  # Normalized error (0-1 scale)
    error_px: float | None  # Error in pixels
    detected: bool


@dataclass
class BallTrackingMetrics:
    """Comprehensive ball tracking metrics."""

    # Detection metrics
    num_gt_frames: int = 0  # Frames with ground truth ball annotation
    num_detected: int = 0  # Frames where ball was detected (pred exists)
    num_matched: int = 0  # Frames where prediction is close enough to GT

    # Error statistics (in normalized coordinates 0-1)
    errors_norm: list[float] = field(default_factory=list)

    # Error statistics (in pixels)
    errors_px: list[float] = field(default_factory=list)

    # Frame comparisons for detailed analysis
    comparisons: list[BallFrameComparison] = field(default_factory=list)

    # Video metadata
    video_width: int = 1920
    video_height: int = 1080
    video_fps: float = 30.0

    @property
    def detection_rate(self) -> float:
        """Fraction of GT frames where ball was detected."""
        return self.num_detected / self.num_gt_frames if self.num_gt_frames > 0 else 0.0

    @property
    def match_rate(self) -> float:
        """Fraction of GT frames where ball was detected within threshold."""
        return self.num_matched / self.num_gt_frames if self.num_gt_frames > 0 else 0.0

    @property
    def mean_error_norm(self) -> float:
        """Mean position error (normalized 0-1 coordinates)."""
        return sum(self.errors_norm) / len(self.errors_norm) if self.errors_norm else 0.0

    @property
    def mean_error_px(self) -> float:
        """Mean position error in pixels."""
        return sum(self.errors_px) / len(self.errors_px) if self.errors_px else 0.0

    @property
    def median_error_px(self) -> float:
        """Median position error in pixels."""
        if not self.errors_px:
            return 0.0
        sorted_errors = sorted(self.errors_px)
        n = len(sorted_errors)
        if n % 2 == 0:
            return (sorted_errors[n // 2 - 1] + sorted_errors[n // 2]) / 2
        return sorted_errors[n // 2]

    @property
    def p90_error_px(self) -> float:
        """90th percentile position error in pixels."""
        if not self.errors_px:
            return 0.0
        sorted_errors = sorted(self.errors_px)
        idx = int(len(sorted_errors) * 0.9)
        return sorted_errors[min(idx, len(sorted_errors) - 1)]

    @property
    def max_error_px(self) -> float:
        """Maximum position error in pixels."""
        return max(self.errors_px) if self.errors_px else 0.0

    @property
    def error_under_20px_rate(self) -> float:
        """Fraction of detections with error under 20 pixels."""
        if not self.errors_px:
            return 0.0
        under_20 = sum(1 for e in self.errors_px if e < 20)
        return under_20 / len(self.errors_px)

    @property
    def error_under_50px_rate(self) -> float:
        """Fraction of detections with error under 50 pixels."""
        if not self.errors_px:
            return 0.0
        under_50 = sum(1 for e in self.errors_px if e < 50)
        return under_50 / len(self.errors_px)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "numGtFrames": self.num_gt_frames,
            "numDetected": self.num_detected,
            "numMatched": self.num_matched,
            "detectionRate": self.detection_rate,
            "matchRate": self.match_rate,
            "meanErrorPx": self.mean_error_px,
            "medianErrorPx": self.median_error_px,
            "p90ErrorPx": self.p90_error_px,
            "maxErrorPx": self.max_error_px,
            "errorUnder20pxRate": self.error_under_20px_rate,
            "errorUnder50pxRate": self.error_under_50px_rate,
            "videoWidth": self.video_width,
            "videoHeight": self.video_height,
        }


def evaluate_ball_tracking(
    ground_truth: list[GroundTruthPosition],
    predictions: list[BallPosition],
    video_width: int = 1920,
    video_height: int = 1080,
    video_fps: float = 30.0,
    match_threshold_px: float = 50.0,
    min_confidence: float = 0.3,
) -> BallTrackingMetrics:
    """Evaluate ball tracking predictions against ground truth.

    Args:
        ground_truth: Ground truth ball positions (from Label Studio).
        predictions: Predicted ball positions from tracker.
        video_width: Video frame width for pixel conversion.
        video_height: Video frame height for pixel conversion.
        video_fps: Video frame rate.
        match_threshold_px: Maximum distance (pixels) for a "match".
        min_confidence: Minimum prediction confidence to consider.

    Returns:
        BallTrackingMetrics with detailed accuracy statistics.
    """
    # Filter GT to ball positions only
    gt_ball = [p for p in ground_truth if p.label == "ball"]

    # Group GT by frame
    gt_by_frame: dict[int, GroundTruthPosition] = {}
    for gt_pos in gt_ball:
        gt_by_frame[gt_pos.frame_number] = gt_pos

    # Group predictions by frame (take highest confidence per frame)
    pred_by_frame: dict[int, BallPosition] = {}
    for pred_pos in predictions:
        if pred_pos.confidence < min_confidence:
            continue
        existing = pred_by_frame.get(pred_pos.frame_number)
        if existing is None or pred_pos.confidence > existing.confidence:
            pred_by_frame[pred_pos.frame_number] = pred_pos

    metrics = BallTrackingMetrics(
        num_gt_frames=len(gt_by_frame),
        video_width=video_width,
        video_height=video_height,
        video_fps=video_fps,
    )

    for frame, gt_pos in gt_by_frame.items():
        matched_pred = pred_by_frame.get(frame)

        if matched_pred is None:
            # No prediction for this frame
            metrics.comparisons.append(
                BallFrameComparison(
                    frame_number=frame,
                    gt_x=gt_pos.x,
                    gt_y=gt_pos.y,
                    pred_x=None,
                    pred_y=None,
                    pred_confidence=None,
                    error_norm=None,
                    error_px=None,
                    detected=False,
                )
            )
            continue

        # Compute error
        dx = matched_pred.x - gt_pos.x
        dy = matched_pred.y - gt_pos.y
        error_norm = math.sqrt(dx * dx + dy * dy)

        # Convert to pixels (use diagonal for consistent scale)
        dx_px = dx * video_width
        dy_px = dy * video_height
        error_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)

        metrics.num_detected += 1
        if error_px <= match_threshold_px:
            metrics.num_matched += 1

        metrics.errors_norm.append(error_norm)
        metrics.errors_px.append(error_px)

        metrics.comparisons.append(
            BallFrameComparison(
                frame_number=frame,
                gt_x=gt_pos.x,
                gt_y=gt_pos.y,
                pred_x=matched_pred.x,
                pred_y=matched_pred.y,
                pred_confidence=matched_pred.confidence,
                error_norm=error_norm,
                error_px=error_px,
                detected=True,
            )
        )

    return metrics


def aggregate_ball_metrics(
    results: list[BallTrackingMetrics],
) -> BallTrackingMetrics:
    """Aggregate ball metrics from multiple rallies.

    Args:
        results: List of per-rally ball tracking metrics.

    Returns:
        Combined BallTrackingMetrics across all rallies.
    """
    combined = BallTrackingMetrics()

    for r in results:
        combined.num_gt_frames += r.num_gt_frames
        combined.num_detected += r.num_detected
        combined.num_matched += r.num_matched
        combined.errors_norm.extend(r.errors_norm)
        combined.errors_px.extend(r.errors_px)
        # Use first rally's video dimensions (should all be same video typically)
        if combined.video_width == 1920 and r.video_width != 1920:
            combined.video_width = r.video_width
            combined.video_height = r.video_height
            combined.video_fps = r.video_fps

    return combined
