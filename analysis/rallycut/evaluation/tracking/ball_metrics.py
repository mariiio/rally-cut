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

    # Velocity error statistics (in pixels/frame)
    velocity_errors_px: list[float] = field(default_factory=list)

    # Temporal jitter (frame-to-frame position variance in pixels)
    temporal_jitter_px: list[float] = field(default_factory=list)

    # Per-confidence-band metrics
    # Key: confidence band (e.g., "0.3-0.5"), Value: list of errors in that band
    errors_by_confidence_band: dict[str, list[float]] = field(default_factory=dict)

    # Video metadata
    video_width: int = 1920
    video_height: int = 1080
    video_fps: float = 30.0

    # Frame offset applied during evaluation (0 = no shift)
    frame_offset: int = 0

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

    @property
    def mean_velocity_error_px(self) -> float:
        """Mean velocity error in pixels/frame."""
        if not self.velocity_errors_px:
            return 0.0
        return sum(self.velocity_errors_px) / len(self.velocity_errors_px)

    @property
    def median_velocity_error_px(self) -> float:
        """Median velocity error in pixels/frame."""
        if not self.velocity_errors_px:
            return 0.0
        sorted_errors = sorted(self.velocity_errors_px)
        n = len(sorted_errors)
        if n % 2 == 0:
            return (sorted_errors[n // 2 - 1] + sorted_errors[n // 2]) / 2
        return sorted_errors[n // 2]

    @property
    def mean_jitter_px(self) -> float:
        """Mean temporal jitter in pixels."""
        if not self.temporal_jitter_px:
            return 0.0
        return sum(self.temporal_jitter_px) / len(self.temporal_jitter_px)

    @property
    def p90_jitter_px(self) -> float:
        """90th percentile temporal jitter in pixels."""
        if not self.temporal_jitter_px:
            return 0.0
        sorted_jitter = sorted(self.temporal_jitter_px)
        idx = int(len(sorted_jitter) * 0.9)
        return sorted_jitter[min(idx, len(sorted_jitter) - 1)]

    def get_error_stats_by_confidence(self) -> dict[str, dict[str, float]]:
        """Get error statistics broken down by confidence band.

        Returns:
            Dict mapping confidence band to error stats (mean, median, count).
        """
        result = {}
        for band, errors in self.errors_by_confidence_band.items():
            if not errors:
                continue
            sorted_errors = sorted(errors)
            n = len(sorted_errors)
            median = (
                (sorted_errors[n // 2 - 1] + sorted_errors[n // 2]) / 2
                if n % 2 == 0
                else sorted_errors[n // 2]
            )
            result[band] = {
                "mean": sum(errors) / n,
                "median": median,
                "count": n,
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
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
            "frameOffset": self.frame_offset,
        }

        # Add velocity and jitter metrics if available
        if self.velocity_errors_px:
            result["meanVelocityErrorPx"] = self.mean_velocity_error_px
            result["medianVelocityErrorPx"] = self.median_velocity_error_px

        if self.temporal_jitter_px:
            result["meanJitterPx"] = self.mean_jitter_px
            result["p90JitterPx"] = self.p90_jitter_px

        # Add per-confidence-band breakdown if available
        confidence_stats = self.get_error_stats_by_confidence()
        if confidence_stats:
            result["errorsByConfidenceBand"] = confidence_stats

        return result


def _get_confidence_band(confidence: float) -> str:
    """Get the confidence band label for a given confidence value."""
    if confidence < 0.3:
        return "0.0-0.3"
    elif confidence < 0.5:
        return "0.3-0.5"
    elif confidence < 0.7:
        return "0.5-0.7"
    elif confidence < 0.9:
        return "0.7-0.9"
    else:
        return "0.9-1.0"


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

    Automatically detects the optimal frame offset (0-5 frames) to handle
    FPS variations and labeling timing differences between videos.

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
    # Auto-detect optimal frame offset
    best_offset, _ = find_optimal_frame_offset(
        ground_truth, predictions,
        video_width=video_width, video_height=video_height,
        match_threshold_px=match_threshold_px,
    )

    # Filter GT to ball positions only
    gt_ball = [p for p in ground_truth if p.label == "ball"]

    # Group GT by frame
    gt_by_frame: dict[int, GroundTruthPosition] = {}
    for gt_pos in gt_ball:
        gt_by_frame[gt_pos.frame_number] = gt_pos

    # Group predictions by frame, applying offset (shift predictions back)
    pred_by_frame: dict[int, BallPosition] = {}
    for pred_pos in predictions:
        if pred_pos.confidence < min_confidence:
            continue
        shifted_frame = pred_pos.frame_number - best_offset
        existing = pred_by_frame.get(shifted_frame)
        if existing is None or pred_pos.confidence > existing.confidence:
            pred_by_frame[shifted_frame] = pred_pos

    metrics = BallTrackingMetrics(
        num_gt_frames=len(gt_by_frame),
        video_width=video_width,
        video_height=video_height,
        video_fps=video_fps,
        frame_offset=best_offset,
    )

    # Initialize confidence band dict
    for band in ["0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]:
        metrics.errors_by_confidence_band[band] = []

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

        # Add to per-confidence-band metrics
        band = _get_confidence_band(matched_pred.confidence)
        metrics.errors_by_confidence_band[band].append(error_px)

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

    # Compute velocity errors (compare GT velocity to prediction velocity)
    # Sort comparisons by frame for velocity computation
    detected_comparisons = [c for c in metrics.comparisons if c.detected]
    detected_comparisons.sort(key=lambda c: c.frame_number)

    for i in range(1, len(detected_comparisons)):
        curr = detected_comparisons[i]
        prev = detected_comparisons[i - 1]

        # Skip if frames are not consecutive
        if curr.frame_number != prev.frame_number + 1:
            continue

        # GT velocity
        gt_vx = (curr.gt_x - prev.gt_x) * video_width
        gt_vy = (curr.gt_y - prev.gt_y) * video_height

        # Predicted velocity
        pred_vx = (curr.pred_x - prev.pred_x) * video_width  # type: ignore
        pred_vy = (curr.pred_y - prev.pred_y) * video_height  # type: ignore

        # Velocity error (magnitude of velocity difference)
        vel_error = math.sqrt(
            (pred_vx - gt_vx) ** 2 + (pred_vy - gt_vy) ** 2
        )
        metrics.velocity_errors_px.append(vel_error)

    # Compute temporal jitter (frame-to-frame variance in prediction error)
    # Jitter measures how much the error changes from frame to frame
    for i in range(1, len(detected_comparisons)):
        curr = detected_comparisons[i]
        prev = detected_comparisons[i - 1]

        # Skip if frames are not consecutive
        if curr.frame_number != prev.frame_number + 1:
            continue

        # Jitter is the change in predicted position between frames
        # (ideally should be smooth)
        pred_dx = (curr.pred_x - prev.pred_x) * video_width  # type: ignore
        pred_dy = (curr.pred_y - prev.pred_y) * video_height  # type: ignore

        gt_dx = (curr.gt_x - prev.gt_x) * video_width
        gt_dy = (curr.gt_y - prev.gt_y) * video_height

        # Jitter is the deviation from expected motion
        jitter = math.sqrt((pred_dx - gt_dx) ** 2 + (pred_dy - gt_dy) ** 2)
        metrics.temporal_jitter_px.append(jitter)

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

    # Initialize confidence band dict
    for band in ["0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]:
        combined.errors_by_confidence_band[band] = []

    for r in results:
        combined.num_gt_frames += r.num_gt_frames
        combined.num_detected += r.num_detected
        combined.num_matched += r.num_matched
        combined.errors_norm.extend(r.errors_norm)
        combined.errors_px.extend(r.errors_px)
        combined.velocity_errors_px.extend(r.velocity_errors_px)
        combined.temporal_jitter_px.extend(r.temporal_jitter_px)

        # Aggregate confidence band errors
        for band, errors in r.errors_by_confidence_band.items():
            if band not in combined.errors_by_confidence_band:
                combined.errors_by_confidence_band[band] = []
            combined.errors_by_confidence_band[band].extend(errors)

        # Use first rally's video dimensions (should all be same video typically)
        if combined.video_width == 1920 and r.video_width != 1920:
            combined.video_width = r.video_width
            combined.video_height = r.video_height
            combined.video_fps = r.video_fps

    return combined


def find_optimal_frame_offset(
    ground_truth: list[GroundTruthPosition],
    predictions: list[BallPosition],
    video_width: int = 1920,
    video_height: int = 1080,
    max_offset: int = 5,
    match_threshold_px: float = 50.0,
) -> tuple[int, float]:
    """Find the optimal frame offset for aligning predictions with ground truth.

    Different videos may have different frame alignment requirements due to:
    - FPS differences (29.97 vs 30.0)
    - Model prediction lag
    - Ground truth labeling timing

    This function tests offsets from 0 to max_offset and returns the one
    that maximizes match rate.

    Args:
        ground_truth: Ground truth ball positions.
        predictions: Predicted ball positions from tracker.
        video_width: Video frame width for pixel conversion.
        video_height: Video frame height for pixel conversion.
        max_offset: Maximum offset to test (default 5 frames).
        match_threshold_px: Distance threshold for counting matches.

    Returns:
        Tuple of (best_offset, best_match_rate).
    """
    gt_ball = [p for p in ground_truth if p.label == "ball"]
    if not gt_ball:
        return 0, 0.0

    gt_by_frame = {p.frame_number: p for p in gt_ball}

    best_offset = 0
    best_match_rate = 0.0

    for offset in range(max_offset + 1):
        # Shift predictions by offset
        pred_by_frame: dict[int, BallPosition] = {}
        for p in predictions:
            if p.confidence < 0.3:
                continue
            shifted_frame = p.frame_number - offset
            existing = pred_by_frame.get(shifted_frame)
            if existing is None or p.confidence > existing.confidence:
                pred_by_frame[shifted_frame] = p

        # Count matches
        num_matched = 0
        for frame, gt_pos in gt_by_frame.items():
            pred = pred_by_frame.get(frame)
            if pred is None:
                continue

            # Compute error in pixels
            dx = (pred.x - gt_pos.x) * video_width
            dy = (pred.y - gt_pos.y) * video_height
            error_px = math.sqrt(dx * dx + dy * dy)

            if error_px <= match_threshold_px:
                num_matched += 1

        match_rate = num_matched / len(gt_by_frame) if gt_by_frame else 0.0

        if match_rate > best_match_rate:
            best_match_rate = match_rate
            best_offset = offset

    return best_offset, best_match_rate
