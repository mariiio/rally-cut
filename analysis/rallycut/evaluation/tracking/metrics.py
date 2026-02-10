"""Enhanced MOT metrics for tracking evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rallycut.cli.commands.compare_tracking import (
    BallMetrics,
    MOTMetrics,
    _match_detections,
    compute_ball_metrics,
)
from rallycut.labeling.ground_truth import GroundTruthResult
from rallycut.tracking.player_tracker import PlayerTrackingResult


@dataclass
class PerPlayerMetrics:
    """Metrics for a single player track."""

    label: str  # player_1, player_2, etc. (from ground truth track_id)
    gt_count: int = 0  # Ground truth detections for this player
    pred_count: int = 0  # Matched predictions
    matches: int = 0  # True positives
    misses: int = 0  # False negatives (frames where GT exists but no match)
    id_switches: int = 0  # Times the matched pred ID changed

    @property
    def precision(self) -> float:
        """Precision for this player."""
        # For per-player, precision = matches / pred_count
        return self.matches / self.pred_count if self.pred_count > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall for this player."""
        return self.matches / self.gt_count if self.gt_count > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 score for this player."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class PerFrameMetrics:
    """Metrics for a single frame."""

    frame_number: int
    gt_count: int = 0  # Ground truth objects in this frame
    pred_count: int = 0  # Predictions in this frame
    matches: int = 0  # True positives
    misses: int = 0  # False negatives
    false_positives: int = 0  # False positives
    id_switches: int = 0  # ID switches in this frame

    @property
    def has_errors(self) -> bool:
        """Whether this frame has any errors."""
        return self.misses > 0 or self.false_positives > 0 or self.id_switches > 0


@dataclass
class TrackingEvaluationResult:
    """Complete evaluation result with aggregate and per-entity breakdowns."""

    rally_id: str
    aggregate: MOTMetrics
    per_player: list[PerPlayerMetrics] = field(default_factory=list)
    per_frame: list[PerFrameMetrics] = field(default_factory=list)
    ball_metrics: BallMetrics | None = None

    @property
    def error_frames(self) -> list[int]:
        """Frame numbers with errors (misses, FPs, or ID switches)."""
        return [f.frame_number for f in self.per_frame if f.has_errors]

    @property
    def worst_player(self) -> PerPlayerMetrics | None:
        """Player with lowest F1 score."""
        if not self.per_player:
            return None
        return min(self.per_player, key=lambda p: p.f1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "rallyId": self.rally_id,
            "aggregate": {
                "mota": self.aggregate.mota,
                "precision": self.aggregate.precision,
                "recall": self.aggregate.recall,
                "f1": self.aggregate.f1,
                "idSwitches": self.aggregate.num_id_switches,
                "numGt": self.aggregate.num_gt,
                "numPred": self.aggregate.num_pred,
                "numMatches": self.aggregate.num_matches,
                "numMisses": self.aggregate.num_misses,
                "numFalsePositives": self.aggregate.num_false_positives,
            },
            "perPlayer": [
                {
                    "label": p.label,
                    "precision": p.precision,
                    "recall": p.recall,
                    "f1": p.f1,
                    "idSwitches": p.id_switches,
                    "gtCount": p.gt_count,
                    "matches": p.matches,
                    "misses": p.misses,
                }
                for p in self.per_player
            ],
            "errorFrames": self.error_frames,
            "errorFrameCount": len(self.error_frames),
        }

        if self.ball_metrics:
            result["ball"] = {
                "detectionRate": self.ball_metrics.detection_rate,
                "meanErrorPx": self.ball_metrics.mean_error_px,
                "numGt": self.ball_metrics.num_gt,
                "numDetected": self.ball_metrics.num_detected,
            }

        return result


def evaluate_rally(
    rally_id: str,
    ground_truth: GroundTruthResult,
    predictions: PlayerTrackingResult,
    iou_threshold: float = 0.5,
    video_width: int | None = None,
    video_height: int | None = None,
) -> TrackingEvaluationResult:
    """Evaluate tracking predictions against ground truth with detailed breakdowns.

    Args:
        rally_id: Identifier for this rally.
        ground_truth: Ground truth annotations.
        predictions: Predicted player positions.
        iou_threshold: Minimum IoU for matching.
        video_width: Video width for ball metrics (optional).
        video_height: Video height for ball metrics (optional).

    Returns:
        TrackingEvaluationResult with aggregate, per-player, and per-frame metrics.
    """
    gt_positions = ground_truth.player_positions
    pred_positions = predictions.positions

    # Group by frame
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)

    for gt_pos in gt_positions:
        gt_by_frame[gt_pos.frame_number].append(
            (gt_pos.track_id, gt_pos.x, gt_pos.y, gt_pos.width, gt_pos.height)
        )

    for pred_pos in pred_positions:
        pred_by_frame[pred_pos.frame_number].append(
            (pred_pos.track_id, pred_pos.x, pred_pos.y, pred_pos.width, pred_pos.height)
        )

    # Get all frames
    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    # Initialize aggregate metrics
    aggregate = MOTMetrics()

    # Per-player tracking
    # Map GT track_id -> per-player metrics
    player_metrics: dict[int, PerPlayerMetrics] = {}
    for gt_track_id in ground_truth.unique_player_tracks:
        player_metrics[gt_track_id] = PerPlayerMetrics(label=f"player_{gt_track_id}")

    # Track last matched prediction ID for each GT track (for ID switch detection)
    last_pred_id: dict[int, int] = {}

    # Per-frame metrics
    frame_metrics: list[PerFrameMetrics] = []

    for frame in all_frames:
        gt_boxes = gt_by_frame.get(frame, [])
        pred_boxes = pred_by_frame.get(frame, [])

        frame_metric = PerFrameMetrics(
            frame_number=frame,
            gt_count=len(gt_boxes),
            pred_count=len(pred_boxes),
        )

        aggregate.num_gt += len(gt_boxes)
        aggregate.num_pred += len(pred_boxes)

        # Count GT appearances per player in this frame
        for gt_id, _, _, _, _ in gt_boxes:
            if gt_id in player_metrics:
                player_metrics[gt_id].gt_count += 1

        if not gt_boxes:
            # No ground truth - all predictions are false positives
            aggregate.num_false_positives += len(pred_boxes)
            frame_metric.false_positives = len(pred_boxes)
            frame_metrics.append(frame_metric)
            continue

        if not pred_boxes:
            # No predictions - all ground truth are misses
            aggregate.num_misses += len(gt_boxes)
            frame_metric.misses = len(gt_boxes)
            # Count misses per player
            for gt_id, _, _, _, _ in gt_boxes:
                if gt_id in player_metrics:
                    player_metrics[gt_id].misses += 1
            frame_metrics.append(frame_metric)
            continue

        # Match detections using Hungarian algorithm
        matches, unmatched_gt, unmatched_pred = _match_detections(
            gt_boxes, pred_boxes, iou_threshold
        )

        aggregate.num_matches += len(matches)
        aggregate.num_misses += len(unmatched_gt)
        aggregate.num_false_positives += len(unmatched_pred)

        frame_metric.matches = len(matches)
        frame_metric.misses = len(unmatched_gt)
        frame_metric.false_positives = len(unmatched_pred)

        # Process matches for per-player and ID switch tracking
        for gt_id, pred_id in matches:
            if gt_id in player_metrics:
                player_metrics[gt_id].matches += 1
                player_metrics[gt_id].pred_count += 1

            # Check for ID switches
            if gt_id in last_pred_id and last_pred_id[gt_id] != pred_id:
                aggregate.num_id_switches += 1
                frame_metric.id_switches += 1
                if gt_id in player_metrics:
                    player_metrics[gt_id].id_switches += 1

            last_pred_id[gt_id] = pred_id

        # Count misses per player
        for gt_id in unmatched_gt:
            if gt_id in player_metrics:
                player_metrics[gt_id].misses += 1

        frame_metrics.append(frame_metric)

    # Sort per-player by label
    per_player = sorted(player_metrics.values(), key=lambda p: p.label)

    # Compute ball metrics if available
    ball_metrics: BallMetrics | None = None
    if ground_truth.ball_positions and predictions.ball_positions:
        width = video_width or ground_truth.video_width or 1920
        height = video_height or ground_truth.video_height or 1080
        ball_metrics = compute_ball_metrics(
            ground_truth.ball_positions,
            predictions.ball_positions,
            width,
            height,
        )

    return TrackingEvaluationResult(
        rally_id=rally_id,
        aggregate=aggregate,
        per_player=per_player,
        per_frame=frame_metrics,
        ball_metrics=ball_metrics,
    )


def aggregate_results(results: list[TrackingEvaluationResult]) -> MOTMetrics:
    """Aggregate multiple evaluation results into overall metrics.

    Args:
        results: List of individual rally evaluation results.

    Returns:
        Combined MOTMetrics across all rallies.
    """
    combined = MOTMetrics()

    for r in results:
        combined.num_gt += r.aggregate.num_gt
        combined.num_pred += r.aggregate.num_pred
        combined.num_matches += r.aggregate.num_matches
        combined.num_misses += r.aggregate.num_misses
        combined.num_false_positives += r.aggregate.num_false_positives
        combined.num_id_switches += r.aggregate.num_id_switches

    return combined
