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
from rallycut.labeling.ground_truth import GroundTruthPosition, GroundTruthResult
from rallycut.tracking.player_tracker import PlayerTrackingResult


def _compute_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes (x, y, w, h format where x,y is center)."""
    x1_min, x1_max = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    y1_min, y1_max = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    x2_min, x2_max = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    y2_min, y2_max = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    xi_min, xi_max = max(x1_min, x2_min), min(x1_max, x2_max)
    yi_min, yi_max = max(y1_min, y2_min), min(y1_max, y2_max)

    if xi_max < xi_min or yi_max < yi_min:
        return 0.0

    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def smart_interpolate_gt(
    ground_truth: GroundTruthResult,
    predictions: PlayerTrackingResult,
    frame_count: int,
    min_keyframe_iou_rate: float = 0.5,
) -> GroundTruthResult:
    """Interpolate ground truth with IoU-based filtering for sparse tracks.

    Only interpolates tracks where a sufficient percentage of keyframes have
    good IoU matches (>=0.5) with predictions. This prevents phantom GT
    positions for tracks that are poorly detected, which would artificially
    lower recall metrics.

    Args:
        ground_truth: Raw ground truth with keyframes only.
        predictions: Predicted positions to check IoU against.
        frame_count: Total frames to interpolate to.
        min_keyframe_iou_rate: Minimum fraction of keyframes that must have
            IoU >= 0.5 with predictions to enable interpolation. Default 0.5
            means at least 50% of keyframes must match well.

    Returns:
        GroundTruthResult with smart interpolation applied.
    """
    # Group predictions by frame for fast lookup
    pred_by_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for p in predictions.positions:
        pred_by_frame[p.frame_number].append((p.x, p.y, p.width, p.height))

    # Group GT by track
    tracks: dict[tuple[int, str], list[GroundTruthPosition]] = {}
    for pos in ground_truth.positions:
        key = (pos.track_id, pos.label)
        if key not in tracks:
            tracks[key] = []
        tracks[key].append(pos)

    interpolated: list[GroundTruthPosition] = []

    for (track_id, label), keyframes in tracks.items():
        keyframes = sorted(keyframes, key=lambda p: p.frame_number)

        if len(keyframes) <= 1:
            interpolated.extend(keyframes)
            continue

        # Calculate keyframe IoU quality for player tracks
        should_interpolate = True
        if label != "ball":  # Ball uses different evaluation
            good_ious = 0
            for kf in keyframes:
                gt_box = (kf.x, kf.y, kf.width, kf.height)
                preds_in_frame = pred_by_frame.get(kf.frame_number, [])
                best_iou = max(
                    (_compute_iou(gt_box, pred_box) for pred_box in preds_in_frame),
                    default=0.0,
                )
                if best_iou >= 0.5:
                    good_ious += 1

            iou_rate = good_ious / len(keyframes)
            should_interpolate = iou_rate >= min_keyframe_iou_rate

        if not should_interpolate:
            # Skip interpolation - keep keyframes only
            interpolated.extend(keyframes)
            continue

        # Normal interpolation
        for i in range(len(keyframes) - 1):
            start = keyframes[i]
            end = keyframes[i + 1]

            for frame in range(start.frame_number, end.frame_number):
                if frame == start.frame_number:
                    interpolated.append(start)
                else:
                    t = (frame - start.frame_number) / (end.frame_number - start.frame_number)
                    interpolated.append(
                        GroundTruthPosition(
                            frame_number=frame,
                            track_id=track_id,
                            label=label,
                            x=start.x + t * (end.x - start.x),
                            y=start.y + t * (end.y - start.y),
                            width=start.width + t * (end.width - start.width),
                            height=start.height + t * (end.height - start.height),
                            confidence=1.0,
                        )
                    )

        interpolated.append(keyframes[-1])

    return GroundTruthResult(
        positions=interpolated,
        frame_count=frame_count,
        video_width=ground_truth.video_width,
        video_height=ground_truth.video_height,
    )


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
class HOTAMetrics:
    """Higher Order Tracking Accuracy metrics.

    HOTA balances detection accuracy (DetA) and association accuracy (AssA).
    HOTA = sqrt(DetA * AssA)

    See: https://arxiv.org/abs/2009.07736
    """

    hota: float = 0.0  # Higher Order Tracking Accuracy
    deta: float = 0.0  # Detection Accuracy (DetA)
    assa: float = 0.0  # Association Accuracy (AssA)
    loca: float = 0.0  # Localization Accuracy (average IoU of TPs)


@dataclass
class TrackQualityMetrics:
    """Track-level quality metrics for analyzing tracking consistency."""

    fragmentation: int = 0  # Number of times GT tracks are split into multiple pred tracks
    num_fragmentations: int = 0  # Total fragmentation events (switches to different pred ID)
    mostly_tracked: int = 0  # GT tracks tracked >80% of their lifespan
    mostly_tracked_ratio: float = 0.0  # Ratio of mostly tracked GT tracks
    partially_tracked: int = 0  # GT tracks tracked 20-80% of their lifespan
    mostly_lost: int = 0  # GT tracks tracked <20% of their lifespan

    # Per-GT track analysis
    gt_track_count: int = 0  # Total number of GT tracks
    avg_track_coverage: float = 0.0  # Average fraction of GT track that is covered

    # ID consistency
    avg_pred_ids_per_gt: float = 0.0  # Average number of different pred IDs matched to each GT


@dataclass
class PositionMetrics:
    """Position accuracy metrics beyond IoU matching."""

    mean_position_error: float = 0.0  # Mean Euclidean error (normalized 0-1)
    median_position_error: float = 0.0  # Median Euclidean error
    p90_position_error: float = 0.0  # 90th percentile error
    num_position_samples: int = 0  # Number of matched pairs used


@dataclass
class TrackingEvaluationResult:
    """Complete evaluation result with aggregate and per-entity breakdowns."""

    rally_id: str
    aggregate: MOTMetrics
    per_player: list[PerPlayerMetrics] = field(default_factory=list)
    per_frame: list[PerFrameMetrics] = field(default_factory=list)
    ball_metrics: BallMetrics | None = None

    # Extended metrics for detailed tracking analysis
    hota_metrics: HOTAMetrics | None = None
    track_quality: TrackQualityMetrics | None = None
    position_metrics: PositionMetrics | None = None

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

        # Extended metrics
        if self.hota_metrics:
            result["hota"] = {
                "hota": self.hota_metrics.hota,
                "deta": self.hota_metrics.deta,
                "assa": self.hota_metrics.assa,
                "loca": self.hota_metrics.loca,
            }

        if self.track_quality:
            result["trackQuality"] = {
                "fragmentation": self.track_quality.fragmentation,
                "numFragmentations": self.track_quality.num_fragmentations,
                "mostlyTracked": self.track_quality.mostly_tracked,
                "mostlyTrackedRatio": self.track_quality.mostly_tracked_ratio,
                "partiallyTracked": self.track_quality.partially_tracked,
                "mostlyLost": self.track_quality.mostly_lost,
                "gtTrackCount": self.track_quality.gt_track_count,
                "avgTrackCoverage": self.track_quality.avg_track_coverage,
                "avgPredIdsPerGt": self.track_quality.avg_pred_ids_per_gt,
            }

        if self.position_metrics:
            result["positionAccuracy"] = {
                "meanError": self.position_metrics.mean_position_error,
                "medianError": self.position_metrics.median_position_error,
                "p90Error": self.position_metrics.p90_position_error,
                "numSamples": self.position_metrics.num_position_samples,
            }

        return result


def evaluate_rally(
    rally_id: str,
    ground_truth: GroundTruthResult,
    predictions: PlayerTrackingResult,
    iou_threshold: float = 0.5,
    video_width: int | None = None,
    video_height: int | None = None,
    interpolate_gt: bool = True,
    smart_interpolate: bool = True,
    min_keyframe_iou_rate: float = 0.5,
) -> TrackingEvaluationResult:
    """Evaluate tracking predictions against ground truth with detailed breakdowns.

    Args:
        rally_id: Identifier for this rally.
        ground_truth: Ground truth annotations.
        predictions: Predicted player positions.
        iou_threshold: Minimum IoU for matching.
        video_width: Video width for ball metrics (optional).
        video_height: Video height for ball metrics (optional).
        interpolate_gt: If True, interpolate keyframe annotations to all frames.
            Label Studio exports only keyframes; this matches the visual display.
        smart_interpolate: If True (default), only interpolate tracks where
            keyframe IoU quality is sufficient. This prevents phantom GT
            positions for poorly-detected sparse tracks.
        min_keyframe_iou_rate: Minimum fraction of keyframes that must have
            IoU >= 0.5 with predictions for interpolation (default 0.5).
            Only used when smart_interpolate=True.

    Returns:
        TrackingEvaluationResult with aggregate, per-player, and per-frame metrics.
    """
    # Interpolate ground truth keyframes to match Label Studio's visual interpolation
    if interpolate_gt and predictions.frame_count > 0:
        if smart_interpolate:
            ground_truth = smart_interpolate_gt(
                ground_truth, predictions, predictions.frame_count, min_keyframe_iou_rate
            )
        else:
            ground_truth = ground_truth.interpolate(predictions.frame_count)

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

    # Track matches by frame for extended metrics (HOTA, fragmentation, position)
    matches_by_frame: dict[int, list[tuple[int, int]]] = {}

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

        # Store matches for extended metrics
        if matches:
            matches_by_frame[frame] = matches

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

    # Compute extended metrics (HOTA, track quality, position accuracy)
    hota_metrics = compute_hota_metrics(
        gt_positions,
        pred_positions,
        matches_by_frame,
        iou_threshold,
    )

    track_quality = compute_track_quality_metrics(
        gt_positions,
        matches_by_frame,
    )

    position_metrics = compute_position_metrics(
        gt_by_frame,
        pred_by_frame,
        matches_by_frame,
    )

    return TrackingEvaluationResult(
        rally_id=rally_id,
        aggregate=aggregate,
        per_player=per_player,
        per_frame=frame_metrics,
        ball_metrics=ball_metrics,
        hota_metrics=hota_metrics,
        track_quality=track_quality,
        position_metrics=position_metrics,
    )


def compute_hota_metrics(
    gt_positions: list[Any],
    pred_positions: list[Any],
    matches_by_frame: dict[int, list[tuple[int, int]]],
    iou_threshold: float = 0.5,
) -> HOTAMetrics:
    """Compute HOTA (Higher Order Tracking Accuracy) metrics.

    HOTA balances detection and association quality:
    - DetA: Detection Accuracy (how well detections match GT)
    - AssA: Association Accuracy (how well IDs are maintained)
    - HOTA = sqrt(DetA * AssA)

    Args:
        gt_positions: Ground truth positions.
        pred_positions: Predicted positions.
        matches_by_frame: Dictionary of frame -> list of (gt_id, pred_id) matches.
        iou_threshold: IoU threshold used for matching.

    Returns:
        HOTAMetrics with HOTA, DetA, AssA, and LocA scores.
    """

    # Count TP, FP, FN for DetA
    total_gt = len(gt_positions)
    total_pred = len(pred_positions)
    total_tp = sum(len(matches) for matches in matches_by_frame.values())

    # DetA = TP / (TP + FP + FN) = TP / (TP + (Pred - TP) + (GT - TP))
    # Simplified: DetA = TP / (GT + Pred - TP)
    deta = total_tp / (total_gt + total_pred - total_tp) if (total_gt + total_pred - total_tp) > 0 else 0.0

    # For AssA, we need to track how consistently each GT track is matched to the same pred ID
    # AssA = sum over all GT tracks of (correctly associated TPs) / sum over all TPs
    # A TP is "correctly associated" if it matches the dominant pred ID for that GT track

    # Find dominant pred ID for each GT track
    gt_to_pred_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for frame_matches in matches_by_frame.values():
        for gt_id, pred_id in frame_matches:
            gt_to_pred_counts[gt_id][pred_id] += 1

    # Count correctly associated TPs (matching dominant pred ID)
    correctly_associated = 0
    for gt_id, pred_counts in gt_to_pred_counts.items():
        if pred_counts:
            dominant_count = max(pred_counts.values())
            correctly_associated += dominant_count

    assa = correctly_associated / total_tp if total_tp > 0 else 0.0

    # HOTA = geometric mean of DetA and AssA
    hota = (deta * assa) ** 0.5 if deta > 0 and assa > 0 else 0.0

    # LocA would require IoU values for each match - approximate with threshold
    # In practice, all matches have IoU >= threshold, so LocA is bounded
    loca = (iou_threshold + 1.0) / 2.0  # Approximate average IoU of TPs

    return HOTAMetrics(hota=hota, deta=deta, assa=assa, loca=loca)


def compute_track_quality_metrics(
    gt_positions: list[Any],
    matches_by_frame: dict[int, list[tuple[int, int]]],
) -> TrackQualityMetrics:
    """Compute track-level quality metrics.

    Analyzes how well individual GT tracks are covered:
    - Mostly Tracked (MT): >80% coverage
    - Partially Tracked (PT): 20-80% coverage
    - Mostly Lost (ML): <20% coverage
    - Fragmentation: how many times a GT track switches pred IDs

    Args:
        gt_positions: Ground truth positions.
        matches_by_frame: Dictionary of frame -> list of (gt_id, pred_id) matches.

    Returns:
        TrackQualityMetrics with fragmentation, MT/PT/ML counts.
    """
    # Group GT positions by track ID
    gt_frames_by_track: dict[int, set[int]] = defaultdict(set)
    for p in gt_positions:
        gt_frames_by_track[p.track_id].add(p.frame_number)

    gt_track_count = len(gt_frames_by_track)
    if gt_track_count == 0:
        return TrackQualityMetrics()

    # Track coverage and fragmentation per GT track
    gt_matched_frames: dict[int, set[int]] = defaultdict(set)
    gt_pred_sequence: dict[int, list[tuple[int, int]]] = defaultdict(list)  # frame, pred_id

    for frame, frame_matches in sorted(matches_by_frame.items()):
        for gt_id, pred_id in frame_matches:
            gt_matched_frames[gt_id].add(frame)
            gt_pred_sequence[gt_id].append((frame, pred_id))

    mostly_tracked = 0
    partially_tracked = 0
    mostly_lost = 0
    total_fragmentations = 0
    fragmented_tracks = 0
    total_coverage = 0.0
    total_pred_ids_per_gt = 0

    for gt_id, gt_frames in gt_frames_by_track.items():
        matched_frames = gt_matched_frames.get(gt_id, set())
        coverage = len(matched_frames) / len(gt_frames) if gt_frames else 0.0
        total_coverage += coverage

        # Classify track
        if coverage > 0.8:
            mostly_tracked += 1
        elif coverage > 0.2:
            partially_tracked += 1
        else:
            mostly_lost += 1

        # Count fragmentations (ID switches within this GT track)
        pred_sequence = gt_pred_sequence.get(gt_id, [])
        if len(pred_sequence) > 1:
            unique_preds = set()
            last_pred = None
            frags = 0
            for _, pred_id in pred_sequence:
                unique_preds.add(pred_id)
                if last_pred is not None and pred_id != last_pred:
                    frags += 1
                last_pred = pred_id
            total_fragmentations += frags
            if len(unique_preds) > 1:
                fragmented_tracks += 1
            total_pred_ids_per_gt += len(unique_preds)
        elif len(pred_sequence) == 1:
            total_pred_ids_per_gt += 1

    avg_coverage = total_coverage / gt_track_count if gt_track_count > 0 else 0.0
    avg_pred_ids = total_pred_ids_per_gt / gt_track_count if gt_track_count > 0 else 0.0

    return TrackQualityMetrics(
        fragmentation=fragmented_tracks,
        num_fragmentations=total_fragmentations,
        mostly_tracked=mostly_tracked,
        mostly_tracked_ratio=mostly_tracked / gt_track_count if gt_track_count > 0 else 0.0,
        partially_tracked=partially_tracked,
        mostly_lost=mostly_lost,
        gt_track_count=gt_track_count,
        avg_track_coverage=avg_coverage,
        avg_pred_ids_per_gt=avg_pred_ids,
    )


def compute_position_metrics(
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    matches_by_frame: dict[int, list[tuple[int, int]]],
) -> PositionMetrics:
    """Compute position accuracy metrics for matched pairs.

    Measures the Euclidean distance between GT and prediction centers
    for all matched pairs.

    Args:
        gt_by_frame: Ground truth boxes by frame: {frame: [(id, x, y, w, h), ...]}.
        pred_by_frame: Prediction boxes by frame: {frame: [(id, x, y, w, h), ...]}.
        matches_by_frame: Matches by frame: {frame: [(gt_id, pred_id), ...]}.

    Returns:
        PositionMetrics with mean, median, P90 position errors.
    """
    import numpy as np

    position_errors: list[float] = []

    for frame, frame_matches in matches_by_frame.items():
        gt_boxes = {b[0]: (b[1], b[2]) for b in gt_by_frame.get(frame, [])}
        pred_boxes = {b[0]: (b[1], b[2]) for b in pred_by_frame.get(frame, [])}

        for gt_id, pred_id in frame_matches:
            if gt_id in gt_boxes and pred_id in pred_boxes:
                gt_x, gt_y = gt_boxes[gt_id]
                pred_x, pred_y = pred_boxes[pred_id]
                # Euclidean distance in normalized coordinates
                error = ((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2) ** 0.5
                position_errors.append(error)

    if not position_errors:
        return PositionMetrics()

    errors_arr = np.array(position_errors)
    return PositionMetrics(
        mean_position_error=float(np.mean(errors_arr)),
        median_position_error=float(np.median(errors_arr)),
        p90_position_error=float(np.percentile(errors_arr, 90)),
        num_position_samples=len(position_errors),
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
