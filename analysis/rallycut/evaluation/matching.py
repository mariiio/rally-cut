"""Rally matching algorithms for evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RallyMatch:
    """A matched pair of ground truth and predicted rally."""

    ground_truth_idx: int
    predicted_idx: int
    iou: float
    start_error_ms: int  # predicted - ground_truth (positive = late start)
    end_error_ms: int  # predicted - ground_truth (positive = late end)


@dataclass
class MatchingResult:
    """Result of matching predicted rallies to ground truth."""

    matches: list[RallyMatch]
    unmatched_ground_truth: list[int]  # Indices of missed rallies (false negatives)
    unmatched_predictions: list[int]  # Indices of false positives

    @property
    def true_positives(self) -> int:
        """Number of correctly detected rallies."""
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        """Number of extra detections (not in ground truth)."""
        return len(self.unmatched_predictions)

    @property
    def false_negatives(self) -> int:
        """Number of missed rallies."""
        return len(self.unmatched_ground_truth)


def compute_iou(
    gt_start: float,
    gt_end: float,
    pred_start: float,
    pred_end: float,
) -> float:
    """Compute Intersection over Union for two time intervals.

    Args:
        gt_start: Ground truth start time (seconds).
        gt_end: Ground truth end time (seconds).
        pred_start: Predicted start time (seconds).
        pred_end: Predicted end time (seconds).

    Returns:
        IoU value between 0.0 and 1.0.
    """
    intersection_start = max(gt_start, pred_start)
    intersection_end = min(gt_end, pred_end)
    intersection = max(0.0, intersection_end - intersection_start)

    gt_duration = gt_end - gt_start
    pred_duration = pred_end - pred_start
    union = gt_duration + pred_duration - intersection

    return intersection / union if union > 0 else 0.0


def match_rallies(
    ground_truth: list[tuple[float, float]],
    predictions: list[tuple[float, float]],
    iou_threshold: float = 0.5,
) -> MatchingResult:
    """Match predicted rallies to ground truth using IoU-based greedy matching.

    Uses greedy matching: for each ground truth rally in order, find the best
    matching prediction with IoU >= threshold. Each prediction can only match
    one ground truth.

    LIMITATION: This greedy algorithm is order-dependent and may produce suboptimal
    matches in edge cases. For example, if GT[0] has IoU 0.51 with Pred[0] and 0.9
    with Pred[1], but GT[1] has IoU 0.95 with Pred[0], the greedy approach matches
    GT[0] to Pred[1] (0.9) and GT[1] to Pred[0] (0.95), which is fine. But if the
    order were different, GT[1] might claim Pred[0] first, leaving GT[0] with a
    lower match.

    For truly optimal matching (maximizing total IoU or TP count), the Hungarian
    algorithm (scipy.optimize.linear_sum_assignment) would be needed. In practice,
    this greedy approach works well when rallies are well-separated temporally.

    Args:
        ground_truth: List of (start_seconds, end_seconds) tuples for ground truth.
        predictions: List of (start_seconds, end_seconds) tuples for predictions.
        iou_threshold: Minimum IoU to consider a match (default 0.5).

    Returns:
        MatchingResult with matches, unmatched ground truth, and unmatched predictions.
    """
    matches: list[RallyMatch] = []
    used_predictions: set[int] = set()
    unmatched_gt: list[int] = []

    # For each ground truth, find best matching prediction
    for gt_idx, (gt_start, gt_end) in enumerate(ground_truth):
        best_pred_idx: int | None = None
        best_iou = 0.0

        for pred_idx, (pred_start, pred_end) in enumerate(predictions):
            if pred_idx in used_predictions:
                continue

            iou = compute_iou(gt_start, gt_end, pred_start, pred_end)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx is not None:
            pred_start, pred_end = predictions[best_pred_idx]
            matches.append(
                RallyMatch(
                    ground_truth_idx=gt_idx,
                    predicted_idx=best_pred_idx,
                    iou=best_iou,
                    start_error_ms=int((pred_start - gt_start) * 1000),
                    end_error_ms=int((pred_end - gt_end) * 1000),
                )
            )
            used_predictions.add(best_pred_idx)
        else:
            unmatched_gt.append(gt_idx)

    # Find unmatched predictions
    unmatched_preds = [i for i in range(len(predictions)) if i not in used_predictions]

    return MatchingResult(
        matches=matches,
        unmatched_ground_truth=unmatched_gt,
        unmatched_predictions=unmatched_preds,
    )
