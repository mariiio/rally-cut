"""Unified evaluation harness for all RallyCut experiments.

Provides a single entry point for evaluating ball tracking, player detection,
rally detection, action classification, and highlight ranking across experiments.
Each eval method uses the same ground truth loading and metric computation,
ensuring consistent comparison across experiment branches.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Ball Tracking Evaluation ──────────────────────────────────────────────────


@dataclass
class BallEvalResult:
    """Ball tracking evaluation result for a single rally."""

    rally_id: str
    detection_rate: float  # % of GT frames with ball detected
    match_rate: float  # % of GT frames within threshold
    mean_error_px: float
    median_error_px: float
    p90_error_px: float
    num_gt_frames: int
    num_detected: int
    num_matched: int
    false_positive_rate: float = 0.0  # FPs / total predictions
    termination_rate: float = 0.0  # % of rally where tracking stops early

    def to_dict(self) -> dict[str, Any]:
        return {
            "rallyId": self.rally_id,
            "detectionRate": self.detection_rate,
            "matchRate": self.match_rate,
            "meanErrorPx": self.mean_error_px,
            "medianErrorPx": self.median_error_px,
            "p90ErrorPx": self.p90_error_px,
            "numGtFrames": self.num_gt_frames,
            "numDetected": self.num_detected,
            "numMatched": self.num_matched,
            "falsePositiveRate": self.false_positive_rate,
            "terminationRate": self.termination_rate,
        }


@dataclass
class BallEvalSummary:
    """Aggregated ball tracking evaluation across rallies."""

    per_rally: list[BallEvalResult] = field(default_factory=list)

    @property
    def mean_detection_rate(self) -> float:
        if not self.per_rally:
            return 0.0
        return float(np.mean([r.detection_rate for r in self.per_rally]))

    @property
    def mean_match_rate(self) -> float:
        if not self.per_rally:
            return 0.0
        return float(np.mean([r.match_rate for r in self.per_rally]))

    @property
    def mean_error_px(self) -> float:
        if not self.per_rally:
            return 0.0
        # Weighted by number of detected frames
        total_error = sum(r.mean_error_px * r.num_detected for r in self.per_rally)
        total_detected = sum(r.num_detected for r in self.per_rally)
        return total_error / total_detected if total_detected > 0 else 0.0

    @property
    def total_gt_frames(self) -> int:
        return sum(r.num_gt_frames for r in self.per_rally)

    @property
    def total_matched(self) -> int:
        return sum(r.num_matched for r in self.per_rally)

    @property
    def overall_match_rate(self) -> float:
        total = self.total_gt_frames
        return self.total_matched / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "meanDetectionRate": self.mean_detection_rate,
            "meanMatchRate": self.mean_match_rate,
            "overallMatchRate": self.overall_match_rate,
            "meanErrorPx": self.mean_error_px,
            "totalGtFrames": self.total_gt_frames,
            "totalMatched": self.total_matched,
            "numRallies": len(self.per_rally),
            "perRally": [r.to_dict() for r in self.per_rally],
        }


# ── Player Detection Evaluation ──────────────────────────────────────────────


@dataclass
class PlayerEvalResult:
    """Player detection evaluation with distance-bucketed recall."""

    rally_id: str
    # Overall metrics
    recall: float
    precision: float
    f1: float
    mota: float
    hota: float
    id_switches: int
    # Distance-bucketed recall (near = bottom 33%, mid = middle 33%, far = top 33%)
    recall_near: float  # Bottom 33% of frame (closest players)
    recall_mid: float  # Middle 33%
    recall_far: float  # Top 33% (farthest players)
    # Counts
    num_gt_detections: int = 0
    num_predictions: int = 0
    fps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rallyId": self.rally_id,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "mota": self.mota,
            "hota": self.hota,
            "idSwitches": self.id_switches,
            "recallNear": self.recall_near,
            "recallMid": self.recall_mid,
            "recallFar": self.recall_far,
            "numGtDetections": self.num_gt_detections,
            "numPredictions": self.num_predictions,
            "fps": self.fps,
        }


@dataclass
class PlayerEvalSummary:
    """Aggregated player detection evaluation."""

    per_rally: list[PlayerEvalResult] = field(default_factory=list)

    @property
    def mean_recall(self) -> float:
        return float(np.mean([r.recall for r in self.per_rally])) if self.per_rally else 0.0

    @property
    def mean_recall_near(self) -> float:
        return float(np.mean([r.recall_near for r in self.per_rally])) if self.per_rally else 0.0

    @property
    def mean_recall_mid(self) -> float:
        return float(np.mean([r.recall_mid for r in self.per_rally])) if self.per_rally else 0.0

    @property
    def mean_recall_far(self) -> float:
        return float(np.mean([r.recall_far for r in self.per_rally])) if self.per_rally else 0.0

    @property
    def mean_mota(self) -> float:
        return float(np.mean([r.mota for r in self.per_rally])) if self.per_rally else 0.0

    @property
    def mean_hota(self) -> float:
        return float(np.mean([r.hota for r in self.per_rally])) if self.per_rally else 0.0

    @property
    def total_id_switches(self) -> int:
        return sum(r.id_switches for r in self.per_rally)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meanRecall": self.mean_recall,
            "meanRecallNear": self.mean_recall_near,
            "meanRecallMid": self.mean_recall_mid,
            "meanRecallFar": self.mean_recall_far,
            "meanMOTA": self.mean_mota,
            "meanHOTA": self.mean_hota,
            "totalIdSwitches": self.total_id_switches,
            "numRallies": len(self.per_rally),
            "perRally": [r.to_dict() for r in self.per_rally],
        }


# ── Rally Detection Evaluation ───────────────────────────────────────────────


@dataclass
class RallyEvalResult:
    """Rally detection evaluation result."""

    video_id: str
    f1_at_iou: dict[float, float]  # IoU threshold -> F1 score
    boundary_error_start_ms: float  # Mean start boundary error
    boundary_error_end_ms: float  # Mean end boundary error
    num_gt_rallies: int
    num_predicted_rallies: int
    true_positives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "videoId": self.video_id,
            "f1AtIoU": {str(k): v for k, v in self.f1_at_iou.items()},
            "boundaryErrorStartMs": self.boundary_error_start_ms,
            "boundaryErrorEndMs": self.boundary_error_end_ms,
            "numGtRallies": self.num_gt_rallies,
            "numPredictedRallies": self.num_predicted_rallies,
            "truePositives": self.true_positives,
            "falsePositives": self.false_positives,
            "falseNegatives": self.false_negatives,
        }


@dataclass
class RallyEvalSummary:
    """Aggregated rally detection evaluation."""

    per_video: list[RallyEvalResult] = field(default_factory=list)

    def mean_f1_at_iou(self, iou: float) -> float:
        values = [r.f1_at_iou.get(iou, 0.0) for r in self.per_video]
        return float(np.mean(values)) if values else 0.0

    @property
    def mean_boundary_error_start_ms(self) -> float:
        return float(np.mean([r.boundary_error_start_ms for r in self.per_video])) if self.per_video else 0.0

    @property
    def mean_boundary_error_end_ms(self) -> float:
        return float(np.mean([r.boundary_error_end_ms for r in self.per_video])) if self.per_video else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "f1AtIoU0.3": self.mean_f1_at_iou(0.3),
            "f1AtIoU0.4": self.mean_f1_at_iou(0.4),
            "f1AtIoU0.5": self.mean_f1_at_iou(0.5),
            "meanBoundaryErrorStartMs": self.mean_boundary_error_start_ms,
            "meanBoundaryErrorEndMs": self.mean_boundary_error_end_ms,
            "numVideos": len(self.per_video),
            "perVideo": [r.to_dict() for r in self.per_video],
        }


# ── Action Classification Evaluation ─────────────────────────────────────────


@dataclass
class ActionEvalResult:
    """Action classification evaluation result."""

    # Per-class metrics
    per_class_f1: dict[str, float]  # action_type -> F1
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    macro_f1: float
    # Confusion matrix as nested dict: true_label -> predicted_label -> count
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    total_actions: int = 0
    correct_actions: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct_actions / self.total_actions if self.total_actions > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "perClassF1": self.per_class_f1,
            "perClassPrecision": self.per_class_precision,
            "perClassRecall": self.per_class_recall,
            "macroF1": self.macro_f1,
            "accuracy": self.accuracy,
            "totalActions": self.total_actions,
            "correctActions": self.correct_actions,
            "confusionMatrix": self.confusion_matrix,
        }


# ── Highlight Evaluation ─────────────────────────────────────────────────────


@dataclass
class HighlightEvalResult:
    """Highlight ranking evaluation result."""

    precision_at_5: float
    precision_at_10: float
    ndcg: float  # Normalized Discounted Cumulative Gain
    inter_rater_agreement: float  # Krippendorff's alpha (if multiple raters)
    num_rallies_ranked: int
    num_human_rated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "precisionAt5": self.precision_at_5,
            "precisionAt10": self.precision_at_10,
            "ndcg": self.ndcg,
            "interRaterAgreement": self.inter_rater_agreement,
            "numRalliesRanked": self.num_rallies_ranked,
            "numHumanRated": self.num_human_rated,
        }


# ── Unified Evaluation Harness ───────────────────────────────────────────────


class EvalHarness:
    """Unified evaluation harness for all RallyCut experiments.

    Provides consistent evaluation methods that work across experiment branches.
    All methods use the same ground truth loading, ensuring fair comparison.
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize evaluation harness.

        Args:
            output_dir: Directory for saving evaluation results.
                       Defaults to analysis/experiments/results/.
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def eval_ball(
        self,
        match_threshold_px: float = 50.0,
        min_confidence: float = 0.3,
        video_ids: list[str] | None = None,
    ) -> BallEvalSummary:
        """Evaluate ball tracking predictions against ground truth.

        Uses the 6 validated ball GT videos. Loads predictions from database
        and evaluates with auto frame-offset detection.

        Args:
            match_threshold_px: Max distance (px) for a "match".
            min_confidence: Min prediction confidence to consider.
            video_ids: Specific video IDs to evaluate (default: all ball GT videos).

        Returns:
            BallEvalSummary with per-rally and aggregate metrics.
        """
        from rallycut.evaluation.tracking.ball_metrics import (
            evaluate_ball_tracking,
            find_optimal_frame_offset,
        )
        from rallycut.evaluation.tracking.db import load_labeled_rallies
        from rallycut.tracking.ball_tracker import BallPosition

        rallies = load_labeled_rallies(ball_gt_only=True)
        if video_ids:
            rallies = [r for r in rallies if r.video_id in video_ids]

        summary = BallEvalSummary()

        for rally in rallies:
            gt = rally.ground_truth
            if gt is None:
                continue

            gt_ball = [p for p in gt.positions if p.label == "ball"]
            if not gt_ball:
                continue

            # Extract ball positions from predictions
            ball_preds: list[BallPosition] = []
            if rally.predictions and rally.predictions.ball_positions:
                ball_preds = rally.predictions.ball_positions

            if not ball_preds:
                continue

            # Auto-detect frame offset
            offset, _ = find_optimal_frame_offset(
                gt.positions,
                ball_preds,
                video_width=rally.video_width,
                video_height=rally.video_height,
                match_threshold_px=match_threshold_px,
            )

            # Apply offset
            if offset > 0:
                adjusted = []
                for bp in ball_preds:
                    adjusted.append(BallPosition(
                        frame_number=bp.frame_number - offset,
                        x=bp.x,
                        y=bp.y,
                        confidence=bp.confidence,
                    ))
                ball_preds = adjusted

            metrics = evaluate_ball_tracking(
                ground_truth=gt.positions,
                predictions=ball_preds,
                video_width=rally.video_width,
                video_height=rally.video_height,
                match_threshold_px=match_threshold_px,
                min_confidence=min_confidence,
            )

            # Compute termination rate
            if gt_ball:
                gt_last_frame = max(p.frame_number for p in gt_ball)
                pred_last_frame = max(
                    (bp.frame_number for bp in ball_preds if bp.confidence >= min_confidence),
                    default=0,
                )
                termination_rate = max(0.0, 1.0 - pred_last_frame / gt_last_frame) if gt_last_frame > 0 else 0.0
            else:
                termination_rate = 0.0

            summary.per_rally.append(BallEvalResult(
                rally_id=rally.rally_id,
                detection_rate=metrics.detection_rate,
                match_rate=metrics.match_rate,
                mean_error_px=metrics.mean_error_px,
                median_error_px=metrics.median_error_px,
                p90_error_px=metrics.p90_error_px,
                num_gt_frames=metrics.num_gt_frames,
                num_detected=metrics.num_detected,
                num_matched=metrics.num_matched,
                termination_rate=termination_rate,
            ))

        logger.info(
            f"Ball eval: {len(summary.per_rally)} rallies, "
            f"match rate={summary.overall_match_rate:.1%}, "
            f"mean error={summary.mean_error_px:.1f}px"
        )

        return summary

    def eval_players(
        self,
        iou_threshold: float = 0.5,
        video_ids: list[str] | None = None,
    ) -> PlayerEvalSummary:
        """Evaluate player detection with distance-bucketed recall.

        Buckets: near (bottom 33% of frame), mid (middle 33%), far (top 33%).

        Args:
            iou_threshold: IoU threshold for matching detections.
            video_ids: Specific video IDs to evaluate.

        Returns:
            PlayerEvalSummary with distance-bucketed metrics.
        """
        from rallycut.evaluation.tracking.db import load_labeled_rallies
        from rallycut.evaluation.tracking.metrics import evaluate_rally

        rallies = load_labeled_rallies()
        if video_ids:
            rallies = [r for r in rallies if r.video_id in video_ids]

        summary = PlayerEvalSummary()

        for rally in rallies:
            if rally.ground_truth is None or rally.predictions is None:
                continue

            result = evaluate_rally(
                rally_id=rally.rally_id,
                ground_truth=rally.ground_truth,
                predictions=rally.predictions,
                iou_threshold=iou_threshold,
                video_width=rally.video_width,
                video_height=rally.video_height,
            )

            # Compute distance-bucketed recall from per-frame metrics
            near_gt, near_matched = 0, 0
            mid_gt, mid_matched = 0, 0
            far_gt, far_matched = 0, 0

            for gt_pos in rally.ground_truth.positions:
                if gt_pos.label != "player":
                    continue
                # Bucket by Y position (0=top=far, 1=bottom=near)
                y = gt_pos.y
                if y >= 0.67:  # Near (bottom 33%)
                    near_gt += 1
                elif y >= 0.33:  # Mid (middle 33%)
                    mid_gt += 1
                else:  # Far (top 33%)
                    far_gt += 1

            # Check which GT were matched
            for pf in result.per_frame:
                for gt_pos in rally.ground_truth.positions:
                    if gt_pos.frame_number != pf.frame_number or gt_pos.label != "player":
                        continue
                    y = gt_pos.y
                    is_matched = pf.matches > 0
                    if y >= 0.67:
                        near_matched += is_matched
                    elif y >= 0.33:
                        mid_matched += is_matched
                    else:
                        far_matched += is_matched

            summary.per_rally.append(PlayerEvalResult(
                rally_id=rally.rally_id,
                recall=result.aggregate.recall,
                precision=result.aggregate.precision,
                f1=result.aggregate.f1,
                mota=result.aggregate.mota,
                hota=result.hota_metrics.hota if result.hota_metrics else 0.0,
                id_switches=result.aggregate.num_id_switches,
                recall_near=near_matched / near_gt if near_gt > 0 else 0.0,
                recall_mid=mid_matched / mid_gt if mid_gt > 0 else 0.0,
                recall_far=far_matched / far_gt if far_gt > 0 else 0.0,
                num_gt_detections=near_gt + mid_gt + far_gt,
                num_predictions=result.aggregate.num_pred,
            ))

        logger.info(
            f"Player eval: {len(summary.per_rally)} rallies, "
            f"recall near={summary.mean_recall_near:.1%}, "
            f"mid={summary.mean_recall_mid:.1%}, "
            f"far={summary.mean_recall_far:.1%}"
        )

        return summary

    def eval_rallies(
        self,
        iou_thresholds: list[float] | None = None,
    ) -> RallyEvalSummary:
        """Evaluate rally detection against ground truth.

        Uses LOO CV by default (train on N-1 videos, evaluate on held-out).

        Args:
            iou_thresholds: IoU thresholds for F1 computation.
                          Default: [0.3, 0.4, 0.5].

        Returns:
            RallyEvalSummary with F1 at each IoU threshold.
        """
        if iou_thresholds is None:
            iou_thresholds = [0.3, 0.4, 0.5]

        from rallycut.evaluation.ground_truth import load_evaluation_videos

        videos = load_evaluation_videos()
        summary = RallyEvalSummary()

        for video in videos:
            gt_rallies = video.ground_truth_rallies
            pred_rallies = video.ml_detected_rallies

            if not gt_rallies or not pred_rallies:
                continue

            f1_at_iou: dict[float, float] = {}
            best_tp, best_fp, best_fn = 0, 0, 0

            for iou_thresh in iou_thresholds:
                tp, fp, fn = 0, 0, 0
                matched_gt: set[int] = set()

                for pred in pred_rallies:
                    best_iou = 0.0
                    best_gt_idx = -1
                    for gi, gt in enumerate(gt_rallies):
                        if gi in matched_gt:
                            continue
                        # Compute temporal IoU
                        overlap_start = max(pred.start_ms, gt.start_ms)
                        overlap_end = min(pred.end_ms, gt.end_ms)
                        overlap = max(0, overlap_end - overlap_start)
                        union = (pred.end_ms - pred.start_ms) + (gt.end_ms - gt.start_ms) - overlap
                        iou = overlap / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gi

                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1

                fn = len(gt_rallies) - len(matched_gt)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                f1_at_iou[iou_thresh] = f1

                if iou_thresh == 0.4:
                    best_tp, best_fp, best_fn = tp, fp, fn

            # Compute boundary errors for matched rallies at IoU=0.4
            boundary_errors_start: list[float] = []
            boundary_errors_end: list[float] = []

            matched_gt_set: set[int] = set()
            for pred in pred_rallies:
                for gi, gt in enumerate(gt_rallies):
                    if gi in matched_gt_set:
                        continue
                    overlap_start = max(pred.start_ms, gt.start_ms)
                    overlap_end = min(pred.end_ms, gt.end_ms)
                    overlap = max(0, overlap_end - overlap_start)
                    union = (pred.end_ms - pred.start_ms) + (gt.end_ms - gt.start_ms) - overlap
                    iou = overlap / union if union > 0 else 0.0
                    if iou >= 0.4:
                        boundary_errors_start.append(abs(pred.start_ms - gt.start_ms))
                        boundary_errors_end.append(abs(pred.end_ms - gt.end_ms))
                        matched_gt_set.add(gi)
                        break

            summary.per_video.append(RallyEvalResult(
                video_id=video.id,
                f1_at_iou=f1_at_iou,
                boundary_error_start_ms=float(np.mean(boundary_errors_start)) if boundary_errors_start else 0.0,
                boundary_error_end_ms=float(np.mean(boundary_errors_end)) if boundary_errors_end else 0.0,
                num_gt_rallies=len(gt_rallies),
                num_predicted_rallies=len(pred_rallies),
                true_positives=best_tp,
                false_positives=best_fp,
                false_negatives=best_fn,
            ))

        logger.info(
            f"Rally eval: {len(summary.per_video)} videos, "
            f"F1@0.4={summary.mean_f1_at_iou(0.4):.1%}, "
            f"F1@0.5={summary.mean_f1_at_iou(0.5):.1%}"
        )

        return summary

    def eval_actions(
        self,
        predictions: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
        action_classes: list[str] | None = None,
    ) -> ActionEvalResult:
        """Evaluate action classification predictions against ground truth.

        Args:
            predictions: List of {"frame": int, "action": str, "player_id": int}.
            ground_truth: List of {"frame": int, "action": str, "player_id": int}.
            action_classes: Action classes to evaluate. Default: serve, receive, set, spike, block.

        Returns:
            ActionEvalResult with per-class and aggregate metrics.
        """
        if action_classes is None:
            action_classes = ["serve", "receive", "set", "spike", "block"]

        # Match predictions to GT by frame proximity (±5 frames)
        matched_pairs: list[tuple[str, str]] = []  # (gt_action, pred_action)
        used_pred: set[int] = set()

        for gt in ground_truth:
            gt_frame = gt["frame"]
            gt_action = gt["action"]
            best_pred_idx = -1
            best_dist = float("inf")

            for pi, pred in enumerate(predictions):
                if pi in used_pred:
                    continue
                dist = abs(pred["frame"] - gt_frame)
                if dist <= 5 and dist < best_dist:
                    best_dist = dist
                    best_pred_idx = pi

            if best_pred_idx >= 0:
                pred_action = predictions[best_pred_idx]["action"]
                matched_pairs.append((gt_action, pred_action))
                used_pred.add(best_pred_idx)
            else:
                matched_pairs.append((gt_action, "missed"))

        # Compute per-class metrics
        per_class_tp: dict[str, int] = {c: 0 for c in action_classes}
        per_class_fp: dict[str, int] = {c: 0 for c in action_classes}
        per_class_fn: dict[str, int] = {c: 0 for c in action_classes}
        confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in action_classes + ["missed"]} for c in action_classes}

        correct = 0
        for gt_action, pred_action in matched_pairs:
            if gt_action in confusion and pred_action in confusion[gt_action]:
                confusion[gt_action][pred_action] += 1

            if gt_action == pred_action:
                correct += 1
                if gt_action in per_class_tp:
                    per_class_tp[gt_action] += 1
            else:
                if gt_action in per_class_fn:
                    per_class_fn[gt_action] += 1
                if pred_action in per_class_fp:
                    per_class_fp[pred_action] += 1

        per_class_f1: dict[str, float] = {}
        per_class_precision: dict[str, float] = {}
        per_class_recall: dict[str, float] = {}

        for c in action_classes:
            tp = per_class_tp[c]
            fp = per_class_fp[c]
            fn = per_class_fn[c]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_class_precision[c] = p
            per_class_recall[c] = r
            per_class_f1[c] = f1

        macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0

        return ActionEvalResult(
            per_class_f1=per_class_f1,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            macro_f1=macro_f1,
            confusion_matrix=confusion,
            total_actions=len(matched_pairs),
            correct_actions=correct,
        )

    def eval_highlights(
        self,
        ranked_rally_ids: list[str],
        human_ratings: dict[str, list[float]],
    ) -> HighlightEvalResult:
        """Evaluate highlight ranking against human ratings.

        Args:
            ranked_rally_ids: Rally IDs ranked by predicted excitement (best first).
            human_ratings: Rally ID -> list of human ratings (1-5 scale, multiple raters).

        Returns:
            HighlightEvalResult with Precision@K and NDCG.
        """
        # Define "exciting" threshold (rating >= 4 out of 5)
        exciting_threshold = 4.0

        def is_exciting(rally_id: str) -> bool:
            ratings = human_ratings.get(rally_id, [])
            if not ratings:
                return False
            return float(np.mean(ratings)) >= exciting_threshold

        # Precision@5
        top5 = ranked_rally_ids[:5]
        p5_hits = sum(1 for r in top5 if is_exciting(r))
        precision_at_5 = p5_hits / min(5, len(top5)) if top5 else 0.0

        # Precision@10
        top10 = ranked_rally_ids[:10]
        p10_hits = sum(1 for r in top10 if is_exciting(r))
        precision_at_10 = p10_hits / min(10, len(top10)) if top10 else 0.0

        # NDCG (using mean human rating as relevance)
        dcg = 0.0
        idcg = 0.0

        relevances = []
        for rally_id in ranked_rally_ids:
            ratings = human_ratings.get(rally_id, [])
            rel = float(np.mean(ratings)) if ratings else 0.0
            relevances.append(rel)

        # DCG
        for i, rel in enumerate(relevances):
            dcg += rel / np.log2(i + 2)  # i+2 because rank starts at 1

        # Ideal DCG (sort by relevance descending)
        ideal_relevances = sorted(relevances, reverse=True)
        for i, rel in enumerate(ideal_relevances):
            idcg += rel / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        # Inter-rater agreement (Krippendorff's alpha simplified)
        # Full implementation requires ordinal alpha; use simplified version
        all_ratings: list[list[float]] = []
        for rally_id in ranked_rally_ids:
            ratings = human_ratings.get(rally_id, [])
            if ratings:
                all_ratings.append(ratings)

        if all_ratings and len(all_ratings[0]) > 1:
            # Simplified agreement: mean pairwise correlation
            num_raters = len(all_ratings[0])
            agreements = []
            for i in range(num_raters):
                for j in range(i + 1, num_raters):
                    rater_i = [r[i] if i < len(r) else 0 for r in all_ratings]
                    rater_j = [r[j] if j < len(r) else 0 for r in all_ratings]
                    if len(set(rater_i)) > 1 and len(set(rater_j)) > 1:
                        corr = float(np.corrcoef(rater_i, rater_j)[0, 1])
                        if not np.isnan(corr):
                            agreements.append(corr)
            inter_rater = float(np.mean(agreements)) if agreements else 0.0
        else:
            inter_rater = 0.0

        return HighlightEvalResult(
            precision_at_5=precision_at_5,
            precision_at_10=precision_at_10,
            ndcg=float(ndcg),
            inter_rater_agreement=inter_rater,
            num_rallies_ranked=len(ranked_rally_ids),
            num_human_rated=len(human_ratings),
        )

    def save_results(self, name: str, results: dict[str, Any]) -> Path:
        """Save evaluation results to JSON.

        Args:
            name: Experiment name (e.g., "exp-ball-tracknet-v1").
            results: Results dict to save.

        Returns:
            Path to saved results file.
        """
        output_path = self.output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
        return output_path

    def compare_results(self, *result_paths: Path) -> dict[str, Any]:
        """Compare multiple experiment results side by side.

        Args:
            result_paths: Paths to result JSON files.

        Returns:
            Comparison dict with aligned metrics.
        """
        comparison: dict[str, Any] = {"experiments": []}

        for path in result_paths:
            with open(path) as f:
                data = json.load(f)
            comparison["experiments"].append({
                "name": path.stem,
                "results": data,
            })

        return comparison
