"""Evaluation metrics for comparing model predictions against ground truth."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassMetrics:
    """Metrics for a single action class."""

    class_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Correct detections / total detections."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Correct detections / total ground truth."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    class_metrics: dict[str, ClassMetrics]
    frame_tolerance: int
    total_ground_truth: int
    total_predictions: int

    @property
    def overall_precision(self) -> float:
        """Macro-averaged precision across all classes."""
        precisions = [m.precision for m in self.class_metrics.values() if m.true_positives + m.false_positives > 0]
        return sum(precisions) / len(precisions) if precisions else 0.0

    @property
    def overall_recall(self) -> float:
        """Macro-averaged recall across all classes."""
        recalls = [m.recall for m in self.class_metrics.values() if m.true_positives + m.false_negatives > 0]
        return sum(recalls) / len(recalls) if recalls else 0.0

    @property
    def overall_f1(self) -> float:
        """Macro-averaged F1 score."""
        if self.overall_precision + self.overall_recall == 0:
            return 0.0
        return 2 * (self.overall_precision * self.overall_recall) / (self.overall_precision + self.overall_recall)


def evaluate_predictions(
    ground_truth: list[dict],
    predictions: list[dict],
    frame_tolerance: int = 15,
    class_mapping: Optional[dict[str, str]] = None,
) -> EvaluationResult:
    """
    Compare model predictions against ground truth annotations.

    Args:
        ground_truth: List of annotations with 'frame' and 'model_class' keys
        predictions: List of detections with 'frame' and 'type' keys
        frame_tolerance: Maximum frame difference for a match (default ±15 frames)
        class_mapping: Optional mapping from prediction type to ground truth class

    Returns:
        EvaluationResult with per-class and overall metrics
    """
    # Normalize class names
    if class_mapping is None:
        class_mapping = {
            "reception": "receive",
            "attack": "spike",
        }

    # Get all unique classes from ground truth
    gt_classes = set(ann["model_class"] for ann in ground_truth)

    # Initialize metrics for each class
    class_metrics = {cls: ClassMetrics(class_name=cls) for cls in gt_classes}

    # Track which ground truth annotations have been matched
    gt_matched = [False] * len(ground_truth)

    # Track which predictions have been matched
    pred_matched = [False] * len(predictions)

    # For each ground truth annotation, find best matching prediction
    for gt_idx, gt_ann in enumerate(ground_truth):
        gt_frame = gt_ann["frame"]
        gt_class = gt_ann["model_class"]

        best_match_idx = None
        best_match_dist = float("inf")

        for pred_idx, pred in enumerate(predictions):
            if pred_matched[pred_idx]:
                continue

            # Map prediction type to ground truth class
            pred_type = pred["type"]
            pred_class = class_mapping.get(pred_type, pred_type)

            # Check if classes match
            if pred_class != gt_class:
                continue

            # Check if within frame tolerance
            pred_frame = pred["frame"]
            frame_dist = abs(pred_frame - gt_frame)

            if frame_dist <= frame_tolerance and frame_dist < best_match_dist:
                best_match_idx = pred_idx
                best_match_dist = frame_dist

        if best_match_idx is not None:
            # True positive
            class_metrics[gt_class].true_positives += 1
            gt_matched[gt_idx] = True
            pred_matched[best_match_idx] = True
        else:
            # False negative (ground truth not detected)
            class_metrics[gt_class].false_negatives += 1

    # Count false positives (predictions not matched to any ground truth)
    for pred_idx, pred in enumerate(predictions):
        if pred_matched[pred_idx]:
            continue

        pred_type = pred["type"]
        pred_class = class_mapping.get(pred_type, pred_type)

        # Add to class metrics if class exists, otherwise create new
        if pred_class not in class_metrics:
            class_metrics[pred_class] = ClassMetrics(class_name=pred_class)

        class_metrics[pred_class].false_positives += 1

    return EvaluationResult(
        class_metrics=class_metrics,
        frame_tolerance=frame_tolerance,
        total_ground_truth=len(ground_truth),
        total_predictions=len(predictions),
    )


def format_evaluation_report(result: EvaluationResult) -> str:
    """Format evaluation results as a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("MODEL EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Frame tolerance: ±{result.frame_tolerance} frames")
    lines.append(f"Ground truth actions: {result.total_ground_truth}")
    lines.append(f"Model predictions: {result.total_predictions}")
    lines.append("")

    # Per-class metrics
    lines.append("Per-Class Metrics:")
    lines.append("-" * 60)
    lines.append(f"{'Class':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    lines.append("-" * 60)

    for cls_name in sorted(result.class_metrics.keys()):
        m = result.class_metrics[cls_name]
        lines.append(
            f"{cls_name:<12} {m.true_positives:>4} {m.false_positives:>4} {m.false_negatives:>4} "
            f"{m.precision:>7.2%} {m.recall:>7.2%} {m.f1_score:>7.2%}"
        )

    lines.append("-" * 60)
    lines.append("")

    # Overall metrics
    lines.append("Overall Metrics (macro-averaged):")
    lines.append(f"  Precision: {result.overall_precision:.2%}")
    lines.append(f"  Recall:    {result.overall_recall:.2%}")
    lines.append(f"  F1 Score:  {result.overall_f1:.2%}")
    lines.append("=" * 60)

    return "\n".join(lines)
