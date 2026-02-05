"""Calibration metrics for probability predictions.

Measures how well the model's confidence scores match actual accuracy.
A well-calibrated model should have:
- 90% accuracy when it predicts with 90% confidence
- 70% accuracy when it predicts with 70% confidence
- etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationMetrics:
    """Calibration metrics for probability predictions."""

    brier_score: float
    expected_calibration_error: float  # ECE
    max_calibration_error: float  # MCE
    reliability_diagram: dict[str, list[float]]


def compute_brier_score(
    probs: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> float:
    """Compute Brier score for probability predictions.

    The Brier score is the mean squared error between predicted probabilities
    and actual outcomes. Lower is better, 0 is perfect.

    Args:
        probs: Predicted probabilities for class 1.
        labels: True labels (0 or 1).

    Returns:
        Brier score (0 to 1, lower is better).
    """
    probs_arr = np.array(probs)
    labels_arr = np.array(labels)
    return float(np.mean((probs_arr - labels_arr) ** 2))


def compute_calibration_error(
    probs: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute calibration error metrics.

    Uses binning to compute Expected Calibration Error (ECE) and
    Maximum Calibration Error (MCE).

    Args:
        probs: Predicted probabilities for class 1.
        labels: True labels (0 or 1).
        n_bins: Number of bins for calibration curve.

    Returns:
        CalibrationMetrics with ECE, MCE, and reliability diagram data.
    """
    probs_arr = np.array(probs)
    labels_arr = np.array(labels)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs_arr, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute per-bin statistics
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = bin_indices == i
        bin_count = in_bin.sum()
        bin_counts.append(int(bin_count))

        if bin_count > 0:
            avg_confidence = probs_arr[in_bin].mean()
            accuracy = labels_arr[in_bin].mean()
            bin_confidences.append(float(avg_confidence))
            bin_accuracies.append(float(accuracy))
        else:
            # Empty bin
            bin_confidences.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(0.0)

    # Compute ECE (weighted average of absolute calibration errors)
    total = len(probs_arr)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        weight = bin_counts[i] / total if total > 0 else 0
        abs_error = abs(bin_confidences[i] - bin_accuracies[i])
        ece += weight * abs_error
        if bin_counts[i] > 0:
            mce = max(mce, abs_error)

    # Brier score
    brier = compute_brier_score(probs_arr, labels_arr)

    return CalibrationMetrics(
        brier_score=brier,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        reliability_diagram={
            "bin_edges": [float(x) for x in bin_edges],
            "confidences": bin_confidences,
            "accuracies": bin_accuracies,
            "counts": bin_counts,
        },
    )


def format_calibration_report(metrics: CalibrationMetrics) -> str:
    """Format calibration metrics as a human-readable report.

    Args:
        metrics: Calibration metrics to format.

    Returns:
        Formatted string report.
    """
    lines = [
        "Calibration Metrics:",
        f"  Brier Score: {metrics.brier_score:.4f} (lower is better)",
        f"  ECE (Expected Calibration Error): {metrics.expected_calibration_error:.4f}",
        f"  MCE (Maximum Calibration Error): {metrics.max_calibration_error:.4f}",
        "",
        "Reliability Diagram:",
        "  Confidence | Accuracy | Count",
        "  -----------|----------|------",
    ]

    diagram = metrics.reliability_diagram
    for i in range(len(diagram["confidences"])):
        conf = diagram["confidences"][i]
        acc = diagram["accuracies"][i]
        count = diagram["counts"][i]
        lines.append(f"     {conf:5.2f}   |   {acc:5.2f}  | {count:5d}")

    return "\n".join(lines)
