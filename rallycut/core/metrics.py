"""Quality metrics for evaluating rally detection accuracy."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RallyAnnotation:
    """A ground truth rally annotation."""

    start_seconds: float
    end_seconds: float
    actions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class ClassMetrics:
    """Precision/recall/F1 metrics for a single class."""

    class_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class QualityMetrics:
    """Complete quality metrics for rally detection."""

    # Segment-level metrics
    rally_recall: float = 0.0  # % of ground truth rally time covered
    rally_precision: float = 0.0  # % of detected play time that is actual rally
    f1_score: float = 0.0

    # Boundary accuracy
    avg_start_error_seconds: float = 0.0
    avg_end_error_seconds: float = 0.0

    # Per-rally coverage
    rallies_detected: int = 0
    rallies_missed: int = 0
    per_rally_coverage: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rally_recall": round(self.rally_recall, 4),
            "rally_precision": round(self.rally_precision, 4),
            "f1_score": round(self.f1_score, 4),
            "avg_start_error_seconds": round(self.avg_start_error_seconds, 2),
            "avg_end_error_seconds": round(self.avg_end_error_seconds, 2),
            "rallies_detected": self.rallies_detected,
            "rallies_missed": self.rallies_missed,
            "per_rally_coverage": self.per_rally_coverage,
        }


def parse_time(time_str: str) -> float:
    """Parse time string like '1:30' or '0:12.5' to seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    return float(time_str)


def load_ground_truth(
    json_path: Path, video_name: str | None = None
) -> dict[str, list[RallyAnnotation]]:
    """
    Load ground truth rally annotations from JSON file.

    Args:
        json_path: Path to ground truth JSON file
        video_name: Optional specific video to load (returns dict with single key)

    Returns:
        Dict mapping video names to lists of RallyAnnotation
    """
    with open(json_path) as f:
        data = json.load(f)

    result: dict[str, list[RallyAnnotation]] = {}

    for vid_name, video_data in data.items():
        if video_name and vid_name != video_name:
            continue

        rallies = []
        for rally in video_data.get("rallies", []):
            start = parse_time(str(rally["start"]))
            end = parse_time(str(rally["end"]))
            actions = rally.get("actions", [])
            rallies.append(RallyAnnotation(start, end, actions))

        result[vid_name] = rallies

    return result


def calculate_quality_metrics(
    detected_segments: list[Any],  # TimeSegment
    ground_truth: list[RallyAnnotation],
    video_duration: float,
    resolution: float = 0.1,  # 0.1s resolution for time masks
) -> QualityMetrics:
    """
    Calculate comprehensive quality metrics comparing detected segments to ground truth.

    Uses time-mask approach at specified resolution for accurate overlap calculation.

    Args:
        detected_segments: List of TimeSegment from analysis
        ground_truth: List of RallyAnnotation from ground truth
        video_duration: Total video duration in seconds
        resolution: Time resolution for comparison (default 0.1s)

    Returns:
        QualityMetrics with precision, recall, F1, and per-rally coverage
    """
    if not ground_truth:
        return QualityMetrics()

    # Create time masks at given resolution
    total_bins = int(video_duration / resolution)

    # Ground truth mask
    gt_bins: set[int] = set()
    for rally in ground_truth:
        start_bin = int(rally.start_seconds / resolution)
        end_bin = int(rally.end_seconds / resolution)
        for t in range(start_bin, end_bin):
            if 0 <= t < total_bins:
                gt_bins.add(t)

    # Detected segments mask (only PLAY/SERVICE states)
    detected_bins: set[int] = set()
    for seg in detected_segments:
        # Check if this segment represents active play
        if hasattr(seg, "state"):
            state_val = seg.state.value if hasattr(seg.state, "value") else str(seg.state)
            if state_val.lower() not in ("play", "service"):
                continue

        start_bin = int(seg.start_time / resolution)
        end_bin = int(seg.end_time / resolution)
        for t in range(start_bin, end_bin):
            if 0 <= t < total_bins:
                detected_bins.add(t)

    # Calculate overlap metrics
    intersection = gt_bins & detected_bins

    rally_recall = len(intersection) / len(gt_bins) if gt_bins else 0.0
    rally_precision = len(intersection) / len(detected_bins) if detected_bins else 0.0

    # F1 score
    f1 = 0.0
    if rally_recall + rally_precision > 0:
        f1 = 2 * rally_recall * rally_precision / (rally_recall + rally_precision)

    # Per-rally coverage
    per_rally_coverage = []
    rallies_detected = 0
    rallies_missed = 0
    start_errors = []
    end_errors = []

    for rally in ground_truth:
        rally_bins: set[int] = set()
        start_bin = int(rally.start_seconds / resolution)
        end_bin = int(rally.end_seconds / resolution)
        for t in range(start_bin, end_bin):
            if 0 <= t < total_bins:
                rally_bins.add(t)

        covered = rally_bins & detected_bins
        coverage = len(covered) / len(rally_bins) if rally_bins else 0.0

        per_rally_coverage.append({
            "start": rally.start_seconds,
            "end": rally.end_seconds,
            "duration": rally.duration,
            "coverage": round(coverage, 3),
        })

        # Count as detected if >50% covered
        if coverage >= 0.5:
            rallies_detected += 1

            # Calculate boundary errors for well-detected rallies
            if covered:
                detected_start = min(covered) * resolution
                detected_end = max(covered) * resolution
                start_errors.append(abs(detected_start - rally.start_seconds))
                end_errors.append(abs(detected_end - rally.end_seconds))
        else:
            rallies_missed += 1

    # Average boundary errors
    avg_start_error = sum(start_errors) / len(start_errors) if start_errors else 0.0
    avg_end_error = sum(end_errors) / len(end_errors) if end_errors else 0.0

    return QualityMetrics(
        rally_recall=rally_recall,
        rally_precision=rally_precision,
        f1_score=f1,
        avg_start_error_seconds=avg_start_error,
        avg_end_error_seconds=avg_end_error,
        rallies_detected=rallies_detected,
        rallies_missed=rallies_missed,
        per_rally_coverage=per_rally_coverage,
    )


def print_quality_metrics(metrics: QualityMetrics) -> None:
    """Print a formatted quality metrics report."""
    print("\n" + "=" * 50)
    print("Quality Metrics")
    print("=" * 50)

    print(f"\nOverall Accuracy:")
    print(f"  Rally Recall:     {metrics.rally_recall * 100:.1f}%")
    print(f"  Rally Precision:  {metrics.rally_precision * 100:.1f}%")
    print(f"  F1 Score:         {metrics.f1_score * 100:.1f}%")

    print(f"\nBoundary Accuracy:")
    print(f"  Avg Start Error:  {metrics.avg_start_error_seconds:.2f}s")
    print(f"  Avg End Error:    {metrics.avg_end_error_seconds:.2f}s")

    print(f"\nRally Detection:")
    total = metrics.rallies_detected + metrics.rallies_missed
    print(f"  Detected:         {metrics.rallies_detected}/{total}")
    print(f"  Missed:           {metrics.rallies_missed}/{total}")

    if metrics.per_rally_coverage:
        print(f"\nPer-Rally Coverage:")
        for i, rally in enumerate(metrics.per_rally_coverage, 1):
            status = "OK" if rally["coverage"] >= 0.5 else "MISS"
            print(
                f"  Rally {i} ({rally['start']:.0f}s-{rally['end']:.0f}s): "
                f"{rally['coverage'] * 100:.0f}% [{status}]"
            )

    print("=" * 50 + "\n")
