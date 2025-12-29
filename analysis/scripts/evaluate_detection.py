#!/usr/bin/env python3
"""Evaluate ML rally detection against ground truth labels.

Compares model predictions to ground truth segments and computes
frame-level accuracy, per-class metrics, and boundary errors.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.analysis.game_state import GameStateAnalyzer
from rallycut.core.config import (
    GameStateConfig,
    RallyCutConfig,
    get_config,
    reset_config,
    set_config,
)
from rallycut.core.models import GameStateResult
from rallycut.core.video import Video

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GroundTruthSegment:
    """A labeled segment from ground truth."""

    start: float
    end: float
    label: str  # "service", "play", "no_play"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ClassMetrics:
    """Precision/Recall/F1 for a single class."""

    name: str
    true_positives: float = 0.0  # seconds
    false_positives: float = 0.0  # seconds
    false_negatives: float = 0.0  # seconds

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class BoundaryError:
    """Error in detecting a state transition."""

    ground_truth_time: float
    predicted_time: float | None
    transition_type: str  # e.g., "no_play->service", "play->no_play"

    @property
    def error(self) -> float | None:
        if self.predicted_time is None:
            return None
        return self.predicted_time - self.ground_truth_time


@dataclass
class EvaluationResult:
    """Complete evaluation results for a video."""

    video_name: str
    duration: float

    # Frame-level accuracy
    total_seconds: float
    correct_seconds: float
    accuracy: float

    # Per-class metrics (2-class: active vs no_play)
    active_metrics: ClassMetrics  # service + play combined
    no_play_metrics: ClassMetrics

    # Per-class metrics (3-class)
    service_metrics: ClassMetrics
    play_metrics: ClassMetrics
    no_play_3class_metrics: ClassMetrics

    # Boundary errors
    boundary_errors: list[BoundaryError] = field(default_factory=list)

    # Rally-level metrics
    expected_rally_count: int = 0
    detected_rally_count: int = 0
    matched_rallies: int = 0

    @property
    def avg_boundary_error(self) -> float | None:
        errors = [abs(e.error) for e in self.boundary_errors if e.error is not None]
        return sum(errors) / len(errors) if errors else None

    @property
    def missed_boundaries(self) -> int:
        return sum(1 for e in self.boundary_errors if e.error is None)


# =============================================================================
# Ground Truth Loader
# =============================================================================


def load_ground_truth(json_path: Path) -> dict[str, dict[str, object]]:
    """Load ground truth from JSON file."""
    with open(json_path) as f:
        data: dict[str, dict[str, object]] = json.load(f)
        return data


def parse_segments(data: dict) -> list[GroundTruthSegment]:
    """Parse segments from ground truth data."""
    return [
        GroundTruthSegment(start=s["start"], end=s["end"], label=s["label"])
        for s in data.get("segments", [])
    ]


# =============================================================================
# Prediction Generator
# =============================================================================


def get_predictions(
    video_path: Path,
    stride: int | None = None,
    limit_seconds: float | None = None,
) -> tuple[list[GameStateResult], float]:
    """Run ML analysis and return predictions with FPS."""
    config = get_config()
    stride = stride or config.game_state.stride

    analyzer = GameStateAnalyzer()
    video = Video(video_path)
    fps = video.info.fps

    results = analyzer.analyze_video(
        video,
        stride=stride,
        limit_seconds=limit_seconds,
        return_raw=False,
    )

    # analyze_video returns list when return_raw=False
    assert isinstance(results, list)
    return results, fps


def predictions_to_time_labels(
    results: list[GameStateResult],
    fps: float,
    duration: float,
    resolution: float = 0.1,  # 100ms resolution
) -> list[tuple[float, str]]:
    """Convert frame-based predictions to time-based labels.

    Returns list of (time, label) tuples at given resolution.
    """
    if not results:
        return []

    labels = []
    t = 0.0

    while t < duration:
        # Find the prediction that covers this time
        frame_idx = int(t * fps)
        label = "no_play"  # default

        for result in results:
            if result.start_frame is not None and result.end_frame is not None:
                if result.start_frame <= frame_idx <= result.end_frame:
                    label = result.state.value
                    break

        labels.append((t, label))
        t += resolution

    return labels


# =============================================================================
# Evaluation Logic
# =============================================================================


def get_ground_truth_label(
    time: float,
    segments: list[GroundTruthSegment],
) -> str:
    """Get ground truth label at a specific time."""
    for seg in segments:
        if seg.start <= time < seg.end:
            return seg.label
    return "no_play"  # Default for gaps


def normalize_label(label: str, mode: Literal["2class", "3class"]) -> str:
    """Normalize label for comparison.

    2class: service and play -> "active", no_play -> "no_play"
    3class: keep original labels
    """
    if mode == "2class":
        if label in ("service", "play"):
            return "active"
        return "no_play"
    return label


def compute_class_metrics(
    time_labels: list[tuple[float, str]],
    segments: list[GroundTruthSegment],
    class_name: str,
    mode: Literal["2class", "3class"],
    resolution: float,
) -> ClassMetrics:
    """Compute precision/recall/F1 for a single class."""
    metrics = ClassMetrics(name=class_name)

    for time, pred_label in time_labels:
        gt_label = get_ground_truth_label(time, segments)

        pred_normalized = normalize_label(pred_label, mode)
        gt_normalized = normalize_label(gt_label, mode)

        pred_is_class = pred_normalized == class_name
        gt_is_class = gt_normalized == class_name

        if pred_is_class and gt_is_class:
            metrics.true_positives += resolution
        elif pred_is_class and not gt_is_class:
            metrics.false_positives += resolution
        elif not pred_is_class and gt_is_class:
            metrics.false_negatives += resolution

    return metrics


def find_boundaries(segments: list[GroundTruthSegment]) -> list[tuple[float, str]]:
    """Find state transition boundaries in ground truth.

    Returns list of (time, transition_type) tuples.
    """
    boundaries = []
    prev_label = "no_play"

    for seg in sorted(segments, key=lambda s: s.start):
        if seg.label != prev_label:
            boundaries.append((seg.start, f"{prev_label}->{seg.label}"))
        prev_label = seg.label

    return boundaries


def find_predicted_boundaries(
    time_labels: list[tuple[float, str]],
    mode: Literal["2class", "3class"],
) -> list[tuple[float, str]]:
    """Find state transitions in predictions."""
    if not time_labels:
        return []

    boundaries = []
    prev_label = normalize_label(time_labels[0][1], mode)

    for time, label in time_labels[1:]:
        curr_label = normalize_label(label, mode)
        if curr_label != prev_label:
            boundaries.append((time, f"{prev_label}->{curr_label}"))
            prev_label = curr_label

    return boundaries


def match_boundaries(
    gt_boundaries: list[tuple[float, str]],
    pred_boundaries: list[tuple[float, str]],
    tolerance: float = 3.0,
    mode: Literal["2class", "3class"] = "2class",
) -> list[BoundaryError]:
    """Match predicted boundaries to ground truth boundaries."""
    errors = []

    for gt_time, gt_type in gt_boundaries:
        # Normalize transition type for matching
        if mode == "2class":
            # Convert service/play to active in transition names
            gt_type_norm = gt_type.replace("service", "active").replace("play", "active")
            # Deduplicate (active->active becomes meaningless)
            parts = gt_type_norm.split("->")
            if parts[0] == parts[1]:
                continue
        else:
            gt_type_norm = gt_type

        # Find closest matching prediction
        best_match = None
        best_dist = float("inf")

        for pred_time, pred_type in pred_boundaries:
            if mode == "2class":
                pred_type_norm = pred_type.replace("service", "active").replace("play", "active")
            else:
                pred_type_norm = pred_type

            # Must be same transition type
            if pred_type_norm != gt_type_norm:
                continue

            dist = abs(pred_time - gt_time)
            if dist < best_dist and dist <= tolerance:
                best_dist = dist
                best_match = pred_time

        errors.append(BoundaryError(
            ground_truth_time=gt_time,
            predicted_time=best_match,
            transition_type=gt_type,
        ))

    return errors


def evaluate_video(
    video_path: Path,
    segments: list[GroundTruthSegment],
    duration: float,
    stride: int | None = None,
    limit_seconds: float | None = None,
    resolution: float = 0.1,
    boundary_tolerance: float = 3.0,
) -> EvaluationResult:
    """Evaluate predictions against ground truth for a single video."""
    # Get predictions
    results, fps = get_predictions(video_path, stride=stride, limit_seconds=limit_seconds)

    # Convert to time-based labels
    eval_duration = min(duration, limit_seconds) if limit_seconds else duration
    time_labels = predictions_to_time_labels(results, fps, eval_duration, resolution)

    # Compute frame-level accuracy
    correct = 0.0
    total = 0.0

    for time, pred_label in time_labels:
        gt_label = get_ground_truth_label(time, segments)

        # Use 2-class comparison for overall accuracy
        pred_norm = normalize_label(pred_label, "2class")
        gt_norm = normalize_label(gt_label, "2class")

        if pred_norm == gt_norm:
            correct += resolution
        total += resolution

    accuracy = correct / total if total > 0 else 0.0

    # Compute per-class metrics (2-class)
    active_metrics = compute_class_metrics(time_labels, segments, "active", "2class", resolution)
    no_play_metrics = compute_class_metrics(time_labels, segments, "no_play", "2class", resolution)

    # Compute per-class metrics (3-class)
    service_metrics = compute_class_metrics(time_labels, segments, "service", "3class", resolution)
    play_metrics = compute_class_metrics(time_labels, segments, "play", "3class", resolution)
    no_play_3class = compute_class_metrics(time_labels, segments, "no_play", "3class", resolution)

    # Compute boundary errors
    gt_boundaries = find_boundaries(segments)
    pred_boundaries = find_predicted_boundaries(time_labels, "2class")
    boundary_errors = match_boundaries(gt_boundaries, pred_boundaries, boundary_tolerance, "2class")

    # Count rallies (service or play segments)
    expected_rallies = sum(1 for s in segments if s.label == "service")

    return EvaluationResult(
        video_name=video_path.name,
        duration=eval_duration,
        total_seconds=total,
        correct_seconds=correct,
        accuracy=accuracy,
        active_metrics=active_metrics,
        no_play_metrics=no_play_metrics,
        service_metrics=service_metrics,
        play_metrics=play_metrics,
        no_play_3class_metrics=no_play_3class,
        boundary_errors=boundary_errors,
        expected_rally_count=expected_rallies,
    )


# =============================================================================
# Reporting
# =============================================================================


def print_report(results: list[EvaluationResult], detailed: bool = False) -> None:
    """Print evaluation report."""
    print("=" * 70)
    print("RALLY DETECTION EVALUATION REPORT")
    print("=" * 70)

    for result in results:
        print(f"\n{result.video_name}")
        print("-" * 50)

        # Overall accuracy
        print(f"  Overall Accuracy:     {result.accuracy:6.1%} ({result.correct_seconds:.1f}s / {result.total_seconds:.1f}s)")

        # 2-class metrics
        print("\n  2-Class Metrics (active vs no_play):")
        print("    Active (service+play):")
        print(f"      Precision: {result.active_metrics.precision:6.1%}")
        print(f"      Recall:    {result.active_metrics.recall:6.1%}")
        print(f"      F1:        {result.active_metrics.f1:6.1%}")
        print("    No Play:")
        print(f"      Precision: {result.no_play_metrics.precision:6.1%}")
        print(f"      Recall:    {result.no_play_metrics.recall:6.1%}")
        print(f"      F1:        {result.no_play_metrics.f1:6.1%}")

        if detailed:
            # 3-class metrics
            print("\n  3-Class Metrics:")
            for metrics in [result.service_metrics, result.play_metrics, result.no_play_3class_metrics]:
                print(f"    {metrics.name.capitalize()}:")
                print(f"      Precision: {metrics.precision:6.1%}  Recall: {metrics.recall:6.1%}  F1: {metrics.f1:6.1%}")

        # Boundary errors
        print("\n  Boundary Detection:")
        print(f"    Total transitions: {len(result.boundary_errors)}")
        print(f"    Missed:            {result.missed_boundaries}")
        if result.avg_boundary_error is not None:
            print(f"    Avg error:         {result.avg_boundary_error:.2f}s")

        if detailed and result.boundary_errors:
            print("\n    Boundary Details:")
            for err in result.boundary_errors:
                if err.error is not None:
                    print(f"      {err.transition_type:20} @ {err.ground_truth_time:6.1f}s -> pred {err.predicted_time:6.1f}s (err: {err.error:+.2f}s)")
                else:
                    print(f"      {err.transition_type:20} @ {err.ground_truth_time:6.1f}s -> MISSED")

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        avg_accuracy = sum(r.accuracy for r in results) / len(results)
        avg_active_f1 = sum(r.active_metrics.f1 for r in results) / len(results)
        total_boundaries = sum(len(r.boundary_errors) for r in results)
        total_missed = sum(r.missed_boundaries for r in results)
        all_errors = [abs(e.error) for r in results for e in r.boundary_errors if e.error is not None]
        avg_boundary = sum(all_errors) / len(all_errors) if all_errors else None

        print(f"  Average Accuracy:       {avg_accuracy:6.1%}")
        print(f"  Average Active F1:      {avg_active_f1:6.1%}")
        print(f"  Boundary Detection:     {total_boundaries - total_missed}/{total_boundaries} ({(total_boundaries - total_missed) / total_boundaries * 100:.0f}%)")
        if avg_boundary is not None:
            print(f"  Avg Boundary Error:     {avg_boundary:.2f}s")


def save_results_json(results: list[EvaluationResult], output_path: Path) -> None:
    """Save evaluation results to JSON."""
    data = []
    for r in results:
        data.append({
            "video": r.video_name,
            "duration": r.duration,
            "accuracy": r.accuracy,
            "active": {
                "precision": r.active_metrics.precision,
                "recall": r.active_metrics.recall,
                "f1": r.active_metrics.f1,
            },
            "no_play": {
                "precision": r.no_play_metrics.precision,
                "recall": r.no_play_metrics.recall,
                "f1": r.no_play_metrics.f1,
            },
            "boundaries": {
                "total": len(r.boundary_errors),
                "missed": r.missed_boundaries,
                "avg_error": r.avg_boundary_error,
            },
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ML rally detection against ground truth"
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        type=Path,
        default=Path(__file__).parent.parent / "tests/fixtures/ground_truth.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--video-dir",
        "-v",
        type=Path,
        default=Path(__file__).parent.parent / "tests/fixtures",
        help="Directory containing test videos",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=int,
        help="Override stride parameter (default: from config)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=float,
        help="Limit analysis to first N seconds",
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=3.0,
        help="Boundary matching tolerance in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed per-boundary results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Evaluate only this video (filename from ground truth)",
    )

    args = parser.parse_args()

    # Load ground truth
    print(f"Loading ground truth from: {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)

    # Apply stride override
    if args.stride:
        reset_config()
        set_config(RallyCutConfig(game_state=GameStateConfig(stride=args.stride)))
        print(f"Using stride: {args.stride}")

    # Evaluate each video
    results = []

    for video_name, data in ground_truth.items():
        if args.video and video_name != args.video:
            continue

        video_path = args.video_dir / video_name

        if not video_path.exists():
            print(f"\nSkipping {video_name}: video not found at {video_path}")
            continue

        print(f"\nEvaluating: {video_name}")

        segments = parse_segments(data)
        duration = float(data.get("duration", 120.0))  # type: ignore[arg-type]

        result = evaluate_video(
            video_path=video_path,
            segments=segments,
            duration=duration,
            stride=args.stride,
            limit_seconds=args.limit,
            boundary_tolerance=args.tolerance,
        )
        results.append(result)

    if not results:
        print("\nNo videos evaluated!")
        return 1

    # Print report
    print_report(results, detailed=args.detailed)

    # Save results
    if args.output:
        save_results_json(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
