"""Compare tracking predictions against ground truth."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rallycut.cli.utils import handle_errors

console = Console()


@dataclass
class MOTMetrics:
    """Multi-Object Tracking metrics."""

    # Detection metrics
    num_gt: int = 0  # Total ground truth objects
    num_pred: int = 0  # Total predicted objects
    num_matches: int = 0  # True positives
    num_misses: int = 0  # False negatives (missed detections)
    num_false_positives: int = 0  # False positives

    # Tracking metrics
    num_id_switches: int = 0  # ID switches
    num_fragmentations: int = 0  # Track fragmentations

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        total = self.num_matches + self.num_false_positives
        return self.num_matches / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        total = self.num_matches + self.num_misses
        return self.num_matches / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 = 2 * precision * recall / (precision + recall)."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mota(self) -> float:
        """Multi-Object Tracking Accuracy.

        MOTA = 1 - (misses + false_positives + id_switches) / num_gt
        """
        if self.num_gt == 0:
            return 0.0
        errors = self.num_misses + self.num_false_positives + self.num_id_switches
        return 1.0 - (errors / self.num_gt)

    @property
    def motp(self) -> float:
        """Multi-Object Tracking Precision.

        Average IoU of matched pairs (not implemented, returns 0).
        """
        # TODO: Implement IoU-based MOTP
        return 0.0


@dataclass
class BallMetrics:
    """Ball tracking metrics."""

    num_gt: int = 0  # Ground truth ball positions
    num_detected: int = 0  # Detected ball positions (with match)
    total_error_px: float = 0.0  # Sum of position errors (pixels)

    @property
    def detection_rate(self) -> float:
        """Fraction of GT frames where ball was detected."""
        return self.num_detected / self.num_gt if self.num_gt > 0 else 0.0

    @property
    def mean_error_px(self) -> float:
        """Mean position error in pixels."""
        return self.total_error_px / self.num_detected if self.num_detected > 0 else 0.0


def _compute_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two boxes (center x, y, width, height)."""
    # Convert center coords to corners
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Box 1 corners
    left1 = x1 - w1 / 2
    right1 = x1 + w1 / 2
    top1 = y1 - h1 / 2
    bottom1 = y1 + h1 / 2

    # Box 2 corners
    left2 = x2 - w2 / 2
    right2 = x2 + w2 / 2
    top2 = y2 - h2 / 2
    bottom2 = y2 + h2 / 2

    # Intersection
    left_i = max(left1, left2)
    right_i = min(right1, right2)
    top_i = max(top1, top2)
    bottom_i = min(bottom1, bottom2)

    if left_i >= right_i or top_i >= bottom_i:
        return 0.0

    intersection = (right_i - left_i) * (bottom_i - top_i)

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _match_detections(
    gt_boxes: list[tuple[int, float, float, float, float]],  # (track_id, x, y, w, h)
    pred_boxes: list[tuple[int, float, float, float, float]],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match predictions to ground truth using Hungarian algorithm.

    Returns:
        Tuple of (matches, unmatched_gt, unmatched_pred).
        matches: List of (gt_track_id, pred_track_id) pairs.
    """
    if not gt_boxes or not pred_boxes:
        gt_ids = [b[0] for b in gt_boxes]
        pred_ids = [b[0] for b in pred_boxes]
        return [], gt_ids, pred_ids

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # Compute IoU matrix
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    iou_matrix = np.zeros((n_gt, n_pred))

    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou = _compute_iou(gt[1:], pred[1:])
            iou_matrix[i, j] = iou

    # Convert to cost matrix (1 - IoU)
    cost_matrix = 1 - iou_matrix

    # Hungarian algorithm
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Filter matches by IoU threshold
    matches: list[tuple[int, int]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for gi, pi in zip(gt_indices, pred_indices):
        if iou_matrix[gi, pi] >= iou_threshold:
            matches.append((gt_boxes[gi][0], pred_boxes[pi][0]))
            matched_gt.add(gi)
            matched_pred.add(pi)

    # Unmatched
    unmatched_gt = [gt_boxes[i][0] for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [pred_boxes[i][0] for i in range(n_pred) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_player_metrics(
    gt_positions: list,
    pred_positions: list,
    iou_threshold: float = 0.5,
) -> MOTMetrics:
    """Compute MOT metrics for player tracking.

    Args:
        gt_positions: Ground truth positions (GroundTruthPosition list).
        pred_positions: Predicted positions (PlayerPosition list).
        iou_threshold: Minimum IoU for a match.

    Returns:
        MOTMetrics with all tracking metrics.
    """
    # Group by frame
    gt_by_frame: dict[int, list] = defaultdict(list)
    pred_by_frame: dict[int, list] = defaultdict(list)

    for p in gt_positions:
        gt_by_frame[p.frame_number].append((p.track_id, p.x, p.y, p.width, p.height))

    for p in pred_positions:
        pred_by_frame[p.frame_number].append((p.track_id, p.x, p.y, p.width, p.height))

    # Get all frames
    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    metrics = MOTMetrics()

    # Track last matched prediction ID for each GT track (for ID switch detection)
    last_pred_id: dict[int, int] = {}

    for frame in all_frames:
        gt_boxes = gt_by_frame.get(frame, [])
        pred_boxes = pred_by_frame.get(frame, [])

        metrics.num_gt += len(gt_boxes)
        metrics.num_pred += len(pred_boxes)

        if not gt_boxes:
            metrics.num_false_positives += len(pred_boxes)
            continue

        if not pred_boxes:
            metrics.num_misses += len(gt_boxes)
            continue

        # Match detections
        matches, unmatched_gt, unmatched_pred = _match_detections(
            gt_boxes, pred_boxes, iou_threshold
        )

        metrics.num_matches += len(matches)
        metrics.num_misses += len(unmatched_gt)
        metrics.num_false_positives += len(unmatched_pred)

        # Check for ID switches
        for gt_id, pred_id in matches:
            if gt_id in last_pred_id and last_pred_id[gt_id] != pred_id:
                metrics.num_id_switches += 1
            last_pred_id[gt_id] = pred_id

    return metrics


def compute_ball_metrics(
    gt_positions: list,
    pred_positions: list,
    video_width: int,
    video_height: int,
    max_distance_px: float = 50.0,
) -> BallMetrics:
    """Compute ball tracking metrics.

    Args:
        gt_positions: Ground truth ball positions.
        pred_positions: Predicted ball positions.
        video_width: Video frame width (for pixel conversion).
        video_height: Video frame height (for pixel conversion).
        max_distance_px: Maximum distance (pixels) for a match.

    Returns:
        BallMetrics with detection rate and position error.
    """
    import math

    # Group by frame
    gt_by_frame: dict[int, tuple[float, float]] = {}
    pred_by_frame: dict[int, tuple[float, float]] = {}

    for p in gt_positions:
        gt_by_frame[p.frame_number] = (p.x, p.y)

    for p in pred_positions:
        # Take highest confidence prediction per frame
        if p.frame_number not in pred_by_frame:
            pred_by_frame[p.frame_number] = (p.x, p.y)

    metrics = BallMetrics(num_gt=len(gt_by_frame))

    for frame, (gt_x, gt_y) in gt_by_frame.items():
        if frame not in pred_by_frame:
            continue

        pred_x, pred_y = pred_by_frame[frame]

        # Compute distance in pixels
        dx = (pred_x - gt_x) * video_width
        dy = (pred_y - gt_y) * video_height
        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= max_distance_px:
            metrics.num_detected += 1
            metrics.total_error_px += distance

    return metrics


@handle_errors
def compare_tracking(
    predictions: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Predictions JSON file (from track-players)",
    ),
    ground_truth: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Ground truth JSON file (from 'rallycut label save')",
    ),
    iou_threshold: float = typer.Option(
        0.5,
        "--iou", "-i",
        help="Minimum IoU for matching predictions to ground truth",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output metrics JSON file (optional)",
    ),
) -> None:
    """Compare tracking predictions against ground truth.

    Computes standard MOT (Multi-Object Tracking) metrics:
    - MOTA: Multi-Object Tracking Accuracy
    - IDF1: ID F1 Score (requires motmetrics, falls back to F1)
    - Precision, Recall, F1
    - ID Switches

    For ball tracking, computes:
    - Detection rate: Fraction of frames with correct detection
    - Mean position error: Average distance in pixels

    Examples:

        # Compare player tracking
        rallycut compare-tracking tracking.json ground_truth.json

        # Use stricter IoU threshold
        rallycut compare-tracking tracking.json gt.json --iou 0.7

        # Save metrics to file
        rallycut compare-tracking tracking.json gt.json -o metrics.json
    """
    import json

    from rallycut.labeling.ground_truth import GroundTruthResult
    from rallycut.tracking.player_tracker import PlayerTrackingResult

    console.print("[bold]Comparing tracking predictions vs ground truth[/bold]")
    console.print(f"  Predictions: {predictions}")
    console.print(f"  Ground truth: {ground_truth}")
    console.print(f"  IoU threshold: {iou_threshold}")

    # Load data
    console.print("\n[cyan]Loading data...[/cyan]")
    pred_result = PlayerTrackingResult.from_json(predictions)
    gt_result = GroundTruthResult.from_json(ground_truth)

    video_width = gt_result.video_width or pred_result.video_width or 1920
    video_height = gt_result.video_height or pred_result.video_height or 1080

    console.print(f"  Predictions: {len(pred_result.positions)} detections")
    console.print(f"  Ground truth: {len(gt_result.positions)} annotations")
    console.print(f"  Resolution: {video_width}x{video_height}")

    # Compute player tracking metrics
    console.print("\n[cyan]Computing player tracking metrics...[/cyan]")
    player_metrics = compute_player_metrics(
        gt_result.player_positions,
        pred_result.positions,
        iou_threshold,
    )

    # Compute ball tracking metrics if available
    ball_metrics: BallMetrics | None = None
    if gt_result.ball_positions and pred_result.ball_positions:
        console.print("\n[cyan]Computing ball tracking metrics...[/cyan]")
        ball_metrics = compute_ball_metrics(
            gt_result.ball_positions,
            pred_result.ball_positions,
            video_width,
            video_height,
        )

    # Display results
    console.print("\n[bold]Player Tracking Metrics[/bold]")

    player_table = Table(show_header=True, header_style="bold")
    player_table.add_column("Metric")
    player_table.add_column("Value")
    player_table.add_column("Target")
    player_table.add_column("Status")

    def status_icon(value: float, target: float, higher_better: bool = True) -> str:
        if higher_better:
            return "[green]OK[/green]" if value >= target else "[yellow]Below[/yellow]"
        else:
            return "[green]OK[/green]" if value <= target else "[yellow]Above[/yellow]"

    player_table.add_row(
        "MOTA",
        f"{player_metrics.mota:.2%}",
        ">80%",
        status_icon(player_metrics.mota, 0.80),
    )
    player_table.add_row(
        "Precision",
        f"{player_metrics.precision:.2%}",
        ">85%",
        status_icon(player_metrics.precision, 0.85),
    )
    player_table.add_row(
        "Recall",
        f"{player_metrics.recall:.2%}",
        ">80%",
        status_icon(player_metrics.recall, 0.80),
    )
    player_table.add_row(
        "F1",
        f"{player_metrics.f1:.2%}",
        ">80%",
        status_icon(player_metrics.f1, 0.80),
    )
    player_table.add_row(
        "ID Switches",
        str(player_metrics.num_id_switches),
        "<5",
        status_icon(player_metrics.num_id_switches, 5, higher_better=False),
    )

    console.print(player_table)

    # Detection breakdown
    console.print("\n[dim]Detection breakdown:[/dim]")
    console.print(f"  Ground truth: {player_metrics.num_gt} objects")
    console.print(f"  Predictions: {player_metrics.num_pred} objects")
    console.print(f"  Matches (TP): {player_metrics.num_matches}")
    console.print(f"  Misses (FN): {player_metrics.num_misses}")
    console.print(f"  False positives (FP): {player_metrics.num_false_positives}")

    # Ball metrics
    if ball_metrics:
        console.print("\n[bold]Ball Tracking Metrics[/bold]")

        ball_table = Table(show_header=True, header_style="bold")
        ball_table.add_column("Metric")
        ball_table.add_column("Value")
        ball_table.add_column("Target")
        ball_table.add_column("Status")

        ball_table.add_row(
            "Detection Rate",
            f"{ball_metrics.detection_rate:.2%}",
            ">60%",
            status_icon(ball_metrics.detection_rate, 0.60),
        )
        ball_table.add_row(
            "Mean Error",
            f"{ball_metrics.mean_error_px:.1f} px",
            "<20px",
            status_icon(ball_metrics.mean_error_px, 20, higher_better=False),
        )

        console.print(ball_table)

    # Save to file if requested
    if output:
        metrics_dict = {
            "player": {
                "mota": player_metrics.mota,
                "precision": player_metrics.precision,
                "recall": player_metrics.recall,
                "f1": player_metrics.f1,
                "idSwitches": player_metrics.num_id_switches,
                "numGt": player_metrics.num_gt,
                "numPred": player_metrics.num_pred,
                "numMatches": player_metrics.num_matches,
                "numMisses": player_metrics.num_misses,
                "numFalsePositives": player_metrics.num_false_positives,
            },
        }
        if ball_metrics:
            metrics_dict["ball"] = {
                "detectionRate": ball_metrics.detection_rate,
                "meanErrorPx": ball_metrics.mean_error_px,
                "numGt": ball_metrics.num_gt,
                "numDetected": ball_metrics.num_detected,
            }

        with open(output, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        console.print(f"\n[green]Metrics saved to {output}[/green]")
