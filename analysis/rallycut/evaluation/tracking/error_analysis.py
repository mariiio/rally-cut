"""Error categorization and analysis for tracking evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rallycut.cli.commands.compare_tracking import _match_detections
from rallycut.labeling.ground_truth import GroundTruthPosition, GroundTruthResult
from rallycut.tracking.player_tracker import PlayerPosition, PlayerTrackingResult


class ErrorType(Enum):
    """Types of tracking errors."""

    MISSED_DETECTION = "missed"  # GT exists, no prediction matched
    FALSE_POSITIVE = "fp"  # Prediction exists, no GT matched
    ID_SWITCH = "id_switch"  # Matched prediction ID changed from previous frame


@dataclass
class ErrorEvent:
    """A single error event in tracking."""

    frame_number: int
    error_type: ErrorType
    player_label: str | None  # e.g., "player_1" for GT-related errors
    gt_position: GroundTruthPosition | None  # For misses
    pred_position: PlayerPosition | None  # For false positives
    previous_pred_id: int | None = None  # For ID switches
    current_pred_id: int | None = None  # For ID switches

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "frameNumber": self.frame_number,
            "errorType": self.error_type.value,
            "playerLabel": self.player_label,
        }

        if self.gt_position:
            result["gtPosition"] = {
                "x": self.gt_position.x,
                "y": self.gt_position.y,
                "width": self.gt_position.width,
                "height": self.gt_position.height,
                "trackId": self.gt_position.track_id,
            }

        if self.pred_position:
            result["predPosition"] = {
                "x": self.pred_position.x,
                "y": self.pred_position.y,
                "width": self.pred_position.width,
                "height": self.pred_position.height,
                "trackId": self.pred_position.track_id,
            }

        if self.error_type == ErrorType.ID_SWITCH:
            result["previousPredId"] = self.previous_pred_id
            result["currentPredId"] = self.current_pred_id

        return result


@dataclass
class ErrorSummary:
    """Summary of errors by type and player."""

    total_errors: int
    by_type: dict[ErrorType, int]
    by_player: dict[str, int]  # player_label -> count
    error_frame_ranges: list[tuple[int, int]]  # Contiguous error ranges

    @property
    def consecutive_error_frames(self) -> int:
        """Maximum consecutive frames with errors."""
        if not self.error_frame_ranges:
            return 0
        return max(end - start + 1 for start, end in self.error_frame_ranges)


def analyze_errors(
    ground_truth: GroundTruthResult,
    predictions: PlayerTrackingResult,
    iou_threshold: float = 0.5,
) -> list[ErrorEvent]:
    """Analyze and categorize all tracking errors.

    Args:
        ground_truth: Ground truth annotations.
        predictions: Predicted player positions.
        iou_threshold: Minimum IoU for matching.

    Returns:
        List of ErrorEvent describing each error.
    """
    gt_positions = ground_truth.player_positions
    pred_positions = predictions.positions

    # Group by frame
    gt_by_frame: dict[int, list[GroundTruthPosition]] = defaultdict(list)
    pred_by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)

    for gt_pos in gt_positions:
        gt_by_frame[gt_pos.frame_number].append(gt_pos)

    for pred_pos in pred_positions:
        pred_by_frame[pred_pos.frame_number].append(pred_pos)

    # Get all frames
    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    # Track last matched prediction ID for each GT track
    last_pred_id: dict[int, int] = {}

    errors: list[ErrorEvent] = []

    for frame in all_frames:
        gt_list = gt_by_frame.get(frame, [])
        pred_list = pred_by_frame.get(frame, [])

        # Convert to box format for matching
        gt_boxes = [(p.track_id, p.x, p.y, p.width, p.height) for p in gt_list]
        pred_boxes = [(p.track_id, p.x, p.y, p.width, p.height) for p in pred_list]

        # Create lookup maps
        gt_by_id = {p.track_id: p for p in gt_list}
        pred_by_id = {p.track_id: p for p in pred_list}

        if not gt_boxes:
            # All predictions are false positives
            for pred in pred_list:
                errors.append(
                    ErrorEvent(
                        frame_number=frame,
                        error_type=ErrorType.FALSE_POSITIVE,
                        player_label=None,
                        gt_position=None,
                        pred_position=pred,
                    )
                )
            continue

        if not pred_boxes:
            # All ground truth are misses
            for gt in gt_list:
                errors.append(
                    ErrorEvent(
                        frame_number=frame,
                        error_type=ErrorType.MISSED_DETECTION,
                        player_label=f"player_{gt.track_id}",
                        gt_position=gt,
                        pred_position=None,
                    )
                )
            continue

        # Match detections
        matches, unmatched_gt, unmatched_pred = _match_detections(
            gt_boxes, pred_boxes, iou_threshold
        )

        # Record misses
        for gt_id in unmatched_gt:
            gt_miss = gt_by_id.get(gt_id)
            if gt_miss is not None:
                errors.append(
                    ErrorEvent(
                        frame_number=frame,
                        error_type=ErrorType.MISSED_DETECTION,
                        player_label=f"player_{gt_id}",
                        gt_position=gt_miss,
                        pred_position=None,
                    )
                )

        # Record false positives
        for pred_id in unmatched_pred:
            pred_fp = pred_by_id.get(pred_id)
            if pred_fp is not None:
                errors.append(
                    ErrorEvent(
                        frame_number=frame,
                        error_type=ErrorType.FALSE_POSITIVE,
                        player_label=None,
                        gt_position=None,
                        pred_position=pred_fp,
                    )
                )

        # Check for ID switches in matches
        for gt_id, pred_id in matches:
            if gt_id in last_pred_id and last_pred_id[gt_id] != pred_id:
                gt_switch = gt_by_id.get(gt_id)
                pred_switch = pred_by_id.get(pred_id)
                errors.append(
                    ErrorEvent(
                        frame_number=frame,
                        error_type=ErrorType.ID_SWITCH,
                        player_label=f"player_{gt_id}",
                        gt_position=gt_switch,
                        pred_position=pred_switch,
                        previous_pred_id=last_pred_id[gt_id],
                        current_pred_id=pred_id,
                    )
                )
            last_pred_id[gt_id] = pred_id

    return errors


def summarize_errors(errors: list[ErrorEvent]) -> ErrorSummary:
    """Summarize errors into aggregate statistics.

    Args:
        errors: List of error events.

    Returns:
        ErrorSummary with counts and patterns.
    """
    by_type: dict[ErrorType, int] = defaultdict(int)
    by_player: dict[str, int] = defaultdict(int)

    for e in errors:
        by_type[e.error_type] += 1
        if e.player_label:
            by_player[e.player_label] += 1

    # Find contiguous error frame ranges
    if not errors:
        return ErrorSummary(
            total_errors=0,
            by_type=dict(by_type),
            by_player=dict(by_player),
            error_frame_ranges=[],
        )

    error_frames = sorted(set(e.frame_number for e in errors))
    ranges: list[tuple[int, int]] = []
    start = error_frames[0]
    end = error_frames[0]

    for frame in error_frames[1:]:
        if frame == end + 1:
            end = frame
        else:
            ranges.append((start, end))
            start = frame
            end = frame
    ranges.append((start, end))

    return ErrorSummary(
        total_errors=len(errors),
        by_type=dict(by_type),
        by_player=dict(by_player),
        error_frame_ranges=ranges,
    )
