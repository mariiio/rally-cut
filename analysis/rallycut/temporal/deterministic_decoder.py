"""Deterministic decoder for rally detection.

Converts per-window rally probabilities to segments using:
1. Probability smoothing (moving average)
2. Hysteresis thresholding (t_on/t_off with patience)
3. Binary cleanup (fill gaps, remove short segments)
4. Anti-overmerge (max duration backstop)

No learned parameters - all hyperparameters tuned via grid search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default grid search parameter ranges
DEFAULT_PARAM_GRID: dict[str, list[float | int]] = {
    "smooth_window": [1, 3, 5],
    "t_on": [0.35, 0.4, 0.45, 0.5],
    "t_off": [0.2, 0.25, 0.3, 0.35],
    "patience": [1, 2, 3],
    "min_segment_windows": [1, 2, 3],
    "max_gap_windows": [1, 2, 3],
}


@dataclass
class DecoderConfig:
    """Configuration for deterministic decoder."""

    # Smoothing
    smooth_window: int = 3  # Moving average window (odd number)

    # Hysteresis thresholding
    t_on: float = 0.4  # Threshold to enter rally (lowered from 0.5 for short rally detection)
    t_off: float = 0.25  # Threshold to exit rally (lowered from 0.3 to match t_on change)
    patience: int = 2  # Windows below t_off before exiting

    # Binary cleanup
    min_segment_windows: int = 1  # Remove segments shorter than this (lowered from 2 for short rally detection)
    max_gap_windows: int = 3  # Fill gaps shorter than this

    # Anti-overmerge
    max_duration_seconds: float = 60.0  # Force split if exceeded

    # Video params (set at runtime)
    fps: float = 30.0
    stride: int = 48
    window_size: int = 16  # Frames per classification window


@dataclass
class DecoderResult:
    """Result from deterministic decoder."""

    segments: list[tuple[float, float]]  # (start_time, end_time)
    smoothed_probs: np.ndarray
    binary_preds: np.ndarray


def smooth_probabilities(probs: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply moving average smoothing to probabilities.

    Args:
        probs: (N,) array of probabilities.
        window: Smoothing window size (odd). Even values will be incremented.

    Returns:
        Smoothed probabilities.
    """
    if window <= 1:
        return probs.copy()

    # Ensure odd window (log if adjusted)
    if window % 2 == 0:
        logger.debug("Adjusting smooth_window from %d to %d (must be odd)", window, window + 1)
        window += 1

    # Pad and convolve
    pad = window // 2
    padded = np.pad(probs, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="valid")

    return smoothed


def hysteresis_threshold(
    probs: np.ndarray,
    t_on: float = 0.4,
    t_off: float = 0.25,
    patience: int = 2,
) -> np.ndarray:
    """Apply hysteresis thresholding to probabilities.

    Once in rally state, stay until seeing `patience` consecutive
    windows below t_off.

    Args:
        probs: (N,) array of probabilities.
        t_on: Threshold to enter rally.
        t_off: Threshold to exit rally.
        patience: Windows below t_off before exiting.

    Returns:
        Binary predictions (0/1).
    """
    n = len(probs)
    preds = np.zeros(n, dtype=int)

    in_rally = False
    below_count = 0

    for i in range(n):
        p = probs[i]

        if not in_rally:
            if p >= t_on:
                in_rally = True
                below_count = 0
                preds[i] = 1
            else:
                preds[i] = 0
        else:  # in_rally
            if p >= t_off:
                below_count = 0
                preds[i] = 1
            else:
                below_count += 1
                if below_count >= patience:
                    in_rally = False
                    # Mark previous patience-1 windows as not rally
                    for j in range(1, patience):
                        if i - j >= 0:
                            preds[i - j] = 0
                    preds[i] = 0
                else:
                    preds[i] = 1  # Still in rally during patience

    return preds


def fill_gaps(preds: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Fill short gaps between rally segments.

    Args:
        preds: (N,) binary predictions.
        max_gap: Maximum gap size to fill.

    Returns:
        Predictions with gaps filled.
    """
    if max_gap <= 0:
        return preds.copy()

    result = preds.copy()
    n = len(result)

    i = 0
    while i < n:
        if result[i] == 1:
            # Find end of this segment
            j = i
            while j < n and result[j] == 1:
                j += 1
            # j is now first 0 after segment (or n)

            # Look for gap
            if j < n:
                gap_start = j
                while j < n and result[j] == 0:
                    j += 1
                gap_end = j
                # j is now first 1 after gap (or n)

                gap_size = gap_end - gap_start
                if gap_size <= max_gap and j < n:
                    # Fill the gap
                    result[gap_start:gap_end] = 1
                    # Continue from end of filled gap
                    i = gap_start
                else:
                    i = gap_end
            else:
                i = j
        else:
            i += 1

    return result


def remove_short_segments(preds: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Remove segments shorter than min_length.

    Args:
        preds: (N,) binary predictions.
        min_length: Minimum segment length in windows.

    Returns:
        Predictions with short segments removed.
    """
    if min_length <= 1:
        return preds.copy()

    result = preds.copy()
    n = len(result)

    i = 0
    while i < n:
        if result[i] == 1:
            # Find segment
            start = i
            while i < n and result[i] == 1:
                i += 1
            end = i

            # Check length
            if end - start < min_length:
                result[start:end] = 0
        else:
            i += 1

    return result


def apply_anti_overmerge(
    preds: np.ndarray,
    probs: np.ndarray,
    max_duration_windows: int,
    min_prob_for_split: float = 0.4,
) -> np.ndarray:
    """Split segments that exceed max duration.

    Finds the lowest probability point within over-long segments
    and splits there.

    Args:
        preds: (N,) binary predictions.
        probs: (N,) probabilities (for finding split points).
        max_duration_windows: Maximum segment length.
        min_prob_for_split: Only split if there's a point below this.

    Returns:
        Predictions with over-long segments split.
    """
    if max_duration_windows <= 0:
        return preds.copy()

    result = preds.copy()
    n = len(result)

    i = 0
    while i < n:
        if result[i] == 1:
            # Find segment
            start = i
            while i < n and result[i] == 1:
                i += 1
            end = i

            # Check if too long
            if end - start > max_duration_windows:
                # Find split point (lowest prob in segment)
                segment_probs = probs[start:end]
                min_idx = np.argmin(segment_probs)
                min_prob = segment_probs[min_idx]

                if min_prob < min_prob_for_split:
                    # Split at this point
                    split_idx = start + min_idx
                    result[split_idx] = 0
                    # Recursively check the two halves
                    i = start  # Re-check from start
                else:
                    # Can't find good split point, skip to next segment
                    i = end
        else:
            i += 1

    return result


def preds_to_segments(
    preds: np.ndarray,
    fps: float,
    stride: int,
    window_size: int = 16,
) -> list[tuple[float, float]]:
    """Convert binary predictions to time segments.

    Args:
        preds: (N,) binary predictions.
        fps: Video FPS.
        stride: Frame stride.
        window_size: Frames per window.

    Returns:
        List of (start_time, end_time) tuples.
    """
    segments = []
    window_duration = stride / fps
    n = len(preds)

    i = 0
    while i < n:
        if preds[i] == 1:
            start = i
            while i < n and preds[i] == 1:
                i += 1
            end = i

            start_time = start * window_duration
            end_time = (end - 1) * window_duration + window_size / fps
            segments.append((start_time, end_time))
        else:
            i += 1

    return segments


def decode(
    probs: np.ndarray,
    config: DecoderConfig,
) -> DecoderResult:
    """Full decoding pipeline.

    Args:
        probs: (N,) per-window rally probabilities.
        config: Decoder configuration.

    Returns:
        DecoderResult with segments and intermediate outputs.
    """
    # 1. Smooth probabilities
    smoothed = smooth_probabilities(probs, config.smooth_window)

    # 2. Hysteresis threshold
    preds = hysteresis_threshold(
        smoothed,
        t_on=config.t_on,
        t_off=config.t_off,
        patience=config.patience,
    )

    # 3. Fill gaps
    preds = fill_gaps(preds, config.max_gap_windows)

    # 4. Remove short segments
    preds = remove_short_segments(preds, config.min_segment_windows)

    # 5. Anti-overmerge
    window_duration = config.stride / config.fps
    max_duration_windows = int(config.max_duration_seconds / window_duration)
    preds = apply_anti_overmerge(preds, smoothed, max_duration_windows)

    # 6. Convert to segments
    segments = preds_to_segments(preds, config.fps, config.stride, config.window_size)

    return DecoderResult(
        segments=segments,
        smoothed_probs=smoothed,
        binary_preds=preds,
    )


# =============================================================================
# Grid Search
# =============================================================================


def compute_segment_metrics(
    ground_truth: list[tuple[float, float]],
    predictions: list[tuple[float, float]],
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute segment-level metrics with IoU matching.

    Args:
        ground_truth: List of (start, end) ground truth segments.
        predictions: List of (start, end) predicted segments.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dict with precision, recall, f1, tp, fp, fn.
    """
    if not ground_truth and not predictions:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not ground_truth:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "tp": 0, "fp": len(predictions), "fn": 0}
    if not predictions:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ground_truth)}

    # Greedy matching by IoU
    matched_gt = set()
    matched_pred = set()

    # Sort predictions by start time for consistent matching
    pred_order = sorted(range(len(predictions)), key=lambda i: predictions[i][0])

    for i in pred_order:
        ps, pe = predictions[i]
        best_iou = 0.0
        best_gt = -1

        for j, (gs, ge) in enumerate(ground_truth):
            if j in matched_gt:
                continue
            intersection = max(0, min(pe, ge) - max(ps, gs))
            union = (pe - ps) + (ge - gs) - intersection
            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            matched_pred.add(i)

    tp = len(matched_pred)
    fp = len(predictions) - tp
    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_overmerge_rate(
    segments: list[tuple[float, float]],
    max_duration: float = 60.0,
) -> dict[str, float | int]:
    """Compute overmerge statistics.

    Args:
        segments: List of (start, end) segments.
        max_duration: Threshold for overmerge.

    Returns:
        Dict with overmerge_count, overmerge_rate, max_duration.
    """
    if not segments:
        return {"overmerge_count": 0, "overmerge_rate": 0.0, "max_segment_duration": 0.0}

    durations = [end - start for start, end in segments]
    overmerge_count = sum(1 for d in durations if d > max_duration)
    max_dur = max(durations)

    return {
        "overmerge_count": overmerge_count,
        "overmerge_rate": overmerge_count / len(segments) if segments else 0,
        "max_segment_duration": max_dur,
    }


def compute_boundary_errors(
    ground_truth: list[tuple[float, float]],
    predictions: list[tuple[float, float]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute boundary errors for matched segments.

    Args:
        ground_truth: Ground truth segments.
        predictions: Predicted segments.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dict with mean/median start/end errors.
    """
    if not ground_truth or not predictions:
        return {
            "mean_start_error": 0.0,
            "mean_end_error": 0.0,
            "median_start_error": 0.0,
            "median_end_error": 0.0,
        }

    # Match segments
    start_errors = []
    end_errors = []
    matched_gt = set()

    for ps, pe in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        best_gt = None

        for j, (gs, ge) in enumerate(ground_truth):
            if j in matched_gt:
                continue
            intersection = max(0, min(pe, ge) - max(ps, gs))
            union = (pe - ps) + (ge - gs) - intersection
            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
                best_gt = (gs, ge)

        if best_iou >= iou_threshold and best_gt is not None:
            matched_gt.add(best_gt_idx)
            start_errors.append(abs(ps - best_gt[0]))
            end_errors.append(abs(pe - best_gt[1]))

    if not start_errors:
        return {
            "mean_start_error": 0.0,
            "mean_end_error": 0.0,
            "median_start_error": 0.0,
            "median_end_error": 0.0,
        }

    return {
        "mean_start_error": float(np.mean(start_errors)),
        "mean_end_error": float(np.mean(end_errors)),
        "median_start_error": float(np.median(start_errors)),
        "median_end_error": float(np.median(end_errors)),
    }


@dataclass
class GridSearchResult:
    """Result from grid search."""

    best_config: DecoderConfig
    best_f1: float
    best_precision: float
    best_recall: float
    best_overmerge_rate: float
    all_results: list[dict] = field(default_factory=list)


def grid_search(
    video_probs: list[np.ndarray],
    video_ground_truths: list[list[tuple[float, float]]],
    video_fps: list[float],
    stride: int = 48,
    iou_threshold: float = 0.5,
    param_grid: dict[str, list[float | int]] | None = None,
) -> GridSearchResult:
    """Grid search for optimal decoder parameters.

    Args:
        video_probs: List of per-window probability arrays (one per video).
        video_ground_truths: List of ground truth segment lists.
        video_fps: List of FPS values.
        stride: Frame stride.
        iou_threshold: IoU threshold for matching.
        param_grid: Parameter grid to search. If None, uses DEFAULT_PARAM_GRID.

    Returns:
        GridSearchResult with best config and metrics.
    """
    # Use provided grid or default
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    best_result: dict[str, Any] | None = None
    best_score = -1.0
    all_results: list[dict[str, Any]] = []

    # Generate all combinations
    keys = list(param_grid.keys())
    param_values: list[list[float | int]] = [param_grid[k] for k in keys]
    for values in product(*param_values):
        params: dict[str, float | int] = dict(zip(keys, values))

        # Skip invalid: t_off should be < t_on
        if params["t_off"] >= params["t_on"]:
            continue

        # Evaluate on all videos
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_overmerge = 0
        total_segments = 0

        for probs, gt, fps in zip(video_probs, video_ground_truths, video_fps):
            config = DecoderConfig(
                smooth_window=int(params["smooth_window"]),
                t_on=float(params["t_on"]),
                t_off=float(params["t_off"]),
                patience=int(params["patience"]),
                min_segment_windows=int(params["min_segment_windows"]),
                max_gap_windows=int(params["max_gap_windows"]),
                max_duration_seconds=60.0,
                fps=fps,
                stride=stride,
            )

            result = decode(probs, config)
            metrics = compute_segment_metrics(gt, result.segments, iou_threshold)
            overmerge = compute_overmerge_rate(result.segments, 60.0)

            total_tp += int(metrics["tp"])
            total_fp += int(metrics["fp"])
            total_fn += int(metrics["fn"])
            total_overmerge += int(overmerge["overmerge_count"])
            total_segments += len(result.segments)

        # Aggregate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        overmerge_rate = total_overmerge / total_segments if total_segments > 0 else 0

        # Score: F1 with penalty for overmerge
        score = f1 * (1 - overmerge_rate)

        result_entry = {
            "params": params,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "overmerge_rate": overmerge_rate,
            "score": score,
        }
        all_results.append(result_entry)

        if score > best_score:
            best_score = score
            best_result = result_entry

    if best_result is None:
        # Return default config
        return GridSearchResult(
            best_config=DecoderConfig(),
            best_f1=0.0,
            best_precision=0.0,
            best_recall=0.0,
            best_overmerge_rate=0.0,
            all_results=all_results,
        )

    best_config = DecoderConfig(
        smooth_window=int(best_result["params"]["smooth_window"]),
        t_on=float(best_result["params"]["t_on"]),
        t_off=float(best_result["params"]["t_off"]),
        patience=int(best_result["params"]["patience"]),
        min_segment_windows=int(best_result["params"]["min_segment_windows"]),
        max_gap_windows=int(best_result["params"]["max_gap_windows"]),
        max_duration_seconds=60.0,
        stride=stride,
    )

    return GridSearchResult(
        best_config=best_config,
        best_f1=float(best_result["f1"]),
        best_precision=float(best_result["precision"]),
        best_recall=float(best_result["recall"]),
        best_overmerge_rate=float(best_result["overmerge_rate"]),
        all_results=sorted(all_results, key=lambda x: x["score"], reverse=True),
    )
