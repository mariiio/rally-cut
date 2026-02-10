"""Grid search over PlayerFilterConfig parameters.

The grid search efficiently evaluates filter configurations by:
1. Caching raw YOLO+ByteTrack positions (slow, one-time)
2. Re-running only the filter pipeline with different configs (fast, per-config)
3. Computing MOT metrics against ground truth
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

from rallycut.cli.commands.compare_tracking import MOTMetrics
from rallycut.evaluation.tracking.metrics import evaluate_rally
from rallycut.evaluation.tracking.param_grid import (
    describe_config_diff,
    generate_filter_configs,
    get_default_config,
)
from rallycut.evaluation.tracking.raw_cache import CachedRallyData
from rallycut.labeling.ground_truth import GroundTruthResult
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_filter import (
    PlayerFilter,
    PlayerFilterConfig,
    stabilize_track_ids,
)
from rallycut.tracking.player_tracker import PlayerPosition, PlayerTrackingResult

logger = logging.getLogger(__name__)


@dataclass
class FilterConfigResult:
    """Result for a single filter config across all rallies."""

    config: PlayerFilterConfig
    aggregate_metrics: MOTMetrics
    per_rally_metrics: list[tuple[str, MOTMetrics]]  # (rally_id, metrics)
    rejected: bool = False  # True if config violated constraints
    rejection_reason: str = ""


@dataclass
class FilterGridSearchResult:
    """Complete grid search result."""

    best_config: PlayerFilterConfig
    best_f1: float
    best_mota: float
    best_id_switches: int
    all_results: list[FilterConfigResult]
    rejected_count: int
    total_configs: int

    # Best config comparison to default
    default_f1: float = 0.0
    default_mota: float = 0.0
    improvement_f1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "best_config": {
                f.name: getattr(self.best_config, f.name)
                for f in fields(PlayerFilterConfig)
            },
            "best_f1": self.best_f1,
            "best_mota": self.best_mota,
            "best_id_switches": self.best_id_switches,
            "default_f1": self.default_f1,
            "default_mota": self.default_mota,
            "improvement_f1": self.improvement_f1,
            "rejected_count": self.rejected_count,
            "total_configs": self.total_configs,
            "all_results": [
                {
                    "config_diff": describe_config_diff(r.config),
                    "f1": r.aggregate_metrics.f1,
                    "mota": r.aggregate_metrics.mota,
                    "precision": r.aggregate_metrics.precision,
                    "recall": r.aggregate_metrics.recall,
                    "id_switches": r.aggregate_metrics.num_id_switches,
                    "rejected": r.rejected,
                    "rejection_reason": r.rejection_reason,
                    "per_rally": [
                        {"rally_id": rid[:8], "f1": m.f1, "mota": m.mota}
                        for rid, m in r.per_rally_metrics
                    ],
                }
                for r in self.all_results
            ],
        }


def apply_filter_config(
    raw_positions: list[PlayerPosition],
    ball_positions: list[BallPosition],
    frame_count: int,
    config: PlayerFilterConfig,
) -> list[PlayerPosition]:
    """Apply a filter config to raw positions.

    This is the fast path - only runs the filter pipeline, not YOLO.

    Args:
        raw_positions: Unfiltered YOLO+ByteTrack output.
        ball_positions: Ball positions for filtering.
        frame_count: Total frames in the rally.
        config: Filter configuration to apply.

    Returns:
        Filtered player positions.
    """
    # Copy positions to avoid modifying originals
    positions = [
        PlayerPosition(
            frame_number=p.frame_number,
            track_id=p.track_id,
            x=p.x,
            y=p.y,
            width=p.width,
            height=p.height,
            confidence=p.confidence,
        )
        for p in raw_positions
    ]

    # Apply track ID stabilization (merging)
    positions, _ = stabilize_track_ids(positions, config)

    # Create filter and analyze tracks
    player_filter = PlayerFilter(
        ball_positions=ball_positions,
        total_frames=frame_count,
        config=config,
    )
    player_filter.analyze_tracks(positions)

    # Group by frame and apply filter
    positions_by_frame: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        if p.frame_number not in positions_by_frame:
            positions_by_frame[p.frame_number] = []
        positions_by_frame[p.frame_number].append(p)

    # Filter each frame
    filtered_positions: list[PlayerPosition] = []
    for frame_num in sorted(positions_by_frame.keys()):
        frame_players = positions_by_frame[frame_num]
        filtered = player_filter.filter(frame_players)
        filtered_positions.extend(filtered)

    return filtered_positions


def evaluate_config(
    config: PlayerFilterConfig,
    rallies: list[tuple[CachedRallyData, GroundTruthResult]],
    iou_threshold: float = 0.5,
    min_rally_f1: float | None = None,
) -> FilterConfigResult:
    """Evaluate a single filter config across all rallies.

    Args:
        config: Filter configuration to evaluate.
        rallies: List of (cached_data, ground_truth) tuples.
        iou_threshold: IoU threshold for matching.
        min_rally_f1: If set, reject configs where any rally drops below.

    Returns:
        FilterConfigResult with metrics and rejection status.
    """
    per_rally_metrics: list[tuple[str, MOTMetrics]] = []
    aggregate = MOTMetrics()

    for cached, gt in rallies:
        # Apply filter config
        filtered_positions = apply_filter_config(
            cached.raw_positions,
            cached.ball_positions,
            cached.frame_count,
            config,
        )

        # Create PlayerTrackingResult for evaluation
        predictions = PlayerTrackingResult(
            positions=filtered_positions,
            frame_count=cached.frame_count,
            video_fps=cached.video_fps,
        )

        # Evaluate against ground truth
        result = evaluate_rally(
            rally_id=cached.rally_id,
            ground_truth=gt,
            predictions=predictions,
            iou_threshold=iou_threshold,
            video_width=cached.video_width,
            video_height=cached.video_height,
        )

        rally_metrics = result.aggregate
        per_rally_metrics.append((cached.rally_id, rally_metrics))

        # Accumulate aggregate metrics
        aggregate.num_gt += rally_metrics.num_gt
        aggregate.num_pred += rally_metrics.num_pred
        aggregate.num_matches += rally_metrics.num_matches
        aggregate.num_misses += rally_metrics.num_misses
        aggregate.num_false_positives += rally_metrics.num_false_positives
        aggregate.num_id_switches += rally_metrics.num_id_switches

    # Check min_rally_f1 constraint
    rejected = False
    rejection_reason = ""
    if min_rally_f1 is not None:
        for rally_id, metrics in per_rally_metrics:
            if metrics.f1 < min_rally_f1:
                rejected = True
                rejection_reason = (
                    f"Rally {rally_id[:8]}... F1={metrics.f1:.1%} < {min_rally_f1:.1%}"
                )
                break

    return FilterConfigResult(
        config=config,
        aggregate_metrics=aggregate,
        per_rally_metrics=per_rally_metrics,
        rejected=rejected,
        rejection_reason=rejection_reason,
    )


def grid_search(
    rallies: list[tuple[CachedRallyData, GroundTruthResult]],
    param_grid: dict[str, list[float | int]],
    iou_threshold: float = 0.5,
    min_rally_f1: float | None = None,
    id_switch_penalty: float = 0.01,
    progress_callback: Callable[[int, int], None] | None = None,
) -> FilterGridSearchResult:
    """Grid search over PlayerFilterConfig parameters.

    Scoring: score = f1 * (1 - id_switch_penalty * id_switches)

    Args:
        rallies: List of (cached_data, ground_truth) tuples.
        param_grid: Dict mapping parameter names to lists of values.
        iou_threshold: IoU threshold for matching.
        min_rally_f1: Reject configs where any rally drops below this F1.
        id_switch_penalty: Penalty per ID switch in scoring.
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        FilterGridSearchResult with best config and all results.
    """
    # Generate all configs
    configs = generate_filter_configs(param_grid)
    total_configs = len(configs)

    logger.info(f"Grid search: {total_configs} configs, {len(rallies)} rallies")

    # Evaluate default config first for comparison
    default_config = get_default_config()
    default_result = evaluate_config(
        default_config, rallies, iou_threshold, min_rally_f1=None
    )
    default_f1 = default_result.aggregate_metrics.f1
    default_mota = default_result.aggregate_metrics.mota

    logger.info(f"Default config: F1={default_f1:.1%}, MOTA={default_mota:.1%}")

    # Evaluate all configs
    all_results: list[FilterConfigResult] = []
    rejected_count = 0

    for i, config in enumerate(configs):
        result = evaluate_config(config, rallies, iou_threshold, min_rally_f1)
        all_results.append(result)

        if result.rejected:
            rejected_count += 1

        if progress_callback:
            progress_callback(i + 1, total_configs)

    # Find best non-rejected config using score
    def compute_score(result: FilterConfigResult) -> float:
        if result.rejected:
            return -1.0
        f1 = result.aggregate_metrics.f1
        switches = result.aggregate_metrics.num_id_switches
        return f1 * (1 - id_switch_penalty * switches)

    valid_results = [r for r in all_results if not r.rejected]

    if not valid_results:
        logger.warning("All configs rejected! Using default config.")
        best_result = default_result
    else:
        best_result = max(valid_results, key=compute_score)

    best_f1 = best_result.aggregate_metrics.f1
    best_mota = best_result.aggregate_metrics.mota
    best_switches = best_result.aggregate_metrics.num_id_switches

    improvement = best_f1 - default_f1

    logger.info(
        f"Best config: F1={best_f1:.1%} (improvement: {improvement:+.1%}), "
        f"MOTA={best_mota:.1%}, ID switches={best_switches}"
    )
    logger.info(f"Config diff: {describe_config_diff(best_result.config)}")

    # Sort results by score (best first)
    all_results.sort(key=compute_score, reverse=True)

    return FilterGridSearchResult(
        best_config=best_result.config,
        best_f1=best_f1,
        best_mota=best_mota,
        best_id_switches=best_switches,
        all_results=all_results,
        rejected_count=rejected_count,
        total_configs=total_configs,
        default_f1=default_f1,
        default_mota=default_mota,
        improvement_f1=improvement,
    )
