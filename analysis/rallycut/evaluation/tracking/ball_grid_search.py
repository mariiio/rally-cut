"""Grid search over BallFilterConfig parameters.

The grid search efficiently evaluates ball filter configurations by:
1. Caching raw ball positions (before Kalman filtering)
2. Re-running only the filter pipeline with different configs (fast, per-config)
3. Computing ball tracking metrics against ground truth
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from rallycut.evaluation.tracking.ball_metrics import (
    BallTrackingMetrics,
    evaluate_ball_tracking,
)
from rallycut.evaluation.tracking.ball_param_grid import (
    describe_ball_config_diff,
    generate_ball_filter_configs,
    get_default_ball_config,
)
from rallycut.labeling.ground_truth import GroundTruthPosition
from rallycut.tracking.ball_filter import BallFilterConfig, BallTemporalFilter
from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)


@dataclass
class CachedBallData:
    """Cached raw ball data for a rally (before Kalman filtering)."""

    rally_id: str
    video_id: str
    raw_ball_positions: list[BallPosition]  # Unfiltered VballNet output
    video_fps: float
    frame_count: int
    video_width: int = 1920
    video_height: int = 1080
    start_ms: int = 0
    end_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "rally_id": self.rally_id,
            "video_id": self.video_id,
            "raw_ball_positions": [
                {
                    "frame_number": bp.frame_number,
                    "x": bp.x,
                    "y": bp.y,
                    "confidence": bp.confidence,
                }
                for bp in self.raw_ball_positions
            ],
            "video_fps": self.video_fps,
            "frame_count": self.frame_count,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedBallData:
        """Create from dict."""
        raw_ball_positions = [
            BallPosition(
                frame_number=bp["frame_number"],
                x=bp["x"],
                y=bp["y"],
                confidence=bp["confidence"],
            )
            for bp in data.get("raw_ball_positions", [])
        ]

        return cls(
            rally_id=data["rally_id"],
            video_id=data["video_id"],
            raw_ball_positions=raw_ball_positions,
            video_fps=data.get("video_fps", 30.0),
            frame_count=data.get("frame_count", 0),
            video_width=data.get("video_width", 1920),
            video_height=data.get("video_height", 1080),
            start_ms=data.get("start_ms", 0),
            end_ms=data.get("end_ms", 0),
        )


@dataclass
class BallRawCache:
    """Cache for raw ball tracking positions (before Kalman filtering).

    Default cache location: ~/.cache/rallycut/ball_grid_search/
    """

    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "rallycut" / "ball_grid_search"
    )

    def __post_init__(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, rally_id: str) -> Path:
        """Get cache file path for a rally."""
        hash_prefix = hashlib.sha256(rally_id.encode()).hexdigest()[:16]
        return self.cache_dir / f"ball_{hash_prefix}.json"

    def has(self, rally_id: str) -> bool:
        """Check if rally is cached."""
        return self._cache_path(rally_id).exists()

    def get(self, rally_id: str) -> CachedBallData | None:
        """Get cached rally data."""
        cache_path = self._cache_path(rally_id)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            cached = CachedBallData.from_dict(data)

            if cached.rally_id != rally_id:
                logger.warning(f"Cache collision: expected {rally_id}, got {cached.rally_id}")
                return None

            return cached
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load ball cache for {rally_id}: {e}")
            return None

    def put(self, data: CachedBallData) -> None:
        """Store rally data in cache."""
        cache_path = self._cache_path(data.rally_id)

        try:
            with open(cache_path, "w") as f:
                json.dump(data.to_dict(), f)
            logger.debug(f"Cached raw ball positions for {data.rally_id[:8]}...")
        except OSError as e:
            logger.warning(f"Failed to cache ball data {data.rally_id}: {e}")

    def delete(self, rally_id: str) -> bool:
        """Delete cached data for a rally."""
        cache_path = self._cache_path(rally_id)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cached data."""
        count = 0
        for cache_file in self.cache_dir.glob("ball_*.json"):
            cache_file.unlink()
            count += 1
        logger.info(f"Cleared {count} cached ball position files")
        return count

    def list_cached(self) -> list[str]:
        """List all cached rally IDs."""
        rally_ids = []
        for cache_file in self.cache_dir.glob("ball_*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                rally_ids.append(data["rally_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return rally_ids

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("ball_*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "count": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
        }


@dataclass
class BallFilterConfigResult:
    """Result for a single ball filter config across all rallies."""

    config: BallFilterConfig
    aggregate_metrics: BallTrackingMetrics
    per_rally_metrics: list[tuple[str, BallTrackingMetrics]]  # (rally_id, metrics)
    rejected: bool = False
    rejection_reason: str = ""


@dataclass
class BallFilterGridSearchResult:
    """Complete grid search result for ball filter."""

    best_config: BallFilterConfig
    best_detection_rate: float
    best_match_rate: float
    best_mean_error_px: float
    all_results: list[BallFilterConfigResult]
    rejected_count: int
    total_configs: int

    # Comparison to default
    default_detection_rate: float = 0.0
    default_match_rate: float = 0.0
    default_mean_error_px: float = 0.0
    improvement_match_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "best_config": {
                f.name: getattr(self.best_config, f.name)
                for f in fields(BallFilterConfig)
            },
            "best_detection_rate": self.best_detection_rate,
            "best_match_rate": self.best_match_rate,
            "best_mean_error_px": self.best_mean_error_px,
            "default_detection_rate": self.default_detection_rate,
            "default_match_rate": self.default_match_rate,
            "default_mean_error_px": self.default_mean_error_px,
            "improvement_match_rate": self.improvement_match_rate,
            "rejected_count": self.rejected_count,
            "total_configs": self.total_configs,
            "all_results": [
                {
                    "config_diff": describe_ball_config_diff(r.config),
                    "detection_rate": r.aggregate_metrics.detection_rate,
                    "match_rate": r.aggregate_metrics.match_rate,
                    "mean_error_px": r.aggregate_metrics.mean_error_px,
                    "median_error_px": r.aggregate_metrics.median_error_px,
                    "error_under_20px_rate": r.aggregate_metrics.error_under_20px_rate,
                    "rejected": r.rejected,
                    "rejection_reason": r.rejection_reason,
                    "per_rally": [
                        {
                            "rally_id": rid[:8],
                            "detection_rate": m.detection_rate,
                            "match_rate": m.match_rate,
                            "mean_error_px": m.mean_error_px,
                        }
                        for rid, m in r.per_rally_metrics
                    ],
                }
                for r in self.all_results
            ],
        }


def apply_ball_filter_config(
    raw_positions: list[BallPosition],
    config: BallFilterConfig,
) -> list[BallPosition]:
    """Apply a ball filter config to raw positions.

    This is the fast path - only runs the Kalman filter, not VballNet.

    Args:
        raw_positions: Unfiltered VballNet output.
        config: Filter configuration to apply.

    Returns:
        Filtered ball positions.
    """
    if not raw_positions:
        return []

    temporal_filter = BallTemporalFilter(config)
    return temporal_filter.filter_batch(raw_positions)


def evaluate_ball_config(
    config: BallFilterConfig,
    rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]],
    match_threshold_px: float = 50.0,
    min_confidence: float = 0.3,
    min_rally_detection_rate: float | None = None,
) -> BallFilterConfigResult:
    """Evaluate a single ball filter config across all rallies.

    Args:
        config: Filter configuration to evaluate.
        rallies: List of (cached_data, ground_truth_positions) tuples.
        match_threshold_px: Maximum distance for a "match".
        min_confidence: Minimum prediction confidence to consider.
        min_rally_detection_rate: If set, reject configs where any rally drops below.

    Returns:
        BallFilterConfigResult with metrics and rejection status.
    """
    from rallycut.evaluation.tracking.ball_metrics import aggregate_ball_metrics

    per_rally_metrics: list[tuple[str, BallTrackingMetrics]] = []

    for cached, gt_positions in rallies:
        # Apply filter config
        filtered_positions = apply_ball_filter_config(
            cached.raw_ball_positions,
            config,
        )

        # Evaluate against ground truth
        metrics = evaluate_ball_tracking(
            ground_truth=gt_positions,
            predictions=filtered_positions,
            video_width=cached.video_width,
            video_height=cached.video_height,
            video_fps=cached.video_fps,
            match_threshold_px=match_threshold_px,
            min_confidence=min_confidence,
        )

        per_rally_metrics.append((cached.rally_id, metrics))

    # Aggregate metrics
    aggregate = aggregate_ball_metrics([m for _, m in per_rally_metrics])

    # Check constraints
    rejected = False
    rejection_reason = ""
    if min_rally_detection_rate is not None:
        for rally_id, metrics in per_rally_metrics:
            if metrics.detection_rate < min_rally_detection_rate:
                rejected = True
                rejection_reason = (
                    f"Rally {rally_id[:8]}... detection={metrics.detection_rate:.1%} "
                    f"< {min_rally_detection_rate:.1%}"
                )
                break

    return BallFilterConfigResult(
        config=config,
        aggregate_metrics=aggregate,
        per_rally_metrics=per_rally_metrics,
        rejected=rejected,
        rejection_reason=rejection_reason,
    )


def ball_grid_search(
    rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]],
    param_grid: dict[str, list[float | int | bool]],
    match_threshold_px: float = 50.0,
    min_confidence: float = 0.3,
    min_rally_detection_rate: float | None = None,
    error_weight: float = 0.5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BallFilterGridSearchResult:
    """Grid search over BallFilterConfig parameters.

    Scoring: score = match_rate - error_weight * (mean_error_px / 100)

    Higher match_rate is better, lower mean_error is better.
    error_weight controls the trade-off (default 0.5 means 10px error = 5% match rate).

    Args:
        rallies: List of (cached_data, ground_truth_positions) tuples.
        param_grid: Dict mapping parameter names to lists of values.
        match_threshold_px: Maximum distance for a "match".
        min_confidence: Minimum prediction confidence to consider.
        min_rally_detection_rate: Reject configs where any rally drops below.
        error_weight: Weight for error in scoring (default 0.5).
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        BallFilterGridSearchResult with best config and all results.
    """
    # Generate all configs
    configs = generate_ball_filter_configs(param_grid)
    total_configs = len(configs)

    logger.info(f"Ball grid search: {total_configs} configs, {len(rallies)} rallies")

    # Evaluate default config first for comparison
    default_config = get_default_ball_config()
    default_result = evaluate_ball_config(
        default_config,
        rallies,
        match_threshold_px,
        min_confidence,
        min_rally_detection_rate=None,
    )
    default_detection_rate = default_result.aggregate_metrics.detection_rate
    default_match_rate = default_result.aggregate_metrics.match_rate
    default_mean_error = default_result.aggregate_metrics.mean_error_px

    logger.info(
        f"Default config: detection={default_detection_rate:.1%}, "
        f"match={default_match_rate:.1%}, error={default_mean_error:.1f}px"
    )

    # Evaluate all configs
    all_results: list[BallFilterConfigResult] = []
    rejected_count = 0

    for i, config in enumerate(configs):
        result = evaluate_ball_config(
            config,
            rallies,
            match_threshold_px,
            min_confidence,
            min_rally_detection_rate,
        )
        all_results.append(result)

        if result.rejected:
            rejected_count += 1

        if progress_callback:
            progress_callback(i + 1, total_configs)

    # Find best non-rejected config using score
    def compute_score(result: BallFilterConfigResult) -> float:
        if result.rejected:
            return -1.0
        match_rate = result.aggregate_metrics.match_rate
        mean_error = result.aggregate_metrics.mean_error_px
        # Normalize error to 0-1 scale (100px = 1.0)
        return match_rate - error_weight * (mean_error / 100.0)

    valid_results = [r for r in all_results if not r.rejected]

    if not valid_results:
        logger.warning("All configs rejected! Using default config.")
        best_result = default_result
    else:
        best_result = max(valid_results, key=compute_score)

    best_detection = best_result.aggregate_metrics.detection_rate
    best_match = best_result.aggregate_metrics.match_rate
    best_error = best_result.aggregate_metrics.mean_error_px

    improvement = best_match - default_match_rate

    logger.info(
        f"Best config: detection={best_detection:.1%}, match={best_match:.1%} "
        f"(improvement: {improvement:+.1%}), error={best_error:.1f}px"
    )
    logger.info(f"Config diff: {describe_ball_config_diff(best_result.config)}")

    # Sort results by score (best first)
    all_results.sort(key=compute_score, reverse=True)

    return BallFilterGridSearchResult(
        best_config=best_result.config,
        best_detection_rate=best_detection,
        best_match_rate=best_match,
        best_mean_error_px=best_error,
        all_results=all_results,
        rejected_count=rejected_count,
        total_configs=total_configs,
        default_detection_rate=default_detection_rate,
        default_match_rate=default_match_rate,
        default_mean_error_px=default_mean_error,
        improvement_match_rate=improvement,
    )
