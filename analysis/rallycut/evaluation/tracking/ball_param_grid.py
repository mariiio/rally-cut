"""Parameter grid configuration for tuning BallFilterConfig and heatmap decoding."""

from __future__ import annotations

from dataclasses import fields
from itertools import product
from typing import Any

from rallycut.tracking.ball_filter import BallFilterConfig

# Quick grid for fast iteration (81 combinations: 3^4)
# Focused on high-impact Kalman filter parameters
BALL_QUICK_GRID: dict[str, list[float | int | bool]] = {
    "process_noise_position": [0.0005, 0.001, 0.002],
    "process_noise_velocity": [0.005, 0.01, 0.02],
    "measurement_noise": [0.002, 0.005, 0.01],
    "lag_frames": [2, 3, 5],
}


# Lag-focused grid (validate current lag compensation setting)
# 14 combinations: 2 * 7
BALL_LAG_GRID: dict[str, list[float | int | bool]] = {
    "enable_lag_compensation": [True, False],
    "lag_frames": [0, 1, 2, 3, 4, 5, 8],
}


# Full grid for comprehensive sweep (486 combinations: 3*3*3*3*3*2)
BALL_FULL_GRID: dict[str, list[float | int | bool]] = {
    "process_noise_position": [0.0005, 0.001, 0.002],
    "process_noise_velocity": [0.005, 0.01, 0.02],
    "measurement_noise": [0.002, 0.005, 0.01],
    "lag_frames": [2, 3, 5],
    "max_velocity": [0.25, 0.30, 0.35],
    "enable_lag_compensation": [True, False],
}


# Confidence-focused grid (27 combinations: 3^3)
# Tune confidence thresholds and occlusion handling
BALL_CONFIDENCE_GRID: dict[str, list[float | int | bool]] = {
    "min_confidence_for_update": [0.2, 0.3, 0.4],
    "max_occlusion_frames": [20, 30, 45],
    "max_velocity": [0.25, 0.30, 0.35],
}


# Heatmap decoding parameters grid (16 combinations: 4*2*2)
# Note: These are passed to HeatmapDecodingConfig in BallTracker, not BallFilterConfig
# Cannot be used with ball_grid_search directly - requires re-running inference
HEATMAP_GRID: dict[str, list[float | str | bool]] = {
    "threshold": [0.3, 0.4, 0.5, 0.6],
    "centroid_method": ["contour", "weighted"],
    "adaptive_threshold": [True, False],
}


# All available grids
BALL_AVAILABLE_GRIDS: dict[str, dict[str, list[Any]]] = {
    "quick": BALL_QUICK_GRID,
    "lag": BALL_LAG_GRID,
    "full": BALL_FULL_GRID,
    "confidence": BALL_CONFIDENCE_GRID,
    "heatmap": HEATMAP_GRID,
}


def get_default_ball_config() -> BallFilterConfig:
    """Get default BallFilterConfig."""
    return BallFilterConfig()


def generate_ball_filter_configs(
    grid: dict[str, list[float | int | bool]],
    base_config: BallFilterConfig | None = None,
) -> list[BallFilterConfig]:
    """Generate all parameter combinations from a grid.

    Args:
        grid: Dict mapping parameter names to lists of values to try.
        base_config: Base config to start from. Defaults to BallFilterConfig().

    Returns:
        List of BallFilterConfig, one for each combination.
    """
    if base_config is None:
        base_config = BallFilterConfig()

    if not grid:
        return [base_config]

    # Validate parameter names
    valid_fields = {f.name for f in fields(BallFilterConfig)}
    for key in grid:
        if key not in valid_fields:
            raise ValueError(
                f"Unknown parameter '{key}'. Valid parameters: {sorted(valid_fields)}"
            )

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    combinations = []
    for combo in product(*values):
        # Start from base config and override with grid values
        config_dict = {f.name: getattr(base_config, f.name) for f in fields(BallFilterConfig)}

        # Apply grid overrides
        for key, value in zip(keys, combo):
            config_dict[key] = value

        combinations.append(BallFilterConfig(**config_dict))

    return combinations


def get_ball_grid(name: str) -> dict[str, list[Any]]:
    """Get a parameter grid by name.

    Args:
        name: Grid name (quick, lag, full, confidence, heatmap).

    Returns:
        Grid dict.

    Raises:
        ValueError: If grid name not found.
    """
    if name not in BALL_AVAILABLE_GRIDS:
        available = ", ".join(BALL_AVAILABLE_GRIDS.keys())
        raise ValueError(f"Unknown grid '{name}'. Available: {available}")
    return BALL_AVAILABLE_GRIDS[name]


def ball_grid_size(grid: dict[str, list[Any]]) -> int:
    """Calculate number of combinations in a grid.

    Args:
        grid: Dict mapping parameter names to lists of values.

    Returns:
        Total number of unique combinations.
    """
    if not grid:
        return 1
    size = 1
    for values in grid.values():
        # Count unique values (handles bool, str, int, float)
        unique_count = len(set(tuple(v) if isinstance(v, list) else v for v in values))
        size *= unique_count
    return size


def describe_ball_config_diff(
    config: BallFilterConfig,
    base: BallFilterConfig | None = None,
) -> str:
    """Describe how a config differs from base (or defaults).

    Args:
        config: Config to describe.
        base: Base config for comparison. Defaults to BallFilterConfig().

    Returns:
        Human-readable string of differences.
    """
    if base is None:
        base = BallFilterConfig()

    diffs = []
    for f in fields(BallFilterConfig):
        new_val = getattr(config, f.name)
        old_val = getattr(base, f.name)
        if new_val != old_val:
            # Format floats nicely
            if isinstance(new_val, float):
                diffs.append(f"{f.name}={new_val:.4g}")
            else:
                diffs.append(f"{f.name}={new_val}")

    return ", ".join(diffs) if diffs else "(default)"
