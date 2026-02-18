"""Parameter grid configuration for tuning BallFilterConfig."""

from __future__ import annotations

from dataclasses import fields
from itertools import product
from typing import Any

from rallycut.tracking.ball_filter import BallFilterConfig

# Outlier removal grid (6 combinations: 3*2)
BALL_OUTLIER_GRID: dict[str, list[float | int | bool]] = {
    "enable_outlier_removal": [True],
    "max_trajectory_deviation": [0.05, 0.08, 0.12],
    "edge_margin": [0.02, 0.03],
}


# Segment pruning grid (18 combinations: 3*3*2)
# Tunes post-processing that removes short false segments at rally boundaries
BALL_SEGMENT_PRUNING_GRID: dict[str, list[float | int | bool]] = {
    "segment_jump_threshold": [0.10, 0.15, 0.20],
    "min_segment_frames": [10, 20, 30],
    "enable_segment_pruning": [True, False],
}


# Oscillation pruning grid (18 combinations: 3*3*2)
# Tunes cluster-based detection of player-locking after ball exits frame
BALL_OSCILLATION_GRID: dict[str, list[float | int | bool]] = {
    "enable_oscillation_pruning": [True],
    "min_oscillation_frames": [6, 8, 12],
    "oscillation_reversal_rate": [0.20, 0.25, 0.35],
    "oscillation_min_displacement": [0.02, 0.03],
}


# Ensemble (WASB+VballNet) filter tuning grid (1152 combinations)
# Tuned for ensemble output where WASB provides high-precision positions and
# VballNet fills gaps. Key differences from VballNet-only grids:
# - Source-aware mode: WASB positions protected from outlier/blip/oscillation removal
# - Motion energy filter disabled (WASB doesn't produce stationary FPs)
# - Shorter min_segment_frames (WASB segments can be short but accurate)
# - Wider blip_max_deviation (VballNet fallback positions deviate from WASB trajectory)
# - Enable/disable toggles for stages that may hurt ensemble output
BALL_ENSEMBLE_GRID: dict[str, list[float | int | bool]] = {
    "enable_motion_energy_filter": [False],
    "ensemble_source_aware": [True, False],
    "min_segment_frames": [3, 5, 8, 10],
    "segment_jump_threshold": [0.15, 0.20, 0.25],
    "blip_max_deviation": [0.10, 0.15, 0.20],
    "max_interpolation_gap": [5, 10],
    "enable_blip_removal": [True, False],
    "enable_outlier_removal": [True, False],
    "enable_oscillation_pruning": [True, False],
}


# WASB-only filter tuning grid (1152 combinations)
# For fine-tuned WASB without VballNet fallback. All positions are WASB source,
# so source_aware=False (otherwise it disables most filter stages).
# Includes stationarity filter (catches WASB player lock-on) and exit_approach_frames.
BALL_WASB_GRID: dict[str, list[float | int | bool]] = {
    "ensemble_source_aware": [False],  # Locked: all positions are WASB, no source distinction
    "enable_stationarity_filter": [True, False],
    "min_segment_frames": [5, 8, 12],
    "segment_jump_threshold": [0.15, 0.20, 0.25],
    "blip_max_deviation": [0.10, 0.15],
    "max_interpolation_gap": [5, 10],
    "exit_approach_frames": [3, 4],
    "enable_blip_removal": [True, False],
    "enable_outlier_removal": [True, False],
    "enable_oscillation_pruning": [True, False],
}


# All available grids
BALL_AVAILABLE_GRIDS: dict[str, dict[str, list[Any]]] = {
    "outlier": BALL_OUTLIER_GRID,
    "segment-pruning": BALL_SEGMENT_PRUNING_GRID,
    "oscillation": BALL_OSCILLATION_GRID,
    "ensemble": BALL_ENSEMBLE_GRID,
    "wasb": BALL_WASB_GRID,
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
        name: Grid name (outlier, segment-pruning, oscillation, ensemble, wasb).

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
