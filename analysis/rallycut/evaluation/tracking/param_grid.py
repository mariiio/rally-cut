"""Parameter grid configuration for tuning PlayerFilterConfig."""

from __future__ import annotations

from dataclasses import fields
from itertools import product

from rallycut.tracking.player_filter import PlayerFilterConfig

# Quick grid for fast iteration (81 combinations: 3^4)
# Focused on high-impact parameters
QUICK_GRID: dict[str, list[float]] = {
    "min_presence_rate": [0.10, 0.20, 0.30],
    "min_position_spread_for_primary": [0.010, 0.015, 0.020],
    "ball_proximity_radius": [0.15, 0.20, 0.25],
    "referee_sideline_threshold": [0.15, 0.20, 0.25],
}


# Full grid for comprehensive sweep (1620 combinations: 5*3*4*3*3*3)
FULL_GRID: dict[str, list[float]] = {
    "min_presence_rate": [0.10, 0.15, 0.20, 0.25, 0.30],
    "min_stability_score": [0.15, 0.20, 0.25],
    "min_position_spread_for_primary": [0.010, 0.015, 0.020, 0.025],
    "ball_proximity_radius": [0.15, 0.20, 0.25],
    "referee_sideline_threshold": [0.15, 0.20, 0.25],
    "min_ball_proximity_for_stationary": [0.03, 0.05, 0.07],
}


# Referee detection focused grid (135 combinations: 5*3*3*3)
REFEREE_GRID: dict[str, list[float]] = {
    "referee_sideline_threshold": [0.15, 0.18, 0.20, 0.22, 0.25],
    "referee_movement_ratio_min": [1.2, 1.5, 1.8],
    "referee_ball_proximity_max": [0.10, 0.12, 0.15],
    "referee_y_std_max": [0.03, 0.04, 0.05],
}


# Stability scoring focused grid (243 combinations: 3^5)
STABILITY_GRID: dict[str, list[float]] = {
    "min_presence_rate": [0.15, 0.20, 0.25],
    "min_stability_score": [0.15, 0.20, 0.25],
    "presence_weight": [0.30, 0.40, 0.50],
    "bbox_area_weight": [0.25, 0.35, 0.45],
    "ball_proximity_weight": [0.15, 0.25, 0.35],
}


# Track merging focused grid (81 combinations: 3^4)
MERGE_GRID: dict[str, list[float | int]] = {
    "max_gap_frames": [60, 90, 120],
    "max_merge_distance": [0.30, 0.40, 0.50],
    "merge_distance_per_frame": [0.006, 0.008, 0.010],
    "min_position_spread_for_primary": [0.012, 0.015, 0.018],
}


# All available grids
AVAILABLE_GRIDS: dict[str, dict[str, list[float | int]]] = {
    "quick": QUICK_GRID,
    "full": FULL_GRID,
    "referee": REFEREE_GRID,
    "stability": STABILITY_GRID,
    "merge": MERGE_GRID,
}


def get_default_config() -> PlayerFilterConfig:
    """Get default PlayerFilterConfig."""
    return PlayerFilterConfig()


def generate_filter_configs(
    grid: dict[str, list[float | int]],
    base_config: PlayerFilterConfig | None = None,
) -> list[PlayerFilterConfig]:
    """Generate all parameter combinations from a grid.

    Args:
        grid: Dict mapping parameter names to lists of values to try.
        base_config: Base config to start from. Defaults to PlayerFilterConfig().

    Returns:
        List of PlayerFilterConfig, one for each combination.
    """
    if base_config is None:
        base_config = PlayerFilterConfig()

    if not grid:
        return [base_config]

    # Validate parameter names
    valid_fields = {f.name for f in fields(PlayerFilterConfig)}
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
        config_dict = {f.name: getattr(base_config, f.name) for f in fields(PlayerFilterConfig)}

        # Apply grid overrides
        for key, value in zip(keys, combo):
            config_dict[key] = value

        combinations.append(PlayerFilterConfig(**config_dict))

    return combinations


def get_grid(name: str) -> dict[str, list[float | int]]:
    """Get a parameter grid by name.

    Args:
        name: Grid name (quick, full, referee, stability, merge).

    Returns:
        Grid dict.

    Raises:
        ValueError: If grid name not found.
    """
    if name not in AVAILABLE_GRIDS:
        available = ", ".join(AVAILABLE_GRIDS.keys())
        raise ValueError(f"Unknown grid '{name}'. Available: {available}")
    return AVAILABLE_GRIDS[name]


def grid_size(grid: dict[str, list[float | int]]) -> int:
    """Calculate number of combinations in a grid."""
    if not grid:
        return 1
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def describe_config_diff(
    config: PlayerFilterConfig,
    base: PlayerFilterConfig | None = None,
) -> str:
    """Describe how a config differs from base (or defaults).

    Args:
        config: Config to describe.
        base: Base config for comparison. Defaults to PlayerFilterConfig().

    Returns:
        Human-readable string of differences.
    """
    if base is None:
        base = PlayerFilterConfig()

    diffs = []
    for f in fields(PlayerFilterConfig):
        new_val = getattr(config, f.name)
        old_val = getattr(base, f.name)
        if new_val != old_val:
            diffs.append(f"{f.name}={new_val}")

    return ", ".join(diffs) if diffs else "(default)"
