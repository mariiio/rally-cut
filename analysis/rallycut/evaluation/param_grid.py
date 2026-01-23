"""Parameter grid configuration for tuning rally detection."""

from __future__ import annotations

from itertools import product

from rallycut.evaluation.cached_analysis import PostProcessingParams

# Default parameters from cutter.py and config.py
DEFAULT_PARAMS = PostProcessingParams(
    min_gap_seconds=5.0,
    rally_continuation_seconds=2.0,
    min_play_duration=1.0,
    padding_seconds=2.0,
    padding_end_seconds=3.0,
    boundary_confidence_threshold=0.35,
    min_active_density=0.25,
    min_active_windows=1,
)


# Quick grid for fast iteration (12 combinations)
QUICK_GRID: dict[str, list[float | int]] = {
    "min_gap_seconds": [3.0, 5.0, 7.0],
    "rally_continuation_seconds": [1.5, 2.0, 3.0],
}


# Full grid for comprehensive sweep (324 combinations)
FULL_GRID: dict[str, list[float | int]] = {
    "min_gap_seconds": [3.0, 5.0, 7.0],
    "rally_continuation_seconds": [1.0, 1.5, 2.0, 3.0],
    "min_play_duration": [0.5, 1.0, 2.0],
    "boundary_confidence_threshold": [0.25, 0.35, 0.45],
    "min_active_density": [0.15, 0.25, 0.35],
}


# Beach volleyball optimized grid (108 combinations)
# Focused on beach volleyball characteristics:
# - Shorter rallies than indoor (lower min_play_duration)
# - More ambient noise between points (higher min_gap_seconds)
# - Outdoor lighting variations (lower boundary_confidence_threshold)
BEACH_GRID: dict[str, list[float | int]] = {
    "min_gap_seconds": [4.0, 5.0, 6.0, 8.0],
    "rally_continuation_seconds": [1.5, 2.0, 2.5],
    "min_play_duration": [0.5, 1.0, 1.5],
    "boundary_confidence_threshold": [0.30, 0.35, 0.40],
}


# Strict grid for high precision (reduces false positives)
STRICT_GRID: dict[str, list[float | int]] = {
    "min_gap_seconds": [5.0, 7.0, 10.0],
    "rally_continuation_seconds": [1.0, 1.5],
    "min_play_duration": [1.5, 2.0, 3.0],
    "boundary_confidence_threshold": [0.40, 0.45, 0.50],
    "min_active_density": [0.30, 0.40],
}


# Relaxed grid for high recall (catches more rallies)
RELAXED_GRID: dict[str, list[float | int]] = {
    "min_gap_seconds": [3.0, 4.0, 5.0],
    "rally_continuation_seconds": [2.0, 2.5, 3.0],
    "min_play_duration": [0.5, 1.0],
    "boundary_confidence_threshold": [0.25, 0.30, 0.35],
    "min_active_density": [0.15, 0.20],
}


AVAILABLE_GRIDS = {
    "quick": QUICK_GRID,
    "full": FULL_GRID,
    "beach": BEACH_GRID,
    "strict": STRICT_GRID,
    "relaxed": RELAXED_GRID,
}


def generate_param_combinations(
    grid: dict[str, list[float | int]],
    base_params: PostProcessingParams | None = None,
) -> list[PostProcessingParams]:
    """Generate all parameter combinations from a grid.

    Args:
        grid: Dict mapping parameter names to lists of values to try.
        base_params: Base parameters to start from. Defaults to DEFAULT_PARAMS.

    Returns:
        List of PostProcessingParams, one for each combination.
    """
    if base_params is None:
        base_params = DEFAULT_PARAMS

    if not grid:
        return [base_params]

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    combinations = []
    for combo in product(*values):
        # Start from base params and override with grid values
        params_dict = {
            "min_gap_seconds": base_params.min_gap_seconds,
            "rally_continuation_seconds": base_params.rally_continuation_seconds,
            "min_play_duration": base_params.min_play_duration,
            "padding_seconds": base_params.padding_seconds,
            "padding_end_seconds": base_params.padding_end_seconds,
            "boundary_confidence_threshold": base_params.boundary_confidence_threshold,
            "min_active_density": base_params.min_active_density,
            "min_active_windows": base_params.min_active_windows,
        }

        # Apply grid overrides
        for key, value in zip(keys, combo):
            params_dict[key] = value

        combinations.append(PostProcessingParams(
            min_gap_seconds=float(params_dict["min_gap_seconds"]),
            rally_continuation_seconds=float(params_dict["rally_continuation_seconds"]),
            min_play_duration=float(params_dict["min_play_duration"]),
            padding_seconds=float(params_dict["padding_seconds"]),
            padding_end_seconds=float(params_dict["padding_end_seconds"]),
            boundary_confidence_threshold=float(params_dict["boundary_confidence_threshold"]),
            min_active_density=float(params_dict["min_active_density"]),
            min_active_windows=int(params_dict["min_active_windows"]),
        ))

    return combinations


def get_grid(name: str) -> dict[str, list[float | int]]:
    """Get a parameter grid by name.

    Args:
        name: Grid name (quick, full, beach, strict, relaxed).

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
