"""Tune ball filter pipeline for WASB+VballNet ensemble output.

Phase A: Ablation study — test each filter stage independently to identify
         which stages help vs hurt ensemble output.

Phase B: Grid search — sweep promising parameters from ablation results.

Requires cached ensemble positions (run cache_ensemble_positions.py first).

Usage:
    cd analysis
    uv run python scripts/tune_ensemble_filter.py                    # Ablation only
    uv run python scripts/tune_ensemble_filter.py --grid             # Ablation + grid search
    uv run python scripts/tune_ensemble_filter.py --grid -o out.json # Save results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.tracking.ball_grid_search import (
    BallRawCache,
    CachedBallData,
    ball_grid_search,
    evaluate_ball_config,
)
from rallycut.evaluation.tracking.ball_param_grid import (
    BALL_ENSEMBLE_GRID,
    describe_ball_config_diff,
)
from rallycut.evaluation.tracking.db import load_labeled_rallies
from rallycut.labeling.ground_truth import GroundTruthPosition
from rallycut.tracking.ball_filter import BallFilterConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
console = Console()

ENSEMBLE_CACHE_DIR = Path.home() / ".cache" / "rallycut" / "ensemble_grid_search"


def _no_filter_config() -> BallFilterConfig:
    """Config that disables all filter stages (passthrough)."""
    return BallFilterConfig(
        enable_kalman=False,
        enable_segment_pruning=False,
        enable_oscillation_pruning=False,
        enable_exit_ghost_removal=False,
        enable_outlier_removal=False,
        enable_blip_removal=False,
        enable_interpolation=False,
        enable_motion_energy_filter=False,
    )


def _all_filter_config() -> BallFilterConfig:
    """Current 'light filter' from eval_ensemble.py."""
    return BallFilterConfig(
        enable_segment_pruning=True,
        segment_jump_threshold=0.20,
        min_segment_frames=5,
        enable_motion_energy_filter=False,
        enable_exit_ghost_removal=True,
        enable_oscillation_pruning=True,
        min_oscillation_frames=12,
        enable_outlier_removal=True,
        enable_blip_removal=True,
        enable_interpolation=True,
        max_interpolation_gap=10,
        min_output_confidence=0.05,
    )


def build_ablation_configs() -> list[tuple[str, BallFilterConfig]]:
    """Build named ablation configs: each stage on/off independently."""
    base_off = _no_filter_config()
    base_on = _all_filter_config()

    configs: list[tuple[str, BallFilterConfig]] = []

    # 1. No filter (unfiltered baseline)
    configs.append(("no_filter", base_off))

    # 2. All filters (current light filter)
    configs.append(("all_filters", base_on))

    # 3-8. Only one stage enabled
    only_stages = [
        ("only_segment_pruning", {"enable_segment_pruning": True, "min_segment_frames": 5,
                                   "segment_jump_threshold": 0.20, "min_output_confidence": 0.05}),
        ("only_oscillation", {"enable_oscillation_pruning": True, "min_oscillation_frames": 12}),
        ("only_exit_ghost", {"enable_exit_ghost_removal": True}),
        ("only_outlier", {"enable_outlier_removal": True}),
        ("only_blip", {"enable_blip_removal": True}),
        ("only_interpolation", {"enable_interpolation": True, "max_interpolation_gap": 10}),
    ]

    for name, overrides in only_stages:
        cfg_dict = {f.name: getattr(base_off, f.name) for f in fields(BallFilterConfig)}
        cfg_dict.update(overrides)
        configs.append((name, BallFilterConfig(**cfg_dict)))

    # 9-11. All except one stage
    except_stages = [
        ("all_except_blip", {"enable_blip_removal": False}),
        ("all_except_outlier", {"enable_outlier_removal": False}),
        ("all_except_oscillation", {"enable_oscillation_pruning": False}),
    ]

    for name, overrides in except_stages:
        cfg_dict = {f.name: getattr(base_on, f.name) for f in fields(BallFilterConfig)}
        cfg_dict.update(overrides)
        configs.append((name, BallFilterConfig(**cfg_dict)))

    return configs


def run_ablation(
    rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]],
) -> list[dict]:
    """Run ablation study and return results."""
    configs = build_ablation_configs()
    results = []

    console.print(f"\n[bold]Phase A: Ablation Study ({len(configs)} configs)[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Config", min_width=25)
    table.add_column("Det%", justify="right")
    table.add_column("Match%", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("<20px", justify="right")
    table.add_column("Delta", justify="right")

    baseline_match = None

    for name, config in configs:
        result = evaluate_ball_config(config, rallies)
        m = result.aggregate_metrics

        if baseline_match is None:
            baseline_match = m.match_rate

        delta = m.match_rate - baseline_match
        delta_str = f"{delta:+.1%}" if delta != 0 else "-"

        # Color coding
        if delta > 0.005:
            match_str = f"[green]{m.match_rate:.1%}[/green]"
            delta_str = f"[green]{delta_str}[/green]"
        elif delta < -0.005:
            match_str = f"[red]{m.match_rate:.1%}[/red]"
            delta_str = f"[red]{delta_str}[/red]"
        else:
            match_str = f"{m.match_rate:.1%}"

        table.add_row(
            name,
            f"{m.detection_rate:.1%}",
            match_str,
            f"{m.mean_error_px:.1f}px",
            f"{m.error_under_20px_rate:.1%}",
            delta_str,
        )

        per_rally = []
        for rid, rm in result.per_rally_metrics:
            per_rally.append({
                "rally_id": rid[:8],
                "detection_rate": rm.detection_rate,
                "match_rate": rm.match_rate,
                "mean_error_px": rm.mean_error_px,
            })

        results.append({
            "name": name,
            "config_diff": describe_ball_config_diff(config, _no_filter_config()),
            "detection_rate": m.detection_rate,
            "match_rate": m.match_rate,
            "mean_error_px": m.mean_error_px,
            "median_error_px": m.median_error_px,
            "error_under_20px_rate": m.error_under_20px_rate,
            "delta_vs_unfiltered": m.match_rate - baseline_match,
            "per_rally": per_rally,
        })

    console.print(table)
    return results


def run_grid_search(
    rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]],
) -> dict:
    """Run grid search over ensemble-specific parameter grid."""
    from rallycut.evaluation.tracking.ball_param_grid import ball_grid_size

    grid_size = ball_grid_size(BALL_ENSEMBLE_GRID)
    console.print(f"\n[bold]Phase B: Grid Search ({grid_size} configs)[/bold]\n")

    from rich.progress import Progress

    with Progress(console=console) as progress:
        task = progress.add_task("Grid search...", total=grid_size)

        def _cb(current: int, total: int) -> None:
            progress.update(task, completed=current)

        result = ball_grid_search(
            rallies=rallies,
            param_grid=BALL_ENSEMBLE_GRID,
            progress_callback=_cb,
        )

    # Print top 10
    console.print("\n[bold]Top 10 Configs (by score):[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Config Diff", min_width=40)
    table.add_column("Det%", justify="right")
    table.add_column("Match%", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("<20px", justify="right")

    for i, r in enumerate(result.all_results[:10]):
        if r.rejected:
            continue
        m = r.aggregate_metrics
        diff = describe_ball_config_diff(r.config)
        table.add_row(
            str(i + 1),
            diff[:60],
            f"{m.detection_rate:.1%}",
            f"{m.match_rate:.1%}",
            f"{m.mean_error_px:.1f}px",
            f"{m.error_under_20px_rate:.1%}",
        )

    console.print(table)

    # Summary
    console.print(f"\nDefault: match={result.default_match_rate:.1%}, "
                  f"error={result.default_mean_error_px:.1f}px")
    console.print(f"Best:    match={result.best_match_rate:.1%}, "
                  f"error={result.best_mean_error_px:.1f}px")
    console.print(f"Improvement: {result.improvement_match_rate:+.1%}")
    console.print(f"Best config: {describe_ball_config_diff(result.best_config)}")
    console.print(f"Rejected: {result.rejected_count}/{result.total_configs}")

    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune ball filter for WASB+VballNet ensemble"
    )
    parser.add_argument(
        "--grid", action="store_true", help="Run grid search after ablation"
    )
    parser.add_argument(
        "-o", "--output", help="Save results to JSON file"
    )
    parser.add_argument(
        "--rally", help="Evaluate specific rally ID (prefix match)"
    )
    args = parser.parse_args()

    # Load cached ensemble positions
    cache = BallRawCache(cache_dir=ENSEMBLE_CACHE_DIR)
    cached_ids = cache.list_cached()

    if not cached_ids:
        console.print("[red]No cached ensemble positions found.[/red]")
        console.print("Run cache_ensemble_positions.py first:")
        console.print("  uv run python scripts/cache_ensemble_positions.py")
        sys.exit(1)

    console.print(f"Found {len(cached_ids)} cached ensemble rallies")

    # Load GT rallies
    rallies_db = load_labeled_rallies()
    if not rallies_db:
        console.print("[red]No rallies with ball GT found[/red]")
        sys.exit(1)

    if args.rally:
        rallies_db = [r for r in rallies_db if r.rally_id.startswith(args.rally)]

    # Pair cached data with GT
    rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]] = []
    for rally in rallies_db:
        cached = cache.get(rally.rally_id)
        if cached is None:
            logger.warning(f"Rally {rally.rally_id[:8]} not in ensemble cache, skipping")
            continue
        rallies.append((cached, rally.ground_truth.positions))

    if not rallies:
        console.print("[red]No rallies matched between cache and GT[/red]")
        sys.exit(1)

    console.print(f"Evaluating {len(rallies)} rallies\n")

    # Phase A: Ablation
    ablation_results = run_ablation(rallies)

    # Phase B: Grid search (optional)
    grid_results = None
    if args.grid:
        grid_results = run_grid_search(rallies)

    # Save results
    if args.output:
        output = {
            "ablation": ablation_results,
        }
        if grid_results:
            output["grid_search"] = grid_results

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        console.print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
