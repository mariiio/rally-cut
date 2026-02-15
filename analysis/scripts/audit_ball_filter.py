"""Comprehensive ball filter audit: ablation, incremental diagnostics, param sensitivity.

Runs a full per-stage ablation on ensemble cached positions against all 7 GT rallies.
Unlike tune_ensemble_filter.py, this tests the production config, stationarity,
exit ghost, interpolation, segment pruning — all stages systematically.

Three analysis modes:
  1. Ablation: production config, no filter, only-X, all-except-X for every stage
  2. Incremental: cumulative stage-by-stage metrics per rally
  3. Sensitivity: parameter sweeps on the most impactful stages

Requires cached ensemble positions (run cache_ensemble_positions.py first).

Usage:
    cd analysis
    uv run python scripts/audit_ball_filter.py                  # Full audit
    uv run python scripts/audit_ball_filter.py --ablation       # Ablation only
    uv run python scripts/audit_ball_filter.py --incremental    # Incremental only
    uv run python scripts/audit_ball_filter.py --sensitivity    # Sensitivity only
    uv run python scripts/audit_ball_filter.py -o results.json  # Save results
    uv run python scripts/audit_ball_filter.py --rally 1bfcbc4f # Single rally
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, fields
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.tracking.ball_grid_search import (
    BallRawCache,
    CachedBallData,
    evaluate_ball_config,
)
from rallycut.evaluation.tracking.ball_metrics import (
    evaluate_ball_tracking,
)
from rallycut.evaluation.tracking.db import load_labeled_rallies
from rallycut.labeling.ground_truth import GroundTruthPosition
from rallycut.tracking.ball_filter import (
    BallFilterConfig,
    BallTemporalFilter,
    get_ensemble_filter_config,
)
from rallycut.tracking.ball_tracker import BallPosition

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
console = Console()

ENSEMBLE_CACHE_DIR = Path.home() / ".cache" / "rallycut" / "ensemble_grid_search"


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _no_filter_config() -> BallFilterConfig:
    """All stages disabled (passthrough)."""
    return BallFilterConfig(
        enable_segment_pruning=False,
        enable_oscillation_pruning=False,
        enable_exit_ghost_removal=False,
        enable_outlier_removal=False,
        enable_blip_removal=False,
        enable_interpolation=False,
        enable_motion_energy_filter=False,
        enable_stationarity_filter=False,
        ensemble_source_aware=False,
    )


def _config_from_production(**overrides: object) -> BallFilterConfig:
    """Clone production config with overrides."""
    prod = get_ensemble_filter_config()
    cfg_dict = {f.name: getattr(prod, f.name) for f in fields(BallFilterConfig)}
    cfg_dict.update(overrides)
    return BallFilterConfig(**cfg_dict)


def _config_from_off(**overrides: object) -> BallFilterConfig:
    """Clone no-filter config with overrides."""
    off = _no_filter_config()
    cfg_dict = {f.name: getattr(off, f.name) for f in fields(BallFilterConfig)}
    cfg_dict.update(overrides)
    return BallFilterConfig(**cfg_dict)


def build_ablation_configs() -> list[tuple[str, BallFilterConfig]]:
    """Build all ablation configs: production, off, only-X, all-except-X."""
    configs: list[tuple[str, BallFilterConfig]] = []

    # Reference configs
    configs.append(("production", get_ensemble_filter_config()))
    configs.append(("no_filter", _no_filter_config()))

    # Only one stage enabled (from no-filter base + source-aware for fair comparison)
    only_stages: list[tuple[str, dict]] = [
        ("only_segment_pruning", {
            "enable_segment_pruning": True,
            "segment_jump_threshold": 0.20,
            "min_segment_frames": 10,
            "min_output_confidence": 0.05,
            "ensemble_source_aware": True,
        }),
        ("only_oscillation", {
            "enable_oscillation_pruning": True,
            "ensemble_source_aware": True,
        }),
        ("only_exit_ghost", {
            "enable_exit_ghost_removal": True,
            "ensemble_source_aware": True,
        }),
        ("only_outlier", {
            "enable_outlier_removal": True,
            "ensemble_source_aware": True,
        }),
        ("only_blip", {
            "enable_blip_removal": True,
            "blip_max_deviation": 0.10,
            "ensemble_source_aware": True,
        }),
        ("only_interpolation", {
            "enable_interpolation": True,
            "max_interpolation_gap": 10,
        }),
        ("only_stationarity", {
            "enable_stationarity_filter": True,
        }),
        ("only_motion_energy", {
            "enable_motion_energy_filter": True,
            "motion_energy_threshold": 0.02,
        }),
    ]

    for name, overrides in only_stages:
        configs.append((name, _config_from_off(**overrides)))

    # All except one stage (from production base)
    except_stages: list[tuple[str, dict]] = [
        ("all_except_segment_pruning", {"enable_segment_pruning": False}),
        ("all_except_oscillation", {"enable_oscillation_pruning": False}),
        ("all_except_exit_ghost", {"enable_exit_ghost_removal": False}),
        ("all_except_outlier", {"enable_outlier_removal": False}),
        ("all_except_blip", {"enable_blip_removal": False}),
        ("all_except_interpolation", {"enable_interpolation": False}),
        ("all_except_stationarity", {"enable_stationarity_filter": False}),
        ("all_except_source_aware", {"ensemble_source_aware": False}),
    ]

    for name, overrides in except_stages:
        configs.append((name, _config_from_production(**overrides)))

    # Combined removals
    configs.append(("no_outlier_no_blip", _config_from_production(
        enable_outlier_removal=False,
        enable_blip_removal=False,
    )))
    configs.append(("no_oscillation_no_blip", _config_from_production(
        enable_oscillation_pruning=False,
        enable_blip_removal=False,
    )))
    configs.append(("minimal", _config_from_off(
        enable_segment_pruning=True,
        segment_jump_threshold=0.20,
        min_segment_frames=10,
        min_output_confidence=0.05,
        enable_interpolation=True,
        max_interpolation_gap=10,
    )))
    configs.append(("minimal_sa", _config_from_off(
        enable_segment_pruning=True,
        segment_jump_threshold=0.20,
        min_segment_frames=10,
        min_output_confidence=0.05,
        enable_interpolation=True,
        max_interpolation_gap=10,
        ensemble_source_aware=True,
    )))

    return configs


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    name: str
    match_rate: float
    detection_rate: float
    mean_error_px: float
    median_error_px: float
    error_under_20px_rate: float
    per_rally: list[dict]


def run_ablation(
    rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]],
) -> list[AblationResult]:
    """Run ablation study across all configs."""
    configs = build_ablation_configs()
    results: list[AblationResult] = []

    console.print(f"\n[bold]== Ablation Study ({len(configs)} configs) ==[/bold]\n")

    # Summary table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Config", min_width=30)
    table.add_column("Det%", justify="right")
    table.add_column("Match%", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("<20px", justify="right")
    table.add_column("Δ Match", justify="right")
    table.add_column("Δ Error", justify="right")

    prod_match = None
    prod_error = None

    for name, config in configs:
        result = evaluate_ball_config(config, rallies)
        m = result.aggregate_metrics

        if prod_match is None:
            prod_match = m.match_rate
            prod_error = m.mean_error_px

        dm = m.match_rate - prod_match
        de = m.mean_error_px - prod_error

        # Color coding
        def _color(val: float, positive_good: bool = True) -> str:
            threshold = 0.005 if abs(val) < 1 else 0.5
            if abs(val) < threshold:
                return f"{val:+.1%}" if abs(val) < 1 else f"{val:+.1f}"
            good = val > 0 if positive_good else val < 0
            s = f"{val:+.1%}" if abs(val) < 1 else f"{val:+.1f}"
            return f"[green]{s}[/green]" if good else f"[red]{s}[/red]"

        match_str = f"[green]{m.match_rate:.1%}[/green]" if dm > 0.005 else (
            f"[red]{m.match_rate:.1%}[/red]" if dm < -0.005 else f"{m.match_rate:.1%}"
        )

        table.add_row(
            name,
            f"{m.detection_rate:.1%}",
            match_str,
            f"{m.mean_error_px:.1f}px",
            f"{m.error_under_20px_rate:.1%}",
            _color(dm, positive_good=True),
            _color(de, positive_good=False),
        )

        per_rally = []
        for rid, rm in result.per_rally_metrics:
            per_rally.append({
                "rally_id": rid[:8],
                "detection_rate": round(rm.detection_rate, 4),
                "match_rate": round(rm.match_rate, 4),
                "mean_error_px": round(rm.mean_error_px, 1),
            })

        results.append(AblationResult(
            name=name,
            match_rate=m.match_rate,
            detection_rate=m.detection_rate,
            mean_error_px=m.mean_error_px,
            median_error_px=m.median_error_px,
            error_under_20px_rate=m.error_under_20px_rate,
            per_rally=per_rally,
        ))

    console.print(table)

    # Per-rally detail table
    console.print("\n[bold]Per-Rally Match Rate Matrix[/bold]\n")

    rally_ids = [r["rally_id"] for r in results[0].per_rally]
    detail = Table(show_header=True, header_style="bold")
    detail.add_column("Config", min_width=30)
    for rid in rally_ids:
        detail.add_column(rid, justify="right")

    # Get production per-rally rates for delta coloring
    prod_rates = {r["rally_id"]: r["match_rate"] for r in results[0].per_rally}

    for res in results:
        row = [res.name]
        for rdata in res.per_rally:
            rid = rdata["rally_id"]
            rate = rdata["match_rate"]
            delta = rate - prod_rates[rid]
            if delta > 0.02:
                row.append(f"[green]{rate:.1%}[/green]")
            elif delta < -0.02:
                row.append(f"[red]{rate:.1%}[/red]")
            else:
                row.append(f"{rate:.1%}")
        detail.add_row(*row)

    console.print(detail)

    return results


# ---------------------------------------------------------------------------
# Incremental stage-by-stage diagnostic
# ---------------------------------------------------------------------------


def _run_incremental_stages(
    raw_positions: list[BallPosition],
    config: BallFilterConfig,
) -> dict[str, list[BallPosition]]:
    """Run pipeline stage-by-stage, returning positions after each cumulative stage."""
    filt = BallTemporalFilter(config)
    stages: dict[str, list[BallPosition]] = {}

    sorted_pos = sorted(raw_positions, key=lambda p: p.frame_number)
    current = list(sorted_pos)
    stages["raw"] = list(current)

    # Stage 1: Motion energy filter
    if config.enable_motion_energy_filter:
        current = filt._motion_energy_filter(current)
    stages["+motion_energy"] = list(current)

    # Stage 2: Stationarity filter
    if config.enable_stationarity_filter:
        current = filt._remove_stationary_runs(current)
    stages["+stationarity"] = list(current)

    # Stage 3: Detect exit ghost ranges (on current data)
    ghost_ranges: list[tuple[int, int]] = []
    if config.enable_exit_ghost_removal:
        ghost_ranges = filt._detect_exit_ghost_ranges(current)

    # Stage 4: Segment pruning
    if config.enable_segment_pruning:
        current = filt._prune_segments(current, ghost_ranges=ghost_ranges)
    stages["+segment_pruning"] = list(current)

    # Stage 5: Apply exit ghost removal
    if ghost_ranges:
        ghost_frames: set[int] = set()
        for start, end in ghost_ranges:
            for p in current:
                if start <= p.frame_number <= end:
                    ghost_frames.add(p.frame_number)
        if ghost_frames:
            current = [p for p in current if p.frame_number not in ghost_frames]
    stages["+exit_ghost"] = list(current)

    # Stage 6: Oscillation pruning
    if config.enable_oscillation_pruning:
        current = filt._prune_oscillating(current)
    stages["+oscillation"] = list(current)

    # Stage 7: Outlier removal
    if config.enable_outlier_removal:
        current = filt._remove_outliers(current)
    stages["+outlier"] = list(current)

    # Stage 8: Blip removal
    if config.enable_blip_removal:
        current = filt._remove_trajectory_blips(current)
    stages["+blip"] = list(current)

    # Stage 9: Re-prune
    before_reprune = len(current)
    if config.enable_oscillation_pruning:
        current = filt._prune_oscillating(current)
    if config.enable_segment_pruning:
        current = filt._prune_segments(current)
    reprune_removed = before_reprune - len(current)
    if reprune_removed == 0:
        # No change — skip re-prune label
        stages["+reprune"] = list(current)
    else:
        stages["+reprune"] = list(current)

    # Stage 10: Interpolation
    if config.enable_interpolation:
        current = filt._interpolate_missing(current)
    stages["+interpolation"] = list(current)

    return stages


def run_incremental(
    rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]],
) -> list[dict]:
    """Run incremental stage-by-stage diagnostics per rally."""
    config = get_ensemble_filter_config()
    results = []

    console.print(f"\n[bold]== Incremental Stage-by-Stage ({len(rallies)} rallies) ==[/bold]")

    for cached, gt_positions in rallies:
        rid = cached.rally_id[:8]
        console.print(f"\n[bold]Rally {rid}[/bold]")

        stages = _run_incremental_stages(cached.raw_ball_positions, config)

        rally_data = {"rally_id": rid, "stages": []}

        prev_match = None
        prev_error = None

        table = Table(show_header=True, header_style="bold")
        table.add_column("Stage", min_width=20)
        table.add_column("Count", justify="right")
        table.add_column("Det%", justify="right")
        table.add_column("Match%", justify="right")
        table.add_column("Error", justify="right")
        table.add_column("Δ Match", justify="right")
        table.add_column("Δ Error", justify="right")

        for stage_name, positions in stages.items():
            metrics = evaluate_ball_tracking(
                ground_truth=gt_positions,
                predictions=positions,
                video_width=cached.video_width,
                video_height=cached.video_height,
                video_fps=cached.video_fps,
            )

            dm = (metrics.match_rate - prev_match) if prev_match is not None else 0
            de = (metrics.mean_error_px - prev_error) if prev_error is not None else 0

            dm_str = "-"
            de_str = "-"
            if prev_match is not None:
                if dm > 0.005:
                    dm_str = f"[green]{dm:+.1%}[/green]"
                elif dm < -0.005:
                    dm_str = f"[red]{dm:+.1%}[/red]"
                else:
                    dm_str = f"{dm:+.1%}"

                if de < -0.5:
                    de_str = f"[green]{de:+.1f}px[/green]"
                elif de > 0.5:
                    de_str = f"[red]{de:+.1f}px[/red]"
                else:
                    de_str = f"{de:+.1f}px"

            table.add_row(
                stage_name,
                str(len(positions)),
                f"{metrics.detection_rate:.1%}",
                f"{metrics.match_rate:.1%}",
                f"{metrics.mean_error_px:.1f}px",
                dm_str,
                de_str,
            )

            rally_data["stages"].append({
                "stage": stage_name,
                "count": len(positions),
                "detection_rate": round(metrics.detection_rate, 4),
                "match_rate": round(metrics.match_rate, 4),
                "mean_error_px": round(metrics.mean_error_px, 1),
            })

            prev_match = metrics.match_rate
            prev_error = metrics.mean_error_px

        console.print(table)
        results.append(rally_data)

    return results


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


def run_sensitivity(
    rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]],
) -> list[dict]:
    """Run parameter sensitivity sweeps on key stages."""
    results = []

    console.print("\n[bold]== Parameter Sensitivity ==[/bold]")

    # Define sweeps: (param_name, values_to_test)
    sweeps: list[tuple[str, str, list]] = [
        # Segment pruning params
        ("segment_jump_threshold", "Segment jump threshold", [0.10, 0.15, 0.20, 0.25, 0.30]),
        ("min_segment_frames", "Min segment frames", [5, 8, 10, 12, 15, 20]),
        # Blip removal params
        ("blip_max_deviation", "Blip max deviation", [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]),
        # Interpolation gap
        ("max_interpolation_gap", "Max interpolation gap", [3, 5, 8, 10, 15, 20]),
        # Stationarity params
        ("stationarity_max_spread", "Stationarity max spread", [0.003, 0.005, 0.007, 0.01]),
        ("stationarity_min_frames", "Stationarity min frames", [6, 8, 10, 12, 15, 20]),
        # Oscillation params
        ("min_oscillation_frames", "Oscillation min frames", [6, 8, 10, 12, 15, 20]),
        ("oscillation_reversal_rate", "Oscillation reversal rate", [0.15, 0.20, 0.25, 0.30, 0.40]),
        # Outlier params
        ("max_trajectory_deviation", "Outlier max deviation", [0.04, 0.06, 0.08, 0.10, 0.15]),
        # Exit ghost params
        ("exit_edge_zone", "Exit edge zone", [0.05, 0.08, 0.10, 0.12, 0.15]),
        ("exit_approach_frames", "Exit approach frames", [2, 3, 4, 5]),
    ]

    for param_name, label, values in sweeps:
        console.print(f"\n[bold]{label} ({param_name})[/bold]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Det%", justify="right")
        table.add_column("Match%", justify="right")
        table.add_column("Error", justify="right")
        table.add_column("<20px", justify="right")

        # Get production value for highlighting
        prod = get_ensemble_filter_config()
        prod_val = getattr(prod, param_name)

        sweep_data = {"param": param_name, "label": label, "results": []}
        best_match = 0.0
        best_val = None

        for val in values:
            config = _config_from_production(**{param_name: val})
            result = evaluate_ball_config(config, rallies)
            m = result.aggregate_metrics

            marker = " *" if val == prod_val else ""
            val_str = f"{val}{marker}"

            if m.match_rate > best_match:
                best_match = m.match_rate
                best_val = val

            table.add_row(
                val_str,
                f"{m.detection_rate:.1%}",
                f"{m.match_rate:.1%}",
                f"{m.mean_error_px:.1f}px",
                f"{m.error_under_20px_rate:.1%}",
            )

            sweep_data["results"].append({
                "value": val,
                "detection_rate": round(m.detection_rate, 4),
                "match_rate": round(m.match_rate, 4),
                "mean_error_px": round(m.mean_error_px, 1),
            })

        console.print(table)
        if best_val != prod_val:
            console.print(
                f"  Best: {param_name}={best_val} ({best_match:.1%} match) "
                f"vs production {prod_val} — [yellow]consider updating[/yellow]"
            )
        else:
            console.print(f"  Production value ({prod_val}) is optimal")

        sweep_data["best_value"] = best_val
        sweep_data["production_value"] = prod_val
        results.append(sweep_data)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_rallies(
    rally_prefix: str | None = None,
) -> list[tuple[CachedBallData, list[GroundTruthPosition]]]:
    """Load cached ensemble positions paired with GT."""
    cache = BallRawCache(cache_dir=ENSEMBLE_CACHE_DIR)
    cached_ids = cache.list_cached()

    if not cached_ids:
        console.print("[red]No cached ensemble positions found.[/red]")
        console.print("Run cache_ensemble_positions.py first:")
        console.print("  uv run python scripts/cache_ensemble_positions.py")
        sys.exit(1)

    console.print(f"Found {len(cached_ids)} cached ensemble rallies")

    rallies_db = load_labeled_rallies()
    if not rallies_db:
        console.print("[red]No rallies with ball GT found[/red]")
        sys.exit(1)

    if rally_prefix:
        rallies_db = [r for r in rallies_db if r.rally_id.startswith(rally_prefix)]

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

    console.print(f"Evaluating {len(rallies)} rallies with ball GT\n")
    return rallies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive ball filter audit"
    )
    parser.add_argument("--ablation", action="store_true", help="Run ablation only")
    parser.add_argument("--incremental", action="store_true", help="Run incremental only")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity only")
    parser.add_argument("--rally", help="Evaluate specific rally ID (prefix match)")
    parser.add_argument("-o", "--output", help="Save results to JSON file")
    args = parser.parse_args()

    run_all = not (args.ablation or args.incremental or args.sensitivity)

    rallies = load_rallies(args.rally)

    output: dict = {}

    if run_all or args.ablation:
        ablation_results = run_ablation(rallies)
        output["ablation"] = [
            {
                "name": r.name,
                "match_rate": round(r.match_rate, 4),
                "detection_rate": round(r.detection_rate, 4),
                "mean_error_px": round(r.mean_error_px, 1),
                "per_rally": r.per_rally,
            }
            for r in ablation_results
        ]

    if run_all or args.incremental:
        incremental_results = run_incremental(rallies)
        output["incremental"] = incremental_results

    if run_all or args.sensitivity:
        sensitivity_results = run_sensitivity(rallies)
        output["sensitivity"] = sensitivity_results

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        console.print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
