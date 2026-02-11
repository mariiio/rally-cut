"""Evaluate player tracking against ground truth from database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from rallycut.cli.utils import handle_errors

if TYPE_CHECKING:
    from rallycut.cli.commands.compare_tracking import BallMetrics, MOTMetrics
    from rallycut.evaluation.tracking.ball_metrics import BallTrackingMetrics
    from rallycut.evaluation.tracking.error_analysis import ErrorEvent
    from rallycut.evaluation.tracking.metrics import PerPlayerMetrics

app = typer.Typer(help="Evaluate player tracking against ground truth")
console = Console()


def _status_icon(value: float, target: float, higher_better: bool = True) -> str:
    """Return status icon based on whether value meets target."""
    if higher_better:
        return "[green]OK[/green]" if value >= target else "[yellow]Below[/yellow]"
    else:
        return "[green]OK[/green]" if value <= target else "[yellow]Above[/yellow]"


@app.callback(invoke_without_command=True)
@handle_errors
def evaluate_tracking(
    ctx: typer.Context,
    rally_id: str = typer.Option(
        None,
        "--rally-id", "-r",
        help="Evaluate specific rally by ID",
    ),
    video_id: str = typer.Option(
        None,
        "--video-id", "-v",
        help="Evaluate all labeled rallies in a video",
    ),
    all_rallies: bool = typer.Option(
        False,
        "--all", "-a",
        help="Evaluate all labeled rallies in database",
    ),
    per_player: bool = typer.Option(
        False,
        "--per-player", "-p",
        help="Show per-player breakdown",
    ),
    analyze_errors: bool = typer.Option(
        False,
        "--analyze-errors", "-e",
        help="Show detailed error analysis",
    ),
    ball_only: bool = typer.Option(
        False,
        "--ball-only", "-b",
        help="Evaluate ball tracking only (skip player metrics)",
    ),
    iou_threshold: float = typer.Option(
        0.5,
        "--iou", "-i",
        help="Minimum IoU for matching predictions to ground truth",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output metrics to JSON file",
    ),
) -> None:
    """Evaluate player and ball tracking predictions against ground truth.

    Loads ground truth and predictions from the database and computes
    MOT (Multi-Object Tracking) metrics including MOTA, precision, recall,
    F1, and ID switches for players. For ball tracking, computes detection
    rate, position error statistics, and accuracy buckets.

    Examples:

        # Evaluate specific rally (players + ball)
        rallycut evaluate-tracking --rally-id abc123

        # Evaluate all labeled rallies in a video
        rallycut evaluate-tracking --video-id a7ee3d38-...

        # Evaluate all labeled data
        rallycut evaluate-tracking --all

        # Ball tracking evaluation only
        rallycut evaluate-tracking --all --ball-only

        # Show per-player breakdown
        rallycut evaluate-tracking --rally-id abc123 --per-player

        # Show detailed error analysis
        rallycut evaluate-tracking --rally-id abc123 --analyze-errors

        # Export to JSON
        rallycut evaluate-tracking --rally-id abc123 -o metrics.json

        # Grid search for optimal filter parameters
        rallycut evaluate-tracking tune-filter --all --grid quick
    """
    # If a subcommand was invoked, don't run the default
    if ctx.invoked_subcommand is not None:
        return

    from rallycut.evaluation.tracking.ball_metrics import (
        aggregate_ball_metrics,
        evaluate_ball_tracking,
    )
    from rallycut.evaluation.tracking.db import load_labeled_rallies
    from rallycut.evaluation.tracking.error_analysis import analyze_errors as get_errors
    from rallycut.evaluation.tracking.metrics import aggregate_results, evaluate_rally

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print(
            "[red]Error:[/red] Specify --rally-id, --video-id, or --all"
        )
        raise typer.Exit(1)

    # Load rallies from database
    console.print("[bold]Loading labeled rallies from database...[/bold]")

    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
        ball_gt_only=ball_only,  # Filter to validated ball GT for ball-only mode
    )

    if not rallies:
        console.print("[yellow]No rallies with ground truth found[/yellow]")
        if rally_id:
            console.print(f"  Rally ID: {rally_id}")
        if video_id:
            console.print(f"  Video ID: {video_id}")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies)} rally(s) with ground truth\n")

    # Ball-only evaluation mode
    if ball_only:
        ball_results = []
        for rally in rallies:
            if rally.predictions is None or not rally.predictions.ball_positions:
                console.print(
                    f"[yellow]Rally {rally.rally_id[:8]}... has no ball predictions, skipping[/yellow]"
                )
                continue

            ball_metrics = evaluate_ball_tracking(
                ground_truth=rally.ground_truth.positions,
                predictions=rally.predictions.ball_positions,
                video_width=rally.video_width,
                video_height=rally.video_height,
                video_fps=rally.video_fps,
            )
            ball_results.append((rally.rally_id, ball_metrics))

        if not ball_results:
            console.print("[red]No rallies with ball tracking data found[/red]")
            raise typer.Exit(1)

        # Display ball-only results
        if len(ball_results) == 1:
            rally_id_str, ball_metrics = ball_results[0]
            console.print(f"[bold]Ball Tracking Evaluation - Rally {rally_id_str[:8]}...[/bold]")
            console.print("=" * 50)
            _display_enhanced_ball_metrics(ball_metrics)
        else:
            console.print(f"[bold]Ball Tracking Evaluation - {len(ball_results)} Rallies[/bold]")
            console.print("=" * 50)

            # Per-rally table
            ball_table = Table(show_header=True, header_style="bold")
            ball_table.add_column("Rally")
            ball_table.add_column("Detection", justify="right")
            ball_table.add_column("Match", justify="right")
            ball_table.add_column("Mean Err", justify="right")
            ball_table.add_column("Median", justify="right")
            ball_table.add_column("P90", justify="right")
            ball_table.add_column("<20px", justify="right")

            for rally_id_str, metrics in ball_results:
                ball_table.add_row(
                    rally_id_str[:8] + "...",
                    f"{metrics.detection_rate:.1%}",
                    f"{metrics.match_rate:.1%}",
                    f"{metrics.mean_error_px:.1f}px",
                    f"{metrics.median_error_px:.1f}px",
                    f"{metrics.p90_error_px:.1f}px",
                    f"{metrics.error_under_20px_rate:.1%}",
                )

            console.print(ball_table)

            # Aggregate metrics
            console.print("\n[bold]Aggregate Metrics[/bold]")
            ball_combined = aggregate_ball_metrics([m for _, m in ball_results])
            _display_enhanced_ball_metrics(ball_combined)

        # Save to file if requested
        if output:
            if len(ball_results) == 1:
                output_data = ball_results[0][1].to_dict()
            else:
                ball_combined = aggregate_ball_metrics([m for _, m in ball_results])
                output_data = {
                    "rallies": [
                        {"rallyId": rid, **m.to_dict()}
                        for rid, m in ball_results
                    ],
                    "aggregate": ball_combined.to_dict(),
                }

            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"\n[green]Metrics saved to {output}[/green]")

        return

    # Evaluate each rally (player + ball combined)
    results = []
    all_errors_list = []

    for rally in rallies:
        if rally.predictions is None:
            console.print(
                f"[yellow]Rally {rally.rally_id[:8]}... has no predictions, skipping[/yellow]"
            )
            continue

        result = evaluate_rally(
            rally_id=rally.rally_id,
            ground_truth=rally.ground_truth,
            predictions=rally.predictions,
            iou_threshold=iou_threshold,
            video_width=rally.video_width,
            video_height=rally.video_height,
        )
        results.append(result)

        if analyze_errors:
            errors = get_errors(
                rally.ground_truth,
                rally.predictions,
                iou_threshold,
            )
            all_errors_list.extend(errors)

    if not results:
        console.print("[red]No rallies with predictions found[/red]")
        raise typer.Exit(1)

    # Display results
    if len(results) == 1:
        # Single rally - show detailed output
        result = results[0]

        console.print(f"[bold]Player Tracking Evaluation - Rally {result.rally_id[:8]}...[/bold]")
        console.print("=" * 50)

        _display_aggregate_metrics(result.aggregate)

        if result.ball_metrics:
            _display_ball_metrics(result.ball_metrics)

        if per_player and result.per_player:
            _display_per_player_metrics(result.per_player)

        # Error frames summary
        error_count = len(result.error_frames)
        total_frames = len(result.per_frame)
        if error_count > 0:
            console.print(f"\n[bold]Error Frames:[/bold] {error_count} of {total_frames} frames")
            if error_count <= 10:
                console.print(f"  Frames: {', '.join(str(f) for f in result.error_frames)}")
            else:
                first_five = result.error_frames[:5]
                last_five = result.error_frames[-5:]
                console.print(
                    f"  First 5: {', '.join(str(f) for f in first_five)}"
                )
                console.print(
                    f"  Last 5: {', '.join(str(f) for f in last_five)}"
                )
        else:
            console.print("\n[green]No error frames![/green]")

        if analyze_errors and all_errors_list:
            _display_error_analysis(all_errors_list)

    else:
        # Multiple rallies - show summary
        console.print(f"[bold]Player Tracking Evaluation - {len(results)} Rallies[/bold]")
        console.print("=" * 50)

        # Per-rally table
        rally_table = Table(show_header=True, header_style="bold")
        rally_table.add_column("Rally")
        rally_table.add_column("MOTA", justify="right")
        rally_table.add_column("Precision", justify="right")
        rally_table.add_column("Recall", justify="right")
        rally_table.add_column("F1", justify="right")
        rally_table.add_column("ID Sw", justify="right")
        rally_table.add_column("Errors", justify="right")

        for result in results:
            error_count = len(result.error_frames)
            rally_table.add_row(
                result.rally_id[:8] + "...",
                f"{result.aggregate.mota:.1%}",
                f"{result.aggregate.precision:.1%}",
                f"{result.aggregate.recall:.1%}",
                f"{result.aggregate.f1:.1%}",
                str(result.aggregate.num_id_switches),
                str(error_count),
            )

        console.print(rally_table)

        # Aggregate metrics
        console.print("\n[bold]Aggregate Metrics[/bold]")
        combined = aggregate_results(results)
        _display_aggregate_metrics(combined)

        if analyze_errors and all_errors_list:
            _display_error_analysis(all_errors_list)

    # Save to file if requested
    if output:
        if len(results) == 1:
            output_data = results[0].to_dict()
        else:
            combined = aggregate_results(results)
            output_data = {
                "rallies": [r.to_dict() for r in results],
                "aggregate": {
                    "mota": combined.mota,
                    "precision": combined.precision,
                    "recall": combined.recall,
                    "f1": combined.f1,
                    "idSwitches": combined.num_id_switches,
                },
            }

        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Metrics saved to {output}[/green]")


@app.command(name="tune-filter")
@handle_errors
def tune_filter(
    rally_id: Annotated[
        str | None,
        typer.Option(
            "--rally-id", "-r",
            help="Tune on specific rally by ID",
        ),
    ] = None,
    video_id: Annotated[
        str | None,
        typer.Option(
            "--video-id", "-v",
            help="Tune on all labeled rallies in a video",
        ),
    ] = None,
    all_rallies: Annotated[
        bool,
        typer.Option(
            "--all", "-a",
            help="Tune on all labeled rallies in database",
        ),
    ] = False,
    grid: Annotated[
        str,
        typer.Option(
            "--grid", "-g",
            help="Grid to search: quick, full, referee, stability, merge",
        ),
    ] = "quick",
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou", "-i",
            help="IoU threshold for matching",
        ),
    ] = 0.5,
    min_rally_f1: Annotated[
        float | None,
        typer.Option(
            "--min-rally-f1",
            help="Reject configs where any rally drops below this F1",
        ),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option(
            "--top", "-n",
            help="Number of top results to show",
        ),
    ] = 10,
    cache_only: Annotated[
        bool,
        typer.Option(
            "--cache-only",
            help="Only cache raw positions, don't run grid search",
        ),
    ] = False,
    clear_cache: Annotated[
        bool,
        typer.Option(
            "--clear-cache",
            help="Clear raw position cache before starting",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Export full results to JSON file",
        ),
    ] = None,
) -> None:
    """Grid search for optimal PlayerFilterConfig parameters.

    Searches over filter parameters to find the configuration that
    maximizes F1 score while minimizing ID switches.

    The key insight: YOLO+ByteTrack is slow (~seconds per rally), but
    the filter pipeline is fast (~milliseconds). By caching raw positions,
    we can re-run filtering with different configs without re-running detection.

    Examples:

        # Cache raw positions first (slow, one-time)
        rallycut evaluate-tracking tune-filter --all --cache-only

        # Quick grid search (36 combinations)
        rallycut evaluate-tracking tune-filter --all --grid quick

        # Full search with constraint
        rallycut evaluate-tracking tune-filter --all --grid full --min-rally-f1 0.70

        # Export results
        rallycut evaluate-tracking tune-filter --all -o results.json
    """
    from rallycut.evaluation.tracking.db import load_labeled_rallies
    from rallycut.evaluation.tracking.grid_search import grid_search
    from rallycut.evaluation.tracking.param_grid import (
        AVAILABLE_GRIDS,
        describe_config_diff,
        get_grid,
        grid_size,
    )
    from rallycut.evaluation.tracking.raw_cache import CachedRallyData, RawPositionCache

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print(
            "[red]Error:[/red] Specify --rally-id, --video-id, or --all"
        )
        raise typer.Exit(1)

    if grid not in AVAILABLE_GRIDS:
        console.print(
            f"[red]Error:[/red] Unknown grid '{grid}'. "
            f"Available: {', '.join(AVAILABLE_GRIDS.keys())}"
        )
        raise typer.Exit(1)

    # Initialize cache
    raw_cache = RawPositionCache()

    if clear_cache:
        count = raw_cache.clear()
        console.print(f"[yellow]Cleared {count} cached raw position files[/yellow]")

    # Load rallies from database
    console.print("[bold]Loading labeled rallies from database...[/bold]")
    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
    )

    if not rallies:
        console.print("[yellow]No rallies with ground truth found[/yellow]")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies)} rally(s) with ground truth\n")

    from rallycut.labeling.ground_truth import GroundTruthResult

    # Check for raw positions in database and build cache
    cached_rallies: list[tuple[CachedRallyData, GroundTruthResult]] = []
    rallies_without_raw: list = []

    for rally in rallies:
        if rally.predictions is None:
            console.print(
                f"[yellow]Rally {rally.rally_id[:8]}... has no predictions, skipping[/yellow]"
            )
            continue

        # First check file cache
        cached = raw_cache.get(rally.rally_id)
        if cached:
            cached_rallies.append((cached, rally.ground_truth))
            continue

        # Check if raw positions are in database
        if rally.raw_positions:
            # Use raw positions from database
            cached_data = CachedRallyData(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                raw_positions=rally.raw_positions,
                ball_positions=rally.predictions.ball_positions or [],
                video_fps=rally.video_fps,
                frame_count=rally.predictions.frame_count,
                video_width=rally.video_width,
                video_height=rally.video_height,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
            )
            raw_cache.put(cached_data)
            cached_rallies.append((cached_data, rally.ground_truth))
        else:
            # No raw positions - need to re-run tracking
            rallies_without_raw.append(rally)

    # Report status
    if rallies_without_raw:
        console.print(
            f"\n[yellow]Warning: {len(rallies_without_raw)} rally(s) have no raw positions.[/yellow]"
        )
        console.print(
            "[dim]Re-run player tracking to store raw positions for parameter tuning.[/dim]"
        )
        for rally in rallies_without_raw:
            console.print(f"  - {rally.rally_id[:8]}...")

    console.print(f"\n  Rallies with raw positions: {len(cached_rallies)}")

    if cache_only:
        stats = raw_cache.stats()
        console.print("\n[green]Raw positions cached successfully![/green]")
        console.print(f"  Cache location: {stats['cache_dir']}")
        console.print(f"  Cached rallies: {stats['count']}")
        console.print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        return

    if not cached_rallies:
        console.print("[red]No rallies available for grid search[/red]")
        raise typer.Exit(1)

    # Run grid search
    param_grid = get_grid(grid)
    num_configs = grid_size(param_grid)

    console.print()
    console.print(f"[bold]Player Filter Grid Search - {num_configs} configs, {len(cached_rallies)} rallies[/bold]")
    console.print("=" * 60)
    console.print(f"Grid: {grid}")
    console.print(f"IoU threshold: {iou_threshold}")
    if min_rally_f1 is not None:
        console.print(f"Min rally F1 constraint: [yellow]{min_rally_f1:.0%}[/yellow]")

    # Show parameters being searched
    console.print("\n[dim]Parameters being searched:[/dim]")
    for param, values in param_grid.items():
        console.print(f"  {param}: {values}")

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=num_configs)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, completed=current)

        result = grid_search(
            rallies=cached_rallies,
            param_grid=param_grid,
            iou_threshold=iou_threshold,
            min_rally_f1=min_rally_f1,
            progress_callback=update_progress,
        )

    # Display results
    console.print()
    console.print("[bold]Best Configuration[/bold]")
    console.print("=" * 60)

    best = result.best_config
    for param in param_grid.keys():
        value = getattr(best, param)
        default_val = getattr(type(best)(), param)
        if value != default_val:
            console.print(f"  [cyan]{param}[/cyan]: {value} [dim](default: {default_val})[/dim]")
        else:
            console.print(f"  [dim]{param}: {value}[/dim]")

    console.print()
    console.print(
        f"Results: F1=[bold green]{result.best_f1:.1%}[/bold green], "
        f"MOTA={result.best_mota:.1%}, "
        f"ID Switches={result.best_id_switches}"
    )

    if result.improvement_f1 != 0:
        improvement_color = "green" if result.improvement_f1 > 0 else "red"
        console.print(
            f"Improvement over default: [{improvement_color}]{result.improvement_f1:+.1%}[/{improvement_color}] "
            f"(default F1={result.default_f1:.1%})"
        )

    if result.rejected_count > 0:
        console.print(
            f"\n[yellow]Rejected {result.rejected_count} configs that violated constraints[/yellow]"
        )

    # Top N configurations
    console.print()
    console.print(f"[bold]Top {min(top_n, len(result.all_results))} Configurations[/bold]")

    top_table = Table(show_header=True, header_style="bold")
    top_table.add_column("Rank", justify="right")
    top_table.add_column("F1", justify="right")
    top_table.add_column("MOTA", justify="right")
    top_table.add_column("ID Sw", justify="right")
    top_table.add_column("Changes from Default")

    for i, config_result in enumerate(result.all_results[:top_n]):
        if config_result.rejected:
            continue

        rank = i + 1
        metrics = config_result.aggregate_metrics
        diff = describe_config_diff(config_result.config)

        # Truncate diff if too long
        if len(diff) > 50:
            diff = diff[:47] + "..."

        f1_style = "green" if metrics.f1 >= 0.80 else ("yellow" if metrics.f1 >= 0.60 else "red")

        top_table.add_row(
            str(rank),
            f"[{f1_style}]{metrics.f1:.1%}[/{f1_style}]",
            f"{metrics.mota:.1%}",
            str(metrics.num_id_switches),
            diff,
        )

    console.print(top_table)

    # Export to file if requested
    if output:
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[green]Full results exported to {output}[/green]")


def _display_aggregate_metrics(metrics: MOTMetrics) -> None:
    """Display aggregate metrics table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")

    table.add_row(
        "MOTA",
        f"{metrics.mota:.2%}",
        ">80%",
        _status_icon(metrics.mota, 0.80),
    )
    table.add_row(
        "Precision",
        f"{metrics.precision:.2%}",
        ">85%",
        _status_icon(metrics.precision, 0.85),
    )
    table.add_row(
        "Recall",
        f"{metrics.recall:.2%}",
        ">80%",
        _status_icon(metrics.recall, 0.80),
    )
    table.add_row(
        "F1",
        f"{metrics.f1:.2%}",
        ">80%",
        _status_icon(metrics.f1, 0.80),
    )
    table.add_row(
        "ID Switches",
        str(metrics.num_id_switches),
        "<5",
        _status_icon(metrics.num_id_switches, 5, higher_better=False),
    )

    console.print("\n[bold]Aggregate Metrics[/bold]")
    console.print(table)

    # Detection breakdown
    console.print("\n[dim]Detection breakdown:[/dim]")
    console.print(f"  Ground truth: {metrics.num_gt} objects")
    console.print(f"  Predictions: {metrics.num_pred} objects")
    console.print(f"  Matches (TP): {metrics.num_matches}")
    console.print(f"  Misses (FN): {metrics.num_misses}")
    console.print(f"  False positives (FP): {metrics.num_false_positives}")


def _display_ball_metrics(ball_metrics: BallMetrics) -> None:
    """Display ball tracking metrics (compact version for combined eval)."""
    from rallycut.evaluation.tracking.ball_metrics import BallTrackingMetrics

    # Handle legacy BallMetrics from compare_tracking
    if not isinstance(ball_metrics, BallTrackingMetrics):
        # Legacy format - just show basic metrics
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status")

        table.add_row(
            "Detection Rate",
            f"{ball_metrics.detection_rate:.2%}",
            ">60%",
            _status_icon(ball_metrics.detection_rate, 0.60),
        )
        table.add_row(
            "Mean Error",
            f"{ball_metrics.mean_error_px:.1f} px",
            "<20px",
            _status_icon(ball_metrics.mean_error_px, 20, higher_better=False),
        )

        console.print("\n[bold]Ball Tracking Metrics[/bold]")
        console.print(table)
        return

    # Enhanced metrics
    _display_enhanced_ball_metrics(ball_metrics)


def _display_enhanced_ball_metrics(metrics: BallTrackingMetrics) -> None:
    """Display enhanced ball tracking metrics with detailed statistics."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")

    # Detection metrics
    table.add_row(
        "Detection Rate",
        f"{metrics.detection_rate:.1%}",
        ">60%",
        _status_icon(metrics.detection_rate, 0.60),
    )
    table.add_row(
        "Match Rate (<50px)",
        f"{metrics.match_rate:.1%}",
        ">50%",
        _status_icon(metrics.match_rate, 0.50),
    )

    # Error metrics
    table.add_row(
        "Mean Error",
        f"{metrics.mean_error_px:.1f} px",
        "<20px",
        _status_icon(metrics.mean_error_px, 20, higher_better=False),
    )
    table.add_row(
        "Median Error",
        f"{metrics.median_error_px:.1f} px",
        "<15px",
        _status_icon(metrics.median_error_px, 15, higher_better=False),
    )
    table.add_row(
        "P90 Error",
        f"{metrics.p90_error_px:.1f} px",
        "<50px",
        _status_icon(metrics.p90_error_px, 50, higher_better=False),
    )
    table.add_row(
        "Max Error",
        f"{metrics.max_error_px:.1f} px",
        "-",
        "",
    )

    # Accuracy buckets
    table.add_row(
        "Accuracy <20px",
        f"{metrics.error_under_20px_rate:.1%}",
        ">60%",
        _status_icon(metrics.error_under_20px_rate, 0.60),
    )
    table.add_row(
        "Accuracy <50px",
        f"{metrics.error_under_50px_rate:.1%}",
        ">80%",
        _status_icon(metrics.error_under_50px_rate, 0.80),
    )

    console.print("\n[bold]Ball Tracking Metrics[/bold]")
    console.print(table)

    # Detection breakdown
    console.print("\n[dim]Detection breakdown:[/dim]")
    console.print(f"  Ground truth frames: {metrics.num_gt_frames}")
    console.print(f"  Detected frames: {metrics.num_detected}")
    console.print(f"  Matched frames (<50px): {metrics.num_matched}")
    miss_count = metrics.num_gt_frames - metrics.num_detected
    console.print(f"  Missed frames: {miss_count}")


def _display_per_player_metrics(per_player: list[PerPlayerMetrics]) -> None:
    """Display per-player metrics table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Player")
    table.add_column("Recall", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ID Swaps", justify="right")
    table.add_column("GT", justify="right")
    table.add_column("Matches", justify="right")

    # Find worst player for highlighting
    worst = min(per_player, key=lambda p: p.f1) if per_player else None

    for player in per_player:
        is_worst = player == worst
        suffix = " [yellow]<- worst[/yellow]" if is_worst else ""

        table.add_row(
            player.label + suffix,
            f"{player.recall:.1%}",
            f"{player.precision:.1%}",
            f"{player.f1:.1%}",
            str(player.id_switches),
            str(player.gt_count),
            str(player.matches),
        )

    console.print("\n[bold]Per-Player Breakdown[/bold]")
    console.print(table)


def _display_error_analysis(errors: list[ErrorEvent]) -> None:
    """Display error analysis summary."""
    from rallycut.evaluation.tracking.error_analysis import ErrorType, summarize_errors

    summary = summarize_errors(errors)

    console.print("\n[bold]Error Analysis[/bold]")
    console.print(f"  Total errors: {summary.total_errors}")

    # By type
    console.print("\n  [dim]By type:[/dim]")
    for error_type in ErrorType:
        count = summary.by_type.get(error_type, 0)
        if count > 0:
            console.print(f"    {error_type.value}: {count}")

    # By player (top 5)
    if summary.by_player:
        console.print("\n  [dim]By player (most errors):[/dim]")
        sorted_players = sorted(
            summary.by_player.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        for label, count in sorted_players:
            console.print(f"    {label}: {count}")

    # Consecutive error frames
    if summary.consecutive_error_frames > 1:
        console.print(
            f"\n  [yellow]Max consecutive error frames: "
            f"{summary.consecutive_error_frames}[/yellow]"
        )


@app.command(name="tune-ball-filter")
@handle_errors
def tune_ball_filter(
    rally_id: Annotated[
        str | None,
        typer.Option(
            "--rally-id", "-r",
            help="Tune on specific rally by ID",
        ),
    ] = None,
    video_id: Annotated[
        str | None,
        typer.Option(
            "--video-id", "-v",
            help="Tune on all labeled rallies in a video",
        ),
    ] = None,
    all_rallies: Annotated[
        bool,
        typer.Option(
            "--all", "-a",
            help="Tune on all labeled rallies in database",
        ),
    ] = False,
    grid: Annotated[
        str,
        typer.Option(
            "--grid", "-g",
            help="Grid to search: quick, lag, full, confidence",
        ),
    ] = "quick",
    match_threshold: Annotated[
        float,
        typer.Option(
            "--match-threshold", "-t",
            help="Max distance (pixels) for a match",
        ),
    ] = 50.0,
    min_rally_detection: Annotated[
        float | None,
        typer.Option(
            "--min-rally-detection",
            help="Reject configs where any rally drops below this detection rate",
        ),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option(
            "--top", "-n",
            help="Number of top results to show",
        ),
    ] = 10,
    cache_only: Annotated[
        bool,
        typer.Option(
            "--cache-only",
            help="Only cache raw positions, don't run grid search",
        ),
    ] = False,
    clear_cache: Annotated[
        bool,
        typer.Option(
            "--clear-cache",
            help="Clear raw position cache before starting",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Export full results to JSON file",
        ),
    ] = None,
) -> None:
    """Grid search for optimal BallFilterConfig parameters.

    Searches over Kalman filter parameters to find the configuration that
    maximizes match rate while minimizing position error.

    Similar to tune-filter, this caches raw (unfiltered) ball positions
    and re-runs the Kalman filter with different configs.

    Examples:

        # Cache raw ball positions first (requires raw_positions in predictions)
        rallycut evaluate-tracking tune-ball-filter --all --cache-only

        # Quick grid search (81 combinations)
        rallycut evaluate-tracking tune-ball-filter --all --grid quick

        # Test lag compensation settings
        rallycut evaluate-tracking tune-ball-filter --all --grid lag

        # Full search
        rallycut evaluate-tracking tune-ball-filter --all --grid full

        # Export results
        rallycut evaluate-tracking tune-ball-filter --all -o results.json
    """
    from rallycut.evaluation.tracking.ball_grid_search import (
        BallRawCache,
        CachedBallData,
        ball_grid_search,
    )
    from rallycut.evaluation.tracking.ball_param_grid import (
        BALL_AVAILABLE_GRIDS,
        ball_grid_size,
        describe_ball_config_diff,
        get_ball_grid,
    )
    from rallycut.evaluation.tracking.db import load_labeled_rallies

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print(
            "[red]Error:[/red] Specify --rally-id, --video-id, or --all"
        )
        raise typer.Exit(1)

    if grid not in BALL_AVAILABLE_GRIDS:
        console.print(
            f"[red]Error:[/red] Unknown grid '{grid}'. "
            f"Available: {', '.join(BALL_AVAILABLE_GRIDS.keys())}"
        )
        raise typer.Exit(1)

    # Initialize cache
    ball_cache = BallRawCache()

    if clear_cache:
        count = ball_cache.clear()
        console.print(f"[yellow]Cleared {count} cached ball position files[/yellow]")

    # Load rallies from database (only videos with validated ball ground truth)
    console.print("[bold]Loading labeled rallies from database...[/bold]")
    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
        ball_gt_only=True,  # Only use videos with validated ball ground truth
    )

    if not rallies:
        console.print("[yellow]No rallies with validated ball ground truth found[/yellow]")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies)} rally(s) with validated ball ground truth\n")

    from rallycut.labeling.ground_truth import GroundTruthPosition

    # Check for raw ball positions and build cache
    # NOTE: Raw ball positions (before Kalman filtering) are not stored in the database.
    # They must be cached via --cache-only after re-running ball tracking with preserve_raw=True.
    # For now, we can use the filtered positions as a baseline - they'll still allow testing
    # different Kalman filter parameters, though the starting point is already filtered.
    cached_rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]] = []
    rallies_using_filtered: list = []

    for rally in rallies:
        if rally.predictions is None or not rally.predictions.ball_positions:
            console.print(
                f"[yellow]Rally {rally.rally_id[:8]}... has no ball predictions, skipping[/yellow]"
            )
            continue

        # First check file cache
        cached = ball_cache.get(rally.rally_id)
        if cached:
            # Filter GT to ball positions
            gt_ball = [p for p in rally.ground_truth.positions if p.label == "ball"]
            cached_rallies.append((cached, gt_ball))
            continue

        # No cached raw positions - use filtered positions as fallback
        # This still allows testing Kalman filter parameters, though results may differ
        # from running on truly raw (unfiltered) positions
        cached_data = CachedBallData(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            raw_ball_positions=rally.predictions.ball_positions,  # Using filtered as fallback
            video_fps=rally.video_fps,
            frame_count=rally.predictions.frame_count,
            video_width=rally.video_width,
            video_height=rally.video_height,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
        )
        ball_cache.put(cached_data)
        gt_ball = [p for p in rally.ground_truth.positions if p.label == "ball"]
        cached_rallies.append((cached_data, gt_ball))
        rallies_using_filtered.append(rally)

    # Report status
    if rallies_using_filtered:
        console.print(
            f"\n[dim]Note: {len(rallies_using_filtered)} rally(s) using already-filtered positions.[/dim]"
        )
        console.print(
            "[dim]For more accurate results, re-run ball tracking with preserve_raw=True.[/dim]"
        )

    console.print(f"\n  Rallies available for grid search: {len(cached_rallies)}")

    if cache_only:
        stats = ball_cache.stats()
        console.print("\n[green]Raw ball positions cached successfully![/green]")
        console.print(f"  Cache location: {stats['cache_dir']}")
        console.print(f"  Cached rallies: {stats['count']}")
        console.print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        return

    if not cached_rallies:
        console.print("[red]No rallies available for grid search[/red]")
        raise typer.Exit(1)

    # Run grid search
    param_grid = get_ball_grid(grid)
    num_configs = ball_grid_size(param_grid)

    console.print()
    console.print(f"[bold]Ball Filter Grid Search - {num_configs} configs, {len(cached_rallies)} rallies[/bold]")
    console.print("=" * 60)
    console.print(f"Grid: {grid}")
    console.print(f"Match threshold: {match_threshold:.0f}px")
    if min_rally_detection is not None:
        console.print(f"Min rally detection constraint: [yellow]{min_rally_detection:.0%}[/yellow]")

    # Show parameters being searched
    console.print("\n[dim]Parameters being searched:[/dim]")
    for param, values in param_grid.items():
        console.print(f"  {param}: {values}")

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=num_configs)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, completed=current)

        result = ball_grid_search(
            rallies=cached_rallies,
            param_grid=param_grid,
            match_threshold_px=match_threshold,
            min_rally_detection_rate=min_rally_detection,
            progress_callback=update_progress,
        )

    # Display results
    console.print()
    console.print("[bold]Best Configuration[/bold]")
    console.print("=" * 60)

    best = result.best_config
    for param in param_grid.keys():
        value = getattr(best, param)
        default_val = getattr(type(best)(), param)
        if value != default_val:
            console.print(f"  [cyan]{param}[/cyan]: {value} [dim](default: {default_val})[/dim]")
        else:
            console.print(f"  [dim]{param}: {value}[/dim]")

    console.print()
    console.print(
        f"Results: Detection=[bold green]{result.best_detection_rate:.1%}[/bold green], "
        f"Match={result.best_match_rate:.1%}, "
        f"Error={result.best_mean_error_px:.1f}px"
    )

    if result.improvement_match_rate != 0:
        improvement_color = "green" if result.improvement_match_rate > 0 else "red"
        console.print(
            f"Improvement over default: [{improvement_color}]{result.improvement_match_rate:+.1%}[/{improvement_color}] "
            f"(default match={result.default_match_rate:.1%})"
        )

    if result.rejected_count > 0:
        console.print(
            f"\n[yellow]Rejected {result.rejected_count} configs that violated constraints[/yellow]"
        )

    # Top N configurations
    console.print()
    console.print(f"[bold]Top {min(top_n, len(result.all_results))} Configurations[/bold]")

    top_table = Table(show_header=True, header_style="bold")
    top_table.add_column("Rank", justify="right")
    top_table.add_column("Detection", justify="right")
    top_table.add_column("Match", justify="right")
    top_table.add_column("Error", justify="right")
    top_table.add_column("<20px", justify="right")
    top_table.add_column("Changes from Default")

    shown = 0
    for i, config_result in enumerate(result.all_results):
        if config_result.rejected:
            continue
        if shown >= top_n:
            break

        rank = shown + 1
        metrics = config_result.aggregate_metrics
        diff = describe_ball_config_diff(config_result.config)

        # Truncate diff if too long
        if len(diff) > 40:
            diff = diff[:37] + "..."

        match_style = "green" if metrics.match_rate >= 0.70 else ("yellow" if metrics.match_rate >= 0.50 else "red")

        top_table.add_row(
            str(rank),
            f"{metrics.detection_rate:.1%}",
            f"[{match_style}]{metrics.match_rate:.1%}[/{match_style}]",
            f"{metrics.mean_error_px:.1f}px",
            f"{metrics.error_under_20px_rate:.1%}",
            diff,
        )
        shown += 1

    console.print(top_table)

    # Export to file if requested
    if output:
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[green]Full results exported to {output}[/green]")


@app.command(name="compare-ball-models")
@handle_errors
def compare_ball_models(
    rally_id: Annotated[
        str | None,
        typer.Option(
            "--rally-id", "-r",
            help="Evaluate specific rally by ID",
        ),
    ] = None,
    video_id: Annotated[
        str | None,
        typer.Option(
            "--video-id", "-v",
            help="Evaluate all labeled rallies in a video",
        ),
    ] = None,
    all_rallies: Annotated[
        bool,
        typer.Option(
            "--all", "-a",
            help="Evaluate all labeled rallies in database",
        ),
    ] = False,
    match_threshold: Annotated[
        float,
        typer.Option(
            "--match-threshold", "-t",
            help="Max distance (pixels) for a match",
        ),
    ] = 50.0,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Export full results to JSON file",
        ),
    ] = None,
) -> None:
    """Compare all available ball tracking models on ground truth.

    Runs each VballNet model variant on the labeled rallies and
    outputs a comparison table with detection rate, accuracy, and error metrics.

    Available models:
    - v2: VballNetV2 (default) - best accuracy (24% match rate at 50px)
    - v1b: VballNetV1b - highest detection rate (98%)
    - fast: VballNetFastV1 - lighter weight, faster inference

    Examples:

        # Compare all models on all labeled rallies
        rallycut evaluate-tracking compare-ball-models --all

        # Compare models on specific video
        rallycut evaluate-tracking compare-ball-models -v a7ee3d38-...

        # Export results to JSON
        rallycut evaluate-tracking compare-ball-models --all -o comparison.json
    """
    from rallycut.evaluation.tracking.ball_metrics import (
        BallTrackingMetrics,  # noqa: F401 - used in type annotation
        aggregate_ball_metrics,
        evaluate_ball_tracking,
    )
    from rallycut.evaluation.tracking.db import get_video_path, load_labeled_rallies
    from rallycut.tracking.ball_tracker import (
        BALL_MODELS,
        BallTracker,
        get_available_ball_models,
    )

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print(
            "[red]Error:[/red] Specify --rally-id, --video-id, or --all"
        )
        raise typer.Exit(1)

    # Load rallies from database (only videos with validated ball ground truth)
    console.print("[bold]Loading labeled rallies from database...[/bold]")
    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
        ball_gt_only=True,  # Only use videos with validated ball ground truth
    )

    if not rallies:
        console.print("[yellow]No rallies with validated ball ground truth found[/yellow]")
        raise typer.Exit(1)

    # Filter to rallies with ball ground truth
    rallies_with_ball = []
    for rally in rallies:
        gt_ball = [p for p in rally.ground_truth.positions if p.label == "ball"]
        if gt_ball:
            rallies_with_ball.append(rally)

    if not rallies_with_ball:
        console.print("[red]No rallies with ball ground truth found[/red]")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies_with_ball)} rally(s) with ball ground truth\n")

    # Get list of models to compare
    model_ids = get_available_ball_models()
    console.print(f"[bold]Comparing {len(model_ids)} Ball Tracking Models[/bold]")
    console.print("=" * 60)

    # Results storage: model_id -> list of (rally_id, metrics)
    model_results: dict[str, list[tuple[str, BallTrackingMetrics]]] = {}

    # Run each model on all rallies
    for model_id in model_ids:
        model_filename = BALL_MODELS[model_id][0]
        console.print(f"\n[bold]Model: {model_id}[/bold] ({model_filename})")

        tracker = BallTracker(model=model_id)
        model_results[model_id] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Running {model_id}...",
                total=len(rallies_with_ball),
            )

            for rally in rallies_with_ball:
                # Get video path from database
                video_path = get_video_path(rally.video_id)
                if video_path is None:
                    console.print(
                        f"[yellow]Rally {rally.rally_id[:8]}... video not found, skipping[/yellow]"
                    )
                    progress.update(task, advance=1)
                    continue

                # Run ball tracking on the rally segment
                try:
                    result = tracker.track_video(
                        video_path,
                        start_ms=rally.start_ms,
                        end_ms=rally.end_ms,
                        enable_filtering=True,  # Use Kalman filter
                    )

                    # Convert prediction frame numbers to rally-relative
                    # Predictions use absolute video frames, GT uses rally-relative
                    # Use actual video FPS (not rally metadata) for consistency with tracker
                    import cv2

                    from rallycut.evaluation.tracking.ball_metrics import (
                        find_optimal_frame_offset,
                    )
                    from rallycut.tracking.ball_tracker import BallPosition

                    cap = cv2.VideoCapture(str(video_path))
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()

                    # Calculate start frame using actual FPS (matches tracker)
                    start_frame = int(rally.start_ms / 1000 * actual_fps)

                    relative_positions = [
                        BallPosition(
                            frame_number=p.frame_number - start_frame,
                            x=p.x,
                            y=p.y,
                            confidence=p.confidence,
                        )
                        for p in result.positions
                    ]

                    # Auto-detect optimal frame offset for this video
                    # Different videos need different offsets due to FPS/labeling timing
                    optimal_offset, _ = find_optimal_frame_offset(
                        gt_ball, relative_positions,
                        rally.video_width, rally.video_height,
                    )

                    # Apply the optimal offset
                    relative_positions = [
                        BallPosition(
                            frame_number=p.frame_number - optimal_offset,
                            x=p.x,
                            y=p.y,
                            confidence=p.confidence,
                        )
                        for p in relative_positions
                    ]

                    # Evaluate against ground truth
                    gt_ball = [p for p in rally.ground_truth.positions if p.label == "ball"]
                    metrics = evaluate_ball_tracking(
                        ground_truth=gt_ball,
                        predictions=relative_positions,
                        video_width=rally.video_width,
                        video_height=rally.video_height,
                        video_fps=rally.video_fps,
                        match_threshold_px=match_threshold,
                    )

                    model_results[model_id].append((rally.rally_id, metrics))

                except Exception as e:
                    console.print(
                        f"[red]Error processing rally {rally.rally_id[:8]}...: {e}[/red]"
                    )

                progress.update(task, advance=1)

    # Display comparison table
    console.print("\n")
    console.print("[bold]Model Comparison Results[/bold]")
    console.print("=" * 80)

    comparison_table = Table(show_header=True, header_style="bold")
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Detection", justify="right")
    comparison_table.add_column("Match", justify="right")
    comparison_table.add_column("Mean Err", justify="right")
    comparison_table.add_column("Median", justify="right")
    comparison_table.add_column("P90", justify="right")
    comparison_table.add_column("<20px", justify="right")
    comparison_table.add_column("Jitter", justify="right")

    # Track best model for each metric
    best_match = 0.0
    best_model = ""

    for model_id in model_ids:
        results = model_results.get(model_id, [])
        if not results:
            comparison_table.add_row(
                model_id, "-", "-", "-", "-", "-", "-", "-"
            )
            continue

        # Aggregate metrics for this model
        metrics_list = [m for _, m in results]
        combined = aggregate_ball_metrics(metrics_list)

        # Track best
        if combined.match_rate > best_match:
            best_match = combined.match_rate
            best_model = model_id

        # Style based on match rate
        match_style = (
            "green" if combined.match_rate >= 0.70
            else ("yellow" if combined.match_rate >= 0.50 else "red")
        )
        detection_style = (
            "green" if combined.detection_rate >= 0.70
            else ("yellow" if combined.detection_rate >= 0.50 else "red")
        )

        comparison_table.add_row(
            model_id,
            f"[{detection_style}]{combined.detection_rate:.1%}[/{detection_style}]",
            f"[{match_style}]{combined.match_rate:.1%}[/{match_style}]",
            f"{combined.mean_error_px:.1f}px",
            f"{combined.median_error_px:.1f}px",
            f"{combined.p90_error_px:.1f}px",
            f"{combined.error_under_20px_rate:.1%}",
            f"{combined.mean_jitter_px:.1f}px" if combined.mean_jitter_px > 0 else "-",
        )

    console.print(comparison_table)

    if best_model:
        console.print(f"\n[green]Best model: {best_model}[/green] (highest match rate: {best_match:.1%})")

    # Export to file if requested
    if output:
        output_data: dict[str, Any] = {
            "rallies": len(rallies_with_ball),
            "match_threshold_px": match_threshold,
            "models": {},
        }

        for model_id in model_ids:
            results = model_results.get(model_id, [])
            if not results:
                continue

            metrics_list = [m for _, m in results]
            combined = aggregate_ball_metrics(metrics_list)

            output_data["models"][model_id] = {
                "aggregate": combined.to_dict(),
                "per_rally": [
                    {"rally_id": rid, **m.to_dict()}
                    for rid, m in results
                ],
            }

        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Full results exported to {output}[/green]")
