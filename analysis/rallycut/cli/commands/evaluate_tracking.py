"""Evaluate player tracking against ground truth from database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from rallycut.cli.utils import handle_errors

if TYPE_CHECKING:
    from rallycut.cli.commands.compare_tracking import BallMetrics, MOTMetrics
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
    """Evaluate player tracking predictions against ground truth.

    Loads ground truth and predictions from the database and computes
    MOT (Multi-Object Tracking) metrics including MOTA, precision, recall,
    F1, and ID switches.

    Examples:

        # Evaluate specific rally
        rallycut evaluate-tracking --rally-id abc123

        # Evaluate all labeled rallies in a video
        rallycut evaluate-tracking --video-id a7ee3d38-...

        # Evaluate all labeled data
        rallycut evaluate-tracking --all

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
    )

    if not rallies:
        console.print("[yellow]No rallies with ground truth found[/yellow]")
        if rally_id:
            console.print(f"  Rally ID: {rally_id}")
        if video_id:
            console.print(f"  Video ID: {video_id}")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies)} rally(s) with ground truth\n")

    # Evaluate each rally
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
    """Display ball tracking metrics."""
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
