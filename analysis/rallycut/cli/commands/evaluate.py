"""Evaluate command - assess rally detection quality against ground truth."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from rallycut.cli.utils import handle_errors
from rallycut.core.config import MODEL_PRESETS
from rallycut.evaluation.cached_analysis import (
    AnalysisCache,
    PostProcessingParams,
    analyze_and_cache,
    apply_post_processing_custom,
)
from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.evaluation.matching import match_rallies
from rallycut.evaluation.metrics import (
    AggregateMetrics,
    VideoEvaluationResult,
    aggregate_metrics,
    compute_metrics,
)
from rallycut.evaluation.param_grid import (
    AVAILABLE_GRIDS,
    DEFAULT_PARAMS,
    generate_param_combinations,
    get_grid,
)
from rallycut.evaluation.video_resolver import VideoResolver

app = typer.Typer(help="Evaluate rally detection against ground truth")
console = Console()


def _print_summary(results: AggregateMetrics, params_used: PostProcessingParams | None) -> None:
    """Print evaluation summary to console."""
    console.print()
    console.print("=" * 60)
    console.print("[bold]RALLY DETECTION EVALUATION SUMMARY[/bold]")
    console.print("=" * 60)

    # Overall metrics table
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    rm = results.rally_metrics
    table.add_row("Videos Evaluated", str(results.video_count))
    table.add_row("Ground Truth Rallies", str(results.total_ground_truth))
    table.add_row("Detected Rallies", str(results.total_predictions))
    table.add_row("True Positives", str(rm.true_positives))
    table.add_row("False Positives", str(rm.false_positives))
    table.add_row("False Negatives (Missed)", str(rm.false_negatives))
    table.add_row("Precision", f"{rm.precision:.1%}")
    table.add_row("Recall", f"{rm.recall:.1%}")
    table.add_row("[bold]F1 Score[/bold]", f"[bold]{rm.f1:.1%}[/bold]")

    console.print(table)

    # Boundary accuracy
    bm = results.boundary_metrics
    if bm.mean_abs_start_error_ms is not None:
        console.print()
        console.print("[bold]Boundary Accuracy:[/bold]")
        console.print(f"  Mean |start error|: {bm.mean_abs_start_error_ms:.0f} ms")
        console.print(f"  Mean |end error|:   {bm.mean_abs_end_error_ms:.0f} ms")
        console.print(f"  Median start error: {bm.median_start_error_ms:.0f} ms")
        console.print(f"  Median end error:   {bm.median_end_error_ms:.0f} ms")

    # Processing time
    if results.total_processing_time is not None:
        console.print()
        console.print(f"Total processing time: {results.total_processing_time:.1f}s")

    # Parameters used
    if params_used:
        console.print()
        console.print("[dim]Parameters used:[/dim]")
        console.print(f"  [dim]min_gap_seconds={params_used.min_gap_seconds}[/dim]")
        console.print(
            f"  [dim]rally_continuation_seconds={params_used.rally_continuation_seconds}[/dim]"
        )
        console.print(f"  [dim]min_play_duration={params_used.min_play_duration}[/dim]")
        console.print(
            f"  [dim]boundary_confidence_threshold={params_used.boundary_confidence_threshold}[/dim]"
        )


def _print_per_video_results(results: list[VideoEvaluationResult]) -> None:
    """Print per-video breakdown."""
    console.print()
    console.print("[bold]Per-Video Results:[/bold]")

    table = Table()
    table.add_column("Video", style="cyan")
    table.add_column("GT", style="dim", justify="right")
    table.add_column("Det", style="dim", justify="right")
    table.add_column("Prec", style="green", justify="right")
    table.add_column("Recall", style="green", justify="right")
    table.add_column("F1", style="bold green", justify="right")
    table.add_column("Time", style="dim", justify="right")

    for r in results:
        rm = r.rally_metrics
        time_str = f"{r.processing_time_seconds:.1f}s" if r.processing_time_seconds else "-"
        # Truncate filename for display
        filename = r.video_filename[:35] + "..." if len(r.video_filename) > 38 else r.video_filename
        table.add_row(
            filename,
            str(r.ground_truth_count),
            str(r.prediction_count),
            f"{rm.precision:.0%}",
            f"{rm.recall:.0%}",
            f"{rm.f1:.0%}",
            time_str,
        )

    console.print(table)


def _export_json(
    results: AggregateMetrics,
    path: Path,
    params: PostProcessingParams,
) -> None:
    """Export results to JSON."""
    data = {
        "parameters": {
            "min_gap_seconds": params.min_gap_seconds,
            "rally_continuation_seconds": params.rally_continuation_seconds,
            "min_play_duration": params.min_play_duration,
            "padding_seconds": params.padding_seconds,
            "padding_end_seconds": params.padding_end_seconds,
            "boundary_confidence_threshold": params.boundary_confidence_threshold,
            "min_active_density": params.min_active_density,
        },
        "summary": {
            "video_count": results.video_count,
            "total_ground_truth": results.total_ground_truth,
            "total_predictions": results.total_predictions,
            "true_positives": results.rally_metrics.true_positives,
            "false_positives": results.rally_metrics.false_positives,
            "false_negatives": results.rally_metrics.false_negatives,
            "precision": results.rally_metrics.precision,
            "recall": results.rally_metrics.recall,
            "f1": results.rally_metrics.f1,
        },
        "boundary_metrics": {
            "mean_abs_start_error_ms": results.boundary_metrics.mean_abs_start_error_ms,
            "mean_abs_end_error_ms": results.boundary_metrics.mean_abs_end_error_ms,
            "median_start_error_ms": results.boundary_metrics.median_start_error_ms,
            "median_end_error_ms": results.boundary_metrics.median_end_error_ms,
        },
        "videos": [
            {
                "video_id": r.video_id,
                "filename": r.video_filename,
                "ground_truth_count": r.ground_truth_count,
                "prediction_count": r.prediction_count,
                "true_positives": r.rally_metrics.true_positives,
                "false_positives": r.rally_metrics.false_positives,
                "false_negatives": r.rally_metrics.false_negatives,
                "precision": r.rally_metrics.precision,
                "recall": r.rally_metrics.recall,
                "f1": r.rally_metrics.f1,
                "processing_time_seconds": r.processing_time_seconds,
            }
            for r in results.per_video_results
        ],
    }
    path.write_text(json.dumps(data, indent=2))


def _export_csv(results: list[VideoEvaluationResult], path: Path) -> None:
    """Export results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_id",
                "filename",
                "ground_truth",
                "predictions",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "processing_time_s",
            ]
        )
        for r in results:
            rm = r.rally_metrics
            writer.writerow(
                [
                    r.video_id,
                    r.video_filename,
                    r.ground_truth_count,
                    r.prediction_count,
                    rm.true_positives,
                    rm.false_positives,
                    rm.false_negatives,
                    f"{rm.precision:.4f}",
                    f"{rm.recall:.4f}",
                    f"{rm.f1:.4f}",
                    f"{r.processing_time_seconds:.2f}" if r.processing_time_seconds else "",
                ]
            )


def _run_evaluation(
    videos: list[EvaluationVideo],
    params: PostProcessingParams,
    iou_threshold: float,
    cache: AnalysisCache,
    resolver: VideoResolver,
    stride: int,
    use_cache: bool,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> list[VideoEvaluationResult]:
    """Run evaluation on a list of videos."""
    results: list[VideoEvaluationResult] = []
    total = len(videos)

    for i, video in enumerate(videos):
        if progress and task_id is not None:
            progress.update(task_id, description=f"Processing {video.filename[:30]}...")
        # Plain text log visible in non-interactive/captured output
        print(f"[{i + 1}/{total}] Processing {video.filename}...", flush=True)

        # Get ground truth as (start, end) tuples
        ground_truth = [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]

        # Get predictions
        start_time = time.time()

        if use_cache:
            # Use cached analysis and apply post-processing
            cached = cache.get(video.content_hash, stride)
            if cached is None:
                # Need to run analysis
                video_path = resolver.resolve(video.s3_key, video.content_hash)
                cached = analyze_and_cache(
                    video_path,
                    video.id,
                    video.content_hash,
                    stride=stride,
                    cache=cache,
                )
            segments = apply_post_processing_custom(cached, params)
        else:
            # Run full analysis
            video_path = resolver.resolve(video.s3_key, video.content_hash)
            cached = analyze_and_cache(
                video_path,
                video.id,
                video.content_hash,
                stride=stride,
                cache=cache,
            )
            segments = apply_post_processing_custom(cached, params)

        processing_time = time.time() - start_time

        # Convert segments to (start, end) tuples
        predictions = [(s.start_time, s.end_time) for s in segments]

        # Match and compute metrics
        matching = match_rallies(ground_truth, predictions, iou_threshold)
        result = compute_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            matching_result=matching,
            video_id=video.id,
            video_filename=video.filename,
            iou_threshold=iou_threshold,
            processing_time=processing_time,
        )
        results.append(result)
        m = result.rally_metrics
        print(
            f"[{i + 1}/{total}] {video.filename}: "
            f"P={m.precision:.0%} R={m.recall:.0%} F1={m.f1:.0%} "
            f"({processing_time:.1f}s)",
            flush=True,
        )

        if progress and task_id is not None:
            progress.update(task_id, advance=1)

    return results


@app.callback(invoke_without_command=True)
@handle_errors
def evaluate(
    ctx: typer.Context,
    video_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--video",
            "-v",
            help="Specific video IDs to evaluate (can be repeated)",
        ),
    ] = None,
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou",
            "-i",
            help="IoU threshold for matching rallies (0.0-1.0)",
        ),
    ] = 0.5,
    stride: Annotated[
        int,
        typer.Option(
            "--stride",
            "-s",
            help="Override stride parameter for ML analysis",
        ),
    ] = 48,
    min_gap: Annotated[
        float | None,
        typer.Option(
            "--min-gap",
            help="Override min_gap_seconds parameter",
        ),
    ] = None,
    rally_continuation: Annotated[
        float | None,
        typer.Option(
            "--rally-continuation",
            help="Override rally_continuation_seconds parameter",
        ),
    ] = None,
    min_play: Annotated[
        float | None,
        typer.Option(
            "--min-play",
            help="Override min_play_duration parameter",
        ),
    ] = None,
    boundary_threshold: Annotated[
        float | None,
        typer.Option(
            "--boundary-threshold",
            help="Override boundary_confidence_threshold (0.0-1.0)",
        ),
    ] = None,
    output_json: Annotated[
        Path | None,
        typer.Option(
            "--json",
            "-o",
            help="Export results to JSON file",
        ),
    ] = None,
    output_csv: Annotated[
        Path | None,
        typer.Option(
            "--csv",
            help="Export results to CSV file",
        ),
    ] = None,
    use_cache: Annotated[
        bool,
        typer.Option(
            "--use-cache/--no-cache",
            help="Use cached ML analysis results (fast iteration)",
        ),
    ] = True,
    cache_analysis: Annotated[
        bool,
        typer.Option(
            "--cache-analysis",
            help="Only cache ML analysis, don't evaluate (initial setup)",
        ),
    ] = False,
    skip_detection: Annotated[
        bool,
        typer.Option(
            "--skip-detection",
            help="Use existing ML rallies from DB instead of re-running detection",
        ),
    ] = False,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model variant: 'indoor' (original) or 'beach' (fine-tuned with beach heuristics)",
        ),
    ] = "indoor",
) -> None:
    """
    Evaluate rally detection against ground truth from the database.

    Ground truth = rallies with confidence IS NULL (manually tagged)
    Predictions = rallies from ML detection with post-processing

    Examples:

        rallycut evaluate                           # Evaluate all videos

        rallycut evaluate -v abc123                 # Specific video

        rallycut evaluate --min-gap 3 --rally-continuation 2.5  # Test parameters

        rallycut evaluate --use-cache               # Fast iteration with cached ML

        rallycut evaluate --cache-analysis          # Initial cache setup

        rallycut evaluate --json results.json       # Export results
    """
    # Skip if subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    console.print()
    console.print("[bold]RallyCut Evaluation Framework[/bold]")
    console.print(f"Model: [yellow]{model}[/yellow]")

    # Get model-specific preset
    preset = MODEL_PRESETS.get(model, MODEL_PRESETS["indoor"])

    # Build parameter overrides, using preset values as defaults
    params = PostProcessingParams(
        min_gap_seconds=min_gap if min_gap is not None else DEFAULT_PARAMS.min_gap_seconds,
        rally_continuation_seconds=(
            rally_continuation
            if rally_continuation is not None
            else preset.get("rally_continuation_seconds", DEFAULT_PARAMS.rally_continuation_seconds)
        ),
        min_play_duration=(
            min_play
            if min_play is not None
            else preset.get("min_play_duration", DEFAULT_PARAMS.min_play_duration)
        ),
        padding_seconds=DEFAULT_PARAMS.padding_seconds,
        padding_end_seconds=DEFAULT_PARAMS.padding_end_seconds,
        boundary_confidence_threshold=(
            boundary_threshold
            if boundary_threshold is not None
            else preset.get("boundary_confidence_threshold", DEFAULT_PARAMS.boundary_confidence_threshold)
        ),
        min_active_density=preset.get("min_active_density", DEFAULT_PARAMS.min_active_density),
        min_active_windows=DEFAULT_PARAMS.min_active_windows,
    )

    # Show parameter overrides
    overrides = []
    if min_gap is not None:
        overrides.append(f"min_gap_seconds={min_gap}")
    if rally_continuation is not None:
        overrides.append(f"rally_continuation_seconds={rally_continuation}")
    if min_play is not None:
        overrides.append(f"min_play_duration={min_play}")
    if boundary_threshold is not None:
        overrides.append(f"boundary_confidence_threshold={boundary_threshold}")

    if overrides:
        console.print(f"Parameter overrides: {', '.join(overrides)}")

    # Load evaluation data from database
    console.print()
    console.print("Loading ground truth from database...")
    videos = load_evaluation_videos(video_ids)

    if not videos:
        console.print("[red]No videos with ground truth found![/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(videos)} videos with ground truth:")
    for v in videos:
        console.print(f"  - {v.filename}: {len(v.ground_truth_rallies)} ground truth rallies")

    # Initialize cache and resolver
    cache = AnalysisCache()
    resolver = VideoResolver()

    # Cache-only mode
    if cache_analysis:
        console.print()
        console.print("[bold]Caching ML analysis results...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Caching...", total=len(videos))

            for video in videos:
                progress.update(task, description=f"Analyzing {video.filename[:30]}...")
                video_path = resolver.resolve(video.s3_key, video.content_hash)

                # Run analysis and cache
                analyze_and_cache(
                    video_path,
                    video.id,
                    video.content_hash,
                    stride=stride,
                    cache=cache,
                )
                progress.update(task, advance=1)

        console.print()
        console.print("[green]ML analysis cached successfully![/green]")
        console.print(f"Cache location: {cache.cache_dir}")
        return

    # Skip detection mode - use existing DB rallies
    if skip_detection:
        console.print()
        console.print("Using existing ML rallies from database...")
        results: list[VideoEvaluationResult] = []

        for video in videos:
            ground_truth = [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]
            predictions = [(r.start_seconds, r.end_seconds) for r in video.ml_detected_rallies]

            matching = match_rallies(ground_truth, predictions, iou_threshold)
            result = compute_metrics(
                ground_truth=ground_truth,
                predictions=predictions,
                matching_result=matching,
                video_id=video.id,
                video_filename=video.filename,
                iou_threshold=iou_threshold,
            )
            results.append(result)

        aggregated = aggregate_metrics(results)
        _print_summary(aggregated, None)
        _print_per_video_results(results)

        if output_json:
            _export_json(aggregated, output_json, params)
            console.print(f"\nResults exported to: [cyan]{output_json}[/cyan]")

        if output_csv:
            _export_csv(results, output_csv)
            console.print(f"CSV exported to: [cyan]{output_csv}[/cyan]")

        return

    # Run evaluation
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(videos))

        results = _run_evaluation(
            videos=videos,
            params=params,
            iou_threshold=iou_threshold,
            cache=cache,
            resolver=resolver,
            stride=stride,
            use_cache=use_cache,
            progress=progress,
            task_id=task,
        )

    # Aggregate and report
    aggregated = aggregate_metrics(results)
    _print_summary(aggregated, params)
    _print_per_video_results(results)

    # Export if requested
    if output_json:
        _export_json(aggregated, output_json, params)
        console.print(f"\nResults exported to: [cyan]{output_json}[/cyan]")

    if output_csv:
        _export_csv(results, output_csv)
        console.print(f"CSV exported to: [cyan]{output_csv}[/cyan]")


@app.command()
@handle_errors
def tune(
    grid_name: Annotated[
        str,
        typer.Option(
            "--grid",
            "-g",
            help=f"Parameter grid to use: {', '.join(AVAILABLE_GRIDS.keys())}",
        ),
    ] = "beach",
    video_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--video",
            "-v",
            help="Specific video IDs to evaluate",
        ),
    ] = None,
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou",
            "-i",
            help="IoU threshold for matching",
        ),
    ] = 0.5,
    stride: Annotated[
        int,
        typer.Option(
            "--stride",
            "-s",
            help="Stride for ML analysis",
        ),
    ] = 48,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file for tuning results",
        ),
    ] = Path("tune_results.json"),
    top_n: Annotated[
        int,
        typer.Option(
            "--top",
            "-n",
            help="Number of top results to show",
        ),
    ] = 5,
) -> None:
    """
    Run parameter sweep to find optimal settings.

    Tests multiple parameter combinations and reports the best
    configuration based on F1 score.

    Examples:

        rallycut evaluate tune --grid beach

        rallycut evaluate tune --grid full -o full_sweep.json

        rallycut evaluate tune --grid quick --top 10
    """
    console.print()
    console.print("[bold]Parameter Tuning[/bold]")

    # Get grid
    try:
        param_grid = get_grid(grid_name)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    combinations = generate_param_combinations(param_grid)

    console.print(f"Grid: {grid_name}")
    console.print(f"Combinations to test: {len(combinations)}")

    # Load videos
    console.print()
    console.print("Loading ground truth from database...")
    videos = load_evaluation_videos(video_ids)

    if not videos:
        console.print("[red]No videos with ground truth found![/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(videos)} videos with ground truth")

    # Initialize
    cache = AnalysisCache()
    resolver = VideoResolver()

    # First ensure all videos are cached
    console.print()
    console.print("Ensuring ML analysis is cached...")
    for video in videos:
        if not cache.has(video.content_hash, stride):
            console.print(f"  Analyzing {video.filename}...")
            video_path = resolver.resolve(video.s3_key, video.content_hash)
            analyze_and_cache(
                video_path,
                video.id,
                video.content_hash,
                stride=stride,
                cache=cache,
            )

    # Run sweep
    console.print()
    all_results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Testing parameters...", total=len(combinations))

        for params in combinations:
            params_desc = f"gap={params.min_gap_seconds}, cont={params.rally_continuation_seconds}"
            progress.update(task, description=f"Testing {params_desc}...")

            video_results = _run_evaluation(
                videos=videos,
                params=params,
                iou_threshold=iou_threshold,
                cache=cache,
                resolver=resolver,
                stride=stride,
                use_cache=True,
            )

            aggregated = aggregate_metrics(video_results)
            all_results.append(
                {
                    "params": {
                        "min_gap_seconds": params.min_gap_seconds,
                        "rally_continuation_seconds": params.rally_continuation_seconds,
                        "min_play_duration": params.min_play_duration,
                        "boundary_confidence_threshold": params.boundary_confidence_threshold,
                        "min_active_density": params.min_active_density,
                    },
                    "f1": aggregated.rally_metrics.f1,
                    "precision": aggregated.rally_metrics.precision,
                    "recall": aggregated.rally_metrics.recall,
                    "true_positives": aggregated.rally_metrics.true_positives,
                    "false_positives": aggregated.rally_metrics.false_positives,
                    "false_negatives": aggregated.rally_metrics.false_negatives,
                }
            )

            progress.update(task, advance=1)

    # Sort by F1 and report
    all_results.sort(key=lambda x: x["f1"], reverse=True)

    console.print()
    console.print(f"[bold]Top {top_n} Configurations:[/bold]")

    table = Table()
    table.add_column("Rank", justify="right")
    table.add_column("F1", style="bold green", justify="right")
    table.add_column("Prec", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Parameters")

    for i, r in enumerate(all_results[:top_n], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        table.add_row(
            str(i),
            f"{r['f1']:.1%}",
            f"{r['precision']:.1%}",
            f"{r['recall']:.1%}",
            params_str,
        )

    console.print(table)

    # Save full results
    output.write_text(json.dumps(all_results, indent=2))
    console.print(f"\nFull results saved to: [cyan]{output}[/cyan]")

    # Show best params for copy-paste
    best = all_results[0]
    console.print()
    console.print("[bold]Best configuration:[/bold]")
    console.print(
        f"  rallycut evaluate "
        f"--min-gap {best['params']['min_gap_seconds']} "
        f"--rally-continuation {best['params']['rally_continuation_seconds']} "
        f"--min-play {best['params']['min_play_duration']} "
        f"--boundary-threshold {best['params']['boundary_confidence_threshold']}"
    )


@app.command()
@handle_errors
def cache_info() -> None:
    """Show cache statistics."""
    cache = AnalysisCache()

    console.print()
    console.print("[bold]Analysis Cache Info[/bold]")
    console.print(f"Location: {cache.cache_dir}")

    cache_files = list(cache.cache_dir.glob("*.json"))
    console.print(f"Cached analyses: {len(cache_files)}")

    if cache_files:
        total_size = sum(f.stat().st_size for f in cache_files)
        console.print(f"Total size: {total_size / 1024 / 1024:.1f} MB")


@app.command()
@handle_errors
def clear_cache(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation",
        ),
    ] = False,
) -> None:
    """Clear the analysis cache."""
    cache = AnalysisCache()

    if not force:
        cache_files = list(cache.cache_dir.glob("*.json"))
        if not cache_files:
            console.print("Cache is empty.")
            return

        confirm = typer.confirm(f"Delete {len(cache_files)} cached analyses?")
        if not confirm:
            console.print("Cancelled.")
            return

    count = cache.clear()
    console.print(f"Deleted {count} cached analyses.")
