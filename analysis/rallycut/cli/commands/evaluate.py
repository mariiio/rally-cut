"""Evaluate command - assess rally detection quality against ground truth."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from rallycut.cli.utils import handle_errors
from rallycut.core.config import MODEL_PRESETS, get_model_path
from rallycut.evaluation.cached_analysis import (
    AnalysisCache,
    CachedAnalysis,
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

logger = logging.getLogger(__name__)

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

    # Overmerge metrics
    om = results.overmerge_metrics
    console.print()
    console.print("[bold]Overmerge Detection:[/bold]")
    console.print(f"  Threshold: {om.overmerge_threshold_seconds:.0f}s")
    console.print(f"  Overmerge count: {om.overmerge_count} / {len(om.segment_durations_seconds)}")
    console.print(f"  Overmerge rate: {om.overmerge_rate:.1%}")
    if om.max_segment_duration_seconds is not None:
        console.print(f"  Max segment duration: {om.max_segment_duration_seconds:.1f}s")

    # Processing time
    if results.total_processing_time is not None:
        console.print()
        console.print(f"Total processing time: {results.total_processing_time:.1f}s")
        if results.inference_time_per_minute is not None:
            console.print(
                f"Inference time per minute of video: {results.inference_time_per_minute:.1f}s/min"
            )

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
        "overmerge_metrics": {
            "threshold_seconds": results.overmerge_metrics.overmerge_threshold_seconds,
            "overmerge_count": results.overmerge_metrics.overmerge_count,
            "total_segments": len(results.overmerge_metrics.segment_durations_seconds),
            "overmerge_rate": results.overmerge_metrics.overmerge_rate,
            "max_segment_duration_seconds": results.overmerge_metrics.max_segment_duration_seconds,
        },
        "processing": {
            "total_time_seconds": results.total_processing_time,
            "total_video_duration_seconds": results.total_video_duration_seconds,
            "inference_time_per_minute": results.inference_time_per_minute,
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
                "video_duration_seconds": r.video_duration_seconds,
                "inference_time_per_minute": r.inference_time_per_minute,
                "overmerge_count": r.overmerge_metrics.overmerge_count,
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


def _apply_binary_head_decoder(
    content_hash: str,
    stride: int,
    feature_cache_dir: Path | None = None,
    binary_head_model_path: Path | None = None,
) -> list | None:
    """Apply binary head + deterministic decoder to cached features.

    Args:
        content_hash: Video content hash to look up cached features.
        stride: Analysis stride in frames.
        feature_cache_dir: Directory containing cached features.
        binary_head_model_path: Path to binary head model (optional).

    Returns:
        List of TimeSegment objects, or None if features/model not found.
    """
    from rallycut.core.config import get_config
    from rallycut.core.models import GameState, TimeSegment
    from rallycut.temporal.deterministic_decoder import DecoderConfig
    from rallycut.temporal.features import FeatureCache
    from rallycut.temporal.inference import load_binary_head_model, run_binary_head_decoder

    config = get_config()

    # Determine model path
    if binary_head_model_path is not None:
        model_path = binary_head_model_path
    else:
        model_path = config.weights_dir / "binary_head" / "best_binary_head.pt"

    if not model_path.exists():
        return None

    # Load features from training feature cache (same default as temporal model)
    cache_dir = feature_cache_dir or Path("training_data/features")
    feature_cache = FeatureCache(cache_dir=cache_dir)
    cached_data = feature_cache.get(content_hash, stride)

    if cached_data is None:
        return None

    features, metadata = cached_data

    # Load model and run decoder
    model = load_binary_head_model(model_path, device="cpu")
    decoder_config = DecoderConfig(fps=metadata.fps, stride=stride)
    result = run_binary_head_decoder(features, model, decoder_config, device="cpu")

    # Convert to TimeSegments
    segments = []
    for start_time, end_time in result.segments:
        start_frame = int(start_time * metadata.fps)
        end_frame = int(end_time * metadata.fps)
        segments.append(
            TimeSegment(
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                state=GameState.PLAY,
            )
        )

    return segments


def _apply_temporal_model(
    cached: CachedAnalysis,
    model: object,  # torch.nn.Module
    stride: int,
    content_hash: str,
    feature_cache_dir: Path | None = None,
) -> list | None:
    """Apply temporal model to cached analysis results.

    Uses pre-extracted features from the specified feature cache directory.

    Args:
        cached: Cached analysis with raw_results
        model: Loaded temporal model
        stride: Analysis stride in frames
        content_hash: Video content hash to look up cached features
        feature_cache_dir: Directory containing cached features.
            Defaults to training_data/features/ if not specified.

    Returns:
        List of RallySegment objects, or None if features not found
    """
    from rallycut.temporal.features import FeatureCache
    from rallycut.temporal.inference import (
        TemporalInferenceConfig,
        run_temporal_inference,
    )

    # Load features from training feature cache
    cache_dir = feature_cache_dir or Path("training_data/features")
    feature_cache = FeatureCache(cache_dir=cache_dir)

    # Load coarse features
    cached_data = feature_cache.get(content_hash, stride)
    if cached_data is None:
        return None  # Features not cached for this video

    features, metadata = cached_data

    # Ensure features match raw_results length
    raw_results = cached.raw_results
    min_len = min(len(features), len(raw_results))
    features = features[:min_len]

    # Create inference config
    config = TemporalInferenceConfig(
        coarse_stride=stride,
        device="cpu",
    )

    # Run temporal inference
    result = run_temporal_inference(
        features=features,
        metadata=metadata,
        model=model,  # type: ignore[arg-type]
        config=config,
    )

    return result.segments


def _run_evaluation(
    videos: list[EvaluationVideo],
    params: PostProcessingParams,
    iou_threshold: float,
    cache: AnalysisCache,
    resolver: VideoResolver,
    stride: int,
    use_cache: bool,
    model_path: Path | None = None,
    model_id: str = "default",
    temporal_model_path: Path | None = None,
    feature_cache_dir: Path | None = None,
    use_binary_head: bool = False,
    use_heuristics: bool = False,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> list[VideoEvaluationResult]:
    """Run evaluation on a list of videos.

    Pipeline priority:
    1. If temporal_model_path: use temporal model (deprecated)
    2. If use_binary_head: use binary head + decoder
    3. If use_heuristics: use heuristics
    4. Auto: use binary head if features cached, else heuristics
    """
    results: list[VideoEvaluationResult] = []
    total = len(videos)

    # Load temporal model if specified
    temporal_model = None
    if temporal_model_path is not None:
        from rallycut.temporal.inference import load_temporal_model

        temporal_model = load_temporal_model(temporal_model_path)
        logger.info("Using temporal model: %s", temporal_model_path)

    for i, video in enumerate(videos):
        if progress and task_id is not None:
            progress.update(task_id, description=f"Processing {video.filename[:30]}...")
        # Plain text log visible in non-interactive/captured output
        print(f"[{i + 1}/{total}] Processing {video.filename}...", flush=True)

        # Get ground truth as (start, end) tuples
        ground_truth = [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]

        # Get predictions
        start_time = time.time()

        # Get cached analysis (or run analysis if needed)
        cached = cache.get(video.content_hash, stride, model_id) if use_cache else None
        if cached is None:
            video_path = resolver.resolve(video.s3_key, video.content_hash)
            cached = analyze_and_cache(
                video_path,
                video.id,
                video.content_hash,
                stride=stride,
                cache=cache,
                model_path=model_path,
                model_id=model_id,
            )

        # Apply post-processing (temporal model, binary head, or heuristics)
        segments = None

        if temporal_model is not None:
            # Deprecated temporal model path
            segments = _apply_temporal_model(
                cached, temporal_model, stride, video.content_hash, feature_cache_dir
            )
            if segments is None:
                logger.warning("No cached features for %s, using heuristics", video.filename)

        elif use_binary_head or (not use_heuristics):
            # Binary head path (explicit or auto-selected)
            segments = _apply_binary_head_decoder(
                video.content_hash, stride, feature_cache_dir
            )
            if segments is None and not use_heuristics:
                # Auto-selection: fallback to heuristics if no features/model
                logger.info("No cached features/model for %s, using heuristics", video.filename)

        # Fallback to heuristics
        if segments is None:
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
            video_duration_seconds=video.duration_seconds,
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
    temporal_model: Annotated[
        Path | None,
        typer.Option(
            "--temporal-model",
            "-t",
            help="Path to temporal model for learned post-processing (replaces heuristics)",
        ),
    ] = None,
    feature_cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--feature-cache",
            help="Directory for cached features (used with --temporal-model or --binary-head)",
        ),
    ] = None,
    binary_head: Annotated[
        bool,
        typer.Option(
            "--binary-head",
            help="Use binary head + decoder for evaluation (84%% F1 at IoU=0.4, default when features cached)",
        ),
    ] = False,
    use_heuristics: Annotated[
        bool,
        typer.Option(
            "--heuristics",
            help="Force heuristics pipeline instead of auto-selecting binary head",
        ),
    ] = False,
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

    # Show deprecation warning for temporal model
    if temporal_model is not None:
        import warnings

        warnings.warn(
            "--temporal-model is deprecated. Use --binary-head instead (84% F1 vs 65% F1).",
            DeprecationWarning,
            stacklevel=2,
        )
        console.print(
            "[yellow]Warning: --temporal-model is deprecated. "
            "Use --binary-head for better results (84% F1).[/yellow]"
        )

    # Show pipeline info
    if binary_head:
        console.print("Pipeline: [green]Binary head + decoder (84% F1)[/green]")
    elif use_heuristics:
        console.print("Pipeline: [dim]Heuristics (57% F1)[/dim]")
    elif temporal_model is None:
        console.print("Pipeline: [dim]Auto-selecting (binary head if features cached)[/dim]")

    # Resolve model weights path
    model_path = get_model_path(model)
    if model_path:
        console.print(f"Weights: [dim]{model_path}[/dim]")
    else:
        console.print(
            f"[yellow]Warning: No local weights found for '{model}', using default[/yellow]"
        )

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
            else preset.get(
                "boundary_confidence_threshold", DEFAULT_PARAMS.boundary_confidence_threshold
            )
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
                    model_path=model_path,
                    model_id=model,
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
                video_duration_seconds=video.duration_seconds,
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
            model_path=model_path,
            model_id=model,
            temporal_model_path=temporal_model,
            feature_cache_dir=feature_cache_dir,
            use_binary_head=binary_head,
            use_heuristics=use_heuristics,
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
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model variant: 'indoor' or 'beach'",
        ),
    ] = "indoor",
) -> None:
    """
    Run parameter sweep to find optimal settings.

    Tests multiple parameter combinations and reports the best
    configuration based on F1 score.

    Examples:

        rallycut evaluate tune --grid beach --model beach

        rallycut evaluate tune --grid full -o full_sweep.json

        rallycut evaluate tune --grid quick --top 10
    """
    console.print()
    console.print("[bold]Parameter Tuning[/bold]")
    console.print(f"Model: [yellow]{model}[/yellow]")

    # Resolve model weights path
    model_path = get_model_path(model)
    if model_path:
        console.print(f"Weights: [dim]{model_path}[/dim]")
    else:
        console.print(
            f"[yellow]Warning: No local weights found for '{model}', using default[/yellow]"
        )

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
        if not cache.has(video.content_hash, stride, model):
            console.print(f"  Analyzing {video.filename}...")
            video_path = resolver.resolve(video.s3_key, video.content_hash)
            analyze_and_cache(
                video_path,
                video.id,
                video.content_hash,
                stride=stride,
                cache=cache,
                model_path=model_path,
                model_id=model,
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
                model_path=model_path,
                model_id=model,
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


@app.command()
@handle_errors
def tune_decoder(
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou",
            "-i",
            help="IoU threshold for matching rallies (0.0-1.0)",
        ),
    ] = 0.4,
    stride: Annotated[
        int,
        typer.Option(
            "--stride",
            "-s",
            help="Frame stride (must match cached features)",
        ),
    ] = 48,
    top_n: Annotated[
        int,
        typer.Option(
            "--top",
            "-n",
            help="Number of top results to show",
        ),
    ] = 10,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file for tuning results",
        ),
    ] = Path("decoder_tune_results.json"),
    feature_cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--feature-cache",
            help="Directory containing cached features",
        ),
    ] = None,
    video_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--video",
            "-v",
            help="Specific video IDs to evaluate",
        ),
    ] = None,
    min_video_f1: Annotated[
        float | None,
        typer.Option(
            "--min-video-f1",
            help="Minimum F1 required for each video (rejects configs where any video drops below)",
        ),
    ] = None,
) -> None:
    """
    Grid search for optimal decoder parameters.

    Searches over decoder parameters (t_on, t_off, patience, etc.) to find
    the configuration that maximizes F1 score with minimal overmerge.

    Use --min-video-f1 to ensure no video drops below a threshold (prevents regressions).

    Requires cached binary head probabilities (run train extract-features first).

    Examples:

        rallycut evaluate tune-decoder --iou 0.4 --top 20

        rallycut evaluate tune-decoder --min-video-f1 0.70  # No video below 70%

        rallycut evaluate tune-decoder --output results.json
    """
    import numpy as np
    import torch

    from rallycut.core.config import get_config
    from rallycut.temporal.deterministic_decoder import (
        DEFAULT_PARAM_GRID,
        DecoderConfig,
        GridSearchResult,
        compute_segment_metrics,
        decode,
        grid_search,
    )
    from rallycut.temporal.features import FeatureCache
    from rallycut.temporal.inference import load_binary_head_model

    console.print()
    console.print("[bold]Decoder Parameter Grid Search[/bold]")
    console.print(f"IoU threshold: {iou_threshold}")
    console.print(f"Stride: {stride}")
    if min_video_f1 is not None:
        console.print(f"Min video F1 constraint: [yellow]{min_video_f1:.0%}[/yellow]")

    # Load binary head model
    config = get_config()
    model_path = config.weights_dir / "binary_head" / "best_binary_head.pt"
    if not model_path.exists():
        console.print(f"[red]Binary head model not found: {model_path}[/red]")
        console.print("Run 'uv run rallycut train binary-head' to train the model.")
        raise typer.Exit(1)

    console.print(f"Binary head model: [dim]{model_path}[/dim]")
    model = load_binary_head_model(model_path, device="cpu")

    # Load feature cache
    cache_dir = feature_cache_dir or Path("training_data/features")
    if not cache_dir.exists():
        console.print(f"[red]Feature cache directory not found: {cache_dir}[/red]")
        console.print("Run 'uv run rallycut train extract-features --stride 48' first.")
        raise typer.Exit(1)

    feature_cache = FeatureCache(cache_dir=cache_dir)
    console.print(f"Feature cache: [dim]{cache_dir}[/dim]")

    # Load ground truth from database
    console.print()
    console.print("Loading ground truth from database...")
    videos = load_evaluation_videos(video_ids)

    if not videos:
        console.print("[red]No videos with ground truth found![/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(videos)} videos with ground truth")

    # Load features and compute probabilities for each video
    video_probs: list[np.ndarray] = []
    video_ground_truths: list[list[tuple[float, float]]] = []
    video_fps: list[float] = []
    video_names: list[str] = []
    valid_videos: list[EvaluationVideo] = []

    console.print()
    console.print("Loading cached features and computing probabilities...")

    for video in videos:
        cached_data = feature_cache.get(video.content_hash, stride)
        if cached_data is None:
            console.print(f"  [yellow]Skipping {video.filename}: no cached features[/yellow]")
            continue

        features, metadata = cached_data

        # Run binary head to get probabilities
        features_t = torch.from_numpy(features).float()
        with torch.no_grad():
            logits = model(features_t)
            probs = torch.sigmoid(logits).cpu().numpy()

        video_probs.append(probs)
        video_ground_truths.append(
            [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]
        )
        video_fps.append(metadata.fps)
        video_names.append(video.filename)
        valid_videos.append(video)
        console.print(f"  [green]{video.filename}[/green]: {len(probs)} windows")

    if not video_probs:
        console.print("[red]No videos with cached features found![/red]")
        console.print("Run 'uv run rallycut train extract-features --stride 48' first.")
        raise typer.Exit(1)

    console.print()
    console.print(f"Loaded features for {len(video_probs)} videos")

    # Calculate total parameter combinations
    total_combos = 1
    for values in DEFAULT_PARAM_GRID.values():
        total_combos *= len(values)
    # Account for invalid t_off >= t_on combinations
    t_on_values = DEFAULT_PARAM_GRID["t_on"]
    t_off_values = DEFAULT_PARAM_GRID["t_off"]
    valid_pairs = sum(1 for t_on in t_on_values for t_off in t_off_values if t_off < t_on)
    valid_combos = total_combos // (len(t_on_values) * len(t_off_values)) * valid_pairs

    console.print(f"Parameter grid: ~{valid_combos} valid combinations")

    # Run grid search
    console.print()
    console.print("[bold]Running grid search...[/bold]")

    result: GridSearchResult = grid_search(
        video_probs=video_probs,
        video_ground_truths=video_ground_truths,
        video_fps=video_fps,
        stride=stride,
        iou_threshold=iou_threshold,
        param_grid=DEFAULT_PARAM_GRID,
        min_video_f1=min_video_f1,
        video_names=video_names,
    )

    # Show constraint info
    if min_video_f1 is not None:
        console.print(
            f"  Rejected {result.rejected_count} configs "
            f"(video F1 < {min_video_f1:.0%})"
        )
        if not result.all_results:
            console.print(
                f"[red]No configurations satisfy min_video_f1={min_video_f1:.0%}![/red]"
            )
            console.print("Try lowering --min-video-f1 threshold.")
            raise typer.Exit(1)

    # Print top configurations
    console.print()
    console.print(f"[bold]Top {top_n} Configurations:[/bold]")

    table = Table()
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Score", style="bold cyan", justify="right")
    table.add_column("F1", style="bold green", justify="right")
    table.add_column("MinF1", style="yellow", justify="right")
    table.add_column("Prec", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("t_on", justify="right")
    table.add_column("t_off", justify="right")
    table.add_column("pat", justify="right")
    table.add_column("sm", justify="right")
    table.add_column("min", justify="right")
    table.add_column("gap", justify="right")

    for i, r in enumerate(result.all_results[:top_n], 1):
        p = r["params"]
        table.add_row(
            str(i),
            f"{r['score']:.1%}",
            f"{r['f1']:.1%}",
            f"{r['min_video_f1']:.0%}",
            f"{r['precision']:.1%}",
            f"{r['recall']:.1%}",
            f"{p['t_on']:.2f}",
            f"{p['t_off']:.2f}",
            str(p["patience"]),
            str(p["smooth_window"]),
            str(p["min_segment_windows"]),
            str(p["max_gap_windows"]),
        )

    console.print(table)

    # Per-video breakdown for best configuration (use cached per_video data)
    console.print()
    console.print("[bold]Per-Video Results (Best Configuration):[/bold]")

    best_result = result.all_results[0]
    best_params = best_result["params"]
    per_video_data = best_result.get("per_video", [])

    per_video_table = Table()
    per_video_table.add_column("Video", style="cyan")
    per_video_table.add_column("GT", justify="right", style="dim")
    per_video_table.add_column("Det", justify="right", style="dim")
    per_video_table.add_column("Prec", justify="right")
    per_video_table.add_column("Recall", justify="right")
    per_video_table.add_column("F1", style="bold green", justify="right")

    for pv in per_video_data:
        # Truncate filename for display
        filename = pv["name"][:35] + "..." if len(pv["name"]) > 38 else pv["name"]
        # Highlight videos below threshold
        f1_style = "bold green"
        if min_video_f1 is not None and pv["f1"] < min_video_f1:
            f1_style = "bold red"
        elif pv["f1"] < 0.70:
            f1_style = "yellow"

        per_video_table.add_row(
            filename,
            str(pv["gt_count"]),
            str(pv["pred_count"]),
            f"{pv['precision']:.0%}",
            f"{pv['recall']:.0%}",
            f"[{f1_style}]{pv['f1']:.0%}[/{f1_style}]",
        )

    console.print(per_video_table)

    # Current defaults comparison
    console.print()
    console.print("[bold]Comparison with Current Defaults:[/bold]")

    # Evaluate current defaults
    current_config = DecoderConfig(stride=stride)
    total_tp_current = 0
    total_fp_current = 0
    total_fn_current = 0
    current_per_video_f1: list[float] = []

    for probs, gt, fps in zip(video_probs, video_ground_truths, video_fps):
        current_config.fps = fps
        decoder_result = decode(probs, current_config)
        metrics = compute_segment_metrics(gt, decoder_result.segments, iou_threshold)
        total_tp_current += int(metrics["tp"])
        total_fp_current += int(metrics["fp"])
        total_fn_current += int(metrics["fn"])
        current_per_video_f1.append(float(metrics["f1"]))

    precision_current = total_tp_current / (total_tp_current + total_fp_current) if (total_tp_current + total_fp_current) > 0 else 0
    recall_current = total_tp_current / (total_tp_current + total_fn_current) if (total_tp_current + total_fn_current) > 0 else 0
    f1_current = 2 * precision_current * recall_current / (precision_current + recall_current) if (precision_current + recall_current) > 0 else 0
    min_f1_current = min(current_per_video_f1) if current_per_video_f1 else 0

    console.print(f"  Current defaults: F1={f1_current:.1%} (P={precision_current:.1%}, R={recall_current:.1%}, MinF1={min_f1_current:.0%})")
    console.print(f"  Best found:       F1={result.best_f1:.1%} (P={result.best_precision:.1%}, R={result.best_recall:.1%}, MinF1={best_result['min_video_f1']:.0%})")

    improvement = result.best_f1 - f1_current
    if improvement > 0:
        console.print(f"  [green]Improvement: +{improvement:.1%}[/green]")
    elif improvement < 0:
        console.print(f"  [yellow]Regression: {improvement:.1%}[/yellow]")
    else:
        console.print("  No change")

    # Save full results
    export_data = {
        "iou_threshold": iou_threshold,
        "stride": stride,
        "num_videos": len(valid_videos),
        "min_video_f1_constraint": min_video_f1,
        "rejected_count": result.rejected_count,
        "current_defaults": {
            "f1": f1_current,
            "precision": precision_current,
            "recall": recall_current,
            "min_video_f1": min_f1_current,
            "params": {
                "smooth_window": current_config.smooth_window,
                "t_on": current_config.t_on,
                "t_off": current_config.t_off,
                "patience": current_config.patience,
                "min_segment_windows": current_config.min_segment_windows,
                "max_gap_windows": current_config.max_gap_windows,
            },
        },
        "best_config": {
            "f1": result.best_f1,
            "precision": result.best_precision,
            "recall": result.best_recall,
            "overmerge_rate": result.best_overmerge_rate,
            "min_video_f1": best_result["min_video_f1"],
            "params": best_params,
            "per_video": per_video_data,
        },
        "all_results": result.all_results[:100],  # Top 100 for analysis
    }
    output.write_text(json.dumps(export_data, indent=2))
    console.print()
    console.print(f"Full results saved to: [cyan]{output}[/cyan]")

    # Print best configuration for copy-paste
    console.print()
    console.print("[bold]Best configuration (for DecoderConfig defaults):[/bold]")
    console.print(f"  smooth_window = {best_params['smooth_window']}")
    console.print(f"  t_on = {best_params['t_on']}")
    console.print(f"  t_off = {best_params['t_off']}")
    console.print(f"  patience = {best_params['patience']}")
    console.print(f"  min_segment_windows = {best_params['min_segment_windows']}")
    console.print(f"  max_gap_windows = {best_params['max_gap_windows']}")
