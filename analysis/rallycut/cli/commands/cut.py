"""Cut command - remove dead time from volleyball videos."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from rallycut.cli.utils import format_time, handle_errors
from rallycut.core.cache import AnalysisCache
from rallycut.core.config import get_config
from rallycut.core.profiler import enable_profiling, get_profiler
from rallycut.core.video import Video
from rallycut.processing.cutter import VideoCutter

app = typer.Typer(help="Remove dead time from volleyball videos")
console = Console()


def _print_diagnostics(console: Console, diagnostic_data: dict, fps: float) -> None:
    """Print diagnostic information about ML classifications."""
    from rallycut.core.models import GameState

    raw_results = diagnostic_data["raw_results"]
    smoothed_results = diagnostic_data["smoothed_results"]
    raw_segments = diagnostic_data["raw_segments"]

    console.print()
    console.print("[bold cyan]═══ Diagnostic Report ═══[/bold cyan]")
    console.print()

    # Raw ML classification distribution
    console.print("[bold]Raw ML Classifications (before smoothing):[/bold]")
    state_counts = {GameState.PLAY: 0, GameState.SERVICE: 0, GameState.NO_PLAY: 0}
    confidence_sums = {GameState.PLAY: 0.0, GameState.SERVICE: 0.0, GameState.NO_PLAY: 0.0}

    for r in raw_results:
        state_counts[r.state] += 1
        confidence_sums[r.state] += r.confidence

    total = len(raw_results)
    for state, count in state_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        avg_conf = (confidence_sums[state] / count) if count > 0 else 0
        console.print(f"  {state.value:8s}: {count:3d} windows ({pct:5.1f}%), avg confidence: {avg_conf:.2f}")

    # Smoothing changes
    console.print()
    console.print("[bold]Temporal Smoothing Changes:[/bold]")
    changes = 0
    for raw, smooth in zip(raw_results, smoothed_results):
        if raw.state != smooth.state:
            changes += 1
    console.print(f"  Windows changed by smoothing: {changes} / {total}")

    # Smoothed distribution
    console.print()
    console.print("[bold]After Smoothing:[/bold]")
    smooth_counts = {GameState.PLAY: 0, GameState.SERVICE: 0, GameState.NO_PLAY: 0}
    for r in smoothed_results:
        smooth_counts[r.state] += 1

    for state, count in smooth_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        console.print(f"  {state.value:8s}: {count:3d} windows ({pct:5.1f}%)")

    # Raw segments (before min_duration filter)
    console.print()
    console.print("[bold]Segments Before Duration Filter:[/bold]")
    for i, seg in enumerate(raw_segments, 1):
        status = "[green]KEPT[/green]" if seg.duration >= 5.0 else "[red]FILTERED (<5s)[/red]"
        console.print(f"  {i}. {seg.start_time:6.1f}s - {seg.end_time:6.1f}s ({seg.duration:5.1f}s) {seg.state.value} {status}")

    # Timeline visualization
    console.print()
    console.print("[bold]Timeline (each char = ~1 second):[/bold]")

    # Build raw timeline
    duration = max(r.end_frame for r in raw_results) / fps if raw_results else 0
    raw_timeline = []
    smooth_timeline = []

    for sec in range(int(duration) + 1):
        # Find result covering this second
        raw_state = "-"
        smooth_state = "-"

        for r in raw_results:
            if r.start_frame / fps <= sec < r.end_frame / fps:
                if r.state == GameState.PLAY:
                    raw_state = "P"
                elif r.state == GameState.SERVICE:
                    raw_state = "S"
                else:
                    raw_state = "N"
                break

        for r in smoothed_results:
            if r.start_frame / fps <= sec < r.end_frame / fps:
                if r.state == GameState.PLAY:
                    smooth_state = "P"
                elif r.state == GameState.SERVICE:
                    smooth_state = "S"
                else:
                    smooth_state = "N"
                break

        raw_timeline.append(raw_state)
        smooth_timeline.append(smooth_state)

    # Print timeline with markers every 10 seconds
    console.print("  Time:    " + "".join(f"{i:<10d}" for i in range(0, len(raw_timeline), 10)))
    console.print("  Raw:     " + "".join(raw_timeline))
    console.print("  Smooth:  " + "".join(smooth_timeline))
    console.print()
    console.print("  Legend: P=PLAY, S=SERVICE, N=NO_PLAY, -=unknown")


@app.callback(invoke_without_command=True)
@handle_errors
def cut(  # noqa: C901
    video: Path = typer.Argument(
        ...,
        help="Path to input video file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Output video path (default: {video}_cut.mp4)",
    ),
    padding: float | None = typer.Option(
        None,
        "--padding", "-p",
        help="Seconds of padding before play segments (default: from config)",
    ),
    padding_end: float | None = typer.Option(
        None,
        "--padding-end",
        help="Seconds of padding after play segments (default: padding + 0.5s)",
    ),
    min_play: float | None = typer.Option(
        None,
        "--min-play",
        help="Minimum play duration in seconds to keep (default: from config)",
    ),
    stride: int | None = typer.Option(
        None,
        "--stride", "-s",
        help="Frames to skip between analyses (higher = faster, less accurate)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run", "-n",
        help="Only analyze and show segments, don't generate video",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--json",
        help="Output segments as JSON to file",
    ),
    input_segments: Path | None = typer.Option(
        None,
        "--segments",
        help="Load segments from JSON file (skip analysis)",
    ),
    gpu: bool | None = typer.Option(
        None,
        "--gpu/--no-gpu",
        help="Enable/disable GPU acceleration (auto-detected by default)",
    ),
    limit: float | None = typer.Option(
        None,
        "--limit", "-l",
        help="Only analyze first N seconds of video (for testing)",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Force re-analysis even if cached",
    ),
    proxy: bool = typer.Option(
        True,
        "--proxy/--no-proxy",
        help="Use 480p@30fps proxy for faster ML analysis (default: on)",
    ),
    min_gap: float | None = typer.Option(
        None,
        "--min-gap",
        help="Min NO_PLAY gap (seconds) before ending a rally (default: from config)",
    ),
    rally_continuation: float | None = typer.Option(
        None,
        "--rally-continuation",
        help="Seconds of consecutive NO_PLAY required to end a rally (default: from config)",
    ),
    auto_stride: bool = typer.Option(
        True,
        "--auto-stride/--no-auto-stride",
        help="Auto-adjust stride based on video FPS (stride 32 @ 30fps = stride 64 @ 60fps)",
    ),
    profile: bool = typer.Option(
        False,
        "--profile",
        help="Enable profiling and show timing breakdown after analysis",
    ),
    profile_json: Path | None = typer.Option(
        None,
        "--profile-json",
        help="Export profiling results to JSON file",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show detailed ML classification diagnostics",
    ),
    model: str = typer.Option(
        "indoor",
        "--model", "-m",
        help="Model variant: 'indoor' (original) or 'beach' (fine-tuned with beach heuristics)",
    ),
    heuristics: bool = typer.Option(
        False,
        "--heuristics",
        help="Force heuristics pipeline (57%% F1) instead of binary head",
    ),
    experimental_temporal: bool = typer.Option(
        False,
        "--experimental-temporal",
        help="[DEPRECATED] Use temporal model for post-processing (use binary head instead)",
    ),
    temporal_model_path: Path | None = typer.Option(
        None,
        "--temporal-model",
        help="Path to temporal model weights (default: weights/temporal/best_temporal_model.pt)",
    ),
    temporal_version: str = typer.Option(
        "v2",
        "--temporal-version",
        help="Temporal model version: v1 (smoothing), v2 (ConvCRF), v3 (BiLSTM)",
    ),
    binary_head: bool = typer.Option(
        False,
        "--binary-head",
        help="Force binary head + decoder (70%% F1). Auto-enabled when features cached.",
    ),
    binary_head_model: Path | None = typer.Option(
        None,
        "--binary-head-model",
        help="Path to binary head model (default: weights/binary_head/best_binary_head.pt)",
    ),
    refine: bool = typer.Option(
        False,
        "--refine",
        help="[EXPERIMENTAL] Enable boundary refinement using fine-stride features",
    ),
) -> None:
    """
    Automatically remove no-play segments from volleyball recordings.

    Analyzes the video to detect SERVICE, PLAY, and NO_PLAY segments,
    then exports a new video containing only the play segments with
    configurable padding.

    Examples:
        rallycut cut match.mp4                     # Full processing
        rallycut cut match.mp4 --dry-run           # Analysis only
        rallycut cut match.mp4 -n --stride 32      # Fast analysis
        rallycut cut match.mp4 --json segments.json
    """
    # Determine output path
    if output is None:
        output = video.parent / f"{video.stem}_cut{video.suffix}"

    # Determine device
    config = get_config()
    if gpu is True:
        device = "cuda" if config.device == "cuda" else "mps"
    elif gpu is False:
        device = "cpu"
    else:
        device = config.device

    # Resolve None parameters from config
    effective_padding = padding if padding is not None else config.segment.padding_seconds
    effective_min_play = min_play if min_play is not None else config.segment.min_play_duration
    effective_stride_base = stride if stride is not None else config.game_state.stride
    effective_min_gap = min_gap if min_gap is not None else config.segment.min_gap_seconds

    console.print("\n[bold]RallyCut[/bold] - Dead Time Removal")
    console.print(f"Input: [cyan]{video}[/cyan]")
    if not dry_run:
        console.print(f"Output: [cyan]{output}[/cyan]")
    console.print(f"Device: [yellow]{device}[/yellow]")
    console.print(f"Model: [yellow]{model}[/yellow]")
    mode_str = "Analysis only" if dry_run else "Full processing"
    console.print(f"Mode: [yellow]{mode_str}[/yellow]")

    # Get video info first to calculate effective stride
    with Video(video) as v:
        video_info = v.info

    # Calculate effective stride (FPS-normalized)
    reference_fps = 30.0
    if auto_stride:
        effective_stride = int(round(effective_stride_base * (video_info.fps / reference_fps)))
        effective_stride = max(1, effective_stride)
        if effective_stride != effective_stride_base:
            console.print(f"Stride: [yellow]{effective_stride_base} → {effective_stride} frames[/yellow] (auto-adjusted for {video_info.fps:.0f}fps)")
        else:
            console.print(f"Stride: [yellow]{effective_stride_base} frames[/yellow]")
    else:
        effective_stride = effective_stride_base
        console.print(f"Stride: [yellow]{effective_stride_base} frames[/yellow]")

    if limit:
        console.print(f"Limit: [yellow]{format_time(limit)} ({limit:.1f}s)[/yellow]")
    console.print()

    console.print(f"Duration: {format_time(video_info.duration)} ({video_info.duration:.1f}s)")
    console.print(f"Resolution: {video_info.width}x{video_info.height}")
    console.print(f"FPS: {video_info.fps:.1f}")

    # Estimate analysis windows using effective stride
    analyze_frames = video_info.frame_count
    if limit:
        analyze_frames = min(analyze_frames, int(limit * video_info.fps))
    total_windows = (analyze_frames - 16) // effective_stride + 1
    console.print(f"Windows to analyze: ~{total_windows}")
    console.print()

    # Enable profiling if requested
    if profile or profile_json:
        profiler = enable_profiling()
        console.print("[dim]Profiling enabled[/dim]")
        console.print()

    # Show deprecation warnings
    if experimental_temporal:
        import warnings

        warnings.warn(
            "--experimental-temporal is deprecated and will be removed in a future release. "
            "Use binary head + decoder instead (70% F1 vs 65% F1).",
            DeprecationWarning,
            stacklevel=2,
        )
        console.print(
            "[yellow]Warning: --experimental-temporal is deprecated. "
            "Binary head + decoder achieves 70% F1 vs temporal model's 65% F1.[/yellow]"
        )

    # Create cutter
    cutter = VideoCutter(
        device=device,
        padding_seconds=effective_padding,
        padding_end_seconds=padding_end,
        min_play_duration=effective_min_play,
        stride=effective_stride_base,
        limit_seconds=limit,
        use_proxy=proxy,
        min_gap_seconds=effective_min_gap,
        auto_stride=auto_stride,
        rally_continuation_seconds=rally_continuation,
        model_variant=model,
        use_temporal_model=experimental_temporal,
        temporal_model_path=temporal_model_path,
        temporal_model_version=temporal_version,
        use_binary_head_decoder=binary_head,
        binary_head_model_path=binary_head_model,
        use_heuristics=heuristics,
        boundary_refinement=refine,
    )

    # Show pipeline info
    if binary_head:
        console.print("[bold green]Binary head + decoder enabled (70% F1)[/bold green]")
    elif experimental_temporal:
        console.print(
            f"[yellow]Temporal model enabled (deprecated):[/yellow] {temporal_version}"
        )
    elif heuristics:
        console.print("[dim]Heuristics pipeline (57% F1)[/dim]")
    else:
        console.print("[dim]Auto-selecting pipeline based on cached features...[/dim]")

    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=100)

        def update_progress(pct: float, msg: str) -> None:
            progress.update(task, completed=int(pct * 100), description=msg)

        # Initialize diagnostic_data (may be populated in debug mode)
        diagnostic_data = None

        if input_segments:
            # Load segments from JSON file
            progress.update(task, completed=50, description="Loading segments...")
            with open(input_segments) as f:
                data = json.load(f)

            from rallycut.core.models import GameState, TimeSegment

            raw_rallies = data["rallies"]
            segments = []

            for rally in raw_rallies:
                start_time = float(rally["start_time"])
                end_time = float(rally["end_time"])

                segments.append(TimeSegment(
                    start_frame=int(start_time * video_info.fps),
                    end_frame=int(end_time * video_info.fps),
                    start_time=start_time,
                    end_time=end_time,
                    state=GameState.PLAY,
                ))

            if not dry_run:
                # Export with loaded segments
                from rallycut.processing.exporter import FFmpegExporter
                exporter = FFmpegExporter()
                exporter.export_segments(
                    video, output, segments,
                    progress_callback=lambda p, m: progress.update(task, completed=int(50 + p * 50), description=m),
                )
            progress.update(task, completed=100, description="Complete!")
        else:
            # Check cache first (only if not using --limit)
            cache = AnalysisCache()
            segments = None

            if not no_cache and limit is None and not debug:
                segments = cache.get(video, effective_stride_base, proxy)
                if segments:
                    console.print("[dim]Using cached analysis[/dim]")

            if segments is not None:
                # Use cached segments
                if not dry_run:
                    from rallycut.processing.exporter import FFmpegExporter
                    exporter = FFmpegExporter()
                    exporter.export_segments(
                        video, output, segments,
                        progress_callback=lambda p, m: progress.update(task, completed=int(p * 100), description=m),
                    )
            elif debug:
                # Debug mode - get full diagnostic data
                diagnostic_data = cutter.analyze_with_diagnostics(
                    video,
                    progress_callback=update_progress,
                )
                segments = diagnostic_data["final_segments"]
                # Don't cache diagnostic results
            elif dry_run:
                # Analysis only
                segments, _suggested = cutter.analyze_only(
                    video,
                    progress_callback=update_progress,
                )
                # Cache results (only if not using --limit)
                if limit is None and segments:
                    cache.set(video, effective_stride_base, segments, proxy)
            else:
                # Full processing
                output_path, segments = cutter.cut_video(
                    video,
                    output,
                    progress_callback=update_progress,
                )
                # Cache results (only if not using --limit)
                if limit is None and segments:
                    cache.set(video, effective_stride_base, segments, proxy)

        progress.update(task, completed=100, description="Complete!")

    # Ensure segments is not None for stats calculation
    if segments is None:
        segments = []

    # Print statistics
    stats = cutter.get_cut_stats(video_info.duration, segments)

    console.print()
    console.print("[bold green]Analysis Complete![/bold green]")
    console.print()

    # Show segments table
    if segments:
        table = Table(title="Play Segments")
        table.add_column("#", style="dim")
        table.add_column("Start", style="cyan")
        table.add_column("End", style="cyan")
        table.add_column("Duration", style="green")
        table.add_column("State", style="yellow")

        for i, seg in enumerate(segments, 1):
            table.add_row(
                str(i),
                format_time(seg.start_time),
                format_time(seg.end_time),
                f"{seg.duration:.1f}s",
                seg.state.value,
            )

        console.print(table)
        console.print()

    console.print("[bold]Statistics:[/bold]")
    console.print(f"  Original duration: {format_time(stats['original_duration'])}")
    console.print(f"  Play duration:     {format_time(stats['kept_duration'])}")
    console.print(f"  Dead time:         {format_time(stats['removed_duration'])} ({stats['removed_percentage']:.1f}%)")
    console.print(f"  Segments found:    {stats['segment_count']}")

    # Print diagnostic output if debug mode
    if debug and diagnostic_data:
        _print_diagnostics(console, diagnostic_data, video_info.fps)

    # Output JSON if requested
    if output_json:
        # Build rally list with unique IDs and metadata for frontend video editor
        rallies = []
        for i, seg in enumerate(segments, 1):
            rally_id = f"rally_{i}"
            # Calculate thumbnail timestamp (midpoint of segment)
            thumbnail_time = seg.start_time + (seg.duration / 2)

            rallies.append({
                "id": rally_id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "start_frame": seg.start_frame,
                "end_frame": seg.end_frame,
                "duration": seg.duration,
                "type": "rally",
                "thumbnail_time": thumbnail_time,
            })

        json_data = {
            "version": "1.0",
            "video": {
                "path": str(video),
                "duration": video_info.duration,
                "fps": video_info.fps,
                "width": video_info.width,
                "height": video_info.height,
                "frame_count": video_info.frame_count,
            },
            "rallies": rallies,
            "stats": stats,
        }
        with open(output_json, "w") as f:
            json.dump(json_data, f, indent=2)
        console.print(f"\nSegments saved to: [cyan]{output_json}[/cyan]")

    if not dry_run:
        console.print(f"\nOutput saved to: [cyan]{output}[/cyan]")

    # Print profiling report if enabled
    if profile or profile_json:
        profiler = get_profiler()
        console.print()
        console.print("[bold]Profiling Results:[/bold]")
        profiler.print_stages_report()

        # Also print component breakdown
        profiler.print_report()

        # Export to JSON if requested
        if profile_json:
            profiler.export_json(profile_json)
            console.print(f"\nProfile saved to: [cyan]{profile_json}[/cyan]")
