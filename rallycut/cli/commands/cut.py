"""Cut command - remove dead time from volleyball videos."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from rallycut.cli.utils import handle_errors
from rallycut.core.cache import AnalysisCache
from rallycut.core.config import get_config
from rallycut.core.video import Video
from rallycut.processing.cutter import VideoCutter

app = typer.Typer(help="Remove dead time from volleyball videos")
console = Console()


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes}:{secs:05.2f}"


def parse_time(time_str: str) -> float:
    """Parse time string like '0:12', '1:30', '1:02:30' to seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        # MM:SS
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        # Assume it's already seconds
        return float(time_str)


@app.callback(invoke_without_command=True)
@handle_errors
def cut(
    video: Path = typer.Argument(
        ...,
        help="Path to input video file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output video path (default: {video}_cut.mp4)",
    ),
    padding: float = typer.Option(
        1.0,
        "--padding", "-p",
        help="Seconds of padding before play segments",
    ),
    padding_end: Optional[float] = typer.Option(
        None,
        "--padding-end",
        help="Seconds of padding after play segments (default: padding + 0.5s)",
    ),
    min_play: float = typer.Option(
        5.0,
        "--min-play",
        help="Minimum play duration in seconds to keep",
    ),
    stride: int = typer.Option(
        32,
        "--stride", "-s",
        help="Frames to skip between analyses (higher = faster, less accurate)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run", "-n",
        help="Only analyze and show segments, don't generate video",
    ),
    quick: bool = typer.Option(
        False,
        "--quick", "-q",
        help="Use fast motion detection instead of ML model (less accurate)",
    ),
    output_json: Optional[Path] = typer.Option(
        None,
        "--json",
        help="Output segments as JSON to file",
    ),
    input_segments: Optional[Path] = typer.Option(
        None,
        "--segments",
        help="Load segments from JSON file (skip analysis)",
    ),
    gpu: Optional[bool] = typer.Option(
        None,
        "--gpu/--no-gpu",
        help="Enable/disable GPU acceleration (auto-detected by default)",
    ),
    limit: Optional[float] = typer.Option(
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
        help="Use 480p@15fps proxy for 2-4x faster ML analysis (default: on)",
    ),
    two_pass: bool = typer.Option(
        True,
        "--two-pass/--full-scan",
        help="Use two-pass: motion scan then ML on motion only (2-3x faster)",
    ),
    min_gap: float = typer.Option(
        3.0,
        "--min-gap",
        help="Min NO_PLAY gap (seconds) before ending a rally (higher = longer rallies)",
    ),
    auto_stride: bool = typer.Option(
        True,
        "--auto-stride/--no-auto-stride",
        help="Auto-adjust stride based on video FPS (stride 32 @ 30fps = stride 64 @ 60fps)",
    ),
):
    """
    Automatically remove no-play segments from beach volleyball recordings.

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

    console.print(f"\n[bold]RallyCut[/bold] - Dead Time Removal")
    console.print(f"Input: [cyan]{video}[/cyan]")
    if not dry_run:
        console.print(f"Output: [cyan]{output}[/cyan]")
    console.print(f"Device: [yellow]{device}[/yellow]")
    mode_str = "Analysis only" if dry_run else "Full processing"
    if quick:
        mode_str += " (quick motion detection)"
    console.print(f"Mode: [yellow]{mode_str}[/yellow]")

    # Get video info first to calculate effective stride
    with Video(video) as v:
        video_info = v.info

    # Calculate effective stride (FPS-normalized)
    reference_fps = 30.0
    if auto_stride:
        effective_stride = int(round(stride * (video_info.fps / reference_fps)))
        effective_stride = max(1, effective_stride)
        if effective_stride != stride:
            console.print(f"Stride: [yellow]{stride} → {effective_stride} frames[/yellow] (auto-adjusted for {video_info.fps:.0f}fps)")
        else:
            console.print(f"Stride: [yellow]{stride} frames[/yellow]")
    else:
        effective_stride = stride
        console.print(f"Stride: [yellow]{stride} frames[/yellow]")

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

    # Create cutter
    cutter = VideoCutter(
        device=device,
        padding_seconds=padding,
        padding_end_seconds=padding_end,
        min_play_duration=min_play,
        stride=stride,
        use_quick_mode=quick,
        use_two_pass=two_pass and not quick,  # Two-pass only when not quick mode
        limit_seconds=limit,
        use_proxy=proxy,  # Proxy handled by TwoPassAnalyzer
        min_gap_seconds=min_gap,
        auto_stride=auto_stride,
    )

    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=100)

        def update_progress(pct: float, msg: str):
            progress.update(task, completed=int(pct * 100), description=msg)

        if input_segments:
            # Load segments from JSON file
            progress.update(task, completed=50, description="Loading segments...")
            with open(input_segments) as f:
                data = json.load(f)

            from rallycut.core.models import GameState, TimeSegment

            raw_segments = data if isinstance(data, list) else data.get("segments", [])
            segments = []

            for s in raw_segments:
                # Support multiple formats:
                # 1. Simple: "0:12-0:19" or "0:12,0:19"
                # 2. Object with time strings: {"start": "0:12", "end": "0:19"}
                # 3. Object with seconds: {"start_time": 12, "end_time": 19}

                if isinstance(s, str):
                    # Parse "0:12-0:19" or "0:12,0:19" format
                    sep = "-" if "-" in s else ","
                    start_str, end_str = s.split(sep)
                    start_time = parse_time(start_str)
                    end_time = parse_time(end_str)
                elif "start" in s:
                    # {"start": "0:12", "end": "0:19"} format
                    start_time = parse_time(str(s["start"]))
                    end_time = parse_time(str(s["end"]))
                else:
                    # {"start_time": 12, "end_time": 19} format
                    start_time = float(s["start_time"])
                    end_time = float(s["end_time"])

                segments.append(TimeSegment(
                    start_frame=int(start_time * video_info.fps),
                    end_frame=int(end_time * video_info.fps),
                    start_time=start_time,
                    end_time=end_time,
                    state=GameState(s.get("state", "play") if isinstance(s, dict) else "play"),
                ))

            if not dry_run:
                # Export with loaded segments
                from rallycut.processing.exporter import FFmpegExporter
                exporter = FFmpegExporter()
                exporter.export_segments(
                    video, output, segments,
                    progress_callback=lambda p, m: progress.update(task, completed=int(50 + p * 50), description=m)
                )
            progress.update(task, completed=100, description="Complete!")
        else:
            # Check cache first (only if not using --limit)
            cache = AnalysisCache()
            segments = None

            if not no_cache and limit is None:
                segments = cache.get(video, stride, quick, proxy)
                if segments:
                    console.print("[dim]Using cached analysis[/dim]")

            if segments is not None:
                # Use cached segments
                if not dry_run:
                    from rallycut.processing.exporter import FFmpegExporter
                    exporter = FFmpegExporter()
                    exporter.export_segments(
                        video, output, segments,
                        progress_callback=lambda p, m: progress.update(task, completed=int(p * 100), description=m)
                    )
            elif dry_run:
                # Analysis only
                segments = cutter.analyze_only(
                    video,
                    progress_callback=update_progress,
                )
                # Cache results (only if not using --limit)
                if limit is None and segments:
                    cache.set(video, stride, quick, segments, proxy)
            else:
                # Full processing
                output_path, segments = cutter.cut_video(
                    video,
                    output,
                    progress_callback=update_progress,
                )
                # Cache results (only if not using --limit)
                if limit is None and segments:
                    cache.set(video, stride, quick, segments, proxy)

        progress.update(task, completed=100, description="Complete!")

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

    # Output JSON if requested
    if output_json:
        json_data = {
            "video": str(video),
            "duration": video_info.duration,
            "fps": video_info.fps,
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "duration": seg.duration,
                    "state": seg.state.value,
                }
                for seg in segments
            ],
            "stats": stats,
        }
        with open(output_json, "w") as f:
            json.dump(json_data, f, indent=2)
        console.print(f"\nSegments saved to: [cyan]{output_json}[/cyan]")

    if not dry_run:
        console.print(f"\nOutput saved to: [cyan]{output}[/cyan]")
