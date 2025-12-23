"""CLI utilities for RallyCut."""

import functools
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()


class RallyCutError(Exception):
    """Base exception for RallyCut errors."""

    def __init__(self, message: str, hint: Optional[str] = None):
        self.message = message
        self.hint = hint
        super().__init__(message)


class VideoError(RallyCutError):
    """Error related to video file."""
    pass


class ModelError(RallyCutError):
    """Error related to ML models."""
    pass


class ExportError(RallyCutError):
    """Error related to video export."""
    pass


def handle_errors(func):
    """Decorator to handle common errors in CLI commands."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RallyCutError as e:
            console.print(f"\n[red]Error:[/red] {e.message}")
            if e.hint:
                console.print(f"[dim]Hint: {e.hint}[/dim]")
            raise typer.Exit(1)
        except FileNotFoundError as e:
            console.print(f"\n[red]Error:[/red] File not found: {e.filename}")
            raise typer.Exit(1)
        except PermissionError as e:
            console.print(f"\n[red]Error:[/red] Permission denied: {e.filename}")
            console.print("[dim]Hint: Check file permissions or try a different output path[/dim]")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            raise typer.Exit(130)
        except MemoryError:
            console.print("\n[red]Error:[/red] Out of memory")
            console.print("[dim]Hint: Try using --stride with a higher value or --limit to process less video[/dim]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]Unexpected error:[/red] {type(e).__name__}: {e}")
            console.print("[dim]Please report this issue at https://github.com/anthropics/rallycut/issues[/dim]")
            raise typer.Exit(1)

    return wrapper


def validate_video_file(path: Path) -> None:
    """Validate that a video file exists and is readable."""
    if not path.exists():
        raise VideoError(
            f"Video file not found: {path}",
            hint="Check the file path and try again"
        )

    if not path.is_file():
        raise VideoError(
            f"Not a file: {path}",
            hint="Provide a path to a video file, not a directory"
        )

    # Check extension
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
    if path.suffix.lower() not in video_extensions:
        raise VideoError(
            f"Unsupported video format: {path.suffix}",
            hint=f"Supported formats: {', '.join(video_extensions)}"
        )


def validate_output_path(path: Path, overwrite: bool = True) -> None:
    """Validate that output path is writable."""
    # Check parent directory exists
    if not path.parent.exists():
        raise ExportError(
            f"Output directory does not exist: {path.parent}",
            hint="Create the directory first or use a different path"
        )

    # Check if file exists and we're not overwriting
    if path.exists() and not overwrite:
        raise ExportError(
            f"Output file already exists: {path}",
            hint="Use a different output path or delete the existing file"
        )


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def check_models_available() -> dict[str, bool]:
    """Check which models are available."""
    from rallycut.core.config import get_config
    config = get_config()

    return {
        "videomae": config.videomae_model_path is not None and config.videomae_model_path.exists(),
        "action_detector": config.action_detector_path is not None and config.action_detector_path.exists(),
        "ball_detector": config.ball_detector_path is not None and config.ball_detector_path.exists(),
    }


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes}:{secs:05.2f}"
