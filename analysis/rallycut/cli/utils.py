"""CLI utilities for RallyCut."""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.console import Console

console = Console()


class RallyCutError(Exception):
    """Base exception for RallyCut errors."""

    def __init__(self, message: str, hint: str | None = None):
        self.message = message
        self.hint = hint
        super().__init__(message)


class VideoError(RallyCutError):
    """Error related to video file."""
    pass


class ExportError(RallyCutError):
    """Error related to video export."""
    pass


F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(func: F) -> F:
    """Decorator to handle common errors in CLI commands."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
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

    return wrapper  # type: ignore[return-value]


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


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes}:{secs:05.2f}"
