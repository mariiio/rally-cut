"""Seamless labeling workflow with Label Studio integration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from rallycut.cli.utils import handle_errors, validate_video_file

console = Console()
app = typer.Typer(help="Ground truth labeling commands")


def _get_video_info(video_path: Path) -> tuple[int, int, float, int]:
    """Get video dimensions, FPS, and frame count."""
    import json

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 1920, 1080, 30.0, 0

    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width", 1920)
            height = stream.get("height", 1080)
            fps_str = stream.get("r_frame_rate", "30/1")
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
            frames = int(stream.get("nb_frames", 0))
            return width, height, fps, frames

    return 1920, 1080, 30.0, 0


def _ensure_label_studio_running() -> bool:
    """Check if Label Studio is running, offer to start if not."""
    from rallycut.labeling.studio_client import LabelStudioClient

    client = LabelStudioClient()
    if client.is_running():
        return True

    console.print("[yellow]Label Studio is not running.[/yellow]")
    console.print("Start it with: python3 -m label_studio start")
    return False


def _get_api_key() -> str | None:
    """Get API key from environment or prompt user."""
    import os

    key = os.environ.get("LABEL_STUDIO_API_KEY")
    if key:
        return key

    console.print("\n[yellow]API key not found.[/yellow]")
    console.print("Get your API key from Label Studio:")
    console.print("  1. Open http://localhost:8080")
    console.print("  2. Click your profile → Account & Settings")
    console.print("  3. Copy the Access Token")
    console.print("\nThen set it:")
    console.print("  export LABEL_STUDIO_API_KEY=your_token_here")
    return None


@app.command(name="open")
@handle_errors
def label_open(
    video_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Video file to label",
    ),
    predictions: Path = typer.Option(
        None,
        "--predictions", "-p",
        help="Tracking predictions JSON (pre-fills annotations)",
    ),
    start_ms: int = typer.Option(
        0,
        "--start", "-s",
        help="Start time in milliseconds",
    ),
) -> None:
    """Open video in Label Studio for annotation.

    Creates a labeling task with pre-filled predictions (if provided)
    and opens it in your browser. After labeling, use 'rallycut label save'
    to export as ground truth.

    Example:
        rallycut label open video.mp4 -p tracking.json
    """
    validate_video_file(video_path)

    if not _ensure_label_studio_running():
        raise typer.Exit(1)

    api_key = _get_api_key()
    if not api_key:
        raise typer.Exit(1)

    from rallycut.labeling.studio_client import LabelStudioClient, build_predictions
    from rallycut.tracking.player_tracker import PlayerTrackingResult

    # Get video info
    video_width, video_height, fps, _ = _get_video_info(video_path)
    frame_offset = int(start_ms / 1000 * fps)

    # Load predictions if provided
    player_positions = []
    ball_positions = None

    if predictions and predictions.exists():
        console.print(f"[dim]Loading predictions from {predictions}...[/dim]")
        result = PlayerTrackingResult.from_json(predictions)
        player_positions = result.positions
        ball_positions = result.ball_positions if result.ball_positions else None

    # Build Label Studio predictions
    ls_predictions = build_predictions(
        player_positions=player_positions,
        ball_positions=ball_positions,
        frame_offset=frame_offset,
        fps=fps,
    )

    console.print("[bold]Setting up Label Studio task...[/bold]")

    client = LabelStudioClient(api_key=api_key)

    # Get or create project
    project_id = client.get_or_create_project()
    console.print(f"  Project: RallyCut Ground Truth (ID: {project_id})")

    # Create task with video path as file:// URL
    video_url = f"file://{video_path.absolute()}"
    task_id = client.create_task(
        project_id=project_id,
        video_path=video_url,
        predictions=ls_predictions if ls_predictions else None,
    )
    console.print(f"  Task created: ID {task_id}")

    if player_positions:
        console.print(f"  Pre-filled: {len(player_positions)} player detections")

    # Open in browser
    console.print("\n[green]Opening in browser...[/green]")
    client.open_task(project_id, task_id)

    console.print(Panel(
        f"[bold]Task ID: {task_id}[/bold]\n\n"
        "1. Correct the bounding boxes in Label Studio\n"
        "2. Click 'Submit' when done\n"
        f"3. Run: rallycut label save {task_id} -o ground_truth.json",
        title="Labeling Started",
    ))

    client.close()


@app.command(name="save")
@handle_errors
def label_save(
    task_id: int = typer.Argument(
        ...,
        help="Label Studio task ID (shown after 'label open')",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output ground truth JSON file",
    ),
    video_width: int = typer.Option(
        1920,
        "--width", "-w",
        help="Video frame width",
    ),
    video_height: int = typer.Option(
        1080,
        "--height", "-h",
        help="Video frame height",
    ),
) -> None:
    """Save completed annotations as ground truth.

    Fetches annotations from Label Studio and saves as ground truth JSON
    for use with 'rallycut compare-tracking'.

    Example:
        rallycut label save 123 -o ground_truth.json
    """
    api_key = _get_api_key()
    if not api_key:
        raise typer.Exit(1)

    from rallycut.labeling.ground_truth import GroundTruthPosition, GroundTruthResult
    from rallycut.labeling.studio_client import LabelStudioClient

    if output is None:
        output = Path(f"ground_truth_task_{task_id}.json")

    console.print(f"[bold]Fetching annotations for task {task_id}...[/bold]")

    client = LabelStudioClient(api_key=api_key)
    annotations = client.get_task_annotations(task_id)
    client.close()

    if not annotations:
        console.print("[yellow]No annotations found. Did you submit the task?[/yellow]")
        raise typer.Exit(1)

    # Convert Label Studio format to ground truth
    positions: list[GroundTruthPosition] = []
    max_frame = 0

    for result in annotations:
        if result.get("type") != "videorectangle":
            continue

        value = result.get("value", {})
        sequence = value.get("sequence", [])
        labels = value.get("labels", ["player"])
        label = labels[0] if labels else "player"

        # Extract track ID from result ID
        result_id = result.get("id", "0")
        try:
            track_id = int(result_id.split("_")[-1])
        except (ValueError, IndexError):
            track_id = hash(result_id) % 1000

        for keyframe in sequence:
            if not keyframe.get("enabled", True):
                continue

            frame = keyframe.get("frame", 1) - 1  # Convert to 0-indexed

            # Convert percentage to normalized
            x_pct = keyframe.get("x", 0)
            y_pct = keyframe.get("y", 0)
            w_pct = keyframe.get("width", 5)
            h_pct = keyframe.get("height", 10)

            x = (x_pct + w_pct / 2) / 100
            y = (y_pct + h_pct / 2) / 100
            width = w_pct / 100
            height = h_pct / 100

            positions.append(GroundTruthPosition(
                frame_number=frame,
                track_id=track_id,
                label=label,
                x=x,
                y=y,
                width=width,
                height=height,
            ))

            max_frame = max(max_frame, frame)

    gt_result = GroundTruthResult(
        positions=positions,
        frame_count=max_frame + 1,
        video_width=video_width,
        video_height=video_height,
    )

    gt_result.to_json(output)

    player_count = len(gt_result.player_positions)
    ball_count = len(gt_result.ball_positions)
    unique_tracks = gt_result.unique_player_tracks

    console.print(f"\n[green]Ground truth saved: {output}[/green]")
    console.print(f"  Players: {player_count} annotations ({len(unique_tracks)} tracks)")
    console.print(f"  Ball: {ball_count} annotations")
    console.print(f"  Frames: 0 - {max_frame}")


@app.command(name="list")
@handle_errors
def label_list() -> None:
    """List labeling tasks in Label Studio."""
    api_key = _get_api_key()
    if not api_key:
        raise typer.Exit(1)

    from rallycut.labeling.studio_client import LabelStudioClient

    client = LabelStudioClient(api_key=api_key)

    try:
        project_id = client.get_or_create_project()
        tasks = client.get_project_tasks(project_id)
    finally:
        client.close()

    if not tasks:
        console.print("[dim]No tasks found.[/dim]")
        return

    console.print("[bold]Tasks in RallyCut Ground Truth project:[/bold]\n")

    for task in tasks:
        task_id = task.get("id")
        annotations = task.get("annotations", [])
        status = "[green]Completed[/green]" if annotations else "[yellow]Pending[/yellow]"
        video = task.get("data", {}).get("video", "unknown")

        # Truncate video path
        if len(video) > 50:
            video = "..." + video[-47:]

        console.print(f"  {task_id}: {status} - {video}")
