"""Track players command - detect and track player positions in volleyball videos."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from rallycut.cli.utils import handle_errors, validate_video_file
from rallycut.tracking.ball_features import detect_ball_phases_v2, detect_server
from rallycut.tracking.ball_tracker import (
    DEFAULT_BALL_MODEL,
    BallPosition,
    BallTracker,
    get_available_ball_models,
)
from rallycut.tracking.player_filter import PlayerFilterConfig
from rallycut.tracking.player_tracker import (
    DEFAULT_TRACKER,
    DEFAULT_YOLO_MODEL,
    PREPROCESSING_CLAHE,
    PREPROCESSING_NONE,
    TRACKER_BOTSORT,
    TRACKER_BYTETRACK,
    YOLO_MODELS,
    BallPhaseInfo,
    PlayerPosition,
    PlayerTracker,
    PlayerTrackingResult,
    ServerInfo,
)

console = Console()


# Colors for different track IDs (BGR format for OpenCV)
TRACK_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 165, 255),  # Orange
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (0, 128, 255),  # Light orange
]


def render_debug_overlay(
    frame: np.ndarray,
    frame_number: int,
    players: list[PlayerPosition],
    ball_positions: list[BallPosition] | None,
    split_y: float | None,
    primary_tracks: set[int],
) -> np.ndarray:
    """Render debug overlay showing two-team split and player positions.

    Camera is always behind baseline, so teams split by horizontal line (Y axis).
    Near team = y > split_y (closer to camera), Far team = y <= split_y.

    Args:
        frame: Video frame (BGR).
        frame_number: Current frame number.
        players: Player positions for this frame.
        ball_positions: All ball positions (for drawing trajectory).
        split_y: Y-coordinate that splits near/far teams (0-1 normalized).
        primary_tracks: Set of primary (stable) track IDs.

    Returns:
        Frame with overlay drawn.
    """
    height, width = frame.shape[:2]
    overlay = frame.copy()

    # Draw court split line (horizontal)
    if split_y is not None:
        y_px = int(split_y * height)
        cv2.line(overlay, (0, y_px), (width, y_px), (0, 255, 255), 2)
        cv2.putText(
            overlay, "FAR", (10, y_px - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.putText(
            overlay, "NEAR", (10, y_px + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

    # Draw ball positions (last 30 frames as trajectory)
    if ball_positions:
        recent_balls = [
            bp for bp in ball_positions
            if frame_number - 30 <= bp.frame_number <= frame_number
            and bp.confidence >= 0.35
        ]
        for bp in recent_balls:
            x_px = int(bp.x * width)
            y_px = int(bp.y * height)
            # Fade based on age
            age = frame_number - bp.frame_number
            alpha = max(0.3, 1.0 - age / 30)
            radius = 5 if age > 0 else 8
            color = (0, int(255 * alpha), int(255 * alpha))  # Yellow fading
            cv2.circle(overlay, (x_px, y_px), radius, color, -1)

    # Draw player bounding boxes with track IDs
    for player in players:
        # Convert normalized coords to pixels
        x1 = int((player.x - player.width / 2) * width)
        y1 = int((player.y - player.height / 2) * height)
        x2 = int((player.x + player.width / 2) * width)
        y2 = int((player.y + player.height / 2) * height)

        # Color based on track ID
        color_idx = player.track_id % len(TRACK_COLORS) if player.track_id >= 0 else 0
        color = TRACK_COLORS[color_idx]

        # Thicker box for primary tracks
        thickness = 3 if player.track_id in primary_tracks else 2

        # Draw bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # Track ID label
        label = f"#{player.track_id}"
        if player.track_id in primary_tracks:
            label += "*"  # Mark primary tracks

        # Determine which side (for debug)
        if split_y is not None:
            side = "F" if player.y <= split_y else "N"
            label += f" ({side})"

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            overlay, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # Frame info
    info_text = f"Frame: {frame_number} | Players: {len(players)}"
    cv2.putText(
        overlay, info_text, (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    return overlay


def create_debug_video(
    video_path: Path,
    output_path: Path,
    tracking_result: PlayerTrackingResult,
    ball_positions: list[BallPosition] | None,
    split_y: float | None,
    primary_tracks: set[int],
    start_ms: int | None = None,
    end_ms: int | None = None,
    stride: int = 1,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    """Create debug video with two-team overlay.

    Args:
        video_path: Input video path.
        output_path: Output video path.
        tracking_result: Player tracking results.
        ball_positions: Ball positions for trajectory.
        split_y: Y-coordinate for court split (horizontal line).
        primary_tracks: Primary track IDs.
        start_ms: Start time in ms.
        end_ms: End time in ms.
        stride: Frame stride used during tracking.
        progress_callback: Progress callback function.
    """
    # Group positions by frame
    positions_by_frame: dict[int, list[PlayerPosition]] = {}
    for p in tracking_result.positions:
        if p.frame_number not in positions_by_frame:
            positions_by_frame[p.frame_number] = []
        positions_by_frame[p.frame_number].append(p)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame range
    start_frame = int(start_ms / 1000 * fps) if start_ms else 0
    end_frame = int(end_ms / 1000 * fps) if end_ms else total_frames

    # Setup video writer
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps / stride, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    frames_written = 0
    total_to_write = (end_frame - start_frame + stride - 1) // stride

    try:
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process strided frames
            if (frame_idx - start_frame) % stride == 0:
                players = positions_by_frame.get(frame_idx, [])
                overlay_frame = render_debug_overlay(
                    frame,
                    frame_idx,
                    players,
                    ball_positions,
                    split_y,
                    primary_tracks,
                )
                out.write(overlay_frame)
                frames_written += 1

                if progress_callback and frames_written % 30 == 0:
                    progress_callback(frames_written / total_to_write)

            frame_idx += 1

        if progress_callback:
            progress_callback(1.0)

    finally:
        cap.release()
        out.release()


@handle_errors
def track_players(
    video: Path = typer.Argument(
        ...,
        exists=True,
        help="Input video file to track player positions",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file for player positions (default: video_player_track.json)",
    ),
    start_ms: int | None = typer.Option(
        None,
        "--start",
        help="Start time in milliseconds",
    ),
    end_ms: int | None = typer.Option(
        None,
        "--end",
        help="End time in milliseconds",
    ),
    confidence: float = typer.Option(
        0.25,
        "--confidence", "-c",
        help="Detection confidence threshold (0-1)",
    ),
    stride: int = typer.Option(
        1,
        "--stride", "-s",
        help="Process every Nth frame (1=all frames, 3=every 3rd frame for faster processing)",
    ),
    filter_court: bool = typer.Option(
        True,
        "--filter/--no-filter",
        help="Filter to court players only (default: enabled)",
    ),
    debug_video: Path | None = typer.Option(
        None,
        "--debug-video",
        help="Output debug video showing two-team split and player boxes",
    ),
    calibration: str | None = typer.Option(
        None,
        "--calibration",
        help="Court calibration corners as JSON array of 4 {x,y} points (normalized 0-1)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output",
    ),
    ball_model: str = typer.Option(
        DEFAULT_BALL_MODEL,
        "--ball-model",
        help=f"Ball tracking model variant ({', '.join(get_available_ball_models())})",
    ),
    # Filter parameter overrides for tuning edge cases
    min_bbox_area: float | None = typer.Option(
        None,
        "--min-bbox-area",
        help="Min bbox area threshold (default: 0.003, lower for far-side players)",
    ),
    min_bbox_height: float | None = typer.Option(
        None,
        "--min-bbox-height",
        help="Min bbox height threshold (default: 0.08, lower for far-side players)",
    ),
    min_position_spread: float | None = typer.Option(
        None,
        "--min-position-spread",
        help="Min position spread for active players (default: 0.015, lower for serve receivers)",
    ),
    min_presence_rate: float | None = typer.Option(
        None,
        "--min-presence-rate",
        help="Min track presence rate (default: 0.20, lower for short rallies)",
    ),
    # Preprocessing options
    preprocessing: str = typer.Option(
        PREPROCESSING_NONE,
        "--preprocessing",
        help=f"Frame preprocessing: {PREPROCESSING_NONE} (default), {PREPROCESSING_CLAHE} (contrast enhancement)",
    ),
    # Tracker selection
    tracker: str = typer.Option(
        DEFAULT_TRACKER,
        "--tracker",
        help=f"Tracking algorithm: {TRACKER_BOTSORT} (default, fewer ID switches), {TRACKER_BYTETRACK}",
    ),
    # YOLO model selection
    yolo_model: str = typer.Option(
        DEFAULT_YOLO_MODEL,
        "--yolo-model",
        help=f"YOLO model size: {', '.join(YOLO_MODELS.keys())} (default: {DEFAULT_YOLO_MODEL})",
    ),
) -> None:
    """Track player positions in a beach volleyball video.

    Uses YOLOv8n for detection with ByteTrack for temporal tracking.
    Outputs player coordinates with track IDs and confidence scores.

    By default, filters to court players only using:
    - Bbox size filtering (removes small background detections)
    - Play area filtering (removes spectators outside court using ball trajectory)
    - Track stability (prefers tracks that appear consistently across frames)
    - Two-team selection (selects top players from each court side - near/far)

    The two-team filter prevents near-side bias by ensuring players from both
    sides of the court are selected (camera is always behind baseline).

    Use --no-filter to include all detected persons (spectators, referees, etc.).

    Example:
        rallycut track-players game.mp4 -o rally_players.json
        rallycut track-players game.mp4 --no-filter  # Include all persons
        rallycut track-players game.mp4 --debug-video out.mp4  # Visualize tracking
    """
    validate_video_file(video)

    # Parse calibration if provided
    calibrator = None
    if calibration:
        import json
        try:
            corners_data = json.loads(calibration)
            if isinstance(corners_data, list) and len(corners_data) == 4:
                from rallycut.court.calibration import CourtCalibrator
                calibrator = CourtCalibrator()
                image_corners = [(c["x"], c["y"]) for c in corners_data]
                calibrator.calibrate(image_corners)
                if not quiet:
                    console.print("[dim]Court calibration: loaded from corners[/dim]")
            else:
                console.print("[yellow]Warning: Calibration must be 4 corners, ignoring[/yellow]")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            console.print(f"[yellow]Warning: Invalid calibration JSON: {e}[/yellow]")

    # Default output path
    if output is None:
        output = video.with_name(f"{video.stem}_player_track.json")

    if not quiet:
        console.print(f"[bold]Player Tracking:[/bold] {video.name}")
        if start_ms is not None or end_ms is not None:
            time_range = f"{start_ms or 0}ms - {end_ms or 'end'}ms"
            console.print(f"[dim]Time range: {time_range}[/dim]")
        if stride > 1:
            console.print(f"[dim]Stride: {stride} (processing every {stride}th frame)[/dim]")
        if filter_court:
            console.print("[dim]Court player filtering: enabled (beach 2v2, max 4 players)[/dim]")

    # Run ball tracking first if filtering enabled
    ball_positions: list[BallPosition] | None = None
    if filter_court:
        if not quiet:
            console.print(f"\n[dim]Running ball tracking (model: {ball_model}) for court filtering...[/dim]")

        ball_tracker = BallTracker(model=ball_model)
        if quiet:
            ball_result = ball_tracker.track_video(
                video,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Tracking ball...", total=100)

                def update_ball_progress(p: float) -> None:
                    progress.update(task, completed=int(p * 100))

                ball_result = ball_tracker.track_video(
                    video,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    progress_callback=update_ball_progress,
                )

        ball_positions = ball_result.positions
        if not quiet:
            console.print(
                f"[dim]Ball detection rate: {ball_result.detection_rate * 100:.1f}%[/dim]"
            )

    # Create filter config with optional overrides (beach volleyball only for now)
    filter_config = None
    if filter_court:
        filter_config = PlayerFilterConfig()
        # Apply CLI overrides for parameter tuning
        if min_bbox_area is not None:
            filter_config.min_bbox_area = min_bbox_area
        if min_bbox_height is not None:
            filter_config.min_bbox_height = min_bbox_height
        if min_position_spread is not None:
            filter_config.min_position_spread_for_primary = min_position_spread
        if min_presence_rate is not None:
            filter_config.min_presence_rate = min_presence_rate

        # Log overrides
        overrides = []
        if min_bbox_area is not None:
            overrides.append(f"min_bbox_area={min_bbox_area}")
        if min_bbox_height is not None:
            overrides.append(f"min_bbox_height={min_bbox_height}")
        if min_position_spread is not None:
            overrides.append(f"min_position_spread={min_position_spread}")
        if min_presence_rate is not None:
            overrides.append(f"min_presence_rate={min_presence_rate}")
        if overrides and not quiet:
            console.print(f"[dim]Filter overrides: {', '.join(overrides)}[/dim]")

    # Create player tracker with optional preprocessing and tracker selection
    if preprocessing != PREPROCESSING_NONE and not quiet:
        console.print(f"[dim]Preprocessing: {preprocessing}[/dim]")
    if tracker != DEFAULT_TRACKER and not quiet:
        console.print(f"[dim]Tracker: {tracker}[/dim]")
    if yolo_model != DEFAULT_YOLO_MODEL and not quiet:
        console.print(f"[dim]YOLO model: {yolo_model}[/dim]")
    player_tracker = PlayerTracker(
        confidence=confidence,
        preprocessing=preprocessing,
        tracker=tracker,
        yolo_model=yolo_model,
    )

    # Track with progress
    if quiet:
        result = player_tracker.track_video(
            video,
            start_ms=start_ms,
            end_ms=end_ms,
            stride=stride,
            ball_positions=ball_positions,
            filter_enabled=filter_court,
            filter_config=filter_config,
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Tracking players...", total=100)

            def update_progress(p: float) -> None:
                progress.update(task, completed=int(p * 100))

            result = player_tracker.track_video(
                video,
                start_ms=start_ms,
                end_ms=end_ms,
                stride=stride,
                progress_callback=update_progress,
                ball_positions=ball_positions,
                filter_enabled=filter_court,
                filter_config=filter_config,
            )

    # Compute ball phases and server detection (if ball tracking available)
    if ball_positions and result.positions:
        if not quiet:
            console.print("\n[dim]Analyzing ball phases...[/dim]")

        # Include ball positions in result for trajectory overlay
        result.ball_positions = ball_positions

        # Detect ball phases using game-flow-aware state machine
        phases = detect_ball_phases_v2(ball_positions, result.positions)
        result.ball_phases = [
            BallPhaseInfo(
                phase=p.phase.value,
                frame_start=p.frame_start,
                frame_end=p.frame_end,
                velocity=p.velocity,
                ball_x=p.ball_position[0],
                ball_y=p.ball_position[1],
            )
            for p in phases
        ]

        # Detect server - only consider primary tracks (actual players)
        primary_set = set(result.primary_track_ids) if result.primary_track_ids else set()
        server_positions = [p for p in result.positions if p.track_id in primary_set] if primary_set else result.positions
        server_result = detect_server(
            server_positions,
            ball_positions,
            rally_start_frame=0,
            calibrator=calibrator,
        )
        if server_result.track_id >= 0:
            result.server_info = ServerInfo(
                track_id=server_result.track_id,
                confidence=server_result.confidence,
                serve_frame=server_result.serve_frame,
                serve_velocity=server_result.serve_velocity,
                is_near_court=server_result.is_near_court,
            )

        if not quiet:
            phase_counts: dict[str, int] = {}
            for p in result.ball_phases:
                phase_counts[p.phase] = phase_counts.get(p.phase, 0) + 1
            console.print(f"[dim]Ball phases: {phase_counts}[/dim]")
            console.print(f"[dim]Ball positions: {len(ball_positions)}[/dim]")
            if result.server_info:
                console.print(
                    f"[dim]Server detected: track #{result.server_info.track_id} "
                    f"(confidence: {result.server_info.confidence:.1%})[/dim]"
                )

    # Save results
    result.to_json(output)

    # Print summary
    if not quiet:
        console.print("\n[green]Player tracking complete![/green]")
        console.print(f"  Frames processed: {result.frame_count}")
        console.print(f"  Avg players/frame: {result.avg_players_per_frame:.1f}")
        console.print(f"  Unique tracks: {result.unique_track_count}")
        console.print(f"  Detection rate: {result.detection_rate * 100:.1f}%")
        console.print(f"  Processing time: {result.processing_time_ms / 1000:.2f}s")
        console.print(f"  Output: {output}")

        # Performance stats
        if result.frame_count > 0 and result.processing_time_ms > 0:
            fps = result.frame_count / (result.processing_time_ms / 1000)
            console.print(f"  Speed: {fps:.1f} FPS")

        # Warning for low detection rate
        if result.detection_rate < 0.5:
            console.print(
                "\n[yellow]Warning:[/yellow] Low detection rate. "
                "Players may be out of frame or difficult to detect."
            )

    # Create debug video if requested
    if debug_video is not None:
        if not quiet:
            console.print("\n[bold]Creating debug video...[/bold]")

        # Use pre-computed filter state from tracking result
        # (Recomputing here would give wrong results without player positions)
        split_y = result.court_split_y
        primary_tracks = set(result.primary_track_ids)

        if filter_court and not quiet:
            if split_y is not None:
                console.print(f"  Court split: y={split_y:.3f}")
            if primary_tracks:
                console.print(f"  Primary tracks: {sorted(primary_tracks)}")

        # Create the debug video
        if quiet:
            create_debug_video(
                video,
                debug_video,
                result,
                ball_positions,
                split_y,
                primary_tracks,
                start_ms,
                end_ms,
                stride,
            )
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Rendering debug video...", total=100)

                def update_video_progress(p: float) -> None:
                    progress.update(task, completed=int(p * 100))

                create_debug_video(
                    video,
                    debug_video,
                    result,
                    ball_positions,
                    split_y,
                    primary_tracks,
                    start_ms,
                    end_ms,
                    stride,
                    update_video_progress,
                )

        if not quiet:
            console.print(f"  Debug video: {debug_video}")
