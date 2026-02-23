"""Detect court command - automatically detect court corners from video."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors, validate_video_file

console = Console()


@handle_errors
def detect_court(
    video: Path = typer.Argument(..., help="Path to the video file"),
    debug: bool = typer.Option(False, "--debug", help="Save debug visualization"),
    output_json: bool = typer.Option(False, "--json", help="Output JSON format"),
    start_ms: int | None = typer.Option(None, "--start-ms", help="Start time in milliseconds"),
    end_ms: int | None = typer.Option(None, "--end-ms", help="End time in milliseconds"),
    debug_dir: Path = typer.Option(
        Path("debug_court_detection"), "--debug-dir",
        help="Directory for debug images",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress console output"),
) -> None:
    """Automatically detect court corners from a volleyball video.

    Uses multi-frame white-line detection on sand + Hough lines + temporal
    aggregation + geometric court model fitting.

    Outputs 4 corners: near-left, near-right, far-right, far-left.
    """
    import cv2

    from rallycut.court.detector import CourtDetectionConfig, CourtDetector

    validate_video_file(video)

    # Convert ms to frames if specified
    start_frame = None
    end_frame = None
    if start_ms is not None or end_ms is not None:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            if start_ms is not None:
                start_frame = int(start_ms / 1000 * fps)
            if end_ms is not None:
                end_frame = int(end_ms / 1000 * fps)

    detector = CourtDetector(CourtDetectionConfig())

    if not output_json and not quiet:
        console.print(f"[bold]Detecting court in:[/bold] {video.name}")

    result = detector.detect(video, start_frame=start_frame, end_frame=end_frame)

    if output_json:
        output = {
            "corners": result.corners,
            "confidence": result.confidence,
            "detected_lines": [
                {
                    "label": dl.label,
                    "p1": {"x": dl.p1[0], "y": dl.p1[1]},
                    "p2": {"x": dl.p2[0], "y": dl.p2[1]},
                    "support": dl.support,
                    "angle_deg": dl.angle_deg,
                }
                for dl in result.detected_lines
            ],
            "warnings": result.warnings,
        }
        print(json.dumps(output, indent=2))
        return

    # Rich output
    if result.corners and len(result.corners) == 4:
        conf_color = (
            "green" if result.confidence >= 0.7
            else "yellow" if result.confidence >= 0.4
            else "red"
        )
        console.print(
            f"\n[bold {conf_color}]Confidence: {result.confidence:.2f}[/bold {conf_color}]"
        )

        console.print("\n[bold]Detected corners:[/bold]")
        labels = ["Near-left ", "Near-right", "Far-right ", "Far-left  "]
        for label, corner in zip(labels, result.corners):
            x, y = corner["x"], corner["y"]
            in_frame = 0 <= x <= 1 and 0 <= y <= 1
            marker = "" if in_frame else " [dim](off-screen)[/dim]"
            console.print(f"  {label}: ({x:.4f}, {y:.4f}){marker}")

        console.print(f"\n[bold]Lines detected:[/bold] {len(result.detected_lines)}")
        for dl in result.detected_lines:
            console.print(
                f"  {dl.label}: angle={dl.angle_deg:.1f}° support={dl.support}"
            )

        # Print calibration JSON for easy copy-paste
        console.print("\n[bold]Calibration JSON:[/bold]")
        cal_json = json.dumps(result.corners, indent=2)
        console.print(f"[dim]{cal_json}[/dim]")

    else:
        console.print("\n[red]Court detection failed[/red]")

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in result.warnings:
            console.print(f"  [yellow]![/yellow] {w}")

    # Debug visualization
    if debug:
        debug_dir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            if ret and frame is not None:
                debug_frame = detector.create_debug_image(frame, result)
                out_path = debug_dir / f"{video.stem}_court_debug.jpg"
                cv2.imwrite(str(out_path), debug_frame)
                console.print(f"\n[dim]Debug image saved to {out_path}[/dim]")
            cap.release()
