"""Quick video quality assessment for early feedback at upload time.

Analyzes 3-5 sample frames with YOLO to estimate scene complexity,
camera distance, and court visibility. Runs in ~2-3 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors, validate_video_file

console = Console()


def _sample_frames(video_path: str, num_samples: int = 5) -> list[np.ndarray]:
    """Sample evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise typer.BadParameter(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_samples:
        num_samples = max(1, total_frames)

    # Skip first/last 10% to avoid title cards
    start = int(total_frames * 0.1)
    end = int(total_frames * 0.9)
    if end <= start:
        start, end = 0, total_frames

    indices = np.linspace(start, end, num_samples, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def _get_video_info(video_path: str) -> dict:
    """Get basic video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "totalFrames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def _assess_frames(frames: list[np.ndarray]) -> dict:
    """Run YOLO on sample frames to assess scene quality."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"error": "YOLO not available", "warnings": ["Install ultralytics for quality assessment"]}

    model = YOLO("yolo11s.pt")
    model.fuse()

    all_person_counts: list[int] = []
    all_heights: list[float] = []
    all_person_areas: list[float] = []

    for frame in frames:
        h, w = frame.shape[:2]
        results = model(frame, imgsz=1280, verbose=False, classes=[0])  # class 0 = person

        person_count = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue
                person_count += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox_h = (y2 - y1) / h
                bbox_area = ((x2 - x1) * (y2 - y1)) / (w * h)
                all_heights.append(bbox_h)
                all_person_areas.append(bbox_area)

        all_person_counts.append(person_count)

    avg_people = np.mean(all_person_counts) if all_person_counts else 0
    median_height = float(np.median(all_heights)) if all_heights else 0

    # Derive quality signals
    warnings: list[str] = []
    quality_signals: dict = {}

    # Camera distance
    if median_height > 0.35:
        camera_distance = "close"
    elif median_height < 0.12:
        camera_distance = "far"
        warnings.append("Camera is very far — player detection may be less accurate")
    elif median_height < 0.20:
        camera_distance = "far"
    else:
        camera_distance = "medium"

    quality_signals["cameraDistance"] = {
        "avgBboxHeight": round(median_height, 3),
        "category": camera_distance,
    }

    # Scene complexity
    scene_category = "complex" if avg_people > 6 else "simple"
    quality_signals["sceneComplexity"] = {
        "avgPeople": round(float(avg_people), 1),
        "category": scene_category,
    }

    if avg_people > 8:
        warnings.append("Crowded scene detected — tracking may have difficulty isolating players")
    if avg_people < 2:
        warnings.append("Few people detected — video may not contain volleyball gameplay")

    # Court visibility heuristic: check if people are spread across frame
    if all_heights:
        height_std = float(np.std(all_heights))
        if height_std > 0.15:
            warnings.append("Large height variance — camera may be too close or players at very different distances")
        quality_signals["heightVariance"] = round(height_std, 3)

    # Resolution check
    if frames:
        h, w = frames[0].shape[:2]
        if w < 1280:
            warnings.append(f"Low resolution ({w}x{h}) — consider recording in 1080p or higher")
        quality_signals["resolution"] = {"width": w, "height": h}

    # Expected quality score (0-1)
    score = 1.0
    if camera_distance == "far":
        score -= 0.2
    if scene_category == "complex":
        score -= 0.15
    if avg_people < 2:
        score -= 0.3
    if frames and frames[0].shape[1] < 1280:
        score -= 0.1
    score = max(0.0, min(1.0, score))

    return {
        "expectedQuality": round(score, 2),
        "warnings": warnings,
        "cameraDistance": quality_signals.get("cameraDistance"),
        "sceneComplexity": quality_signals.get("sceneComplexity"),
    }


@handle_errors
def assess_quality(
    video: Path = typer.Argument(..., help="Input video file"),
    num_samples: int = typer.Option(5, "--samples", "-n", help="Number of frames to analyze"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress console output"),
) -> None:
    """Assess video quality for volleyball tracking (quick, ~2-3s)."""
    validate_video_file(video)

    if not quiet:
        console.print(f"[bold]Assessing video quality:[/bold] {video.name}")

    frames = _sample_frames(str(video), num_samples)
    if not frames:
        console.print("[red]Failed to read frames from video[/red]")
        raise typer.Exit(code=1)

    video_info = _get_video_info(str(video))
    result = _assess_frames(frames)
    result["videoInfo"] = video_info

    if json_output or quiet:
        print(json.dumps(result))
    else:
        score = result.get("expectedQuality", 0)
        color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
        console.print(f"\n[bold]Expected Quality:[/bold] [{color}]{score:.0%}[/{color}]")

        cam = result.get("cameraDistance", {})
        if cam:
            console.print(f"  Camera distance: {cam.get('category', '?')} (avg bbox height: {cam.get('avgBboxHeight', 0):.3f})")

        scene = result.get("sceneComplexity", {})
        if scene:
            console.print(f"  Scene complexity: {scene.get('category', '?')} ({scene.get('avgPeople', 0):.1f} people/frame)")

        warnings = result.get("warnings", [])
        if warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for w in warnings:
                console.print(f"  [yellow]• {w}[/yellow]")
        else:
            console.print("\n[green]No issues detected — video looks good for tracking.[/green]")
