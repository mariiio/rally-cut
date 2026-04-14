"""`rallycut preview-check <dir>` — run camera-geometry against a directory
of JPEG frames (no video decoding required). Used by the web pre-upload gate
before the upload commits.

CLIP classifier is deferred to Project C; for A1 this is purely the
court-keypoint-based check (which also catches the "not volleyball" case via
low confidence).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import typer

from rallycut.quality.camera_geometry import CourtCorners, check_camera_geometry
from rallycut.quality.types import QualityReport


def preview_check(
    frames_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    width: int = typer.Option(..., "--width"),
    height: int = typer.Option(..., "--height"),
    duration_s: float = typer.Option(..., "--duration-s"),
    as_json: bool = typer.Option(True, "--json/--no-json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Run camera-geometry check against a directory of JPEG frames."""
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        typer.echo(json.dumps({"version": 2, "issues": []}))
        raise typer.Exit(0)

    # Load the first frame for court detection (BGR uint8)
    bgr = cv2.imread(str(frame_paths[0]))
    if bgr is None:
        typer.echo(json.dumps({"version": 2, "issues": []}))
        raise typer.Exit(0)

    try:
        corners = _detect_corners_from_frame(bgr, width=width, height=height)
    except Exception:
        corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.0)

    geom = check_camera_geometry(corners)

    report = QualityReport.from_checks(
        [geom], source="preview", duration_ms=int(duration_s * 1000)
    )
    typer.echo(json.dumps(report.to_dict()))


def _detect_corners_from_frame(bgr: np.ndarray, width: int, height: int) -> CourtCorners:
    """Return CourtCorners from a single BGR frame using the keypoint model.

    Uses CourtKeypointDetector.detect_from_frame() which runs YOLO-pose on a
    single frame with bottom-padding and returns a CourtDetectionResult with
    normalized corner coordinates and a confidence score.
    """
    from rallycut.court.keypoint_detector import CourtKeypointDetector

    detector = CourtKeypointDetector()
    result = detector.detect_from_frame(bgr)

    if not result.corners or len(result.corners) < 4:
        return CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.0)

    # corners order from keypoint_detector: near-left, near-right, far-right, far-left
    # CourtCorners: tl=far-left, tr=far-right, br=near-right, bl=near-left
    # (top = far side of court, bottom = near side)
    nl = result.corners[0]  # near-left
    nr = result.corners[1]  # near-right
    fr = result.corners[2]  # far-right
    fl = result.corners[3]  # far-left

    return CourtCorners(
        tl=(fl["x"], fl["y"]),  # far-left = top-left from camera POV
        tr=(fr["x"], fr["y"]),  # far-right = top-right from camera POV
        br=(nr["x"], nr["y"]),  # near-right = bottom-right from camera POV
        bl=(nl["x"], nl["y"]),  # near-left = bottom-left from camera POV
        confidence=result.confidence,
    )
