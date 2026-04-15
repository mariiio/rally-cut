"""`rallycut preview-check <dir>` — run court-geometry + beach-VB classifier
against a directory of JPEG frames (no video decoding required). Used by the
web pre-upload gate before the upload commits.

Both checks run on the same 5 client-extracted frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import typer

from rallycut.quality.beach_vb_classifier import classify_is_beach_vb
from rallycut.quality.camera_geometry import CourtCorners, check_camera_geometry
from rallycut.quality.types import CheckResult, QualityReport


def preview_check(
    frames_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    width: int = typer.Option(..., "--width"),
    height: int = typer.Option(..., "--height"),
    duration_s: float = typer.Option(..., "--duration-s"),
    as_json: bool = typer.Option(True, "--json/--no-json"),
    quiet: bool = typer.Option(False, "--quiet"),  # noqa: ARG001 (reserved for future UX)
) -> None:
    """Run preview-time checks against a directory of JPEG frames."""
    report = _run(frames_dir, width=width, height=height, duration_s=duration_s)
    if as_json:
        typer.echo(json.dumps(report.to_dict()))


def _run(frames_dir: Path, *, width: int, height: int, duration_s: float) -> QualityReport:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        return QualityReport.from_checks(
            [], source="preview", duration_ms=int(duration_s * 1000)
        )

    # Load first frame for court detection (BGR uint8)
    bgr = cv2.imread(str(frame_paths[0]))

    # Court geometry check (requires a valid decoded frame)
    if bgr is not None:
        try:
            corners = _detect_corners_from_frame(bgr, width=width, height=height)
        except Exception:  # noqa: BLE001
            corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.0)
        geom_result = check_camera_geometry(corners)
    else:
        geom_result = CheckResult()  # skip court check when frame can't be decoded

    # Score all available frames with CLIP (failure-tolerant: on any error,
    # skip the check entirely, preserving A1 behavior)
    try:
        probs = _score_beach_vb_for_frames(frame_paths)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[preview-check] beach-VB scoring failed: {exc}", err=True)
        probs = []
    vb_result = classify_is_beach_vb(probs)

    return QualityReport.from_checks(
        [geom_result, vb_result], source="preview", duration_ms=int(duration_s * 1000)
    )


def _detect_corners_from_frame(bgr: np.ndarray, width: int, height: int) -> CourtCorners:
    """Return CourtCorners from a single BGR frame using the keypoint model."""
    from rallycut.court.keypoint_detector import CourtKeypointDetector

    detector = CourtKeypointDetector()
    result = detector.detect_from_frame(bgr)

    if not result.corners or len(result.corners) < 4:
        return CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.0)

    nl = result.corners[0]
    nr = result.corners[1]
    fr = result.corners[2]
    fl = result.corners[3]
    return CourtCorners(
        tl=(fl["x"], fl["y"]),
        tr=(fr["x"], fr["y"]),
        br=(nr["x"], nr["y"]),
        bl=(nl["x"], nl["y"]),
        confidence=result.confidence,
    )


def _score_beach_vb_for_frames(frame_paths: list[Path]) -> list[float]:
    """Load JPEGs as PIL.Image and run open-clip. Returns beach-VB probs."""
    from PIL import Image

    from rallycut.quality.beach_vb_classifier import embed_and_score_frames

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    return embed_and_score_frames(images)
