"""`rallycut tilt-detect <frames-dir>` — emit {tiltDeg, courtConfidence, framesScored}.

Used at POST /v1/videos/:id/confirm to decide whether the optimize pass should
append an FFmpeg rotate filter. Pure computation layer (`_compute_tilt_from_frames`)
is unit-tested separately from the frame-loading + detector instantiation.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import cv2
import typer

from rallycut.quality.camera_geometry import CourtCorners, baseline_tilt_deg

MIN_FRAME_CONF = 0.5  # frames below this are excluded from the median


def tilt_detect(
    frames_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    as_json: bool = typer.Option(True, "--json/--no-json"),
) -> None:
    """Detect tilt from a directory of JPEG frames. Emit JSON to stdout."""
    from rallycut.court.keypoint_detector import CourtKeypointDetector

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    frames = [cv2.imread(str(p)) for p in frame_paths]
    frames = [f for f in frames if f is not None]

    result = _compute_tilt_from_frames(CourtKeypointDetector(), frames=frames)
    if as_json:
        typer.echo(json.dumps(result))


def _compute_tilt_from_frames(detector: Any, frames: list) -> dict:
    """Pure compute layer. `detector` must have `detect_from_frame(frame) -> result`
    with `result.corners` (list of {x, y} dicts) and `result.confidence` (float).
    """
    tilt_degs: list[float] = []
    confs: list[float] = []
    for frame in frames:
        det = detector.detect_from_frame(frame)
        if not det.corners or len(det.corners) < 4 or det.confidence < MIN_FRAME_CONF:
            continue
        # keypoint order: [nl, nr, fr, fl] -> our CourtCorners: tl=fl, tr=fr, br=nr, bl=nl
        nl, nr, fr, fl = det.corners[0], det.corners[1], det.corners[2], det.corners[3]
        corners = CourtCorners(
            tl=(fl["x"], fl["y"]),
            tr=(fr["x"], fr["y"]),
            br=(nr["x"], nr["y"]),
            bl=(nl["x"], nl["y"]),
            confidence=det.confidence,
        )
        tilt_degs.append(baseline_tilt_deg(corners))
        confs.append(det.confidence)

    if not tilt_degs:
        return {"tiltDeg": 0.0, "courtConfidence": 0.0, "framesScored": 0}
    return {
        "tiltDeg": statistics.median(tilt_degs),
        "courtConfidence": statistics.median(confs),
        "framesScored": len(tilt_degs),
    }
