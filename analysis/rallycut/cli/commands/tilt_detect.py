"""`rallycut tilt-detect <frames-dir>` — emit {tiltDeg, linesScored}.

Classical Hough-line tilt detection: find near-horizontal line segments in the
image (net top, horizon, sideline of a sponsor board), take the length-weighted
median of their angles relative to horizontal. No ML model — fast (~10ms/frame),
rotation-equivariant, and independent of whether the footage looks like beach
volleyball.

Replaces the earlier YOLO-pose court-keypoint approach (2026-04-15), whose
confidence plateaued at 0.6–0.75 on rotated footage and whose tilt measurement
collapsed at >~10° rotation.

Used at POST /v1/videos/:id/confirm to decide whether the optimize pass should
append an FFmpeg rotate filter.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np
import typer

# Only count lines within this many degrees of horizontal. Generous enough to
# handle moderate tilts while rejecting obviously non-horizontal edges (court
# sidelines, player silhouettes).
ANGLE_FILTER_DEG = 30.0

# Line must span at least this fraction of the frame's shorter dimension.
# The beach-VB net top and horizon are both near frame-width; this cutoff
# rejects short edges from clothing, sand texture, sponsor logos, etc.
MIN_LINE_LENGTH_FRAC = 0.25


def tilt_detect(
    frames_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    as_json: bool = typer.Option(True, "--json/--no-json"),
) -> None:
    """Detect tilt from a directory of JPEG frames. Emit JSON to stdout."""
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    frames = [cv2.imread(str(p)) for p in frame_paths]
    frames = [f for f in frames if f is not None]

    result = compute_tilt_from_frames(frames)
    if as_json:
        typer.echo(json.dumps(result))


def compute_tilt_from_frames(frames: list[np.ndarray]) -> dict:
    """Return `{tiltDeg, linesScored}`.

    `tiltDeg` is signed: positive = top of the frame is tilted clockwise
    (right side lower than left in image coordinates). A rotation filter
    correcting this should rotate by `-tiltDeg`.

    `linesScored` is the total number of near-horizontal line segments
    contributing to the median across all input frames.
    """
    collected: list[tuple[float, float]] = []  # (angle_deg, length_px)
    for f in frames:
        collected.extend(_near_horizontal_lines(f))
    if not collected:
        return {"tiltDeg": 0.0, "linesScored": 0}
    tilt = _weighted_median(collected)
    return {"tiltDeg": float(tilt), "linesScored": len(collected)}


def _near_horizontal_lines(bgr: np.ndarray) -> list[tuple[float, float]]:
    """Return (angle_deg, length_px) for every near-horizontal line in `bgr`.

    Uses a Canny + probabilistic Hough pipeline. Each returned angle is
    normalized to (-90, 90] and filtered to |angle| < ANGLE_FILTER_DEG.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Light blur suppresses pixel-scale noise that would otherwise spawn
    # spurious edges (sand grain, jersey texture).
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    min_dim = min(gray.shape)
    min_line_length = max(40, int(min_dim * MIN_LINE_LENGTH_FRAC))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=20,
    )
    if lines is None:
        return []

    out: list[tuple[float, float]] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0 and dy == 0:
            continue
        length = math.hypot(dx, dy)
        angle_deg = math.degrees(math.atan2(dy, dx))
        # Normalize to (-90, 90]: a line going "right-and-up" in image coords
        # (where y increases downward) has dy < 0, so its atan2 angle is
        # negative. That matches the sign convention we need.
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg <= -90:
            angle_deg += 180
        if abs(angle_deg) < ANGLE_FILTER_DEG:
            out.append((angle_deg, length))
    return out


def _weighted_median(values_and_weights: list[tuple[float, float]]) -> float:
    """Weighted median: value where cumulative weight crosses half of total."""
    sorted_pairs = sorted(values_and_weights, key=lambda p: p[0])
    total = sum(w for _, w in sorted_pairs)
    if total == 0:
        return statistics.median(v for v, _ in sorted_pairs)
    half = total / 2
    cum = 0.0
    for v, w in sorted_pairs:
        cum += w
        if cum >= half:
            return v
    return sorted_pairs[-1][0]
