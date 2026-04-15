"""Tests for the Hough-based tilt detection pure functions."""
from __future__ import annotations

import math

import cv2
import numpy as np

from rallycut.cli.commands.tilt_detect import (
    _near_horizontal_lines,
    _weighted_median,
    compute_tilt_from_frames,
)


def _black_frame_with_line(angle_deg: float, width: int = 640, height: int = 360) -> np.ndarray:
    """Draw a long white line tilted by `angle_deg` degrees on a black frame.

    The line runs through the frame center and spans nearly the full width,
    which guarantees it passes the minLineLength cutoff in _near_horizontal_lines.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width / 2.0, height / 2.0
    rad = math.radians(angle_deg)
    half_len = min(width, height) * 0.4
    dx = math.cos(rad) * half_len
    dy = math.sin(rad) * half_len
    p1 = (int(cx - dx), int(cy - dy))
    p2 = (int(cx + dx), int(cy + dy))
    cv2.line(img, p1, p2, color=(255, 255, 255), thickness=3)
    return img


def test_weighted_median_simple():
    assert _weighted_median([(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]) == 2.0


def test_weighted_median_respects_weights():
    # Heavy weight on the second value drags the median there
    assert _weighted_median([(1.0, 1.0), (5.0, 100.0), (10.0, 1.0)]) == 5.0


def test_straight_line_detects_zero_tilt():
    frame = _black_frame_with_line(0.0)
    result = compute_tilt_from_frames([frame])
    assert abs(result["tiltDeg"]) < 1.0
    assert result["linesScored"] >= 1


def test_positive_tilt_detected_with_correct_sign():
    frame = _black_frame_with_line(7.0)
    result = compute_tilt_from_frames([frame])
    # 7° image-space tilt (clockwise in image coords = positive dy/dx)
    assert abs(result["tiltDeg"] - 7.0) < 1.5
    assert result["linesScored"] >= 1


def test_negative_tilt_detected_with_correct_sign():
    frame = _black_frame_with_line(-10.0)
    result = compute_tilt_from_frames([frame])
    assert abs(result["tiltDeg"] - (-10.0)) < 1.5
    assert result["linesScored"] >= 1


def test_near_vertical_line_is_filtered_out():
    # 60° from horizontal — outside the ANGLE_FILTER_DEG=30° window
    frame = _black_frame_with_line(60.0)
    result = compute_tilt_from_frames([frame])
    # No qualifying lines → no-op response
    assert result["tiltDeg"] == 0.0
    assert result["linesScored"] == 0


def test_empty_input_is_noop():
    result = compute_tilt_from_frames([])
    assert result["tiltDeg"] == 0.0
    assert result["linesScored"] == 0


def test_median_across_multiple_frames():
    frames = [
        _black_frame_with_line(5.0),
        _black_frame_with_line(6.0),
        _black_frame_with_line(7.0),
    ]
    result = compute_tilt_from_frames(frames)
    assert 5.0 <= result["tiltDeg"] <= 7.0
    assert result["linesScored"] >= 3


def test_near_horizontal_lines_returns_empty_for_blank_frame():
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    assert _near_horizontal_lines(blank) == []
