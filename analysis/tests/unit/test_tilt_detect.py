"""Tests for the tilt-detect CLI's pure computation layer."""
from __future__ import annotations

from unittest.mock import MagicMock

from rallycut.cli.commands.tilt_detect import _compute_tilt_from_frames


def _mock_detect(corners_conf_pairs):
    """Build a fake CourtKeypointDetector that returns canned results per call."""
    calls = iter(corners_conf_pairs)

    def _detect_from_frame(_bgr):
        corners, confidence = next(calls)
        result = MagicMock()
        result.corners = corners
        result.confidence = confidence
        return result

    det = MagicMock()
    det.detect_from_frame.side_effect = _detect_from_frame
    return det


def _corners_tilted(deg: float):
    import math
    # Build corners that baseline_tilt_deg will read as `deg` degrees
    # (baseline is TL → TR; keypoint format: [nl, nr, fr, fl])
    dx = 0.4
    dy = dx * math.tan(math.radians(deg))
    # keypoint_detector order: [nl, nr, fr, fl]
    return [
        {"x": 0.2, "y": 0.8},  # nl → bl
        {"x": 0.8, "y": 0.8},  # nr → br
        {"x": 0.7, "y": 0.4 + dy},  # fr → tr
        {"x": 0.3, "y": 0.4},  # fl → tl
    ]


def test_tilt_detect_returns_median_of_high_confidence_frames():
    detector = _mock_detect([
        (_corners_tilted(7), 0.85),
        (_corners_tilted(9), 0.90),
        (_corners_tilted(6), 0.82),
    ])
    result = _compute_tilt_from_frames(detector, frames=[b"a", b"b", b"c"])
    assert abs(result["tiltDeg"] - 7.0) < 0.5  # median of 6,7,9
    assert abs(result["courtConfidence"] - 0.85) < 0.01  # median of 0.82, 0.85, 0.90
    assert result["framesScored"] == 3


def test_tilt_detect_filters_low_confidence_frames():
    detector = _mock_detect([
        (_corners_tilted(8), 0.85),    # kept
        (_corners_tilted(20), 0.10),   # dropped (low conf)
        (_corners_tilted(7), 0.80),    # kept
    ])
    result = _compute_tilt_from_frames(detector, frames=[b"a", b"b", b"c"])
    assert abs(result["tiltDeg"] - 7.5) < 0.5
    assert result["framesScored"] == 2


def test_tilt_detect_no_confident_frames_returns_zero_conf():
    detector = _mock_detect([
        (_corners_tilted(8), 0.20),
        (_corners_tilted(12), 0.30),
    ])
    result = _compute_tilt_from_frames(detector, frames=[b"a", b"b"])
    assert result["courtConfidence"] == 0.0
    assert result["framesScored"] == 0
