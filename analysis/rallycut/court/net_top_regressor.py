"""Learned regression of `net_top_y` from court calibration corners.

Predicts the normalized image-y of the net top tape from the 4 court
calibration corners alone — no ball positions, no frame pixels, no
inference engine. Just 9 floating-point ops per video.

Background
----------
The original `contact_detector.estimate_net_position` derives `net_y`
from the ball-trajectory midpoint between local Y minima and maxima.
On a 77-video user-labeled GT corpus the median |Δ| is 0.025 and
worst-case |Δ| 0.178 (14 videos with |Δ| > 0.05). The error is
algorithmic, not random — the ball trajectory midpoint is a noisy
proxy for the net top, especially at 60fps where temporal aliasing
biases the sampling.

This module is the production-recommended replacement for that
estimator: a ridge regression over 8 perspective-geometry features
derived from the 4 corners. LOO-CV on the 77-video GT:

    median |Δ|  =  0.008       (vs A0 baseline 0.025, 3× better)
    mean   |Δ|  =  0.010
    worst  |Δ|  =  0.034       (vs A0 0.178, 5× better)
    >0.05 errors:   0          (vs A0 14)
    >0.10 errors:   0          (vs A0 2)

Top-5 A0-worst-video rescue: all five (caca, hehe, michu, macho, kiki)
go from |Δ| ∈ [0.077, 0.178] to |Δ| ∈ [0.006, 0.014].

Visually validated on representative frames (probe X-M); see
`analysis/scripts/probe_X_m_visual_validation.py` and the saved
overlays under `/tmp/net_top_validation_m4/`.

Coefficients
------------
Coefficients below were fit by ridge regression (α=0.01) on all 77
labeled samples in `videos.court_calibration_net_top_y` and the
corresponding `courtCalibrationJson` corners. See
`analysis/scripts/probe_X_l_net_top_regression.py` for the fit script.

Inputs
------
The 4 corners are normalized image-y coordinates (0-1), in the order
expected by `CourtCalibrationPanel`:
    0 = bottom-left (near baseline, left sideline)
    1 = bottom-right
    2 = top-right (far baseline, right sideline)
    3 = top-left

When corners are missing or invalid, callers should fall back to the
existing ball-trajectory estimator (`estimate_net_position`).
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# M4 ridge-regression coefficients (probe X-L, fit on 77 user-GT samples)
# --------------------------------------------------------------------------
_INTERCEPT = -0.0736
_COEF_MIDLINE_Y = +0.2983
_COEF_NEAR_Y = +0.3083
_COEF_FAR_Y = +0.3966
_COEF_COURT_DEPTH_Y = -0.0882
_COEF_MIDLINE_WIDTH_X = -0.1879
_COEF_NEAR_WIDTH_X = -0.0708
_COEF_FAR_WIDTH_X = -0.1226
_COEF_TRAPEZOID_ASPECT = +0.0038


@dataclass(frozen=True)
class _Pt:
    x: float
    y: float


def _line_intersect(
    p1: _Pt, p2: _Pt, p3: _Pt, p4: _Pt,
) -> tuple[float, float] | None:
    """Intersection of line p1-p2 with p3-p4. None if (near-)parallel."""
    denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
    if abs(denom) < 1e-9:
        return None
    t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom
    return (p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))


def _features(corners: list[dict]) -> dict[str, float] | None:
    """Compute the 8 geometric features used by the M4 model.

    Returns None when corners are malformed (wrong length, degenerate).
    """
    if not corners or len(corners) != 4:
        return None
    try:
        pts = [_Pt(float(c["x"]), float(c["y"])) for c in corners]
    except (KeyError, TypeError, ValueError):
        return None
    near_y = (pts[0].y + pts[1].y) / 2
    far_y = (pts[2].y + pts[3].y) / 2
    near_width_x = abs(pts[1].x - pts[0].x)
    far_width_x = abs(pts[2].x - pts[3].x)
    court_depth_y = near_y - far_y
    if court_depth_y <= 1e-6:
        return None
    diag = _line_intersect(pts[0], pts[2], pts[1], pts[3])
    midline_y = diag[1] if diag else (near_y + far_y) / 2
    # Midline width: use the perspective-correct endpoints when a
    # baseline vanishing point exists; otherwise average of near/far
    # widths is an OK fallback.
    midline_width_x = (near_width_x + far_width_x) / 2
    if diag is not None:
        baseline_vp = _line_intersect(pts[0], pts[1], pts[3], pts[2])
        if baseline_vp is not None:
            diag_pt = _Pt(diag[0], diag[1])
            vp_pt = _Pt(baseline_vp[0], baseline_vp[1])
            left = _line_intersect(diag_pt, vp_pt, pts[3], pts[0])
            right = _line_intersect(diag_pt, vp_pt, pts[2], pts[1])
            if left is not None and right is not None:
                midline_width_x = abs(right[0] - left[0])
    trapezoid_aspect = ((near_width_x + far_width_x) / 2) / court_depth_y
    return {
        "midline_y": midline_y,
        "near_y": near_y,
        "far_y": far_y,
        "court_depth_y": court_depth_y,
        "midline_width_x": midline_width_x,
        "near_width_x": near_width_x,
        "far_width_x": far_width_x,
        "trapezoid_aspect": trapezoid_aspect,
    }


def predict_net_top_y(corners: list[dict]) -> float | None:
    """Predict normalized image-y of the net top tape from 4 calibration
    corners. Returns None when corners are malformed; caller should
    fall back to the ball-trajectory estimator in that case.

    Output is clamped to [0, 1] (image bounds).
    """
    f = _features(corners)
    if f is None:
        return None
    y = (
        _INTERCEPT
        + _COEF_MIDLINE_Y * f["midline_y"]
        + _COEF_NEAR_Y * f["near_y"]
        + _COEF_FAR_Y * f["far_y"]
        + _COEF_COURT_DEPTH_Y * f["court_depth_y"]
        + _COEF_MIDLINE_WIDTH_X * f["midline_width_x"]
        + _COEF_NEAR_WIDTH_X * f["near_width_x"]
        + _COEF_FAR_WIDTH_X * f["far_width_x"]
        + _COEF_TRAPEZOID_ASPECT * f["trapezoid_aspect"]
    )
    return max(0.0, min(1.0, y))
