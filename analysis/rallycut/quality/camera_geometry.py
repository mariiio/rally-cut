"""Camera geometry checks derived from court keypoints.

The court-keypoint model returns four corners (TL, TR, BR, BL) in normalized
coordinates with a confidence score. We use them to detect:
  - tilt (baseline not horizontal → advisory; C will later auto-rotate)
  - wrong camera angle (camera not behind baseline → hard block)
  - no court detected (confidence too low → hard block, also subsumes 'not beach volleyball')

Thresholds are conservative defaults; see calibrate_quality_checks.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rallycut.quality.types import CheckResult, Issue, Tier

TILT_ADVISORY_DEG = 5.0  # baseline vs horizontal
MIN_COURT_CONFIDENCE = 0.6


@dataclass(frozen=True)
class CourtCorners:
    tl: tuple[float, float]
    tr: tuple[float, float]
    br: tuple[float, float]
    bl: tuple[float, float]
    confidence: float


def _baseline_tilt_deg(corners: CourtCorners) -> float:
    """Return the absolute tilt in degrees of the top baseline."""
    dx = corners.tr[0] - corners.tl[0]
    dy = corners.tr[1] - corners.tl[1]
    return abs(math.degrees(math.atan2(dy, dx)))


def _is_behind_baseline(corners: CourtCorners) -> bool:
    """Heuristic: camera is behind a baseline when both baselines are visible
    and the bottom baseline is wider than the top (perspective foreshortening).
    Side-view or overhead cameras fail this."""
    top_width = math.hypot(corners.tr[0] - corners.tl[0], corners.tr[1] - corners.tl[1])
    bot_width = math.hypot(corners.br[0] - corners.bl[0], corners.br[1] - corners.bl[1])
    if top_width == 0 or bot_width == 0:
        return False
    ratio = bot_width / max(top_width, 1e-6)
    # Perspective: bottom wider than top by > 5%
    return ratio > 1.05


def check_camera_geometry(corners: CourtCorners) -> CheckResult:
    issues: list[Issue] = []
    metrics: dict[str, float] = {"courtConfidence": corners.confidence}

    if corners.confidence < MIN_COURT_CONFIDENCE:
        issues.append(Issue(
            id="wrong_angle_or_not_volleyball",
            tier=Tier.BLOCK,
            severity=1.0,
            message="We couldn't find a beach volleyball court in this video. Make sure the camera is behind the baseline and the whole court is visible.",
            source="preflight",
            data={"courtConfidence": corners.confidence},
        ))
        return CheckResult(issues=issues, metrics=metrics)

    if not _is_behind_baseline(corners):
        issues.append(Issue(
            id="wrong_angle_or_not_volleyball",
            tier=Tier.BLOCK,
            severity=0.9,
            message="The camera doesn't look like it's behind the baseline. Tracking needs footage filmed from behind one end of the court.",
            source="preflight",
            data={"courtConfidence": corners.confidence},
        ))
        return CheckResult(issues=issues, metrics=metrics)

    tilt = _baseline_tilt_deg(corners)
    metrics["tiltDeg"] = tilt
    if tilt > TILT_ADVISORY_DEG:
        issues.append(Issue(
            id="video_rotated",
            tier=Tier.ADVISORY,
            severity=min(1.0, (tilt - TILT_ADVISORY_DEG) / 20.0),
            message=f"Video is tilted about {tilt:.0f}° — straightening the camera improves tracking accuracy.",
            source="preflight",
            data={"tiltDeg": tilt},
        ))

    return CheckResult(issues=issues, metrics=metrics)
