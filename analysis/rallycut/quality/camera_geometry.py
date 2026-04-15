"""Camera geometry checks derived from court keypoints.

The court-keypoint model returns four corners (TL, TR, BR, BL) in normalized
coordinates with a confidence score. We use them to detect:
  - no court detected (confidence too low → hard block)
  - wrong camera angle (not behind baseline → hard block)

The tilt-advisory (`video_rotated`) was dropped on 2026-04-15: it never fired
in calibration (0 of 66 GT videos) or validation fixtures, and Project C will
re-add it bundled with auto-rotation once that lever is worth pulling.

MIN_COURT_CONFIDENCE=0.6 is the only empirically-supported threshold in the
A1 check set (best_lift 2.13 at 0.6, 1.77 at 0.75 on GT). Validation
fixtures confirm it blocks 3 of 5 obvious negatives; the two that slip
through (beach-trained model confidently finds a "court" in indoor /
unrelated footage) are tracked as Project C scope (richer non-VB rejection,
e.g. CLIP zero-shot).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rallycut.quality.types import CheckResult, Issue, Tier

MIN_COURT_CONFIDENCE = 0.6


@dataclass(frozen=True)
class CourtCorners:
    tl: tuple[float, float]
    tr: tuple[float, float]
    br: tuple[float, float]
    bl: tuple[float, float]
    confidence: float


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

    return CheckResult(issues=issues, metrics=metrics)


def baseline_tilt_deg(corners: CourtCorners) -> float:
    """Absolute tilt in degrees of the top baseline vs. horizontal.

    Returns 0.0 for degenerate corners (all points coincident). Re-added in
    Project C to feed the tilt-detect CLI; kept as a pure helper (no issue
    emission) so the dropped `video_rotated` advisory stays dropped.
    """
    dx = corners.tr[0] - corners.tl[0]
    dy = corners.tr[1] - corners.tl[1]
    if dx == 0 and dy == 0:
        return 0.0
    return abs(math.degrees(math.atan2(dy, dx)))
