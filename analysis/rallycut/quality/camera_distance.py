"""Camera-distance heuristic: median player bbox height over a 10-frame sample.

The actual YOLO inference lives in `rallycut.detection` — this module only
scores Detection lists, so it stays cheap to unit-test.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from rallycut.quality.types import CheckResult, Issue, Tier

BBOX_HEIGHT_FAR_THRESHOLD = 0.10  # normalized


@dataclass(frozen=True)
class Detection:
    x: float  # center x (normalized)
    y: float  # center y
    w: float  # width (normalized)
    h: float  # height (normalized)


def check_camera_distance(frames_detections: list[list[Detection]]) -> CheckResult:
    """`frames_detections[i]` = detections in the i-th sampled frame (person class only)."""
    all_heights = [d.h for frame in frames_detections for d in frame]
    if not all_heights:
        return CheckResult(issues=[], metrics={})

    median_h = statistics.median(all_heights)
    metrics = {"medianBboxHeight": median_h}
    issues: list[Issue] = []
    if median_h < BBOX_HEIGHT_FAR_THRESHOLD:
        issues.append(Issue(
            id="camera_too_far",
            tier=Tier.GATE,
            severity=min(1.0, (BBOX_HEIGHT_FAR_THRESHOLD - median_h) / BBOX_HEIGHT_FAR_THRESHOLD),
            message="Camera is very far from the court — moving closer gives better player and ball tracking.",
            source="preflight",
            data={"medianBboxHeight": median_h},
        ))
    return CheckResult(issues=issues, metrics=metrics)
