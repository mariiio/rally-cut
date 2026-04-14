"""Crowd-density check: average number of person detections *outside* the
court polygon, per frame. Purely a function of detections + court ROI."""
from __future__ import annotations

import statistics

from rallycut.quality.camera_distance import Detection
from rallycut.quality.types import CheckResult, Issue, Tier

AVG_SPECTATORS_GATE_THRESHOLD = 5.0  # avg non-court people per frame


def _center_in_bbox(d: Detection, bbox: tuple[float, float, float, float]) -> bool:
    xmin, ymin, xmax, ymax = bbox
    return xmin <= d.x <= xmax and ymin <= d.y <= ymax


def check_crowd_density(
    frames_detections: list[list[Detection]],
    court_bbox: tuple[float, float, float, float],
) -> CheckResult:
    counts_outside = [
        sum(1 for d in frame if not _center_in_bbox(d, court_bbox))
        for frame in frames_detections
    ]
    if not counts_outside:
        return CheckResult(issues=[], metrics={})

    avg = statistics.mean(counts_outside)
    metrics = {"avgNonCourtPersons": avg}
    issues: list[Issue] = []
    if avg > AVG_SPECTATORS_GATE_THRESHOLD:
        issues.append(Issue(
            id="crowded_scene",
            tier=Tier.GATE,
            severity=min(1.0, (avg - AVG_SPECTATORS_GATE_THRESHOLD) / AVG_SPECTATORS_GATE_THRESHOLD),
            message="There are a lot of people off the court — tracking may occasionally include spectators.",
            source="preflight",
            data={"avgNonCourtPersons": avg},
        ))
    return CheckResult(issues=issues, metrics=metrics)
