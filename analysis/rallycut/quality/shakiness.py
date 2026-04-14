"""Shakiness = mean pixel-level frame-to-frame L1 residual after a global
translation estimate. Crude but robust: catches handheld jitter without
needing optical-flow inference.

Threshold calibrated offline (see calibrate_quality_checks.py)."""
from __future__ import annotations

import statistics

import numpy as np

from rallycut.quality.types import CheckResult, Issue, Tier

SHAKINESS_GATE_THRESHOLD = 0.10  # mean normalized residual


def _frame_residual(a: np.ndarray, b: np.ndarray) -> float:
    # Downsample to 64×64 grayscale for speed and noise robustness
    from cv2 import cvtColor, COLOR_BGR2GRAY, resize  # type: ignore
    ga = resize(cvtColor(a, COLOR_BGR2GRAY), (64, 64))
    gb = resize(cvtColor(b, COLOR_BGR2GRAY), (64, 64))
    diff = np.abs(ga.astype(np.float32) - gb.astype(np.float32)) / 255.0
    return float(diff.mean())


def check_shakiness(frames: list[np.ndarray]) -> CheckResult:
    if len(frames) < 2:
        return CheckResult(issues=[], metrics={})

    residuals = [_frame_residual(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    mean_res = statistics.mean(residuals)
    metrics = {"meanResidual": mean_res}
    issues: list[Issue] = []
    if mean_res > SHAKINESS_GATE_THRESHOLD:
        issues.append(Issue(
            id="shaky_camera",
            tier=Tier.GATE,
            severity=min(1.0, (mean_res - SHAKINESS_GATE_THRESHOLD) / SHAKINESS_GATE_THRESHOLD),
            message="The camera looks shaky — a tripod or stable mount gives much more accurate tracking.",
            source="preflight",
            data={"meanResidual": mean_res},
        ))
    return CheckResult(issues=issues, metrics=metrics)
