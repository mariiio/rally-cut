"""Quality-check data types shared across checks and runner.

Each check returns a CheckResult (zero or more Issues). The runner merges them
into a QualityReport that the API serializes to qualityReportJson.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class Tier(str, Enum):
    BLOCK = "block"
    GATE = "gate"
    ADVISORY = "advisory"


_TIER_ORDER = {Tier.BLOCK: 0, Tier.GATE: 1, Tier.ADVISORY: 2}


@dataclass(frozen=True)
class Issue:
    id: str
    tier: Tier
    severity: float  # 0..1
    message: str
    source: str  # 'preview' | 'upload' | 'preflight' | 'tracking'
    data: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
            "detectedAt": datetime.now(UTC).isoformat(),
            "data": self.data,
        }


@dataclass
class CheckResult:
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # raw metrics for debugging


@dataclass(frozen=True)
class CourtDetection:
    """Court-keypoint detection output for the API's auto-calibration path.

    `corners` follows the courtCalibrationJson convention: 4 points ordered
    near-left, near-right, far-right, far-left, normalised to [0,1]^2.
    """

    corners: list[dict[str, float]]
    confidence: float


@dataclass
class QualityReport:
    issues: list[Issue]
    source: str
    sample_seconds: int | None = None
    duration_ms: int | None = None
    court: CourtDetection | None = None

    @classmethod
    def from_checks(
        cls,
        results: list[CheckResult],
        source: str,
        sample_seconds: int | None = None,
        duration_ms: int | None = None,
        court: CourtDetection | None = None,
    ) -> QualityReport:
        flat: list[Issue] = [i for r in results for i in r.issues]
        flat.sort(key=lambda i: (_TIER_ORDER[i.tier], -i.severity, i.id))
        return cls(
            issues=flat[:3],
            source=source,
            sample_seconds=sample_seconds,
            duration_ms=duration_ms,
            court=court,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": 2,
            "issues": [i.to_dict() for i in self.issues],
            self.source: {
                "ranAt": datetime.now(UTC).isoformat(),
                "sampleSeconds": self.sample_seconds,
                "durationMs": self.duration_ms,
            },
        }
        if self.court is not None:
            d["court"] = {
                "corners": self.court.corners,
                "confidence": self.court.confidence,
            }
        return d
