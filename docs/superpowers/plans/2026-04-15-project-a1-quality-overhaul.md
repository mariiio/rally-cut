# Project A1 — Quality Overhaul + Upload Gate + Insights Removal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `VideoInsightsBanner` + ad-hoc `assessVideoQuality` with a single data-driven `QualityReport` surface; add pre-upload hard-fail gate that rejects non-volleyball / wrong-angle uploads before they commit; delete the `characteristicsJson` column.

**Architecture:** A new `analysis/rallycut/quality/` module implements each check as an isolated pure function over sampled frames + metadata. A calibration harness runs every check against the 63-video GT and produces `P(pipeline_failure | fires)` lift; checks below 3× lift are dropped. API exposes `runUploadChecks` (fast, metadata-only) and `runPreflightChecks` (heavy, CLI-spawned). Frontend adds an ffmpeg.wasm thumbnail extractor + client preview POST that gates presigned URL issuance. A new `QualityReportBanner` reads the single `Video.qualityReportJson` column — everything else is deleted.

**Tech Stack:** Python (uv, pytest, ultralytics YOLO, open-clip for zero-shot), Prisma + PostgreSQL, Express + Zod, Next.js + Zustand, ffmpeg.wasm.

**Spec:** See `/Users/mario/.claude/plans/squishy-bubbling-turtle.md` for the approved design. This plan implements **Project A1 only** (quality overhaul + upload gate + insights removal); A2 (orchestrator rewrite, progressive UX, edit-during-tracking) is a follow-up plan.

**Scope boundary:** A1 does NOT change the orchestrator control flow (`analysisStore` phase ordering stays as-is), does NOT implement edit-during-tracking, and does NOT add webhook idempotency. Those are all A2.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `analysis/rallycut/quality/__init__.py` | Package marker, re-exports. |
| `analysis/rallycut/quality/types.py` | `Issue`, `Tier`, `CheckResult` dataclasses; `QualityReport` builder. |
| `analysis/rallycut/quality/metadata.py` | ffprobe-based duration/resolution/fps/brightness checks (fast). |
| `analysis/rallycut/quality/camera_geometry.py` | Tilt + behind-baseline detection from court keypoints. |
| `analysis/rallycut/quality/camera_distance.py` | Median player bbox height via fast YOLO sample. |
| `analysis/rallycut/quality/crowd_density.py` | Non-court person count via YOLO sample. |
| `analysis/rallycut/quality/shakiness.py` | Optical-flow residual over a 5-second sample. |
| `analysis/rallycut/quality/beach_vb_classifier.py` | Zero-shot CLIP classifier for "beach volleyball match". |
| `analysis/rallycut/quality/runner.py` | Top-level entry point: runs all checks, returns `QualityReport`. |
| `analysis/rallycut/cli/commands/preflight.py` | `rallycut preflight <video>` CLI command. |
| `analysis/rallycut/cli/commands/assess_quality.py` | **Deleted** — old CLI. |
| `analysis/scripts/calibrate_quality_checks.py` | Calibration harness: runs every check against GT, computes lift per threshold. |
| `analysis/tests/unit/test_quality_metadata.py` | Unit tests for metadata check. |
| `analysis/tests/unit/test_quality_camera_geometry.py` | Unit tests for geometry check. |
| `analysis/tests/unit/test_quality_camera_distance.py` | Unit tests for camera-distance check. |
| `analysis/tests/unit/test_quality_crowd_density.py` | Unit tests for crowd-density check. |
| `analysis/tests/unit/test_quality_shakiness.py` | Unit tests for shakiness check. |
| `analysis/tests/unit/test_quality_beach_vb.py` | Unit tests for beach-VB classifier. |
| `analysis/tests/unit/test_quality_runner.py` | Unit tests for top-level report building. |
| `api/prisma/schema.prisma` | Add `qualityReportJson`, `VideoStatus.REJECTED`; remove `characteristicsJson`. |
| `api/prisma/migrations/<timestamp>_quality_report/migration.sql` | Migration. |
| `api/src/services/qualityService.ts` | Rewrite: `runUploadChecks`, `runPreflightChecks`, `buildQualityReport`, `runPreviewChecks`. |
| `api/src/routes/videos.ts` | New `POST /v1/videos/preflight-preview`; updated `/assess-quality` response. |
| `api/src/services/playerTrackingService.ts` | Stop writing `characteristicsJson`; merge tracking metrics into `qualityReportJson` instead. |
| `api/src/services/processingService.ts` | Merge brightness into `qualityReportJson` instead of `characteristicsJson`. |
| `api/tests/qualityService.test.ts` | New — tier merging, top-3 selection, report building. |
| `web/src/utils/extractPreviewFrames.ts` | ffmpeg.wasm / WebCodecs thumbnail extractor. |
| `web/src/components/QualityReportBanner.tsx` | New banner — top-3 issues, tier-colored. |
| `web/src/components/VideoInsightsBanner.tsx` | **Deleted.** |
| `web/src/components/EditorLayout.tsx` | Swap banner. |
| `web/src/components/UploadFlow.tsx` | Integrate pre-upload preview gate. |
| `web/src/types/rally.ts` | Drop `VideoCharacteristics`; add `QualityReport`. |
| `web/src/services/api.ts` | `preflightPreview()` + updated `assessQuality()` types. |

---

## Task 1: Quality module scaffolding + shared types

**Files:**
- Create: `analysis/rallycut/quality/__init__.py`
- Create: `analysis/rallycut/quality/types.py`
- Test: `analysis/tests/unit/test_quality_types.py`

- [ ] **Step 1: Write failing test for `Issue` dataclass + `QualityReport.from_checks` builder**

```python
# analysis/tests/unit/test_quality_types.py
from rallycut.quality.types import Issue, Tier, QualityReport, CheckResult


def test_issue_serializes_to_user_facing_dict():
    issue = Issue(
        id="camera_too_far",
        tier=Tier.GATE,
        severity=0.8,
        message="Camera is very far — player tracking may be less accurate.",
        source="preflight",
        data={"medianBboxHeight": 0.08},
    )
    out = issue.to_dict()
    assert out["id"] == "camera_too_far"
    assert out["tier"] == "gate"
    assert out["severity"] == 0.8
    assert out["source"] == "preflight"
    assert out["data"]["medianBboxHeight"] == 0.08


def test_report_picks_top_3_by_tier_then_severity():
    results = [
        CheckResult(issues=[Issue("c", Tier.ADVISORY, 0.9, "c", "preflight")]),
        CheckResult(issues=[Issue("a", Tier.BLOCK, 0.4, "a", "preflight")]),
        CheckResult(issues=[Issue("b", Tier.GATE, 0.95, "b", "preflight")]),
        CheckResult(issues=[Issue("d", Tier.GATE, 0.2, "d", "preflight")]),
    ]
    report = QualityReport.from_checks(results, source="preflight")
    ids = [i["id"] for i in report.to_dict()["issues"]]
    assert ids == ["a", "b", "c"]  # block first, then gate by severity, then advisory


def test_report_preserves_from_source_metadata():
    report = QualityReport.from_checks([], source="preflight", sample_seconds=60, duration_ms=12345)
    d = report.to_dict()
    assert d["preflight"]["sampleSeconds"] == 60
    assert d["preflight"]["durationMs"] == 12345
```

- [ ] **Step 2: Run test — verify it fails with ImportError**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_types.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'rallycut.quality'`.

- [ ] **Step 3: Create `__init__.py` and `types.py`**

```python
# analysis/rallycut/quality/__init__.py
from rallycut.quality.types import Issue, Tier, CheckResult, QualityReport

__all__ = ["Issue", "Tier", "CheckResult", "QualityReport"]
```

```python
# analysis/rallycut/quality/types.py
"""Quality-check data types shared across checks and runner.

Each check returns a CheckResult (zero or more Issues). The runner merges them
into a QualityReport that the API serializes to qualityReportJson.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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
            "detectedAt": datetime.now(timezone.utc).isoformat(),
            "data": self.data,
        }


@dataclass
class CheckResult:
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # raw metrics for debugging


@dataclass
class QualityReport:
    issues: list[Issue]
    source: str
    sample_seconds: int | None = None
    duration_ms: int | None = None

    @classmethod
    def from_checks(
        cls,
        results: list[CheckResult],
        source: str,
        sample_seconds: int | None = None,
        duration_ms: int | None = None,
    ) -> QualityReport:
        flat: list[Issue] = [i for r in results for i in r.issues]
        # Deterministic sort: tier (block first), then severity desc, then id
        flat.sort(key=lambda i: (_TIER_ORDER[i.tier], -i.severity, i.id))
        return cls(issues=flat[:3], source=source, sample_seconds=sample_seconds, duration_ms=duration_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "issues": [i.to_dict() for i in self.issues],
            self.source: {
                "ranAt": datetime.now(timezone.utc).isoformat(),
                "sampleSeconds": self.sample_seconds,
                "durationMs": self.duration_ms,
            },
        }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_types.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/quality/__init__.py analysis/rallycut/quality/types.py analysis/tests/unit/test_quality_types.py
git commit -m "feat(quality): scaffold quality module with Issue/Tier/QualityReport types"
```

---

## Task 2: Metadata check (duration, resolution, fps)

**Files:**
- Create: `analysis/rallycut/quality/metadata.py`
- Test: `analysis/tests/unit/test_quality_metadata.py`

- [ ] **Step 1: Write failing tests**

```python
# analysis/tests/unit/test_quality_metadata.py
from rallycut.quality.metadata import check_metadata, VideoMetadata
from rallycut.quality.types import Tier


def test_short_video_hard_blocks():
    meta = VideoMetadata(duration_s=5, width=1920, height=1080, fps=30)
    result = check_metadata(meta)
    ids = {i.id for i in result.issues}
    assert "video_too_short" in ids
    blocks = [i for i in result.issues if i.tier == Tier.BLOCK]
    assert len(blocks) == 1


def test_low_resolution_soft_gates():
    meta = VideoMetadata(duration_s=120, width=640, height=480, fps=30)
    result = check_metadata(meta)
    issue = next(i for i in result.issues if i.id == "resolution_too_low")
    assert issue.tier == Tier.GATE


def test_low_fps_soft_gates():
    meta = VideoMetadata(duration_s=120, width=1920, height=1080, fps=15)
    result = check_metadata(meta)
    issue = next(i for i in result.issues if i.id == "fps_too_low")
    assert issue.tier == Tier.GATE


def test_normal_metadata_produces_no_issues():
    meta = VideoMetadata(duration_s=600, width=1920, height=1080, fps=30)
    result = check_metadata(meta)
    assert result.issues == []


def test_duration_is_bounded_below_for_severity():
    # A 0-length video should still produce a block, not crash
    meta = VideoMetadata(duration_s=0, width=1920, height=1080, fps=30)
    result = check_metadata(meta)
    assert any(i.id == "video_too_short" for i in result.issues)
```

- [ ] **Step 2: Run — expect 5 failures (module missing)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_metadata.py -v
```
Expected: 5 errors (import failure).

- [ ] **Step 3: Implement `metadata.py`**

```python
# analysis/rallycut/quality/metadata.py
"""Metadata-only quality checks (duration, resolution, framerate).

These are cheap and run at upload time (no decoding required beyond ffprobe).
Thresholds are conservative defaults; `calibrate_quality_checks.py` tunes
them against the 63-video GT and updates `DEFAULT_THRESHOLDS` if needed.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from rallycut.quality.types import CheckResult, Issue, Tier

# Thresholds — update after calibration (see analysis/scripts/calibrate_quality_checks.py).
MIN_DURATION_S = 10.0
MIN_WIDTH = 1280  # 720p
MIN_FPS = 24.0


@dataclass(frozen=True)
class VideoMetadata:
    duration_s: float
    width: int
    height: int
    fps: float

    @classmethod
    def from_ffprobe(cls, video_path: str) -> VideoMetadata:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1",
            video_path,
        ]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        num, den = kv.get("r_frame_rate", "0/1").split("/")
        fps = float(num) / float(den) if float(den) > 0 else 0.0
        return cls(
            duration_s=float(kv.get("duration", 0) or 0),
            width=int(kv.get("width", 0) or 0),
            height=int(kv.get("height", 0) or 0),
            fps=fps,
        )


def check_metadata(meta: VideoMetadata) -> CheckResult:
    issues: list[Issue] = []

    if meta.duration_s < MIN_DURATION_S:
        issues.append(Issue(
            id="video_too_short",
            tier=Tier.BLOCK,
            severity=1.0,
            message=f"Video is only {meta.duration_s:.1f}s long — we need at least {MIN_DURATION_S:.0f}s to find rallies.",
            source="upload",
            data={"durationS": meta.duration_s},
        ))

    if meta.width < MIN_WIDTH and meta.width > 0:
        issues.append(Issue(
            id="resolution_too_low",
            tier=Tier.GATE,
            severity=min(1.0, (MIN_WIDTH - meta.width) / MIN_WIDTH),
            message=f"Resolution is {meta.width}×{meta.height}. Recording in 720p or higher gives noticeably better tracking.",
            source="upload",
            data={"width": float(meta.width), "height": float(meta.height)},
        ))

    if 0 < meta.fps < MIN_FPS:
        issues.append(Issue(
            id="fps_too_low",
            tier=Tier.GATE,
            severity=min(1.0, (MIN_FPS - meta.fps) / MIN_FPS),
            message=f"Frame rate is {meta.fps:.1f} fps — {MIN_FPS:.0f} fps or higher gives better ball tracking.",
            source="upload",
            data={"fps": meta.fps},
        ))

    metrics = {
        "durationS": meta.duration_s,
        "width": float(meta.width),
        "height": float(meta.height),
        "fps": meta.fps,
    }
    return CheckResult(issues=issues, metrics=metrics)
```

- [ ] **Step 4: Run tests — all pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_metadata.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/quality/metadata.py analysis/tests/unit/test_quality_metadata.py
git commit -m "feat(quality): metadata check (duration, resolution, fps) with conservative thresholds"
```

---

## Task 3: Brightness check

**Files:**
- Modify: `analysis/rallycut/quality/metadata.py` (add `check_brightness`)
- Test: `analysis/tests/unit/test_quality_metadata.py` (extend)

- [ ] **Step 1: Add failing tests**

Append to `analysis/tests/unit/test_quality_metadata.py`:

```python
import numpy as np
from rallycut.quality.metadata import check_brightness


def test_dark_video_soft_gates():
    # mean luma around 0.05 (very dark)
    frames = [np.full((480, 640, 3), 12, dtype=np.uint8) for _ in range(5)]
    result = check_brightness(frames)
    issue = next(i for i in result.issues if i.id == "too_dark")
    assert issue.tier == Tier.GATE


def test_overexposed_video_soft_gates():
    frames = [np.full((480, 640, 3), 240, dtype=np.uint8) for _ in range(5)]
    result = check_brightness(frames)
    issue = next(i for i in result.issues if i.id == "overexposed")
    assert issue.tier == Tier.GATE


def test_normal_brightness_produces_no_issues():
    frames = [np.full((480, 640, 3), 128, dtype=np.uint8) for _ in range(5)]
    result = check_brightness(frames)
    assert result.issues == []
    assert 0.45 < result.metrics["meanLuma"] < 0.55
```

- [ ] **Step 2: Run — expect 3 failures**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_metadata.py -v
```
Expected: `ImportError: cannot import name 'check_brightness'`.

- [ ] **Step 3: Add `check_brightness`**

Add to `analysis/rallycut/quality/metadata.py`:

```python
import numpy as np

LUMA_DARK_THRESHOLD = 0.15
LUMA_BRIGHT_THRESHOLD = 0.85


def check_brightness(frames: list[np.ndarray]) -> CheckResult:
    """Compute mean luma across sampled BGR/RGB frames (uint8, shape HxWx3)."""
    if not frames:
        return CheckResult(issues=[], metrics={})

    # Use Rec. 601 luma coefficients on the assumption the frames are BGR or RGB;
    # either way the mean across three channels is a close approximation.
    lumas = [float(f.mean()) / 255.0 for f in frames]
    mean_luma = float(np.mean(lumas))

    issues: list[Issue] = []
    if mean_luma < LUMA_DARK_THRESHOLD:
        issues.append(Issue(
            id="too_dark",
            tier=Tier.GATE,
            severity=min(1.0, (LUMA_DARK_THRESHOLD - mean_luma) / LUMA_DARK_THRESHOLD),
            message="Video looks very dim — brighter lighting helps player and ball detection.",
            source="upload",
            data={"meanLuma": mean_luma},
        ))
    elif mean_luma > LUMA_BRIGHT_THRESHOLD:
        issues.append(Issue(
            id="overexposed",
            tier=Tier.GATE,
            severity=min(1.0, (mean_luma - LUMA_BRIGHT_THRESHOLD) / (1.0 - LUMA_BRIGHT_THRESHOLD)),
            message="Video looks overexposed — washed-out colors reduce tracking accuracy.",
            source="upload",
            data={"meanLuma": mean_luma},
        ))

    return CheckResult(issues=issues, metrics={"meanLuma": mean_luma})
```

- [ ] **Step 4: Run tests — all pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_metadata.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/quality/metadata.py analysis/tests/unit/test_quality_metadata.py
git commit -m "feat(quality): brightness check (too dark / overexposed) over sampled frames"
```

---

## Task 4: Camera geometry check (tilt + behind-baseline)

**Files:**
- Create: `analysis/rallycut/quality/camera_geometry.py`
- Test: `analysis/tests/unit/test_quality_camera_geometry.py`

Uses the existing court keypoint output: `detect_court()` returns 4 corners in normalized (x, y) coords, ordered top-left, top-right, bottom-right, bottom-left (this is the convention in `analysis/rallycut/court/`). From these we derive (a) tilt angle of baselines and (b) an "angle OK" boolean based on whether both baselines are visible and the net is roughly horizontal.

- [ ] **Step 1: Write failing tests**

```python
# analysis/tests/unit/test_quality_camera_geometry.py
import math

from rallycut.quality.camera_geometry import check_camera_geometry, CourtCorners
from rallycut.quality.types import Tier


def _square_corners() -> CourtCorners:
    # Perfect straight-on court, centered
    return CourtCorners(
        tl=(0.3, 0.4), tr=(0.7, 0.4),
        br=(0.8, 0.8), bl=(0.2, 0.8),
        confidence=0.9,
    )


def test_straight_court_passes():
    result = check_camera_geometry(_square_corners())
    assert result.issues == []


def test_tilted_court_produces_advisory():
    # Rotate baseline by 10 degrees
    angle = math.radians(10)
    c = math.cos(angle); s = math.sin(angle)
    def rot(p):
        return (0.5 + (p[0]-0.5)*c - (p[1]-0.5)*s, 0.5 + (p[0]-0.5)*s + (p[1]-0.5)*c)
    sq = _square_corners()
    corners = CourtCorners(
        tl=rot(sq.tl), tr=rot(sq.tr), br=rot(sq.br), bl=rot(sq.bl),
        confidence=0.9,
    )
    result = check_camera_geometry(corners)
    issue = next(i for i in result.issues if i.id == "video_rotated")
    assert issue.tier == Tier.ADVISORY


def test_no_court_hard_blocks():
    corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.1)
    result = check_camera_geometry(corners)
    issue = next(i for i in result.issues if i.id == "wrong_angle_or_not_volleyball")
    assert issue.tier == Tier.BLOCK


def test_side_view_hard_blocks():
    # Baselines nearly vertical (camera is beside the court, not behind the baseline)
    corners = CourtCorners(
        tl=(0.4, 0.2), tr=(0.45, 0.8),
        br=(0.55, 0.8), bl=(0.5, 0.2),
        confidence=0.9,
    )
    result = check_camera_geometry(corners)
    issue = next(i for i in result.issues if i.id == "wrong_angle_or_not_volleyball")
    assert issue.tier == Tier.BLOCK
```

- [ ] **Step 2: Run — expect 4 failures (module missing)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_camera_geometry.py -v
```

- [ ] **Step 3: Implement**

```python
# analysis/rallycut/quality/camera_geometry.py
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
```

- [ ] **Step 4: Run tests — all pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_camera_geometry.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/quality/camera_geometry.py analysis/tests/unit/test_quality_camera_geometry.py
git commit -m "feat(quality): camera geometry check (tilt advisory + behind-baseline block)"
```

---

## Task 5: Camera distance check (via fast YOLO sample)

**Files:**
- Create: `analysis/rallycut/quality/camera_distance.py`
- Test: `analysis/tests/unit/test_quality_camera_distance.py`

- [ ] **Step 1: Write failing tests with mocked detector**

```python
# analysis/tests/unit/test_quality_camera_distance.py
from rallycut.quality.camera_distance import check_camera_distance, Detection
from rallycut.quality.types import Tier


def test_players_large_in_frame_passes():
    # bbox height ~0.3 of frame — normal
    dets = [[Detection(x=0.5, y=0.5, w=0.1, h=0.3) for _ in range(4)] for _ in range(10)]
    result = check_camera_distance(dets)
    assert result.issues == []


def test_players_very_small_gates():
    # bbox height 0.05 — far away
    dets = [[Detection(x=0.5, y=0.5, w=0.02, h=0.05) for _ in range(4)] for _ in range(10)]
    result = check_camera_distance(dets)
    issue = next(i for i in result.issues if i.id == "camera_too_far")
    assert issue.tier == Tier.GATE


def test_empty_detections_produces_no_issue():
    # If no detections at all, this check can't decide — silent
    result = check_camera_distance([[]] * 10)
    assert result.issues == []
```

- [ ] **Step 2: Run — expect failures**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_camera_distance.py -v
```

- [ ] **Step 3: Implement**

```python
# analysis/rallycut/quality/camera_distance.py
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
```

- [ ] **Step 4: Run — pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_camera_distance.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/quality/camera_distance.py analysis/tests/unit/test_quality_camera_distance.py
git commit -m "feat(quality): camera-distance check (median bbox height gate)"
```

---

## Task 6: Crowd density check

**Files:**
- Create: `analysis/rallycut/quality/crowd_density.py`
- Test: `analysis/tests/unit/test_quality_crowd_density.py`

- [ ] **Step 1: Write failing tests**

```python
# analysis/tests/unit/test_quality_crowd_density.py
from rallycut.quality.crowd_density import check_crowd_density
from rallycut.quality.camera_distance import Detection
from rallycut.quality.types import Tier


def test_empty_court_passes():
    # 4 players, 0 spectators
    dets = [[Detection(0.5, 0.5, 0.1, 0.3) for _ in range(4)] for _ in range(10)]
    court_bbox = (0.2, 0.3, 0.8, 0.9)  # xmin, ymin, xmax, ymax
    result = check_crowd_density(dets, court_bbox)
    assert result.issues == []


def test_many_spectators_outside_court_gates():
    # 4 players on court + 12 people outside
    players = [Detection(0.5, 0.5, 0.1, 0.3) for _ in range(4)]
    spectators = [Detection(0.05 + i * 0.01, 0.1, 0.05, 0.15) for i in range(12)]
    dets = [players + spectators for _ in range(10)]
    court_bbox = (0.2, 0.3, 0.8, 0.9)
    result = check_crowd_density(dets, court_bbox)
    issue = next(i for i in result.issues if i.id == "crowded_scene")
    assert issue.tier == Tier.GATE


def test_people_inside_court_not_counted_as_crowd():
    # 6 people all inside court bbox — still a normal game
    dets = [[Detection(0.5, 0.5, 0.1, 0.3) for _ in range(6)] for _ in range(10)]
    court_bbox = (0.2, 0.3, 0.8, 0.9)
    result = check_crowd_density(dets, court_bbox)
    assert result.issues == []
```

- [ ] **Step 2: Run — expect failures**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_crowd_density.py -v
```

- [ ] **Step 3: Implement**

```python
# analysis/rallycut/quality/crowd_density.py
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
```

- [ ] **Step 4: Run — pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_crowd_density.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/quality/crowd_density.py analysis/tests/unit/test_quality_crowd_density.py
git commit -m "feat(quality): crowd-density check (avg non-court people per frame)"
```

---

## Task 7: Shakiness check (optical flow residual)

**Files:**
- Create: `analysis/rallycut/quality/shakiness.py`
- Test: `analysis/tests/unit/test_quality_shakiness.py`

- [ ] **Step 1: Write failing tests**

```python
# analysis/tests/unit/test_quality_shakiness.py
import numpy as np

from rallycut.quality.shakiness import check_shakiness
from rallycut.quality.types import Tier


def test_static_footage_passes():
    # Identical frames → zero flow → no issue
    frames = [np.full((120, 160, 3), 128, dtype=np.uint8) for _ in range(10)]
    result = check_shakiness(frames)
    assert result.issues == []


def test_high_jitter_footage_gates():
    rng = np.random.default_rng(42)
    # Random noise per frame → huge frame-to-frame residual
    frames = [rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8) for _ in range(10)]
    result = check_shakiness(frames)
    issue = next(i for i in result.issues if i.id == "shaky_camera")
    assert issue.tier == Tier.GATE
```

- [ ] **Step 2: Run — expect failures**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_shakiness.py -v
```

- [ ] **Step 3: Implement**

```python
# analysis/rallycut/quality/shakiness.py
"""Shakiness = mean pixel-level frame-to-frame L1 residual after a global
translation estimate. Crude but robust: catches handheld jitter without
needing optical-flow inference.

Threshold calibrated offline (see calibrate_quality_checks.py)."""
from __future__ import annotations

import statistics

import numpy as np

from rallycut.quality.types import CheckResult, Issue, Tier

SHAKINESS_GATE_THRESHOLD = 0.15  # mean normalized residual


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
```

- [ ] **Step 4: Run — pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_shakiness.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/quality/shakiness.py analysis/tests/unit/test_quality_shakiness.py
git commit -m "feat(quality): shakiness check (downsampled frame-to-frame residual)"
```

---

## Task 8: Beach-volleyball classifier (CLIP zero-shot)

**Files:**
- Create: `analysis/rallycut/quality/beach_vb_classifier.py`
- Test: `analysis/tests/unit/test_quality_beach_vb.py`

- [ ] **Step 1: Write failing tests with a stubbed CLIP backend**

```python
# analysis/tests/unit/test_quality_beach_vb.py
from rallycut.quality.beach_vb_classifier import classify_is_beach_vb, BeachVBProbabilities
from rallycut.quality.types import Tier


def test_high_beach_vb_probability_passes():
    probs = BeachVBProbabilities(beach_vb=0.93, indoor_vb=0.04, other=0.03)
    result = classify_is_beach_vb([probs] * 5)
    assert result.issues == []


def test_very_low_beach_vb_probability_blocks():
    probs = BeachVBProbabilities(beach_vb=0.05, indoor_vb=0.10, other=0.85)
    result = classify_is_beach_vb([probs] * 5)
    issue = next(i for i in result.issues if i.id == "wrong_angle_or_not_volleyball")
    assert issue.tier == Tier.BLOCK


def test_ambiguous_does_not_block():
    # We only block on confident-not-beach-VB. Ambiguous stays silent.
    probs = BeachVBProbabilities(beach_vb=0.4, indoor_vb=0.3, other=0.3)
    result = classify_is_beach_vb([probs] * 5)
    assert result.issues == []
```

- [ ] **Step 2: Run — expect failures**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_beach_vb.py -v
```

- [ ] **Step 3: Implement**

```python
# analysis/rallycut/quality/beach_vb_classifier.py
"""Zero-shot "is this beach volleyball" classifier using open-clip ViT-B/32.

The actual model inference is wrapped behind `embed_and_score_frames()` so
unit tests can pass precomputed probabilities without loading the model.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from rallycut.quality.types import CheckResult, Issue, Tier

BEACH_VB_BLOCK_THRESHOLD = 0.20  # average beach_vb prob below this = block

PROMPTS = {
    "beach_vb": "a beach volleyball match on sand",
    "indoor_vb": "an indoor volleyball match",
    "other": "a video that is not volleyball",
}


@dataclass(frozen=True)
class BeachVBProbabilities:
    beach_vb: float
    indoor_vb: float
    other: float


def classify_is_beach_vb(per_frame_probs: list[BeachVBProbabilities]) -> CheckResult:
    if not per_frame_probs:
        return CheckResult(issues=[], metrics={})

    avg_beach = statistics.mean(p.beach_vb for p in per_frame_probs)
    metrics = {"avgBeachVbProb": avg_beach}
    issues: list[Issue] = []
    if avg_beach < BEACH_VB_BLOCK_THRESHOLD:
        issues.append(Issue(
            id="wrong_angle_or_not_volleyball",
            tier=Tier.BLOCK,
            severity=1.0 - avg_beach,
            message="This doesn't look like a beach volleyball match. RallyCut is tuned for beach volleyball filmed from behind the baseline.",
            source="preflight",
            data={"avgBeachVbProb": avg_beach},
        ))
    return CheckResult(issues=issues, metrics=metrics)


def embed_and_score_frames(frames) -> list[BeachVBProbabilities]:
    """Run open-clip on each frame, return softmax over the three prompts.

    Kept out of unit tests — integration-tested via `rallycut preflight`.
    """
    import open_clip  # local import: heavy
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    labels = list(PROMPTS.keys())
    text_tokens = tokenizer([PROMPTS[k] for k in labels])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_tensors = torch.stack([preprocess(f) for f in frames])
        image_features = model.encode_image(image_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ text_features.T) * 100.0
        probs = logits.softmax(dim=-1).cpu().numpy()

    out: list[BeachVBProbabilities] = []
    for row in probs:
        kv = dict(zip(labels, row))
        out.append(BeachVBProbabilities(
            beach_vb=float(kv["beach_vb"]),
            indoor_vb=float(kv["indoor_vb"]),
            other=float(kv["other"]),
        ))
    return out
```

- [ ] **Step 4: Run — pass (model is not loaded for unit tests)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_beach_vb.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/quality/beach_vb_classifier.py analysis/tests/unit/test_quality_beach_vb.py
git commit -m "feat(quality): beach-VB zero-shot classifier (CLIP ViT-B/32)"
```

---

## Task 9: Runner + CLI entry point

**Files:**
- Create: `analysis/rallycut/quality/runner.py`
- Create: `analysis/rallycut/cli/commands/preflight.py`
- Modify: `analysis/rallycut/cli/app.py` (register the new command)
- Test: `analysis/tests/unit/test_quality_runner.py`

Before Step 1: locate the CLI app file and confirm command registration pattern.

- [ ] **Step 1: Locate CLI app module**

```bash
grep -rn "assess-quality" analysis/rallycut/cli/ --include='*.py'
```
Expected: finds the Typer app registering the old command. Note the file path and pattern — the new `preflight` command follows the same pattern.

- [ ] **Step 2: Write failing test for `run_full_preflight()`**

```python
# analysis/tests/unit/test_quality_runner.py
from unittest.mock import patch

import numpy as np

from rallycut.quality.runner import run_full_preflight
from rallycut.quality.metadata import VideoMetadata
from rallycut.quality.camera_geometry import CourtCorners
from rallycut.quality.beach_vb_classifier import BeachVBProbabilities


def test_runner_produces_serializable_report_with_good_video(tmp_path):
    meta = VideoMetadata(duration_s=600, width=1920, height=1080, fps=30)
    frames = [np.full((120, 160, 3), 128, dtype=np.uint8) for _ in range(10)]
    corners = CourtCorners(tl=(0.3, 0.4), tr=(0.7, 0.4), br=(0.8, 0.8), bl=(0.2, 0.8), confidence=0.9)
    probs = [BeachVBProbabilities(0.93, 0.04, 0.03)] * 5

    with patch("rallycut.quality.runner._load_video_inputs", return_value=(meta, frames, corners, [[], [], []], (0.2, 0.3, 0.8, 0.9), probs)):
        report = run_full_preflight("/tmp/fake.mp4", sample_seconds=60)

    d = report.to_dict()
    assert d["version"] == 2
    assert d["preflight"]["sampleSeconds"] == 60
    assert d["issues"] == []  # all checks should be silent on good inputs


def test_runner_surfaces_block_issue_for_bad_video(tmp_path):
    meta = VideoMetadata(duration_s=600, width=1920, height=1080, fps=30)
    frames = [np.full((120, 160, 3), 128, dtype=np.uint8) for _ in range(10)]
    bad_corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.1)
    probs = [BeachVBProbabilities(0.93, 0.04, 0.03)] * 5

    with patch("rallycut.quality.runner._load_video_inputs", return_value=(meta, frames, bad_corners, [[], [], []], (0.2, 0.3, 0.8, 0.9), probs)):
        report = run_full_preflight("/tmp/fake.mp4", sample_seconds=60)

    d = report.to_dict()
    assert any(i["id"] == "wrong_angle_or_not_volleyball" and i["tier"] == "block" for i in d["issues"])
```

- [ ] **Step 3: Run — expect import failures**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_runner.py -v
```

- [ ] **Step 4: Implement `runner.py`**

```python
# analysis/rallycut/quality/runner.py
"""Top-level preflight runner.

Loads inputs once (metadata, sampled frames, court corners, person detections,
beach-VB probabilities), then runs each check and merges their results into a
QualityReport.

`_load_video_inputs` is deliberately extracted so unit tests can patch it.
"""
from __future__ import annotations

from rallycut.quality.beach_vb_classifier import classify_is_beach_vb
from rallycut.quality.camera_distance import check_camera_distance
from rallycut.quality.camera_geometry import check_camera_geometry
from rallycut.quality.crowd_density import check_crowd_density
from rallycut.quality.metadata import check_brightness, check_metadata
from rallycut.quality.shakiness import check_shakiness
from rallycut.quality.types import QualityReport


def _load_video_inputs(video_path: str, sample_seconds: int):
    """Return (metadata, sampled_frames, court_corners, person_detections, court_bbox, clip_probs).

    Integration-only: wraps ffprobe, frame sampling, court-keypoint inference,
    a fast YOLO pass, and CLIP zero-shot scoring. Heavy; unit tests patch this.
    """
    from rallycut.detection.video_io import sample_frames  # existing
    from rallycut.court.keypoint_detector import detect_court_corners  # existing
    from rallycut.detection.yolo_person import detect_persons_in_frames  # existing (fast alias)
    from rallycut.quality.beach_vb_classifier import embed_and_score_frames
    from rallycut.quality.camera_geometry import CourtCorners
    from rallycut.quality.metadata import VideoMetadata

    meta = VideoMetadata.from_ffprobe(video_path)
    frames = sample_frames(video_path, n=10, max_seconds=sample_seconds)

    raw_corners = detect_court_corners(video_path, max_seconds=sample_seconds)
    corners = CourtCorners(
        tl=raw_corners.tl, tr=raw_corners.tr, br=raw_corners.br, bl=raw_corners.bl,
        confidence=raw_corners.confidence,
    )

    per_frame_dets = detect_persons_in_frames(frames)
    court_bbox = (
        min(corners.tl[0], corners.bl[0]),
        min(corners.tl[1], corners.tr[1]),
        max(corners.tr[0], corners.br[0]),
        max(corners.bl[1], corners.br[1]),
    )

    clip_probs = embed_and_score_frames(frames[:5])

    return meta, frames, corners, per_frame_dets, court_bbox, clip_probs


def run_full_preflight(video_path: str, sample_seconds: int = 60) -> QualityReport:
    meta, frames, corners, dets, court_bbox, clip_probs = _load_video_inputs(
        video_path, sample_seconds=sample_seconds
    )
    results = [
        check_metadata(meta),
        check_brightness(frames),
        check_camera_geometry(corners),
        check_camera_distance(dets),
        check_crowd_density(dets, court_bbox),
        check_shakiness(frames),
        classify_is_beach_vb(clip_probs),
    ]
    return QualityReport.from_checks(
        results, source="preflight", sample_seconds=sample_seconds, duration_ms=int(meta.duration_s * 1000)
    )
```

- [ ] **Step 5: Implement `preflight` CLI command**

```python
# analysis/rallycut/cli/commands/preflight.py
"""`rallycut preflight <video>` — run all quality checks, emit JSON."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from rallycut.quality.runner import run_full_preflight


def preflight(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    sample_seconds: int = typer.Option(60, "--sample-seconds"),
    as_json: bool = typer.Option(True, "--json/--no-json"),
    quiet: bool = typer.Option(False, "--quiet"),
):
    """Run the full preflight quality check and print a JSON QualityReport."""
    report = run_full_preflight(str(video), sample_seconds=sample_seconds)
    d = report.to_dict()
    if as_json:
        typer.echo(json.dumps(d, indent=None if quiet else 2))
    else:
        for issue in d["issues"]:
            typer.echo(f"[{issue['tier'].upper()}] {issue['message']}")
```

- [ ] **Step 6: Register the command in `analysis/rallycut/cli/app.py`**

Using the pattern found in Step 1, add a registration line for `preflight`. Example (adjust to the project's actual pattern):

```python
# analysis/rallycut/cli/app.py  — add near existing command registrations
from rallycut.cli.commands.preflight import preflight
app.command(name="preflight")(preflight)
```

- [ ] **Step 7: Run unit tests — all pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_quality_runner.py -v
```
Expected: 2 passed.

- [ ] **Step 8: Smoke-test CLI against a known GT video**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run rallycut preflight <path-to-any-labeled-video> --json
```
Expected: a JSON blob with `version: 2`, `issues: []` (or at most advisories), and a non-null `preflight.durationMs`.

- [ ] **Step 9: Commit**

```bash
git add analysis/rallycut/quality/runner.py analysis/rallycut/cli/commands/preflight.py analysis/rallycut/cli/app.py analysis/tests/unit/test_quality_runner.py
git commit -m "feat(quality): preflight runner + rallycut preflight CLI"
```

---

## Task 10: Calibration harness against 63-video GT

**Files:**
- Create: `analysis/scripts/calibrate_quality_checks.py`
- Create: `analysis/reports/quality_calibration_2026_04_15.json` (run output)

**Goal:** For every candidate check, compute `P(pipeline_failure | check fires)` vs. `P(pipeline_failure | !fires)`; table the lift for a sweep of thresholds; output the recommended threshold per check. **Any check that can't clear 3× lift is dropped or deferred to Project C.**

This task is run-once and produces a committed report. It doesn't follow the strict TDD structure of earlier tasks.

- [ ] **Step 1: Write the harness**

```python
# analysis/scripts/calibrate_quality_checks.py
"""Calibrate quality-check thresholds against the 63-video GT.

For each check, sweep candidate thresholds. For each threshold, compute:
  lift = P(failure | fires) / P(failure | !fires)
where `failure` = (HOTA < 0.60) OR (action_acc < 0.75) OR (score_acc < 0.75).

Emit `analysis/reports/quality_calibration_<date>.json` with the best threshold
per check and its lift. Print a human-readable table.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

# The GT loader already exists — reuse it
from rallycut.evaluation.gt_loader import load_all_gt_videos  # existing
from rallycut.evaluation.metrics import compute_video_metrics  # existing
from rallycut.quality.runner import _load_video_inputs
from rallycut.quality.metadata import check_brightness, check_metadata
from rallycut.quality.camera_geometry import check_camera_geometry
from rallycut.quality.camera_distance import check_camera_distance
from rallycut.quality.crowd_density import check_crowd_density
from rallycut.quality.shakiness import check_shakiness
from rallycut.quality.beach_vb_classifier import classify_is_beach_vb

FAIL = lambda m: (m.hota < 0.60) or (m.action_acc < 0.75) or (m.score_acc < 0.75)  # noqa: E731
MIN_LIFT = 3.0


@dataclass
class CheckCalibration:
    check_id: str
    metric_key: str
    candidate_thresholds: list[float]
    best_threshold: float | None
    best_lift: float
    n_fires: int
    n_total: int
    false_positive_rate: float
    recommendation: str  # "ship" | "drop" | "defer_to_C"


def calibrate(check_fn, metric_key, candidate_thresholds, direction, video_samples, failures) -> CheckCalibration:
    """direction = '<' if firing means metric < threshold, else '>'."""
    best_lift, best_t = 0.0, None
    n_fires_best = 0
    for t in candidate_thresholds:
        fires = [(metric_key in m and ((m[metric_key] < t) if direction == "<" else (m[metric_key] > t))) for m in video_samples]
        n_fires = sum(fires)
        n_total = len(fires)
        if n_fires == 0 or n_fires == n_total:
            continue
        p_fail_fires = sum(1 for i, f in enumerate(fires) if f and failures[i]) / n_fires
        p_fail_nofires = sum(1 for i, f in enumerate(fires) if (not f) and failures[i]) / (n_total - n_fires)
        lift = p_fail_fires / max(p_fail_nofires, 1e-6)
        if lift > best_lift:
            best_lift, best_t, n_fires_best = lift, t, n_fires

    if best_lift < MIN_LIFT:
        recommendation = "drop"
    else:
        recommendation = "ship"

    fpr = 0.0
    if best_t is not None:
        fires = [(metric_key in m and ((m[metric_key] < best_t) if direction == "<" else (m[metric_key] > best_t))) for m in video_samples]
        n_fires = sum(fires)
        if n_fires > 0:
            fpr = sum(1 for i, f in enumerate(fires) if f and not failures[i]) / n_fires

    return CheckCalibration(
        check_id=check_fn.__name__,
        metric_key=metric_key,
        candidate_thresholds=list(candidate_thresholds),
        best_threshold=best_t,
        best_lift=best_lift,
        n_fires=n_fires_best,
        n_total=len(video_samples),
        false_positive_rate=fpr,
        recommendation=recommendation,
    )


def main():
    gt_videos = load_all_gt_videos()
    print(f"Loaded {len(gt_videos)} GT videos")
    assert len(gt_videos) >= 50, f"Expected ≥50 GT videos, got {len(gt_videos)}"

    all_metrics: list[dict] = []
    failures: list[bool] = []

    for i, video in enumerate(gt_videos):
        print(f"[{i+1}/{len(gt_videos)}] {video.id}: extracting features", flush=True)
        try:
            inputs = _load_video_inputs(video.path, sample_seconds=60)
        except Exception as exc:
            print(f"  SKIP ({exc})")
            continue
        meta, frames, corners, dets, court_bbox, probs = inputs
        m: dict[str, float] = {}
        m.update(check_metadata(meta).metrics)
        m.update(check_brightness(frames).metrics)
        m.update(check_camera_geometry(corners).metrics)
        m.update(check_camera_distance(dets).metrics)
        m.update(check_crowd_density(dets, court_bbox).metrics)
        m.update(check_shakiness(frames).metrics)
        m.update(classify_is_beach_vb(probs).metrics)
        all_metrics.append(m)

        metrics = compute_video_metrics(video.id)
        fail = FAIL(metrics)
        failures.append(fail)
        print(f"  HOTA={metrics.hota:.2f} action_acc={metrics.action_acc:.2f} score_acc={metrics.score_acc:.2f} fail={fail}")

    # Define which metric key and direction each check uses
    sweeps = [
        ("camera_too_far", "medianBboxHeight", [0.05, 0.08, 0.10, 0.12, 0.15], "<"),
        ("crowded_scene", "avgNonCourtPersons", [2, 3, 5, 8, 12], ">"),
        ("shaky_camera", "meanResidual", [0.05, 0.10, 0.15, 0.20, 0.30], ">"),
        ("video_rotated", "tiltDeg", [3, 5, 8, 12, 20], ">"),
        ("too_dark", "meanLuma", [0.08, 0.12, 0.15, 0.20], "<"),
        ("overexposed", "meanLuma", [0.80, 0.85, 0.90, 0.95], ">"),
        ("wrong_angle_or_not_volleyball:clip", "avgBeachVbProb", [0.10, 0.20, 0.30, 0.40], "<"),
        ("wrong_angle_or_not_volleyball:court", "courtConfidence", [0.3, 0.5, 0.6, 0.75], "<"),
    ]

    results = []
    for check_id, key, thresholds, direction in sweeps:
        cal = CheckCalibration(
            check_id=check_id, metric_key=key,
            candidate_thresholds=[], best_threshold=None, best_lift=0.0,
            n_fires=0, n_total=0, false_positive_rate=0.0, recommendation="",
        )
        cal = calibrate(lambda m: None, key, thresholds, direction, all_metrics, failures)
        cal.check_id = check_id
        results.append(cal)

    report = {
        "date": str(date.today()),
        "n_videos": len(all_metrics),
        "n_failures": sum(failures),
        "min_lift_bar": MIN_LIFT,
        "checks": [asdict(r) for r in results],
    }
    Path("analysis/reports").mkdir(exist_ok=True)
    out = Path(f"analysis/reports/quality_calibration_{date.today()}.json")
    out.write_text(json.dumps(report, indent=2))
    print("\n=== RESULTS ===")
    for r in results:
        print(f"  {r.check_id:40s}  best_t={r.best_threshold}  lift={r.best_lift:.2f}  fpr={r.false_positive_rate:.2f}  -> {r.recommendation}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run calibration (background; ~30-60 min)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python scripts/calibrate_quality_checks.py 2>&1 | tee reports/quality_calibration_run.log
```
Use `run_in_background: true`. Expected: per-video progress lines (per project rule: no truncated output), then a final table. If a video errors, skip and continue. If >10% of videos fail to load, stop and debug inputs before proceeding.

- [ ] **Step 3: Review results**

Open `analysis/reports/quality_calibration_<date>.json`. For each check:
- If `recommendation == "drop"`: document in the plan comments and remove the check from `runner.py` (issue stops being emitted even if metric is still computed). Commit this change as a separate "drop <check> — calibration showed <lift>× lift, below 3× bar" commit.
- If `recommendation == "ship"`: update the threshold constant in the check's module file (`metadata.py`, `camera_distance.py`, etc.) to the calibrated `best_threshold`.

- [ ] **Step 4: Re-run unit tests to confirm calibrated thresholds don't break existing fixtures**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/ -k quality -v
```
If any unit test fails because a fixture now sits on the wrong side of a calibrated threshold, update the fixture (not the threshold) to be unambiguously in-range or out-of-range.

- [ ] **Step 5: Commit**

```bash
git add analysis/scripts/calibrate_quality_checks.py analysis/reports/quality_calibration_*.json analysis/rallycut/quality/
git commit -m "feat(quality): calibrate check thresholds against 63-video GT"
```

---

## Task 11: Prisma migration — add `qualityReportJson`, `VideoStatus.REJECTED`, drop `characteristicsJson`

**Files:**
- Modify: `api/prisma/schema.prisma`
- Create: `api/prisma/migrations/<timestamp>_quality_report/migration.sql`

- [ ] **Step 1: Modify schema**

Locate `model Video` and `enum VideoStatus` in `api/prisma/schema.prisma`. Apply:

```diff
 enum VideoStatus {
   PENDING
   UPLOADED
   DETECTING
   DETECTED
+  REJECTED
   ERROR
 }
```

```diff
 model Video {
   ...
-  characteristicsJson Json?
+  qualityReportJson   Json?
   ...
 }
```

- [ ] **Step 2: Generate migration**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api && npx prisma migrate dev --name quality_report --create-only
```
Expected: creates a migration file with `ALTER TABLE "Video" ADD COLUMN "qualityReportJson" JSONB` and `DROP COLUMN "characteristicsJson"`. Open the generated SQL file.

- [ ] **Step 3: Rewrite migration for safe data preservation**

Replace the generated SQL with a non-destructive data-preserving version:

```sql
-- Add new column
ALTER TABLE "Video" ADD COLUMN "qualityReportJson" JSONB;

-- Migrate useful fields (brightness, resolution) out of characteristicsJson
UPDATE "Video"
SET "qualityReportJson" = jsonb_build_object(
  'version', 2,
  'issues', '[]'::jsonb,
  'brightness', "characteristicsJson"->'brightness',
  'resolution', "characteristicsJson"->'resolution'
)
WHERE "characteristicsJson" IS NOT NULL;

-- Add REJECTED enum value
ALTER TYPE "VideoStatus" ADD VALUE 'REJECTED';

-- Drop old column
ALTER TABLE "Video" DROP COLUMN "characteristicsJson";
```

- [ ] **Step 4: Apply migration locally**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api && npx prisma migrate dev && npx prisma generate
```
Expected: migration applies cleanly; Prisma client regenerates with `qualityReportJson` and the new enum value.

- [ ] **Step 5: Spot-check data preservation via psql**

```bash
psql postgresql://postgres:postgres@localhost:5436/rallycut -c "SELECT id, \"qualityReportJson\"->'brightness' AS brightness FROM \"Video\" WHERE \"qualityReportJson\" IS NOT NULL LIMIT 5;"
```
Expected: rows where `brightness` is present (non-null) in the new column for videos that had it before.

- [ ] **Step 6: Commit**

```bash
git add api/prisma/schema.prisma api/prisma/migrations/
git commit -m "feat(db): migrate characteristicsJson -> qualityReportJson; add VideoStatus.REJECTED"
```

---

## Task 12: Rewrite `qualityService.ts`

**Files:**
- Modify: `api/src/services/qualityService.ts` (near-total rewrite)
- Create: `api/tests/qualityService.test.ts`

- [ ] **Step 1: Write failing vitest for report merging**

```typescript
// api/tests/qualityService.test.ts
import { describe, it, expect } from 'vitest';
import { mergeQualityReports, pickTopIssues } from '../src/services/qualityService';

describe('mergeQualityReports', () => {
  it('combines upload + preflight issues and sorts by tier then severity', () => {
    const upload = {
      version: 2,
      issues: [
        { id: 'too_dark', tier: 'gate', severity: 0.4, message: '', source: 'upload', detectedAt: '', data: {} },
      ],
    };
    const preflight = {
      version: 2,
      issues: [
        { id: 'wrong_angle_or_not_volleyball', tier: 'block', severity: 1.0, message: '', source: 'preflight', detectedAt: '', data: {} },
        { id: 'camera_too_far', tier: 'gate', severity: 0.9, message: '', source: 'preflight', detectedAt: '', data: {} },
      ],
    };
    const merged = mergeQualityReports([upload, preflight]);
    expect(merged.issues.map((i) => i.id)).toEqual(['wrong_angle_or_not_volleyball', 'camera_too_far', 'too_dark']);
  });
});

describe('pickTopIssues', () => {
  it('caps display to 3 items', () => {
    const issues = Array.from({ length: 6 }, (_, i) => ({
      id: `i${i}`, tier: 'advisory' as const, severity: 1 - i * 0.1,
      message: '', source: 'preflight' as const, detectedAt: '', data: {},
    }));
    expect(pickTopIssues(issues)).toHaveLength(3);
  });
});
```

- [ ] **Step 2: Run — expect module/export failures**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api && npm test -- qualityService
```

- [ ] **Step 3: Rewrite `qualityService.ts`**

Replace the file's contents with the new service. The runner logic (spawning Python) mirrors the old `runQualityAssessmentCli` / `runCourtDetectionCli` — keep that infrastructure, but swap the command to `rallycut preflight`.

```typescript
// api/src/services/qualityService.ts
/**
 * Quality report service — upload-time and preflight-time video quality checks.
 *
 * Upload-time checks are fast (metadata + light sampling) and run on
 * `POST /v1/videos/:id/confirm`. Preflight checks are heavy (CLIP, court
 * keypoints, YOLO sample) and run on `POST /v1/videos/:id/assess-quality`
 * just before detection.
 *
 * Both write to `Video.qualityReportJson`. The banner UI picks the top 3.
 */
import { spawn } from 'child_process';
import { createWriteStream } from 'fs';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { fileURLToPath } from 'url';

import { Prisma, VideoStatus } from '@prisma/client';
import { prisma } from '../lib/prisma.js';
import { generateDownloadUrl } from '../lib/s3.js';
import { NotFoundError } from '../middleware/errorHandler.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ANALYSIS_DIR = path.resolve(__dirname, '../../../analysis');
const TEMP_DIR = path.join(os.tmpdir(), 'rallycut-quality');
const PREFLIGHT_TIMEOUT_MS = 120_000;

export type Tier = 'block' | 'gate' | 'advisory';

export interface Issue {
  id: string;
  tier: Tier;
  severity: number;
  message: string;
  source: 'preview' | 'upload' | 'preflight' | 'tracking';
  detectedAt: string;
  data?: Record<string, number>;
}

export interface QualityReport {
  version: 2;
  issues: Issue[];
  preflight?: { ranAt: string; sampleSeconds: number; durationMs: number } | null;
  brightness?: number | null;
  resolution?: { width: number; height: number } | null;
}

const TIER_ORDER: Record<Tier, number> = { block: 0, gate: 1, advisory: 2 };

export function pickTopIssues(issues: Issue[], max = 3): Issue[] {
  return [...issues]
    .sort((a, b) => TIER_ORDER[a.tier] - TIER_ORDER[b.tier] || b.severity - a.severity || a.id.localeCompare(b.id))
    .slice(0, max);
}

export function mergeQualityReports(reports: Array<Partial<QualityReport>>): QualityReport {
  const all: Issue[] = reports.flatMap((r) => r.issues ?? []);
  const brightness = reports.map((r) => r.brightness).find((v) => v != null) ?? null;
  const resolution = reports.map((r) => r.resolution).find((v) => v != null) ?? null;
  const preflight = reports.map((r) => r.preflight).find((v) => v != null) ?? null;
  return {
    version: 2,
    issues: pickTopIssues(all),
    preflight,
    brightness,
    resolution,
  };
}

export async function runPreflightChecks(videoId: string, userId: string): Promise<QualityReport> {
  const video = await prisma.video.findFirst({ where: { id: videoId, userId, deletedAt: null } });
  if (!video) throw new NotFoundError('Video', videoId);

  await fs.mkdir(TEMP_DIR, { recursive: true });
  const suffix = `_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const ext = path.extname(video.filename || '.mp4');
  const localPath = path.join(TEMP_DIR, `preflight_${videoId}${suffix}${ext}`);

  try {
    await downloadFromS3(video.s3Key, localPath);
    const report = await runPreflightCli(localPath);

    const existing = (video.qualityReportJson as Partial<QualityReport> | null) ?? {};
    const merged = mergeQualityReports([existing, report]);
    await prisma.video.update({
      where: { id: videoId },
      data: { qualityReportJson: merged as unknown as Prisma.InputJsonValue },
    });

    // Hard-block handling — if preflight found a block-tier issue, mark video REJECTED
    if (merged.issues.some((i) => i.tier === 'block')) {
      await prisma.video.update({
        where: { id: videoId },
        data: { status: VideoStatus.REJECTED },
      });
    }

    return merged;
  } finally {
    await fs.unlink(localPath).catch(() => {});
  }
}

export async function saveUploadReport(videoId: string, report: Partial<QualityReport>) {
  const existing = await prisma.video.findUnique({ where: { id: videoId } });
  const prior = (existing?.qualityReportJson as Partial<QualityReport> | null) ?? {};
  const merged = mergeQualityReports([prior, report]);
  await prisma.video.update({
    where: { id: videoId },
    data: { qualityReportJson: merged as unknown as Prisma.InputJsonValue },
  });
}

function runPreflightCli(videoPath: string): Promise<QualityReport> {
  return new Promise((resolve, reject) => {
    const args = ['run', 'rallycut', 'preflight', videoPath, '--json', '--quiet'];
    const child = spawn('uv', args, { cwd: ANALYSIS_DIR, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env } });

    let stdout = '';
    let stderr = '';
    let settled = false;

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { child.kill(); } catch { /* ignore */ }
      reject(new Error('Preflight timed out'));
    }, PREFLIGHT_TIMEOUT_MS);

    child.stdout?.on('data', (d: Buffer) => { stdout += d.toString(); });
    child.stderr?.on('data', (d: Buffer) => {
      stderr += d.toString();
      const line = d.toString().trim();
      if (line) console.log(`[PREFLIGHT:PY] ${line}`);
    });
    child.on('error', (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`Preflight failed to start: ${err.message}`));
    });
    child.on('exit', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`Preflight exited ${code}: ${(stderr || stdout).slice(-500)}`));
        return;
      }
      try {
        const json = JSON.parse(stdout.match(/\{[\s\S]*\}/)![0]);
        resolve(json as QualityReport);
      } catch (e) {
        reject(new Error(`Preflight output parse failed: ${e}`));
      }
    });
  });
}

async function downloadFromS3(s3Key: string, destPath: string): Promise<void> {
  const url = await generateDownloadUrl(s3Key);
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`S3 download failed: ${response.status}`);
  await pipeline(Readable.fromWeb(response.body as never), createWriteStream(destPath));
}

// Keep getAnalysisPipelineStatus and savePlayerMatchingGt functions unchanged from the old file
// (they don't depend on characteristicsJson). Copy them over verbatim and update only the
// `quality` shape in `getAnalysisPipelineStatus` to read from qualityReportJson.
export { getAnalysisPipelineStatus, savePlayerMatchingGt, getPlayerMatchingGt } from './qualityService-legacy-helpers.js';
```

The `qualityService-legacy-helpers.ts` side-file holds `getAnalysisPipelineStatus`, `savePlayerMatchingGt`, and `getPlayerMatchingGt` verbatim from the old file with one change: in `getAnalysisPipelineStatus` replace the `characteristicsJson` read with:

```typescript
const qr = video.qualityReportJson as QualityReport | null;
const qualityAssessment = qr
  ? { expectedQuality: 1 - (qr.issues[0]?.severity ?? 0), warnings: qr.issues.map((i) => i.message) }
  : null;
```

- [ ] **Step 4: Run vitest — expect pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api && npm test -- qualityService
```
Expected: 2 passed.

- [ ] **Step 5: Run type check + lint**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api && npx tsc --noEmit
```
Fix any type errors surfaced by removing `characteristicsJson` — this is expected; other files (processingService, playerTrackingService, routes/videos) will be updated in later tasks.

If type errors are from files not in this task's scope, add a temporary `// @ts-expect-error — cleaned up in task N` comment **only if necessary to unblock**; prefer inlining the cleanup in the next task instead.

- [ ] **Step 6: Commit**

```bash
git add api/src/services/qualityService.ts api/src/services/qualityService-legacy-helpers.ts api/tests/qualityService.test.ts
git commit -m "refactor(api): rewrite qualityService around QualityReport + preflight CLI"
```

---

## Task 13: New `POST /v1/videos/preflight-preview` route

**Files:**
- Modify: `api/src/routes/videos.ts`
- Modify: `api/src/services/qualityService.ts` (add `runPreviewChecks`)
- Create: `api/tests/preflightPreview.test.ts`

The pre-upload preview endpoint accepts a small multipart payload of 5 JPEG frames + video metadata, runs **only checks 5 and 6** (beach-VB + camera geometry) using CLIP + court keypoints on the frames, and returns `{ pass: boolean, issues: Issue[] }`. It does NOT persist anything to the DB (no video row exists yet). If the response is `pass: false`, the frontend cancels the upload flow.

- [ ] **Step 1: Write failing route test**

```typescript
// api/tests/preflightPreview.test.ts
import { describe, it, expect } from 'vitest';
import request from 'supertest';
import { app } from '../src/index'; // assume app export; create one if missing (task notes below)
import fs from 'fs';

describe('POST /v1/videos/preflight-preview', () => {
  it('returns pass=true for a beach-volleyball thumbnail', async () => {
    const frame = fs.readFileSync('api/tests/fixtures/beach_vb_frame.jpg');
    const res = await request(app)
      .post('/v1/videos/preflight-preview')
      .set('X-Visitor-Id', '00000000-0000-0000-0000-000000000001')
      .attach('frames', frame, 'f1.jpg')
      .attach('frames', frame, 'f2.jpg')
      .attach('frames', frame, 'f3.jpg')
      .field('width', '1920')
      .field('height', '1080')
      .field('durationS', '300');
    expect(res.status).toBe(200);
    expect(res.body.pass).toBe(true);
  });

  it('returns pass=false with a block issue for a non-VB thumbnail', async () => {
    const frame = fs.readFileSync('api/tests/fixtures/non_vb_frame.jpg');
    const res = await request(app)
      .post('/v1/videos/preflight-preview')
      .set('X-Visitor-Id', '00000000-0000-0000-0000-000000000001')
      .attach('frames', frame, 'f1.jpg')
      .field('width', '1920').field('height', '1080').field('durationS', '300');
    expect(res.status).toBe(200);
    expect(res.body.pass).toBe(false);
    expect(res.body.issues.some((i: any) => i.tier === 'block')).toBe(true);
  });
});
```

Supply real fixture frames in `api/tests/fixtures/` by sampling a single JPEG from a known-good GT video and an unrelated video (e.g., any non-sport content).

- [ ] **Step 2: Run — expect failure (route not registered)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api && npm test -- preflightPreview
```

- [ ] **Step 3: Add route + handler**

In `api/src/routes/videos.ts`, add:

```typescript
import multer from 'multer';
const previewUpload = multer({ limits: { fileSize: 2 * 1024 * 1024, files: 10 } });

router.post(
  '/preflight-preview',
  requireUser,
  previewUpload.array('frames', 10),
  async (req, res, next) => {
    try {
      const files = (req.files as Express.Multer.File[] | undefined) ?? [];
      if (files.length === 0) {
        return res.status(400).json({ error: 'At least one frame is required' });
      }
      const width = Number(req.body.width);
      const height = Number(req.body.height);
      const durationS = Number(req.body.durationS);
      const result = await qualityService.runPreviewChecks({
        frames: files.map((f) => f.buffer),
        width, height, durationS,
      });
      res.json({ pass: !result.issues.some((i) => i.tier === 'block'), issues: result.issues });
    } catch (err) {
      next(err);
    }
  },
);
```

In `qualityService.ts`, add:

```typescript
export interface PreviewInput {
  frames: Buffer[];
  width: number;
  height: number;
  durationS: number;
}

export async function runPreviewChecks(input: PreviewInput): Promise<{ issues: Issue[] }> {
  // Write frames to a temp dir, call `rallycut preview-check <dir>` (see Task 13b), parse JSON.
  await fs.mkdir(TEMP_DIR, { recursive: true });
  const dir = path.join(TEMP_DIR, `preview_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);
  await fs.mkdir(dir, { recursive: true });
  try {
    await Promise.all(
      input.frames.map((buf, i) => fs.writeFile(path.join(dir, `frame_${i}.jpg`), buf)),
    );
    const issues = await runPreviewCli(dir, input);
    return { issues };
  } finally {
    await fs.rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}

function runPreviewCli(dir: string, meta: { width: number; height: number; durationS: number }): Promise<Issue[]> {
  return new Promise((resolve, reject) => {
    const args = [
      'run', 'rallycut', 'preview-check', dir,
      '--width', String(meta.width),
      '--height', String(meta.height),
      '--duration-s', String(meta.durationS),
      '--json', '--quiet',
    ];
    const child = spawn('uv', args, { cwd: ANALYSIS_DIR, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    const timer = setTimeout(() => { try { child.kill(); } catch {} reject(new Error('Preview timed out')); }, 20_000);
    child.stdout?.on('data', (d: Buffer) => { stdout += d.toString(); });
    child.stderr?.on('data', (d: Buffer) => { stderr += d.toString(); });
    child.on('exit', (code) => {
      clearTimeout(timer);
      if (code !== 0) return reject(new Error(`preview-check exited ${code}: ${stderr.slice(-500)}`));
      try {
        const m = stdout.match(/\{[\s\S]*\}/);
        if (!m) return reject(new Error('No JSON from preview-check'));
        const parsed = JSON.parse(m[0]);
        resolve(parsed.issues ?? []);
      } catch (e) { reject(e); }
    });
  });
}
```

- [ ] **Step 4: Add the `preview-check` Python CLI command**

Create `analysis/rallycut/cli/commands/preview_check.py`:

```python
"""`rallycut preview-check <dir>` — run beach-VB + camera-geometry on a
directory of JPEG frames (no video decoding required). Used by the web
pre-upload gate before the upload commits.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from PIL import Image

from rallycut.court.keypoint_detector import detect_court_from_frames
from rallycut.quality.beach_vb_classifier import embed_and_score_frames, classify_is_beach_vb
from rallycut.quality.camera_geometry import CourtCorners, check_camera_geometry
from rallycut.quality.types import QualityReport


def preview_check(
    frames_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    width: int = typer.Option(..., "--width"),
    height: int = typer.Option(..., "--height"),
    duration_s: float = typer.Option(..., "--duration-s"),
    as_json: bool = typer.Option(True, "--json/--no-json"),
    quiet: bool = typer.Option(False, "--quiet"),
):
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    if not frames:
        typer.echo(json.dumps({"issues": []}))
        raise typer.Exit(0)

    probs = embed_and_score_frames(frames)
    vb_result = classify_is_beach_vb(probs)

    # Court keypoints from the first frame only (fast)
    raw = detect_court_from_frames([frames[0]])
    geom_result = check_camera_geometry(CourtCorners(
        tl=raw.tl, tr=raw.tr, br=raw.br, bl=raw.bl, confidence=raw.confidence,
    ))

    report = QualityReport.from_checks([vb_result, geom_result], source="preview", duration_ms=int(duration_s * 1000))
    typer.echo(json.dumps(report.to_dict()))
```

Register it in `analysis/rallycut/cli/app.py` the same way as `preflight`.

`detect_court_from_frames` needs to accept a list of PIL images — if the existing `detect_court_corners` only takes a video path, add a thin helper in `analysis/rallycut/court/keypoint_detector.py` that runs the model on a list of PIL images. Keep the helper small; reuse the existing model-loading code.

- [ ] **Step 5: Run vitest — pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api && npm test -- preflightPreview
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add api/src/routes/videos.ts api/src/services/qualityService.ts api/tests/preflightPreview.test.ts api/tests/fixtures/ analysis/rallycut/cli/commands/preview_check.py analysis/rallycut/cli/app.py analysis/rallycut/court/keypoint_detector.py
git commit -m "feat(api): POST /v1/videos/preflight-preview + rallycut preview-check CLI"
```

---

## Task 14: Frontend — client-side preview frame extractor

**Files:**
- Create: `web/src/utils/extractPreviewFrames.ts`
- Create: `web/src/utils/__tests__/extractPreviewFrames.test.ts` (if vitest is wired up; otherwise verify manually)

Strategy: use the browser's `HTMLVideoElement` with a blob URL + `<canvas>` drawing for a zero-dependency implementation (ffmpeg.wasm is overkill for extracting 5 thumbnails).

- [ ] **Step 1: Implement extractor**

```typescript
// web/src/utils/extractPreviewFrames.ts
/**
 * Extract N evenly-spaced JPEG thumbnails from a local video File.
 * No external deps: uses <video> + <canvas>. ~100-300ms for a typical clip.
 */
export async function extractPreviewFrames(file: File, count = 5, maxWidth = 640): Promise<Blob[]> {
  const url = URL.createObjectURL(file);
  try {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Failed to load video metadata'));
    });
    const duration = video.duration || 0;
    if (duration < 1) throw new Error('Video too short to extract preview frames');

    const scale = Math.min(1, maxWidth / video.videoWidth);
    const w = Math.round(video.videoWidth * scale);
    const h = Math.round(video.videoHeight * scale);
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas context unavailable');

    const blobs: Blob[] = [];
    for (let i = 0; i < count; i++) {
      const t = ((i + 0.5) / count) * duration;
      await new Promise<void>((resolve, reject) => {
        const onSeeked = () => { video.removeEventListener('seeked', onSeeked); resolve(); };
        video.addEventListener('seeked', onSeeked, { once: true });
        video.addEventListener('error', () => reject(new Error('Video seek failed')), { once: true });
        video.currentTime = t;
      });
      ctx.drawImage(video, 0, 0, w, h);
      const blob: Blob = await new Promise((resolve, reject) => {
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/jpeg', 0.85);
      });
      blobs.push(blob);
    }
    return blobs;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export interface PreviewMetadata {
  width: number;
  height: number;
  durationS: number;
}

export async function readVideoMetadata(file: File): Promise<PreviewMetadata> {
  const url = URL.createObjectURL(file);
  try {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Failed to read metadata'));
    });
    return { width: video.videoWidth, height: video.videoHeight, durationS: video.duration };
  } finally {
    URL.revokeObjectURL(url);
  }
}
```

- [ ] **Step 2: Manual verification**

```bash
cd /Users/mario/Personal/Projects/RallyCut/web && npm run dev
```
Open the upload page in a browser, select a known-good video, observe (via a temporary `console.log` inside the function) that 5 blobs are produced within a few seconds. Remove the log before committing.

- [ ] **Step 3: Type check**

```bash
cd /Users/mario/Personal/Projects/RallyCut/web && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/utils/extractPreviewFrames.ts
git commit -m "feat(web): client-side preview frame extractor (HTMLVideoElement + canvas)"
```

---

## Task 15: Frontend — wire preview gate into upload flow

**Files:**
- Modify: `web/src/services/api.ts` (add `preflightPreview()`)
- Modify: Upload component (identify via Step 1)
- Modify: `web/src/types/rally.ts` (add `QualityReport`, `Issue`, `Tier` types; drop `VideoCharacteristics`)

- [ ] **Step 1: Locate the upload trigger**

```bash
grep -rn "multipart/init\|upload-url\|startUpload" web/src/ --include='*.ts' --include='*.tsx'
```
Identify the component/method that calls `POST /v1/videos/upload-url` or the multipart init. The preview gate runs **before** that call.

- [ ] **Step 2: Add types in `web/src/types/rally.ts`**

Remove the `VideoCharacteristics` interface entirely. Add:

```typescript
export type Tier = 'block' | 'gate' | 'advisory';

export interface QualityIssue {
  id: string;
  tier: Tier;
  severity: number;
  message: string;
  source: 'preview' | 'upload' | 'preflight' | 'tracking';
  detectedAt: string;
  data?: Record<string, number>;
}

export interface QualityReport {
  version: 2;
  issues: QualityIssue[];
  preflight?: { ranAt: string; sampleSeconds: number; durationMs: number } | null;
  brightness?: number | null;
  resolution?: { width: number; height: number } | null;
}
```

Update the `Match` type: replace `characteristicsJson?: VideoCharacteristics` with `qualityReportJson?: QualityReport | null`. Touch every consumer that reads `characteristicsJson` (expect type errors and fix them inline — most should resolve once `VideoInsightsBanner` is deleted in Task 16).

- [ ] **Step 3: Add `preflightPreview` API client**

In `web/src/services/api.ts`:

```typescript
export interface PreflightPreviewResponse {
  pass: boolean;
  issues: QualityIssue[];
}

export async function preflightPreview(
  frames: Blob[],
  metadata: { width: number; height: number; durationS: number },
): Promise<PreflightPreviewResponse> {
  const form = new FormData();
  for (const b of frames) form.append('frames', b, 'frame.jpg');
  form.append('width', String(metadata.width));
  form.append('height', String(metadata.height));
  form.append('durationS', String(metadata.durationS));

  const res = await fetch(`${API_URL}/v1/videos/preflight-preview`, {
    method: 'POST',
    headers: { 'X-Visitor-Id': getVisitorId() },
    body: form,
  });
  if (!res.ok) throw new Error(`preflight-preview failed: ${res.status}`);
  return res.json();
}
```

- [ ] **Step 4: Integrate the gate in the upload flow**

In the upload component located in Step 1, wrap the existing upload trigger:

```typescript
import { extractPreviewFrames, readVideoMetadata } from '@/utils/extractPreviewFrames';
import { preflightPreview } from '@/services/api';

async function handleFileSelected(file: File) {
  // Pre-upload preview gate — runs before any bytes go to S3
  try {
    const meta = await readVideoMetadata(file);
    const frames = await extractPreviewFrames(file, 5);
    const preview = await preflightPreview(frames, meta);
    if (!preview.pass) {
      setUploadError({
        title: 'Upload cancelled',
        issues: preview.issues, // renderable with <QualityReportBanner />
      });
      return;  // do NOT proceed to upload-url
    }
  } catch (err) {
    // Preview is advisory-only on failure — if the preview pipeline crashes,
    // we don't want to block legitimate uploads. Log and continue.
    console.warn('[upload] preview gate failed, continuing:', err);
  }

  // Existing upload flow continues below
  await startUpload(file);
}
```

- [ ] **Step 5: Manual test with a non-VB clip**

Start dev server (`make dev`) and attempt to upload a random non-volleyball video. Expect: upload aborts with a clear block message referencing "beach volleyball" / "camera angle". No `upload-url` request hits the API (verify in Network panel).

Attempt to upload a legitimate match video. Expect: preview passes (network shows a 200 from `/v1/videos/preflight-preview` with `pass: true`), then the normal upload begins.

- [ ] **Step 6: Type check + lint**

```bash
cd /Users/mario/Personal/Projects/RallyCut/web && npx tsc --noEmit && npm run lint
```
Expected: no new errors.

- [ ] **Step 7: Commit**

```bash
git add web/src/services/api.ts web/src/types/rally.ts web/src/components/<upload-component>
git commit -m "feat(web): pre-upload preview gate — hard-fail wrong sport / wrong angle before upload"
```

---

## Task 16: New `QualityReportBanner` + delete `VideoInsightsBanner`

**Files:**
- Create: `web/src/components/QualityReportBanner.tsx`
- Delete: `web/src/components/VideoInsightsBanner.tsx`
- Modify: `web/src/components/EditorLayout.tsx`

- [ ] **Step 1: Implement new banner**

```tsx
// web/src/components/QualityReportBanner.tsx
'use client';

import { Alert, AlertTitle, Box, Chip, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useState, useEffect, useMemo } from 'react';
import type { QualityIssue, QualityReport } from '@/types/rally';

const TIER_COLOR: Record<QualityIssue['tier'], string> = {
  block: '#f44336',
  gate: '#FF9800',
  advisory: '#2196F3',
};

const TIER_SEVERITY: Record<QualityIssue['tier'], 'error' | 'warning' | 'info'> = {
  block: 'error',
  gate: 'warning',
  advisory: 'info',
};

interface QualityReportBannerProps {
  report: QualityReport | null | undefined;
  videoId: string;
}

function isDismissed(videoId: string): boolean {
  try { return sessionStorage.getItem(`quality-banner-dismissed-${videoId}`) === '1'; }
  catch { return false; }
}

export function QualityReportBanner({ report, videoId }: QualityReportBannerProps) {
  const [dismissed, setDismissed] = useState(() => isDismissed(videoId));
  useEffect(() => { setDismissed(isDismissed(videoId)); }, [videoId]);

  const issues = report?.issues ?? [];
  const topTier = useMemo<QualityIssue['tier']>(() => {
    if (issues.some((i) => i.tier === 'block')) return 'block';
    if (issues.some((i) => i.tier === 'gate')) return 'gate';
    return 'advisory';
  }, [issues]);

  if (!issues.length || dismissed) return null;

  const handleDismiss = () => {
    setDismissed(true);
    try { sessionStorage.setItem(`quality-banner-dismissed-${videoId}`, '1'); } catch {}
  };

  return (
    <Alert
      severity={TIER_SEVERITY[topTier]}
      sx={{ mb: 1 }}
      action={topTier !== 'block' ? (
        <IconButton size="small" onClick={handleDismiss} aria-label="dismiss">
          <CloseIcon fontSize="small" />
        </IconButton>
      ) : undefined}
    >
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mb: 0.5 }}>
        <AlertTitle sx={{ mb: 0 }}>Video quality</AlertTitle>
        {issues.map((i) => (
          <Chip
            key={i.id}
            label={i.tier}
            size="small"
            sx={{
              bgcolor: TIER_COLOR[i.tier],
              color: 'white',
              fontSize: '0.7rem',
              height: 18,
              textTransform: 'capitalize',
              '& .MuiChip-label': { px: 0.75 },
            }}
          />
        ))}
      </Box>
      {issues.map((i) => (
        <div key={i.id}>{i.message}</div>
      ))}
    </Alert>
  );
}
```

Key differences from the old `VideoInsightsBanner`:
- Dismiss uses **session**Storage, not localStorage — reappears on each visit.
- Block-tier issues are not dismissable.
- Displays only the `QualityReport.issues[]` array directly, with no merging logic (the API side handles all merging).

- [ ] **Step 2: Update `EditorLayout.tsx`**

```diff
-import { VideoInsightsBanner } from './VideoInsightsBanner';
+import { QualityReportBanner } from './QualityReportBanner';
```

```diff
-<VideoInsightsBanner currentMatch={currentMatch} />
+{currentMatch && (
+  <QualityReportBanner report={currentMatch.qualityReportJson ?? null} videoId={currentMatch.id} />
+)}
```

- [ ] **Step 3: Delete the old banner**

```bash
rm web/src/components/VideoInsightsBanner.tsx
```

- [ ] **Step 4: Type check**

```bash
cd /Users/mario/Personal/Projects/RallyCut/web && npx tsc --noEmit
```
Expected: no errors referencing `VideoInsightsBanner` or `characteristicsJson`.

- [ ] **Step 5: Manual test**

Start dev server, load a video that has preflight issues. Expect: new banner appears with at most 3 colored chips matching the tier palette; close button works for gate/advisory issues; block issues show with no close button.

- [ ] **Step 6: Commit**

```bash
git add web/src/components/QualityReportBanner.tsx web/src/components/EditorLayout.tsx
git rm web/src/components/VideoInsightsBanner.tsx
git commit -m "feat(web): QualityReportBanner replaces VideoInsightsBanner"
```

---

## Task 17: Stop writing `characteristicsJson` from tracking + processing services

**Files:**
- Modify: `api/src/services/playerTrackingService.ts`
- Modify: `api/src/services/processingService.ts`

These two services used to write to `characteristicsJson`. Now that column is gone — they must write their outputs into `qualityReportJson` instead.

- [ ] **Step 1: Locate writes**

```bash
grep -rn "characteristicsJson" api/src/ --include='*.ts'
```
There should be two writers (processingService for brightness; playerTrackingService for cameraDistance/sceneComplexity). Any remaining reads live in `qualityService-legacy-helpers.ts` (updated in Task 12).

- [ ] **Step 2: Update `processingService.ts` brightness writer**

Replace any `characteristicsJson: { brightness, ... }` update with:

```typescript
import { mergeQualityReports } from './qualityService.js';

const existing = (video.qualityReportJson as QualityReport | null) ?? null;
const updated = mergeQualityReports([existing ?? {}, { version: 2, issues: [], brightness }]);
await prisma.video.update({
  where: { id: videoId },
  data: { qualityReportJson: updated as unknown as Prisma.InputJsonValue },
});
```

- [ ] **Step 3: Update `playerTrackingService.ts` post-tracking writer**

Replace the characteristics update with an issue emitter:

```typescript
// After batch tracking finishes, convert per-rally qualityReport signals
// into video-level issues (source='tracking'). We DON'T re-surface the
// internal suggestions; we only emit user-facing issues.
const trackingIssues: Issue[] = [];
const lowBallDetectionRate = medianBallDetectionRate < 0.30;
if (lowBallDetectionRate) {
  trackingIssues.push({
    id: 'poor_ball_detection',
    tier: 'advisory',
    severity: 1 - medianBallDetectionRate / 0.30,
    message: 'Ball was hard to see in many rallies — better lighting or a higher-contrast background helps.',
    source: 'tracking',
    detectedAt: new Date().toISOString(),
    data: { ballDetectionRate: medianBallDetectionRate },
  });
}
// ... (repeat for camera-distance signal, stability signal, etc., using post-tracking data)

const existing = (video.qualityReportJson as QualityReport | null) ?? null;
const updated = mergeQualityReports([existing ?? {}, { version: 2, issues: trackingIssues }]);
await prisma.video.update({
  where: { id: videoId },
  data: { qualityReportJson: updated as unknown as Prisma.InputJsonValue },
});
```

- [ ] **Step 4: Type check**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api && npx tsc --noEmit
```
Expected: no more `characteristicsJson` references anywhere in `api/src`.

- [ ] **Step 5: Verify with grep**

```bash
grep -rn "characteristicsJson" api/src/ web/src/ --include='*.ts' --include='*.tsx'
```
Expected: **zero** results.

- [ ] **Step 6: Commit**

```bash
git add api/src/services/playerTrackingService.ts api/src/services/processingService.ts
git commit -m "refactor(api): route brightness + tracking signals through qualityReportJson"
```

---

## Task 18: End-to-end verification

**Files:**
- No code changes unless issues surface.

This task exists to prove the system behaves correctly end-to-end.

- [ ] **Step 1: Start dev stack**

```bash
cd /Users/mario/Personal/Projects/RallyCut && make dev
```
Verify: API on :3001, web on :3000, MinIO on :9000 all healthy.

- [ ] **Step 2: Scenario A — non-VB upload is rejected pre-upload**

Open web UI, choose a random non-volleyball video. Expect:
- Within ~3 seconds, an error banner appears stating this isn't a beach volleyball match.
- Network panel shows `POST /v1/videos/preflight-preview` → 200 with `pass: false`.
- **No** `POST /v1/videos/upload-url` request was made.
- Zero bytes uploaded to MinIO.

- [ ] **Step 3: Scenario B — legitimate upload + clean quality**

Upload a known-good GT video. Expect:
- `preflight-preview` returns `pass: true`.
- Upload completes normally.
- `qualityReportJson.issues` array is empty or contains only advisory items.
- `QualityReportBanner` either hides or shows only advisory-tier items.

- [ ] **Step 4: Scenario C — hard-block post-upload**

Upload a legitimate-but-borderline video where the pre-upload preview passes but the full preflight surfaces a block issue (e.g., court only visible in some sections). Expect:
- Upload completes.
- Clicking "Analyze Match" runs preflight; surfaces block issue; `Video.status` = REJECTED; banner shows block without dismiss button.
- Detection does NOT start.

- [ ] **Step 5: Scenario D — soft-gate flow**

Upload a dim clip. Expect: upload succeeds, preflight returns a `gate` issue; analysis proceeds (behaviour unchanged in A1 — the modal gating is an A2 responsibility). The banner shows the gate.

- [ ] **Step 6: Regression checks**

```bash
grep -rn "VideoInsightsBanner\|characteristicsJson\|VideoCharacteristics" web/src/ api/src/ analysis/ --include='*.ts' --include='*.tsx' --include='*.py'
```
Expected: zero results.

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/ -k quality -v
cd /Users/mario/Personal/Projects/RallyCut/api && npm test && npx tsc --noEmit
cd /Users/mario/Personal/Projects/RallyCut/web && npm run lint && npx tsc --noEmit
```
All must pass.

- [ ] **Step 7: Re-run calibration regression**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python scripts/calibrate_quality_checks.py
```
Confirm every shipped check still clears the 3× lift bar at its configured threshold. If not, investigate (likely a threshold change during implementation drifted away from the calibrated value).

- [ ] **Step 8: Final commit if any fix-ups**

```bash
git commit -am "chore: e2e verification — all scenarios pass"
```

---

## Self-review

**Spec coverage:**
- Check set (1–10 from spec): covered in Tasks 2–8. Note: check #5 (Not beach volleyball) and #6 (Wrong camera angle) share issue ID `wrong_angle_or_not_volleyball` intentionally (spec says issues can share a root cause).
- Tiered gating: Task 12 + Task 16 (tier-colored UI, block not dismissable).
- Data-driven calibration (≥3× lift bar): Task 10. Mandatory before shipping any preflight check.
- Remove VideoInsightsBanner + characteristicsJson: Tasks 11, 16, 17.
- Upload-time hard-fail for wrong sport / wrong angle: Tasks 13 (backend route) + 14 (client extractor) + 15 (gate integration).
- Progressive UX, edit-during-tracking, resilience: **OUT OF SCOPE for A1** — see spec and A2 follow-up plan.

**Placeholder scan:** every step has concrete code or commands. Calibration thresholds are placeholders-by-design until Task 10 runs; that's called out explicitly in each check module's docstring. Task 9 Step 1 requires locating the CLI app file (grep before editing), not a blind edit.

**Type consistency:**
- `QualityReport` shape matches across Python (`types.py::QualityReport.to_dict`), API (`qualityService.ts::QualityReport`), and web (`types/rally.ts::QualityReport`). All three use `version: 2`, `issues[]`, `preflight`.
- `Issue` fields match across the three layers: `id`, `tier`, `severity`, `message`, `source`, `detectedAt`, `data`.
- `Tier` values identical: `'block' | 'gate' | 'advisory'`.
- `source` values identical: `'preview' | 'upload' | 'preflight' | 'tracking'`.

**Follow-ups created:**
- Analyze-anyway modal for soft-gate (currently passes through — spec allows this, A2 adds the modal).
- `BatchTrackingJob.appendRally`, webhook idempotency, debounced match-analysis, `AnalysisPhase` refactor — all A2.
