# Project C — Sport-Sanity + Auto-Rotate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a beach-VB CLIP classifier to the client-side pre-upload gate (Component A) and a silent server-side auto-rotate during optimize (Component B), closing two gaps A1's post-ship validation surfaced.

**Architecture:** Two independent features that each slot into an existing pipeline stage. (A) extends the existing `rallycut preview-check` CLI with open-clip ViT-B/32 zero-shot scoring on the same 5 frames it already runs court-detection on. (B) adds a lightweight `rallycut tilt-detect` CLI that runs at `POST /v1/videos/:id/confirm`, and an FFmpeg `rotate` filter branch inside the existing optimize pass. New `qualityReportJson.autoFixes` field keeps "what we did" separate from `issues` (what the user should do).

**Tech Stack:** Python 3.12 (analysis/), open-clip-torch, PyTorch, FFmpeg, TypeScript/Express (api/), Prisma, React/Next.js (web/), `uv` package manager.

**Spec:** `docs/superpowers/specs/2026-04-15-project-c-sport-sanity-design.md`

**Prerequisite:** Create an isolated worktree before starting (project convention per A2a/A2b):
```bash
git -C /Users/mario/Personal/Projects/RallyCut worktree add .worktrees/c-sport-sanity -b feat/c-sport-sanity
cd .worktrees/c-sport-sanity
```

---

## Task 1: Generate manufactured-tilt fixtures

Fixtures are a hard prerequisite for Component B's tests (tilt-detect CLI, idempotency, regression). Generate them first so every downstream test has something to run against. They live locally (not checked in); we deferred S3 backup in the A1 session.

**Files:**
- Create: `analysis/scripts/generate_tilt_fixtures.sh`

- [ ] **Step 1: Write the shell script**

```bash
#!/usr/bin/env bash
# Generate manufactured-tilt fixtures for Project C Component B tests.
# Usage: bash analysis/scripts/generate_tilt_fixtures.sh
set -euo pipefail

SRC_DIR="$HOME/Desktop/rallies/Matches"
OUT_DIR="$HOME/Desktop/rallies/Negative"
SRC="$SRC_DIR/match.mp4"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: source fixture not found: $SRC"
  exit 1
fi
mkdir -p "$OUT_DIR"

for deg in 3 6 10 15; do
  out="$OUT_DIR/tilt_${deg}deg.mp4"
  echo "Generating ${out} (tilt ${deg}°)..."
  ffmpeg -y -i "$SRC" \
    -vf "rotate=${deg}*PI/180:ow=iw:oh=ih:c=black" \
    -c:a copy \
    -t 120 \
    "$out"
done
echo "Done. 4 fixtures at $OUT_DIR/tilt_*deg.mp4"
```

Note: `-t 120` limits each fixture to 2 minutes so generation + test runs stay fast.

- [ ] **Step 2: Run the script**

```bash
bash analysis/scripts/generate_tilt_fixtures.sh
```

Expected: four files at `~/Desktop/rallies/Negative/tilt_{3,6,10,15}deg.mp4`, each ~120s.

- [ ] **Step 3: Commit**

```bash
git add analysis/scripts/generate_tilt_fixtures.sh
git commit -m "feat(c): add manufactured-tilt fixture generator script"
```

---

## Task 2: Add open-clip-torch as an optional dependency

Keep CLIP out of the base install so CI / fresh checkouts that don't run preflight don't have to download ~150MB of weights.

**Files:**
- Modify: `analysis/pyproject.toml`

- [ ] **Step 1: Add the optional extra**

In `analysis/pyproject.toml`, under `[project.optional-dependencies]`, alongside the existing `dev` block, add:

```toml
preflight = [
    "open-clip-torch>=2.24.0",
]
```

- [ ] **Step 2: Install the extra**

```bash
cd analysis
uv sync --extra preflight
```

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "import open_clip; print(open_clip.__version__)"
```

Expected: version prints cleanly, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add analysis/pyproject.toml analysis/uv.lock
git commit -m "feat(c): add open-clip-torch as optional preflight extra"
```

---

## Task 3: Rewrite beach_vb_classifier to binary prompts + failing tests

The existing stub uses 3-way prompts (`beach_vb`, `indoor_vb`, `other`). Binary is cleaner — the decision is beach-VB or not. Following TDD: update tests first, watch them fail, then rewrite the classifier.

**Files:**
- Modify: `analysis/tests/unit/test_quality_beach_vb.py`
- Modify: `analysis/rallycut/quality/beach_vb_classifier.py`

- [ ] **Step 1: Rewrite the tests for binary shape**

Replace the contents of `analysis/tests/unit/test_quality_beach_vb.py` with:

```python
"""Tests for the binary beach-VB classifier.

The classifier returns a single per-frame `beach_vb_prob` (softmax prob of the
beach-volleyball prompt vs. the not-beach-volleyball prompt). A video blocks
when the average across sampled frames falls below BEACH_VB_BLOCK_THRESHOLD.
"""
from rallycut.quality.beach_vb_classifier import (
    BEACH_VB_BLOCK_THRESHOLD,
    classify_is_beach_vb,
)
from rallycut.quality.types import Tier


def test_high_beach_vb_probability_passes():
    # All frames clearly look like beach VB
    probs = [0.95, 0.92, 0.90, 0.88, 0.94]
    result = classify_is_beach_vb(probs)
    assert result.issues == []
    assert result.metrics["avgBeachVbProb"] > 0.85


def test_very_low_beach_vb_probability_blocks():
    # All frames look like non-beach content (indoor, or not volleyball)
    probs = [0.12, 0.18, 0.15, 0.21, 0.19]
    result = classify_is_beach_vb(probs)
    assert any(i.id == "not_beach_volleyball" for i in result.issues)
    block = next(i for i in result.issues if i.id == "not_beach_volleyball")
    assert block.tier == Tier.BLOCK
    assert block.data["avgBeachVbProb"] < BEACH_VB_BLOCK_THRESHOLD


def test_ambiguous_does_not_block():
    # Right at the threshold — must NOT fire (favor false-accept)
    probs = [BEACH_VB_BLOCK_THRESHOLD + 0.01] * 5
    result = classify_is_beach_vb(probs)
    assert result.issues == []


def test_empty_input_is_noop():
    result = classify_is_beach_vb([])
    assert result.issues == []
    assert result.metrics == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd analysis
uv run pytest tests/unit/test_quality_beach_vb.py -v
```

Expected: FAIL — `classify_is_beach_vb` currently takes `list[BeachVBProbabilities]`, not `list[float]`.

- [ ] **Step 3: Rewrite `beach_vb_classifier.py` to binary prompts**

Replace `analysis/rallycut/quality/beach_vb_classifier.py` with:

```python
"""Binary zero-shot "is this beach volleyball" classifier using open-clip.

Scores each frame against two prompts and returns the softmax prob of the
beach-VB prompt. The runtime path (`embed_and_score_frames`) imports open-clip
lazily so unit tests can pass precomputed probabilities without loading the
model.

Calibration principle (see spec §Guiding Principle): threshold is set below
the lowest-scoring positive, not at the midpoint between positives and
negatives. Favor false-accept over false-reject.
"""
from __future__ import annotations

import statistics
from typing import Any

from rallycut.quality.types import CheckResult, Issue, Tier

# Calibrated against 5 negatives + positives (see
# analysis/reports/beach_vb_calibration_<date>.json). Post-Task 10 this value
# may be refined; keep the constant as the single source of truth.
BEACH_VB_BLOCK_THRESHOLD = 0.50

PROMPTS = (
    "a beach volleyball match played on sand",
    "a video that is not beach volleyball",
)


def classify_is_beach_vb(per_frame_beach_vb_probs: list[float]) -> CheckResult:
    """Classify a video from its per-frame beach-VB probabilities.

    Args:
        per_frame_beach_vb_probs: softmax prob of PROMPTS[0] per frame, in [0,1].

    Returns a CheckResult. Empty input is a no-op (no issues, no metrics).
    """
    if not per_frame_beach_vb_probs:
        return CheckResult(issues=[], metrics={})

    avg = statistics.mean(per_frame_beach_vb_probs)
    metrics = {"avgBeachVbProb": avg}
    issues: list[Issue] = []

    if avg < BEACH_VB_BLOCK_THRESHOLD:
        issues.append(Issue(
            id="not_beach_volleyball",
            tier=Tier.BLOCK,
            severity=1.0 - avg,
            message="This doesn't look like a beach volleyball match. RallyCut is tuned for beach volleyball only.",
            source="preview",
            data={"avgBeachVbProb": avg},
        ))
    return CheckResult(issues=issues, metrics=metrics)


def embed_and_score_frames(frames: list[Any]) -> list[float]:
    """Run open-clip ViT-B/32 on each frame and return PROMPTS[0] softmax probs.

    `frames` is a list of PIL.Image objects. Integration-tested via
    `rallycut preview-check`, not unit-tested.
    """
    import open_clip  # type: ignore[import-not-found]
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    text_tokens = tokenizer(list(PROMPTS))
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_tensors = torch.stack([preprocess(f) for f in frames])
        image_features = model.encode_image(image_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ text_features.T) * 100.0
        probs = logits.softmax(dim=-1).cpu().numpy()

    # Column 0 is PROMPTS[0] (beach VB). Return per-frame beach-VB prob.
    return [float(row[0]) for row in probs]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_quality_beach_vb.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Type-check**

```bash
uv run mypy rallycut/quality/beach_vb_classifier.py
```

Expected: `Success: no issues found in 1 source file`.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/quality/beach_vb_classifier.py analysis/tests/unit/test_quality_beach_vb.py
git commit -m "feat(c): binary beach-VB classifier (open-clip zero-shot)"
```

---

## Task 4: Wire CLIP check into preview-check CLI

Extend the existing CLI that already runs court-keypoint detection on the 5 client-extracted frames. Same frames, second check.

**Files:**
- Modify: `analysis/rallycut/cli/commands/preview_check.py`
- Create: `analysis/tests/unit/test_preview_check_wiring.py`

- [ ] **Step 1: Write a failing integration test**

Create `analysis/tests/unit/test_preview_check_wiring.py`:

```python
"""Confirm preview-check merges court-geometry + beach-VB issues."""
from unittest.mock import patch

from rallycut.quality.camera_geometry import CourtCorners
from rallycut.quality.types import Tier


def _good_corners() -> CourtCorners:
    return CourtCorners(
        tl=(0.3, 0.4), tr=(0.7, 0.4),
        br=(0.8, 0.8), bl=(0.2, 0.8),
        confidence=0.9,
    )


def test_preview_check_fires_beach_vb_block_when_scores_low(tmp_path, monkeypatch):
    from rallycut.cli.commands import preview_check as pc

    # Fake frame files — preview-check only needs the first to exist
    (tmp_path / "frame_0.jpg").write_bytes(b"\x00" * 16)

    with patch.object(pc, "_detect_corners_from_frame", return_value=_good_corners()), \
         patch.object(pc, "_score_beach_vb_for_frames", return_value=[0.1, 0.15, 0.12, 0.2, 0.18]):
        report = pc._run(tmp_path, width=640, height=360, duration_s=60.0)

    ids = {i.id for i in report.issues}
    assert "not_beach_volleyball" in ids
    block = next(i for i in report.issues if i.id == "not_beach_volleyball")
    assert block.tier == Tier.BLOCK


def test_preview_check_passes_when_scores_high(tmp_path):
    from rallycut.cli.commands import preview_check as pc

    (tmp_path / "frame_0.jpg").write_bytes(b"\x00" * 16)

    with patch.object(pc, "_detect_corners_from_frame", return_value=_good_corners()), \
         patch.object(pc, "_score_beach_vb_for_frames", return_value=[0.85, 0.9, 0.88, 0.92, 0.87]):
        report = pc._run(tmp_path, width=640, height=360, duration_s=60.0)

    assert report.issues == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_preview_check_wiring.py -v
```

Expected: FAIL — `pc._run` and `pc._score_beach_vb_for_frames` don't exist yet.

- [ ] **Step 3: Refactor preview-check to expose an internal `_run` + new `_score_beach_vb_for_frames`**

Replace `analysis/rallycut/cli/commands/preview_check.py` with:

```python
"""`rallycut preview-check <dir>` — run court-geometry + beach-VB classifier
against a directory of JPEG frames (no video decoding required). Used by the
web pre-upload gate before the upload commits.

Both checks run on the same 5 client-extracted frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import typer

from rallycut.quality.beach_vb_classifier import classify_is_beach_vb
from rallycut.quality.camera_geometry import CourtCorners, check_camera_geometry
from rallycut.quality.types import QualityReport


def preview_check(
    frames_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    width: int = typer.Option(..., "--width"),
    height: int = typer.Option(..., "--height"),
    duration_s: float = typer.Option(..., "--duration-s"),
    as_json: bool = typer.Option(True, "--json/--no-json"),
    quiet: bool = typer.Option(False, "--quiet"),  # noqa: ARG001 (reserved for future UX)
) -> None:
    """Run preview-time checks against a directory of JPEG frames."""
    report = _run(frames_dir, width=width, height=height, duration_s=duration_s)
    if as_json:
        typer.echo(json.dumps(report.to_dict()))


def _run(frames_dir: Path, *, width: int, height: int, duration_s: float) -> QualityReport:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        return QualityReport.from_checks(
            [], source="preview", duration_ms=int(duration_s * 1000)
        )

    # Load first frame for court detection (BGR uint8)
    bgr = cv2.imread(str(frame_paths[0]))
    if bgr is None:
        return QualityReport.from_checks(
            [], source="preview", duration_ms=int(duration_s * 1000)
        )

    try:
        corners = _detect_corners_from_frame(bgr, width=width, height=height)
    except Exception:  # noqa: BLE001
        corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.0)
    geom_result = check_camera_geometry(corners)

    # Score all available frames with CLIP (failure-tolerant: on any error,
    # skip the check entirely, preserving A1 behavior)
    try:
        probs = _score_beach_vb_for_frames(frame_paths)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[preview-check] beach-VB scoring failed: {exc}", err=True)
        probs = []
    vb_result = classify_is_beach_vb(probs)

    return QualityReport.from_checks(
        [geom_result, vb_result], source="preview", duration_ms=int(duration_s * 1000)
    )


def _detect_corners_from_frame(bgr: np.ndarray, width: int, height: int) -> CourtCorners:
    """Return CourtCorners from a single BGR frame using the keypoint model."""
    from rallycut.court.keypoint_detector import CourtKeypointDetector

    detector = CourtKeypointDetector()
    result = detector.detect_from_frame(bgr)

    if not result.corners or len(result.corners) < 4:
        return CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.0)

    nl = result.corners[0]
    nr = result.corners[1]
    fr = result.corners[2]
    fl = result.corners[3]
    return CourtCorners(
        tl=(fl["x"], fl["y"]),
        tr=(fr["x"], fr["y"]),
        br=(nr["x"], nr["y"]),
        bl=(nl["x"], nl["y"]),
        confidence=result.confidence,
    )


def _score_beach_vb_for_frames(frame_paths: list[Path]) -> list[float]:
    """Load JPEGs as PIL.Image and run open-clip. Returns beach-VB probs."""
    from PIL import Image

    from rallycut.quality.beach_vb_classifier import embed_and_score_frames

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    return embed_and_score_frames(images)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_preview_check_wiring.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Ensure existing preview-check behavior is preserved**

```bash
uv run pytest tests/unit/test_quality_camera_geometry.py tests/unit/test_quality_beach_vb.py -v
```

Expected: all pass.

- [ ] **Step 6: Type-check**

```bash
uv run mypy rallycut/
```

Expected: success.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/cli/commands/preview_check.py analysis/tests/unit/test_preview_check_wiring.py
git commit -m "feat(c): wire CLIP beach-VB classifier into preview-check CLI"
```

---

## Task 5: Calibration script for beach-VB threshold

Before the classifier can ship, the threshold has to clear the ≥ 0.15 gap bar (spec §Guiding Principle + §Calibration). Build the sweep, run it, write the report, and lock the final threshold.

**Files:**
- Create: `analysis/scripts/calibrate_beach_vb.py`

- [ ] **Step 1: Write the calibration script**

Create `analysis/scripts/calibrate_beach_vb.py`:

```python
r"""Calibrate the beach-VB classifier threshold.

Runs open-clip on the 5 negative fixtures + positive examples, reports
per-video beach-VB probability, and selects a threshold where:

    threshold = max(lowest_positive_score - 0.10, 0.30)

Ships only if `lowest_positive - highest_negative >= 0.15`.

Usage:
    cd analysis
    uv run python scripts/calibrate_beach_vb.py \
        --negatives ~/Desktop/rallies/Negative/bad\ angle.mp4 \
        --negatives ~/Desktop/rallies/Negative/very\ bad\ angle.mp4 \
        --negatives ~/Desktop/rallies/Negative/indoor\ 1.mp4 \
        --negatives ~/Desktop/rallies/Negative/indoor\ 2.mp4 \
        --negatives ~/Desktop/rallies/Negative/not\ related.mp4 \
        --positives ~/Desktop/rallies/Matches/match.mp4 \
        --positives "~/Desktop/rallies/Matches/Newport Beach Games 10_12_25 Paul_Nate vs Colin_Tim - Nathan Gali (1080p, h264).mp4"

Writes analysis/reports/beach_vb_calibration_<date>.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path


def _extract_frames(video_path: Path, n_frames: int = 5) -> list[Path]:
    """Sample n_frames evenly across the video via ffmpeg."""
    out_dir = Path(tempfile.mkdtemp(prefix="beach_vb_calib_"))
    for i in range(n_frames):
        frac = (i + 0.5) / n_frames
        out = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{frac * 100:.2f}%",
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out),
            ],
            check=True,
        )
    # Some ffmpeg builds don't support percentage seeks; fall back to duration-based
    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
    return frames


def _score_video(video_path: Path) -> float:
    """Return the average beach-VB prob for video_path."""
    from PIL import Image

    from rallycut.quality.beach_vb_classifier import embed_and_score_frames

    frames = _extract_frames(video_path, n_frames=5)
    images = [Image.open(p).convert("RGB") for p in frames]
    probs = embed_and_score_frames(images)
    return sum(probs) / len(probs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", action="append", required=True, type=Path)
    parser.add_argument("--positives", action="append", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / f"beach_vb_calibration_{date.today().isoformat()}.json",
    )
    args = parser.parse_args()

    print(f"[calibrate] scoring {len(args.negatives)} negatives, {len(args.positives)} positives")
    scores: dict[str, float] = {}
    for v in args.negatives + args.positives:
        print(f"  scoring {v.name} ...", flush=True)
        scores[str(v)] = _score_video(v)
        print(f"    = {scores[str(v)]:.3f}")

    neg_scores = [scores[str(v)] for v in args.negatives]
    pos_scores = [scores[str(v)] for v in args.positives]
    highest_neg = max(neg_scores)
    lowest_pos = min(pos_scores)
    gap = lowest_pos - highest_neg
    threshold = max(lowest_pos - 0.10, 0.30)

    ships = gap >= 0.15
    report = {
        "date": date.today().isoformat(),
        "per_video": scores,
        "highest_negative": highest_neg,
        "lowest_positive": lowest_pos,
        "gap": gap,
        "chosen_threshold": threshold,
        "ship_gate_passed": ships,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[calibrate] wrote {args.output}")
    print(f"  highest_neg={highest_neg:.3f}  lowest_pos={lowest_pos:.3f}  gap={gap:.3f}")
    print(f"  chosen_threshold={threshold:.3f}  SHIPS={ships}")

    return 0 if ships else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the calibration sweep**

```bash
cd analysis
uv run python scripts/calibrate_beach_vb.py \
  --negatives "$HOME/Desktop/rallies/Negative/bad angle.mp4" \
  --negatives "$HOME/Desktop/rallies/Negative/very bad angle.mp4" \
  --negatives "$HOME/Desktop/rallies/Negative/indoor 1.mp4" \
  --negatives "$HOME/Desktop/rallies/Negative/indoor 2.mp4" \
  --negatives "$HOME/Desktop/rallies/Negative/not related.mp4" \
  --positives "$HOME/Desktop/rallies/Matches/match.mp4" \
  --positives "$HOME/Desktop/rallies/Matches/Newport Beach Games 10_12_25 Paul_Nate vs Colin_Tim - Nathan Gali (1080p, h264).mp4"
```

Expected:
- Report written to `analysis/reports/beach_vb_calibration_<date>.json`.
- Script prints `SHIPS=True` AND `gap >= 0.15`.
- If `SHIPS=False`: **STOP** and treat as a go/no-go decision point. Options: retune prompts, sample more positives/negatives, or drop Component A and ship only B. Document the decision in the A1-style memory update (Task 12).

- [ ] **Step 3: Update BEACH_VB_BLOCK_THRESHOLD to the chosen value**

Read the `chosen_threshold` field from the report JSON and update the constant in `analysis/rallycut/quality/beach_vb_classifier.py`:

```python
BEACH_VB_BLOCK_THRESHOLD = <value from report, e.g. 0.55>
```

- [ ] **Step 4: Re-run unit tests to confirm they pass at the calibrated threshold**

```bash
uv run pytest tests/unit/test_quality_beach_vb.py -v
```

The `test_ambiguous_does_not_block` test uses `BEACH_VB_BLOCK_THRESHOLD + 0.01`, so it stays green regardless of the chosen value.

- [ ] **Step 5: Commit**

```bash
git add analysis/scripts/calibrate_beach_vb.py analysis/reports/beach_vb_calibration_*.json analysis/rallycut/quality/beach_vb_classifier.py
git commit -m "feat(c): calibrate beach-VB threshold against 5 negatives + 2 positives"
```

---

## Task 6: Re-add `_baseline_tilt_deg` helper + test

This helper was deleted on 2026-04-15 when `video_rotated` was dropped. Re-add it as a pure helper (no issue emission) so Component B can import it without bringing back an advisory we don't want.

**Files:**
- Modify: `analysis/rallycut/quality/camera_geometry.py`
- Create: `analysis/tests/unit/test_camera_geometry_tilt.py`

- [ ] **Step 1: Write failing tests**

Create `analysis/tests/unit/test_camera_geometry_tilt.py`:

```python
import math

from rallycut.quality.camera_geometry import CourtCorners, baseline_tilt_deg


def test_straight_baseline_is_zero_tilt():
    corners = CourtCorners(
        tl=(0.3, 0.4), tr=(0.7, 0.4),
        br=(0.8, 0.8), bl=(0.2, 0.8),
        confidence=0.9,
    )
    assert baseline_tilt_deg(corners) == 0.0


def test_10deg_baseline_returns_10deg():
    angle = math.radians(10)
    c, s = math.cos(angle), math.sin(angle)
    def rot(p):
        return (0.5 + (p[0] - 0.5) * c - (p[1] - 0.5) * s,
                0.5 + (p[0] - 0.5) * s + (p[1] - 0.5) * c)
    base = CourtCorners(
        tl=(0.3, 0.4), tr=(0.7, 0.4),
        br=(0.8, 0.8), bl=(0.2, 0.8),
        confidence=0.9,
    )
    tilted = CourtCorners(
        tl=rot(base.tl), tr=rot(base.tr), br=rot(base.br), bl=rot(base.bl),
        confidence=0.9,
    )
    assert abs(baseline_tilt_deg(tilted) - 10.0) < 0.1


def test_degenerate_corners_return_zero():
    corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.0)
    assert baseline_tilt_deg(corners) == 0.0
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/unit/test_camera_geometry_tilt.py -v
```

Expected: FAIL — `baseline_tilt_deg` doesn't exist.

- [ ] **Step 3: Add the helper to `camera_geometry.py`**

Append to `analysis/rallycut/quality/camera_geometry.py` (do not re-introduce the `video_rotated` advisory):

```python
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
```

(If the existing file no longer has `import math` at the top after the A1 deletion, add it back.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_camera_geometry_tilt.py tests/unit/test_quality_camera_geometry.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/quality/camera_geometry.py analysis/tests/unit/test_camera_geometry_tilt.py
git commit -m "feat(c): re-add baseline_tilt_deg helper (pure, no advisory emission)"
```

---

## Task 7: Create `rallycut tilt-detect` CLI

One-shot CLI: take a directory of JPEG frames, return median tilt + median confidence from the subset with confidence ≥ 0.5. Matches the preview-check input shape so the API can reuse its frame-extraction helpers.

**Files:**
- Create: `analysis/rallycut/cli/commands/tilt_detect.py`
- Create: `analysis/tests/unit/test_tilt_detect.py`

- [ ] **Step 1: Write failing tests**

Create `analysis/tests/unit/test_tilt_detect.py`:

```python
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
    # fl = TL, fr = TR: make TR offset downward by tan(deg)*dx
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
```

- [ ] **Step 2: Run tests to verify failure**

```bash
uv run pytest tests/unit/test_tilt_detect.py -v
```

Expected: FAIL — the module doesn't exist.

- [ ] **Step 3: Implement the CLI + pure helper**

Create `analysis/rallycut/cli/commands/tilt_detect.py`:

```python
"""`rallycut tilt-detect <frames-dir>` — emit {tiltDeg, courtConfidence, framesScored}.

Used at POST /v1/videos/:id/confirm to decide whether the optimize pass should
append an FFmpeg rotate filter. Pure computation layer (`_compute_tilt_from_frames`)
is unit-tested separately from the frame-loading + detector instantiation.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import cv2
import typer

from rallycut.quality.camera_geometry import CourtCorners, baseline_tilt_deg

MIN_FRAME_CONF = 0.5  # frames below this are excluded from the median


def tilt_detect(
    frames_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    as_json: bool = typer.Option(True, "--json/--no-json"),
) -> None:
    """Detect tilt from a directory of JPEG frames. Emit JSON to stdout."""
    from rallycut.court.keypoint_detector import CourtKeypointDetector

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    frames = [cv2.imread(str(p)) for p in frame_paths]
    frames = [f for f in frames if f is not None]

    result = _compute_tilt_from_frames(CourtKeypointDetector(), frames=frames)
    if as_json:
        typer.echo(json.dumps(result))


def _compute_tilt_from_frames(detector: Any, frames: list) -> dict:
    """Pure compute layer. `detector` must have `detect_from_frame(frame) -> result`
    with `result.corners` (list of {x, y} dicts) and `result.confidence` (float).
    """
    tilt_degs: list[float] = []
    confs: list[float] = []
    for frame in frames:
        det = detector.detect_from_frame(frame)
        if not det.corners or len(det.corners) < 4 or det.confidence < MIN_FRAME_CONF:
            continue
        # keypoint order: [nl, nr, fr, fl] -> our CourtCorners: tl=fl, tr=fr, br=nr, bl=nl
        nl, nr, fr, fl = det.corners[0], det.corners[1], det.corners[2], det.corners[3]
        corners = CourtCorners(
            tl=(fl["x"], fl["y"]),
            tr=(fr["x"], fr["y"]),
            br=(nr["x"], nr["y"]),
            bl=(nl["x"], nl["y"]),
            confidence=det.confidence,
        )
        tilt_degs.append(baseline_tilt_deg(corners))
        confs.append(det.confidence)

    if not tilt_degs:
        return {"tiltDeg": 0.0, "courtConfidence": 0.0, "framesScored": 0}
    return {
        "tiltDeg": statistics.median(tilt_degs),
        "courtConfidence": statistics.median(confs),
        "framesScored": len(tilt_degs),
    }
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/test_tilt_detect.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/cli/commands/tilt_detect.py analysis/tests/unit/test_tilt_detect.py
git commit -m "feat(c): tilt-detect CLI (median of high-conf frame tilts)"
```

---

## Task 8: Register `tilt-detect` in CLI main

**Files:**
- Modify: `analysis/rallycut/cli/main.py`

- [ ] **Step 1: Add the import + registration**

In `analysis/rallycut/cli/main.py`:

1. Add this import alongside the other command imports, in alphabetical order:
```python
from rallycut.cli.commands.tilt_detect import tilt_detect as tilt_detect_command
```

2. Add this line alongside the other `app.command()` registrations (look for `preflight_command` registration and add adjacent):
```python
app.command(name="tilt-detect")(tilt_detect_command)
```

- [ ] **Step 2: Verify the command is registered**

```bash
uv run rallycut tilt-detect --help
```

Expected: typer help text renders, showing `--json/--no-json` options.

- [ ] **Step 3: End-to-end smoke test against a real frame directory**

Extract 3 frames from a tilt fixture and run:
```bash
mkdir -p /tmp/tilt_smoke && \
  ffmpeg -y -v error -i "$HOME/Desktop/rallies/Negative/tilt_10deg.mp4" \
    -vf "select='eq(n,30)+eq(n,300)+eq(n,900)'" -vsync vfr /tmp/tilt_smoke/frame_%02d.jpg && \
  uv run rallycut tilt-detect /tmp/tilt_smoke
```

Expected JSON with `tiltDeg ≈ 10 ± 2`, `courtConfidence > 0.6`, `framesScored >= 2`.
If the real court model reports lower confidence on the rotated fixture, `framesScored` may be 0 — in that case, also note the result, it's a signal worth saving for Task 13's regression.

- [ ] **Step 4: Commit**

```bash
git add analysis/rallycut/cli/main.py
git commit -m "feat(c): register tilt-detect command"
```

---

## Task 9: Extend QualityReport types + mergeQualityReports

Add `autoFixes`, `autoRotated`, `tiltDeg`, `courtConfidence` to the shared type. Update merge behavior: `autoFixes` arrays concat with dedupe-by-id (first wins).

**Files:**
- Modify: `api/src/services/qualityReport.ts`
- Modify: `api/tests/qualityReport.test.ts` (add if missing — check existing test file name)
- Modify: `web/src/types/rally.ts`

- [ ] **Step 1: Extend the API type**

In `api/src/services/qualityReport.ts`, add the new type and extend `QualityReport`:

```ts
// After the existing Issue type definition
export interface AutoFix {
  id: string;
  message: string;
  appliedAt: string; // ISO timestamp
  data?: Record<string, number>;
}

// Extend the existing QualityReport interface
export interface QualityReport {
  version: 2;
  issues: Issue[];
  autoFixes?: AutoFix[];
  preflight?: { ranAt: string; sampleSeconds: number; durationMs: number } | null;
  brightness?: number | null;
  resolution?: { width: number; height: number } | null;
  // NEW state fields used by Project C auto-rotate:
  autoRotated?: boolean;
  tiltDeg?: number | null;
  courtConfidence?: number | null;
}
```

- [ ] **Step 2: Extend `mergeQualityReports` for autoFixes concat + dedupe**

Still in `api/src/services/qualityReport.ts`, update the function so the returned object includes:

```ts
export function mergeQualityReports(reports: Array<Partial<QualityReport>>): QualityReport {
  const allIssues: Issue[] = reports.flatMap((r) => r.issues ?? []);
  const brightness = reports.map((r) => r.brightness).find((v) => v != null) ?? null;
  const resolution = reports.map((r) => r.resolution).find((v) => v != null) ?? null;
  const preflight = reports.map((r) => r.preflight).find((v) => v != null) ?? null;

  // autoFixes: concat all arrays, then dedupe by id keeping the FIRST occurrence.
  // Mirrors the "first non-null wins" semantics used for scalar fields — callers
  // pass the existing persisted report first, so prior auto-fixes survive.
  const seenFixIds = new Set<string>();
  const autoFixes: AutoFix[] = [];
  for (const r of reports) {
    for (const fx of r.autoFixes ?? []) {
      if (seenFixIds.has(fx.id)) continue;
      seenFixIds.add(fx.id);
      autoFixes.push(fx);
    }
  }

  const autoRotated =
    reports.map((r) => r.autoRotated).find((v) => v != null) ?? undefined;
  const tiltDeg =
    reports.map((r) => r.tiltDeg).find((v) => v != null) ?? null;
  const courtConfidence =
    reports.map((r) => r.courtConfidence).find((v) => v != null) ?? null;

  return {
    version: 2,
    issues: pickTopIssues(allIssues),
    preflight,
    brightness,
    resolution,
    autoFixes: autoFixes.length ? autoFixes : undefined,
    autoRotated,
    tiltDeg,
    courtConfidence,
  };
}
```

- [ ] **Step 3: Add a failing test for the new merge behavior**

Find the existing tests. If `api/tests/qualityService.test.ts` already exists, append these cases. Otherwise, create `api/tests/qualityReport.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { mergeQualityReports } from '../src/services/qualityReport.js';

describe('mergeQualityReports — autoFixes', () => {
  it('concatenates autoFixes arrays across reports', () => {
    const merged = mergeQualityReports([
      { version: 2, issues: [], autoFixes: [{ id: 'auto_straightened', message: 'A', appliedAt: '2026-04-15T00:00:00Z' }] },
      { version: 2, issues: [], autoFixes: [{ id: 'other_fix', message: 'B', appliedAt: '2026-04-15T00:00:01Z' }] },
    ]);
    expect(merged.autoFixes).toHaveLength(2);
    expect(merged.autoFixes?.map((f) => f.id).sort()).toEqual(['auto_straightened', 'other_fix']);
  });

  it('dedupes by id, keeping the first occurrence', () => {
    const merged = mergeQualityReports([
      { version: 2, issues: [], autoFixes: [{ id: 'auto_straightened', message: 'FIRST', appliedAt: '2026-04-15T00:00:00Z' }] },
      { version: 2, issues: [], autoFixes: [{ id: 'auto_straightened', message: 'SECOND', appliedAt: '2026-04-15T00:00:01Z' }] },
    ]);
    expect(merged.autoFixes).toHaveLength(1);
    expect(merged.autoFixes?.[0].message).toBe('FIRST');
  });

  it('returns undefined autoFixes when all inputs are empty', () => {
    const merged = mergeQualityReports([{ version: 2, issues: [] }]);
    expect(merged.autoFixes).toBeUndefined();
  });
});

describe('mergeQualityReports — new scalar fields', () => {
  it('keeps first non-null autoRotated', () => {
    const merged = mergeQualityReports([
      { version: 2, issues: [], autoRotated: true },
      { version: 2, issues: [], autoRotated: false },
    ]);
    expect(merged.autoRotated).toBe(true);
  });

  it('keeps first non-null tiltDeg', () => {
    const merged = mergeQualityReports([
      { version: 2, issues: [], tiltDeg: 7.3 },
      { version: 2, issues: [], tiltDeg: 0 },
    ]);
    expect(merged.tiltDeg).toBe(7.3);
  });
});
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd api
npx vitest run tests/qualityReport.test.ts  # or the file you appended to
```

Expected: all tests pass.

- [ ] **Step 5: tsc check**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Mirror types to web**

In `web/src/types/rally.ts`, locate the `QualityReport` type and apply the same addition:

```ts
export interface AutoFix {
  id: string;
  message: string;
  appliedAt: string;
  data?: Record<string, number>;
}

export interface QualityReport {
  version: 2;
  issues: Issue[];
  autoFixes?: AutoFix[];
  preflight?: { ranAt: string; sampleSeconds: number; durationMs: number } | null;
  brightness?: number | null;
  resolution?: { width: number; height: number } | null;
  autoRotated?: boolean;
  tiltDeg?: number | null;
  courtConfidence?: number | null;
}
```

- [ ] **Step 7: web tsc check**

```bash
cd ../web
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
cd ..
git add api/src/services/qualityReport.ts api/tests/qualityReport.test.ts web/src/types/rally.ts
git commit -m "feat(c): extend QualityReport with autoFixes + auto-rotate state fields"
```

---

## Task 10: Add tilt-detect frame extraction + CLI spawn helper

Server-side helper that extracts 5 evenly-sampled JPEG frames from S3, calls `rallycut tilt-detect`, returns `{tiltDeg, courtConfidence}`. Mirrors the existing `runPreflightCli` / `runPreviewCli` patterns in `qualityService.ts`.

**Files:**
- Modify: `api/src/services/qualityService.ts`

- [ ] **Step 1: Add the public function**

At the end of `api/src/services/qualityService.ts`, add:

```ts
export interface TiltDetectResult {
  tiltDeg: number;
  courtConfidence: number;
  framesScored: number;
}

/**
 * Run `rallycut tilt-detect` against a video at `localPath`. Extracts 5
 * evenly-sampled frames via ffmpeg, spawns the CLI against the frame dir,
 * parses the JSON response.
 *
 * Fails soft: any error returns `{tiltDeg: 0, courtConfidence: 0, framesScored: 0}`.
 * Callers treat that as "no rotation; no flag." Never throws.
 */
export async function runTiltDetect(localPath: string): Promise<TiltDetectResult> {
  const frameDir = path.join(
    TEMP_DIR,
    `tilt_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
  );
  try {
    await fs.mkdir(frameDir, { recursive: true });
    await extractFramesForTiltDetect(localPath, frameDir, 5);
    return await runTiltDetectCli(frameDir);
  } catch (err) {
    console.warn(`[tilt-detect] failed, returning zero result: ${err}`);
    return { tiltDeg: 0, courtConfidence: 0, framesScored: 0 };
  } finally {
    await fs.rm(frameDir, { recursive: true, force: true }).catch(() => {});
  }
}

async function extractFramesForTiltDetect(
  videoPath: string,
  outDir: string,
  count: number,
): Promise<void> {
  // Get duration via ffprobe
  const { spawn } = await import('child_process');
  const duration = await new Promise<number>((resolve, reject) => {
    let out = '';
    const p = spawn('ffprobe', [
      '-v', 'error',
      '-show_entries', 'format=duration',
      '-of', 'default=noprint_wrappers=1:nokey=1',
      videoPath,
    ]);
    p.stdout.on('data', (d) => { out += d.toString(); });
    p.on('exit', (code) => {
      if (code !== 0) return reject(new Error('ffprobe failed'));
      const secs = parseFloat(out.trim());
      if (!Number.isFinite(secs) || secs <= 0) return reject(new Error('bad duration'));
      resolve(secs);
    });
  });

  // Extract `count` frames evenly across the duration
  await Promise.all(
    Array.from({ length: count }, (_, i) => {
      const ts = ((i + 0.5) / count) * duration;
      const outPath = path.join(outDir, `frame_${String(i).padStart(2, '0')}.jpg`);
      return new Promise<void>((resolve, reject) => {
        const p = spawn('ffmpeg', [
          '-y', '-v', 'error',
          '-ss', ts.toFixed(2),
          '-i', videoPath,
          '-frames:v', '1',
          '-q:v', '2',
          outPath,
        ]);
        p.on('exit', (code) => code === 0 ? resolve() : reject(new Error(`ffmpeg exit ${code}`)));
      });
    }),
  );
}

function runTiltDetectCli(frameDir: string): Promise<TiltDetectResult> {
  return new Promise((resolve, reject) => {
    const args = ['run', 'rallycut', 'tilt-detect', frameDir, '--json'];
    const child = spawn('uv', args, {
      cwd: ANALYSIS_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env },
    });
    let stdout = '';
    let stderr = '';
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { child.kill(); } catch { /* ignore */ }
      reject(new Error('tilt-detect timed out'));
    }, 30_000);
    child.stdout?.on('data', (d: Buffer) => { stdout += d.toString(); });
    child.stderr?.on('data', (d: Buffer) => { stderr += d.toString(); });
    child.on('error', (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`tilt-detect failed to start: ${err.message}`));
    });
    child.on('exit', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code !== 0) {
        return reject(new Error(`tilt-detect exited ${code}: ${stderr.slice(-500)}`));
      }
      try {
        const m = stdout.match(/\{[\s\S]*\}/);
        if (!m) return reject(new Error('No JSON from tilt-detect'));
        const parsed = JSON.parse(m[0]);
        resolve({
          tiltDeg: Number(parsed.tiltDeg ?? 0),
          courtConfidence: Number(parsed.courtConfidence ?? 0),
          framesScored: Number(parsed.framesScored ?? 0),
        });
      } catch (e) {
        reject(new Error(`tilt-detect parse failed: ${e}`));
      }
    });
  });
}
```

- [ ] **Step 2: tsc check**

```bash
cd api
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
cd ..
git add api/src/services/qualityService.ts
git commit -m "feat(c): add runTiltDetect helper (ffmpeg frame extract + CLI spawn)"
```

---

## Task 11: Wire tilt-detect into confirm + rotation into optimize

This task is where Component B's logic actually flips on. It modifies the existing confirm + optimize flow in `processingService.ts`.

**Files:**
- Modify: `api/src/services/processingService.ts`

- [ ] **Step 1: Call tilt-detect during confirm (alongside brightness)**

In `processingService.ts`, find the section (around line 94) that computes brightness and merges it into `qualityReportJson`. Add tilt-detect right after the brightness block. Use mergeQualityReports (prior values for these fields are null at this stage, so merge is safe):

```ts
// ... existing brightness block ...

// Project C Component B: detect tilt at confirm time so optimize can rotate.
let tiltResult: TiltDetectResult | null = null;
try {
  tiltResult = await runTiltDetect(inputPath);
  console.log(
    `[TILT] Video ${videoId} tiltDeg=${tiltResult.tiltDeg.toFixed(2)} ` +
    `confidence=${tiltResult.courtConfidence.toFixed(2)} ` +
    `framesScored=${tiltResult.framesScored}`,
  );
} catch (err) {
  console.log(`[TILT] Failed to detect tilt for ${videoId}:`, err);
}

// ... existing qualityReportPatch build ...
// Add tilt fields to the patch if we got them:
if (tiltResult) {
  qualityReportPatch = mergeQualityReports([
    qualityReportPatch ?? { version: 2, issues: [] },
    {
      version: 2,
      issues: [],
      tiltDeg: tiltResult.tiltDeg,
      courtConfidence: tiltResult.courtConfidence,
      autoRotated: false,
    },
  ]);
}
```

Import `runTiltDetect` and `TiltDetectResult` at the top of the file:
```ts
import { mergeQualityReports, runTiltDetect } from './qualityService.js';
import type { QualityReport, TiltDetectResult } from './qualityService.js';
```

- [ ] **Step 2: Add rotation filter branch to optimize**

Find the `optimizeVideo` function. Before the FFmpeg invocation, load the quality report and decide whether to rotate:

```ts
// Project C Component B: read persisted quality report to decide if we rotate.
const vid = await prisma.video.findUnique({ where: { id: videoId } });
const qr = (vid?.qualityReportJson as Partial<QualityReport> | null) ?? {};
const shouldRotate =
  !qr.autoRotated &&
  (qr.tiltDeg ?? 0) > 5 &&
  (qr.courtConfidence ?? 0) > 0.8;
const rotationRad = shouldRotate ? -((qr.tiltDeg ?? 0) * Math.PI) / 180 : 0;
```

Append the rotate filter to the existing FFmpeg argument list when `shouldRotate`:

```ts
const vfArgs: string[] = [];
// ... existing -vf entries the function already builds, if any ...
if (shouldRotate) {
  vfArgs.push(`rotate=${rotationRad}:ow=iw:oh=ih:c=black`);
}
const ffmpegArgs = [
  // ... existing args ...
  ...(vfArgs.length ? ['-vf', vfArgs.join(',')] : []),
  // ... existing output path ...
];
```

*(Adapt the exact insertion point to whatever pattern the current function uses — some functions build `-vf` differently. The invariant is: append the filter when `shouldRotate`, leave it out otherwise.)*

- [ ] **Step 3: After a successful rotate, update state via read-mutate-write**

Inside the same `optimizeVideo`, after the FFmpeg process exits successfully AND `shouldRotate` was true:

```ts
if (shouldRotate) {
  // Read-mutate-write (NOT merge): merge's "first non-null wins" would keep
  // the old tiltDeg + autoRotated: false. See design spec §Merge caveat.
  await prisma.$transaction(async (tx) => {
    const row = await tx.video.findUnique({ where: { id: videoId } });
    const prior = (row?.qualityReportJson as QualityReport | null) ?? { version: 2, issues: [] };
    const updated: QualityReport = {
      ...prior,
      autoRotated: true,
      tiltDeg: 0,
      autoFixes: [
        ...(prior.autoFixes ?? []),
        {
          id: 'auto_straightened',
          message: `Auto-straightened by ${Math.round(qr.tiltDeg ?? 0)}°`,
          appliedAt: new Date().toISOString(),
          data: { originalTiltDeg: qr.tiltDeg ?? 0 },
        },
      ],
    };
    await tx.video.update({
      where: { id: videoId },
      data: {
        qualityReportJson: updated as unknown as Prisma.InputJsonValue,
        courtCalibrationJson: null, // invalidated by rotation
      },
    });
  });
}
```

- [ ] **Step 4: tsc check**

```bash
cd api
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 5: Unit test for the rotate-filter decision (add to existing qualityService.test.ts if present, else new file)**

Because the decision logic is a small predicate embedded in `optimizeVideo`, it's easier to validate end-to-end than to mock the FFmpeg child process. Write a unit test for the pure predicate — extract it to a named function if the test surface demands it:

```ts
// api/tests/processingService.rotateDecision.test.ts
import { describe, expect, it } from 'vitest';

function shouldAutoRotate(qr: {
  autoRotated?: boolean; tiltDeg?: number | null; courtConfidence?: number | null;
}): boolean {
  return !qr.autoRotated && (qr.tiltDeg ?? 0) > 5 && (qr.courtConfidence ?? 0) > 0.8;
}

describe('shouldAutoRotate', () => {
  it('fires when tilt > 5 and confidence > 0.8', () => {
    expect(shouldAutoRotate({ tiltDeg: 7, courtConfidence: 0.85 })).toBe(true);
  });
  it('does not fire when tilt is 5 (strict >)', () => {
    expect(shouldAutoRotate({ tiltDeg: 5, courtConfidence: 0.9 })).toBe(false);
  });
  it('does not fire when confidence is 0.8 (strict >)', () => {
    expect(shouldAutoRotate({ tiltDeg: 10, courtConfidence: 0.8 })).toBe(false);
  });
  it('does not fire when already autoRotated', () => {
    expect(shouldAutoRotate({ autoRotated: true, tiltDeg: 10, courtConfidence: 0.9 })).toBe(false);
  });
  it('does not fire on null fields', () => {
    expect(shouldAutoRotate({})).toBe(false);
  });
});
```

If you wrote the predicate inline inside `optimizeVideo`, extract it to a local exported helper (`export function shouldAutoRotate(qr): boolean`) and import it in the test. This is worth the refactor — the predicate is safety-critical.

- [ ] **Step 6: Run the new test**

```bash
npx vitest run tests/processingService.rotateDecision.test.ts
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
cd ..
git add api/src/services/processingService.ts api/tests/processingService.rotateDecision.test.ts
git commit -m "feat(c): auto-rotate during optimize when tilt>5 AND courtConf>0.8"
```

---

## Task 12: Render autoFixes in the web UI

Light-touch: a single-line checkmark note beneath `QualityReportBanner` when `autoFixes` is present. Not banner real estate, not a modal.

**Files:**
- Create: `web/src/components/AutoFixNote.tsx`
- Modify: `web/src/components/QualityReportBanner.tsx` (or its parent layout where the banner is rendered)

- [ ] **Step 1: Create the component**

```tsx
// web/src/components/AutoFixNote.tsx
import React from 'react';
import type { AutoFix } from '@/types/rally';

export function AutoFixNote({ fixes }: { fixes: AutoFix[] | undefined }) {
  if (!fixes || fixes.length === 0) return null;
  return (
    <ul
      style={{
        margin: '4px 0 0 0',
        padding: 0,
        listStyle: 'none',
        fontSize: 12,
        color: '#4caf50',
      }}
    >
      {fixes.map((fx) => (
        <li key={fx.id}>✓ {fx.message}</li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 2: Render it adjacent to the banner**

Find where `QualityReportBanner` is consumed (grep `QualityReportBanner` in `web/src/`). Immediately after it, render:

```tsx
<AutoFixNote fixes={qualityReport?.autoFixes} />
```

- [ ] **Step 3: tsc check**

```bash
cd web
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Smoke-render manually**

Start dev server:
```bash
cd ..
make dev
```
Open a video whose `qualityReportJson.autoFixes` is non-empty (you can hand-insert one via `npx prisma studio` for the test). Confirm the green checkmark note renders below the banner.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/AutoFixNote.tsx web/src/components/QualityReportBanner.tsx
git commit -m "feat(c): render quality-report autoFixes as checkmark note"
```

---

## Task 13: Idempotency + end-to-end regression

Verify three invariants:
1. Running confirm twice on the same tilted fixture rotates it exactly once.
2. A 3° fixture (below threshold) is left alone.
3. The A1 validation harness on the 7 A1 fixtures still reports the expected A1 baseline — now strengthened for the 2 negatives that slipped A1.

**Files:**
- Create: `analysis/scripts/validate_project_c.py` (extension of `validate_quality_checks.py` approach)

- [ ] **Step 1: Write the validation script**

```python
r"""End-to-end Project C validation:
  1. A1 regression — 5 negatives + 2 positives → same A1 behavior.
  2. Component A — CLIP classifier blocks indoor + non-VB negatives.
  3. Component B — manufactured tilt fixtures trigger rotation idempotently.

Usage:
    uv run python analysis/scripts/validate_project_c.py

Writes a report to /tmp/project_c_validation_<date>.{log,json}.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

A1_NEGATIVES = [
    "~/Desktop/rallies/Negative/bad angle.mp4",
    "~/Desktop/rallies/Negative/very bad angle.mp4",
    "~/Desktop/rallies/Negative/indoor 1.mp4",
    "~/Desktop/rallies/Negative/indoor 2.mp4",
    "~/Desktop/rallies/Negative/not related.mp4",
]
A1_POSITIVES = [
    "~/Desktop/rallies/Matches/match.mp4",
    "~/Desktop/rallies/Matches/Newport Beach Games 10_12_25 Paul_Nate vs Colin_Tim - Nathan Gali (1080p, h264).mp4",
]
TILT_FIXTURES = [
    ("~/Desktop/rallies/Negative/tilt_3deg.mp4", 3.0, False),   # below threshold, no rotate
    ("~/Desktop/rallies/Negative/tilt_6deg.mp4", 6.0, True),
    ("~/Desktop/rallies/Negative/tilt_10deg.mp4", 10.0, True),
    ("~/Desktop/rallies/Negative/tilt_15deg.mp4", 15.0, True),
]


def _extract_frames(video_path: Path, n: int = 5) -> Path:
    d = Path(tempfile.mkdtemp(prefix="proj_c_valid_"))
    # Evenly sample frames via ffmpeg
    for i in range(n):
        ts = f"{((i + 0.5) / n) * 100:.2f}%"
        out = d / f"frame_{i:02d}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", ts, "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(out)],
            check=True,
        )
    return d


def _preview_check(frames_dir: Path, width: int, height: int, duration_s: float) -> dict:
    out = subprocess.run(
        ["uv", "run", "rallycut", "preview-check", str(frames_dir),
         "--width", str(width), "--height", str(height),
         "--duration-s", str(duration_s), "--json"],
        capture_output=True, text=True, check=True,
    ).stdout
    import re
    m = re.search(r"\{[\s\S]*\}", out)
    assert m, f"no json in preview-check output: {out}"
    return json.loads(m.group(0))


def _tilt_detect(frames_dir: Path) -> dict:
    out = subprocess.run(
        ["uv", "run", "rallycut", "tilt-detect", str(frames_dir), "--json"],
        capture_output=True, text=True, check=True,
    ).stdout
    import re
    m = re.search(r"\{[\s\S]*\}", out)
    assert m
    return json.loads(m.group(0))


def _ffprobe_meta(video_path: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout
    kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    return int(kv["width"]), int(kv["height"]), float(kv.get("duration", 0))


def main() -> int:
    results = {"a1_regression": [], "component_a": [], "component_b": []}
    failures: list[str] = []

    # A1 regression — all 5 negatives must block, all 2 positives must pass
    print("== A1 regression ==", flush=True)
    for vs in A1_NEGATIVES:
        v = Path(vs).expanduser()
        w, h, d = _ffprobe_meta(v)
        frames = _extract_frames(v)
        r = _preview_check(frames, w, h, d)
        blocked = any(i["tier"] == "block" for i in r.get("issues", []))
        ids = [i["id"] for i in r.get("issues", [])]
        print(f"  {v.name}: blocked={blocked}, issue_ids={ids}", flush=True)
        results["a1_regression"].append({"video": v.name, "blocked": blocked, "ids": ids})
        if not blocked:
            failures.append(f"A1 negative not blocked: {v.name}")

    for vs in A1_POSITIVES:
        v = Path(vs).expanduser()
        w, h, d = _ffprobe_meta(v)
        frames = _extract_frames(v)
        r = _preview_check(frames, w, h, d)
        blocked = any(i["tier"] == "block" for i in r.get("issues", []))
        ids = [i["id"] for i in r.get("issues", [])]
        print(f"  {v.name}: blocked={blocked}, issue_ids={ids}", flush=True)
        results["a1_regression"].append({"video": v.name, "blocked": blocked, "ids": ids})
        if blocked:
            failures.append(f"A1 positive incorrectly blocked: {v.name}  ids={ids}")

    # Component B — tilt-detect must report within tolerance
    print("== Component B — tilt detection ==", flush=True)
    for vs, expected_deg, expected_rotate in TILT_FIXTURES:
        v = Path(vs).expanduser()
        if not v.exists():
            print(f"  SKIP {v.name} (missing; run generate_tilt_fixtures.sh)", flush=True)
            continue
        frames = _extract_frames(v, n=5)
        t = _tilt_detect(frames)
        got = t.get("tiltDeg", 0)
        conf = t.get("courtConfidence", 0)
        within = abs(got - expected_deg) <= 2.0  # ± 2° tolerance
        would_rotate = got > 5 and conf > 0.8
        print(f"  {v.name}: tilt={got:.2f}° (exp {expected_deg}±2, within={within}) conf={conf:.2f} would_rotate={would_rotate}", flush=True)
        results["component_b"].append({
            "video": v.name, "expected_deg": expected_deg, "got_deg": got,
            "conf": conf, "would_rotate": would_rotate, "within_tolerance": within,
        })
        if not within:
            failures.append(f"B: {v.name} tilt detection off: expected ≈{expected_deg}, got {got:.2f}")
        if would_rotate != expected_rotate:
            failures.append(
                f"B: {v.name} rotation decision wrong: expected {expected_rotate}, got {would_rotate}"
            )

    # Write report
    report_path = Path(f"/tmp/project_c_validation_{time.strftime('%Y%m%d_%H%M%S')}.json")
    report_path.write_text(json.dumps({"results": results, "failures": failures}, indent=2))
    print(f"\nWrote {report_path}", flush=True)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

```bash
cd analysis
uv run python scripts/validate_project_c.py
```

Expected: exit 0, "ALL CHECKS PASSED". If any failure: stop and investigate before shipping. Specific expected outcomes:

| Fixture | Expected |
|---|---|
| 3 A1 negatives (bad angle, very bad angle, indoor 1) | blocked by `wrong_angle_or_not_volleyball` |
| `indoor 2` | blocked by `not_beach_volleyball` (NEW, Component A) |
| `not related` | blocked by `not_beach_volleyball` (NEW, Component A) |
| 2 A1 positives | not blocked |
| `tilt_3deg` | tilt ≈ 3, would_rotate = False |
| `tilt_{6,10,15}deg` | tilt ≈ N ± 2, would_rotate = True |

- [ ] **Step 3: Commit the script + the run log**

```bash
cp /tmp/project_c_validation_*.json analysis/reports/
git add analysis/scripts/validate_project_c.py analysis/reports/project_c_validation_*.json
git commit -m "feat(c): end-to-end validation harness (A1 regression + A block + B detect)"
```

---

## Task 14: Update memory + CLAUDE.md docs

Following the pattern A1 set post-validation: update the memory index + the two CLAUDE.md files so future sessions see the current state.

**Files:**
- Create: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/project_c_sport_sanity_2026_04_15.md`
- Modify: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` (add pointer line)
- Modify: `analysis/CLAUDE.md`
- Modify: `api/CLAUDE.md`

- [ ] **Step 1: Write the memory file**

```markdown
---
name: Project C Sport Sanity + Auto-Rotate (2026-04-15)
description: Project C shipped — CLIP beach-VB classifier closes A1's indoor/non-VB slip-through; silent auto-rotate during optimize fixes tilted uploads; qualityReportJson gains autoFixes field.
type: project
---

**Status:** MERGED as <sha>. Branch `feat/c-sport-sanity`. 14 implementation commits.

**Why:** A1 post-merge validation surfaced two holes that A1 architecturally couldn't close: (1) beach-trained court model confidently finds "courts" in indoor/non-VB footage (indoor 2 → conf 0.78, not related → conf 0.83), so the `wrong_angle_or_not_volleyball` block misses them; (2) tilt detection was dropped because no fixtures existed, but manufactured tilt fixtures make it testable and the auto-fix keeps it out of the advisory tier (which would just annoy users without helping).

**Shipped surface:**
- Analysis: `beach_vb_classifier.py` (binary prompts, open-clip ViT-B/32 lazy import), `rallycut preview-check` extended to run CLIP on the same 5 frames, new `rallycut tilt-detect` CLI (median of high-conf frame tilts), `_baseline_tilt_deg` helper restored.
- API: `QualityReport` gains `autoFixes`, `autoRotated`, `tiltDeg`, `courtConfidence`; `mergeQualityReports` concat+dedupes autoFixes; `processingService.ts` calls `tilt-detect` at confirm + appends FFmpeg rotate filter at optimize; read-mutate-write for rotate-success state (merge's first-non-null-wins would keep the old tiltDeg).
- Web: `AutoFixNote` renders a checkmark line beneath `QualityReportBanner`.

**Calibration locked in (do not reopen without new data):**
- BEACH_VB_BLOCK_THRESHOLD = <value>; ship gate = `lowest_positive - highest_negative >= 0.15` passed at <date>. Report: `analysis/reports/beach_vb_calibration_<date>.json`.
- Auto-rotate fires only when `tiltDeg > 5° AND courtConfidence > 0.8`. Both are strict `>`; either alone admits edge cases.

**Decisions locked in:**
- **Client-side gate only for A.** No server-side CLIP fallback at preflight. If edge cases slip through, add then — don't pre-build.
- **Silent auto-rotate for B.** No user-confirm prompt. AND-composed thresholds gate the safety-critical case.
- **autoFixes separate field.** Not in `issues`. Not a new `INFO` tier. Read-mutate-write for state transitions; merge for cross-source combining.
- **open-clip as optional [preflight] extra.** Keep CI fast; download weights only when needed.

**How to apply:**
- When adding a new preflight check: empirical evidence required (calibration ≥ 0.15 gap for A-style classifiers; for invariants, justify each threshold in the spec). A1's experience: shipping "conservative default" thresholds produces checks that get ripped out post-validation.
- When mutating `qualityReportJson` state in place (e.g., setting `autoRotated = true`), use read-mutate-write inside a transaction. `mergeQualityReports` is wrong for state transitions.
- `autoFixes` id dedupe is first-occurrence; prior values survive a re-merge.

**Known gaps / follow-ups:**
- Preflight-stage CLIP fallback (design §Out of Scope).
- Undo-auto-rotate UX control (only add if false-positive reports arrive in production).
- S3 backup of negative + tilt fixtures.
```

- [ ] **Step 2: Add MEMORY.md pointer**

At the top of `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`, insert a new line beneath the A2b line:

```markdown
- [**Project C Sport Sanity — MERGED 2026-04-15**](project_c_sport_sanity_2026_04_15.md) — CLIP beach-VB classifier (client-side pre-upload) closes A1's indoor/non-VB slip-through; silent auto-rotate during optimize; qualityReportJson.autoFixes field. Calibration gap <value>, threshold <value>.
```

And update the e2e pipeline redesign pointer to mark C complete.

- [ ] **Step 3: Update `analysis/CLAUDE.md`**

Find the line describing `rallycut preflight` and add:

```
uv run rallycut tilt-detect <frames-dir>         # Emit {tiltDeg, courtConfidence, framesScored} from a directory of JPEGs. Used server-side at POST /v1/videos/:id/confirm to decide auto-rotate (Project C).
```

In the preview-check description, note: "Runs court-keypoint detection + camera_geometry + CLIP beach-VB classifier (Project C, open-clip ViT-B/32 via [preflight] extra)."

- [ ] **Step 4: Update `api/CLAUDE.md`**

In the "Analysis Pipeline" section, update the assess-quality description:

```
- Preflight checks: metadata invariants + camera_geometry (block on court-conf < 0.6 OR wrong angle).
- Project C adds: CLIP beach-VB classifier at the preview gate (blocks indoor / non-VB uploads before they start); silent auto-rotate during optimize when tiltDeg > 5° AND courtConfidence > 0.8 (paired with qualityReportJson.autoFixes).
```

- [ ] **Step 5: Commit**

```bash
git add analysis/CLAUDE.md api/CLAUDE.md
# The memory dir is outside the repo; commit that separately:
cd ~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory
# (if this is under git elsewhere — otherwise just save is fine, harness persists)
cd /Users/mario/Personal/Projects/RallyCut
git commit -m "docs(c): document Project C in CLAUDE.md files + memory"
```

---

## Task 15: Final CI pass + PR

- [ ] **Step 1: Run the full analysis test suite**

```bash
cd analysis
uv run pytest tests/unit -v
```

Expected: all green.

- [ ] **Step 2: Run mypy + ruff**

```bash
uv run mypy rallycut/
uv run ruff check rallycut/
```

Expected: both clean.

- [ ] **Step 3: Run api + web tsc**

```bash
cd ../api
npx tsc --noEmit
cd ../web
npx tsc --noEmit
```

Expected: both clean.

- [ ] **Step 4: Re-run Project C validation harness**

```bash
cd ../analysis
uv run python scripts/validate_project_c.py
```

Expected: exit 0, ALL CHECKS PASSED.

- [ ] **Step 5: Push + open PR**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git push -u origin feat/c-sport-sanity
gh pr create --title "feat: Project C — sport-sanity + silent auto-rotate" --body "$(cat <<'EOF'
## Summary

- Component A: CLIP beach-VB classifier wired into the client-side pre-upload gate. Closes the indoor/non-VB slip-through A1 validation surfaced.
- Component B: silent auto-rotate during the existing FFmpeg optimize pass when `tiltDeg > 5° AND courtConfidence > 0.8`.
- Schema addition: `qualityReportJson.autoFixes` (separate field from `issues`).

## Design spec

`docs/superpowers/specs/2026-04-15-project-c-sport-sanity-design.md`

## Test plan

- [ ] Manufactured tilt fixtures generated (`bash analysis/scripts/generate_tilt_fixtures.sh`).
- [ ] `uv run python analysis/scripts/calibrate_beach_vb.py ...` reports ship gate passed.
- [ ] `uv run python analysis/scripts/validate_project_c.py` exits 0.
- [ ] `uv run pytest tests/unit -v` green.
- [ ] `npx tsc --noEmit` green in api/ and web/.
- [ ] Manual: upload `indoor 2.mp4` → upload blocked pre-upload.
- [ ] Manual: upload a `tilt_10deg.mp4` → confirm the editor shows a straight video with a "✓ Auto-straightened by 10°" note.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review (completed during plan authoring)

**Spec coverage check:**
- A architecture (client-side gate, open-clip, binary prompts) → Tasks 2, 3, 4.
- A calibration (≥ 0.15 gap ship gate) → Task 5.
- B tilt-detect at confirm → Tasks 6, 7, 8, 10, 11.
- B FFmpeg rotate at optimize → Task 11.
- B idempotency → Task 11 (state write) + Task 13 (validation harness covers the predicate).
- `autoFixes` schema addition (API + web) → Task 9.
- UI rendering → Task 12.
- Merge caveat for state transitions → Task 11 (explicit read-mutate-write).
- Calibration + fixture validation (ship gates) → Tasks 5, 13.
- Memory + docs update → Task 14.

**Placeholder scan:** No TBDs. One literal `<value>` placeholder in Task 14's memory file — it's correct as-written because the calibrated threshold isn't chosen until Task 5 runs. The engineer fills it in at write time.

**Type consistency:** `AutoFix`, `QualityReport`, `TiltDetectResult`, `BEACH_VB_BLOCK_THRESHOLD`, `baseline_tilt_deg` names used consistently across Tasks 3, 6, 7, 9, 10, 11.

**Ambiguity check:** Task 11 Step 2 says "adapt the exact insertion point to whatever pattern the current function uses" — that's flagged as guidance not specification, because the current optimize-pass `-vf` shape isn't one the plan can read without looking at the file. The invariant (append the filter when `shouldRotate`) is unambiguous.

**Scope check:** Single coherent feature. A + B are independent but share the `QualityReport` schema change (Task 9), which is the tie-breaker for bundling into one plan per the spec's "A + B bundled" decision.
