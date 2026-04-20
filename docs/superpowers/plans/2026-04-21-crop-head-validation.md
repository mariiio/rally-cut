# Per-Candidate Crop Head Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine in 2-3 days whether a per-candidate player-bbox + ball-patch crop head can learn contact-vs-non-contact signal that beats the 88.0% F1 / 91.2% Action Acc trajectory GBM baseline — with explicit kill gates pre-registered so we stop early if the architecture doesn't work.

**Architecture:** Frozen pretrained backbone (ImageNet ResNet-18 or CLIP ViT-B/32) extracts per-frame features from player-bbox crops over a ±4f window. Features mean-pooled across time, fed to a simple MLP head. Binary classification: P(contact). **This is the frozen-backbone sanity probe only — Phase 1 of a 4-phase plan.** No end-to-end fine-tuning in Phase 1.

**Tech Stack:** PyTorch + torchvision (ResNet-18 ImageNet pretrained), OpenCV for crop extraction, numpy, scikit-learn for baseline logistic probe. uv for Python env. Modal GPU for inference if CLIP chosen.

**Context for the engineer:**
- Phase 0 feasibility diagnostic PASSED on 2026-04-20. 100% of target cases (32 `looks_fixable` + 56 `classifier_boundary` FNs) have both player bbox and ball within ±4f window. See `memory/crop_head_phase0_2026_04_20.md`.
- VideoMAE at scene-level failed (36.9% F1) because of signal localization. E2E-Spot per-frame on full frames failed (~50%). See `memory/videomae_contact_nogo_2026_04_19.md`.
- Canonical baseline is 88.0% Contact F1 / 91.2% Action Acc on 68 videos / 364 rallies / 2095 contacts via `scripts/eval_loo_video.py`. Locked. All measurements compare to this.
- 2095 positives is TIGHT for training. Use pretrained backbone + frozen.

---

## File Structure

| File | Purpose |
|---|---|
| `analysis/scripts/extract_crop_dataset.py` | Extracts (player_bbox_crop[T], ball_patch_crop[T], label) tuples to disk; one-shot |
| `analysis/rallycut/ml/crop_head/dataset.py` | `CropContactDataset` — torch Dataset reading cached tensors |
| `analysis/rallycut/ml/crop_head/backbone.py` | Frozen ResNet-18 feature extractor (stateless) |
| `analysis/rallycut/ml/crop_head/head.py` | Simple 2-layer MLP head |
| `analysis/scripts/train_crop_head_probe.py` | Training loop + orthogonality+negctl eval |
| `analysis/scripts/crop_head_orthogonality.py` | THE kill-gate test adapted from `orthogonality_probe.py` |
| `analysis/reports/crop_head_phase1_probe_2026_04_YY.md` | Phase 1 decision memo |
| `tests/unit/test_crop_head_dataset.py` | Dataset shape + label correctness tests |
| `tests/unit/test_crop_head_backbone.py` | Backbone stateless-inference test |

---

## Pre-registered ship gates (DO NOT modify after training starts)

**Phase 1 pass gates (ALL must pass):**
1. Test AUC ≥ 0.75 on held-out videos
2. Orthogonality + neg-control: `P(held-out GT contact ≥ 0.5) − P(random non-contact ≥ 0.5) ≥ 0.15` absolute gap
3. Hard-negative performance: AUC on GBM-conf ∈ [0.2, 0.3] candidates ≥ 0.65 (not just "player present" detector)

**Fail ANY gate → NO-GO.** Document findings, redirect to pose-event classifier (separate 2-3 day plan) or GT attribution work.

**Phase 1 pass → Phase 2 is scheduled in a subsequent session.** Do NOT escalate to Phase 2 in the same execution.

---

## Task 1: Feasibility infrastructure setup

**Files:**
- Create: `analysis/rallycut/ml/crop_head/__init__.py` (empty)
- Create: `analysis/rallycut/ml/__init__.py` (if not present)
- Create: `analysis/rallycut/ml/crop_head/dataset.py`
- Create: `tests/unit/test_crop_head_dataset.py`

- [ ] **Step 1.1: Create the dataset module**

```python
# analysis/rallycut/ml/crop_head/dataset.py
"""Per-candidate crop dataset for contact classification.

Loads cached (player_crop, ball_patch, label, meta) tuples from disk.
Crops are pre-extracted by scripts/extract_crop_dataset.py to avoid
re-decoding video on every epoch (slow + non-reproducible if decoding
is non-deterministic).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class CropContactDataset(Dataset):
    """Dataset of pre-extracted (player_crop_T_3_64_64, ball_patch_T_3_32_32, label).

    Crops stored as .npz files, one per candidate, under a root directory
    organized by video_id for easy held-out splitting.
    """

    def __init__(self, cache_root: Path, video_ids: list[str]) -> None:
        self.cache_root = Path(cache_root)
        self.video_ids = list(video_ids)
        self.items: list[tuple[str, str]] = []  # (video_id, file_path)
        for vid in self.video_ids:
            vid_dir = self.cache_root / vid
            if not vid_dir.exists():
                continue
            for f in sorted(vid_dir.glob("*.npz")):
                self.items.append((vid, str(f)))
        if not self.items:
            raise ValueError(f"No crops found under {cache_root} for {len(self.video_ids)} videos")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        vid, path = self.items[idx]
        data = np.load(path)
        return {
            "video_id": vid,
            "rally_id": str(data["rally_id"]),
            "frame": int(data["frame"]),
            "player_crop": torch.from_numpy(data["player_crop"]).float(),  # (T,3,64,64)
            "ball_patch": torch.from_numpy(data["ball_patch"]).float(),    # (T,3,32,32)
            "label": int(data["label"]),
            "gbm_conf": float(data["gbm_conf"]),
            "source": str(data["source"]),  # "gt_positive" | "hard_negative" | "random_negative"
        }
```

- [ ] **Step 1.2: Write the dataset test**

```python
# tests/unit/test_crop_head_dataset.py
"""Unit tests for CropContactDataset."""
from pathlib import Path
import numpy as np
import pytest
import torch

from rallycut.ml.crop_head.dataset import CropContactDataset


def _make_fake_npz(path: Path, video_id: str, label: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rally_id="rally-fake-0001",
        frame=100,
        player_crop=np.random.rand(9, 3, 64, 64).astype(np.float32),
        ball_patch=np.random.rand(9, 3, 32, 32).astype(np.float32),
        label=label,
        gbm_conf=0.4,
        source="gt_positive" if label else "random_negative",
    )


def test_dataset_loads_and_yields_expected_shapes(tmp_path: Path) -> None:
    _make_fake_npz(tmp_path / "vid_a" / "001.npz", "vid_a", 1)
    _make_fake_npz(tmp_path / "vid_a" / "002.npz", "vid_a", 0)
    _make_fake_npz(tmp_path / "vid_b" / "001.npz", "vid_b", 1)

    ds = CropContactDataset(tmp_path, ["vid_a", "vid_b"])
    assert len(ds) == 3
    item = ds[0]
    assert item["player_crop"].shape == (9, 3, 64, 64)
    assert item["ball_patch"].shape == (9, 3, 32, 32)
    assert item["label"] in (0, 1)


def test_dataset_filters_by_video_id(tmp_path: Path) -> None:
    _make_fake_npz(tmp_path / "vid_a" / "001.npz", "vid_a", 1)
    _make_fake_npz(tmp_path / "vid_b" / "001.npz", "vid_b", 1)
    ds = CropContactDataset(tmp_path, ["vid_a"])
    assert len(ds) == 1
    assert ds[0]["video_id"] == "vid_a"


def test_dataset_raises_on_empty_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No crops found"):
        CropContactDataset(tmp_path, ["does-not-exist"])
```

- [ ] **Step 1.3: Run tests to verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_crop_head_dataset.py -v`
Expected: all 3 pass.

- [ ] **Step 1.4: Commit**

```bash
git add analysis/rallycut/ml/__init__.py analysis/rallycut/ml/crop_head/__init__.py analysis/rallycut/ml/crop_head/dataset.py tests/unit/test_crop_head_dataset.py
git commit -m "$(cat <<'EOF'
feat(ml): crop-head dataset module — Phase 1 foundation

First step in the crop-head Phase 1 validation plan. Dataset reads
pre-extracted (player_crop, ball_patch, label) tuples from disk;
video-level filtering supports LOO splits. Extraction script (next task)
populates the cache.

Scoped as pure reader — no training yet. See
docs/superpowers/plans/2026-04-21-crop-head-validation.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Crop extraction script

**Files:**
- Create: `analysis/scripts/extract_crop_dataset.py`

**Context:**
- For each rally, iterate over its ground-truth contacts + the candidate frames that the trajectory generators produce.
- Positive samples: GT contact frames (from `rally.gt_labels`).
- Hard-negative samples: candidate frames NOT within ±7f of any GT with GBM conf ∈ [0.2, 0.3] (from running the production `detect_contacts` with `use_classifier=True` and recording ALL scored candidates via the `_decoder_inputs` pattern that's NOT currently in production).
- Random-negative samples: candidate frames NOT within ±7f of any GT with GBM conf < 0.2.
- For each sample: T=9 frames centered on the target frame, extract player bbox crop (resize to 64×64) and ball patch crop (resize to 32×32).
- Video decoding: use `cv2.VideoCapture` with sequential reads where possible. Cache decoded frames in RAM per-rally.

**Target sizes:**
- Positives: ~2095 (all GT contacts across 68 videos)
- Hard negatives: ~1500 (aim for 15× per rally)
- Random negatives: ~3000 (aim for 30× per rally)

**Step-by-step:**

- [ ] **Step 2.1: Write the extraction script skeleton**

Create `analysis/scripts/extract_crop_dataset.py`:

```python
"""Extract per-candidate crops for Phase 1 crop-head probe.

Outputs: outputs/crop_dataset/<video_id>/<seq>.npz containing
  - rally_id, frame, label, gbm_conf, source
  - player_crop: (T=9, 3, 64, 64) float32 normalized [0,1]
  - ball_patch: (T=9, 3, 32, 32) float32 normalized [0,1]

Usage (cd analysis):
    uv run python scripts/extract_crop_dataset.py
    uv run python scripts/extract_crop_dataset.py --video <id>
    uv run python scripts/extract_crop_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rallycut.tracking.contact_classifier import load_contact_classifier
from rallycut.tracking.contact_detector import ContactDetectionConfig
from scripts.eval_action_detection import load_rallies_with_action_gt
from scripts.eval_loo_video import _precompute

T_FRAMES = 9   # ±4 around target
PLAYER_CROP_SIZE = 64
BALL_PATCH_SIZE = 32
BALL_PATCH_HALF_NORM = 0.04  # ±4% of frame = ~34px at 854 width

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "crop_dataset"


@dataclass
class SampleSpec:
    """A sample to extract."""
    rally_id: str
    video_id: str
    video_path: str
    target_frame: int
    label: int
    gbm_conf: float
    source: str
    # Precomputed at dispatch time:
    player_bbox_by_frame: dict  # {frame: (cx, cy, w, h)} normalized
    ball_by_frame: dict  # {frame: (x, y)} normalized


def _crop_player(img: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Extract player bbox from image, resize to PLAYER_CROP_SIZE square, return HWC float32 [0,1]."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    # Use bbox directly. Add 10% padding to ensure arms/head visible.
    pad = 0.10
    x0 = max(0, int((cx - bw/2 - pad*bw) * w))
    x1 = min(w, int((cx + bw/2 + pad*bw) * w))
    y0 = max(0, int((cy - bh/2 - pad*bh) * h))
    y1 = min(h, int((cy + bh/2 + pad*bh) * h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((PLAYER_CROP_SIZE, PLAYER_CROP_SIZE, 3), dtype=np.float32)
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, (PLAYER_CROP_SIZE, PLAYER_CROP_SIZE))
    return (crop.astype(np.float32) / 255.0)


def _crop_ball(img: np.ndarray, ball_xy: tuple[float, float]) -> np.ndarray:
    """Extract ball-centered patch, resize, return HWC float32 [0,1].
    When ball unavailable, returns zeros."""
    if ball_xy is None:
        return np.zeros((BALL_PATCH_SIZE, BALL_PATCH_SIZE, 3), dtype=np.float32)
    h, w = img.shape[:2]
    bx, by = ball_xy
    x0 = max(0, int((bx - BALL_PATCH_HALF_NORM) * w))
    x1 = min(w, int((bx + BALL_PATCH_HALF_NORM) * w))
    y0 = max(0, int((by - BALL_PATCH_HALF_NORM) * h))
    y1 = min(h, int((by + BALL_PATCH_HALF_NORM) * h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((BALL_PATCH_SIZE, BALL_PATCH_SIZE, 3), dtype=np.float32)
    patch = img[y0:y1, x0:x1]
    patch = cv2.resize(patch, (BALL_PATCH_SIZE, BALL_PATCH_SIZE))
    return (patch.astype(np.float32) / 255.0)


def extract_one_sample(
    cap: cv2.VideoCapture,
    spec: SampleSpec,
    frame_cache: dict[int, np.ndarray],
) -> dict | None:
    """Extract (player_crop_T, ball_patch_T) tensors for one sample.
    Returns None if essential data missing."""
    t_center = spec.target_frame
    half = T_FRAMES // 2

    player_crops = np.zeros((T_FRAMES, 3, PLAYER_CROP_SIZE, PLAYER_CROP_SIZE), dtype=np.float32)
    ball_patches = np.zeros((T_FRAMES, 3, BALL_PATCH_SIZE, BALL_PATCH_SIZE), dtype=np.float32)

    for i, offset in enumerate(range(-half, half + 1)):
        f = t_center + offset
        if f not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            frame_cache[f] = img
        img = frame_cache[f]
        # Player: use CLOSEST available bbox within ±2f (track may skip frames)
        bbox = spec.player_bbox_by_frame.get(f)
        if bbox is None:
            for d in (1, -1, 2, -2):
                bbox = spec.player_bbox_by_frame.get(f + d)
                if bbox:
                    break
        if bbox is not None:
            crop_hwc = _crop_player(img, bbox)
            player_crops[i] = crop_hwc.transpose(2, 0, 1)  # CHW
        # Ball
        ball = spec.ball_by_frame.get(f)
        if ball is None:
            for d in (1, -1, 2, -2):
                ball = spec.ball_by_frame.get(f + d)
                if ball:
                    break
        if ball is not None:
            patch_hwc = _crop_ball(img, ball)
            ball_patches[i] = patch_hwc.transpose(2, 0, 1)

    return {
        "rally_id": spec.rally_id,
        "frame": spec.target_frame,
        "player_crop": player_crops,
        "ball_patch": ball_patches,
        "label": spec.label,
        "gbm_conf": spec.gbm_conf,
        "source": spec.source,
    }


# [implement main() with: iterate rallies, resolve video path,
#  build SampleSpec list per rally (positives + hard-negs + random-negs),
#  open VideoCapture, extract each sample, save .npz files under OUTPUT_ROOT]
```

**Finish the main() loop** in this step to completion — it should:
- Load all 364 rallies via `load_rallies_with_action_gt()`
- For each rally, build the SampleSpec list:
  - Positives: one per GT contact frame
  - Hard-negs: sample up to 5 per rally from candidate frames where GBM conf ∈ [0.2, 0.3] AND distance from any GT > 7f
  - Random-negs: sample up to 10 per rally from candidate frames with conf < 0.2 AND distance from any GT > 7f
- Open video once per rally, process all samples, save .npz files
- Print per-rally progress (`[17/364] rally_id: 5 pos + 3 hard + 8 rnd saved (3.2s)`)
- Emit a summary at the end with total counts

- [ ] **Step 2.2: Run on 5 rallies as smoke test**

Run: `cd analysis && uv run python scripts/extract_crop_dataset.py --video <pick-one-video-id>`
Expected: ~5-15 .npz files under `outputs/crop_dataset/<video_id>/`. Verify they load via:

```python
import numpy as np
d = np.load("outputs/crop_dataset/<vid>/<file>.npz")
assert d["player_crop"].shape == (9, 3, 64, 64)
assert d["ball_patch"].shape == (9, 3, 32, 32)
assert d["label"] in (0, 1)
```

- [ ] **Step 2.3: Run on full 68 videos**

Run: `cd analysis && uv run python scripts/extract_crop_dataset.py 2>&1 | tee reports/crop_extraction_2026_04_YY.log`
Expected wall-clock: 30-60 min (64 videos × ~30 samples × ~100ms per sample). Monitor via tail-f.

Sanity:
- Total samples ~6500 (2095 pos + 1500 hard + 3000 rnd)
- One directory per video under `outputs/crop_dataset/`
- No videos missing

- [ ] **Step 2.4: Commit extraction script (not the data)**

`outputs/` is gitignored. Commit just the script.

```bash
git add analysis/scripts/extract_crop_dataset.py
git commit -m "$(cat <<'EOF'
feat(ml): crop dataset extractor for Phase 1 probe

One-shot extractor that reads 68 videos and writes .npz tuples of
(player_crop[9,3,64,64], ball_patch[9,3,32,32], label, gbm_conf, source)
to outputs/crop_dataset/<video_id>/. Samples = 2095 GT positives +
1500 hard negatives (GBM conf 0.2-0.3) + 3000 random negatives.

Reuses scripts/eval_loo_video.py's _precompute for player/ball lookups.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Frozen backbone + MLP head

**Files:**
- Create: `analysis/rallycut/ml/crop_head/backbone.py`
- Create: `analysis/rallycut/ml/crop_head/head.py`
- Create: `tests/unit/test_crop_head_backbone.py`

**Design:**
- Backbone: ImageNet-pretrained ResNet-18 from torchvision. Remove final fc, take avgpool (512-dim).
- Apply to each frame × 2 inputs (player + ball), concatenate: 9 × (512+512) = 9×1024 per sample.
- Head: mean-pool across T → 1024 → MLP(1024, 256, 1).
- Frozen backbone: set `requires_grad=False` on all backbone params.

- [ ] **Step 3.1: Implement backbone wrapper**

```python
# analysis/rallycut/ml/crop_head/backbone.py
"""Frozen ImageNet-pretrained ResNet-18 feature extractor for crop inputs."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class FrozenResNet18(nn.Module):
    """Per-frame feature extractor. Output: 512-dim feature per input image.

    Applied to (B*T, 3, H, W) → returns (B*T, 512). Caller reshapes to
    (B, T, 512). All weights frozen — this module contributes zero
    gradient to the training loop.
    """

    def __init__(self) -> None:
        super().__init__()
        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # drop final fc
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.eval()  # BN in eval mode
        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        feat = self.backbone(x)  # (B, 512, 1, 1)
        return feat.flatten(1)   # (B, 512)
```

- [ ] **Step 3.2: Implement MLP head**

```python
# analysis/rallycut/ml/crop_head/head.py
"""MLP classifier head over mean-pooled per-frame features."""
from __future__ import annotations

import torch
import torch.nn as nn


class CropHeadMLP(nn.Module):
    """Takes (B, T, D_player + D_ball) → (B, 1) binary logit.

    Mean-pools across T, then MLP. Minimal architecture for the Phase 1
    frozen-backbone sanity probe. If this works, Phase 2 ablates pooling
    choice (mean vs attention) and temporal window.
    """

    def __init__(self, d_in: int = 1024, d_hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        pooled = x.mean(dim=1)  # (B, D)
        return self.net(pooled).squeeze(-1)  # (B,)
```

- [ ] **Step 3.3: Write backbone test**

```python
# tests/unit/test_crop_head_backbone.py
"""Tests for the frozen ResNet-18 backbone."""
import torch

from rallycut.ml.crop_head.backbone import FrozenResNet18
from rallycut.ml.crop_head.head import CropHeadMLP


def test_backbone_output_shape() -> None:
    bb = FrozenResNet18()
    x = torch.rand(4, 3, 64, 64)
    out = bb(x)
    assert out.shape == (4, 512)


def test_backbone_is_frozen() -> None:
    bb = FrozenResNet18()
    assert all(not p.requires_grad for p in bb.parameters())


def test_backbone_deterministic_under_eval() -> None:
    """Frozen backbone with BN in eval mode must be deterministic."""
    bb = FrozenResNet18()
    x = torch.rand(2, 3, 64, 64)
    out_a = bb(x)
    out_b = bb(x)
    assert torch.allclose(out_a, out_b)


def test_mlp_head_shape() -> None:
    head = CropHeadMLP(d_in=1024)
    x = torch.rand(4, 9, 1024)
    out = head(x)
    assert out.shape == (4,)
```

- [ ] **Step 3.4: Run tests**

Run: `cd analysis && uv run pytest tests/unit/test_crop_head_backbone.py -v`
Expected: all 4 pass.

- [ ] **Step 3.5: Commit**

```bash
git add analysis/rallycut/ml/crop_head/backbone.py analysis/rallycut/ml/crop_head/head.py tests/unit/test_crop_head_backbone.py
git commit -m "$(cat <<'EOF'
feat(ml): frozen ResNet-18 backbone + MLP head for crop-head Phase 1

ResNet-18 ImageNet-pretrained, frozen (requires_grad=False, BN eval).
MLP head: mean-pool temporal → Linear(1024, 256, 1). Tests verify
stateless inference and correct output shapes.

This is deliberately the simplest architecture that could work. Phase 1
tests whether the architectural premise holds; Phase 2 ablates choices.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Training script with orthogonality + neg-control gates

**Files:**
- Create: `analysis/scripts/train_crop_head_probe.py`

**Design:**
- Video-level split: **53 train / 5 val / 10 test** (hold out the 10 test videos consistently across Phase 1). Rationale: 10 test videos ≈ 300 samples total; at 30-50 positives per group the binomial SE is ≈3pp, making the 15pp orthogonality gap gate robust at 95% confidence. A smaller test set (e.g. 3 videos) leaves the gate marginally detectable and risks false-NO-GO on noise. Select the 10 test videos deterministically (sorted video_id, every 7th) for reproducibility.
- Training: frozen backbone (no gradients to it), only MLP head trained. AdamW(lr=1e-3, weight_decay=1e-4). BCEWithLogitsLoss with `pos_weight` to balance classes. Batch size 64. Epochs 15.
- Eval on held-out 3 test videos:
  - AUC (overall) and per-class (serve/receive/set/attack/dig/block)
  - Confusion at threshold 0.5 (and threshold sweep 0.3–0.7)
- Orthogonality + negative control (the decisive test):
  - For each GT contact in test videos: record the model's output at that frame
  - For a random non-contact frame in the same rally: record model output
  - Compare distributions. Gap ≥ 0.15 at threshold 0.5 = pass; < 0.15 = FAIL (VideoMAE-style indistinguishability).
- Hard-negative eval: AUC on the subset of test-video samples tagged `source=hard_negative`.

- [ ] **Step 4.1: Write training + eval script**

```python
# analysis/scripts/train_crop_head_probe.py
"""Phase 1 crop-head frozen-backbone probe.

Trains a simple MLP head over frozen ResNet-18 features extracted from
pre-cached player + ball crops. Validates against the pre-registered
ship gates in docs/superpowers/plans/2026-04-21-crop-head-validation.md.

Usage (cd analysis):
    uv run python scripts/train_crop_head_probe.py
    uv run python scripts/train_crop_head_probe.py --epochs 20 --seed 1337
"""
# [Full implementation including train/val/test split, training loop,
#  AUC + threshold-sweep eval, orthogonality + neg-control test, hard-neg
#  eval, per-class stratified report. Output: reports/crop_head_phase1_probe_<date>.md
#  with PASS/FAIL against each gate.]
```

Due to length, the engineer implements the full script following the design above. Key eval block:

```python
def orthogonality_test(model, test_loader, device) -> dict:
    """THE decisive test. Returns gap metrics per VideoMAE-style neg-control."""
    probs_at_gt = []
    probs_at_noncontact = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            for p, src in zip(probs, batch["source"]):
                if src == "gt_positive":
                    probs_at_gt.append(p)
                else:
                    probs_at_noncontact.append(p)
    gt_ge_05 = (np.array(probs_at_gt) >= 0.5).mean()
    nc_ge_05 = (np.array(probs_at_noncontact) >= 0.5).mean()
    return {
        "gt_ge_05": gt_ge_05,
        "nc_ge_05": nc_ge_05,
        "absolute_gap": gt_ge_05 - nc_ge_05,
        "ship_gate_pass": (gt_ge_05 - nc_ge_05) >= 0.15,
    }
```

- [ ] **Step 4.2: Run training**

```bash
cd analysis && uv run python scripts/train_crop_head_probe.py 2>&1 | tee reports/crop_head_phase1_training.log
```
Expected wall-clock: 30-90 min on CPU (backbone is frozen, extraction cached). If GPU available, much faster.

- [ ] **Step 4.3: Read the report + check ship gates**

Read `analysis/reports/crop_head_phase1_probe_<date>.md`. Three gates:
- Test AUC ≥ 0.75 ?
- Orthogonality absolute gap ≥ 0.15 ?
- Hard-negative AUC ≥ 0.65 ?

Note which gates passed/failed. This is the input to the decision in Step 4.4.

- [ ] **Step 4.4: Decision branch — PASS or NO-GO**

**If ALL THREE gates pass**: commit the probe as "Phase 1 pass — Phase 2 scheduled" and STOP THIS PLAN. Phase 2 is a separate plan to be written in a subsequent session.

**If any gate fails**: write the NO-GO memo documenting which gate failed + what the failure mode tells us (e.g., indistinguishable distributions = architectural dead-end per VideoMAE). Commit as NO-GO.

In either case:

```bash
git add analysis/scripts/train_crop_head_probe.py analysis/reports/crop_head_phase1_probe_*.md analysis/reports/crop_head_phase1_training.log
git commit -m "$(cat <<'EOF'
ml(crop-head): Phase 1 probe — <PASS|NO-GO> against pre-registered gates

Test AUC: <value> (gate: ≥0.75)
Orthogonality gap: <value>pp (gate: ≥15pp)
Hard-neg AUC: <value> (gate: ≥0.65)

See reports/crop_head_phase1_probe_<date>.md.

<If PASS>: Phase 2 (architecture ablations) scheduled as separate plan.
<If NO-GO>: Architectural dead-end confirmed; redirect to [pose-event
classifier | GT attribution] per memory/crop_head_phase0_2026_04_20.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Verification summary

After Phase 1 complete:

1. `uv run pytest tests/unit -q` — all tests pass (target 239+ tests).
2. `uv run ruff check rallycut/` — clean.
3. `uv run mypy rallycut/` — clean.
4. `reports/crop_head_phase1_probe_<date>.md` exists with PASS/FAIL decision.
5. Production `detect_contacts` unchanged — baseline 88.0% F1 / 91.2% Action Acc preserved.

**Regression criterion:** Phase 1 is opt-in (a probe). Production code is untouched. If the probe fails, revert the probe script but keep the module files (they're scaffolding for any future attempt — Phase 2 may revive them).

---

## Honest scope caveats

- Phase 1 alone does NOT ship the crop head. It answers a binary question: does this architecture have signal?
- If Phase 1 PASSES, Phase 2 is still 2-3 days of ablation work (T window, input combinations, pooling). Phase 3 is 3-5 days of fine-tuning + LOO. Phase 4 is 2-3 days of cross-workstream validation. Full crop-head ship is 11-15 days TOTAL.
- If Phase 1 FAILS, redirect to dedicated pose-event classifier (2-3 days) or GT attribution repair (variable). Both are smaller, principled investments.
- This plan targets the 88 tagged cases (32 `looks_fixable` + 56 `classifier_boundary`). If the crop head only rescues ~30-50 TPs from those, we're adding ~1-2pp F1. The ship memo's 90-92% F1 projection was speculative; adjust expectations accordingly.
- Do NOT enter Phase 2 in the same session as Phase 1. Phase 1's output is a decision; act on the decision in a fresh session with full context.

## Self-Review

**Spec coverage:** Phase 1 has clear pass gates. Phase 0's feasibility results are honored (100% data availability means the extraction step is viable). Kill criteria derived from VideoMAE post-mortem (orthogonality + neg-control).

**Placeholder scan:** One placeholder in Task 4 Step 4.1 — the full training script is too long to inline in the plan; the design is fully specified (architecture, splits, loss, eval). Engineer implements from the spec. All other code is complete.

**Type consistency:** `FrozenResNet18` output is (B, 512); applied separately to player and ball inputs, concatenated to (B, T, 1024), consumed by `CropHeadMLP(d_in=1024)`. Dataset emits `player_crop` as (T, 3, 64, 64) float32 — loader permutes to match backbone input shape (B*T, 3, 64, 64). `source` field on samples is `str` with values `gt_positive` / `hard_negative` / `random_negative`, used consistently in training loop and orthogonality test.
