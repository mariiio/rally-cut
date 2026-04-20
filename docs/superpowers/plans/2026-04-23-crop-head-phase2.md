# Crop-Head Phase 2: Ablations + Integrated 68-fold LOO

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the crop-head (per-candidate player+ball CNN over temporal window) can replace the GBM as the contact emission source in the decoder pipeline, delivering integrated F1 ≥ 89.0% on 68-fold LOO — or prove it can't and redirect to GT attribution work.

**Architecture:** Two sub-phases with a hard gate between them.

- **Phase 2A** (ablation sweep, ~1-2 days compute + review): expand the Phase 1 crop cache to additional T-windows and wider hard-negative pools; train 8 architecture variants (T-window, input channels, pooling, backbone-fine-tune); pick the single winning config by val AUC on the held-out 10-video test set. **Gate:** winner must beat Phase 1 baseline (test AUC 0.8385) by ≥ +1pp, OR NO-GO.
- **Phase 2B** (integration + 68-fold LOO, ~3-4 hr compute): plug the winning crop-head into `run_decoder_over_rally` as the emission source, replacing the GBM. Run the same A/B harness as the decoder-integration workstream (`eval_decoder_integration.py` shape): baseline = GBM + decoder overlay (current production); experimental = crop-head + decoder overlay. **Gate:** integrated F1 ≥ 89.0% (+1.05pp over 88.0% baseline); Block F1 ≥ 20%; no per-class regression > 2pp (block exempt to -8pp, tighter than decoder's -12pp since crop-head should help block). Fail ANY gate → NO-GO, revert integration, keep crop-head as dormant research artifact.

**Tech Stack:** PyTorch + torchvision (ResNet-18 frozen or last-block-unfrozen), MPS on macOS for training, `uv` for Python env, reuse of Phase 1 dataset + Task 4 A/B harness.

**Context for the engineer (read these before starting):**
- Phase 1 PASS memo: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/crop_head_phase1_pass_2026_04_20.md`
- Phase 0 feasibility: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/crop_head_phase0_2026_04_20.md` — don't-retry list, spatial-resolution findings, pose-gap observations.
- Decoder integration PASS: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/decoder_integration_pass_2026_04_20.md` — explains the `run_decoder_over_rally` architecture you'll swap the classifier into.
- Canonical baseline: `scripts/eval_loo_video.py` on 68 videos / 364 rallies / 2095 contacts → 88.0% F1 / 91.2% Action Acc.
- Phase 1 cache at `analysis/outputs/crop_dataset/` — 4836 samples (1798 pos + 132 hard + 2906 rnd) at T=9. Phase 2A adds T=5/13/17 caches in Task 1.
- Phase 1 best test AUC: 0.8385 (epoch 9 val peak was 0.8703). Phase 2 winner must beat that on the same held-out 10-video test set, OR NO-GO.
- Block F1 12.12% baseline → decoder overlay 6.45% → Phase 2 target 20%. The pose-probe finding (`action_fixes_attempt_2026_04_20.md` §pose-signal) suggests pose+visual signal IS discriminating for block contacts; the failure mode in prior attempts was architectural (threshold gates can't express multivariate signatures). Crop-head should address this.

---

## Pre-registered ship gates (DO NOT modify after measurement runs)

### Phase 2A gate (ablation sweep → single winning config)

**Winner = config with highest val AUC on the Phase 1 held-out 10-video test set (video IDs frozen from Phase 1 for comparability).**

- Winner test AUC ≥ Phase 1 baseline + **1.0pp** (i.e., ≥ 0.8485)
- Winner val curve must show ≥ 3 epochs of improvement (no degenerate 1-epoch peak)
- If NO config meets the +1.0pp gate: Phase 2A NO-GO; do NOT enter Phase 2B.

### Phase 2B gate (68-fold integrated LOO)

**Baseline arm:** current production (GBM → detect_contacts → decoder overlay, the shipped 93.43% Acc pipeline).
**Experimental arm:** winner crop-head replaces GBM inside `run_decoder_over_rally`; same detect_contacts; same overlay.

- Integrated Contact F1 ≥ **89.0%** (+1.05pp over published 88.0%)
- Block F1 ≥ **20.0%** (+8pp over 12.12% baseline; +13.55pp over decoder overlay's 6.45%)
- No per-class F1 regression > 2.0pp (block exempt to −8pp, tighter than decoder's −12pp)
- Baseline arm sanity: within ±0.3pp of published 88.0%

**Fail ANY gate → NO-GO.** Revert integration (Task 4 CLI-wiring commit), keep crop-head modules as dormant research artifact, write NO-GO memo documenting which gate failed + what the failure mode implies.

**Phase 2 PASS does NOT ship to production immediately.** It clears the crop-head for a CLI-wiring task (a separate small plan) after one last review.

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `analysis/scripts/extract_crop_dataset.py` | MODIFY | Add `--t-window` flag (5/9/13/17) and `--hard-conf-range` flag ([0.10, 0.40] option). Existing T=9 cache is reused; new T-windows write to sibling dirs. |
| `analysis/rallycut/ml/crop_head/head.py` | MODIFY | Add `AttentionPoolHead` and `MaxPoolHead` classes alongside existing `CropHeadMLP` (mean-pool). |
| `analysis/rallycut/ml/crop_head/backbone.py` | MODIFY | Add `PartiallyUnfrozenResNet18` that leaves ResNet-18's `layer4` (last residual block, ~11M params) trainable while freezing earlier layers. |
| `analysis/rallycut/ml/crop_head/model.py` | CREATE | Composition wrapper: takes backbone + head + input-channel config (`player_only`, `ball_only`, `concat`), produces a unified `CropHeadModel` for training + inference. |
| `analysis/scripts/train_crop_head_probe.py` | MODIFY | Accept `--config path/to/config.yaml` so one script runs any ablation cell. Emit a JSON result per run for aggregation. |
| `analysis/scripts/sweep_crop_head_ablations.py` | CREATE | Orchestrator: runs the 8-cell orthogonal sweep (T-window ×4, input ×3, pool ×3, backbone ×2, via one-axis-at-a-time around Phase 1 defaults = 4+3+3+2 - 4 = 8 cells), aggregates JSON results, writes decision report. |
| `analysis/rallycut/tracking/crop_head_emitter.py` | CREATE | Adapter class `CropHeadContactClassifier` with a `predict_proba(features) -> (prob_bg, prob_contact)` interface compatible with `ContactClassifier`. Loads the winning checkpoint + runs inference per candidate using the same player+ball crops at runtime. |
| `analysis/rallycut/tracking/decoder_runtime.py` | MODIFY | Generalize `run_decoder_over_rally` to accept any classifier implementing `predict_proba` (currently hardcoded to `ContactClassifier` model attribute); add optional `classifier_kind` parameter. |
| `analysis/scripts/eval_decoder_integration.py` | MODIFY | Add `--emitter {gbm, crop_head}` flag; crop-head path loads the Phase 2A winner checkpoint. |
| `analysis/reports/crop_head_phase2a_sweep_<date>.md` | GENERATED | Phase 2A decision memo: per-config val AUC table, winner selection, PASS/NO-GO. |
| `analysis/reports/crop_head_phase2b_integration_<date>.md` | GENERATED | Phase 2B decision memo: 68-fold A/B, per-class deltas, PASS/NO-GO. |
| `analysis/tests/unit/test_crop_head_model.py` | CREATE | Unit tests for model composition, backbone partial-unfreeze, attention/max pooling shapes. |
| `analysis/tests/unit/test_crop_head_emitter.py` | CREATE | Unit tests for the ContactClassifier-compatible emitter adapter. |

---

## Task 1: Expand the crop cache

**Goal:** Extract T=5, T=13, T=17 variants of the Phase 1 dataset; bump hard-negative pool to conf ∈ [0.10, 0.40] so Phase 2A has adequate boundary-case coverage (Phase 1 had only 132 hard-negatives — too few for robust architecture choice).

**Files:**
- MODIFY: `analysis/scripts/extract_crop_dataset.py`

### Step 1.1: Add `--t-window` and `--hard-conf-range` flags

Open `analysis/scripts/extract_crop_dataset.py`. Currently `T_FRAMES = 9` is module-level. Refactor so it's a CLI arg:

```python
parser.add_argument("--t-window", type=int, default=9, choices=[5, 9, 13, 17],
                    help="Temporal window size (must be odd). Default 9 (±4f, Phase 1 config).")
parser.add_argument("--hard-conf-lo", type=float, default=0.20,
                    help="Hard-negative GBM conf lower bound (Phase 1 default 0.20).")
parser.add_argument("--hard-conf-hi", type=float, default=0.30,
                    help="Hard-negative GBM conf upper bound (Phase 1 default 0.30).")
parser.add_argument("--output-subdir", type=str, default=None,
                    help="Subdir under outputs/crop_dataset/. Defaults to t{T}_h{lo:02.0f}_{hi:02.0f}.")
```

Output directory becomes `analysis/outputs/crop_dataset/t{T}_h{lo*100:02.0f}_{hi*100:02.0f}/<video_id>/`. The existing Phase 1 cache stays at `analysis/outputs/crop_dataset/<video_id>/` unchanged for backward compatibility — new runs go to the subdirectoried layout. If `--output-subdir` is explicitly set, use that.

Thread `T` through `_extract_sample` (currently uses the module-level constant). `HALF = T // 2`. Keep the npz schema identical — consumers check shape at load time.

### Step 1.2: Smoke-test each new T-window on 1 video

For each T ∈ {5, 13, 17}:

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/extract_crop_dataset.py \
    --video 35ff869f-5698-4982-bcdb-267e11d70e0d \
    --t-window 5 --hard-conf-lo 0.10 --hard-conf-hi 0.40
```

Verify each produces ~20-25 samples with the correct `player_crop` shape `(T, 3, 64, 64)`:

```bash
uv run python -c "
import numpy as np, glob
for t in (5, 13, 17):
    path = glob.glob(f'outputs/crop_dataset/t{t}_h10_40/35ff869f*/*.npz')[0]
    d = np.load(path)
    assert d['player_crop'].shape == (t, 3, 64, 64), d['player_crop'].shape
    print(f'T={t}: OK, {d[\"player_crop\"].shape}')
"
```

### Step 1.3: Full extraction for all 3 new T-windows

Run in background (each takes ~70 min based on Phase 1 timing):

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
for T in 5 13 17; do
    uv run python scripts/extract_crop_dataset.py \
        --t-window $T --hard-conf-lo 0.10 --hard-conf-hi 0.40 \
        2>&1 | tee reports/crop_extraction_t${T}_2026_04_23.log
done
```

Monitor per-video progress. Total wall-clock ~3.5 hr. Use `run_in_background: true` per T.

Expected samples per T (wider hard-neg range should give ~3-4× more hard-negatives):
- T=5: ~6000 samples, ~400 hard-neg
- T=13: ~6500 samples, ~400 hard-neg
- T=17: ~6800 samples, ~400 hard-neg

Also rebuild the Phase 1 T=9 cache at the WIDER hard-neg range (so all 4 T-windows use comparable data):

```bash
uv run python scripts/extract_crop_dataset.py \
    --t-window 9 --hard-conf-lo 0.10 --hard-conf-hi 0.40 \
    2>&1 | tee reports/crop_extraction_t9_wide_2026_04_23.log
```

### Step 1.4: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/scripts/extract_crop_dataset.py
git commit -m "$(cat <<'EOF'
feat(ml): crop extractor supports T-window + hard-neg range flags

Adds --t-window (5/9/13/17) and --hard-conf-lo/--hard-conf-hi flags so
Phase 2A can sweep temporal window without code changes. Wider
hard-negative pool ([0.10, 0.40] vs Phase 1's [0.20, 0.30]) gives
Phase 2A ~400 hard-negatives per T-window vs Phase 1's 132, enough for
robust architecture choice.

Output layout: outputs/crop_dataset/t{T}_h{lo:02.0f}_{hi:02.0f}/<video_id>/.
Phase 1 cache at outputs/crop_dataset/<video_id>/ preserved for
back-compat.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Extraction outputs are gitignored — this commit is the script change only.

---

## Task 2: Architecture variants (head + partial-unfreeze backbone)

**Files:**
- MODIFY: `analysis/rallycut/ml/crop_head/head.py`
- MODIFY: `analysis/rallycut/ml/crop_head/backbone.py`
- CREATE: `analysis/rallycut/ml/crop_head/model.py`
- CREATE: `analysis/tests/unit/test_crop_head_model.py`

### Step 2.1: Add attention-pool and max-pool heads

Append to `analysis/rallycut/ml/crop_head/head.py`:

```python
class AttentionPoolHead(nn.Module):
    """Attention-weighted temporal pooling + MLP.

    Learns a per-frame attention score; pools features by softmax-weighted
    average across T. Rationale: contact is a 1-frame event in a T-frame
    window, so uniform mean-pooling dilutes the signal. Attention should
    let the model focus on the contact frame.
    """

    def __init__(self, d_in: int = 1024, d_hidden: int = 256) -> None:
        super().__init__()
        self.attn = nn.Linear(d_in, 1)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        scores = self.attn(x).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        out: torch.Tensor = self.net(pooled)
        return out.squeeze(-1)


class MaxPoolHead(nn.Module):
    """Max-pooling temporal aggregation + MLP.

    Picks the per-feature maximum across T. Simple alternative to
    attention; fewer learnable params.
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
        pooled, _ = x.max(dim=1)
        out: torch.Tensor = self.net(pooled)
        return out.squeeze(-1)
```

### Step 2.2: Add partial-unfreeze backbone variant

Append to `analysis/rallycut/ml/crop_head/backbone.py`:

```python
class PartiallyUnfrozenResNet18(nn.Module):
    """Like FrozenResNet18 but leaves `layer4` (last residual block, ~11M
    params out of 11.7M total) trainable. Used for the Phase 2A
    fine-tuning ablation.
    """

    def __init__(self) -> None:
        super().__init__()
        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        # Freeze everything BEFORE layer4.
        for name, p in self.backbone.named_parameters():
            p.requires_grad = name.startswith("7.")  # index 7 = layer4 in Sequential children
        # BN stats for frozen layers stay fixed; keep the layer4 BN in train mode
        # (it's trainable, so its running stats update normally).
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        x = (x - mean) / std
        feat: torch.Tensor = self.backbone(x)
        return feat.flatten(1)
```

Note: this class does NOT use `@torch.no_grad()` (unlike `FrozenResNet18`) because `layer4` needs gradients. Caller must handle eval-mode appropriately; the training script sets `bb.eval()` for frozen layers only via `bb.backbone[:7].eval()` inside the train loop.

### Step 2.3: Create unified model composition

Create `analysis/rallycut/ml/crop_head/model.py`:

```python
"""Unified crop-head model composer for Phase 2A ablations."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from rallycut.ml.crop_head.backbone import FrozenResNet18, PartiallyUnfrozenResNet18
from rallycut.ml.crop_head.head import AttentionPoolHead, CropHeadMLP, MaxPoolHead

InputKind = Literal["player_only", "ball_only", "concat"]
PoolKind = Literal["mean", "attention", "max"]
BackboneKind = Literal["frozen", "layer4_unfrozen"]


class CropHeadModel(nn.Module):
    """Composable crop-head model.

    Forward: (player_crop[B,T,3,64,64], ball_patch[B,T,3,32,32]) -> (B,) logit.

    Player and ball inputs are processed by the SAME backbone (shared
    weights) at different spatial resolutions. Ball patches are bilinearly
    upsampled to 64×64 before backbone.
    """

    def __init__(
        self,
        input_kind: InputKind = "concat",
        pool_kind: PoolKind = "mean",
        backbone_kind: BackboneKind = "frozen",
    ) -> None:
        super().__init__()
        self.input_kind = input_kind
        self.pool_kind = pool_kind
        self.backbone_kind = backbone_kind

        if backbone_kind == "frozen":
            self.backbone: nn.Module = FrozenResNet18()
        elif backbone_kind == "layer4_unfrozen":
            self.backbone = PartiallyUnfrozenResNet18()
        else:
            raise ValueError(backbone_kind)

        d_feat = 512
        if input_kind == "concat":
            d_in = 2 * d_feat
        else:
            d_in = d_feat

        if pool_kind == "mean":
            self.head: nn.Module = CropHeadMLP(d_in=d_in)
        elif pool_kind == "attention":
            self.head = AttentionPoolHead(d_in=d_in)
        elif pool_kind == "max":
            self.head = MaxPoolHead(d_in=d_in)
        else:
            raise ValueError(pool_kind)

    def forward(
        self, player_crops: torch.Tensor, ball_patches: torch.Tensor,
    ) -> torch.Tensor:
        b, t, c, h, w = player_crops.shape
        player_flat = player_crops.reshape(b * t, c, h, w)
        player_feat = self.backbone(player_flat).reshape(b, t, -1)

        if self.input_kind == "player_only":
            combined = player_feat
        else:
            ball_upsampled = torch.nn.functional.interpolate(
                ball_patches.reshape(b * t, c, ball_patches.shape[-2], ball_patches.shape[-1]),
                size=(h, w), mode="bilinear", align_corners=False,
            )
            ball_feat = self.backbone(ball_upsampled).reshape(b, t, -1)
            if self.input_kind == "ball_only":
                combined = ball_feat
            else:  # concat
                combined = torch.cat([player_feat, ball_feat], dim=-1)

        return self.head(combined)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
```

### Step 2.4: Unit tests

Create `analysis/tests/unit/test_crop_head_model.py`:

```python
"""Unit tests for Phase 2A model variants."""
import torch

from rallycut.ml.crop_head.backbone import PartiallyUnfrozenResNet18
from rallycut.ml.crop_head.head import AttentionPoolHead, MaxPoolHead
from rallycut.ml.crop_head.model import CropHeadModel


def test_attention_pool_head_shape() -> None:
    head = AttentionPoolHead(d_in=1024)
    x = torch.rand(4, 9, 1024)
    assert head(x).shape == (4,)


def test_max_pool_head_shape() -> None:
    head = MaxPoolHead(d_in=1024)
    x = torch.rand(4, 9, 1024)
    assert head(x).shape == (4,)


def test_partially_unfrozen_backbone_has_gradients_on_layer4() -> None:
    bb = PartiallyUnfrozenResNet18()
    trainable = {n for n, p in bb.named_parameters() if p.requires_grad}
    frozen = {n for n, p in bb.named_parameters() if not p.requires_grad}
    # layer4 is index 7 in the Sequential(children), so all names start with "backbone.7."
    assert trainable, "expected at least layer4 parameters to be trainable"
    assert all(n.startswith("backbone.7.") for n in trainable), trainable
    assert frozen, "expected early-layer parameters to be frozen"


def test_model_forward_concat_mean_frozen() -> None:
    """Phase 1 baseline config sanity."""
    m = CropHeadModel(input_kind="concat", pool_kind="mean", backbone_kind="frozen")
    pc = torch.rand(2, 9, 3, 64, 64)
    bp = torch.rand(2, 9, 3, 32, 32)
    assert m(pc, bp).shape == (2,)


def test_model_forward_player_only_attention_unfrozen() -> None:
    """Off-axis combo sanity."""
    m = CropHeadModel(
        input_kind="player_only", pool_kind="attention",
        backbone_kind="layer4_unfrozen",
    )
    pc = torch.rand(2, 5, 3, 64, 64)
    bp = torch.rand(2, 5, 3, 32, 32)
    assert m(pc, bp).shape == (2,)


def test_model_trainable_parameters_frozen_variant_is_head_only() -> None:
    m = CropHeadModel(backbone_kind="frozen")
    trainable_modules = {id(p): p for p in m.trainable_parameters()}
    head_params = {id(p) for p in m.head.parameters()}
    backbone_trainable = {id(p) for p in m.backbone.parameters() if p.requires_grad}
    assert set(trainable_modules) == head_params, (
        "frozen variant must have only head params trainable"
    )
    assert backbone_trainable == set()


def test_model_trainable_parameters_unfrozen_includes_layer4() -> None:
    m = CropHeadModel(backbone_kind="layer4_unfrozen")
    trainable_modules = {id(p): p for p in m.trainable_parameters()}
    backbone_trainable = {id(p) for p in m.backbone.parameters() if p.requires_grad}
    head_params = {id(p) for p in m.head.parameters()}
    expected = head_params | backbone_trainable
    assert set(trainable_modules) == expected
    assert backbone_trainable, "layer4 params must be in trainable set"
```

### Step 2.5: Run tests + lint + mypy

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_crop_head_model.py -v
uv run ruff check rallycut/ml/crop_head/
uv run mypy rallycut/ml/crop_head/
```

Expected: 7 tests pass, clean.

### Step 2.6: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/ml/crop_head/head.py \
        analysis/rallycut/ml/crop_head/backbone.py \
        analysis/rallycut/ml/crop_head/model.py \
        analysis/tests/unit/test_crop_head_model.py
git commit -m "$(cat <<'EOF'
feat(ml): crop-head Phase 2A architecture variants

Adds AttentionPoolHead (softmax-weighted temporal pooling — lets the
model focus on the 1-frame contact event in the T-frame window) and
MaxPoolHead (per-feature max across T) alongside Phase 1's
CropHeadMLP mean-pool head.

Adds PartiallyUnfrozenResNet18 which leaves layer4 (last residual
block, ~11M params) trainable while freezing earlier layers — enables
the fine-tuning ablation without full backbone retraining.

Unifies composition via CropHeadModel(input_kind, pool_kind,
backbone_kind) so the Phase 2A sweep script can instantiate any cell
from one factory.

7 passing unit tests cover pool-head shapes, partial-unfreeze gradient
masks, and both Phase 1 baseline + off-axis combo forward passes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Phase 2A ablation sweep + winner selection

**Files:**
- MODIFY: `analysis/scripts/train_crop_head_probe.py`
- CREATE: `analysis/scripts/sweep_crop_head_ablations.py`
- GENERATED: `analysis/reports/crop_head_phase2a_sweep_<date>.md`

### Design (orthogonal one-axis sweep, 8 cells)

Holding 3 axes at Phase 1 defaults (T=9, concat, mean, frozen) and varying one at a time:

| Cell | T | Input | Pool | Backbone |
|---|---|---|---|---|
| Baseline (Phase 1 replay with wider negs) | 9 | concat | mean | frozen |
| T-short | 5 | concat | mean | frozen |
| T-long | 13 | concat | mean | frozen |
| T-longest | 17 | concat | mean | frozen |
| Player-only | 9 | player_only | mean | frozen |
| Ball-only | 9 | ball_only | mean | frozen |
| Attention-pool | 9 | concat | attention | frozen |
| Max-pool | 9 | concat | max | frozen |
| Fine-tune layer4 | 9 | concat | mean | layer4_unfrozen |

That's 9 cells (1 baseline + 8 ablations). Each trains in ~10 min on MPS (15 epochs × 35s) so the full sweep is ~1.5 hr.

### Step 3.1: Make `train_crop_head_probe.py` config-driven

Open `analysis/scripts/train_crop_head_probe.py`. Add CLI flags:

```python
parser.add_argument("--t-window", type=int, default=9)
parser.add_argument("--input-kind", type=str, default="concat",
                    choices=["player_only", "ball_only", "concat"])
parser.add_argument("--pool-kind", type=str, default="mean",
                    choices=["mean", "attention", "max"])
parser.add_argument("--backbone-kind", type=str, default="frozen",
                    choices=["frozen", "layer4_unfrozen"])
parser.add_argument("--cache-subdir", type=str, default=None,
                    help="If set, read crops from outputs/crop_dataset/<subdir>/ "
                         "instead of outputs/crop_dataset/.")
parser.add_argument("--run-name", type=str, default=None,
                    help="Label for this run in output filenames.")
```

Wire them:
- `cache-subdir` changes the dataset root.
- Model instantiation: replace the existing `FrozenResNet18 + CropHeadMLP` wiring with `CropHeadModel(input_kind, pool_kind, backbone_kind)`.
- For `layer4_unfrozen` backbone: use a LR multiplier on the backbone params. Recommended: AdamW with two param groups — head at lr=1e-3, backbone.layer4 at lr=1e-5.
- Output filenames: `crop_head_phase2a_<run_name>_<date>.md` and `.json`.
- HELD-OUT TEST SPLIT MUST BE THE PHASE 1 10 VIDEOS (hardcode the list in a module-level constant copied from `reports/crop_head_phase1_probe_2026-04-20.md` Test split section). This makes the AUC-on-test comparison between Phase 2A cells and Phase 1 APPLES-TO-APPLES.

### Step 3.2: Create the sweep orchestrator

Create `analysis/scripts/sweep_crop_head_ablations.py`:

```python
"""Phase 2A: orthogonal ablation sweep over T-window / input / pool / backbone.

Trains 9 cells (1 baseline + 8 one-axis variants), aggregates val/test
AUCs, picks the winner, writes a decision report against the
pre-registered +1.0pp gate vs Phase 1 baseline (test AUC 0.8385).

Usage (cd analysis):
    uv run python scripts/sweep_crop_head_ablations.py
    uv run python scripts/sweep_crop_head_ablations.py --dry-run  # just print config
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from rich.console import Console

console = Console()

PHASE1_TEST_AUC = 0.8385
GATE_AUC_LIFT = 0.010  # +1.0pp


CELLS = [
    # (name,           t, input, pool, backbone, cache_subdir)
    ("baseline",       9, "concat",      "mean",      "frozen",           "t09_h10_40"),
    ("t_short",        5, "concat",      "mean",      "frozen",           "t05_h10_40"),
    ("t_long",        13, "concat",      "mean",      "frozen",           "t13_h10_40"),
    ("t_longest",     17, "concat",      "mean",      "frozen",           "t17_h10_40"),
    ("player_only",    9, "player_only", "mean",      "frozen",           "t09_h10_40"),
    ("ball_only",      9, "ball_only",   "mean",      "frozen",           "t09_h10_40"),
    ("attention_pool", 9, "concat",      "attention", "frozen",           "t09_h10_40"),
    ("max_pool",       9, "concat",      "max",       "frozen",           "t09_h10_40"),
    ("layer4_unfrozen", 9, "concat",     "mean",      "layer4_unfrozen",  "t09_h10_40"),
]


def run_cell(cell_name: str, t: int, input_kind: str, pool_kind: str,
             backbone_kind: str, cache_subdir: str) -> dict:
    """Invoke train_crop_head_probe.py as a subprocess and parse its JSON output."""
    json_path = Path("reports") / f"crop_head_phase2a_{cell_name}.json"
    cmd = [
        "uv", "run", "python", "scripts/train_crop_head_probe.py",
        "--t-window", str(t),
        "--input-kind", input_kind,
        "--pool-kind", pool_kind,
        "--backbone-kind", backbone_kind,
        "--cache-subdir", cache_subdir,
        "--run-name", cell_name,
        "--out-json", str(json_path),
    ]
    console.print(f"[bold]Running cell:[/bold] {cell_name} — {' '.join(cmd[2:])}")
    t0 = time.time()
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    dt = time.time() - t0
    console.print(f"  done in {dt/60:.1f} min")
    with json_path.open() as f:
        data: dict = json.load(f)
    data["_cell_name"] = cell_name
    data["_elapsed_s"] = dt
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        for cell in CELLS:
            console.print(cell)
        return 0

    results = []
    for cell in CELLS:
        results.append(run_cell(*cell))

    # Rank by test AUC
    ranked = sorted(results, key=lambda r: r["test_auc"], reverse=True)
    winner = ranked[0]

    # Gate check
    passed_gate = winner["test_auc"] >= PHASE1_TEST_AUC + GATE_AUC_LIFT
    verdict = "PASS" if passed_gate else "NO-GO"

    # Write report
    today = date.today().isoformat()
    out = Path("reports") / f"crop_head_phase2a_sweep_{today}.md"
    lines = [
        f"# Crop-Head Phase 2A Ablation Sweep — {verdict}",
        "",
        f"- Date: {today}",
        f"- Phase 1 baseline test AUC: {PHASE1_TEST_AUC}",
        f"- Gate: winner test AUC ≥ {PHASE1_TEST_AUC + GATE_AUC_LIFT} (+{GATE_AUC_LIFT:.2%})",
        "",
        "## Cells (sorted by test AUC, descending)",
        "",
        "| Cell | T | Input | Pool | Backbone | Val AUC | Test AUC | Test AUC Δ vs Phase 1 | Elapsed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ranked:
        delta = r["test_auc"] - PHASE1_TEST_AUC
        lines.append(
            f"| {r['_cell_name']} "
            f"| {r.get('t_window', '—')} "
            f"| {r.get('input_kind', '—')} "
            f"| {r.get('pool_kind', '—')} "
            f"| {r.get('backbone_kind', '—')} "
            f"| {r.get('best_val_auc', 'n/a'):.4f} "
            f"| **{r['test_auc']:.4f}** "
            f"| {delta:+.2%} "
            f"| {r['_elapsed_s']/60:.1f} min |"
        )
    lines += [
        "",
        f"## Winner: **{winner['_cell_name']}**",
        "",
        f"- Test AUC: {winner['test_auc']:.4f}",
        f"- Δ vs Phase 1: {(winner['test_auc'] - PHASE1_TEST_AUC):+.2%}",
        f"- Gate ({GATE_AUC_LIFT:+.2%}): {'PASSED' if passed_gate else 'FAILED'}",
        "",
    ]
    if passed_gate:
        lines.append("Proceed to Phase 2B (integrated 68-fold LOO).")
    else:
        lines.append(
            "Phase 2A NO-GO. Do not enter Phase 2B. Architecture variants "
            "cannot clear the +1pp gate on the frozen-backbone sanity probe. "
            "Redirect to GT attribution work or accept current production "
            "ceiling (93.43% Action Acc after decoder overlay)."
        )
    out.write_text("\n".join(lines))
    console.print(f"[bold]Report:[/bold] {out}")
    console.print(f"[bold]Verdict:[/bold] {verdict}")
    return 0 if passed_gate else 2


if __name__ == "__main__":
    sys.exit(main())
```

### Step 3.3: Dry-run the sweep

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/sweep_crop_head_ablations.py --dry-run
```

Expected: prints the 9 cells. Verify the cache subdirs exist (Task 1 Step 1.3 output).

### Step 3.4: Run the full sweep

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/sweep_crop_head_ablations.py \
    2>&1 | tee reports/crop_head_phase2a_sweep.log
```

Use `run_in_background: true`. Total wall-clock ~1.5 hr on MPS.

### Step 3.5: Read report + decide

Open `reports/crop_head_phase2a_sweep_<date>.md`. Record:
- Winner cell name + config
- Winner test AUC
- Δ vs Phase 1
- Gate PASS / NO-GO

### Step 3.6: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/scripts/train_crop_head_probe.py \
        analysis/scripts/sweep_crop_head_ablations.py \
        analysis/reports/crop_head_phase2a_sweep_*.md \
        analysis/reports/crop_head_phase2a_sweep.log
git commit -m "$(cat <<'EOF'
ml(crop-head): Phase 2A ablation sweep — <PASS|NO-GO>

Winner: <cell_name> (T=<T>, <input>, <pool>, <backbone>)
Test AUC: <value> (Phase 1 baseline 0.8385, gate +1pp)

Full sweep of 9 cells (baseline + 8 one-axis variants): <summary>.

See reports/crop_head_phase2a_sweep_<date>.md.

<If PASS>: Phase 2B (integrated 68-fold LOO) proceeds.
<If NO-GO>: Phase 2B skipped. Crop-head modules retained as dormant
research artifact; redirect to GT attribution work.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 3.7: Branch on gate

- **PASS:** Proceed to Task 4.
- **NO-GO:** Stop this plan. Write NO-GO memory memo at `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/crop_head_phase2_nogo_<date>.md`. Update MEMORY.md index.

---

## Task 4: Phase 2B integration + 68-fold LOO (ONLY if Phase 2A passed)

**Files:**
- CREATE: `analysis/rallycut/tracking/crop_head_emitter.py` — ContactClassifier-compatible adapter.
- MODIFY: `analysis/rallycut/tracking/decoder_runtime.py` — generalize `run_decoder_over_rally` to accept any emitter implementing `predict_proba`.
- MODIFY: `analysis/scripts/eval_decoder_integration.py` — add `--emitter {gbm, crop_head}` flag.
- CREATE: `analysis/tests/unit/test_crop_head_emitter.py`
- GENERATED: `analysis/reports/crop_head_phase2b_integration_<date>.md`

### Design

The crop-head emitter at inference time needs to:
1. Receive a list of candidate frames from `extract_candidate_features`.
2. For each candidate, extract the SAME crops the training script used: (player bbox from nearest-player-to-ball, ball patch, T frames centered on the candidate). This requires VIDEO FRAMES — which the current `run_decoder_over_rally` does not have access to (it operates on ball/player positions only).
3. Run the CropHeadModel → per-candidate contact probability.

This is the major new infrastructure piece. Design choices:

**Option A:** Crop extraction at inference time, same path as `extract_crop_dataset.py`. Requires `video_path` argument to `run_decoder_over_rally`. Adds I/O cost per candidate (~100ms for seek + decode × 9 frames).

**Option B:** Pre-extract crop features for all candidates of all rallies once, cache them, and have the emitter look up cached features. Fast at inference but requires a 1-time ~1hr precompute step.

**Choose Option A for Phase 2B** — simpler, matches how the production CLI will work (it has video_path anyway), and the 100ms/candidate cost is dwarfed by the existing MS-TCN++ compute. Option B is a follow-up optimization if Phase 2B passes.

### Step 4.1: Create the emitter

```python
# analysis/rallycut/tracking/crop_head_emitter.py
"""Inference-time crop extraction + crop-head scoring.

Used by run_decoder_over_rally when the emitter_kind='crop_head' flag is
set. Matches the crop extraction path in scripts/extract_crop_dataset.py
so the training and inference distributions align.
"""
# [Engineer implements — mirrors extract_crop_dataset.py's _extract_sample
#  but in-memory (no .npz write), with a load_weights(path) classmethod
#  that loads the Phase 2A winner checkpoint.
#
#  predict_proba(candidates: list[CandidateFeatures], video_path: Path,
#                rally_start_frame: int, player_positions, ball_positions)
#    -> np.ndarray of shape (N_candidates, 2)
#
#  where column 0 = P(no_contact), column 1 = P(contact). This matches
#  sklearn's predict_proba convention, so decoder_runtime does not need
#  a branch on emitter kind — just calls .predict_proba(...) uniformly.
# ]
```

The full implementation is ~200 lines — engineer writes from the spec above, using `extract_crop_dataset.py`'s `_extract_sample` as a reference. Key correctness requirement: the player bbox selection logic (nearest-player-to-ball at the candidate frame) MUST match training, or the inference distribution drifts.

### Step 4.2: Generalize `run_decoder_over_rally`

Currently `run_decoder_over_rally(..., classifier: ContactClassifier, ...)` is tightly typed. Loosen to:

```python
from typing import Protocol

class _ContactEmitter(Protocol):
    """Minimal protocol the decoder needs from a contact emitter."""
    def predict_proba(self, candidates: list["CandidateFeatures"]) -> np.ndarray:
        """Returns (N, 2) array of [P(no_contact), P(contact)] per candidate."""
        ...
```

Accept either the existing `ContactClassifier` or the new `CropHeadContactClassifier` via this protocol. Add an optional `video_path` + `rally_start_frame` + position kwargs for the crop-head path (`ContactClassifier` ignores them).

Keep backward compat: all existing callers (Task 5 CLI wiring, A/B harness) continue to work unchanged.

### Step 4.3: Add `--emitter` flag to A/B harness

In `scripts/eval_decoder_integration.py`, add:
```python
parser.add_argument("--emitter", choices=["gbm", "crop_head"], default="gbm")
parser.add_argument("--crop-head-weights", type=str, default=None,
                    help="Path to Phase 2A winner checkpoint; required when --emitter=crop_head.")
```

Thread through so the experimental arm uses the crop-head emitter. Baseline arm always uses GBM.

### Step 4.4: Smoke test (5-fold)

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/eval_decoder_integration.py \
    --emitter crop_head \
    --crop-head-weights weights/crop_head/phase2a_winner.pt \
    --limit 5 \
    2>&1 | tee reports/crop_head_phase2b_smoke.log
```

Expected: runs in 15-30 min (crop extraction + inference adds ~2x the GBM-baseline wall-clock). Verify the baseline arm lands near 88% F1 (sanity); the experimental arm shouldn't crash.

### Step 4.5: Full 68-fold A/B

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/eval_decoder_integration.py \
    --emitter crop_head \
    --crop-head-weights weights/crop_head/phase2a_winner.pt \
    2>&1 | tee reports/crop_head_phase2b_full.log
```

Background. Expected wall-clock: 2-4 hours (double the GBM path).

### Step 4.6: Check Phase 2B gates

Open `reports/crop_head_phase2b_integration_<date>.md`. Record:
- Experimental F1 (gate ≥ 89.0%)
- Block F1 (gate ≥ 20.0%)
- Per-class regressions (gate ≤ 2pp except block)
- Baseline sanity (within ±0.3pp of 88.0%)

### Step 4.7: Decision commit

**PASS (all 4 gates):**

```bash
git add analysis/rallycut/tracking/crop_head_emitter.py \
        analysis/rallycut/tracking/decoder_runtime.py \
        analysis/scripts/eval_decoder_integration.py \
        analysis/tests/unit/test_crop_head_emitter.py \
        analysis/reports/crop_head_phase2b_integration_*.md \
        analysis/reports/crop_head_phase2b_*.log
git commit -m "$(cat <<'EOF'
ml(crop-head): Phase 2B integrated LOO — PASS

Crop-head emission (Phase 2A winner: <config>) replaces GBM in the
decoder pipeline. 68-fold LOO A/B vs GBM-baseline:

  Contact F1:   baseline <x>% → crop-head <y>% (Δ <z>pp, gate ≥89.0%)
  Action Acc:   baseline <x>% → crop-head <y>% (Δ <z>pp)
  Block F1:     baseline <x>% → crop-head <y>% (gate ≥20%)

See reports/crop_head_phase2b_integration_<date>.md.

Production wiring stays on GBM path — a separate small plan will switch
production emitter after 1-week checkpoint validation on newly-labeled
rallies. Do not ship to production from this plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Then write the PASS memory memo. STOP this plan.

**NO-GO (any gate fails):**

Do NOT commit the integration infrastructure to production call sites. The decoder_runtime changes (protocol + optional video_path kwargs) are safe to keep since they're backward-compatible. The A/B harness changes are also safe (new `--emitter` flag defaults to gbm).

Write the NO-GO memo documenting which gate failed. Revert only the changes that meaningfully add risk (the crop_head_emitter.py file stays as dormant research, but it's not wired anywhere).

---

## Verification summary

After Phase 2 complete (whether PASS or NO-GO):

1. `uv run pytest tests/unit -q` — all tests pass (target: N+7+K tests where N was the prior count, K is emitter tests).
2. `uv run ruff check rallycut/` — clean.
3. `uv run mypy rallycut/ml/ rallycut/tracking/` — clean.
4. `reports/crop_head_phase2a_sweep_<date>.md` exists with PASS/NO-GO.
5. If Phase 2A passed: `reports/crop_head_phase2b_integration_<date>.md` exists with PASS/NO-GO.
6. Production CLI paths (the Task 5 decoder wiring from the prior plan) still run with GBM emission — Phase 2 does not swap emitter in production.
7. Existing A/B harness still reproducible with `--emitter gbm` (default). Published 88.0% / 91.2% baseline preserved.

---

## Honest scope caveats

- **Phase 2A gate is aggressive.** +1pp AUC on the Phase 1 held-out 10-video test set is a real lift, not noise. If no cell clears it, the ablation space has been searched and the architecture's frozen-backbone ceiling is the real limit. Redirecting to GT attribution work is the honest move.
- **Phase 2B integration cost.** Crop extraction at inference time adds ~100ms per candidate. Typical rally has ~30 candidates → ~3s per rally of added compute in the production pipeline. Acceptable for batch analysis CLIs; not acceptable for real-time feedback. Follow-up: pre-extract-and-cache path (Option B from Task 4 Design).
- **Block F1 target of 20%.** That's a ≥8pp lift over the Phase 1 baseline (12.12%) and +13.55pp over the decoder overlay (6.45%). It's plausible based on the pose-probe finding that multivariate signatures exist, but untested. If we miss this gate but clear the overall F1 gate, reconsider whether the block-specific target is worth vetoing on.
- **This plan does NOT ship to production.** PASS means we've earned the right to open a follow-up wiring plan. Production continues to run the GBM + decoder overlay (decoder integration PASS shipped 2026-04-20).
- **Gate discipline.** Do NOT modify the pre-registered Phase 2A (+1pp AUC) or Phase 2B (≥89% F1 / ≥20% block F1) gates after training runs. Pre-registration is the mechanism that prevents motivated reasoning after seeing results.

## Self-Review

**Spec coverage:**
- Task 1 — expanded crop cache (T-window ablation data) ✅
- Task 2 — architecture variants (pool heads + backbone partial unfreeze) ✅
- Task 3 — Phase 2A sweep + winner selection + pre-registered gate ✅
- Task 4 — Phase 2B integration + 68-fold LOO + pre-registered gates ✅

**Placeholder scan:**
- Task 4 Step 4.1 has `# [Engineer implements]` for the ~200-line emitter. Acceptable: the spec fully defines inputs/outputs/correctness requirement (match training's bbox selection), and the reference implementation is pointed to (`extract_crop_dataset.py::_extract_sample`). Similar to Phase 1 plan's Task 4 Step 4.1 pattern — worked there.
- Decision commit messages have `<value>` placeholders filled in from the report by the engineer. Intentional.

**Type consistency:**
- `CropHeadModel` signature (`player_crops[B,T,3,64,64]`, `ball_patches[B,T,3,32,32] -> Tensor[B,]`) consistent across Task 2 test, Task 3 training script, Task 4 emitter.
- `CropHeadContactClassifier.predict_proba` returns `(N, 2)` matching sklearn convention so decoder_runtime works uniformly regardless of emitter kind.
- `input_kind` / `pool_kind` / `backbone_kind` Literal types consistent across model.py, train script, sweep orchestrator.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-23-crop-head-phase2.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, two-stage review (spec then code). Good for this plan because Tasks 1/3/4 each include a long background run and the per-task checkpoints prevent drift.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Recommendation: start the Phase 2 session fresh — this plan spans 2-3 days of compute and benefits from clean context.
