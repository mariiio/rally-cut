# Crop-Head Phase 2: Kill-Test → Ablations → Integrated 68-fold LOO

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the crop-head (per-candidate player+ball CNN over temporal window) can replace the GBM as the contact emission source in the decoder pipeline, delivering integrated F1 ≥ 89.0% on 68-fold LOO — or prove it can't, cheaply, and redirect to GT attribution work.

**Architecture:** Three sub-phases with hard gates between them, designed to kill cheaply if the premise is wrong.

- **Phase 2.0** (kill-test spike, ~3 hr compute): plug the EXISTING Phase 1 checkpoint (no retraining) into the A/B harness as a crop-head emitter. **Gate:** integrated F1 delta ≥ +0.3pp OR Block F1 delta ≥ +3pp. This proves the architecture-swap is worth investing in BEFORE we spend 2-3 days on ablations. If the Phase 1 checkpoint already helps, a trained-better winner will help more; if it doesn't help at all, ablations can't save it.
- **Phase 2A** (ablation sweep, ~1-2 days compute + review): 9-cell orthogonal sweep around Phase 1 defaults (T-window × input × pool × backbone). **Gate:** winner beats Phase 1 test AUC by ≥ +1.0pp AND multi-seed (3 seeds) variance confirms signal (std ≤ half the delta). Orthogonality+neg-control re-check on the winner must still clear Phase 1's gate. If top-2 cells vary different axes, run ONE full-crossed cell to confirm no interaction lift.
- **Phase 2B** (integration + 68-fold LOO, ~3-4 hr compute): plug winner into `run_decoder_over_rally` as the emitter replacing the GBM. Same A/B harness. **Gate:** integrated F1 ≥ 89.0% OR F1 delta ≥ +0.5pp over baseline; Block F1 ≥ 20.0% OR Block TP ≥ 40; no per-class regression > 2pp (block exempt to −8pp).
- **Phase 2C** (ensemble fallback branch, ~2 hr, only if Phase 2B partially passes): per-class ensemble (use crop-head prob where it beat GBM per-class in Phase 2B; GBM elsewhere). Same gates as Phase 2B. If still NO-GO, close the workstream.

**Tech Stack:** PyTorch + torchvision, MPS on macOS, `uv`, reuse of Phase 1 dataset + decoder integration A/B harness.

**Context for the engineer (read these before starting):**
- Phase 1 PASS memo: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/crop_head_phase1_pass_2026_04_20.md`
- Phase 0 feasibility: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/crop_head_phase0_2026_04_20.md` — don't-retry list, spatial-resolution findings.
- Decoder integration PASS: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/decoder_integration_pass_2026_04_20.md` — explains `run_decoder_over_rally` + overlay architecture.
- Canonical baseline: `scripts/eval_loo_video.py` on 68 videos / 364 rallies / 2095 contacts → 88.0% F1 / 91.2% Action Acc.
- Production current state: GBM + decoder overlay = 87.95% F1 / 93.43% Action Acc.
- Phase 1 best test AUC: 0.8385 (epoch 9 val peak 0.8703). Winner must beat on the SAME 10-video held-out test set.
- Block F1 12.12% → decoder overlay 6.45% → Phase 2 target ≥20% OR ≥40 TPs. Block has 165 total GT contacts across the corpus; +8pp F1 ≈ +10 TPs, which can be within fold noise. TP-count fallback makes the gate robust at low class size.
- Phase 1 checkpoint: saved during the Phase 1 training run via `best_state` (best val AUC at epoch 9). **CRITICAL for Phase 2.0:** Phase 1's script (`train_crop_head_probe.py`) kept `best_state` in memory but only wrote the final-epoch state. Task 0 must either (a) re-run Phase 1 training for 9 epochs at seed=42 to reproduce the checkpoint, or (b) rerun with `--epochs 9 --save-checkpoint weights/crop_head/phase1_repro.pt` added.

---

## Pre-registered ship gates (DO NOT modify after measurement runs)

### Phase 2.0 gate (kill-test, ~3 hr)

**Experimental arm:** Phase 1 checkpoint used as emitter, UNCHANGED (no retraining, no ablation).

**Gate (either clauses suffices; must pass at least one):**
- Integrated Contact F1 delta ≥ **+0.3pp** vs GBM baseline
- Block F1 delta ≥ **+3.0pp** (from baseline 6.45%)

**Fail BOTH → NO-GO for entire Phase 2.** Do NOT enter Phase 2A. Write NO-GO memo; redirect to GT attribution work.

Rationale: if the off-the-shelf Phase 1 model (optimally-trained for AUC, not for integrated F1) doesn't move either metric meaningfully on the full LOO, a +1pp AUC improvement from ablations won't rescue a 0pp integrated shift. The kill-test costs 3 hr; the ablations cost 2-3 days. Ordering matters.

### Phase 2A gate (ablation sweep)

**Winner = highest-test-AUC config on the Phase 1 held-out 10-video test set (same videos, frozen for comparability).**

All three must hold:
1. Winner test AUC ≥ Phase 1 baseline + **1.0pp** (i.e., ≥ 0.8485)
2. **Multi-seed confirmation:** re-run the winner config at seeds {42, 1337, 7} and compute mean test AUC + std. Gate: mean delta vs Phase 1 ≥ +1.0pp AND std ≤ 0.5 × mean_delta (i.e., signal-to-noise > 2×).
3. **Orthogonality+neg-control re-check:** P(GT ≥ 0.5) − P(non-contact ≥ 0.5) ≥ 0.15 on the winner's test set. (Phase 1 gate applied again; protects against overfitting cells that game AUC without preserving the VideoMAE kill discriminator.)

**Interaction sanity check:** if the top-2 cells vary DIFFERENT axes (e.g., T=13 is #1 and attention-pool is #2), run ONE additional cell combining both. If it beats either parent by ≥ 0.5pp, that's the winner; otherwise the top-1 wins.

**Fail ANY gate → Phase 2A NO-GO.** Do not enter Phase 2B.

### Phase 2B gate (integrated 68-fold LOO)

**Baseline arm:** current production (GBM + decoder overlay).
**Experimental arm:** winner crop-head replaces GBM inside `run_decoder_over_rally`; same detect_contacts; same overlay.

Gates (ALL must hold, with OR-clauses where noted):
1. Contact F1 ≥ **89.0%** **OR** F1 delta ≥ **+0.5pp** over baseline (whichever is easier to clear — recognizes that a +0.5pp lift on the published 87.95% baseline is material even if it doesn't cross the 89% absolute threshold)
2. Block F1 ≥ **20.0%** **OR** Block TP ≥ **40** (TP-count fallback prevents small-sample F1 noise from sinking the gate)
3. No per-class F1 regression > **2.0pp** (block exempt to **−8pp**, tighter than decoder overlay's −12pp exemption since a crop-head that CAN'T improve block isn't worth the integration cost)
4. Baseline arm sanity: within ±0.3pp of published 87.95%

**Fail ANY gate → consider Phase 2C (ensemble fallback). If Phase 2C also fails, revert integration, keep crop-head as dormant research artifact.**

### Phase 2C gate (per-class ensemble fallback — conditional)

**Trigger:** Phase 2B clears gate 4 (sanity) but fails gate 1, 2, or 3 ONLY if per-class analysis shows crop-head beats GBM on ≥ 2 classes.

**Experimental arm:** per-class ensemble — use crop-head probability for classes where it won in Phase 2B; GBM probability for classes where it lost. Class-level switching at the `decoder_runtime` emitter boundary.

Gates:
1. All Phase 2B gates (re-evaluated on the ensemble arm)

**Fail → close workstream.** Final NO-GO memo. Crop-head modules retained as dormant research infrastructure (reusable for future variants).

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `analysis/scripts/train_crop_head_probe.py` | MODIFY | Add `--save-checkpoint PATH` flag. Add `--seed` flag (already exists — verify it's honored). Add `--t-window`, `--input-kind`, `--pool-kind`, `--backbone-kind`, `--cache-subdir` flags. |
| `analysis/scripts/extract_crop_dataset.py` | MODIFY | Add `--t-window` and `--hard-conf-lo/--hard-conf-hi` flags. |
| `analysis/rallycut/ml/crop_head/head.py` | MODIFY | Add `AttentionPoolHead` and `MaxPoolHead`. |
| `analysis/rallycut/ml/crop_head/backbone.py` | MODIFY | Add `PartiallyUnfrozenResNet18`. |
| `analysis/rallycut/ml/crop_head/model.py` | CREATE | Unified `CropHeadModel(input_kind, pool_kind, backbone_kind)`. |
| `analysis/rallycut/tracking/crop_head_emitter.py` | CREATE | `CropHeadContactClassifier` adapter with `predict_proba` returning (N, 2) array, used by `run_decoder_over_rally` when emitter=crop_head. |
| `analysis/rallycut/tracking/decoder_runtime.py` | MODIFY | Accept any emitter implementing `predict_proba`. Optional `video_path` + `rally_start_frame` for crop extraction. |
| `analysis/scripts/eval_decoder_integration.py` | MODIFY | Add `--emitter {gbm, crop_head, ensemble}` and `--crop-head-weights PATH`. |
| `analysis/scripts/sweep_crop_head_ablations.py` | CREATE | Orchestrator: 9-cell sweep + winner multi-seed + interaction check. |
| `analysis/reports/crop_head_phase2_0_killtest_<date>.md` | GENERATED | Task 0 PASS/NO-GO. |
| `analysis/reports/crop_head_phase2a_sweep_<date>.md` | GENERATED | Task 3 PASS/NO-GO + winner config. |
| `analysis/reports/crop_head_phase2b_integration_<date>.md` | GENERATED | Task 5 PASS/NO-GO. |
| `analysis/reports/crop_head_phase2c_ensemble_<date>.md` | GENERATED (conditional) | Task 6 PASS/NO-GO. |
| `analysis/tests/unit/test_crop_head_model.py` | CREATE | Model composition + partial-unfreeze tests. |
| `analysis/tests/unit/test_crop_head_emitter.py` | CREATE | `predict_proba` protocol compliance + crop-at-inference correctness. |

---

## Task 0: Kill-test with existing Phase 1 checkpoint (~3 hr, hard gate)

**Goal:** Cheap 68-fold A/B using the UNMODIFIED Phase 1 checkpoint as an emitter. Either proves the architectural premise survives integration, or kills Phase 2 before we spend 2-3 days on ablations.

**Files:**
- MODIFY: `analysis/scripts/train_crop_head_probe.py` — add `--save-checkpoint PATH` flag.
- CREATE: `analysis/rallycut/tracking/crop_head_emitter.py`
- MODIFY: `analysis/rallycut/tracking/decoder_runtime.py`
- MODIFY: `analysis/scripts/eval_decoder_integration.py`
- CREATE: `analysis/tests/unit/test_crop_head_emitter.py`

### Step 0.1: Add checkpoint-saving to the training script

Open `analysis/scripts/train_crop_head_probe.py`. Find the final `best_state` handling (around the end-of-training block). Add:

```python
parser.add_argument("--save-checkpoint", type=str, default=None,
                    help="Path to save the best (val-AUC-peak) state_dict.")

# ... later, just after best_state is finalized and before test-set eval:
if args.save_checkpoint and best_state is not None:
    ckpt_path = Path(args.save_checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "input_kind": getattr(args, "input_kind", "concat"),
        "pool_kind": getattr(args, "pool_kind", "mean"),
        "backbone_kind": getattr(args, "backbone_kind", "frozen"),
        "t_window": getattr(args, "t_window", 9),
        "best_val_auc": best_val_auc,
        "epoch": best_epoch,  # if tracked; else final epoch
        "seed": args.seed,
    }, ckpt_path)
    console.print(f"[bold]Saved checkpoint:[/bold] {ckpt_path}")
```

If `input_kind` / `pool_kind` / `backbone_kind` / `t_window` args don't exist yet (they'll be added in Task 2), the `getattr` falls back to Phase 1 defaults. That's intentional — Task 0 uses Phase 1 defaults.

### Step 0.2: Reproduce the Phase 1 checkpoint

Re-run Phase 1 training at seed 42 for 9 epochs (matches Phase 1's best-val-AUC epoch), saving the checkpoint:

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/train_crop_head_probe.py \
    --epochs 9 --seed 42 \
    --save-checkpoint weights/crop_head/phase1_repro_seed42.pt \
    --out-json reports/crop_head_phase1_repro.json \
    2>&1 | tee reports/crop_head_phase1_repro.log
```

Expected: runs in ~5-6 min on MPS. Val AUC should peak around 0.87 (Phase 1 hit 0.8703 at epoch 9). Test AUC should land near 0.8385. If either metric is wildly off, reproducibility is broken — stop and investigate before Task 0 measurement.

### Step 0.3: Create the emitter adapter

```python
# analysis/rallycut/tracking/crop_head_emitter.py
"""Inference-time crop-head emitter. Adapts CropHeadModel to the
ContactClassifier.predict_proba protocol the decoder uses.

At inference time, extracts crops on-the-fly from the source video so
training + inference distributions align. Uses the same bbox selection
logic as scripts/extract_crop_dataset.py — nearest-player-to-ball at
the candidate frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from rallycut.ml.crop_head.model import CropHeadModel
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos


@dataclass
class _Checkpoint:
    state_dict: dict
    input_kind: str
    pool_kind: str
    backbone_kind: str
    t_window: int


class CropHeadContactClassifier:
    """predict_proba(candidates) -> (N, 2) array of [P(bg), P(contact)].

    Requires video_path + rally_start_frame + position lookups at construction
    time so it can extract crops per candidate during predict_proba.

    Crop extraction matches scripts/extract_crop_dataset.py:
      - player crop from nearest-to-ball track's bbox at candidate frame,
        with ±2f nearest-neighbor fallback
      - ball patch from ball position at candidate frame (same ±2f fallback)
      - T frames centered on the candidate (T=9 for Phase 1 checkpoint)
      - 64×64 player crop with 10% padding; 32×32 ball patch at ±4% of frame width.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        video_path: Path,
        rally_start_frame: int,
        ball_positions: list[BallPos],
        player_positions: list[PlayerPos],
        device: str = "auto",
    ):
        self.video_path = Path(video_path)
        self.rally_start_frame = rally_start_frame
        self.ball_positions = list(ball_positions)
        self.player_positions = list(player_positions)
        self.device = self._resolve_device(device)
        self._ckpt = self._load_checkpoint(checkpoint_path)
        self.model = CropHeadModel(
            input_kind=self._ckpt.input_kind,
            pool_kind=self._ckpt.pool_kind,
            backbone_kind=self._ckpt.backbone_kind,
        ).to(self.device)
        self.model.load_state_dict(self._ckpt.state_dict)
        self.model.eval()
        self.is_trained = True  # protocol alignment with ContactClassifier
        # Exposed for decoder_runtime's pad/truncate logic (not used — crop-head
        # doesn't take the 26-dim feature vector, but the protocol expects it):
        self.model_dummy_n_features_in_ = 1

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    @staticmethod
    def _load_checkpoint(path: Path) -> _Checkpoint:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        return _Checkpoint(
            state_dict=raw["state_dict"],
            input_kind=raw.get("input_kind", "concat"),
            pool_kind=raw.get("pool_kind", "mean"),
            backbone_kind=raw.get("backbone_kind", "frozen"),
            t_window=raw.get("t_window", 9),
        )

    def predict_proba(self, candidates: list[Any]) -> np.ndarray:
        """Returns (N, 2) array: column 0 = P(bg), column 1 = P(contact).

        `candidates` is the list of CandidateFeatures (from
        extract_candidate_features). Only their `frame` attribute is used.
        """
        if not candidates:
            return np.zeros((0, 2))
        # Build per-frame lookups (mirrors extract_crop_dataset.py logic)
        ball_by_frame: dict[int, BallPos] = {}
        for bp in self.ball_positions:
            if bp.x > 0 or bp.y > 0:
                ball_by_frame[bp.frame_number] = bp

        # Open video once and batch crops
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            crops_pc, crops_bp = [], []
            for c in candidates:
                pc, bp = self._extract_for_frame(cap, c.frame, ball_by_frame)
                crops_pc.append(pc)
                crops_bp.append(bp)
        finally:
            cap.release()

        pc_tensor = torch.tensor(np.stack(crops_pc), dtype=torch.float32, device=self.device)
        bp_tensor = torch.tensor(np.stack(crops_bp), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(pc_tensor, bp_tensor)  # (N,)
            probs = torch.sigmoid(logits).cpu().numpy()

        out = np.zeros((len(candidates), 2))
        out[:, 0] = 1.0 - probs
        out[:, 1] = probs
        return out

    def _extract_for_frame(
        self,
        cap: cv2.VideoCapture,
        candidate_rally_frame: int,
        ball_by_frame: dict[int, BallPos],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (player_crop[T,3,64,64], ball_patch[T,3,32,32]) for one candidate."""
        # [Engineer mirrors extract_crop_dataset.py::_extract_sample here. Key
        #  difference: no np.savez, return numpy arrays directly. Use the same
        #  ±2f nearest-neighbor fallback for bbox + ball. Nearest-player-to-ball
        #  picks the track_id whose (x, y) is closest to the ball at the
        #  candidate frame. Extract the T frames centered on
        #  self.rally_start_frame + candidate_rally_frame. Return float32 [0,1].
        #
        #  Important: if the candidate is within T//2 of rally start, pad the
        #  window — zero-fill the missing frames. Same for rally end. Matches
        #  training distribution where edge candidates had incomplete windows.
        # ]
        raise NotImplementedError("engineer fills in from extract_crop_dataset.py reference")
```

Engineer fills in `_extract_for_frame` using `extract_crop_dataset.py`'s `_extract_sample` as the reference. Must match training distribution exactly — nearest-player-to-ball track, same padding, same fallback order.

### Step 0.4: Generalize `run_decoder_over_rally` to accept any emitter

Open `analysis/rallycut/tracking/decoder_runtime.py`. Currently the function takes `classifier: ContactClassifier` and calls `classifier.model.predict_proba(X)` on the 26-dim feature vector. Refactor so:

1. The `classifier` parameter is typed as a Protocol that requires `is_trained: bool` + `predict_proba(features)`.
2. `predict_proba`'s argument is made polymorphic: either the 26-dim feature matrix (GBM) or the list of `CandidateFeatures` (crop-head). The GBM path already needs the feature matrix; keep that path unchanged. Add a branch:

```python
if hasattr(classifier, "predict_proba") and not hasattr(classifier, "model"):
    # New-style emitter (e.g., CropHeadContactClassifier) — takes candidates directly
    gbm_probs = classifier.predict_proba(feats_list)[:, 1]
else:
    # Legacy ContactClassifier path — uses feature matrix
    x_mat = np.array([f.to_array() for f in feats_list], dtype=np.float64)
    expected = classifier.model.n_features_in_
    if x_mat.shape[1] > expected:
        x_mat = x_mat[:, :expected]
    elif x_mat.shape[1] < expected:
        pad = np.zeros((x_mat.shape[0], expected - x_mat.shape[1]))
        x_mat = np.hstack([x_mat, pad])
    gbm_probs = classifier.model.predict_proba(x_mat)[:, 1]
```

Back-compat: all existing callers (decoder Task 5 CLI wiring, integration A/B) still work with `ContactClassifier` unchanged.

### Step 0.5: Wire `--emitter crop_head` into A/B harness

In `analysis/scripts/eval_decoder_integration.py`, add:
```python
parser.add_argument("--emitter", choices=["gbm", "crop_head"], default="gbm")
parser.add_argument("--crop-head-weights", type=str, default=None,
                    help="Path to the crop-head checkpoint. Required when --emitter=crop_head.")
```

When `--emitter crop_head`, the experimental arm instantiates `CropHeadContactClassifier` per rally (requires `video_path`, `rally_start_frame`, positions — all available in the per-fold loop from the `RallyPrecomputed` + DB lookup). Thread `video_path` through via `get_video_path(rally.video_id)` with caching (avoid re-downloading per rally — cache `{video_id: video_path}` in the eval script).

Baseline arm always uses GBM.

### Step 0.6: Write unit tests for the emitter

```python
# analysis/tests/unit/test_crop_head_emitter.py
"""Unit tests for the CropHeadContactClassifier adapter."""
# Tests:
# 1. predict_proba returns shape (N, 2) on valid inputs
# 2. predict_proba returns [] with shape (0, 2) for empty candidates
# 3. is_trained = True after load
# 4. Checkpoint load roundtrip: save with save_checkpoint, load, verify config survives
# 5. predict_proba output sums to 1 per row (valid probability distribution)
```

Engineer implements. Target: 5 passing tests.

### Step 0.7: Smoke test (1 rally)

Pick one rally with action GT. Confirm the emitter runs end-to-end:

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python -c "
from pathlib import Path
from rallycut.tracking.crop_head_emitter import CropHeadContactClassifier
from rallycut.evaluation.tracking.db import get_video_path
from scripts.eval_action_detection import load_rallies_with_action_gt
from scripts.eval_loo_video import _precompute
from rallycut.tracking.contact_detector import ContactDetectionConfig
from scripts.train_contact_classifier import extract_candidate_features

rallies = load_rallies_with_action_gt()
r = rallies[0]
pre = _precompute(r, ContactDetectionConfig())
video_path = get_video_path(r.video_id)
start_frame = round(r.start_ms * r.fps / 1000)
clf = CropHeadContactClassifier(
    Path('weights/crop_head/phase1_repro_seed42.pt'),
    video_path, start_frame,
    pre.ball_positions, pre.player_positions,
)
feats, frames = extract_candidate_features(r, config=ContactDetectionConfig())
probs = clf.predict_proba(feats)
print(f'candidates={len(feats)} probs.shape={probs.shape} probs[:3]={probs[:3]}')
"
```

Expected: shape (N, 2), row sums = 1, no errors.

### Step 0.8: Full 68-fold A/B with Phase 1 checkpoint as emitter

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/eval_decoder_integration.py \
    --emitter crop_head \
    --crop-head-weights weights/crop_head/phase1_repro_seed42.pt \
    2>&1 | tee reports/crop_head_phase2_0_killtest.log
```

Use `run_in_background: true`. Expected wall-clock: 1-3 hr (double the GBM path because of video-read I/O per candidate; acceptable for a one-shot gate).

### Step 0.9: Check Task 0 gates

Read `reports/decoder_integration_<date>.md` (the harness writes to its standard path; rename to `crop_head_phase2_0_killtest_<date>.md` for traceability). Compute:
- Integrated F1 delta (experimental − GBM baseline)
- Block F1 delta
- Per-class F1 deltas
- Baseline sanity (baseline should be ~87.95% — confirms harness sane)

### Step 0.10: Decision

**PASS (F1 Δ ≥ +0.3pp OR Block F1 Δ ≥ +3pp):**

```bash
git add analysis/scripts/train_crop_head_probe.py \
        analysis/rallycut/tracking/crop_head_emitter.py \
        analysis/rallycut/tracking/decoder_runtime.py \
        analysis/scripts/eval_decoder_integration.py \
        analysis/tests/unit/test_crop_head_emitter.py \
        analysis/reports/crop_head_phase2_0_killtest_*.md \
        analysis/reports/crop_head_phase2_0_killtest.log \
        analysis/reports/crop_head_phase1_repro.log \
        analysis/reports/crop_head_phase1_repro.json
git commit -m "..."  # descriptive; mention the gate passed
```

Proceed to Task 1.

**NO-GO (both gates fail):**

Same commit pattern (the infrastructure is still worth keeping), but then STOP the plan. Write NO-GO memo to `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/crop_head_phase2_nogo_<date>.md` explaining:
- Which gate failed and by how much
- What the failure implies: Phase 1's AUC lift doesn't translate to integrated F1; architectural premise of the overlay integration is broken for this emitter
- Redirect recommendation: GT attribution work or accept current production ceiling (93.43% Action Acc after decoder overlay)

Update MEMORY.md index with the NO-GO. Close the workstream.

---

## Task 1: Expand the crop cache (ONLY if Task 0 PASSED)

**Goal:** Extract T=5, T=13, T=17 variants + wider hard-negative pool for Phase 2A's sweep.

**Files:**
- MODIFY: `analysis/scripts/extract_crop_dataset.py`

### Step 1.1: Add `--t-window` and `--hard-conf-lo/--hard-conf-hi` flags

```python
parser.add_argument("--t-window", type=int, default=9, choices=[5, 9, 13, 17])
parser.add_argument("--hard-conf-lo", type=float, default=0.20)
parser.add_argument("--hard-conf-hi", type=float, default=0.30)
parser.add_argument("--output-subdir", type=str, default=None,
                    help="Subdir under outputs/crop_dataset/. Default: t{T}_h{lo:02.0f}_{hi:02.0f}")
```

Thread `T` through `_extract_sample` (currently uses module-level `T_FRAMES`). `HALF = T // 2`. Keep npz schema identical. Default output layout: `analysis/outputs/crop_dataset/t09_h20_30/<video_id>/`. Phase 1's cache stays at `analysis/outputs/crop_dataset/<video_id>/` untouched.

### Step 1.2: Smoke-test each T on 1 video

Run `--video <small-rally-video-id> --t-window {5,13,17} --hard-conf-lo 0.10 --hard-conf-hi 0.40` and verify:
```bash
uv run python -c "
import numpy as np, glob
for t in (5, 13, 17):
    path = glob.glob(f'outputs/crop_dataset/t{t:02d}_h10_40/*/*.npz')[0]
    d = np.load(path)
    assert d['player_crop'].shape == (t, 3, 64, 64)
    print(f'T={t}: {d[\"player_crop\"].shape}')
"
```

### Step 1.3: Full extraction (T=5, T=9 wide, T=13, T=17)

4 runs in sequence (each ~70 min). Run in background per T. Total ~5 hr wall-clock, monitored.

### Step 1.4: Commit the script change

```bash
git add analysis/scripts/extract_crop_dataset.py
git commit -m "feat(ml): crop extractor T-window + hard-neg range flags (Phase 2A prep)"
```

---

## Task 2: Architecture variants (~1 hr dev + 5 min tests)

**Files:**
- MODIFY: `analysis/rallycut/ml/crop_head/head.py` (AttentionPoolHead + MaxPoolHead)
- MODIFY: `analysis/rallycut/ml/crop_head/backbone.py` (PartiallyUnfrozenResNet18)
- CREATE: `analysis/rallycut/ml/crop_head/model.py`
- CREATE: `analysis/tests/unit/test_crop_head_model.py`

### Step 2.1: Attention + max pool heads

Append to `head.py`:

```python
class AttentionPoolHead(nn.Module):
    """Attention-weighted temporal pooling. Lets the model focus on the
    1-frame contact event in the T-frame window instead of averaging it
    into noise (the failure mode of mean-pool at long T)."""

    def __init__(self, d_in: int = 1024, d_hidden: int = 256) -> None:
        super().__init__()
        self.attn = nn.Linear(d_in, 1)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        out: torch.Tensor = self.net(pooled)
        return out.squeeze(-1)


class MaxPoolHead(nn.Module):
    """Per-feature max across T. Simpler than attention, zero learnable
    pooling params. Often competitive with attention on simple tasks."""

    def __init__(self, d_in: int = 1024, d_hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = x.max(dim=1)
        out: torch.Tensor = self.net(pooled)
        return out.squeeze(-1)
```

### Step 2.2: Partially-unfrozen backbone

Append to `backbone.py`:

```python
class PartiallyUnfrozenResNet18(nn.Module):
    """FrozenResNet18 variant with layer4 (last residual block) trainable.
    ~11M params out of 11.7M total remain frozen; layer4's 0.6M trainable."""

    def __init__(self) -> None:
        super().__init__()
        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        for name, p in self.backbone.named_parameters():
            # layer4 is at index 7 in the Sequential's child list
            p.requires_grad = name.startswith("7.")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        x = (x - mean) / std
        feat: torch.Tensor = self.backbone(x)
        return feat.flatten(1)
```

Note: no `@torch.no_grad()` (layer4 needs gradients). Training loop must put frozen layers (indices 0-6) in eval mode: `for i in range(7): model.backbone.backbone[i].eval()` before each forward.

### Step 2.3: Unified model composer

Create `analysis/rallycut/ml/crop_head/model.py`:

```python
"""Unified crop-head model for Phase 2A ablations."""
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
        self.backbone: nn.Module = (
            FrozenResNet18() if backbone_kind == "frozen"
            else PartiallyUnfrozenResNet18()
        )
        d_feat = 512
        d_in = 2 * d_feat if input_kind == "concat" else d_feat
        self.head: nn.Module = {
            "mean": CropHeadMLP(d_in=d_in),
            "attention": AttentionPoolHead(d_in=d_in),
            "max": MaxPoolHead(d_in=d_in),
        }[pool_kind]

    def forward(self, player_crops: torch.Tensor, ball_patches: torch.Tensor) -> torch.Tensor:
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
            combined = ball_feat if self.input_kind == "ball_only" else torch.cat([player_feat, ball_feat], dim=-1)

        return self.head(combined)
```

### Step 2.4: Tests

Create `analysis/tests/unit/test_crop_head_model.py` covering: forward shapes for all 3×3×2 = 18 combos sampled representatively (at least 4 cells including Phase 1 baseline, player-only + attention + layer4-unfrozen); partial-unfreeze gradient mask correct; trainable-params filtering correct.

Target: ≥ 7 passing tests.

### Step 2.5: Commit

```bash
git add analysis/rallycut/ml/crop_head/ analysis/tests/unit/test_crop_head_model.py
git commit -m "feat(ml): Phase 2A architecture variants (attention/max pool + layer4 unfreeze)"
```

---

## Task 3: Phase 2A ablation sweep + multi-seed + interaction check (~2-3 hr compute)

**Files:**
- MODIFY: `analysis/scripts/train_crop_head_probe.py` (accept config flags, use `CropHeadModel`)
- CREATE: `analysis/scripts/sweep_crop_head_ablations.py`
- GENERATED: `analysis/reports/crop_head_phase2a_sweep_<date>.md`

### Design (9 cells + winner multi-seed + optional interaction)

| Cell | T | Input | Pool | Backbone | Cache subdir |
|---|---|---|---|---|---|
| baseline | 9 | concat | mean | frozen | t09_h10_40 |
| t_short | 5 | concat | mean | frozen | t05_h10_40 |
| t_long | 13 | concat | mean | frozen | t13_h10_40 |
| t_longest | 17 | concat | mean | frozen | t17_h10_40 |
| player_only | 9 | player_only | mean | frozen | t09_h10_40 |
| ball_only | 9 | ball_only | mean | frozen | t09_h10_40 |
| attention | 9 | concat | attention | frozen | t09_h10_40 |
| max | 9 | concat | max | frozen | t09_h10_40 |
| ft_layer4 | 9 | concat | mean | layer4_unfrozen | t09_h10_40 |

**Multi-seed:** after the 9-cell sweep identifies a top-1 winner, re-run that config at seeds {42 (already done), 1337, 7}. Gate: mean test AUC lift vs Phase 1 ≥ 1.0pp AND std ≤ 0.5× mean lift.

**Interaction:** if top-2 differ on different axes (e.g., t_long is #1 at 0.86 AUC, attention is #2 at 0.855), run ONE crossed cell (t=13 AND attention). If crossed beats top-1 by ≥0.5pp, it wins; otherwise top-1 wins. Crossed cell gets the multi-seed treatment too.

**Orthogonality+neg-control re-check:** on the final winner checkpoint (best of 3 seeds), run Phase 1's orthogonality probe on the test set. Gate ≥ 0.15 gap (same as Phase 1).

### Step 3.1: Make `train_crop_head_probe.py` config-driven

Add flags: `--t-window`, `--input-kind`, `--pool-kind`, `--backbone-kind`, `--cache-subdir`, `--run-name`, `--save-checkpoint`. Replace the hard-coded `FrozenResNet18 + CropHeadMLP` with `CropHeadModel(input_kind, pool_kind, backbone_kind)`.

For `layer4_unfrozen`: use `AdamW` with two param groups — head at lr=1e-3, backbone params at lr=1e-5 (10× smaller for fine-tune stability).

HARDCODE the Phase 1 10-video test set at module level (copy the list from `reports/crop_head_phase1_probe_2026-04-20.md` — exact video IDs). Split the dataset by this list, not by every-7th. Prevents test-set drift between cells.

### Step 3.2: Sweep orchestrator

Create `analysis/scripts/sweep_crop_head_ablations.py` with:
- `CELLS` constant (the 9 cells above)
- `run_cell(config)` function that invokes the probe script via subprocess, reads its JSON output
- Main loop: run all 9 cells, rank by test AUC, identify top-1
- Multi-seed: re-run top-1 at seeds 1337 and 7, aggregate
- Interaction: if top-2 on different axes, run crossed cell (+ multi-seed for it)
- Orthogonality re-check: load top-1 checkpoint, run orthogonality probe, record gap
- Write `reports/crop_head_phase2a_sweep_<date>.md` with PASS/NO-GO against all 3 Phase 2A gates

### Step 3.3: Smoke-run the orchestrator

```bash
uv run python scripts/sweep_crop_head_ablations.py --dry-run
```
Expected: prints 9 cells. Then run `--limit-cells 2` (add flag to orchestrator) to confirm end-to-end flow on 2 cells.

### Step 3.4: Full sweep

```bash
uv run python scripts/sweep_crop_head_ablations.py \
    2>&1 | tee reports/crop_head_phase2a_sweep.log
```
Background. ~1.5 hr base + 3 × 10 min multi-seed + maybe 1 crossed cell × 10 min = ~2.5 hr.

### Step 3.5: Check all 3 Phase 2A gates

Open the generated report. Record:
- Winner config + seed-averaged test AUC
- Multi-seed std (must be ≤ 0.5 × delta)
- Orthogonality gap on winner (must be ≥ 0.15)

### Step 3.6: Commit + branch

```bash
git add analysis/scripts/train_crop_head_probe.py \
        analysis/scripts/sweep_crop_head_ablations.py \
        analysis/reports/crop_head_phase2a_sweep_*.md \
        analysis/reports/crop_head_phase2a_sweep.log
git commit -m "ml(crop-head): Phase 2A sweep — <PASS|NO-GO>, winner=<cell>"
```

- **PASS:** Proceed to Task 4.
- **NO-GO:** STOP. Write NO-GO memo. Close workstream.

---

## Task 4: Phase 2B integrated 68-fold LOO (~3-4 hr)

**Files:**
- `eval_decoder_integration.py` already supports `--crop-head-weights` from Task 0.
- GENERATED: `analysis/reports/crop_head_phase2b_integration_<date>.md`

### Step 4.1: Update `CropHeadContactClassifier` to handle the winner config

The Task 0 emitter hardcoded Phase 1 defaults. If the Phase 2A winner uses `T=13` / `attention` / `layer4_unfrozen`, the emitter's `CropHeadModel(...)` instantiation should already read config from the checkpoint dict — verify the Task 0 implementation honored the `input_kind` / `pool_kind` / `backbone_kind` / `t_window` keys (it should).

If the winner uses T=13 or T=17, the emitter must extract crops at that T. Verify `_extract_for_frame` respects `self._ckpt.t_window`.

### Step 4.2: Full 68-fold A/B with winner

```bash
uv run python scripts/eval_decoder_integration.py \
    --emitter crop_head \
    --crop-head-weights weights/crop_head/phase2a_winner.pt \
    2>&1 | tee reports/crop_head_phase2b.log
```
Background. ~3-4 hr.

### Step 4.3: Check all 4 Phase 2B gates

Record per-class F1 table. Specifically:
- F1 vs 89% OR delta ≥ +0.5pp
- Block F1 vs 20% OR Block TP ≥ 40
- Per-class regressions (block exempt to -8pp)
- Baseline sanity

### Step 4.4: Decision branch

**PASS (all 4 gates):** commit report, write PASS memo, STOP plan. A follow-up production wiring plan handles the actual swap after additional checkpoint validation.

**FAIL but partial (at least 2 classes win):** proceed to Task 5 (Phase 2C ensemble).

**FAIL (no class wins, or overall regression > any floor):** commit report as NO-GO, STOP.

---

## Task 5: Phase 2C per-class ensemble fallback (conditional, ~2 hr)

**Triggered only if Task 4 failed but ≥ 2 classes showed crop-head > GBM.**

**Files:**
- MODIFY: `analysis/rallycut/tracking/crop_head_emitter.py` — add `EnsembleContactClassifier` that takes `{class: emitter}` mapping.
- MODIFY: `analysis/scripts/eval_decoder_integration.py` — add `--emitter ensemble` with `--ensemble-class-map` JSON flag.
- GENERATED: `analysis/reports/crop_head_phase2c_ensemble_<date>.md`

### Design

The decoder emits `{frame, action, team}` with `action ∈ {serve, receive, set, attack, dig, block}`. The ensemble emitter returns per-candidate `[P(bg), P(contact)]` just like the single emitters — but the `P(contact)` prediction for the decoder also influences the decoder's ACTION choice via the `action_probs` computed from `sequence_probs`. The ensemble works at the EMISSION stage, not the ACTION stage.

Simplest ensemble: predict twice (once with GBM, once with crop-head), then combine:
- For "classes-where-crop-head-won": use crop-head's P(contact)
- For "classes-where-GBM-won": use GBM's P(contact)

But this requires a per-candidate expected action to pick the emitter. Use `argmax(sequence_probs[1:])` as the expected action → select emitter.

### Step 5.1: Implement `EnsembleContactClassifier`

```python
class EnsembleContactClassifier:
    """Per-candidate switches between crop-head and GBM based on the
    ms-tcn-argmax expected action class and a pre-chosen per-class winner map.
    """
    def __init__(
        self,
        gbm: ContactClassifier,
        crop_head: CropHeadContactClassifier,
        class_winner_map: dict[str, str],  # {"serve": "crop_head", "block": "gbm", ...}
        sequence_probs: np.ndarray | None,
    ):
        ...

    def predict_proba(self, candidates) -> np.ndarray:
        gbm_probs = self.gbm.predict_proba(...)
        crop_probs = self.crop_head.predict_proba(candidates)
        out = np.zeros_like(gbm_probs)
        for i, c in enumerate(candidates):
            expected_action = self._expected_action(c.frame)
            winner = self.class_winner_map.get(expected_action, "gbm")
            out[i] = crop_probs[i] if winner == "crop_head" else gbm_probs[i]
        return out
```

### Step 5.2: Build the class_winner_map from Phase 2B results

From Task 4 report, compute per-class delta (crop-head − GBM). For each class where delta > 0, winner = "crop_head"; else "gbm". Save as JSON, pass via `--ensemble-class-map`.

### Step 5.3: Run 68-fold A/B with ensemble emitter

### Step 5.4: Check Phase 2B gates on ensemble

If PASS: commit, write PASS memo with ensemble caveat (production wiring is per-class switching, not single-model swap).

If FAIL: write final NO-GO memo. Close workstream. Crop-head modules remain as dormant research infrastructure.

---

## Verification summary

After plan completion (whether PASS or NO-GO at any stage):

1. `uv run pytest tests/unit -q` — all crop-head + emitter tests pass.
2. `uv run ruff check rallycut/` — clean.
3. `uv run mypy rallycut/ml/ rallycut/tracking/` — clean.
4. All generated report files exist for the phases that ran:
   - `reports/crop_head_phase2_0_killtest_<date>.md` (always)
   - `reports/crop_head_phase2a_sweep_<date>.md` (if Task 0 passed)
   - `reports/crop_head_phase2b_integration_<date>.md` (if Task 3 passed)
   - `reports/crop_head_phase2c_ensemble_<date>.md` (if Task 4 partially passed)
5. Production CLI paths (Task 5 decoder wiring from prior plan) still run with GBM — Phase 2 does NOT swap production emitter.
6. `scripts/eval_decoder_integration.py` still supports `--emitter gbm` (default) and reproduces published 87.95% baseline within ±0.3pp.

---

## Honest scope caveats

- **Task 0 is the load-bearing gate.** If the Phase 1 checkpoint (already optimally-trained for AUC) can't shift integrated F1 or Block F1 meaningfully, the architectural premise of emitter-swap is broken — ablations won't fix a 0pp delta.
- **Multi-seed gate is essential.** AUC swings of ±1pp across seeds are common on 902-sample test sets. Declaring a winner without seed confirmation is motivated reasoning.
- **Interaction check is a compromise, not a solution.** A true ablation design would run all 36 cells; the 9-cell orthogonal sweep + 1 crossed cell is a compute-saving heuristic. If the winning combination lies off the one-axis-at-a-time grid AND outside the top-2 interaction, the sweep misses it. That's a known limitation.
- **Block F1 is noisy.** 165 GT blocks total; 10 TPs = 6pp F1 swing. The OR-clause on TP count (≥40) makes the gate robust at low class frequency.
- **Inference cost at production.** Crop extraction per candidate adds ~100ms; ~30 candidates/rally = ~3s added per rally. Acceptable for batch CLI, potentially unacceptable for real-time. Pre-cache optimization is a follow-up if Phase 2B passes.
- **This plan does NOT ship to production.** PASS at Phase 2B (or 2C) earns a follow-up wiring plan, which includes additional checkpoint validation on newly-labeled rallies before swapping production emitter.
- **Gate pre-registration is not optional.** Do NOT modify Task 0, 2A, 2B, or 2C gates after their measurement runs. If a gate is marginal (e.g. Phase 2B F1 = 88.7%, short of 89%), that's NO-GO. Relaxing is motivated reasoning; re-running at different hyperparameters is gate-gaming.

## Self-Review

**Spec coverage:**
- Task 0 (kill-test with Phase 1 checkpoint + pre-registered gate) ✅
- Task 1 (expanded crop cache) ✅
- Task 2 (architecture variants + tests) ✅
- Task 3 (9-cell sweep + multi-seed + interaction check + orthogonality re-check + pre-registered gates) ✅
- Task 4 (Phase 2B integration + 4 pre-registered gates with OR-clauses) ✅
- Task 5 (Phase 2C ensemble fallback, conditional) ✅

**Placeholder scan:**
- Task 0 Step 0.3's `_extract_for_frame` body is `raise NotImplementedError("...")` — acceptable: the spec fully defines the contract (match `extract_crop_dataset.py::_extract_sample`).
- Task 5 Step 5.1 `EnsembleContactClassifier` `__init__` body `...` — acceptable: design fully specified.
- Decision commit messages have `<value>` placeholders — intentional.

**Type consistency:**
- `CropHeadModel(input_kind, pool_kind, backbone_kind)` signature consistent across model.py, train script, sweep orchestrator, emitter.
- `CropHeadContactClassifier.predict_proba` returns `(N, 2)` sklearn-compatible — works with the generalized `run_decoder_over_rally` Protocol.
- `EnsembleContactClassifier` composes the two emitters without changing the public `predict_proba` signature — decoder runtime unaware of ensemble existence.
- Checkpoint dict schema (`state_dict`, `input_kind`, `pool_kind`, `backbone_kind`, `t_window`, `seed`) consistent between save (train script) and load (emitter).

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-04-23-crop-head-phase2.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review per task. Good for this plan because Task 0 is load-bearing (if it fails, hard stop) and per-task review prevents sunk-cost bias.

**2. Inline Execution** — same session with executing-plans, batch execution.

**Recommended kickoff prompt for the fresh session:**

> Execute `docs/superpowers/plans/2026-04-23-crop-head-phase2.md` via subagent-driven development. Start at Task 0 (kill-test). If Task 0 clears the gate, proceed through Tasks 1-4 with per-task commits. If Task 0 fails, STOP and write the NO-GO memo. Do NOT modify any pre-registered gate. Phase 2 only — no production wiring.

Fresh session, direct on main, pre-registered gate discipline.
