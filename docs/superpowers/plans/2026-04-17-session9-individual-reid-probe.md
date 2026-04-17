# Session 9 — Individual-Identity ReID Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the within-team ReID head with individual player identity (not team membership) as the SupCon positive class. Re-run the anchor viability probe to assess whether anchor-based identity propagation is viable.

**Architecture:** Change the identity encoding in `build_identity_encoding` from `(rally_id, canonical_id)` to `(video_id, canonical_id)` so SupCon pulls same-player crops together across rallies. Add a v4 variant config. Train. Extend the probe script to accept a custom checkpoint (extracting crops from video + running through backbone+head inline). Report.

**Tech Stack:** Python 3.11, PyTorch, DINOv2 ViT-S/14 (frozen backbone), SupCon loss, existing training infrastructure in `training/within_team_reid/`.

**Spec reference:** `docs/superpowers/specs/2026-04-17-session9-individual-reid-probe-design.md`

---

## File Structure

### Modified files

- `analysis/training/within_team_reid/data/dataset.py` — change `build_identity_encoding` key from `(rally_id, canonical_id)` to `(video_id, canonical_id)`. Change `PairDataset.__getitem__` lookup to match.
- `analysis/training/within_team_reid/config.py` — add v4 variant.
- `analysis/scripts/probe_anchor_viability.py` — add `--checkpoint` flag for inline crop extraction + new-head inference.

### New files

- `analysis/weights/within_team_reid/variant_v4/best.pt` — trained checkpoint (gitignored).
- `analysis/reports/merge_veto/anchor_viability_probe_v2.md` — probe results with new head.

---

## Task 1: Change identity encoding + add v4 variant

**Files:**
- Modify: `analysis/training/within_team_reid/data/dataset.py:45-55`
- Modify: `analysis/training/within_team_reid/data/dataset.py:125-126`
- Modify: `analysis/training/within_team_reid/config.py:87-91`

- [ ] **Step 1: Read the current `build_identity_encoding` function**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
```

Read `training/within_team_reid/data/dataset.py` lines 45-55 and lines 125-126.

Current code (lines 45-55):
```python
def build_identity_encoding(pairs: list[Pair]) -> dict[tuple[str, int], int]:
    """Stable mapping (rally_id, canonical_id) → int across the corpus."""
    seen: list[tuple[str, int]] = []
    seen_set: set[tuple[str, int]] = set()
    for p in pairs:
        for tk in (p.track_a, p.track_b):
            key = (p.rally_id, tk.canonical_id)
            if key not in seen_set:
                seen_set.add(key)
                seen.append(key)
    return {k: i for i, k in enumerate(seen)}
```

Current lookup (lines 125-126):
```python
ident_a = self.identity_encoding[(pair.rally_id, pair.track_a.canonical_id)]
ident_b = self.identity_encoding[(pair.rally_id, pair.track_b.canonical_id)]
```

- [ ] **Step 2: Change the encoding to use `video_id` instead of `rally_id`**

In `dataset.py`, replace the `build_identity_encoding` function:

```python
def build_identity_encoding(pairs: list[Pair]) -> dict[tuple[str, int], int]:
    """Stable mapping (video_id, canonical_id) → int across the corpus.

    Session 9: keyed by video_id (not rally_id) so SupCon treats the same
    player across different rallies as ONE identity class. This trains the
    head to pull same-player embeddings together across rallies and push
    teammates apart — the right objective for anchor-based propagation.
    """
    seen: list[tuple[str, int]] = []
    seen_set: set[tuple[str, int]] = set()
    for p in pairs:
        for tk in (p.track_a, p.track_b):
            key = (p.video_id, tk.canonical_id)
            if key not in seen_set:
                seen_set.add(key)
                seen.append(key)
    return {k: i for i, k in enumerate(seen)}
```

Update the lookups in `PairDataset.__getitem__` (lines 125-126):

```python
ident_a = self.identity_encoding[(pair.video_id, pair.track_a.canonical_id)]
ident_b = self.identity_encoding[(pair.video_id, pair.track_b.canonical_id)]
```

- [ ] **Step 3: Add v4 variant to config**

In `analysis/training/within_team_reid/config.py`, add v4 to `VARIANT_CONFIGS`:

```python
VARIANT_CONFIGS: dict[str, TrainConfig] = {
    "v1": TrainConfig(variant_id="v1", lam_tm=0.50, label_smoothing_mid=0.05),
    "v2": TrainConfig(variant_id="v2", lam_tm=1.00, label_smoothing_mid=0.02),
    "v3": TrainConfig(variant_id="v3", lam_tm=0.25, label_smoothing_mid=0.00),
    "v4": TrainConfig(variant_id="v4", lam_tm=0.50, label_smoothing_mid=0.05),
}
```

Note: v4 uses v1's hyperparameters (middle-road lam_tm=0.50). The encoding change is the experimental variable, not the hyperparams.

- [ ] **Step 4: Verify the docstring on the `Pair` class still references correct semantics**

In `analysis/training/within_team_reid/data/manifest.py` line 32-33, the docstring says:
```
Identity label for SupCon = (rally_id, track_a.canonical_id) or (rally_id, track_b.canonical_id).
```

Update to:
```
Identity label for SupCon = (video_id, track_a.canonical_id) or (video_id, track_b.canonical_id).
```

- [ ] **Step 5: Run lint**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run ruff check training/within_team_reid/data/dataset.py training/within_team_reid/config.py training/within_team_reid/data/manifest.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/training/within_team_reid/data/dataset.py analysis/training/within_team_reid/config.py analysis/training/within_team_reid/data/manifest.py
git commit -m "$(cat <<'EOF'
feat(analysis): Session 9 — cross-rally identity encoding for ReID head

Changes build_identity_encoding from (rally_id, canonical_id) to
(video_id, canonical_id). SupCon now treats the same player across
rallies as ONE identity class, training the head to separate teammates
rather than cluster them. Adds v4 variant config.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Train the v4 variant

**Files:**
- No code changes. Runs existing training CLI.

- [ ] **Step 1: Verify training data exists**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
ls training_data/within_team_reid/candidate_pairs.jsonl
wc -l training_data/within_team_reid/candidate_pairs.jsonl
ls training_data/within_team_reid/crops/ | wc -l
```

Expected: `candidate_pairs.jsonl` exists with ~7600 lines, crops directory with ~40+ rally subdirs. If missing, run:
```bash
uv run python scripts/harvest_within_team_pairs.py mine
uv run python scripts/harvest_within_team_pairs.py extract
```

- [ ] **Step 2: Verify eval cache exists**

```bash
ls training_data/within_team_reid/eval_cache.npz
```

If missing, build it:
```bash
uv run python -m training.within_team_reid.cli build-eval-cache
```

- [ ] **Step 3: Train v4**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python -m training.within_team_reid.cli train --variant v4 2>&1 | tee /tmp/session9_train.log
```

Expected runtime: ~30-60 min (MLP-only training, backbone frozen, ~7600 pairs).

Expected output: per-epoch log lines with `held_out_rank1`, `cross_rally_rank1`, `teammate_margin_mean`. Key signals:

- `held_out_rank1` should INCREASE vs Session 3 V3's 0.500 — the head is now trained to discriminate individuals, not just teams.
- `cross_rally_rank1` should STAY ≥ 0.683 (the cross-rally guard from Session 3).
- `teammate_margin_mean` should be strongly positive (the head pushes teammates apart).

Checkpoint saved to: `weights/within_team_reid/variant_v4/best.pt`.

- [ ] **Step 4: Verify checkpoint exists and report training metrics**

```bash
ls -la weights/within_team_reid/variant_v4/best.pt
```

Read the epoch log to find best-epoch metrics:
```bash
cat weights/within_team_reid/variant_v4/epochs.jsonl | python3 -c "
import json, sys
best = None
for line in sys.stdin:
    d = json.loads(line)
    if best is None or d['held_out_rank1'] > best['held_out_rank1']:
        best = d
print(f'Best epoch {best[\"epoch\"]}: held_out_rank1={best[\"held_out_rank1\"]:.3f}, cross_rally_rank1={best[\"cross_rally_rank1\"]:.3f}')
"
```

- [ ] **Step 5: Decision gate**

- If `held_out_rank1 > 0.600` AND `cross_rally_rank1 >= 0.683`: proceed to Task 3 (probe).
- If `held_out_rank1 ≤ 0.500` (same as Session 3 V3): the encoding change had no effect. Investigate before proceeding — check if the new encoding actually produced fewer identity classes (`Identity classes: N` in the training log; should be ~136 instead of ~172).
- If `cross_rally_rank1 < 0.683`: the cross-rally guard fired. Check if this is a threshold issue or a fundamental regression.

---

## Task 3: Extend probe for custom checkpoint

**Files:**
- Modify: `analysis/scripts/probe_anchor_viability.py`

The existing probe uses pre-computed embeddings from the retrack cache's `LearnedEmbeddingStore` (built with Session 3's head). To test the new v4 head, we need to compute embeddings inline.

- [ ] **Step 1: Read the existing probe script**

Read `analysis/scripts/probe_anchor_viability.py` fully. Understand:
- `_load_post_processed_state(rally_id)` — loads cached retrack data, runs post-processing, returns (result, learned_store).
- `_cosine_pair(learned_store, ...)` — computes cosine between two tracks' median embeddings from the store.
- `_process_rally(rally_id, ...)` — orchestrates per-rally processing.

- [ ] **Step 2: Add `--checkpoint` flag and inline embedding extraction**

Add these imports at the top of the file:

```python
import argparse
import cv2
import torch
```

Add a function to extract crops and compute embeddings using a custom head:

```python
def _compute_track_embedding_from_video(
    head: torch.nn.Module,
    backbone: torch.nn.Module,
    device: torch.device,
    video_path: Path,
    positions: list,
    track_id: int,
    frames: set[int],
    max_samples: int = 30,
    min_frames: int = 5,
) -> np.ndarray | None:
    """Extract crops from video, run through backbone+head, return L2-normalized median."""
    sorted_frames = sorted(frames)
    if len(sorted_frames) > max_samples:
        step = len(sorted_frames) / max_samples
        sorted_frames = [sorted_frames[int(i * step)] for i in range(max_samples)]

    # Build frame→bbox lookup from positions
    bbox_by_frame: dict[int, tuple[float, float, float, float]] = {}
    for p in positions:
        if p.track_id == track_id and p.frame_number in set(sorted_frames):
            bbox_by_frame[p.frame_number] = (p.x, p.y, p.width, p.height)

    if len(bbox_by_frame) < min_frames:
        return None

    # Extract crops from video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    crops: list[np.ndarray] = []
    target_frames = sorted(bbox_by_frame.keys())
    frame_idx = 0
    current_target = 0

    while current_target < len(target_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == target_frames[current_target]:
            x, y, w, h = bbox_by_frame[frame_idx]
            fh, fw = frame.shape[:2]
            x1 = max(0, int((x - w / 2) * fw))
            y1 = max(0, int((y - h / 2) * fh))
            x2 = min(fw, int((x + w / 2) * fw))
            y2 = min(fh, int((y + h / 2) * fh))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
            current_target += 1
        frame_idx += 1
    cap.release()

    if len(crops) < min_frames:
        return None

    # Run through backbone + head
    from training.within_team_reid.data.dataset import augment_eval

    tensors = torch.stack([augment_eval(c) for c in crops]).to(device)
    with torch.no_grad():
        backbone_feats = backbone(tensors)
        head_feats = head(backbone_feats)  # L2-normalized by head's forward()

    embeddings = head_feats.cpu().numpy()
    med = np.median(embeddings, axis=0)
    norm = float(np.linalg.norm(med))
    if norm < 1e-8:
        return None
    return (med / norm).astype(np.float32)
```

**Important**: The `augment_eval` function is imported from the training pipeline's dataset module. It resizes crops to the standard input size and normalizes. Check the exact import path and function signature before implementing.

Modify `main()` to accept `--checkpoint`:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a custom head checkpoint. When set, extracts "
                             "crops from video and computes embeddings inline instead "
                             "of using the retrack cache's learned_store.")
    parser.add_argument("--report-out", type=Path,
                        default=OUT_DIR / "anchor_viability_probe.md")
    args = parser.parse_args()

    if args.checkpoint:
        args.report_out = OUT_DIR / "anchor_viability_probe_v2.md"
    ...
```

When `--checkpoint` is set:
- Load the head from checkpoint: `payload = torch.load(args.checkpoint, map_location=device); head.load_state_dict(payload["head_state_dict"])`
- Load the DINOv2 backbone: `from training.within_team_reid.model.backbone import BackboneRunner; backbone = BackboneRunner(device=device)`
- In `_process_rally`, instead of calling `_cosine_pair(learned_store, ...)`, compute embeddings via `_compute_track_embedding_from_video` using the head+backbone.

The video path for each rally needs to be looked up from the database. Use the same DB loading as the main probe flow:
```python
from rallycut.evaluation.ground_truth import load_labeled_rallies
rallies = load_labeled_rallies(all_rallies=True)
# Each rally has .video_path
```

Note: video frame extraction with `cv2.VideoCapture` sequential read + seek is simpler than random access. For each rally, the start/end frames are known from the cached data (`cached_data.frame_count`, rally start/end ms). Position frames are rally-relative (0-indexed from rally start).

**The video_path needs rally-level seeking**: the rally starts at `rally.start_ms` in the full video. Positions are rally-relative (frame 0 = rally start). So set `cap.set(cv2.CAP_PROP_POS_MSEC, rally.start_ms)` before reading frames.

- [ ] **Step 3: Test the modified probe with the EXISTING Session 3 checkpoint first**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/probe_anchor_viability.py --checkpoint weights/within_team_reid/best.pt
```

Expected: results should be SIMILAR to the v1 probe (median cos ~0.784). Not necessarily identical (different crop sampling, augmentation pipeline), but within ±0.05. This validates the inline extraction path works.

If results diverge wildly (median cos differs by > 0.1), the crop extraction or inference path has a bug. Fix before proceeding.

- [ ] **Step 4: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/scripts/probe_anchor_viability.py
git commit -m "$(cat <<'EOF'
feat(analysis): probe_anchor_viability --checkpoint flag

Extends the within-team cosine probe to accept a custom head checkpoint.
When set, extracts crops from video at primary track positions and runs
through DINOv2 backbone + MLP head inline, instead of using the retrack
cache's pre-computed learned_store. Enables testing new head variants
(Session 9) without rebuilding the retrack cache.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Run probe with v4 checkpoint + report

**Files:**
- Output: `analysis/reports/merge_veto/anchor_viability_probe_v2.md`

- [ ] **Step 1: Run the probe with the v4 checkpoint**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/probe_anchor_viability.py \
  --checkpoint weights/within_team_reid/variant_v4/best.pt \
  2>&1 | tee /tmp/session9_probe.log
```

Expected runtime: ~3-5 min (DINOv2 inference on ~3400 crops + video decoding).

Expected output: per-rally cosine pairs, distribution summary, anchor coverage.

- [ ] **Step 2: Read the generated report**

```bash
cat reports/merge_veto/anchor_viability_probe_v2.md
```

Interpret against gates:
- **Viable** (≥70% of videos with anchor cos < 0.5): scope Session 10 propagation.
- **Marginal** (30-70%): note per-video breakdown, decide.
- **Not viable** (<30%): close workstream.

- [ ] **Step 3: Compare v1 vs v2 probe results**

Key comparisons:
- Median within-team cosine: v1 was 0.784. Target: < 0.5.
- % rallies with clear separation (cos < 0.5): v1 was 16%. Target: > 50%.
- % videos with usable anchor: v1 was 18%. Target: > 70%.

---

## Task 5: Final commit + memory update

**Files:**
- `analysis/reports/merge_veto/anchor_viability_probe_v2.md`
- Memory files (outside git)

- [ ] **Step 1: Commit report**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/reports/merge_veto/anchor_viability_probe_v2.md
git commit -m "$(cat <<'EOF'
report(analysis): Session 9 anchor viability probe v2

Individual-identity ReID head (v4) probe results. Identity encoding
changed from (rally_id, canonical_id) to (video_id, canonical_id).
SupCon now trains cross-rally individual identity.

<VERDICT — fill in after running>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Write memory file**

Create `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/session9_individual_reid_probe_2026_04_17.md`:

```markdown
---
name: Session 9 Individual-Identity ReID Probe
description: Retrained within-team head with individual player identity (video_id, canonical_id) instead of team membership (rally_id, canonical_id). Probe result: <VERDICT>.
type: project
---

# Session 9 — Individual-Identity ReID Probe

**Date**: 2026-04-17
**Outcome**: <VIABLE / MARGINAL / NOT VIABLE>

## What changed
- Identity encoding: (rally_id, canonical_id) → (video_id, canonical_id)
- SupCon now pulls same-player across rallies, pushes teammates apart
- Same architecture (DINOv2 ViT-S/14 → 384→192→128 MLP), same loss (SupCon + TeammateMargin)

## Training metrics
- held_out_rank1: <value> (Session 3 V3 was 0.500)
- cross_rally_rank1: <value> (Session 3 V3 was 0.694)
- Identity classes: <N> (Session 3 had ~172, Session 9 should have ~136)

## Probe results
- Median within-team cosine: <value> (Session 8 probe was 0.784)
- % rallies with clear separation (cos < 0.5): <value>% (was 16%)
- % videos with usable anchor: <value>% (was 18%)

## Next step
- If VIABLE: scope Session 10 — anchor propagation pipeline
- If NOT VIABLE: close within-team workstream definitively
```

- [ ] **Step 3: Update MEMORY.md**

Add under the Track-ID Stability section, before the Session 8 entry:

```
- [**Session 9 Individual-Identity ReID Probe 2026-04-17**](session9_individual_reid_probe_2026_04_17.md) — <one-line verdict>
```

---

## Self-review

**Spec coverage**: All 4 spec sections mapped to tasks:
- Training (§1) → Tasks 1-2
- Probe (§2) → Tasks 3-4
- Gates (§3) → Task 4 step 2
- Files (§4) → all tasks

**Placeholder scan**: Task 5's commit message and memory file have `<VERDICT>` and `<value>` placeholders — these are intentionally deferred to runtime (filled in by the implementer after the probe runs).

**Type consistency**: `build_identity_encoding` signature stays `dict[tuple[str, int], int]` — the tuple key type doesn't change (still `(str, int)`), just the semantic of the first element changes from rally_id to video_id.
