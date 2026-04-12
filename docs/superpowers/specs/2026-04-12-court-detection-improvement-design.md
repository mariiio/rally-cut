# Court Detection Improvement Investigation

## Context

The court keypoint detection model (YOLO11s-pose, 6 keypoints) was last trained on 35 videos (525 frames). Since then, 31 more videos have been manually calibrated in the DB (66 total), plus 8 external screenshot images with keypoint annotations. Current MCD is ~0.095 (normalized). Two known problem videos with narrow-angle cameras (`4f2bd66a`, `b026dc6c`) have manual GT but are not in the training set, causing downstream scoring failures (formation detection completely abstains). This investigation retrains with all available GT, diagnoses remaining failures, and explores architecture variants if needed.

## Current State

- **Model**: YOLO11s-pose, 6 keypoints (4 corners + 2 net-sideline intersections)
- **Training data**: 41 videos exported (623 frames), trained on 525 frames from 35 videos
- **DB GT**: 66 videos with `court_calibration_json IS NOT NULL` (all `source='manual'`)
- **External GT**: 8 annotated screenshots in `datasets/court_keypoints_external/`
- **MCD**: ~0.08-0.10 normalized (varies by measurement — 0.0735 reported for v3, 0.095 for current with refinement). Near corners ~7x worse than far corners (off-screen)
- **Key weakness**: Near corners often below frame (y > 1.0), detected with ~0.001 confidence. Refinement via center-point projection + VP fallback reduces error ~33%.
- **Problem videos**: `4f2bd66a` (22/23 formation abstentions), `b026dc6c` (29/29 abstentions) — narrow-angle cameras, not in training set

## Investigation Phases

### Phase 1: Data Expansion

Re-export dataset with all 66 DB videos + 8 external images.

**Script**: `analysis/scripts/export_court_keypoint_dataset.py`

**Modification needed**: Add `--force-val` flag to pin specific video IDs to the validation set. Currently the script does random video-level splitting.

```bash
uv run python scripts/export_court_keypoint_dataset.py \
  --output-dir datasets/court_keypoints_v5 \
  --frames-per-video 15 \
  --pad-ratio 0.3 \
  --val-split 0.2 \
  --force-val 4f2bd66a-61a1-49ac-8137-fd2576e0e851 \
  --force-val b026dc6c-0858-42f2-8513-fc5ee7dbdd93 \
  --force-val 0a383519-ecaa-411a-8e5e-e0aadc835725 \
  --force-val dd042609-e22e-4f60-83ed-038897c88c32 \
  --external-dir datasets/court_keypoints_external
```

**Expected output**: ~990 video frames + 8 external = ~998 total frames. ~53 train / ~13 val videos.

### Phase 2: Baseline Retrain

Train YOLO11s-pose on the expanded v5 dataset. Same hyperparameters as current to isolate the data effect.

**Script**: `analysis/scripts/train_court_keypoint_model.py`

```bash
uv run python scripts/train_court_keypoint_model.py \
  --data datasets/court_keypoints_v5/court_keypoints.yaml \
  --epochs 200 --batch 8
```

**Evaluation** via `analysis/scripts/eval_court_detection.py`, comparing v5 model vs current model:

| Metric | Description | Current | Target |
|--------|-------------|---------|--------|
| Overall MCD | Average corner distance (normalized) | ~0.095 | <0.07 |
| Near-corner MCD | Near-left + near-right average | ~0.15 | <0.10 |
| Far-corner MCD | Far-left + far-right average | ~0.02 | Maintain |
| Court IoU | Polygon overlap (200x200 grid) | ~0.71 | >0.85 |
| Success rate | % videos with MCD < 0.05 | Unknown | >90% |
| Problem video MCD | 4 specific videos individually | Fail | Any detection |

**Decision gate**: If MCD improves >10% on val set, data expansion is validated. Proceed to Phase 3 diagnosis regardless.

### Phase 3: Failure Diagnosis

Run per-video diagnostics on the worst performers after retraining.

**Script**: `analysis/scripts/diagnose_court_detection.py`

Categorize remaining failures into:
- **Near-corner dominated**: MCD driven by near corners with <0.01 confidence (visibility/refinement ceiling)
- **Narrow-angle/extreme perspective**: perspective ratio >4, camera geometry outlier
- **Detection rate**: low bbox confidence, frames missed entirely
- **Refinement failure**: raw keypoints reasonable but center-point/VP refinement regresses

Also compare keypoint model vs classical pipeline on failures via `eval_court_detection.py --compare-keypoint`.

**Output**: Ranked table of remaining failures with root cause category. This determines whether Phase 4 is needed.

### Phase 4: Architecture Exploration (Conditional)

Triggered if Phase 2 shows MCD still >0.07 or problem videos still fail.

**Modification needed**: Add `--model` flag to `train_court_keypoint_model.py` (currently hardcodes `yolo11s-pose.pt`).

| Variant | Model | Params | Hypothesis | Batch |
|---------|-------|--------|-----------|-------|
| A (baseline) | YOLO11s-pose | ~11M | Data alone is enough | 8 |
| B | YOLO11m-pose | ~20M | More capacity helps near-corner extrapolation | 4 |
| C | YOLO11s-pose unfrozen | ~11M | Frozen backbone limits domain adaptation | 8 |

All variants trained on the same v5 dataset, evaluated on the same val split.

**Key comparisons**:
- B > A: model capacity matters, consider YOLO11l-pose
- C > A: frozen backbone too restrictive
- A ~ B ~ C: architecture isn't the bottleneck

**Step 4b — SOTA research**: If YOLO variants plateau, research alternative approaches:
- Sports-field-specific models (TVCalib, SportsFieldLocalization)
- Higher-resolution keypoint models (HRNet-pose, RTMPose)
- Direct homography regression

This is a lightweight research step before committing to a new architecture.

### Phase 5: Deploy

Ship the best model variant:
- Copy to `weights/court_keypoint/court_keypoint_best.pt`
- Update Modal deployment if model file changes
- Re-run `eval_court_detection.py` on all GT videos to confirm production metrics
- Update confidence thresholds in `detector.py` if model confidence distribution changes

## Files to Modify

| File | Change |
|------|--------|
| `analysis/scripts/export_court_keypoint_dataset.py` | Add `--force-val` flag to pin video IDs to val split |
| `analysis/scripts/train_court_keypoint_model.py` | Add `--model` flag to support yolo11m-pose.pt |
| `analysis/rallycut/court/keypoint_detector.py` | Potentially tune refinement params after diagnosis |
| `analysis/rallycut/court/detector.py` | Potentially adjust confidence thresholds |

## Files to Read (no changes)

| File | Purpose |
|------|---------|
| `analysis/scripts/eval_court_detection.py` | Evaluation harness |
| `analysis/scripts/diagnose_court_detection.py` | Per-video failure diagnosis |
| `analysis/rallycut/court/keypoint_detector.py` | Near-corner refinement logic |

## Verification

1. After data export: confirm frame count matches expected (~998), problem videos in val split
2. After training: run `eval_court_detection.py` comparing v5 model vs current model. Print per-video table.
3. After diagnosis: inspect saved frames from worst-performing videos
4. After architecture variants: compare all on same val split in single table
5. Before deploy: run full eval on all 66 GT videos with winning model, confirm no regressions on previously-good videos
