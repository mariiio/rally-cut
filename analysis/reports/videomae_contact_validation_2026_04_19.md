# VideoMAE-Based Contact Detection — Validation Decision Memo

**Date**: 2026-04-19
**Decision**: **NO-GO**. Close this branch; trajectory GBM remains production.
**Follow-up**: redirect investment to (a) better ball tracking to reduce the 12% trajectory-FN set, or (b) labeling more contacts in hard scenes, not visual featurization.

## TL;DR

Investigated whether a VideoMAE-based contact/action head could beat the production trajectory-GBM (88.0% F1 / 91.2% action accuracy LOO-per-video). Answer: **no**, and the diagnosis is principled, not a training bug.

Three independent tests converged on the same conclusion:

1. **Binary linear probe (logistic regression on 768-dim CLS)** — **36.9% F1** LOO.
2. **Binary MLP 768→64→2** — **33.3% F1** LOO (overfits on 68 videos).
3. **7-class MSTCN + binary focal-BCE (single fold sanity)** — **35.6% F1** with confident peaks (max p=0.954).

Plus a prior **E2E-Spot** run (per-frame RegNet-Y + GSM + BiGRU, ~50% F1, 2026-03-30 — already on the dead-end list).

**Root cause** (now clear): VideoMAE's 16-frame window averages the single-frame contact event into noise. Temporal context across windows can't recover what window-level aggregation lost.

**Orthogonality check (the decisive test):** VideoMAE probs at held-out GT contacts vs held-out non-contact in-rally frames are **statistically indistinguishable** (GT ≥0.5: 41.1%, non-contact ≥0.5: 43.7%). Non-contacts actually fire *more often*. Signals are redundant, not complementary → Phase 5 fusion cannot help.

## What was built (kept in-tree)

| Artefact | Path | Status |
|---|---|---|
| Modal VideoMAE feature extractor | `analysis/rallycut/service/platforms/modal_features.py` | Deployed, reusable for any future VideoMAE work |
| Local driver | `analysis/scripts/extract_contact_features.py` | Idempotent, parallel, pre-flight validated |
| MinIO→Modal video uploader | `analysis/scripts/upload_videos_to_modal.py` | Uploads 68 videos to `rallycut-features-data` volume |
| 68 stride=4 features | `~/.cache/rallycut/features/*.npy` (68 files, 768-dim) | Local FeatureCache |
| Phase 0 LOO-per-video baseline script | `analysis/scripts/eval_loo_video.py` | Re-usable for any future baseline re-capture |
| Phase 2 linear probe script | `analysis/scripts/train_videomae_contact_probe.py` | Re-runnable on any new features |
| Phase 3 MSTCN head script + module | `analysis/scripts/train_videomae_contact_head.py`, `analysis/rallycut/actions/videomae_contact_head.py` | 7-class, soft-Gaussian targets |
| Orthogonality probe | `analysis/scripts/orthogonality_probe.py` | The definitive Phase-5 go/no-go test |
| FP16 fix in VideoMAE encoder | `analysis/lib/volleyball_ml/video_mae.py` | Bug fix, benefits other code |

Total Modal spend: ~$12 (video uploads + 68 stride=4 feature extractions).

## Measurements

### Phase 0 — LOO-per-video baseline

Re-measured the 92.3% F1 / 90.3% Action Acc figures from memory under proper video-level LOO (memory number was per-rally LOO, which leaks).

| Metric | Value |
|---|---|
| **Contact F1** | **88.0%** (P=91.1%, R=85.0%; TP=1781 FP=174 FN=314) |
| **Action Accuracy** | **91.2%** (1624/1781 matched) |
| Per-class F1 | serve 71.9% • receive 82.3% • set 85.4% • attack 88.3% • dig 69.0% • block 11.8% |

Report: `reports/contact_baseline_loo_video_2026_04_19.md`

### Phase 2 — Linear probe threshold sweep (binary contact, LOO-per-video)

Best F1 at threshold 0.30:
- 27.8% precision, 54.7% recall, **36.9% F1** (TP=1146, FP=2978, FN=949)

Recall stays below 55%; precision never exceeds 38% at any threshold. Report: `reports/videomae_probe_loo_video_2026_04_19.md`.

### Phase 3 — MSTCN head sanity (1 fold, decisive)

Weighted BCE pos_weight=50, MSTCN (2 stages × 8 layers × hidden 64), 20 epochs:
- Max predicted prob reached 0.954 (confident peaks), but fold F1 = **35.6%** — same ceiling as probe.

This ruled out training-setup as cause; the ceiling is the feature ceiling.

### Orthogonality (decisive)

For each GT contact we computed max VideoMAE prob in ±233ms window, with LOO probes (no in-sample leakage). **Paired with negative control** — max VideoMAE prob at random in-rally non-contact frames.

| Group | N | mean prob | frac ≥ 0.3 | frac ≥ 0.5 | frac ≥ 0.7 |
|---|---|---|---|---|---|
| GT contacts | 180 | 0.493 | 51.7% | 41.1% | 41.1% |
| Random non-contact frames | 332 | 0.464 | 45.2% | **43.7%** | 41.0% |

Distributions are indistinguishable. Non-contact frames fire at ≥0.5 more often than GT contacts. **This is the test that proves fusion cannot help**: no matter how we calibrate a threshold, VideoMAE can't tell contacts from non-contacts at held-out.

Report: `reports/orthogonality_negctl_2026_04_19.md` and `reports/videomae_orthogonality_2026_04_19.md`.

## Why visual features fail this task (architectural)

- **Ball trajectory at contact is a physics discontinuity.** Direction change, velocity magnitude, arc-fit residual — these are deterministic signals with sub-pixel resolution, and contact produces a sharp transition detectable to ±1 frame.
- **VideoMAE CLS at stride=4 averages a 16-frame window into a single 768-dim vector.** The 1-frame contact event gets blurred across ~530ms. Recall-at-window is ~55% (the window *containing* a contact is identifiable half the time), but precision-at-frame collapses because every adjacent window fires too.
- **Temporal context (MSTCN, BiGRU) cannot recover what the window aggregation lost.** This is why both MSTCN on VideoMAE features AND per-frame E2E-Spot backbone (~50% F1) sit 35-50pp below the trajectory GBM.

A per-frame backbone with much finer temporal resolution *might* eventually close the gap, but the existing E2E-Spot attempt was already a dead-end at the state-of-the-art for sports action spotting. This is a hard architectural gap, not a tuning problem.

## What would have to be true for visual features to help

1. Either a drastically different featurization (per-frame backbone with sub-frame precision) catches contacts **different from** the ones trajectory catches, **with low FP rate** on non-contact frames. E2E-Spot at ~50% F1 shows this is hard.
2. Or trajectory-style physics features already capture every recoverable contact, and the remaining 12% FN set is fundamentally ambiguous (occluded ball, partial views, visually identical setup vs. dig).

Our orthogonality test directly measured (1) and it failed. We can't measure (2) without crop-level human review, which is itself a non-trivial project.

## Follow-ups that would actually move the needle

These are *not* visual-featurization investments. They target the real bottleneck.

1. **Ball-tracker recall on occluded frames.** Many of the 314 GBM-miss contacts correspond to ball trajectory gaps. Improving WASB's recall in occlusion (2D inpainting, longer temporal windows) directly reduces trajectory FN without new visual heads.
2. **More labeled contacts.** 2095 contacts across 364 rallies is tight for any classifier. 5000+ would also help the GBM alone.
3. **Per-player-crop action head.** Block F1 is 11.8% because blockers are at the net with weak ball-trajectory signal — a per-player-crop ViT on the blocker bbox is the structural fix (out of scope here; see memory `block_detection.md`).
4. **Pose keypoint density.** Pose features already helped (Oct memo): wrist-arm-ball geometry is the best visual discriminator. Extending keypoint coverage (currently ~77% of positions) is more valuable than global scene embeddings.

## What we bank from this investigation

Infrastructure reusable for future ML work:
- Modal + MinIO video-staging pipeline (unblocks any future Modal-GPU experiment)
- Idempotent per-video feature extraction with robust retry + atomic writes
- Negative-control-paired orthogonality test (can reuse for any feature-vs-feature complementarity question before spending on fusion)

Debugging lessons:
- Class-balanced logistic regression on 768-dim features produces **statistically meaningless** high probabilities everywhere when signal is weak. Always pair with a negative control.
- Modal's function return-value buffer is cleared on client disconnect — use `return_exceptions=True` + unordered `order_outputs=False` + per-result local write so kill-mid-run loses at most the in-flight batch.
- `np.save('foo.npy.tmp', arr)` silently appends `.npy` → `.npy.tmp.npy`. Always `np.save(open(path, 'wb'), arr)` for atomic tmp-file writes.

Canonical baseline locked: **88.0% F1 / 91.2% Action Acc LOO-per-video** on 68 videos / 364 rallies / 2095 contacts. This is the honest number — memory's 92.3%/90.3% was per-rally LOO and leaked.
