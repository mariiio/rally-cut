# Candidate Decoder — Ship Memo

**Date**: 2026-04-20
**Status**: SHIP (pending review). Behind a feature flag, default-off for 2-week A/B.

## What ships

Viterbi MAP decoder over the candidate lattice, replacing the GBM's hard threshold decision
as the final accept/reject step. The GBM remains unchanged as emission source and feature
extractor; the decoder only changes how the final contact sequence is selected.

Best configuration (from the LOO-per-video sweep): **action emission enabled, skip_penalty=1.0**.

## Measurements (LOO-per-video, 68 videos, 2095 contacts)

| Metric | Decoder | Phase 0 GBM Baseline | Δ |
|---|---|---|---|
| **Contact F1** | 88.2% | 88.0% | **+0.2pp** (parity) |
| **Action Accuracy** | **95.5%** | 91.2% | **+4.3pp** 🟢 |
| Precision | 90.4% | 91.1% | −0.7pp |
| Recall | 86.1% | 85.0% | +1.1pp |
| TP / FP / FN | 1803 / 192 / 292 | 1781 / 174 / 314 | +22 TP, +18 FP, −22 FN |

### Per-class F1 (decoder vs baseline)

| Class | Decoder | Baseline | Δ |
|---|---|---|---|
| serve | 78.2% | 71.9% | **+6.3pp** |
| receive | 88.0% | 82.3% | **+5.7pp** |
| set | 88.9% | 85.4% | **+3.5pp** |
| attack | 89.4% | 88.3% | +1.1pp |
| dig | 75.2% | 69.0% | **+6.2pp** |
| block | 0.0% | 11.8% | −11.8pp (accepted — block is structurally hard; not an optimization target this cycle) |

**5 of 6 action classes improved materially.** Aggregate F1 is tied within measurement noise;
the shipped value is primarily the **+4.3pp action accuracy** for downstream match statistics.

## Why F1 didn't lift further

Oracle ceiling analysis (2026-04-20): 219 of 314 GBM-miss GT contacts had a candidate within
±7f (rescuable by any downstream filter). The decoder rescued **22 of those 219 (10%)**.
Remaining 197 have GBM emission probs so low (~0.03–0.10) that no reasonable transition
prior elevates them above skip cost. **The emission quality is the ceiling now, not the
decoder.**

Tested and rejected emission-fix: retraining the GBM without `frames_since_last` dropped F1 to
85.8% even with the decoder. The feature carries non-linear interactions the decoder's
coarse gap-bucketed transitions cannot replace.

## Integration

**Code locations**:
- `analysis/rallycut/tracking/candidate_decoder.py` — decoder module (Viterbi + transitions loader).
- `analysis/rallycut/tracking/contact_detector.py` — wire decoder as optional post-process.
- `analysis/reports/transition_matrix_2026_04_20.json` — learned transition matrix (checked in).

**Feature flag**: `ENABLE_CANDIDATE_DECODER=0|1` (env var). Default 0 for first ship.

**Dependencies**: no new Python packages. No Modal work.

**Rollout**:
1. Land decoder code + feature flag (default off).
2. A/B on production videos for 2 weeks against Phase 0 baseline — monitor per-class F1 and
   action accuracy.
3. Flip default to on if A/B confirms LOO numbers.

## Known limitations

1. **Block F1 = 0%** (vs baseline 11.8%). Decoder never predicts block because MS-TCN++
   seq_probs emit block weakly (training has only 30 block contacts). Block is documented as
   structurally hard (`memory/block_detection.md`); not an optimization target this cycle.
   If this becomes unacceptable, gate via minimum-block-prob floor or swap action emission
   to the learned `ActionTypeClassifier` at block candidates. Cost: maybe +0.5pp block F1.
2. **Serve recall unchanged** at 67% (248/368 vs baseline 237/364). The decoder cannot
   rescue serves that have **no candidate** (74 of 127 serve FNs) — those are
   candidate-generation failures, not classifier-rejection failures. Follow-up work tracked
   in `analysis/reports/serve_candidate_generator_plan_2026_04_20.md`.
3. **10% rescue capture of the oracle ceiling** (22/219). Further lift requires emission
   improvement (richer per-candidate features), not architectural changes.

## Follow-ups (separate tracks)

- **Serve candidate generator** (next session) — 1 day, targets the 74 no-candidate serve
  FNs. Expected +2-3pp aggregate F1. See follow-up plan file.
- **Per-candidate crop-level emission** (next sprint) — 2-3 weeks, expected path to 90-92% F1.
  The crop-level visual featurization (player-bbox + ball-patch sequences around each
  candidate) addresses the emission-quality ceiling. Not shippable without significant ML work.
- **Block F1** remains at 0% — not this cycle.

## Verification

Baseline replication confirmed:
- `scripts/eval_loo_video.py` produces 88.0% F1 / 91.2% action acc on 68 videos.
- `scripts/eval_candidate_decoder.py --skip-penalty 1.0` produces 88.2% / 95.5%.
- `scripts/sweep_candidate_decoder.py` cross-validates 6 config variants.

All artifacts under `analysis/reports/`:
- `candidate_decoder_sweep_2026_04_20.{md,json}` — full sweep results
- `decoder_no_fsl_sweep_2026_04_20.{md,json}` — no-FSL ablation (rejected)
- `transition_analysis_2026_04_20.md` — CRF-0 transition matrix discovery
- `oracle_candidate_coverage_2026_04_20.md` — 93.7% oracle ceiling + per-action breakdown
- `no_candidate_fn_diagnosis_2026_04_20.md` — serve-dominant failure mode

## Canonical baseline correction

Memory's 92.3% F1 / 90.3% action acc was per-rally LOO (leaks across same-video rallies).
**Honest video-level LOO baseline is 88.0% / 91.2%** (re-captured in this investigation).
All numbers in this memo are against that honest baseline. Any comparison to the older
92.3% is wrong.
