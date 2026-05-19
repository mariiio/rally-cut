# Phase 2.5 — contact_frame_regressor retrain: STRUCTURAL FINDING (NO-SHIP)

**Date:** 2026-05-19
**Decision:** No-ship. Production restored to v4 across all 4 models (contact_classifier, contact_frame_regressor, action_classifier, dynamic_attribution_scorer).

## Hypothesis being tested

Phase 2 showed coordinated retrain of `contact_classifier` + `action_classifier` + `dynamic_attribution_scorer` couldn't close the Action Acc gap (−0.6pp, driven by Dig F1 −2.7pp). Phase 2.5 tested whether retraining the **contact_frame_regressor** (the model that snaps the final contact frame after GBM acceptance) on the new pipeline output could pull frames back toward GT-aligned positions where the action_classifier rules expect them.

## Method

1. Swapped candidate contact_classifier (from Phase 1) back to default.
2. Modified `extract_contact_frame_training_data_2026_05_17.py` with `--in-memory` flag to regenerate training data using `detect_contacts` in-memory (new GBM in the loop, no DB writes). 2775 examples extracted.
3. Ran `train_contact_frame_regressor_2026_05_17.py` on the new training data. LOO MAE 1.517 → 1.275 (+42 NET within ±5 of GT).
4. Ran `eval_action_detection.py --only-videos <trusted-31>` with: **NEW contact_classifier + NEW regressor + V4 action_classifier + V4 scorer**.

## Trusted-31 results (234 rallies, 1445 GT contacts)

| Metric | v4 baseline | Phase 2 (all 3 new) | **Phase 2.5 (contact+regressor only)** |
|--------|-------------|---------------------|----------------------------------------|
| Contact F1 | 93.9% | 94.2% | 94.1% (+0.2) |
| Contact Recall | 92.8% | 92.9% | 92.9% |
| Contact Precision | 95.1% | 95.7% | 95.4% |
| **Action Accuracy** | **91.0%** | **90.4%** | **90.4% (−0.6)** ❌ |
| Player Attribution | 87.9% | 88.0% | 87.1% (−0.8) ❌ |
| **Dig F1** | 81.4% | 78.7% (−2.7) | **83.1% (+1.7 ✅)** |
| Serve F1 | 89.5% | 90.7% | 87.8% (−1.7) ❌ |
| Receive F1 | 88.6% | 87.8% | 86.0% (−2.6) ❌ |
| Set F1 | 86.8% | 87.8% | 86.3% (−0.5) |
| Attack F1 | 85.6% | 85.0% | 86.2% (+0.6) |
| Block Attribution | 22.2% | 36.4% | **37.5% (+15.3 ✅)** |

**Hypothesis confirmed at one level, refuted at another:**
- ✅ Regressor retrain DID fix the Dig regression (Phase 2: −2.7pp → Phase 2.5: +1.7pp = +4.4pp swing)
- ❌ But it CREATED new regressions on Serve (−1.7pp) and Receive (−2.6pp)
- ❌ Aggregate Action Accuracy is identical to Phase 2 (−0.6pp vs v4)

The regression just **moved** to different action types.

## The structural finding

We've now run three configurations:

| Configuration | Outcome |
|---------------|---------|
| Phase 1 (gate scaling) | Regressed everything ❌ |
| Phase 2 (all 3 retrained) | Action Acc −0.6pp, Dig F1 −2.7pp ❌ |
| Phase 2.5 (contact + regressor, v4 action/scorer) | Action Acc −0.6pp, Serve/Receive F1 −1.7/−2.6pp ❌ |

**No combination of model retraining closes the Action Acc gap on trusted-31.** Each retrain configuration moves the per-action regression around, but the aggregate Action Accuracy is consistently −0.6pp below v4.

**This is structural, not a knob.** The new contact_classifier produces a frame-distribution shift that the rule-based logic in `action_classifier.py` cannot accommodate even when downstream models are co-retrained. The rules themselves (Rule 3 dedupe, Rule 6 attack→attack→set, time-gap sanity gates, dig-vs-attack disambiguation, repair rules) are calibrated to v4-pipeline-shaped contact-frame distributions and need redesign — not retraining — for the new distribution.

## What this means for the 60fps bug

The 60fps degradation is real and structural:
- **Phase 1 diagnostic proved:** 60fps GT actions are missed 15.1% of the time vs 30fps 6.5% (+8.6pp), with the dominant +4.5pp gap being GBM rejection of valid candidates (class A).
- **Phase 2 proved:** retraining the contact_classifier on a multi-fps corpus partially closes the class-A gap but causes downstream Action Acc regression.
- **Phase 2.5 proved:** retraining the regressor doesn't fix this — it just trades Dig regression for Serve/Receive regression.

The bug exists. We can't fix it with retraining alone. The "proper" fix requires either:

1. **action_classifier rule audit + redesign** (multi-day workstream). Find every rule that depends on frame-position relative-to-event and either make it frame-position-tolerant (use ±N windows instead of exact frames) or use physical-time semantics. Then coordinated retrain. This is the "robust correct" path.

2. **Per-fps separate model paths.** Train and ship 30fps-specific + 60fps-specific contact_classifier/regressor/action_classifier triples. Production routes by input fps. Doubles training and maintenance.

3. **Accept the current limitation.** Document the +4.5pp class-A FN gap on 60fps as a known issue. Revisit when a dedicated workstream is feasible.

## Recoverable artifacts

For any future attempt:
- `weights/contact_classifier/contact_classifier.pkl.candidate_2026_05_19_60fps_retrain`
- `weights/contact_frame_regressor/best_model.candidate_2026_05_19_60fps.joblib`
- `reports/contact_frame_regressor_2026_05_17/training_data_post_60fps_retrain.csv`
- `scripts/extract_contact_frame_training_data_2026_05_17.py` now supports `--in-memory` flag for regenerating training data without DB writes
- `scripts/diagnose_contact_fn_60fps.py` (`--in-memory` mode for A/B against new classifier)
- A/B baselines: `eval_v4_trusted31.txt`, `eval_new_trusted31.txt`, `eval_new_contact_regressor_v4_action_scorer_trusted31.txt`

Regenerable in ~30 min: swap candidate contact_classifier → default, run `extract_contact_frame_training_data_2026_05_17.py --in-memory`, run `train_contact_frame_regressor_2026_05_17.py`.

## Production state

All four production models restored to v4 (`md5 -q` verified against `.backup_pre_2026_05_19_60fps_*` snapshots). DB is in v4 state (restored after Phase 1 v5 rollback).
