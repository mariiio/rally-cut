# Phase 2 — Coordinated 3-model retrain: NO-SHIP

**Date:** 2026-05-19
**Decision:** No-ship. v4 restored as default across all three models (contact_classifier, action_classifier, dynamic_attribution_scorer).

## What we did

Per the v8→v11 ship pattern (`retrain_action_models_plan_2026_05_18.md`), coordinated retrain of all three models:

1. **contact_classifier** retrained on full current corpus (10,486 candidates / 2,642 positive, 492 LOO folds). LOO F1 = 86.6%. Saved as `contact_classifier.pkl.candidate_2026_05_19_60fps_retrain`.
2. **action_classifier** retrained with new contact frames in the loop (4,856 samples, 486 LOO folds, no-hybrid per v11 audit). LOO F1 = 89.0% (vs v11 baseline 88.9% — essentially unchanged).
3. **dynamic_attribution_scorer** retrained (1245 GT rows, 4407 candidate rows, 6 per-action GBMs).

Then trusted-31 eval A/B (234 rallies, 1445 GT contacts):

## Results

| Metric | v4 baseline | New triple | Δ |
|--------|-------------|------------|---|
| **Contact F1** | 93.9% | 94.2% | **+0.3pp ✅** |
| Contact Recall | 92.8% | 92.9% | +0.1pp |
| Contact Precision | 95.1% | 95.7% | +0.6pp ✅ |
| **Action Accuracy** | **91.0%** | **90.4%** | **−0.6pp ⚠️** |
| Player Attribution | 87.9% | 88.0% | +0.1pp |
| Court-Side | 87.9% | 87.2% | −0.7pp |
| Sanity violations | 31 | 35 | +4 ⚠️ |

**Per-action F1:** Serve +1.2pp ✅, Set +1.0pp ✅, Receive −0.8pp, Attack −0.6pp, **Dig −2.7pp ⚠️**
**Per-action Attribution:** **Block +14.2pp ✅**, Serve +2.2pp ✅, Attack +0.4pp ✅, Set −0.3pp, Receive −0.5pp, Dig −1.4pp

## Verdict against project ship gate

The v8→v11 memo's documented ship criteria: *"0 per-class F1 regressions >1pp, Action Accuracy delta >+2pp."*

- Per-class F1 regressions >1pp: **Dig −2.7pp ❌**
- Action Accuracy delta: **−0.6pp ❌** (need ≥+2pp)

**NO-SHIP** by the project's own gate.

## Why coordinated retrain didn't fix it

The new contact_classifier shifts detected contact frames by a few frames (consistent with the v4 history note "v4 shifts frames ~3f earlier on average"). Even with action_classifier AND scorer retrained on those new frames, the Action Acc still regresses 0.6pp, dominated by Dig confusion.

The implication: `action_classifier.py` contains rule-based logic (action-type heuristics, repair rules, sanity-violation gates) that's calibrated to v4 contact-frame positions. Retraining the GBM portion isn't enough — those rules misfire when frame placement shifts. Specifically:
- Dig-vs-attack disambiguation rules
- Same-side action-repeat detection
- Time-gap sanity gates

These rules would need redesign for the new frame distribution, not just retraining.

## Why this is structural, not a knob

We tried the right architectural fix and it didn't close the gap. This rules out the "simpler" path. Remaining options for a future attempt:

1. **Audit + redesign action_classifier rules** for fps-invariance. Significant rule-design work, hard to estimate.
2. **Phase 3 — physical-time features.** Would shift contact frames even MORE (because feature windows change with fps), so action_classifier rule misfires would get WORSE. Not a path forward unless paired with (1).
3. **Per-fps separate classifiers / per-fps thresholds.** Treat 60fps as a separate model path. Doubles training/maintenance.
4. **Accept current state.** 60fps cohort is degraded by ~+4.5pp class-A FN. Production users with 60fps content see fewer detected contacts. Not catastrophic — system still works.

## Production state (restored)

All three models reverted to v4 baseline. Hashes verified (`md5 -q` matches `.backup_pre_2026_05_19_60fps_coordinated` files). Recoverable candidates:
- `weights/contact_classifier/contact_classifier.pkl.candidate_2026_05_19_60fps_retrain` (preserved)
- action_classifier + scorer candidates were overwritten during baseline restore. **Regenerable** by swapping in the contact_classifier candidate and re-running `train_action_classifier.py` + `train_and_save_dynamic_scorer_2026_05_14.py` (~20 min total).

## What we learned

1. **Phase-1 diagnostic was correct on the mechanism** — GBM rejection IS the 60fps FN driver (Class B = 0 in both cohorts).
2. **Simple GBM retrain partly helps** (60fps class A 5.4 → 3.2) but introduces a coupled −0.6pp Action Acc regression.
3. **Coordinated 3-model retrain doesn't fix the Action Acc regression** — the action_classifier RULES (not just the GBM) are calibrated to v4 contact-frame positions.
4. **The "robust correct" fix isn't a coordinated retrain.** It's a rule-redesign-plus-retrain, which is significantly larger scope.
5. **The v8→v11 ship gate is enforced.** −2.7pp Dig F1 and −0.6pp Action Acc miss the bar; restoration is the disciplined response.

## Recommendation

For now: **accept the current 60fps degradation as a known limitation**, document it, and revisit in a dedicated workstream that includes action_classifier rule audit. The diagnostic scripts (`diagnose_contact_fn_60fps.py`, `dump_contacts.py`, `contact_density_cohort.py`) and this report provide the substrate for any future attempt.

If 60fps support becomes a priority, the next investigation should be:
- Per-rally trace of the Dig misclassifications introduced by the coordinated retrain
- Identify which specific rules in `action_classifier.py` fire differently on the shifted frames
- Decide between rule redesign and per-fps model paths
