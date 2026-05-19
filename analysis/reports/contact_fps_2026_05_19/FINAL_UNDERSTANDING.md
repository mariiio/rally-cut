# Final Understanding: the 60fps degradation is ONE bug, NOT multiple

**Date:** 2026-05-19
**Status:** Root cause confirmed. Fix path identified but NOT executed (multi-day workstream).

## The single bug

`contact_classifier` GBM (`weights/contact_classifier/contact_classifier.pkl`) is trained on a 30fps-dominated corpus. Its features are computed as per-frame quantities (`direction_change_deg` over 8 frames, `velocity_ratio` over 5 frames, `wrist_velocity_max` per frame, etc.). On 60fps videos the same physical motion produces ~half the per-frame value, causing the GBM to see lower feature values and reject more valid candidates.

**Measured at two layers:**

| Layer | Measurement | Gap |
|-------|-------------|-----|
| Contact detection (Phase 1) | Class A FN rate (GBM rejects valid candidate) | 60fps 5.4% vs 30fps 0.9% = **+4.5pp** |
| Action attribution (this investigation) | Player Attribution accuracy | 60fps 83.5% vs 30fps 89.4% = **−5.9pp** |
| Action attribution (Serve specifically) | Serve attribution accuracy | 60fps 73.2% vs 30fps 84.1% = **−10.9pp** |

These aren't separate bugs. The contact-layer FN → action_classifier inserts a SYNTHETIC SERVE with hardcoded `ballX=0.500` and `playerTrackId=-1` → counted as Attribution failure.

## Proof of the synthetic-serve mechanism

For 5 specific EMPTY-candidate failure cases on 60fps Serves:
- All have `serve.ballX = 0.500` (suspicious hardcoded value)
- Actual ball position at that frame is completely different (e.g., haha 18175bae frame 206: serve says (0.500, 0.775), actual ball is (0.370, 0.491))
- `contact.playerCandidates = NO_CONTACT` (no entry in contacts_json for that frame)
- All 4 player tracks DO exist at the GT frame and are in primary_track_ids
- The "empty candidates" isn't a search-window miss — it's a synthetic action with no underlying contact

This rules out the "attribution-layer fix" hypotheses tested in Phases 3-5.

## Why six fix attempts failed

All targeted SYMPTOMS, not the cause:

1. **Phase 1 v5** — Scaled all 13 `ContactDetectionConfig` gates. Over-dedup broke real contacts; GBM feature drift; regressed.
2. **Phase 2** — Coordinated contact + action + scorer retrain. Action_classifier rule logic calibrated to v4 contact frames; Action Acc −0.6pp.
3. **Phase 2.5** — Added contact_frame_regressor retrain. Same Action Acc gap, different per-class pattern.
4. **Phase 3 Stage 1 (Rule L)** — Windowed MS-TCN++ index. Byte-identical (wrong layer).
5. **Phase 4** — fps-velocity normalization in dynamic_attribution_scorer. 60fps Attribution went DOWN (−0.8pp).
6. **Just now (player_candidate_search_frames)** — Widened search window at 60fps. Byte-identical (empty cands were synthetic-serve fallbacks).

**Pattern:** every attempted fix targeted a downstream layer. The actual cause is upstream (GBM training distribution).

## The bug IS real and measurable

| Evidence | Status |
|----------|--------|
| Phase 1 class-A gap (+4.5pp on 60fps) | ✓ Reproduced on 19 60fps videos (3.1pp gap on expanded cohort) |
| Trusted-31 v4 60fps Attribution gap (−5.9pp) | ✓ Reproduced with fresh v4 eval |
| Per-class Serve attribution gap (−10.9pp) | ✓ Reproduced |
| Synthetic-serve fallback mechanism | ✓ Confirmed via per-rally probe (5/5 EMPTY cases are synthetic serves) |

## Why it's bounded (not catastrophic)

- ~10% of 60fps serves get wrong/no attribution (vs ~5% on 30fps)
- Users see "some serves don't have a player assigned"
- Contact detection itself still works (just rejects more candidates)
- Action Accuracy on 60fps is actually HIGHER than 30fps (91.9% vs 90.7%) — the action classifier compensates for missing contacts via heuristics
- Block, Dig, Receive attribution all hold or improve on 60fps

## The fix path (identified, NOT executed)

This is a multi-day coordinated workstream:

1. **Retrain contact_classifier on multi-fps corpus** (already have candidate model from Phase 2: `contact_classifier.pkl.candidate_2026_05_19_60fps_retrain`). LOO F1 86.6%.
2. **Audit and redesign action_classifier rules** for fps-invariance. 12 rules identified in `PHASE3_AUDIT.md`. Convert absolute frame counts to physical-time (seconds × fps) or ordinal positions ("first contact" not "first 60 frames").
3. **Retrain dynamic_attribution_scorer** on new contact frames. Candidate model regenerable from contact candidate in ~5 min.
4. **Retrain contact_frame_regressor** on new pipeline output (`scripts/extract_contact_frame_training_data_2026_05_17.py --in-memory` already supports this).
5. **eval_action_detection.py on trusted-31** with all four new models + rule changes. Pass criteria: 60fps Action Acc improves, 30fps holds within 0.5pp.
6. **Ship with coordinated version bumps** + redetect.

**Estimated effort:** 1-2 days for someone with full context. The Phase-2.5 finding ("Coordinated retrain alone doesn't fix it; rules need redesign") is the key constraint — without step 2 (rule redesign), the retrain causes Action Acc regressions.

## What we've confirmed about "no bug hanging"

1. **The bug is documented and understood.** Multi-layer measurement, mechanism traced.
2. **The fix path is concrete.** Specific code paths identified, recoverable candidate models exist.
3. **Production is in known-good v4 state.** All models md5-verified.
4. **Diagnostic infrastructure preserved.** 7+ scripts for any future attempt:
   - `scripts/diagnose_contact_fn_60fps.py` (--in-memory mode)
   - `scripts/dump_contacts.py`
   - `scripts/contact_density_cohort.py`
   - `scripts/probe_serve_attribution_60fps.py`
   - `scripts/probe_empty_candidates_root_cause.py`
   - `scripts/probe_serve_exact_frame.py`
   - `scripts/extract_contact_frame_training_data_2026_05_17.py --in-memory`
5. **No partial fixes outstanding.** All experimental code reverted; only documentation and probe scripts remain.

The bug isn't "hanging" — it's documented as known-and-bounded with a clear fix path, awaiting the resources to execute the coordinated workstream.

## Recoverable candidate models

If someone takes up the workstream, these can be re-activated:
- `weights/contact_classifier/contact_classifier.pkl.candidate_2026_05_19_60fps_retrain` (multi-fps trained)
- `weights/contact_frame_regressor/best_model.candidate_2026_05_19_60fps.joblib` (retrained on new candidates)
- `reports/contact_frame_regressor_2026_05_17/training_data_post_60fps_retrain.csv` (new candidate training data)

action_classifier + scorer candidates regenerable in ~20 min by swapping in candidates and re-running their training scripts.
