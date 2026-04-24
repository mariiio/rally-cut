# Parallel Viterbi decoder (as `detect_contacts` replacement) — NO-GO

**Date:** 2026-04-24
**Status:** Workstream closed. Code removed (see commit log).
**Reason:** Does not pass pre-registered ship gates in production-realistic measurement. The originally-reported +3.5pp Action Acc lift was an eval-methodology artifact.

---

## Summary

The parallel Viterbi decoder was investigated as a replacement for `detect_contacts` in the production action-classification path. Initial validation via `eval_candidate_decoder.py` showed +3.5pp Action Acc / +0.4pp Contact F1. A code-review-driven production smoke test revealed that this lift was achieved only when the eval passed GT contact frames into the trainer's `extract_candidate_features` — a proxy unavailable in production.

Closing the eval-vs-production methodology gap (two-pass scheme mirroring legacy `detect_contacts`'s per-iteration `prev_accepted_frame` update) drops Contact F1 to 86.7% (**−3.4pp below canonical 90.1%**) while preserving the +5.2pp Action Acc lift — a trade that fails the pre-registered gates.

## Production-realistic 68-fold LOO (`--include-synthetic`, MS-TCN++ v5 + GBM v5)

| Metric | Baseline | Decoder (two-pass, production-real) | Δ | Gate | Result |
|---|---:|---:|---:|---|---|
| Contact F1 | 90.1% | 86.7% | **−3.4pp** | ≥ −1.0pp | ❌ FAIL |
| Action Acc | 90.6% | 95.8% | **+5.2pp** | ≥ +2.5pp | ✅ PASS |
| serve F1 | 78.6% | 82.7% | +4.1pp | ≥ −1.5pp | ✅ PASS |
| receive F1 | 81.9% | 86.3% | +4.4pp | ≥ −1.5pp | ✅ PASS |
| dig F1 | 73.0% | 71.7% | −1.3pp | ≥ −1.5pp | ✅ PASS |
| block F1 | 9.3% | 14.3% | +5.0pp | exempt | ✅ bonus |
| FPs | 165 | 215 | **+50** | ≤ baseline | ❌ FAIL |

**3/5 gates fail** — Contact F1 regression and FP budget overshoot.

## Why the original measurement was misleading

The trainer's `extract_candidate_features` uses `gt_frames` as a proxy for "previously-accepted contacts" when computing `frames_since_last`. With `gt_frames=None` (production), it treats every candidate as accepted, so `frames_since_last` is always small, and the GBM systematically penalizes candidates. Result: production-mode decoder produces **−37% actions** vs legacy on a 10-rally side-by-side smoke.

With `gt_frames = list of GT contacts` (eval-only), the feature matches what a perfect detector would have seen, which is an optimistic upper bound — not a production proxy. That's what `eval_candidate_decoder.py` was measuring.

The two-pass scheme closes this gap by first running the GBM with the broken feature, collecting candidates with prob ≥ threshold as the "accepted" proxy, then re-extracting and re-running. This matches legacy production semantics where `prev_accepted_frame` tracks GBM-accepted contacts online.

## Durable findings from this workstream

1. **The `extract_candidate_features(gt_frames=None)` semantics are wrong for production** — `frames_since_last` treats every candidate as accepted. This also affects the existing label-only overlay (`decoder_runtime.run_decoder_over_rally` → `run_decoder_for_production` → currently shipping at +2.64pp Action Acc via Task 5 overlay). **Follow-up attempt:** a two-pass fix was tested in `decoder_runtime` (Pass 1 identifies GBM-threshold-accepted candidates → Pass 2 re-extracts features with that set as the `frames_since_last` proxy). 68-fold LOO with `--include-synthetic` showed:

   | Metric | Baseline (broken semantics) | Two-pass fix | Δ |
   |---|---:|---:|---:|
   | Contact F1 | 89.8% | 89.8% | 0.0pp (overlay doesn't touch contacts) |
   | Action Acc | 91.3% | 90.8% | **−0.5pp** |

   The fix was reverted. **Root diagnosis:** training/inference mismatch, not a production-only bug. The GBM was trained with `gt_frames=[GT contacts]` (see `scripts/train_contact_classifier.main()` line 538). Production inference with `gt_frames=None` produces DIFFERENT `frames_since_last` distributions than training, and "fixing" to correct semantics moves inference into a region the model isn't calibrated for. The observed +2.64pp Action Acc from Task 4's overlay ship is real — the model has co-adapted to the production distribution despite the training mismatch.

   **Proper fix requires GBM retraining, not production-only changes.** Two viable paths:
   - **(a) Retrain GBM with `gt_frames=None`** (cheapest, ~1 day training + LOO A/B). Training now matches production directly. Expected to recover the +0.5pp that the two-pass couldn't get without retraining.
   - **(b) Retrain GBM with GBM-threshold-accepted proxy** at training time (requires a pre-pass in the training harness). More faithful to production semantics; same order of effort.

   Do not attempt production-only semantic "fixes" to this issue without retraining. The measurement result showed the cleanest possible semantic fix regresses; any subsequent production-only change (including a full Option B refactor of the feature builder) would reproduce the same regression architecturally.
2. **Eval-methodology rigor.** The `_eval_gt_frames` backdoor was intended as an eval shortcut but silently overstated production behavior. Future decoder-integration experiments need production-realistic semantics BEFORE ship decisions, not after.
3. **The decoder's Viterbi grammar + transition matrix are genuinely useful for action labeling.** The +5.2pp Action Acc lift is real; the regression is on the contact-emission side. The existing label-only overlay already captures +2.64pp of this lift safely. Enhancing it (option 2 in the decision memo) is a potential follow-up.

## Durable work kept from the workstream

Code that survives the closure (all byte-identical-guarded or net-positive independent of the decoder):

- **`_prepare_candidates` refactor** (`analysis/rallycut/tracking/contact_detector.py`): 280-line helper extracted from `detect_contacts`. Snapshot-test-locked, enables future reuse without duplication.
- **Snapshot test + fixtures** (`analysis/tests/integration/test_detect_contacts_snapshot.py`, `tests/fixtures/detect_contacts_snapshot/`): guards `detect_contacts` byte-identity. Useful for any future refactor.
- **`team_label` promoted public** (`analysis/rallycut/tracking/action_classifier.py`): pure cleanup, backward-compat alias preserved.
- **Modal `scripts/` bundling** (`modal_app.py`, `modal_tracking.py`): fixes a latent import-crash in the existing `decoder_runtime.run_decoder_over_rally` when a trained classifier lands on Modal. Unrelated to this workstream but surfaced by it.
- **Two pre-existing CI test failures fixed** (`test_segment_merging`, `test_gap_merging`): full suite now green at 1096/1096.

## What was removed

- `detect_contacts_via_decoder` function
- `_features_to_classifier_matrix` helper
- `Contact.decoder_action` + `Contact.is_synthetic` fields
- `analysis/rallycut/tracking/decoder_actions.py` (builder module)
- `analysis/tests/unit/test_detect_contacts_via_decoder.py` (5 tests)
- `USE_PARALLEL_DECODER` flag wiring in `analyze.py` + `track_player.py`
- `--use-decoder` / `--decoder-skip-penalty` flags in `eval_loo_video.py`
- `analysis/scripts/decoder_smoke_2026_04_24.py` (smoke harness)
- The ship plan `docs/superpowers/plans/2026-04-24-parallel-decoder-ship.md`
- Intermediate A/B reports (Phase 3 / Phase 4 / smoke)

## Don't retry without new evidence

- Don't re-measure with `gt_frames=<GT contacts>` — that methodology is known-biased.
- Don't try to tune skip_penalty / transition matrix to recover F1 — the architecture trades F1 for Action Acc; hyperparameters don't fix that.
- Don't re-open this workstream without a production-realistic eval that shows Contact F1 ≥ 89% AND Action Acc ≥ +2.5pp.

## What's actually worth doing next (from this workstream's evidence)

1. ~~Audit the existing label-only overlay~~ **DONE 2026-04-24.** See finding §1 above — the two-pass fix was tested and reverted (−0.5pp Action Acc regression). The proper fix requires GBM retraining. Do NOT retry production-only fixes.
2. **GBM retraining with production-matching `frames_since_last` semantics** (~1 day) — cheapest path to a real overlay Action Acc lift. Train with `gt_frames=None` so training matches production inference directly. Expected recovery: some fraction of the missing +0.5-1.5pp that the two-pass couldn't capture without retraining.
3. **If more Action Acc lift is wanted without retraining,** enhance the existing label-only overlay with richer Viterbi grammar + transitions (separate from the `frames_since_last` issue). F1-safe by construction (the overlay only relabels accepted contacts). ~2 days. Lift ceiling bounded by the training/inference mismatch above — worth measuring but won't fully unlock the decoder's theoretical upper bound.

None of these paths are blocking. Production baseline with the shipped +2.64pp overlay is already a meaningful win.

## Known acceptable tech debt

- `decoder_runtime.run_decoder_over_rally` imports `extract_candidate_features` from `scripts/train_contact_classifier.py`. This is a layering violation (production code importing from scripts). **Accepted as debt** because:
  - Modal images now bundle `scripts/` (fixes any deployment-time crash).
  - Removing the import requires extracting the feature builder into `rallycut/`, which is ~150 lines of careful refactor.
  - Per finding §1, the refactor alone won't improve overlay Action Acc — requires GBM retraining.
  - Fix naturally follows from path (2) above if that work happens.
