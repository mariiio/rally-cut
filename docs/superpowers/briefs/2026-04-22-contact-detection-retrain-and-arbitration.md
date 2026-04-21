# Contact Detection — MS-TCN++ retrain + meta-classifier arbitration

**For:** fresh-session CV/ML engineer picking up after the 2026-04-21 review.
**Prior brief:** `docs/superpowers/briefs/2026-04-21-contact-detection-full-review.md` (6-phase diagnostic review, completed).
**Commits from prior session:** `5b0a1a5` (freshness infra), `1aff8f5` (diagnostic scripts), `0d7cb32` (trainer helper), `82883dd` (review reports), `a0dd13a` (Phase 6 v2-update).

## TL;DR

1. MS-TCN++ is 16 days stale against at least 6.6% new GT (24 rallies added post-train) + unknown in-place GT edits. Retrain.
2. The "GBM contact + MS-TCN++" architecture forces unprincipled hierarchy — GBM vetoes MS-TCN++ even when MS-TCN++ is more confident. Design a meta-classifier arbitration stage (Option B in prior session's analysis).
3. `scripts/train_production_sequence.py` has a 27-dim-GBM mismatch vs current 26-dim production — audit before running.

**Target: clear path to 89.5-90.5% F1 from current v2 baseline of 88.31%.** 92% target requires Path D (WASB retrain), separate workstream.

## Ground truth you can trust

- **Current production F1: 88.31% / Action Acc 93.59%** on 68-fold LOO (v2 corpus at `analysis/outputs/action_errors/corpus_eval_reconciled_v2.jsonl`). Canary-fingerprinted; verify before analyzing.
- **Non-block FN total: 225.** Category distribution in `analysis/reports/contact_fn_phase4_counts_2026_04_21.md`.
- **131 FNs are classifier-rejection inside `detect_contacts`** — largest single mechanism, confirmed by `outputs/cat7_mechanism_trace_v2.log` (77 of 78 Cat 7 are `loop_rejected`, not dedup).
- **49 FNs are WASB tracker-recall** (Cat 6 + Cat 5b) — confirmed via `outputs/wasb_gap_inspection_2026_04_21.jsonl` (19/34 Cat 6 have ZERO WASB detections in GT ± 5f). **No classifier change fixes these.**

**Do NOT use:** `corpus_eval_reconciled.jsonl` (pre-rebuild, stale), `corpus_eval_reconciled_v3.jsonl` (Fix I-A+H-B failure), `corpus_eval_reconciled_v4.jsonl` (Fix R1+R2 failure). All failed gates.

## Prerequisite infrastructure (already built)

- **Freshness pre-flight.** `rallycut.evaluation.corpus_freshness.verify_corpus_fresh()` re-runs the canary fold and compares a fingerprint. Every analysis script that reads a corpus should call this. `scripts/phase4_categorize_fns.py` already does. ~1 min overhead, catches stale corpus artifacts.
- **Canary `_meta` header.** `scripts/build_eval_reconciled_corpus.py` writes a `_meta` header line on every corpus it produces. Will write for v5 automatically after retrain.
- **Six reusable diagnostic scripts:** `trace_dedup_winners.py`, `trace_cat7_mechanism.py`, `inspect_wasb_gaps.py`, `decompose_gbm_rejections.py`, `sample_phase2_fns.py`, `diagnose_fn_stage_attribution.py`. Covered in `1aff8f5`.
- **14 unit tests** for freshness module pass.

## Day 1 checklist (verify before any ML work)

1. **Read the four feedback memories** (in `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/`):
   - `feedback_corpus_freshness.md` — canary-fingerprint verification is mandatory
   - `feedback_small_sample_probes.md` — small probes don't predict 68-fold LOO (learned the hard way this session)
   - `feedback_unbuffered_output.md` — always `PYTHONUNBUFFERED=1 python -u` for long scripts
   - `feedback_no_truncate_output.md` — never pipe through `head`/`tail -N`
2. **Run the freshness check on v2.** `cd analysis && uv run python -c "from rallycut.evaluation.corpus_freshness import verify_corpus_fresh, iter_errors; from scripts.build_eval_reconciled_corpus import reproduce_single_fold; from pathlib import Path; verify_corpus_fresh(Path('outputs/action_errors/corpus_eval_reconciled_v2.jsonl'), reproduce_canary_fn=reproduce_single_fold)"`. Must pass. ~1 min.
3. **Sanity check the MS-TCN++ training script.** See § "Retrain script audit" below.
4. **Confirm GT stability.** Nobody editing GT mid-retrain. Check `SELECT MAX(created_at) FROM player_tracks WHERE action_ground_truth_json IS NOT NULL` shouldn't advance between you starting retrain and it finishing.

## Primary objective — MS-TCN++ retrain + v5 baseline

### Retrain script audit (BEFORE running)

`analysis/scripts/train_production_sequence.py` has a dual problem:
- It trains MS-TCN++ (what we want) — good.
- It ALSO retrains a 27-dim GBM (20 trajectory + 7 per-class seq probs) — but production uses a 26-dim GBM (25 trajectory + `seq_max_nonbg` scalar), per the 2026-04-07 removal of zero-importance per-class seq features.

If you run it as-is, it overwrites `weights/contact_classifier/contact_classifier.pkl` with a 27-dim model. At inference, `ContactClassifier.predict`'s pad-to-size logic papers over the dim mismatch BUT the feature at index 25 in training is `seq_p_serve` (or similar), while index 25 at inference is `seq_max_nonbg`. **Silent feature-index misalignment → large silent regression.**

**Fix options (pick one):**

1. **Fork the trainer to MS-TCN++-only.** Copy `train_production_sequence.py` → `train_mstcn_only.py`. Strip the GBM section (`train_contact_gbm` call + GBM save). Keep only MS-TCN++ training + save. ~30 min edit. Safe and narrow.
2. **Fix the trainer to output a 26-dim GBM.** Update the GBM-training half to use `scripts/train_contact_classifier.py::extract_candidate_features` (current semantics) + saves at production path. ~1 hour edit. Bigger change but keeps one trainer authoritative.
3. **Run as-is but delete the 27-dim GBM output and retrain GBM separately.** Most error-prone; not recommended.

**Recommendation: option 1.** Smallest possible change, biggest safety.

### Execution

1. Verify MS-TCN++ training workflow runs with a smoke test (1-2 epochs, compare loss trajectory to expected).
2. Run full retrain (60 epochs default, ~2-4 hours on a reasonable CPU; GPU would be faster but CPU is sufficient).
3. Save ONLY MS-TCN++ weights to `weights/sequence_action/ms_tcn_production.pt`. Do not overwrite the GBM yet.
4. Smoke-check new model: inference on 3 rallies, verify shape `(7, T)` output, non-bg probabilities look reasonable.

### Retrain the GBM on current features

Once MS-TCN++ is retrained, the GBM's `seq_max_nonbg` feature reads slightly different values (MS-TCN++ output shifted). Retrain GBM via the production training loop (whatever `eval_loo_video._train_fold` uses for LOO is already correct — but for production save, use the equivalent all-data trainer). ~30 seconds CPU. Updates `weights/contact_classifier/contact_classifier.pkl` with a 26-dim GBM matching current inference.

### Build v5 corpus

`cd analysis && PYTHONUNBUFFERED=1 uv run python -u scripts/build_eval_reconciled_corpus.py --out outputs/action_errors/corpus_eval_reconciled_v5.jsonl 2>&1 | tee outputs/action_errors/corpus_v5_rebuild.log`

~20 min. Canary fingerprint updates automatically. Compare F1/Action Acc to v2.

**Pre-registered gate for retrain-alone:**
- F1 Δ ≥ 0pp (no regression)
- Action Acc Δ ≥ +0.3pp (target for retrain gain)
- No fold regresses > 0.8pp F1
- Canary fingerprint changes (expected — new model)

If retrain alone passes the gate, ship as v5 baseline and proceed to Option B. If it regresses, investigate before proceeding.

### Update Phase 4 categorization on v5

`uv run python scripts/phase4_categorize_fns.py` (after updating `CORPUS_PATH` in that script to point at v5). Compare category distribution to v2. Expect Cat 2 / Cat 7 to shrink if MS-TCN++ predictions sharpened.

## Secondary objective — Option B meta-classifier arbitration

### Design

Add a new stage AFTER `detect_contacts`'s per-candidate GBM call that arbitrates between GBM's `accept/reject` and MS-TCN++'s per-class endorsement:

**New module:** `rallycut/tracking/contact_arbitrator.py`

**Inputs (10-12 scalars per candidate):**
- `gbm_contact_prob` (from current 26-dim GBM)
- `seq_bg_prob`, `seq_serve_prob`, `seq_receive_prob`, `seq_set_prob`, `seq_attack_prob`, `seq_dig_prob`, `seq_block_prob` (MS-TCN++ per-class at candidate frame, max in ±5f window)
- `player_distance_capped = min(player_distance, 1.0)`
- `player_missing = 1.0 if player_distance==inf else 0.0`
- `ball_gap_frames` (from ball tracker output)
- `direction_change_deg` (raw feature)

**Model:** `sklearn.ensemble.GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)`. Small, tested, auditable.

**Training:** LOO-per-video, same methodology as current GBM. Positive label = candidate within ±5f of GT contact. Threshold 0.30.

**Integration:** in `detect_contacts` loop, after the current GBM's predict, call `arbitrator.predict` with the 10-12 scalars. Replace `is_validated` assignment with arbitrator's decision.

### Pre-registered gates (STRICT — this is the big swing)

- F1 Δ ≥ +0.5pp vs v5 post-retrain
- FP increase ≤ +20 absolute
- No fold regresses > 0.8pp F1
- Action Acc Δ ≥ −0.3pp (near-neutral OK given arbitrator focuses on contact accept/reject)
- Canary fingerprint changes (expected — arbitrator changes accept decisions)

### Expected outcome

If the diagnostic's framing is right (classifier rejects cases MS-TCN++ confidently handles), arbitrator should recover 15-40 FNs from the 131-case classifier-rejection pool without adding FPs (because MS-TCN++ distinguishes real contacts from near-contact noise better than the GBM alone).

**NO-GO conditions (respect these):**
- Any fold F1 regression > 0.8pp → hard fail
- FP increase > 20 → hard fail
- Action Acc regression > 0.3pp → hard fail (would suggest arbitrator accepts too many mis-classed contacts)

**If arbitrator passes:** ship as v6, update production. This is a meaningful architectural improvement.
**If arbitrator fails:** we've exhausted the surgical-classifier path. Pivot to Path D (WASB retrain) as the remaining lever.

## Known dead ends — do NOT retry

From the 2026-04-21 review:

1. **Adaptive dedup (Fix A) or confident-dedup-bypass (Fix B).** Only 1 of 225 FNs is actual dedup elimination. Wrong lever. Invalidated by `trace_cat7_mechanism.py` diagnostic.
2. **WASB confidence-threshold lowering (Fix D).** 19/34 Cat 6 FNs have ZERO WASB detections in GT ± 5f. Threshold is not the lever. Invalidated by `inspect_wasb_gaps.py`.
3. **Narrower `direction_change_deg` window (Fix F).** Direction change is top-1 blocker in only 3% of Cat 2 FNs. Invalidated by `decompose_gbm_rejections.py`.
4. **Dropping `frames_since_last` feature (Fix I-A).** Probe suggested +10.8pp F1; full 68-fold LOO showed **−3.31pp F1** with 136 new FNs. The feature is net-positive despite train/inference divergence. Lesson in `feedback_small_sample_probes.md`.
5. **Adding `player_missing` binary feature (Fix H-B).** Only tested combined with I-A; regressed jointly. Could be re-tested alone, but low-leverage vs Option B.
6. **R1 attack→defense dedup rescue + R2 dual-occlusion rescue (narrow rules).** Combined +0.08pp F1 vs +0.10pp gate — near-miss. Rescued 3 FNs but added 5 wrong_actions (rescued contacts got mis-labeled downstream). Patched symptoms, not the hierarchy problem.
7. **Seq-only rescue variants** (e.g., `gbm<0.10 AND seq≥0.95` or any blanket seq threshold). Prior session NO-GO (+17pp candidate F1 → −4.82pp integration F1). Brief §never-do line 53.
8. **Per-candidate crop-head emitter.** Brief §closed 2026-04-20 Task 0 NO-GO. Don't revive without a fresh architectural hypothesis — brief §never-do line 200.

## Parallel workstream (don't block on primary)

### Path D — WASB tracker retrain (~50 FNs at stake)

Cat 6 (31 FNs) + Cat 5b (18 FNs) — WASB genuinely misses the ball frames. Not a threshold / not a classifier problem. Requires training-data augmentation.

**Scope:** 1-2 weeks data curation + ~$10 GPU retrain via `uv run rallycut train wasb-modal`.

**Data to curate:**
- Hand-contact-occlusion clips (Cat 6's "ball briefly hidden by player hand" signature).
- Serve-start late-detection clips (Cat 5b's "WASB detects ball 5-10 frames into the serve").

**NOT this session's primary scope.** Should be its own brief once a priority owner is identified.

## Decision tree — when to pick which path

```
After MS-TCN++ retrain + v5 corpus:
├── F1 lift >= +1pp  →  Ship v5, defer Option B (retrain alone hit the target)
├── F1 lift +0.3-0.9pp  →  Ship v5 as new baseline, build Option B on top
├── F1 lift 0 to +0.3pp  →  Ship v5, build Option B (original plan)
└── F1 regression  →  Investigate. Likely MS-TCN++ config issue, not data issue.

After Option B:
├── Gate passes  →  Ship v6, close workstream. Target F1 target re-evaluated.
├── Gate fails by < 0.2pp  →  Ablate arbitrator features, find the single-feature win, re-A/B.
└── Gate fails by > 0.2pp  →  Option B doesn't work at scale. Pivot entirely to Path D.
```

## Files & locations

**Canonical baseline:** `analysis/outputs/action_errors/corpus_eval_reconciled_v2.jsonl` (F1 88.31%).

**Phase 4 category assignments (225 non-block FNs):** `analysis/outputs/phase4_category_assignments.jsonl`.

**Trace evidence (keep for reference):**
- `analysis/outputs/cat7_mechanism_trace_v2.log` — 77/78 Cat 7 are `loop_rejected`
- `analysis/outputs/wasb_gap_inspection_2026_04_21.jsonl` — Cat 6 ZERO-WASB-detection evidence
- `analysis/outputs/gbm_decomposition_2026_04_21.jsonl` — per-feature attribution for Cat 2
- `analysis/outputs/probe_frames_since_last.log` — 4-probe diagnostic for Fix I/H choice

**Review reports:** `analysis/reports/contact_fn_phase{3,4,5,6}_*.md` + `contact_fn_visual_log_2026_04_21.md`. Phase 6 is canonical ("v2 rewrite").

**Pipeline reference:** `analysis/docs/contact_detection_pipeline.md` (full stage-by-stage reference).

**Diagnostic scripts (reusable):** `analysis/scripts/trace_{dedup_winners,cat7_mechanism}.py`, `inspect_wasb_gaps.py`, `decompose_gbm_rejections.py`, `sample_phase2_fns.py`.

## Open questions for this session to resolve

1. **Does MS-TCN++ retrain alone lift F1 meaningfully?** Expected +0.3-1.0pp Action Acc from 6.6% new GT. Unknown for F1.
2. **Does Option B arbitrator pass the gate?** Open question. If yes, big win. If no, close the classifier-side work and move to Path D.
3. **Does the wrong_action regression pattern from R1/R2 recur under the arbitrator?** The arbitrator decides accept/reject only; downstream action labeling is via MS-TCN++ override + decoder. If arbitrator accepts more contacts but some get wrong action labels, Action Acc can regress. Pre-register Action Acc gate (≥ −0.3pp) to catch this.
4. **How many current FNs flip to TP under retrain alone?** If many → Option B's marginal lift is smaller. If few → Option B is the bigger lever.

## Success criteria for this session

**Minimum acceptable:** v5 corpus built from retrained MS-TCN++ + GBM, canary-fingerprinted, at least matches v2 F1. Review closes with refreshed baseline.

**Target:** v6 with Option B arbitrator, F1 ≥ v5 + 0.5pp, Action Acc stable or improved.

**Stretch:** Ship v6 and scope Path D (WASB retrain) as the next workstream brief.

## What NOT to do (tired-engineer traps)

- Don't run `train_production_sequence.py` without the trainer audit.
- Don't relax pre-registered gates even for "marginal" near-misses.
- Don't propose architectural rework without exhausting Option B's arbitration first.
- Don't revisit Path D, R1/R2, or any Fix in "dead ends" without new evidence.
- Don't skip the freshness pre-flight on corpus v2 before analyzing — 10 min saved is weeks of wrong conclusions risked.

## Appendix: kickoff prompt for this session

```
Read docs/superpowers/briefs/2026-04-22-contact-detection-retrain-and-arbitration.md.

Execute in order:
1. Day 1 checklist (verify freshness, memories, GT stability). Do not skip.
2. Retrain script audit + fork (option 1 in the brief). Confirm safe.
3. MS-TCN++ retrain.
4. v5 corpus build + gate check.
5. Based on retrain outcome, proceed to Option B arbitrator design + implementation + 68-fold LOO A/B.
6. Report v6 result against pre-registered gates.

Pause for review between each step. Honor all pre-registered gates. If a gate
fails, report and stop — do not propose alternatives in the same session.
```
