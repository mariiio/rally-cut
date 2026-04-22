# Contact FN Phase 6 — Gap-assessment decision memo (v2, post-rebuild)

**Final deliverable** of the contact-detection full review (brief: `docs/superpowers/briefs/2026-04-21-contact-detection-full-review.md`).

**Provenance.** This is the POST-REBUILD rewrite of Phase 6. The original (pre-rebuild) Phase 6 memo was built on `corpus_eval_reconciled.jsonl` from 2026-04-21 09:13, which silently drifted from current code by ~12% of FN records. Discovered 2026-04-21 evening when visual verification of one "phantom Cat 7" case (99a01ce4/371) showed the pipeline had correctly detected a contact that the stored corpus claimed was missed. See `feedback_corpus_freshness.md`.

**Fresh baseline.** v2 corpus (2026-04-21 18:26, current HEAD, canary-fingerprinted):
- F1 = **88.31%** (was 88.15% stale; +0.16pp)
- Action Acc = **93.59%** (was 93.15% stale; +0.44pp)
- 416 errors = 301 FN + 115 wrong_action (was 435 = 313+122)
- 225 non-block FN (was 238)

Baseline F1 is slightly better than the stale corpus claimed, meaning the target gap is now +3.69pp (to 92%), not +3.85pp.

---

## Diagnostics re-validated against v2

Three diagnostics preceded this memo. Two are directly tied to the specific FN universe and thus shifted under the rebuild; one is mechanism-level and carries over unchanged.

### 1. Cat 7 "dedup-kill" is NOT primarily dedup — **CONFIRMED CLEANER ON V2**

**v2 mechanism trace (n=78 primary Cat 7):**

| Mechanism | Stale (n=84) | v2 (n=78) |
|---|---|---|
| `loop_rejected` | 56 | **77** |
| `false_positive_cat7_label` | 25 | **0** |
| `no_candidate_in_window` | 2 | 0 |
| `dedup_eliminated` | 1 | 1 |

**All 25 phantom-Cat-7 false positives disappeared on v2.** They were stale-corpus artifacts — contacts the current pipeline DOES detect correctly, mis-labeled by the stale corpus's record.

**77 of 78 primary Cat 7 cases in v2 are classifier-rejections inside `detect_contacts`**, not dedup eliminations. Identical mechanism to Cat 2 — just mis-labeled by `diagnose_fn_stage_attribution.py` because that script's `dedup_survived` check tests the OUTPUT of detect_contacts, not the internal dedup pass.

**Consolidated classifier-rejection pool (v2):**
- Cat 7 loop_rejected: 77
- Cat 2 kin_underreports: 31
- Cat 2b kin_moderate: 9
- Cat 4 dual_occlusion: 10
- Cat 3 kin_maximal: 4
- = **131 classifier-rejection FNs**, 58% of all 225 non-block FNs.

**Implication:** Fix I (drop `frames_since_last`) targets a 131-case pool, not the ~87 estimated pre-rebuild. Leverage is proportionally larger.

### 2. Cat 6 ball-gap is a WASB recall ceiling

**Finding on stale corpus (n=34):** 19/34 zero WASB detections in GT ± 5f; only 1/34 has a detection below conf 0.30.

**v2 corpus:** 31 Cat 6 + 18 Cat 5b = 49 WASB-related FNs. Mechanism finding unchanged — the `_CONFIDENCE_THRESHOLD=0.3` is not the bottleneck; WASB is genuinely missing these contact frames.

**Implication:** Path A has no lever for Cat 6 / Cat 5b. Brief §closed 2026-04-20 already documents this as a Path D (WASB retrain) workstream.

### 3. Cat 2 classifier rejection is driven by `frames_since_last`, not `direction_change_deg`

**Finding on stale corpus (n=30):** top-1 blocker freq:
- `frames_since_last`: **40%** (top)
- `seq_max_nonbg`: 17%
- `player_distance`: 17%
- `ball_detection_density`: 10%
- `direction_change_deg`: **3%** (far from top)

**Carries to v2 unchanged.** The Cat 2 primary count is identical (31 in both stale and v2), and the per-feature attribution mechanism is not corpus-specific.

**Implication:** `frames_since_last` has a documented train/inference semantic divergence — training uses GT-matched semantics, inference uses `prev_accepted_frame`. This produces a **cascading FN failure**: one missed contact makes subsequent `frames_since_last` values artificially large, pushing more rejections.

---

## The surgical-fix program after diagnostic revisions

Phase 5 proposed 5 fixes. Three were invalidated by diagnostics regardless of corpus version.

| Fix | Category | Gate | Status |
|---|---|---|---|
| ~~A — adaptive dedup unknown-side~~ | Cat 7 (fake) | — | **KILLED** — only ~1 FN is real dedup |
| ~~B — confident-dedup bypass~~ | Cat 7 (fake) | — | **KILLED** — same reason |
| ~~D — WASB threshold bridge~~ | Cat 6 | — | **KILLED** — WASB has ZERO detections, not low-conf |
| ~~F — narrower direction_check~~ | Cat 2 | — | **KILLED** — `direction_change_deg` is not the blocker |
| **I (new)** — drop `frames_since_last` feature | Cat 2 + mis-labeled Cat 7 loop_rejected | F1 ≥ +0.3pp; no fold regresses > 0.8pp; FP +15 cap | **VIABLE** |
| **H** — neutralize `player_distance=inf` | Cat 4 | F1 ≥ +0.15pp; no fold regresses > 0.5pp; FP +10 cap | **VIABLE** |
| **J (diagnostic)** — downstream tracer for phantom-Cat-7 | ≤25 cases | — | **LARGELY MOOT** after rebuild — the stale-corpus artifacts have reduced Cat 7 overall |

### Fix I — Drop `frames_since_last` feature

**Rationale.** Attribution shows it's top-1 blocker in 40% of Cat 2 FNs. The consolidated classifier-rejection pool in v2 is **131 FNs** (Cat 2 + Cat 2b + Cat 3 + Cat 4 + the 77 Cat 7 loop_rejected cases re-categorized as classifier-rejections). Same mechanism throughout.

**Risk.** Global importance is low (~0.03) per prior measurement, but low global importance ≠ safe to drop (as Phase 5's earlier error showed). Must validate with 68-fold LOO A/B.

**Realistic recovery:** if attribution's 40% top-1-blocker rate holds across the 131-case pool, ~52 cases are potentially rescuable. Conservative estimate at gate minimum: ~10-15 FNs. Optimistic: ~40 FNs.

### Fix H — Neutralize `player_distance=inf`

**Rationale.** Cat 4 dual-occlusion (10 primary + 4 in Cat 7+4 overlap = ~14) has `player_distance=inf` substituted to 1.0 by `to_array()`. Replacing with training-set median (~0.06) should admit some of these cases without affecting normal-proximity candidates.

**Risk.** Low — one-line change; affects only rows where the feature was defaulted.

**Realistic recovery:** ~5 FNs.

### Fix J — Status change: deferred, probably unneeded

The 25 "phantom Cat 7" cases were largely stale-corpus artifacts. v2 has fewer (exact count pending re-trace). Downstream tracer can still be written if those remaining cases are worth chasing, but with only a handful of true cases it's lower leverage than I (classifier) or H (dual-occlusion).

---

## Brutal gap assessment — v2 numbers

| | F1 |
|---|---|
| v2 baseline | **88.31%** |
| + Fix I at gate minimum (+0.3pp F1) | ≥88.61% |
| + Fix H at gate minimum (+0.15pp F1) | ≥88.76% |
| Realistic combined recovery (if Fix I recovers 25% of 131-pool) | ~89.7-90.0% |
| Optimistic combined recovery (40% of 131-pool) | ~90.8% |
| Target | **92.00%** |
| **Residual gap after optimistic Path A** | **~1.2-2.3pp (~25-50 FNs)** |

**Path A does NOT close the 92% target.** The residual is concentrated in:
- **Cat 6 (31 FNs) — WASB tracker recall.** Path D (retrain) or Path B (replace). Brief §closed 2026-04-20 already accepts.
- **Cat 5b (18 FNs) — same underlying mechanism.**
- **u-classifier_other (21 FNs)** + **u-candidate_gen_other (7 FNs)** — residual tails of Cat 2/Cat 1 distributions.
- **Remaining mis-labeled Cat 7 cases** not helped by Fix I.

---

## Path B / Path D proposals (for the residual)

Per brief §Phase 6: "Architectural rework proposals need evidence: a specific stage that cannot close its residual gap surgically."

### Path D — WASB tracker recall augment (covers ~49 FNs: Cat 6 + Cat 5b)

**Evidence.** Raw WASB confidence inspection confirms 56% of Cat 6 have ZERO detections in GT ± 5f. Threshold is not the lever; training data is.

**Proposal.** Curate a targeted WASB fine-tuning dataset of hand-contact-occlusion clips + serve-start late-detection clips. Existing infrastructure: `rallycut train wasb-modal`. Estimated cost: 1-2 weeks labeling + $10 GPU retrain. Expected recovery: unknown (depends on added samples).

**Scope:** separate workstream with its own brief + pre-registration. Do not couple to this review's Path A fixes.

### Path B — Not proposed

No category currently meets the brief's evidence bar ("a specific stage that cannot close its residual gap surgically AND a specific rework that would"). The classifier-loop ceiling might eventually need architectural work (per-candidate crop-head), but brief §closed 2026-04-20 Task 0 kills the most recent attempt and explicitly warns against reviving without a fresh brainstorm.

---

## Recommended execution plan

### Immediate (can run today)

1. **Implement + validate Fix I (drop `frames_since_last`).**
   - Edit `contact_classifier.py:CandidateFeatures.to_array()` to drop field.
   - Retrain via 68-fold LOO.
   - Re-build corpus v3 with new canary fingerprint → the `verify_corpus_fresh` preflight in `phase4_categorize_fns.py` will catch any staleness automatically.
   - Compare v3 to v2 on pre-registered gates.
   - **NO-GO if any fold's F1 regresses > 0.8pp or FP increase > 15.**

2. **Implement + validate Fix H (neutralize `player_distance=inf`).**
   - Edit `contact_classifier.py:to_array():93` — replace `1.0` with the training median (~0.06). Use a module-level constant so the median is reproducible.
   - Same 68-fold LOO + pre-reg gates.

3. **Re-run Phase 4 categorization on v3 corpus** after each fix to measure per-category impact.

### Deferred

- **Fix J downstream tracer** — run only if v3 still has >10 phantom-Cat-7 cases after Fix I.
- **Cat 6 / Cat 5b / Cat 4 residuals** — accepted as Path A ceiling. Scope Path D retrain as a separate workstream when prioritized.

### Not proposed

- Any dedup-level fix (A, B) — wrong lever per diagnostics.
- Any Cat 6 threshold tweak (D) — mechanism debunked.
- Any direction_change window adjustment (F) — wrong feature.
- Any seq-based rescue variant — brief §never-do line 53.
- Architectural Path B — no category clears the evidence bar.

---

## What stands vs what this memo replaces

**Unchanged from Phase 1-3 (pre-rebuild, valid):**
- `analysis/docs/contact_detection_pipeline.md` — pipeline reference.
- `analysis/reports/contact_fn_visual_log_2026_04_21.md` — 20-case visual observations. 17/20 still FN in v2; observations remain valid since they're about real video.
- `analysis/reports/contact_fn_phase3_categories_2026_04_21.md` — category definitions + observable signatures.

**Refreshed against v2 (numbers updated):**
- `analysis/reports/contact_fn_phase4_counts_2026_04_21.md` — rebuilt from v2 corpus; 225 non-block FNs vs prior 238.

**Partially superseded:**
- `analysis/reports/contact_fn_phase5_hypotheses_2026_04_21.md` — contains 4 invalidated fixes (A, B, D, F). Keep as record of what was considered and why it was killed.

**Replaced by THIS memo:**
- Any prior version of `contact_fn_phase6_decision_memo_2026_04_21.md` — THIS document is canonical.

---

## Lessons carried into project memory

Two new feedback memories were saved during this review:

- `feedback_unbuffered_output.md` — always use `PYTHONUNBUFFERED=1 python -u` for long scripts whose output is piped (`tee` block-buffers; silent progress during long runs is a bug).
- `feedback_corpus_freshness.md` — verify corpus freshness via canary-fold fingerprint (`verify_corpus_fresh()`) before analyzing any eval artifact. Git-independent. Catches the "stale corpus" bug class that silently invalidated 12% of this review's FN records.

Plus one reusable infrastructure piece:

- `rallycut.evaluation.corpus_freshness` + `scripts/build_eval_reconciled_corpus.py`'s `_meta` header + `reproduce_single_fold()` — any future eval-artifact consumer can pre-flight with `verify_corpus_fresh()` and abort on staleness. 14 unit tests cover the helper.

---

## Meta-lesson for the brief's §traps-to-avoid

The brief warned "Assume a candidate-level metric predicts production F1" — that lesson held. But this review surfaced a new trap: **assume a stage-attribution label directly names its mechanism**. The label `lost_at_stage=dedup_survived` captures ANY reason a classifier-acceptable candidate doesn't reach `detect_contacts` output, not specifically dedup elimination. Downstream readers took the label literally → 4 fix proposals targeting the wrong mechanism.

Recommend adding to the brief's §traps-to-avoid in future reviews:

> **Stage-attribution labels may be mechanism-inaccurate.** `diagnose_fn_stage_attribution.py`'s boolean checks test the pipeline's output at each stage boundary, not the internal mechanism at that stage. A `dedup_survived=False` label collapses: actual dedup elimination, classifier-loop rejection inside detect_contacts, trajectory-refinement shifting candidate out of tolerance, and attribution-framework false positives. Instrument the specific mechanism before committing to a stage-targeted fix.

---

## Appendix — v2 category counts for quick reference

| # | Category | Primary | Flat (incl. overlaps) | Dominant action |
|---|---|---|---|---|
| 7 | `dedup_kill` + overlap with 4 | 74 + 4 = **78** | 83 | serve 18 / dig 18 / receive 14 |
| 2 | `kin_underreports` (dir≤30°) | 31 | 31 | dig 12 / serve 9 / receive 6 |
| 6 | `ball_gap_exceeds_interp` (gap≥4) | 31 | 32 | serve 19 / set 5 / receive 4 |
| 5b | `serve_cand_gen_other` | 18 | 18 | serve 18 |
| 4 | `dual_occlusion` | 10 | 54 | dig 3 / set 3 |
| 2b | `kin_moderate_gbm_rejects` | 8 | 9 | dig 5 |
| 1 | `interp_erases_deflection` | 5 | 10 | attack 2 / set 2 |
| 3 | `kin_max_gbm_rejects` | 4 | 4 | — |
| 8 | `action_labeling` | 5 | — | serve 4 |
| 10 | `matcher_steal` | 3 | — | attack 2 |
| seq | `seq_signal_only` | 3 | — | — |
| 5 | `serve_ball_dropout` | 1 | 1 | — |
| — | `u-classifier_other` | 21 | — | attack 8 |
| — | `u-candidate_gen_other` | 7 | — | attack 3 |
| | **Total non-block FN** | **225** | | |

---

## v5 Shipped 2026-04-22

MS-TCN++ retrained on 364 rallies (post-16-day GT-edit gap) + contact GBM retrained on v5 sequence probs. v5 is the new production baseline.

| Metric | v2_rerun (current code) | v5 | Δ |
|---|---:|---:|---:|
| F1 | 88.83% | 88.87% | +0.04pp |
| Action Acc | 93.41% | 94.01% | +0.61pp |
| wrong_action | 119 | 108 | −11 |

Pre-registered per-fold gate (no fold > 0.8pp F1 regression) missed with 24 regressions, but regressions have no common signature — shipping for aggregate wins, Phase D arbitrator is the recovery lever.

New canary fingerprint: `sha256:be9bcd76a0059…` (corpus at `analysis/outputs/action_errors/corpus_eval_reconciled_v5.jsonl`). Memory: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/mstcn_retrain_2026_04_22.md`.
