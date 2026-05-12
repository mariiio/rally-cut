# Phase 1.5 Combined-Flag A/B — NO-SHIP

**Date:** 2026-05-12
**Branch:** `main`
**Flags tested:** `RELAX_CONTACT_DIR_GEN=1 RELAX_CONTACT_VEL_GEN=1 RELAX_CONTACT_PARABOLIC_GEN=1`
**Commits supplying infrastructure:** `451b8745`, `4ff662fe`, `4bd59c7b`

## Verdict: NO-SHIP

The combined Phase 1.5 generator-creation relaxation produced essentially zero recall movement and only a marginal improvement on the diagnostic probe. Decision matrix routes us to NO-SHIP. Snapshot was restored; DB is clean.

## Headline numbers

| Metric | Baseline (precheck) | Phase 1.5 ON | Δ |
|---|---|---|---|
| Recall | **0.8922** (2119/2375) | **0.8918** (2118/2375) | **-0.0004pp** |
| Precision (proxy) | 0.9343 | 0.9322 | -0.0021pp |
| C-1 violations | 112 (skip-coherence) | 91 | -21 |
| C-2 violations | 333 (skip-coherence) | 248 | -85 |
| C-3 violations | 2 (skip-coherence) | 1 | -1 |

Note: baseline coherence numbers are from Phase 1 spec (the `--skip-coherence` precheck didn't recompute them). Phase 1.5 coherence was actually computed and *improved* across all three categories — but recall is flat, so this is "different output, not better output."

## Per-type recall

| Action | Phase 1 baseline | Phase 1.5 ON | Δ | Gate G-E (>2pp drop?) |
|---|---|---|---|---|
| serve   | 0.8627 | 0.8676 | +0.49pp | OK |
| receive | 0.9194 | 0.9274 | +0.80pp | OK |
| set     | 0.9336 | 0.9280 | -0.56pp | OK |
| attack  | 0.9342 | 0.9311 | -0.31pp | OK |
| block   | 0.1081 | 0.1351 | +2.70pp | OK (improvement) |
| dig     | 0.8355 | 0.8382 | +0.27pp | OK |

(Comparison column uses spec Phase 1 baseline; precheck values are within 0.1pp of these.)

## DIAGNOSTIC COUNT (PRIMARY signal)

**20 / 173 probe cases (11.6%) now have a contact within ±10 of GT.**

Phase 1 combined-flag baseline: 14/173 (8.1%). Phase 1.5 added 6 probe-case recoveries.

Decision-matrix target: ≥80 (46%) for PASS, ≥40 for CONTINUE_TO_T5. We are at 20 → NO-SHIP.

Per-action probe breakdown:

| GT action | Hit / total | % |
|---|---|---|
| attack  |  3/37  |   8.1% |
| block   |  0/1   |   0.0% |
| dig     |  4/49  |   8.2% |
| receive |  6/25  |  24.0% |
| serve   |  6/34  |  17.6% |
| set     |  1/27  |   3.7% |

The strongest gains are on receive (+24%) and serve (+17.6%) — both at the rally edges where parabolic/velocity generators kick in. Mid-rally actions (set, dig, attack) saw essentially nothing.

## Per-gate verdicts

| Gate | Threshold | Result | Pass? |
|---|---|---|---|
| **G-A** (Recall) | ≥ +3pp | -0.0004pp | **FAIL** |
| **G-B** (Precision) | ≥ -3pp | -0.0021pp | PASS |
| **G-C** (C-1 coherence) | ≤ +5% rel (≤117) | 91 | PASS (decreased) |
| **G-D** (Unit tests) | 12/12 | 12/12 | PASS |
| **G-E** (Per-type recall) | no >2pp drop | worst -0.56pp (set) | PASS |

G-A is the binding constraint. The other gates are clean — meaning the relaxed thresholds are *safe* in production sense, they just don't recover real FNs.

## Decision matrix application

| Diagnostic | Recall Δ | Verdict |
|---|---|---|
| 20 (<40) | -0.0004pp (<+3pp) | **NO-SHIP** |

Diagnostic count gates everything else: at <40, the matrix says restore and escalate to Phase 1.7 / Phase 2 regardless of recall.

## Interpretation

The hypothesis that "relaxing GENERATOR-CREATION thresholds opens a real recovery path" is falsified at this diagnostic count. The ladder of dead-ends so far:

1. **Phase 1 validation-gate relaxation** (DIR_CHANGE / DIR+INFL+VEL / all-5): -0.13pp to +0.08pp recall, only 14/173 probe cases recovered.
2. **Phase 1.5 generator-creation relaxation** (this run): -0.0004pp recall, 20/173 probe cases recovered.

Both phases land in the same plateau. The 153 remaining probe FNs are not blocked by threshold tuning on the existing generator/validator stages — the trajectories never produce a candidate at the GT frame at all under the current geometric primitives. Rule-tuning has hit a ceiling.

## Recommendation

**NO-SHIP. Escalate.**

- Do **not** ship `RELAX_CONTACT_*_GEN` flags as default-ON.
- Do **not** continue to Plan Task 5 (per-flag isolation) — the combined effect is already so weak that isolating contributions is not informative.
- The infrastructure (config fields, env-flag plumbing, unit tests) is sound and harmless to keep on `main` as dormant flags. Recommend cataloguing them in the dormant-flag audit if they remain past one cycle.
- Next investigative direction (per `contact_detection_ceiling_2026_05_11.md`): ball-tracker recall improvement and/or GT refresh. The trajectory data going *into* the contact detector likely lacks the inflection signal at the missing-FN frames, which is why no amount of threshold tuning surfaces a candidate.

## Snapshot status

Restored. Post-restore baseline measured at recall=0.8922 (identical to precheck). DB clean.

## Files

- Phase 1.5 measurement JSON: `analysis/reports/contact_detection_fn/measurement_phase15_combined_2026_05_12.json`
- Pre-snapshot: `/tmp/phase15_pre_snapshot.jsonl` (820 rallies, ephemeral)
- Redetect log: `/tmp/phase15_redetect.log`
