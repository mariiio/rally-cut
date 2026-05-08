# Phase 3 — Pattern A (bookend-fill complement rescue)

**Date:** 2026-04-24
**Status:** NO-GO on impact. Technical gates pass, substantive lift cosmetic (+0.2pp).
**Scope:** 9 fixtures / 69 rallies / 468 GT actions.
**Artifacts:**
- `scripts/phase3_pattern_a.py` — rescuer
- `scripts/phase3_diagnose_bookend.py` + `_v2.py` — opportunity diagnostics
- `reports/attribution_rebuild/phase3_pattern_a.json` — experiment output
- `reports/attribution_rebuild/bench_phase3_pattern_a.json` — bench diff

## Outcome

| metric | baseline | phase3 | Δ |
|---|---|---|---|
| correct | 205 (43.8%) | 206 (44.0%) | +0.2pp |
| wrong   | 191 (40.8%) | 191 (40.8%) | 0.0pp |
| missing |  72 (15.4%) |  71 (15.2%) | −0.2pp |

Transition matrix: exactly **1 baseline-missing → experiment-correct**. No other cells move.

Per-fixture: only **tata** moves (+0.9pp correct / −0.9pp missing). All 8 other fixtures unchanged.

## Kill-gate check (plan §3.3)

| gate | result |
|---|---|
| rescue precision ≥95% | ✅ 100% (1/1 emitted rescues that collided with GT matched correctly) |
| wrong_rate strictly monotonic-lower | ⚠️ tied (0.0pp) — no wrongs introduced, but no reduction either |
| missing_rate lower than baseline | ✅ −0.2pp |
| no per-fixture wrong_rate regression | ✅ every fixture 0.0pp on wrong |

Technically passes. But the point of the kill gate was to catch rescues that break the no-regression contract — not to certify cosmetic lift. The impact is ~1/20th of the memory-estimated headroom (10-20 rescues).

## Why impact is 1, not 10-20

Pre-registered estimate in memory: "Expected headroom: 10-20 rescues of 71 detection_limit." Reality: 1.

**Structural reason — detection misses are correlated.** Pattern A needs both anchors intact (clean receive/dig before + clean attack/spike after, same pid, gap ≤120f, no opposing touch between, no same-team middle). For a bookend to be fillable, the contact detector must be *selectively* wrong only on the middle touch — missing the set but catching the receive and the attack.

Diagnostic on the 15 missing GT sets (the only target class for Pattern A):

| per-missing-set surroundings | count |
|---|---|
| no anchors at all (both receive/dig before AND attack/spike after are missing or mis-attributed) | 9 |
| no receive/dig anchor before | 2 |
| no attack/spike anchor after | 1 |
| 2-touch shape (X≠Y same team) | 1 |
| gap > 120 frames | 1 |
| side mismatch / other | 1 |
| **clean bookend exists and fires** | **1** |

When the detector misses one contact, it typically misses neighbors too. Rallies with broken set-frames also have broken receive/attack frames. The 71 `detection_limit` bucket from Phase 1.5 is not a clean rescue pool — it's a cluster pool. Rescue patterns that depend on intact anchors only help the uncorrelated-miss minority.

## Headroom by action type (re-scoped)

Pattern A targets set-class misses only. Missing actions by type:

| type | count | Pattern A-addressable? |
|---|---|---|
| serve | 22 | no — needs Pattern B (server-occluded) |
| set | 15 | yes — 1 actual rescue |
| attack | 14 | no — terminal action, no "middle" shape |
| dig | 11 | no — receive-side inbound, not middle |
| receive | 5 | no |
| block | 5 | no |

So Pattern A's *maximum possible* ceiling on this corpus is 15 (all missing sets rescued), and the detected addressable subset is **1**.

## Emitted rescues (9 total, 1 matched GT)

All 9 have the shape `{receive|dig}(pid_a) → set(pid_b, inferred) → {attack|spike}(pid_a)`. Of 9:

- 1 aligned with a missing GT set within ±10f and inferred the correct pid → **+1 correct**.
- 8 were phantom — emitted in rally gaps where no GT action existed → **no-op on the bench**.

Per-rally:
- tata `d934f57a`: inject `set(pid=4)@289` — matches GT `set(pid=4)@288` ✅ correct
- tata (2 phantom), lala (4 phantom), rere (1 phantom), yeye (1 phantom) — no GT collision, no effect.

The phantom rescues are structurally safe on this corpus (no collision with any GT action), but they add noise to downstream consumers that read pipeline_actions directly (stats, UI). Shipping would wire provenance-tagged synthetic contacts into the stream.

## Decision

**NO-GO.** Recommending not to ship Pattern A rescue code in its current form:

1. +1 correct is within measurement noise — not distinguishable from a single GT labeling error.
2. Adds 9 synthetic pipeline_actions per 69 rallies with provenance `rescuePattern=bookend_fill` that downstream consumers must ignore or re-tag.
3. Not scalable — the structural reason (correlated misses) is architectural, not tunable.

**Keep:**
- `scripts/phase3_pattern_a.py` and `scripts/phase3_diagnose_bookend*.py` as repeatable tooling for future corpus runs. Cheap to re-run when the corpus expands.
- The diagnostic framework (per-missing-action-type surroundings) generalises to Patterns B/C.

**Implications for the Phase 3 plan (§3.2):**

- Pattern A's headroom estimate in the plan was derived from the coarse 71 `detection_limit` partition. The partition doesn't account for correlated misses. Future rescue-pattern design should size headroom by *explicit anchor availability*, not by category counts.
- Pattern B (server-occluded serve, 22 missing serves) needs a parallel diagnostic before coding. Probably also ceiling'd by chain-breakage: if the serve is missing, the receive after is often also misattributed.
- Pattern C (teammate-role check) targets the abstention path, not the missing-action path — orthogonal to Pattern A and still plausibly useful *after* a confidence-gated chooser lands. Phase 2 offline chooser is NO-GO, so Pattern C is de-facto blocked at Phase 3 entry too.

## What to try next (order of leverage)

The session-structural insight stands: **the contact detector is the architectural ceiling** (22% of GT actions have no candidate). Rescue patterns operating on pipeline-output-only cannot breach that ceiling. Options, roughly by leverage:

1. **Contact-detector recall investment** — 4 prior NO-GOs in memory (VideoMAE, crop-head Phase 2, E2E-Spot, contact arbitrator). Do not reopen without a new architectural hypothesis. Pose-based seeding and WASB ball-tracker recall are the named dormant candidates.
2. **Held-out validation of primitive baseline** — label 2-3 more fixtures. Cheap sanity check that the 9-fixture baseline generalises, and tightens corpus confidence for future kill gates.
3. **Relax the nearest-candidate-guard** in `action_classifier.py:2917` with a sweep — flagged in phase2_summary as 10-30 potential gains on serve-chain-declared overrides, but risky. Needs a margin sweep on the locked baseline.

Pattern A and the chooser workstream are closed. Phase 3 in its primitive-rescue form is closed.
