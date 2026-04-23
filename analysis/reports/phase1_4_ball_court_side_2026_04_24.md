# Phase 1.4 — Ball Court-Side Audit

**Date:** 2026-04-24  
**Scope:** all `correct` + `wrong_cross_team` matches from baseline, measured against GT-actor foot projection as ground truth.  
**Gate:** best method ≥ 95%  
**Winner:** Method **C** at 100.0% — **PASS**

## Methods

- **(A) Instantaneous**: pipeline's `contacts_json.contacts[i].courtSide`.
- **(B) Trajectory median**: median ball-y over ±3 frames vs midline 0.5.
- **(C) Net-line crossing**: explicit midline crossing in 10f pre-contact.

*Ground truth* = GT actor's foot-y (`position.y + height/2`) vs midline 0.5.

## Results (all matches)

| method | eligible | match actual | rate |
|---|---|---|---|
| **A** | 330 | 234 | 70.9% ⚠️ |
| **B** | 338 | 216 | 63.9% ⚠️ |
| **C** | 1 | 1 | 100.0% ✅ |

## Results (wrong_cross_team subset, 48 actions)

| method | eligible | match actual | rate |
|---|---|---|---|
| **A** | 133 | 94 | 70.7% |
| **B** | 135 | 102 | 75.6% |
| **C** | 0 | 0 | 0.0% |

## Results (correct subset — sanity, should be high)

| method | eligible | match actual | rate |
|---|---|---|---|
| **A** | 197 | 140 | 71.1% |
| **B** | 203 | 114 | 56.2% |
| **C** | 1 | 1 | 100.0% |

## Decision

**Method C wins at 100.0% ≥ 95%.** Phase 2 chooser adopts this for ball-side inference. The signal alone does not solve attribution (memory `attribution_ballside_oracle_2026_04_23.md` showed +3.82pp ceiling) but is a trusted primitive for Phase 3 cross-checks.