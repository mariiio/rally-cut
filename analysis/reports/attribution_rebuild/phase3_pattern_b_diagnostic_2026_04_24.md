# Phase 3 Pattern B — Pre-implementation diagnostic

**Date:** 2026-04-24
**Status:** NO-GO before implementation. Ceiling 1/22 on the missing-serve bucket.
**Script:** `scripts/phase3_diagnose_pattern_b.py`

## Spec under test (plan §3.2)

> "No contact at expected serve frame but next team-action within 60 frames is
> an across-net receive by the opposing team's visible player → infer serve by
> expected team's expected server (if unambiguous via 3.1)"

## Result on the 22 missing GT serves

| precondition | pass | of | notes |
|---|---|---|---|
| opposing receive/dig anchor within 60f of GT serve | 4 | 22 | 18 have no post-serve activity in pipeline at all |
| ... AND estimated frame (anchor − 30f) matches GT ±10f | 1 | 22 | 3 anchors exist but frame estimate drifts |
| serving team has exactly 1 candidate (unambiguous) | 0 | 22 | 2v2 → always 2 candidates; alternation required |
| gt_pid among serving-team primaries | 18 | 22 | 4 cases have GT actor not on serving team's primaries (primitive error) |
| **CEILING — anchor + frame match + gt in candidates** | **1** | **22** | Even with perfect serve-alternation resolution |
| **Unambiguous 1-shot rescue (no alternation)** | **0** | **22** | — |

## Failure-mode breakdown

| reason | count |
|---|---|
| no opposing anchor within 60f | 18 |
| anchor exists but frame estimate misses ±10f | 3 |
| ambiguous (2 serving candidates, alternation needed) | 1 |

## Why (same pattern as A)

Correlated-miss strikes again. When the contact detector misses a serve, it also typically misses the opposing-team receive/dig that would anchor the rescue. 18 of 22 missing serves land in rallies where no post-serve contact survives into pipeline output.

Additionally:

- **Server-unambiguity never holds in 2v2.** Both primary tids on `servingTeam` are visible at rally start. Pattern B's plan wording ("if unambiguous via 3.1") presupposed a single visible candidate — that case doesn't exist in the corpus.
- **Serve alternation is required for EVERY case.** Implementing Pattern B means tracking "who served last per team" across rally boundaries — stateful, cross-rally logic.
- **4/22 have primitive team-assignment errors**, independent of Pattern B: `gt_pid` is not even on `servingTeam`'s primary list. These need stage-2 primitive fixes, not a rescue pattern.

## Decision

Close Pattern B before coding. The ceiling is 1 rescue (same order as Pattern A) and it would require:
- Cross-rally serve-alternation state
- Frame estimation heuristic (30-frame anchor offset is eyeballed; real serve-to-receive flight varies)
- 4 additional primitive fixes for the gt_pid-not-in-candidates cases

Cost-benefit vs +1 rescue on 468-action corpus: not worth it. Same call as Pattern A.

## What this confirms (cross-pattern insight)

The **correlated-miss hypothesis is now evidenced on two independent rescue patterns** (A: 1/15 missing sets rescuable; B: 1/22 missing serves rescuable). Every rescue pattern operating on pipeline-output alone will hit the same ceiling because when the detector misses one action, it misses the surrounding context that a rescue pattern would need to anchor against.

**New rule, lifted to plan §10.2:** size any future rescue-pattern headroom by explicit anchor availability on the target subset, BEFORE writing code. Category counts from Phase 1.5 (detection_limit 71, missing 72) systematically over-project rescue surface.

## Artifacts retained

- `scripts/phase3_diagnose_pattern_b.py` — the diagnostic. Reusable on corpus expansion.
- This report.

No Pattern B implementation written.
