# Phase 3 — Nearest-candidate-guard relaxation opportunity diagnostic

**Date:** 2026-04-24
**Status:** Marginal opportunity (+0.85pp ceiling). Full pipeline sweep not recommended without first improving serve-chain reliability.
**Script:** `scripts/phase3_diagnose_nearest_guard.py`

## The guard under test

`analysis/rallycut/tracking/action_classifier.py:2930-2935` in `reattribute_players`:

```python
if (
    not is_unmapped
    and contact.player_candidates
    and contact.player_candidates[0][0] == action.player_track_id
):
    continue  # don't override the nearest candidate
```

Blocks serve-chain-declared corrections on mapped tracks when the current attribution is the nearest candidate. Rationale: proximity is hard physical evidence.

## Error-structure measurement on baseline

141 `wrong_cross_team` errors on locked baseline. Current pl_pid's rank in `contact.playerCandidates`:

| pl_pid rank | count | comment |
|---|---|---|
| 0 (nearest) | 131 | **guard-blocked** — the serve chain wants to override but the guard holds |
| 1 | 10 | chain already fired (non-nearest override) |

Of the 131 guard-blocked, GT actor's rank:

| GT rank | count | reachable via relaxed guard? |
|---|---|---|
| 0 | 4 | already there; team-label mismatch upstream (teamAssignment error) |
| 1 | **41** | ✅ reachable with a single rank-0→rank-1 swap |
| 2+ | 53 | ❌ deeper reach required — higher FP risk |
| NOT in candidates | 38 | ❌ architectural ceiling (detector didn't surface GT) |

**The rank-1 bucket is the real target surface: 41 errors, ~8.8% of the 468-action corpus.**

## Margin-threshold sweep (upper bound)

For each guard-blocked rec, compute `margin = (dist_rank2 − dist_rank1) / dist_rank1`. At threshold T, a relaxed guard fires only when `margin < T` (proximity evidence is weak). Sweep:

| thr | TP (w_cross→correct) | FP ceiling (correct→wrong) | net |
|---|---|---|---|
| 0.05 | 3 | 2 | +1 |
| 0.10 | 3 | 3 | +0 |
| 0.15 | 8 | 4 | **+4** |
| 0.20 | 9 | 5 | +4 |
| 0.30 | 10 | 9 | +1 |
| 0.50 | 14 | 15 | −1 |
| 1.00 | 25 | 57 | −32 |

**Peak net upper bound: +4 correct at threshold 0.15-0.20 = +0.85pp on the 468-action corpus.**

## Caveats on the +4 number

### TP is an upper bound

These TP counts assume the serve-chain declares the correct `expected_team` on every triggered case. In reality, chain reliability is a ceiling:

- Serve attribution is the chain's root — mis-attributed serves cascade wrong `expected_team` forward.
- Phase 2 summary documented that the team-pair fix doesn't lift end-to-end accuracy because chain quality is the propagation bottleneck.
- True TP < 8. Plausible range: 3-6 actual corrections.

### FP ceiling is a full upper bound too

FP_ceiling counts every `correct` record with `pl at rank 0` and `margin < T`, regardless of whether the chain would disagree. The chain only flips the attribution if `expected_team ≠ current_team`. On correct records the chain often agrees (since both are correct) and doesn't trigger the guard path at all.

**True FP is a fraction of FP_ceiling** — probably 20-50% depending on chain disagreement rate on marginally-close records.

### Net estimate: +1 to +3 correct, plus noise

Given TP 3-6 and FP 1-2, best case real-world net is +1 to +5 correct. Small in absolute terms; signal-level indistinguishable from GT labeling noise at this sample size.

## Why this differs from Patterns A/B (clean NO-GOs)

- Patterns A/B had ceilings of 1 rescue on correlated-miss data. Their ceilings were dominated by anchor absence.
- Guard-relax has a ceiling of 41 (GT at rank 1) which is 10× larger — proximity/chain-disagreement is a different failure mode than detector-miss.
- But ceiling ≠ achievable: the chain-reliability bottleneck caps real-world TP, and real FP threatens the monotonic-wrong-rate contract.

## Decision: conditional NO-GO

**Do not run the full pipeline sweep (env-flag guard relaxation + re-attribute + bench) at this time.** The offline diagnostic caps best-case lift at +0.85pp, and the honest measurement path (full pipeline run per THR value, ~1 hour each) would be expensive for a likely +1 to +3 correct result.

**Conditions under which this becomes ship-worthy:**

1. **Serve-chain reliability improves** — if we can raise chain accuracy materially, the real TP approaches the +8 ceiling and FP drops. This is contingent on contact-detector recall work (serve emission is the current bottleneck).
2. **Per-fixture signature identified** — if the 41 rank-1 errors concentrate in 1-2 fixtures, a per-fixture relax might ship separately without touching the rest.

### Per-fixture TP concentration (below): is the opportunity fixture-skewed?

*(Not computed in this pass — worth a 5-min follow-up if the concept interests.)*

## Cross-session conclusion

**Three independent rescue/override workstreams in one session, all closed or marginal on the same corpus:**

- Pattern A (bookend rescue): +0.2pp ceiling / 1 rescue, correlated-miss
- Pattern B (server-occluded serve): +0.2pp ceiling / 1 rescue, correlated-miss + 4 primitive errors
- Guard relaxation: +0.85pp ceiling, chain-reliability bottleneck

None of these breach the structural ceiling imposed by contact-detector recall (22% of GT has no candidate). The leverage is elsewhere: either the contact detector gets better (architectural hypothesis needed; 4 prior NO-GOs), or the primitive-level team assignment gets upstream-fixed (the 4 rank-0 errors in guard-blocking + the 4 missing-serve primitive errors point to specific stage-2 cases worth investigating).

## Artifacts retained

- `scripts/phase3_diagnose_nearest_guard.py` — diagnostic. Reusable per-fixture.
- This report.

No code change to the guard. No pipeline sweep run.
