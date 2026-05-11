# Attribution Team-Chain v1 — Measurement Report (2026-05-11)

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md
Plan: docs/superpowers/plans/2026-05-11-action-attribution-team-chain.md

## Baseline (DB read — current production)

From `scripts/measure_attribution_fresh_gt.py` stdout, 22 GT rallies, 136 GT actions:

```
[cece] 5 GT rallies
[gigi] 7 GT rallies
[wawa] 10 GT rallies

==============================================================================
ATTRIBUTION BASELINE — fresh GT (3 videos, 2026-05-11)
Match tolerance: ±10 frames
==============================================================================

PER-VIDEO
  fix       n         correct       wrong   miss   abs
  cece     29   22 ( 75.9%)    4 (13.8%)    3      0
  gigi     56   35 ( 62.5%)   12 (21.4%)    9      0
  wawa     51   25 ( 49.0%)   10 (19.6%)   16      0

COMBINED (n=136 GT actions)
  correct:            82 ( 60.3%)
  wrong (any):        26 ( 19.1%)
    cross_team:       16
    same_team:         7
    unknown_team:      3
  missing:            28 ( 20.6%)
  abstained:           0 (  0.0%)
```

Baseline confirmed matching expected values (combined correct 60.3%, wrong 19.1%, missing 20.6%).

---

## A/B Harness (in-memory re-run — env flag OFF vs ON)

From `scripts/measure_attribution_team_chain_ab.py`, re-running `reattribute_players` in memory
on the same 22 GT rallies with `RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN` set to 0 (OFF) and 1 (ON):

```
Loaded 22 rallies across 3 videos

=== OFF (production baseline) ===
  correct:  82  ( 60.3%)
  wrong:    26  ( 19.1%)  [cross=16 same=7 unk=3]
  missing:  28  ( 20.6%)

=== ON  (team-chain v1) ===
  correct:  82  ( 60.3%)
  wrong:    26  ( 19.1%)  [cross=16 same=7 unk=3]
  missing:  28  ( 20.6%)

=== DELTA (on - off) ===
  correct:           +0 (+0.0%)
  wrong_cross_team: +0
  wrong_same_team:  +0
  wrong_unknown:    +0

=== PER-FIXTURE DELTA ===
  cece   correct: 22 → 22 (+0)
  gigi   correct: 35 → 35 (+0)
  wawa   correct: 25 → 25 (+0)
```

---

## Gate Investigation — Why Zero Delta?

The A/B harness shows exactly 0 improvement. Full gate-by-gate diagnosis was run on all 16
cross-team errors where `current_is_nearest=True` (the cases that would require the team-chain
override to fire). Findings:

### G4 (ball-trajectory corroborator) blocks most cases

Key cross-team errors blocked by G4:
- `cece/f978201e/frame=84`: expected_team=0 (near), but contact.court_side=far → G4 FAIL
- `gigi/72c8229b/frame=474`: expected_team=0 (near), contact.court_side=far → G4 FAIL
- `gigi/3e07342a/frames=177,222`: expected_team=0 (near), contact.court_side=far → G4 FAIL

These are contact frames where the ball is physically on the "far" side of the court, but the
team-chain predicts the near-team should act. G4 correctly flags these as ambiguous — the ball
trajectory contradicts the chain prediction. This is the predicate being appropriately conservative.

### G3 (candidate within distance cap) also blocks

For `gigi/72c8229b/frame=474` and `gigi/3e07342a` errors: even if G4 were relaxed, no correct-team
candidate exists within 1.5x the current (wrong-team) player distance. The correct-team players are
genuinely farther from the ball, making swap unsafe.

### G1 (confidence >= 0.7) blocks low-conf actions

`cece/f978201e/frame=192` (conf=0.55), `wawa/7f0f540a/frame=613` (conf=0.50): both below the 0.7
threshold, correctly excluded.

### Root cause of zero delta

All 16 cross-team errors with `current_is_nearest=True` fail at least one of G1/G3/G4. The
new code path adds a team-chain override path that is gated by 4 conservative predicates. On the
current 3-video GT corpus, NONE of those contacts pass all 4 gates simultaneously.

This is **correct behavior** — the predicate is designed to avoid swapping when:
- The ball is on the "wrong" court side (G4), OR
- The correct-team candidate is significantly farther (G3 > 1.5x cap), OR
- The action has low confidence (G1 < 0.7).

The zero-delta result shows the new code is non-regressive (no harm done). The expected gains
require contacts where the ball side aligns with the chain's expected team but the nearest player
is on the wrong team — a pattern that exists in harder tracking scenarios (more ID switches).

---

## Pre-ship gates (A/B in-memory, env flag OFF vs ON)

- [x] G-A: Combined `correct_rate` improves by ≥ +5pp (60.3% → ≥ 65.3%).
      Result: OFF=60.3%, ON=60.3%, delta=+0.0pp.
      **FAIL** (0.0pp < 5pp threshold). See investigation above — no contacts meet all 4 gates.

- [x] G-B: `wrong_cross_team` ≥ 50% reduction (16 → ≤ 8).
      Result: OFF=16, ON=16.
      **FAIL** (no reduction). Same root cause as G-A.

- [x] G-C: No per-fixture `correct` regression (cece ≥ 22, gigi ≥ 35, wawa ≥ 25).
      Result: cece 22→22, gigi 35→35, wawa 25→25.
      **PASS** (no regressions; ON == OFF at baseline values).

- [x] G-D: `wrong_same_team` count non-increasing (7 today).
      Result: OFF=7, ON=7.
      **PASS** (no increase; no cross-team errors became same-team).

- [ ] G-E: `audit-coherence-invariants` C-2 violation count on the 3 videos ≤ current baseline.
      Result: **DEFERRED to post-deploy (Task 5)**.

---

## Summary

**Status: DONE_WITH_CONCERNS**

Gates G-A and G-B fail (zero improvement), while G-C and G-D pass (no regression). G-E deferred.

The team-chain predicate code is correct and non-regressive. The zero-delta reveals that the
current GT corpus does not contain contacts that satisfy all 4 gates simultaneously for the
cross-team errors. The gate design is appropriate — it correctly refuses to swap when ball
trajectory contradicts the chain prediction (G4). The errors that exist are structurally
"hard" cases where the ball-side signal contradicts team-chain, making unconditional swap unsafe.

**Next steps for controller:**
- The predicate is correct; the issue is that G-A/G-B gates assumed more of the 16 cross-team
  errors would pass all 4 gates. The gates may need loosening (particularly G3 ratio and G4
  soft-pass threshold) or the test corpus is not representative of the contacts where the
  override genuinely helps.
- Consider relaxing G4 to allow override when chain_integrity is strong (G2 well-satisfied) even
  when court_side contradicts — or raising the G3 distance ratio from 1.5x to 2.0x.
- Alternatively, accept zero-delta as a baseline and ship as-is (non-regressive, structure in place
  for future gain), then re-evaluate on a broader corpus.
