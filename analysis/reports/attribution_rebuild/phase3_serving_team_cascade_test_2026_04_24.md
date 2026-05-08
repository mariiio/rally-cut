# Phase 3 — servingTeam flip cascade test

**Date:** 2026-04-24 (original); **correction note 2026-04-24 evening**
**Status:** NO-GO for attribution conclusion stands. Rally-set used was partly wrong (see correction). Cosmetic bug only; no end-to-end attribution lift under either rally-set.
**Script:** `scripts/phase3_simulate_serving_team_flip.py`

## Correction (2026-04-24)

The 4 rallies this simulation ran on (tata 03144243, toto 51aa8083, lulu 35ca5d33, lulu 71c642dc) were identified by the flawed investigation in `phase3_team_assignment_investigation_2026_04_24.md` (see its correction section). The corrected rally split:

- tata `03144243` — convention-only mismatch, NOT a bug
- toto `51aa8083` — live-DB inversion only (baseline had match_gt="B" agreement)
- lulu `35ca5d33` — baseline-locked inversion (real, metric-visible)
- lulu `71c642dc` — convention-only mismatch, NOT a bug

So this simulation's 4-rally set was only 1/4 correctly identified. The corrected investigation found **2 baseline-locked inversions** (lulu 35ca5d33 + tata 689854e3) and 5 additional live-DB inversions.

**Does the correction change the conclusion?** No. The cascade simulation's core finding — that flipping `serving_team` produces **0 attribution corrections** because the nearest-candidate guard absorbs the chain's bad flip attempts — is independent of which specific rallies you simulate on. The mechanism is the same: per-action `team` labels come from `team_assignments[pid]` (correct per Phase 1.3), so the chain comparing current-team to expected-team produces no override even under inverted `serving_team`.

If you wanted to re-run the simulation on the corrected 2-rally set (tata 689854e3, lulu 35ca5d33), the expected outcome is still +0. The simulation's mechanism doesn't depend on which rallies you pick.

The wrong rally set does invalidate some specifics (the "~5pp cascade corpus-wide" estimate was always wrong, but it was wrong at a different magnitude than the original memo admitted). Net: the NO-GO call stands; the memo's reasoning is correct; only the identity of the target rallies was partly wrong.

---

## Original analysis (rally set partly wrong — see correction above)

## Revised sizing (earlier estimate was wrong)

Prior memo (`phase3_team_assignment_investigation_2026_04_24.md`) estimated ~5pp corpus-wide lift from fixing servingTeam on 4 rallies (4 rallies × ~6 cascaded corrections each). **Wrong.** The actual numbers on the 4 target rallies:

| rally | fixture | correct | wrong_cross | missing | cascade UB |
|---|---|---|---|---|---|
| tata `03144243` | tata | 5 | 1 | 1 | +0 |
| toto `51aa8083` | toto | — | — | — | empty pipeline, cannot simulate |
| lulu `35ca5d33` | lulu | 0 | 1 | 5 | +0 |
| lulu `71c642dc` | lulu | 4 | 3 | 5 | +0 |
| **TOTAL (3 sim-able)** | | 9 | 5 | 11 | **+0** |

Upper bound of attribution-level lift from flipping servingTeam: **+0 corrections**.

## Why cascading doesn't fire

The `reattribute_players` serve-chain compares `current_team = team_assignments[action.pid]` against `expected_team` derived from `serving_team` + action-type alternation. Phase 1.3 verified `team_assignments` is 100% correct post-match_tracker-fix, so per-action team labels are reliable.

Case analysis:

1. **servingTeam correct, chain correct, current matches expected → no override, correct outcome.**
2. **servingTeam inverted, chain inverted, current (correct per-action) doesn't match expected (inverted) → chain WANTS to override.**
3. **But** the nearest-candidate guard at `action_classifier.py:2930` blocks the override when `current_pid` is at rank 0.
4. **Net:** the 4 rallies retain their correct per-action attributions, despite inverted servingTeam, because the nearest-guard absorbs the chain's bad flip attempts.

In other words: the servingTeam inversion + nearest-guard are already compensating each other at the attribution level. Fixing one without the other would risk regression.

## What the sanity check DID confirm

4/4 target rallies have `gt_team == flip_team(serving_team)` — the inversion diagnosis is right. GT actor's team matches what servingTeam WOULD be under the flipped value.

So the bug is real, but its impact is:

- **Attribution (468-action corpus): 0 corrections** (cascade fully absorbed by nearest-guard).
- **Rally-level servingTeam display: 4 rallies wrong** (affects UI, stats).
- **Score evaluation accuracy: unmeasured** (separate eval surface — score_accuracy baseline). Would improve on these 4 rallies' serving/receiving team display.

## Decision

**Do not ship a servingTeam fix as an attribution workstream.** No measurable impact on the 9-fixture attribution baseline. The inversion is a real primitive bug but its effect is neutralized by the nearest-guard.

**Does worth tracking as a score/stats workstream.** If the score_accuracy or UI-correctness evaluation picks up these 4 rallies' inverted servingTeam as errors, a fix would deliver direct improvement there. That's a separate eval surface not in scope today.

## Diagnostic retained

`scripts/phase3_simulate_serving_team_flip.py` — servingTeam-flip cascade simulator. Reusable if a future workstream tests servingTeam interventions against the score_accuracy baseline.

## Session conclusion (updated — attribution front is empty)

Five consecutive attribution-front workstreams this session, all diagnosed and closed:

| workstream | ceiling | verdict |
|---|---|---|
| Pattern A (bookend rescue) | +0.2pp | NO-GO on impact |
| Pattern B (server-occluded serve) | 1/22 | NO-GO before code |
| Guard relaxation | +0.85pp | Conditional NO-GO |
| Class A team-assignment | noise | Closed |
| Class B servingTeam inversion | **+0pp attribution / some stats** | NO-GO for attribution; possibly ship for stats |

**The current primitive/chooser/rescue architecture is at its ceiling on this corpus.** Every remaining attribution improvement requires one of:

1. **Contact-detector recall investment** — the architectural bottleneck (22%, 105/468 actions with no candidate). 4 prior NO-GOs; needs new hypothesis.
2. **Held-out corpus labeling** — validates the 9-fixture baseline generalises; may surface different failure modes.
3. **Orthogonal evaluation surface** — e.g., score_accuracy with the 4 servingTeam inversions addressed.

Attribution-front iteration in-session has reached diminishing returns. Time to either invest in (1) with a new hypothesis or pivot workstreams.
