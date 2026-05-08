# Phase 3 — Stage-2 teamAssignment primitive error investigation

**Date:** 2026-04-24 (original); **corrected 2026-04-24 evening**
**Status:** Original 4-rally inversion claim was a diagnostic error. Real picture is 2 baseline-locked + 5 additional live-DB inversions. Details in the **Correction (2026-04-24)** section at the top. The original analysis below is retained for audit; read the correction first.
**Script:** `scripts/phase3_investigate_team_assignment_errors.py` (rewritten to use match-level GT)

---

## Correction (2026-04-24)

The original investigation below compared two different letter conventions as if they were the same:

- `team_assignments` in the baseline JSON is **rally-local** (near=A convention, stamped at `reattribute_actions.py:701-704` with `team 0 → "A"`). A pid's letter here can flip across rallies when side-switches flip which team is on near.
- `rally.serving_team` in the DB is **match-level** (via `resolve_serving_team`'s `label_a` calibration). It's fixed across rallies.

The old script derived `gt_team = team_assignments[gt_pid]` (rally-local) and compared it to `rally.serving_team` (match-level). Two letter systems; disagreement doesn't imply inversion.

**Authoritative match-level GT is `rallies.gt_serving_team` in the DB.** Comparing pipeline against that field gives the real picture:

| classification | count | definition |
|---|---|---|
| baseline-locked inversion | **2** | baseline JSON's `serving_team` ≠ `gt_serving_team`. These drive attribution metrics on the locked baseline. |
| live-DB inversion | **7** | current `rallies.serving_team` ≠ `gt_serving_team`. Superset of baseline (DB may have been updated since baseline lock). |
| convention-only mismatch | **2** | rally-local `team_assignments[gt_pid]` ≠ match-level `gt_serving_team`. **Not a bug** — side-switch artifact. |
| clean | 12 | no issue |

**Baseline-locked inversions (metric-visible):**
- tata `689854e3` gt_serve pid=2  baseline=A  match_gt=B
- lulu `35ca5d33` gt_serve pid=2  baseline=B  match_gt=A

**Additional live-DB inversions (not in locked baseline — pipeline output has drifted in the DB):**
- tata `30ffb876` pid=4  db=B  match_gt=A
- toto `51aa8083` pid=2  db=A  match_gt=B
- lulu `060e5898` pid=1  db=B  match_gt=A
- wawa `fb6e37bf` pid=1  db=B  match_gt=A
- wawa `7f0f540a` pid=2  db=A  match_gt=B
- wawa `06c13117` pid=3  db=A  match_gt=B

**Convention-only mismatches (flagged by the buggy old script; NOT real bugs):**
- tata `03144243` pid=2  rally_local=A  match_gt=B  — pid 2's letter is rally-local A in this rally but match-level B at video scope. Both consistent with a post-side-switch state.
- lulu `71c642dc` pid=4  rally_local=A  match_gt=B  — same pattern.

### What the old analysis got right vs wrong

Right:
- The "Class A / wrong_cross_team with GT at rank 0" analysis (4 rallies). Those are genuine chain errors, unchanged.
- The structural observation that missing-serve rallies are where the pipeline has to infer `serving_team` from alternatives (Viterbi, formation fallback) and is more error-prone there.

Wrong:
- "4 rallies with servingTeam inversion" — 2 of those 4 (tata 03144243, lulu 71c642dc) were convention mismatches, not inversions.
- "All 4 have the same pattern gt_team=A, servingTeam=B" — that pattern only held because of the letter-system confusion.
- "Rough sizing ~5pp cascade corpus-wide" — the real baseline-locked inversion count is 2, not 4, and the subsequent cascade simulation (see `phase3_serving_team_cascade_test_2026_04_24.md`) showed 0 attribution corrections even under upper-bound assumptions, regardless of the inversion count.

### What to do with this finding

The 2 baseline-locked inversions are real but the cascade simulation in `phase3_serving_team_cascade_test_2026_04_24.md` demonstrated the inversion doesn't affect end-to-end attribution (nearest-candidate guard absorbs the bad chain flips). Impact is still on rally-level display / score_accuracy eval, not on attribution. Conclusion unchanged: **cosmetic for attribution, potentially meaningful for score evaluation**, but in either case much smaller scope than initially claimed.

The additional 5 live-DB inversions suggest the DB state has drifted from the 2026-04-24-morning baseline lock. Worth investigating if anyone's been manually editing `rally.serving_team` or if `reattribute-actions` has been re-run with different parameters — but unrelated to the attribution rebuild scope.

**Fix priority: low.** Neither the attribution nor the score_accuracy baselines have been re-measured against the corrected picture. If `serving_team`/`gt_serving_team` agreement is a real score-eval metric, these 2 baseline-locked rallies are the first to correct; a proper fix requires tracing which inference path (formation, Viterbi) flips on them.

---

## Original analysis (superseded — read correction above first)


## Hypothesis tested

Two error classes from the Pattern B and guard-relax diagnostics looked like they might share a primitive-level root cause (analogous to the match_tracker pair fix that lifted Phase 1.3 from 26% to 100%).

## Class A — NOT a primitive bug (ruled out)

4 `wrong_cross_team` errors where GT actor is at rank 0 in contact candidates but pipeline emitted a rank-1 pick.

| rally | fixture | frame | GT | PL (rank 1) |
|---|---|---|---|---|
| `4e7e589c` | tata | 475 | pid=1 team=A | pid=3 team=B |
| `f127f3d5` | cuco | 222 | pid=2 team=B | pid=4 team=A |
| `cbf17cce` | yeye | 806 | pid=3 team=B | pid=2 team=A |
| `2d3cb54b` | yeye | 90  | pid=4 team=A | pid=1 team=B |

No shared root cause — `team_assignments` are consistent with GT in all 4 cases. The pipeline picked a wrong cross-team candidate at rank 1 rather than the correct nearest candidate. This is chain/attribution noise driven by misattributed upstream anchors (serve-chain errors), not a teamAssignment primitive bug. A fix here would require per-case chain tracing, not a primitive-level correction.

Close this class; revisit only if individual rallies trigger downstream investigation.

## Class B — **Real `servingTeam` inversion bug, 4 rallies**

All 4 missing-GT-serve cases where `gt_pid` is not among serving-team primaries share the **same pattern**:

| rally | fixture | gt_pid | gt_team | servingTeam | serving primaries | team_assignments |
|---|---|---|---|---|---|---|
| `03144243` | tata | 2 | A | **B** | [4, 1] | {1:B, 2:A, 3:A, 4:B} |
| `51aa8083` | toto | 2 | A | **B** | [1, 3] | {1:B, 2:A, 3:B, 4:A} |
| `35ca5d33` | lulu | 2 | A | **B** | [3, 4] | {1:A, 2:A, 3:B, 4:B} |
| `71c642dc` | lulu | 4 | A | **B** | [1, 2] | {1:B, 2:B, 3:A, 4:A} |

In every case `gt_team ≠ servingTeam` and `team_assignments[gt_pid]` is consistent across the rally's other actions. The failure mode is **`servingTeam`** being flipped, not team labels.

## Why this matters

`servingTeam` anchors the serve-chain in `reattribute_players`:
- serve's team = servingTeam → receive's team = opposite → set's team = same → attack's team = same → ...

If `servingTeam` is inverted on a rally, **every subsequent action in that rally has an inverted `expected_team`**, and `reattribute_players` flips attributions toward the wrong team. This is likely a large downstream multiplier — each of these 4 rallies probably has multiple chain-propagated `wrong_cross_team` errors beyond just the missing serve.

Rough sizing: 4 rallies × ~6 actions/rally ≈ 24 cascaded errors potentially correctable by fixing servingTeam on these 4 rallies. That's ~5pp of the 468-action corpus.

## Fix sites (not yet implemented)

- `analysis/rallycut/tracking/action_classifier.py:574` — `_find_serving_team_by_formation`, the preferred servingTeam inference. This is the likely origin of the inversion.
- `analysis/rallycut/tracking/team_identity.py:183` — `resolve_serving_team`, the final resolver that decides between formation vs contact-based vs Viterbi-decoded sources.
- `analysis/rallycut/scoring/cross_rally_viterbi.py:370` — Viterbi's `decode_serving_teams_with_noisy_labels`, which can over-ride formation. If Viterbi is flipping correct formations, this is a second candidate site.

## Common failure trigger

All 4 cases have the serve itself missing from pipeline output (correlated-miss — the serve contact wasn't emitted). The `servingTeam` inference then has to rely on either formation (who stood in serving position at rally start) or the first post-serve contact (which, if it's an opposing-team receive, could be mis-classified as the serve and invert the team assignment).

Testable hypothesis: on these 4 rallies, does the first emitted contact belong to the opposing team (receive), and did the formation detector misfire or fall back to contact-based inference? A 1-hour trace on one rally (e.g. tata `03144243`) would validate or invalidate.

## Recommendation

**Scope a separate workstream** to fix `servingTeam` inference on missing-serve rallies. Estimated sizing: +5pp corpus-wide (4 rallies × cascade multiplier), conditional on:

1. Validating the formation-fallback-inverts hypothesis on 1-2 specific rallies (1-2 hours).
2. Implementing a guard in `_find_serving_team_by_formation` or `resolve_serving_team` that detects and either fixes or abstains on the inversion trigger (1-2 days).
3. Re-running the baseline and benching — this is a primitive fix (like the match_tracker pair fix), visible via end-to-end re-attribution.

Per-fixture: the bug spans tata, toto, lulu (2 cases). Not fixture-isolated — suggests a generic inference-path bug, not per-video miscalibration.

**Second recommendation (cheaper, complementary):** add `scripts/phase3_investigate_team_assignment_errors.py` to the audit framework. Re-run on any future corpus expansion — catches `servingTeam` inversions automatically.

## Combined session verdict (updated)

| workstream | result | action |
|---|---|---|
| Pattern A (bookend rescue) | +0.2pp | Closed, tooling kept |
| Pattern B (server-occluded serve) | 1/22 ceiling | Closed before code |
| Guard relaxation | +0.85pp ceiling | Conditional NO-GO |
| Class A team-assignment errors | noise, no root | Closed |
| **Class B `servingTeam` inversion** | **4 rallies → ~5pp ceiling** | **Worth pursuing** |

The `servingTeam` inversion on 4 rallies is the first workstream this session with a clear bug hypothesis, a probable fix site, and a cascade-multiplier sizing story. Bigger potential than any rescue pattern and independent of contact-detector recall work.
