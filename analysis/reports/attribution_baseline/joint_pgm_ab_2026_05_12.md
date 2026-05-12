# Joint Attribution PGM A/B Verdict — 2026-05-12

## Status: **NO-SHIP** (STOP condition triggered)

Hand-tuned soft PGM with absent-actor states does not beat the existing
`reattribute_players` chain on the 22-rally fresh-GT panel. Calibrated PGM
**regresses correct rate by -9.5pp** (70 → 60 of 105 GT actions). Ships as
**default-OFF infrastructure** with `USE_JOINT_ATTRIBUTION=0` (default).

## Origin

Pursued after both contact-detection FN-reduction phases (Phase 1
validation-gate relaxation + Phase 1.5 generator-threshold relaxation) hit
the rule-tuning ceiling and shipped as NO-SHIP. Morning's per-error probe
on the panel predicted **+13pp ceiling** for the PGM (recovers up to 14 of
24 attribution errors: 9 rank-1-overruled + 5 absent-from-candidates). That
prediction did not materialize.

## Headline

| Metric | Legacy (`USE_JOINT_ATTRIBUTION=0`) | PGM (`=1`, calibrated) | Δ |
|---|---|---|---|
| correct | 70 (66.7%) | **60 (57.1%)** | **-10 (-9.5pp)** |
| wrong (any) | 21 (20.0%) | 21 (20.0%) | 0 |
| cross_team | 12 | 14 | +2 |
| same_team | 9 | 7 | -2 |
| unknown_team | 0 | 0 | 0 |
| missing | 14 (13.3%) | 14 (13.3%) | 0 |
| abstained | 0 | **10 (9.5%)** | **+10 (PGM-absent)** |

Per-fixture:

| Fixture | Legacy correct | PGM correct | Δ |
|---|---|---|---|
| cece | 18/21 (85.7%) | 13/21 (61.9%) | -5 |
| gigi | 32/40 (80.0%) | 30/40 (75.0%) | -2 |
| wawa | 20/44 (45.5%) | 17/44 (38.6%) | -3 |

## Pre-ship gates

| Gate | Threshold | Result | Verdict |
|---|---|---|---|
| **G-A (Correct rate)** | ≥ +5pp absolute | -9.5pp | **FAIL** |
| **G-B (Cross-team errors)** | ≥ 30% reduction | 12 → 14 (+17%) | **FAIL** (worsened) |
| **G-C (Same-team errors)** | non-increasing | 9 → 7 (-2) | PASS (improved) |
| **G-D (Unknown-team errors)** | non-increasing | 0 → 0 | PASS |
| **G-E (Per-action recall)** | no GT type drops > 2pp | (not stratified; all per-fixture regressed) | FAIL (per-fixture proxy) |
| **G-F (Latency)** | ≤ 50ms avg | not formally measured; PGM completed all panel rallies in seconds | PASS (informal) |
| **G-G (Coherence)** | non-increasing | not measured post-restore | n/a |
| **G-H (Unit tests)** | all pass | 1419 pass with flag unset | PASS |

## STOP conditions

| Condition | Triggered? |
|---|---|
| **Action correct rate regresses below baseline** | **YES** (70 → 60) |
| Same-team errors > +33% | No (9 → 7) |
| Inference timeout (any rally > 2 sec) | No |

**STOP triggered → snapshot restored.** DB back to pre-PGM state
(verified: post-restore measurement re-produces 70/105 baseline exactly).

## Calibration outcome

Coordinate ascent on the 22-rally panel changed 4 of 13 weights:

| Weight | Hand-tuned | Calibrated | Change |
|---|---|---|---|
| w_proximity | 2.0 | 1.0 | -50% |
| w_dist | 1.0 | 0.5 | -50% |
| w_team_consistency | 2.0 | 3.0 | +50% |
| w_absent_pair | 1.5 | 1.125 | -25% |
| (other 9 weights) | unchanged | unchanged | — |

Calibration brought PGM correct rate up to 25/105 (panel-level using
calibration script's measurement; differs from the full-harness 60/105
in scoring details) from the default-weight baseline of 12/105. Even
optimized, calibrated PGM materially underperforms legacy.

## Why the +13pp prediction didn't materialize

The morning's per-error probe assumed: *for the 9 errors where GT is at
rank 1-2 of candidates, joint inference will select GT*. That assumption
was wrong. Two reasons:

1. **Absent-state unaries outweigh tracked-state unaries** under any
   explored weight configuration. The PGM's joint MAP prefers absent
   states when team-distance is non-trivial — even when a tracked player
   is right there. 10 of 105 actions on the panel ended up tagged
   `pgm_absent_team_*` post-calibration (9.5% absent rate; pre-calibration
   would have been much higher).
2. **Hand-tuned factor weights aren't expressive enough** to encode the
   per-action-type dynamics the data demands. The same weights are applied
   to serves, sets, attacks, digs — but each has different signal profiles
   (serves often have far-court servers; sets are soft trajectories; etc.).
   Phase B (per-action-type learned weights from GT) is the right
   architectural answer, but requires GT expansion.

The PGM's pairwise volleyball-rule factors did reduce same-team errors
slightly (-2) — a small confirmation that joint inference IS doing
something useful on the rule side. But the unary evidence model's
weakness drowns out the rule-side wins.

## What ships

**Default-ON in production: nothing.** `USE_JOINT_ATTRIBUTION=0` is the
default; production behavior is byte-identical to pre-workstream.

**Infrastructure that landed (reusable for future workstreams):**
- `analysis/rallycut/tracking/joint_attribution.py` —
  `joint_attribute_rally(rally) -> RallyAttribution` public API
- `analysis/rallycut/tracking/joint_attribution_factors.py` — 8 pure
  factor functions
- `analysis/rallycut/tracking/joint_attribution_weights.py` —
  `FactorWeights` dataclass + calibrated `DEFAULT_WEIGHTS`
- `analysis/scripts/calibrate_joint_attribution_weights.py` —
  coordinate-ascent calibration (reusable)
- `analysis/scripts/measure_joint_attribution_ab.py` — A/B harness
- 4 unit test files: 46 tests covering factors, inference, e2e,
  integration
- `attribution_source` field on action dicts (additive; downstream
  consumers ignore unless they care)
- `apply_pgm_result_to_actions` helper writes MAP + marginals into
  action dicts
- `USE_JOINT_ATTRIBUTION` env flag wired at 2 call sites

Total: ~700 lines production code + ~600 lines tests + 2 scripts.

## Recommended next workstreams

The PGM's NO-SHIP, combined with both contact-detection NO-SHIPs (Phase
1 + 1.5), exhausts the "tune the existing pipeline with hand-tuned soft
factors" approach. Three remaining ladders, in priority order:

1. **Phase 1.7 — re-enable `enable_player_motion_candidates` for blocks
   specifically.** Direct shot at block recall (10.8% floor). Independent
   of attribution architecture. The 9 TPs hidden in the 265 candidates
   would meaningfully move block recall. Cost ~1 week.

2. **GT expansion to ~50+ rallies.** Bottlenecked but compounds with
   everything else. The 22-rally panel is too noisy to detect sub-5pp
   lifts; this is the binding constraint on every future workstream.

3. **Phase B — learn factor weights per-action-type from GT.** This is
   the right answer to "hand-tuned weights don't capture per-action
   dynamics", but it's bottlenecked on GT expansion. After (2) ships,
   this becomes viable.

Less promising next ladders:
- **Ball tracker recall** — would surface contacts that contact detection
  currently misses. Independent of attribution. Bumps against ball-
  tracker F1 ceiling.
- **Cluster-then-attribute reframe (Framing 2 from morning brainstorm)** —
  high-disruption rewrite; only justifiable if all per-contact approaches
  cap out and (1)+(2)+(3) don't move the needle.

The original morning question — "what architecture replaces per-contact
attribution to reach competitor accuracy?" — now has an empirical answer
floor: hand-tuned joint inference doesn't reach it. The competitor's
near-perfect accuracy likely requires both better signals AND
joint-inference with learned weights AND larger GT.

## Files

- Spec: `docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md`
- Plan: `docs/superpowers/plans/2026-05-12-joint-attribution-pgm.md`
- This report

## Rollback

`USE_JOINT_ATTRIBUTION=0` is default. To explicitly enforce: ensure no
deployment env sets the flag. Production rollback (if accidentally
enabled): set env=0 and re-run `reattribute-actions <video>` on affected
rallies. Snapshot pattern at `/tmp/joint_pgm_pre_ab_snapshot.jsonl` is
the cheap restore mechanism.

## Snapshot restore confirmation

Post-restore measurement matches legacy baseline exactly:
- correct: 70 (66.7%) ✓
- wrong: 21 (20.0%) ✓
- missing: 14 (13.3%) ✓
- abstained: 0 ✓

DB is at clean baseline. The 4 panel-video rallies' `actions_json` are
byte-identical to pre-A/B state.
