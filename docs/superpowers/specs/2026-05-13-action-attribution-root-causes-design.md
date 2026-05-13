# Action-Attribution Root-Cause Workstreams — Design (2026-05-13)

## Status

Design phase. Three independently-shippable workstreams (A / B / C). Each plan and ship gate is self-contained; nothing in this design requires all three to land together.

## Origin

Brainstorm session 2026-05-13 working from four concrete current-pipeline failures observed in the C-4 coherence catalog ([[coherence_repair_sub_2b_2026_05_13]]):

- **Cascade** — rally `a0881d82` (video `2e984c43`/titi): five consecutive contacts attributed to player 2 (team B) where teammate is 2–7× farther. Survives full pipeline refresh.
- **F3 — set→attack cross-team via occlusion** — rally `0144acfb` (video `bfd1decd`/keke), frames 174→223. User-observed: far-side attacker occluded by near-side blocker; contact attributed to nearest visible (the blocker).
- **F4 — synthetic + real duplicate serve** — rally `88a04529` (keke): synth serve correctly prepended for team B at frame 106, then real action at frame 111 wrongly classified as serve (should be receive). `repair_action_sequence.Rule 3` only handles non-synth duplicates.
- **F5 — mid-rally possession-back** — rally `99091ec6` (keke): sequence `attack(A) → receive(B) → set(B) → attack(B)` with suspect `attack(A)` at frame 184 (conf 0.694), most likely a block mis-typed as attack.

GT inspection on the cascade rally (added during the brainstorm session) resolved the architectural puzzle: the "5 consecutive p2 attributions" is **not** a single failure mode. Bbox-center matching from GT snapshot bboxes to current-pipeline tracks (each match distance < 0.002 normalized) shows:

| Frame | Pipeline | GT (current track) | Verdict |
|:--|:--|:--|:--|
| 76 RECEIVE | track 2 | track 2 | ✓ correct |
| 128 SET | track 2 | **track 1** | ✗ within-team swap (B→B) |
| 176 ATTACK | track 2 | track 2 | ✓ correct |
| 191 ATTACK | — missed — | track 3 | ✗ contact-detection FN |
| 225 DIG | track 2 | **track 1** | ✗ within-team swap (B→B) |
| 276 SET | track 2 | track 2 | ✓ correct |
| 326 ATTACK | track 1 | track 1 | ✓ correct |
| 340 ATTACK | — missed — | track 3 | ✗ contact-detection FN |

The cascade decomposes into **2 within-team mis-attributions** (frames 128, 225) plus **2 contact-detection FNs** (frames 191, 340). C-4 fires because three correct attributions to track 2 are sandwiched between two wrong ones, surfacing as five same-player back-to-back pairs.

The contact-detection FN ladder is already documented as needing new signals ([[contact_detection_fn_v1_2026_05_12]]); those two frames are **out of scope** here. The within-team swap at frames 128 and 225 has the same shape as F3 (occlusion) and F5 (block-mis-typed): **the producer's single-frame local signal is structurally insufficient.** This design addresses that.

## Hard constraints (from session prompt)

- No "patch" / "hack" / "quick rule tweak" downstream of the producer.
- No reviving Sub-2.B Phase 2 as a soft-veto repair pass — its 2× distance cap was the binding constraint that excluded the cascade.
- No reviving the joint-attribution PGM with hand-tuned global weights ([[joint_attribution_pgm_2026_05_12]] NO-SHIP — `-9.5pp` panel regression).
- Each workstream must have a falsifiable ship gate.

## Failure taxonomy

| Cluster | Failures | Shape of root cause | Ceilings hit |
|:--|:--|:--|:--|
| **A** — Producer single-frame signal lies | cascade frames 128/225; F3; F5 | Contact-detector + action-classifier + reattribute_players reach a local decision from a single-frame 2D-distance + per-frame action-conf signal. Truth requires *structural* info: pose, 3D, trajectory-endpoint, anti-self-touch, sequence-coherence. | Pass 2c (within-team swap) blocked by confidence floor; PGM hand-tuned weights cap; Sub-2.B Phase 2 2× cap can't reach 2–7× cases. |
| **B** — Post-processor dedupe gap | F4 | `repair_action_sequence.Rule 3` only dedupes non-synth duplicate serves. Synth + real is the missing symmetric branch. | None — incomplete rule. |
| **C** — Audit blind spots | F3, F5 don't surface | `coherence_invariants.C-2` punts on mid-possession crossovers (deliberate v1 punt). | None — additive. |

## Out of scope

- Missing contacts at cascade frames 191 / 340 — belongs to [[contact_detection_fn_v1_2026_05_12]] ladder.
- GT resolver bug — `rally_action_ground_truth` rows for cascade rally have `SNAPSHOT_EXACT` resolved_source with stale `snapshot_track_id` values (6, 7) that don't exist in current tracking. Bbox-center matching is the reliable reconciliation. Separate small fix; flagged as a follow-up.
- Reviving Sub-2.B Phase 2 — superseded by A1.

---

# Workstream A — Type-aware producer attribution upgrade

Move player attribution from "nearest-player at contact_detector time" to **type-aware attribution that runs after the action classifier picks a type**. Per-action-type hard rules, no soft weights, no threshold tuning. Ships rule-by-rule (A1 → A2 → A3); each rule is an independent ship gate.

## A1 — Anti-self-touch for SET / RECEIVE / DIG (smallest, ships first within A)

### Constraint

For each consecutive action pair `(prev, curr)` in a rally's action sequence:

```
if prev.team == curr.team
   and prev.player == curr.player
   and prev.type != BLOCK:
       # Volleyball-rule violation. Re-attribute curr.
       alt = closest same-team player to ball at curr.frame
             that is not prev.player
       if dist(alt, ball) <= ABSTAIN_BOUND:
           curr.player = alt
       else:
           curr.attribution_uncertain = True
```

`ABSTAIN_BOUND = 0.3` (normalized court distance). Picked as a sanity bound, not a tuning knob — any value in [0.2, 0.4] produces identical panel outcome.

### Why this is not Sub-2.B Phase 2 reborn

| Aspect | Sub-2.B Phase 2 (parked) | A1 |
|:--|:--|:--|
| Distance cap | 2× (binding constraint) | None — replaced by abstention |
| Signal mix | Soft convergence over 4 signals (type_fit, team_geometry, alt_ratio, confidence) | None — hard volleyball rule, no signals to tune |
| Confidence floor | < 0.3 skip | None — constraint is hard regardless of conf |
| Block exception | Strict | Strict (same) |
| When alt is implausible | "skip" recommendation in catalog | Abstain — mark `attribution_uncertain=True`, don't flip |

Two structural differences: drop the 2× cap (the cascade has alt_ratios up to 36×; the cap was the *binding constraint* that parked Sub-2.B, not a soundness requirement) and drop the soft signal mix (signals were where the hand-tuning happened).

### Pipeline location

New function `_attribution_volleyball_rule_pass` in `analysis/rallycut/tracking/action_classifier.py`. Called from `reattribute_players` after classifier-type is assigned, **before** existing Pass 2c (within-team proximity swap). Single forward pass; fixed-point (re-running is a no-op).

Env flag `USE_VOLLEYBALL_RULE_ATTRIBUTION=1` for default-OFF rollout. Default-OFF behavior is byte-identical to pre-workstream.

### Cascade worked example

- Pair `(176 attack p2, 225 dig p2)` → fires. Closest team-B alt at 225: track 1 at 0.041 normalized. Within 0.3 bound. **Flip 225 → track 1.** ✓ matches GT.
- Pair `(128 set p2, 176 attack p2)` → fires. Closest team-B alt at 128: track 1. **Flip 128 → track 1.** ✓ matches GT.
- Pair `(76 receive p2, 128 set p2)` — after the 128 flip — now `(76 p2, 128 p1)`, different players, rule doesn't refire. ✓

Three correct flips, no abstentions, the cascade resolves.

### Ship gate (must clear all)

1. 22-rally panel: same_team errors down by ≥ 2, total correct-rate not worse.
2. 73-video catalog regen: C-4 violation count drops by ≥ 80 % on rallies where at least one same-team back-to-back pair has `alt_ratio ∈ [2×, 50×]`.
3. Fleet F1 (action attribution) not worse than the current baseline at the time of A/B (currently 0.833 resolved-only / 0.930 all-GT pending refresh per [[action_gt_decouple_2026_05_12]] — measured at A/B time, not pinned here).
4. Abstention rate ≤ 5 % of contacts.

### Falsifiable failure modes

- Panel correct-rate drops → NO-SHIP.
- Abstention rate > 5 % → threshold or rule needs revisiting; NO-SHIP this round.

### Tests

`analysis/tests/unit/test_volleyball_rule_attribution.py`:
- Cascade fixture (frames 128, 225 from rally `a0881d82`).
- F3-shape fixture (cross-team violation).
- Block-exception fixture.
- Abstention fixture (no plausible alt).
- Multi-pair within one rally.

A/B harness: existing `measure_pid_accuracy.py` + the catalog regen script (`analysis/scripts/catalog_c4_violations.py`).

## A2 — Pose-driven attribution for ATTACK

### Constraint

For each contact predicted as `ATTACK`, compute for each same-team candidate:
- wrist-keypoint → ball distance (normalized).
- attack-pose indicator: arm above shoulder (wrist Y < shoulder Y in image coords) AND body in jump (ankle Y delta from prior frames > jump-threshold).

Selection: prefer candidate with `attack_pose=True` AND nearest wrist-to-ball distance. If no candidate has `attack_pose=True`, fall back to nearest-2D (current behavior).

### Why this fixes F3

In F3, the occluded far-side attacker has wrist-keypoint detected near the ball with arm-raised pose; the near-side blocker has wrist near ball but blocker-pose (stationary, arms up, no jump). Pose-discrimination + wrist-position together select the actual attacker.

### Probe-first kill-switch

Before building A2, run a 10-attack probe (including F3) to verify pose-wrist-to-ball is more discriminating than bbox-center-to-ball. **If probe shows < 50 % disambiguation improvement, NO-SHIP A2 and skip to A3.**

Probe script: `analysis/scripts/probe_pose_attack_attribution.py`. Inputs: 10 hand-picked attacks with known GT. Output: per-case GT alignment for bbox-center selection vs pose-wrist selection. Disambiguation improvement = `(pose_correct - bbox_correct) / pose_evaluable_n`.

### Pipeline location

`_attribution_attack_pose_pass` in `action_classifier.py`. Runs after A1's volleyball-rule pass. Env flag `USE_POSE_ATTACK_ATTRIBUTION`.

### Pose data source

Pose inference needs to be available at attribution time. Three options, decided in plan:
1. Inline at `contact_detector` (cost: ~5–10 ms/contact × N contacts).
2. Pre-batch on rally frames as a producer pre-step (cost: significant but once per rally, cached).
3. Reuse pose data from existing `motion_pose_experiment` / `contact_classifier_pose_experiment` infrastructure if available.

### Ship gate

1. 22-rally panel: cross-team errors at net-contacts down by ≥ 2 (F3-shape).
2. Pose-fallback rate ≤ 20 % of attacks (otherwise pose is unreliable on this corpus → NO-SHIP).
3. ATTACK action attribution F1 not worse than current.
4. Added per-rally latency ≤ 50 ms.

### Falsifiable failure modes

- Pose detection unreliable on volleyball footage → many fallbacks → NO-SHIP.
- Pose picks wrong wrist → A2 regresses → NO-SHIP.

## A3 — BLOCK type reclassification

### Constraint

For each contact predicted as `ATTACK`, reclassify to `BLOCK` iff all four hold:
- (a) player at-net (court-y in near-net band, parametrized per court calibration).
- (b) wrist above net-height (pose-keypoint Y).
- (c) ball direction-change at contact ≤ 90° (ball deflected, didn't reverse fully).
- (d) contact is opposing-team response to attacking-team's set/attack within the possession.

### Why this fixes F5

Frame 184 "attack p2 A": team B was in possession (receive at 150), p2 is at the net, ball didn't fully reverse — all three structural cues match a block, not an attack. The fourth condition (opposing-team response) confirms.

### Pipeline location

`_action_classifier_block_reclassification_pass` in `action_classifier.py`. Runs after type prediction. Env flag `USE_BLOCK_RECLASSIFICATION`.

### Ship gate

1. 22-rally panel: F5-shape patterns (mid-rally cross-team "attack" that's really block) down by ≥ 1.
2. Reclassification rate ≤ 3 % of predicted attacks (over-eager → NO-SHIP).
3. Block recall up; attack precision not significantly down.

### Risk and mitigation

Block recall is currently 10.8 % ([[contact_detection_fn_v1_2026_05_12]]). Most missed blocks are not detected at all (no contact). A3 doesn't add detection; it only reclassifies existing attacks. So block recall improves modestly (subset of blocks that are detected as attacks). Risk: over-reclassifying real attacks at the net. Mitigation: tight gating (all four conditions must hold).

---

# Workstream B — F4 dedupe symmetry

### Constraint

Extend `repair_action_sequence.Rule 3`: if two consecutive serves AND exactly one is synthetic, convert the non-synthetic serve to RECEIVE. Current behavior when both are non-synthetic (convert the second to RECEIVE) is unchanged.

### Scope

One commit, one unit test, ships standalone. No env flag — closed-form rule extension; the existing Rule 3 already runs in production.

### Tests

Add fixture for F4-shape pattern (synth + real serve pair) to `test_repair_action_sequence.py`.

### Ship gate

1. F4 rally `88a04529` resolves on panel (real action at frame 111 reclassified as RECEIVE).
2. Existing non-synth dedupe behavior unchanged.

---

# Workstream C — C-5 mid-possession crossover audit

### Constraint

New invariant in `analysis/rallycut/tracking/coherence_invariants.py`:

```
def check_c5_mid_possession_crossover(actions) -> list[Violation]:
    for i in range(len(actions) - 1):
        if (actions[i].team != actions[i+1].team
            and actions[i].type not in {attack, serve, block}):
            fire C-5 violation
```

Detection-only. Bumps `audit-coherence-invariants` CLI to report C-1..C-5 (currently C-1..C-4).

### Why useful

Surfaces F3 / F5 patterns fleet-wide. C-2 currently punts on mid-possession crossovers; C-5 closes that audit gap. Composes with the [[coherence_invariants_v1_2026_05_10]] family. Produces fleet baseline for prioritizing A2 / A3 by prevalence.

### Pipeline location

New function in `coherence_invariants.py`; wired into `run_all` dispatch alongside C-1..C-4. CLI gets `--c5` flag (and `--all` includes it). Catalog script (`catalog_c4_violations.py`) gets a parallel C-5 catalog if useful (optional in plan).

### Tests

`analysis/tests/unit/test_coherence_invariants.py::TestC5*`:
- Cross-team after attack/serve/block → no fire (legal).
- Cross-team after receive/set/dig → fire.
- Multiple crossovers in one rally.
- Same-team transitions → no fire.

### Ship gate

1. Tests pass.
2. Fleet baseline produced (catalog row count for C-5 across 73 videos).
3. F3 and F5 rallies fire C-5 in their respective frames.

---

# Sequencing and rollout

Order: **C → B → A1 → A2-probe → A2 → A3.**

1. **C first** — get fleet visibility on F3/F5 prevalence before designing A2/A3 rollout priority. ~1–2 days.
2. **B second** — closed-form, low-risk, single commit. ~1 day.
3. **A1 third** — smallest A-rule, falsifiable in days, fixes the cascade directly. ~3–5 days incl. A/B.
4. **A2-probe** — gate the A2 build on the 10-attack pose-disambiguation probe. ~½ day.
5. **A2 fourth** (if probe passes) — pose-driven attribution. ~1–2 weeks incl. pose plumbing.
6. **A3 fifth** — block reclassification. ~1 week.

Each workstream is shippable independently. C and B can land in any order; A1 depends on neither.

# Why this design avoids the past traps

- **Not PGM-shaped.** No soft factor weights. The volleyball-rule is hard; pose-wrist-distance is a single signal; block-reclass is four hard conditions ANDed together.
- **Not Sub-2.B-shaped.** A1 is not a downstream repair pass with a tuned distance cap — it's an upgrade of the attribution stage itself, the cap is replaced by abstention, and the soft signal mix is replaced by the hard volleyball rule.
- **Not contact-detection-FN-shaped.** Not threshold tuning on existing signals. A2 introduces a *new* signal (pose); A1 uses a hard rule, no thresholds; A3 introduces structural reclassification.

# Composes with

- [[coherence_invariants_v1_2026_05_10]] — C-5 extends the C-1..C-4 family.
- [[coherence_repair_sub_2b_2026_05_13]] — A1 replaces parked Phase 2 with a structurally different mechanism.
- [[joint_attribution_pgm_2026_05_12]] — same lesson applied: avoid soft weights.
- [[contact_detection_fn_v1_2026_05_12]] — out of scope for this design; cascade FN frames 191/340 belong there.
- [[adaptive_candidate_window_v30_2026_05_11]] / [[attribution_team_chain_v1_2026_05_11]] — A1 lives upstream of Pass 2c in `reattribute_players`.
- [[serve_peak_prepend_v13_2026_05_11]] / synth-serve placement — B closes the synth+real dedupe gap left by these.

# Files (anticipated)

**Code:**
- `analysis/rallycut/tracking/action_classifier.py` (A1, A2, A3 functions)
- `analysis/rallycut/tracking/coherence_invariants.py` (C-5)
- `analysis/rallycut/tracking/action_sequence_repair.py` (B Rule 3 extension)

**Scripts:**
- `analysis/scripts/probe_pose_attack_attribution.py` (A2 probe)

**Tests:**
- `analysis/tests/unit/test_volleyball_rule_attribution.py` (A1)
- `analysis/tests/unit/test_pose_attack_attribution.py` (A2)
- `analysis/tests/unit/test_block_reclassification.py` (A3)
- `analysis/tests/unit/test_action_sequence_repair.py` (B fixture addition)
- `analysis/tests/unit/test_coherence_invariants.py::TestC5*` (C)

**Plans (separate docs via writing-plans):**
- `docs/superpowers/plans/2026-05-13-c5-mid-possession-crossover-audit.md`
- `docs/superpowers/plans/2026-05-13-f4-dedupe-symmetry.md`
- `docs/superpowers/plans/2026-05-13-a1-volleyball-rule-attribution.md`
- A2 / A3 plans authored after A1 ships and the pose probe is run.

# Production impact

All workstreams ship default-OFF (env-flag-gated where applicable). Default-OFF behavior is byte-identical to pre-workstream. Each workstream's env flag flips independently.

Env flags introduced:
- `USE_VOLLEYBALL_RULE_ATTRIBUTION` (A1)
- `USE_POSE_ATTACK_ATTRIBUTION` (A2)
- `USE_BLOCK_RECLASSIFICATION` (A3)

C-5 is detection-only (no production behavior change). B is a closed-form rule extension (no flag).
