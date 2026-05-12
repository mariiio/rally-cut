# Joint Attribution PGM (Phase A + Absent States) — 2026-05-12

## Origin

This brainstorm followed the contact-detection FN reduction workstream
(Phase 1 + Phase 1.5 both NO-SHIP — see
`contact_detection_fn_v1_2026_05_12.md`), which conclusively showed that
threshold tuning at the contact detector cannot move action recall on this
corpus. The morning session had brainstormed three attribution-architecture
framings; this spec realizes Framing 1 (soft probabilistic joint inference
with absent-actor latent states), grounded in:

- The morning's per-error probe (24 attribution errors broken down: 9
  with GT in candidates at rank 1-2 but pipeline overruled, 5 with GT
  absent from candidates entirely, 6 with GT at low rank, 4 contact-
  detection misses)
- The user's stated principle: attribution should be resilient to missing
  contacts (a missing contact should not cascade-fail other contacts'
  attributions)
- The Phase 1 v2 NO-GO post-mortem: hard-rule beam search amplifies
  upstream signal noise; soft factors with explicit uncertainty are the
  fix

The morning identified Framing 1 as the right architecture but deferred
its implementation pending contact-detection ladder exhaustion. That
ladder is now exhausted.

## Goal

Replace `reattribute_players` with a per-rally Probabilistic Graphical
Model that performs joint MAP inference over per-contact player
assignments, with explicit absent-actor latent states. Hand-tuned soft
factors; exhaustive enumeration over the small joint configuration space.

**Targeted lift on the 22-rally fresh-GT panel:** +10pp ceiling
(recovers up to 14 of 24 attribution errors). Pre-ship gate is +5pp
absolute (60.3% → ≥ 65.3%) — well below the ceiling, accommodating
panel noise.

## Scope IN

- New module `analysis/rallycut/tracking/joint_attribution.py` exposing
  `joint_attribute_rally(rally: RallyContext) -> RallyAttribution`
- Companion modules `joint_attribution_factors.py` (pure factor
  functions) and `joint_attribution_weights.py` (typed weights config)
- Latent absent-actor states `{ABSENT_TEAM_A, ABSENT_TEAM_B}` per
  contact, with synthetic-action emission when MAP picks absent (writes
  `playerTrackId=None`, `attribution_source="pgm_absent_team_X"`,
  `team` populated)
- Hand-tuned factor weights with a calibration sweep on the 22-rally
  panel (offline script, ~5 min runtime, locks in defaults)
- Integration: `redetect_all_actions.py` and the `reattribute-actions`
  CLI flip to PGM via `USE_JOINT_ATTRIBUTION` env flag (default-OFF
  initially; default-ON after panel validation)
- Per-contact posterior marginals exposed for downstream consumers
- New `attribution_source` field on each action: one of
  `"action_classifier"`, `"action_classifier_abstained"`,
  `"pgm_committed"`, `"pgm_absent_team_A"`, `"pgm_absent_team_B"`
- Measurement: re-uses existing `measure_attribution_fresh_gt.py` panel
  harness; add `measure_joint_attribution_ab.py` wrapper for
  pre-vs-post deltas
- Unit tests: factor truth tables, inference correctness, end-to-end
  synthetic rallies, integration env-flag branch
- A/B verdict report

## Scope OUT (deferred)

- Learned factor weights from GT (Phase B — bottlenecked on GT expansion)
- Per-action-type factor variants (Phase B)
- Junction Tree / loopy BP (Approach 2 — only if rallies grow >10 contacts)
- Cluster-then-attribute reframe (Framing 2 from morning — different
  architecture)
- Visual signal enrichment beyond what's in `playerCandidates`
  (re-running OSNet at attribution time — adds latency; can be added later)
- Block-recall improvements (orthogonal — Phase 1.7 contact-detection
  work)
- Contact-detection improvements (orthogonal)
- GT corpus expansion (separate workstream)

**Non-goal:** Not a rewrite of `action_classifier` or `contact_detector`.
The PGM operates on their outputs and the `team_assignments` /
`serving_team` fields already present in DB.

## Success criteria

| Gate | Threshold | Rationale |
|---|---|---|
| **G-A (Correct rate)** | ≥ +5pp absolute (60.3% → ≥ 65.3%) | Below this not worth the maintenance cost vs `reattribute_players` |
| **G-B (Cross-team errors)** | Reduces by ≥ 30% (17 → ≤ 12) | Dominant error type the PGM should help |
| **G-C (Same-team errors)** | Non-increasing (≤ 9) | Don't trade cross-team wins for same-team losses |
| **G-D (Unknown-team errors)** | Non-increasing (≤ 0) | Sanity check |
| **G-E (Per-action recall)** | No GT type's matched-rate drops by > 2pp | Type-specific regression guard |
| **G-F (Inference latency)** | ≤ 50ms per rally average | Performance budget |
| **G-G (Coherence audit)** | C-1, C-2, C-3 fleet-wide non-increasing | Cross-validation against ungated data |
| **G-H (Unit tests)** | All existing + new joint-attribution tests pass | |

**STOP conditions** (any halts the workstream):
- Correct rate regresses below 60.3%
- Same-team errors > 12 (33% increase)
- Inference timeout (any rally > 2 sec)

If +5pp threshold isn't met but no STOP triggered, ship as default-OFF
infrastructure with the verdict report; escalate to Phase B (learned
weights) as a follow-up workstream.

## Formal model

### Variables

For a rally with N contacts (typically 3-6), define N discrete random
variables:

```
X_t ∈ {P1, P2, P3, P4, ABSENT_TEAM_A, ABSENT_TEAM_B}    for t = 0..N-1
```

The `team_assignments` map (already in DB per rally) tells us which `Pi`
belongs to team A vs team B. The `serving_team` is treated as a known
input, not a latent variable.

### Joint distribution

```
P(X_0, ..., X_{N-1} | evidence) ∝ exp(
    Σ_t φ_unary(X_t)
  + Σ_t ψ_pairwise(X_t, X_{t+1})
  + Σ_t ψ_higher(X_t..X_{t+k})
)
```

We work in log-space throughout — no exp/normalization needed for MAP.

### Unary factors φ_unary(X_t = i) — per-contact evidence

Each contributes a log-likelihood for "contact t was performed by entity i":

| Source | Contribution to φ(X_t = Pi) | Contribution to φ(X_t = ABSENT_TEAM_X) |
|---|---|---|
| `playerCandidates` proximity rank | `log(1 / (rank_i + 1)) * w_proximity` (rank 1 best). If Pi not in candidates, `-LARGE` | `0` (absent doesn't claim proximity) |
| `playerCandidates` distance | `-distance_i * w_dist` (closer → less penalty) | `+best_team_X_distance * w_dist_team` (positive bonus proportional to team's nearest-tracked-player distance; far team → larger bonus for absent state, i.e., absent is more plausible when team's tracked players are far) |
| Visual similarity (cross-rally PID profile) | `cosine_sim(crop_i, profile_Pi) * w_visual` (only when crop available) | `0` |
| Pose endorsement | `log(P_pose_touching_i) * w_pose` (when pose model loads) | `0` |
| `action_classifier` prior PID | `log(0.6) * w_prior` if Pi == initial PID, else `log(0.1) * w_prior` (soft, low weight) | `log(0.05) * w_prior` (very small mass on absent unless evidence strong) |
| Action-type prior | `0` | `log(prior_absent_for_type) * w_action` (e.g., serves have higher absent prior) |

Hand-tuned starting weights:

```
w_proximity = 2.0
w_dist = 1.0
w_dist_team = 0.5
w_visual = 1.5  (when present)
w_pose = 1.5    (when present)
w_prior = 0.5
w_action = 1.0
```

These are starting points. The calibration sweep tunes them on the
22-rally panel.

### Pairwise factors ψ_pairwise(X_t, X_{t+1})

| Rule | Condition | Penalty |
|---|---|---|
| `R_no_back_to_back` | X_t == X_{t+1} (same player twice consecutively, both not absent) | `-w_back_to_back` (e.g., -3.0) |
| `R_back_to_back_block_attack` | Same player consecutive AND first action was a block | `+0` (allowed exception) |
| `R_alternating_teams_cross_net` | team(X_t) == team(X_{t+1}) AND a net crossing occurred between them | `-w_alternation` (e.g., -3.0) |
| `R_alternating_teams_same_side` | team(X_t) != team(X_{t+1}) AND no net crossing between them | `-w_team_consistency` (e.g., -2.0) |
| `R_absent_pair_penalty` | Both X_t and X_{t+1} are ABSENT_* states | `-w_absent_pair` (e.g., -1.5) |

### Higher-order factors

| Rule | Condition | Penalty |
|---|---|---|
| `R_3_contact_per_side` | 4+ consecutive contacts by the same team without a net crossing | `-w_3_contact` per extra contact (e.g., -4.0) |
| `R_serve_first` | X_0 not a member of `serving_team` (or X_0 is `ABSENT_TEAM_<other>`) | `-w_serve_first` (e.g., -3.0). If X_0 is `ABSENT_TEAM_<serving>`, small penalty (-0.5) |

### Net-crossing detection

Each contact has `courtSide` (`"near"`, `"far"`, `"unknown"`) in
`contacts_json`. A net crossing is detected when `courtSide` flips between
consecutive contacts (`unknown` neither flips nor confirms — treated as
"don't update"). This is the input to alternation rules.

### Action-type contextualization

Factor weights could vary by action type (e.g., serves stronger
`w_serve_first`; sets weaker `w_back_to_back`). For MVP, use global
weights. Per-action-type weights are Phase B.

## Inference algorithm

Exhaustive enumeration over all 6^N joint configurations:

```python
def joint_attribute_rally(rally: RallyContext) -> RallyAttribution:
    contacts = rally.contacts
    candidates = build_evidence(rally)  # per-contact unary scores per state
    n = len(contacts)
    state_domain = build_state_domain(rally.team_assignments)

    best_score = -inf
    best_config = None
    for config in itertools.product(state_domain, repeat=n):
        score = sum(candidates[t][config[t]] for t in range(n))
        for t in range(n - 1):
            score += pairwise_factor(config[t], config[t+1], contacts[t], contacts[t+1])
        for t in range(n - 2):
            score += three_contact_factor(config[t], config[t+1], config[t+2], contacts)
        score += serve_first_factor(config[0], rally.serving_team)
        if score > best_score:
            best_score, best_config = score, config

    return RallyAttribution(map=best_config, score=best_score, marginals=...)
```

**Per-rally cost:** N=6 → 6^6 = 47k configs × ~30 factor evaluations =
~1.4M operations. ~10-30ms in Python; ~3ms with numpy vectorization.

**Cap on rally length:** For N > 8 contacts (>1.7M configs), fall back to
**beam search with width 100**. Empirically equivalent to exhaustive for
our problem class but bounded compute. Logged so we can audit when
fallback fires.

**Marginals:** computed during enumeration by accumulating
`exp(score - max_score)` per state per contact. Used for confidence
reporting and debugging.

## Integration

### Current pipeline

```
contacts_json (from contact_detector)
  ↓
classify_rally_actions (initial per-contact attribution)
  ↓
reattribute_players (multi-pass post-hoc PID fixes)
  ↓
actions_json
```

### New pipeline with PGM

```
contacts_json
  ↓
classify_rally_actions (unchanged — produces initial PIDs as soft prior)
  ↓
joint_attribute_rally (NEW — exhaustive joint MAP)        IF USE_JOINT_ATTRIBUTION=1
                  OR
reattribute_players (LEGACY — kept for rollback)          IF USE_JOINT_ATTRIBUTION=0
  ↓
actions_json (with new attribution_source + attribution_confidence fields)
```

### Call sites

- `analysis/scripts/redetect_all_actions.py` — env-flag branch on
  `USE_JOINT_ATTRIBUTION`
- `analysis/rallycut/cli/commands/reattribute_actions.py` — same

### Synthetic-action emission

When MAP assigns `ABSENT_TEAM_A` or `ABSENT_TEAM_B` to contact t:

```json
{
  "frame": <contact frame>,
  "action": <action_classifier label>,
  "playerTrackId": null,
  "team": "A" | "B",
  "confidence": <action confidence>,
  "attribution_source": "pgm_absent_team_A" | "pgm_absent_team_B",
  "attribution_confidence": <pgm marginal>
}
```

Downstream consumers see `playerTrackId=None` (existing semantics for
"abstained"), `team` populated (so team-level stats work), and
`attribution_source` distinguishes PGM-absent from action_classifier-
abstained.

### attribution_source values

| Value | Meaning |
|---|---|
| `"action_classifier"` | Initial classifier confident in a player |
| `"action_classifier_abstained"` | Classifier had no confident pick |
| `"pgm_committed"` | PGM picked a tracked player (P1..P4) |
| `"pgm_absent_team_A"` / `"pgm_absent_team_B"` | PGM picked an absent state |

### Composition with existing systems

- **synthetic-serve** (already shipped): runs BEFORE
  `classify_rally_actions`. PGM sees the synthetic serve as just another
  contact. Composes cleanly.
- **assignment-anchor** (cross-rally matcher): PGM reads
  `team_assignments` and `serving_team` which derive from the matcher;
  doesn't write back. One-way data flow.
- **coherence audit** (C-1/C-2/C-3): audit reads `actions_json`. PGM's
  MAP should reduce coherence violations (it directly optimizes for rule
  compliance). Audit is a secondary success signal.

### Rollback

`USE_JOINT_ATTRIBUTION=0` reverts to existing `reattribute_players`
chain. Read at call time (matching existing
`RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN` pattern). No code rollback needed.

## Calibration loop (pre-ship)

Offline script `calibrate_joint_attribution_weights.py`. Uses
**coordinate ascent** (NOT cartesian-product grid search — 9 weights ×
5 values = 1.95M configs, too expensive):

1. 22-rally panel = calibration set
2. Initialize weights to hand-tuned starting values (the table above)
3. **Coordinate ascent**: in fixed cyclic order (`w_proximity`, `w_dist`,
   `w_dist_team`, `w_visual`, `w_pose`, `w_prior`, `w_action`,
   `w_back_to_back`, `w_alternation`, `w_team_consistency`,
   `w_absent_pair`, `w_3_contact`, `w_serve_first`):
   - Fix all other weights at their current values
   - Sweep this weight in 5 steps: -50%, -25%, current, +25%, +50%
     (multiplicative around current value; clamped to [0.1, 10.0])
   - Pick step that maximizes panel correct-rate; advance that weight
   - If no step beats current, leave unchanged
4. Repeat the coordinate-ascent cycle up to 3 times (typical convergence
   is in 1-2 cycles; stop early if no weight changes during a full cycle)
5. **Anti-overfit gate**: reject any single-weight change that improves
   panel by recovering only 1 case but degrades the rally with the next
   highest score (heuristic: if recall delta is +0.7pp = +1 case AND
   any single rally's correct count drops, consider it overfit)
6. Lock final weights as `DEFAULT_WEIGHTS` in
   `joint_attribution_weights.py`
7. Document the calibration trajectory (per-cycle weight changes +
   panel deltas) in the verdict report

Per-iteration cost: 5 panel evaluations × 13 weights × 3 cycles =
195 panel runs. Each panel run takes ~10s with PGM (22 rallies ×
~50ms each + harness overhead). Total ~30 min, ran once offline.

## A/B execution

```
1. Snapshot DB (820 rallies, contacts_json + actions_json)
2. Pre-PGM baseline: USE_JOINT_ATTRIBUTION=0; capture all gates
3. Apply PGM: USE_JOINT_ATTRIBUTION=1; re-run reattribute-actions on
   panel rallies
4. Post-PGM measurement: capture all gates with flag ON
5. Compute deltas vs pre-ship gates
6. Verdict: PASS (ship default-ON) / DONE_WITH_CONCERNS (ship
   default-OFF infrastructure) / FAIL (revert)
7. If FAIL: restore snapshot
```

## Files

**New production code:**

| File | Responsibility |
|---|---|
| `analysis/rallycut/tracking/joint_attribution.py` | `joint_attribute_rally(rally) -> RallyAttribution` public API; `RallyContext` dataclass (input bundle); `RallyAttribution` dataclass (MAP + marginals + score); evidence builder; exhaustive enumeration with beam-search fallback |
| `analysis/rallycut/tracking/joint_attribution_factors.py` | Pure functions: `_unary_proximity`, `_unary_distance`, `_unary_visual`, `_unary_action_prior`, `_pairwise_no_back_to_back`, `_pairwise_alternation`, `_higher_3_contact`, `_higher_serve_first`. Each takes simple typed inputs, returns log-likelihood. |
| `analysis/rallycut/tracking/joint_attribution_weights.py` | `FactorWeights` dataclass + `DEFAULT_WEIGHTS` constant (single source of truth for tuning). |

**Modified:**

| File | Change |
|---|---|
| `analysis/scripts/redetect_all_actions.py` | Read `USE_JOINT_ATTRIBUTION` env flag; branch between `reattribute_players(...)` (default) and PGM path |
| `analysis/rallycut/cli/commands/reattribute_actions.py` | Same env-flag branch |
| Action data model | Add `attribution_source` field (always populated; existing code defaults to `"action_classifier"` or `"action_classifier_abstained"`); add `attribution_confidence` field (optional) |

**New scripts:**

| File | Purpose |
|---|---|
| `analysis/scripts/calibrate_joint_attribution_weights.py` | Sweeps factor weights on 22-rally panel; outputs calibrated `DEFAULT_WEIGHTS` |
| `analysis/scripts/measure_joint_attribution_ab.py` | A/B harness wrapping `measure_attribution_fresh_gt.py`; runs OFF then ON; emits delta table |

**New unit test files:**

| File | Coverage |
|---|---|
| `analysis/tests/unit/test_joint_attribution_factors.py` | Truth tables for each factor (per-rule per-input-pattern). 30+ tests. |
| `analysis/tests/unit/test_joint_attribution_inference.py` | Exhaustive enumeration correctness; MAP-finds-known-best-config; beam-search-fallback test. |
| `analysis/tests/unit/test_joint_attribution_e2e.py` | Synthetic `RallyContext` → `joint_attribute_rally` → expected MAP. ~10 cases (clean rally, off-screen serve, missing-middle-contact, etc.) |
| `analysis/tests/unit/test_joint_attribution_integration.py` | Verify env-flag branch in `redetect_all_actions` invokes correct path; verify `actions_json` shape with `attribution_source` field |

## Testing strategy

- **TDD** for factor functions, inference correctness, fallback semantics
- **Property-based** (hypothesis): random `RallyContext` inputs, assert
  `joint_attribute_rally` always returns valid configuration with
  `score >= per_contact_initial_score` (joint MAP can't be worse than
  per-contact MAP)
- **Integration smoke test**: single panel rally end-to-end
- **Calibration validation**: any weight at sweep boundary is suspect
  (might want wider sweep — flag in report)
- **Regression guard**: existing `tests/unit/test_action_attribution_team_chain.py`
  and other reattribute_players tests pass when `USE_JOINT_ATTRIBUTION=0`

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Hand-tuned weights overfit 22-rally panel | +5pp gate is conservative (well below +10pp ceiling); calibration sweep step large; G-G fleet coherence audit cross-validates against ungated data |
| Inference latency exceeds 50ms on long rallies | Beam-search fallback at N>8; profile panel's longest rally during calibration; if any exceeds 50ms, lower beam threshold or vectorize with numpy |
| `attribution_source` field breaks downstream consumers | Audit consumers (web frontend, lambda export, match-stats aggregator) before shipping; make field optional in consumers' types initially |
| PGM picks ABSENT aggressively → many actions unattributed | `w_dist_team` and `R_absent_pair_penalty` discourage absent assignments; calibration sweep verifies absent-rate stays below 15% on panel |
| 22-rally panel too small to detect +5pp lift reliably | Acknowledged; G-G fleet coherence audit provides cross-validation; if panel ambiguous, ship as default-OFF infrastructure pending GT expansion |
| Calibration runs differently on different machines | Pin to specific commit; document weight versions in verdict report |

## Rollback

- All new code feature-gated behind `USE_JOINT_ATTRIBUTION`. Default OFF
  until panel validation passes.
- Post-deploy issues: `USE_JOINT_ATTRIBUTION=0` on host running
  `redetect-actions`, re-run on affected rallies. No DB schema migration;
  no code revert needed.
- `attribution_source` field is additive (existing code can ignore it).

## Out-of-scope hooks left in place

- `attribution_confidence` field per action (PGM marginal probability) —
  exposed but no consumer uses it yet. Future Phase B (per-action-type
  weight learning) needs it.
- `pgm_marginals` per action — optional debug output. Disabled by default
  to keep `actions_json` size sane.

## References

- Origin brainstorm: this session
  (`originSessionId: 791c9269-c44c-47e5-a4dc-283c0474c939`)
- Phase 1 contact-detection NO-SHIP context:
  `contact_detection_fn_v1_2026_05_12.md`
- Per-error probe used to estimate +10pp ceiling:
  `analysis/reports/attribution_baseline/per_error_classification_2026_05_12.json`
- Morning's joint-v2 NO-GO post-mortem (hard-rule beam search; lessons):
  `analysis/reports/attribution_baseline/joint_v2_2026_05_11.md`
- Existing `reattribute_players` implementation:
  `analysis/rallycut/tracking/action_classifier.py`
- Existing measurement harness:
  `analysis/scripts/measure_attribution_fresh_gt.py`
