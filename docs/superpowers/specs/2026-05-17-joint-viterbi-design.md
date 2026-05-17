# Joint Viterbi over (action_type, player_id) — Design (2026-05-17)

## Context

Post-v3.2 ship, the dynamic attribution scorer hits 91.6% matched
accuracy on trusted-29. The remaining residual is dominated by:

1. **Upstream contact-detection FN** (~18% of GT rows unmatched)
2. **Coupled-regression risk** when bolting new rules onto the 12-stage
   classify→repair→viterbi→reattribute→scorer cascade (proven by the
   3rd-contact rule's C-2 +22 regression in [[repair_cascade_audit_2026_05_17]])

The architectural fix is **joint structured prediction**: replace the
cascade of action-type repair rules + per-action scorer + sequence
override with a single Viterbi over the joint `(action_type, player_id)`
state space. Hard volleyball constraints encoded as near-zero transition
probabilities; soft typical-transitions encoded as priors; emissions from
the existing learned models.

## State space

For each contact i:
- `action_type ∈ {SERVE, RECEIVE, SET, ATTACK, DIG, BLOCK}` (6 types)
- `player_track_id ∈ contact.player_candidates` (typically up to 4 tracks)
- Joint state size ≤ 24 per contact

For an N-contact rally: O(N × 24²) Viterbi DP = trivial cost (~5K ops
for 10 contacts).

## Emissions

```
P(observe | state=(at, tid)) = P_classifier(at) × P_scorer(tid | at)
```

- `P_classifier(at)`: full distribution from `ActionTypeClassifier.predict_proba` (added in this session, lines 678-702 of `action_type_classifier.py`).
- `P_scorer(tid | at)`: per-action-type GBM from
  `DynamicAttributionScorer.score(at, candidates)`, normalised so the
  sum over candidates equals 1 within each action_type.

## Transitions

Hard constraints (transition prob → 0.001):
- Net-crossing action (SERVE / ATTACK) MUST flip team possession
- Non-net-crossing action MUST stay on same team

Soft constraints (multiplicative discounts):
- Same-player back-to-back: × 0.05
- 3rd-contact-on-same-team-not-ATTACK: × 0.10

Typical-transition prior matrix (hand-coded in
`_TYPICAL_NEXT_GIVEN_PREV`): encodes empirical priors like
SERVE→RECEIVE 0.85, RECEIVE→SET 0.75, SET→ATTACK 0.85, etc. Will be
refined from trusted-29 GT statistics in Phase 2.

## Phases

### Phase 1A — Algorithm skeleton ✅ (this session)

- `rallycut/tracking/joint_viterbi.py`: `joint_viterbi()`,
  `StateCandidate`, `transition_log_prob()`
- Smoke test on synthetic 4-contact rally → correctly picks
  team-coherent path (avoids cross-team DIG, picks ATTACK as 3rd
  contact, avoids same-player back-to-back)

### Phase 1B — Real-data smoke test ✅ (this session)

- `scripts/probe_joint_viterbi_2026_05_17.py`: pulls a rally from DB,
  builds emissions from real classifier + scorer, runs Viterbi
- Verified on kiki r7 — produces volleyball-rule-coherent output
  (every possession alternates teams, 3rd contacts are ATTACK)
- Within-team attribution accuracy still bounded by existing scorer
  emissions (no free lunch)
- Added `ActionTypeClassifier.predict_proba` (full distribution API,
  needed for Viterbi emissions)

### Phase 1C — Production integration (next, ~hours)

- Add env-flag `USE_JOINT_VITERBI` (default OFF)
- When ON: route through joint Viterbi *instead of* repair cascade
- Output a `RallyActions` for downstream consumers
- Keep cascade as fallback when emissions unavailable

### Phase 2 — A/B on trusted-29 (next, ~hours)

- Measure attribution accuracy + coherence violations vs current
  pipeline
- Decision criteria: ship if matched accuracy ≥ current AND
  coherence violations ≤ current

### Phase 3 — Cascade simplification (longer)

When Phase 2 validates, deprecate:
- Most of `repair_action_sequence` (only structural repairs remain)
- `viterbi_decode_actions` (subsumed by joint Viterbi)
- Most of `_apply_dynamic_scorer_attribution` (its work is now inside
  Viterbi emissions)
- `apply_sequence_override` (Viterbi's joint optimum supersedes
  MS-TCN++ argmax)

### Phase 4 — Learned transition probabilities (future)

Hand-coded priors are a starting point. With trusted-29 GT:
- Compute empirical action-type transition frequencies per
  (prev_action_type, this_action_type, count_on_team) bin
- Replace hand-coded `_TYPICAL_NEXT_GIVEN_PREV` matrix

### Phase 5 — Synthetic-contact insertion (future)

When the team-chain rule conflicts with the observed sequence (e.g.,
team A has only 2 contacts before team B contacts), the joint Viterbi
detects but can't repair. A new state representing "missed contact"
could be inserted to extend the search space, jointly inferring
contact-FN frames as part of the optimisation.

## Files added this session

- `analysis/rallycut/tracking/joint_viterbi.py` — algorithm
- `analysis/scripts/probe_joint_viterbi_2026_05_17.py` — real-data smoke test
- `analysis/rallycut/tracking/action_type_classifier.py` — added `predict_proba` method (API for Viterbi emissions; also useful for any future structured-prediction work)

## Status

Phase 1A + 1B complete. Phase 1C (production integration with env-flag)
ready to start when next session resumes.

## Architectural anchor

[[repair_cascade_audit_2026_05_17]] — the post-v3.2 audit that
motivated this work, including the cascade analysis and the case
against the 3rd-contact rule patch.
