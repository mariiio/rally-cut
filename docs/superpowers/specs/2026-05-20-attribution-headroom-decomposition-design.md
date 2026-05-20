# Attribution headroom decomposition — converged understanding (2026-05-20)

## TL;DR

The 2026-05-20 Rule-5 NO-GO triggered a proposal to replace the action+attribution
cascade with a joint beam-search decoder ([[pipeline_architecture_review_2026_05_20]]).
Two empirical probes killed that framing and surfaced a different design space:

- **Joint decoding has zero headroom** over independent single-axis correction on
  trusted-32 (264 baseline violations, `fixed_only_jointly = 0`).
- **The 2026-05-17 joint-Viterbi probe was already** over `(action_type,
  player_track_id)` with full-distribution emissions and GT-derived empirical
  transitions — the architecture proposed as "the unprobed variant" had in fact
  been probed and rejected (−6.6pp matched vs cascade).
- **The attribution lever decomposes into three independent sub-levers** with
  measured per-lever ceilings totalling ~115-130 of 264 violations (~44-49%
  reduction, ~88% → 91-92% attribution accuracy on trusted-31). None requires
  architectural-rethink-scale change.
- **A fourth lever (action GBM SET/DIG boundary)** owns the residual ~15 set→set
  violations as a separate, cheap workstream.

First implementation prong (this spec → writing-plans next): **Sub-lever 1 —
downstream-override audit**. 28 of 122 B-only flip-targets have the v2 scorer's
top-1 candidate equal to GT — i.e. some cascade stage AFTER the scorer is
overriding the correct pick. Identify and gate that stage. Zero new ML work,
~28 violations fixable.

## Background — why the joint-decoder framing was wrong

### Trigger

Re-validating Rule 5 (set→set same-side → second becomes attack) on v11 trusted-31
showed −5 net (2 right, 7 wrong on 10 GT-resolved cases). 6/10 had the FIRST set
mislabeled as DIG per GT — the violation was real but the root cause was upstream
(action GBM mistyping), not addressable by any rewrite rule. The user's
[[pipeline_architecture_review_2026_05_20]] memo escalated to architecture review
and floated a joint beam-search decoder over `(action_type, player, team)` with
transitions learned on the 32-video corpus.

### Pushback #1 — the "unprobed variant" claim is wrong

Recovered the deleted `joint_viterbi.py` and `probe_joint_viterbi_2026_05_17.py`
from commits `c426fe4d` and `81d1db52`. The 2026-05-17 probe:

- State space was **joint** `(action_type, player_track_id)` — not action-only.
- Emissions expanded the full grid of (action_type × candidate_player), with
  `emission_prob = action_GBM_proba[at] × normalized_scorer_proba[at][candidate]`
  — full distribution, not argmax.
- Transitions were **GT-derived empirical action priors** loaded via
  `JOINT_VITERBI_TRANSITIONS_PATH` env override.
- Hard team coherence via `_DISCOUNT_WRONG_TEAM_TRANSITION = 0.001`.

Result: 85.0% matched vs cascade's 91.6% (−6.6pp). The deletion-commit diagnosis:
"remaining gap is structural — synth-SERVE insertion, MS-TCN++ sequence override,
reattribute_players Pass 2 hard team-swap, emission calibration — not addressable
by transition learning." The architecture the user called "the unprobed variant"
had in fact been probed and lost.

### Pushback #2 — Oracle decomposition kills coupling hypothesis

Built `analysis/scripts/probe_violation_oracle_decomp_2026_05_20.py`. For each
v11 production violation on trusted-32 (243 rallies, 264 baseline violations),
counted residual under 4 conditions: baseline / Oracle-A (sub GT action_type) /
Oracle-B (sub GT playerTrackId) / Oracle-AB (sub both).

| Bucket | base | A_only | B_only | both | **joint** | neither |
|---|---:|---:|---:|---:|---:|---:|
| set→attack cross-team | 39 | 3 | 21 | 0 | **0** | 15 |
| attack→dig same-team | 32 | 1 | 17 | 1 | **0** | 13 |
| serve→receive same-team | 22 | 6 | 10 | 0 | **0** | 6 |
| set→set any-team | 16 | 15 | 0 | 0 | **0** | 1 |
| C-4 same-player back-to-back | 37 | 0 | 21 | 0 | **0** | 16 |
| C-5 mid-possession crossover | 118 | 17 | 51 | 2 | **0** | 48 |
| **TOTAL** | **264** | **42** | **120** | **3** | **0** | **99** |

Headline: **`fixed_only_jointly` is empty across every bucket**. No violation
requires correct action AND correct attribution simultaneously to fix; every
fixable violation is fixable by correcting one axis. This directly refutes the
"parallel architecture is root cause" framing.

Of the 99 "neither" cases, 62 have GT for both contacts but the violation
persists under Oracle-AB — these are residual issues in `teamAssignments` or
GT-resolver edges, not action/attribution errors and not in scope.

## The three attribution sub-levers (B1 ceiling probe)

Built `analysis/scripts/probe_scorer_rank2_ceiling_2026_05_20.py`. For each
B-only violation (120 violations → 122 flip-target contacts, since some
violations need both prev and curr flipped), determined which contact(s) the
fix relies on, then re-ran the v2 scorer for that contact and reported GT
player's rank.

| Outcome | n | % | Interpretation |
|---|---:|---:|---|
| **rank_1** | 28 | 23.0% | Scorer top-1 IS GT. Pipeline pick was overridden downstream of the scorer. Bug subset, not lift subset. |
| **rank_2** | 33 | 27.0% | GT is scorer's second pick. Coherence-aware reranker recovers. |
| **rank_3** | 10 | 8.2% | GT is third. Reranker reachable. |
| **rank_4+** | 2 | 1.6% | Trivial. |
| **scored_but_dropped** | 30 | 24.6% | GT player IS a candidate but `extract_features` returns None (no bbox within ±5 frames of contact). Player-tracker coverage problem at contact frame. |
| **not_in_candidates** | 18 | 14.8% | GT player absent from `playerCandidates`. Upstream candidate-generation gap. |

### Sub-lever 1 (FREE) — downstream-override audit

28 flip-targets where scorer-top-1 already equals GT, but the production
`actions_json` records a different player. Some cascade stage after the scorer
flipped the answer. Candidate culprits per [[repair_cascade_audit_2026_05_17]]:
`reattribute_players` Pass 2 hard team-swap; MS-TCN++ sequence override;
synthetic-serve placement.

Estimated effort: 1-2 days. Estimated lift: up to 28 violations recovered.
Lowest-cost, no new ML.

### Sub-lever 2 (CHEAP) — coherence-aware reranker

45 flip-targets (rank_2 + rank_3 + rank_4+ = 33 + 10 + 2) where GT is in
the scorer's ranked output beyond top-1. Build a post-scorer pass: when a
sequence-illegal configuration (any of the 6 coherence buckets) is detected,
attempt to swap the offending contact's pick to the next-best-scoring
candidate that resolves the violation. Gate on env flag, A/B against
cascade with LOO CV.

Estimated effort: 3-5 days. Estimated lift: up to 45 violations recovered
(43 from rank_2 + rank_3, +2 marginal from rank_4+).

### Sub-lever 3 (UPSTREAM) — player-tracker contact-frame coverage

30 flip-targets where the GT player IS in `playerCandidates` but bbox is
absent within ±5 frames of contact, so `_find_pos` returns None and the
candidate is dropped from scoring. Two minimal interventions:

- (a) Widen `_find_pos` tolerance from ±5 to ±10 or ±15.
- (b) Interpolate positions across short gaps when a track was active before
  and after the contact frame.

Estimated effort: 2-3 days. Estimated lift: up to 30 violations recovered
(if the widened-window scoring picks GT, which is empirically untested).

## The action-side sub-lever (WS-2) — prev-action GBM feature

The single bucket where Oracle-A dominates is set→set (15/15 fixable cases).
The Rule 5 trace empirically validates: first SET is mislabeled (typically as
DIG) by the v11 action GBM. Cheapest fix: add a categorical `prev_action_class`
feature to the GBM training inputs, retrain to v12, LOO-CV gate.

Estimated effort: 2-3 days. Estimated lift: up to 15-30 violations recovered
(set→set + likely some of A_only's 17 C-5 cases).

## Combined ceiling estimate

If all four levers land at their measured ceiling:

| Lever | Ceiling | Cost |
|---|---:|---|
| Sub-lever 1 (downstream-override audit) | ~28 violations | 1-2 days |
| Sub-lever 2 (coherence reranker) | ~45 violations | 3-5 days |
| Sub-lever 3 (position-coverage widening) | ~30 violations | 2-3 days |
| WS-2 (GBM prev-action feature) | ~15-30 violations | 2-3 days |
| **Total** | **~118-133 of 264 violations** | |

Trusted-31 attribution accuracy: **88% → 91-92%** (vs the joint-decoder
projection of 88% → 90-92% on much more architectural scope).

Hard ceilings:
- ~99 "neither" cases (38% of violations) unaddressable by these levers
  (team_assignments + GT-resolver edges).
- The original [[attribution_ceiling_2026_05_14]] visual-signal exhaustion
  verdict still holds — past ~92%, the VLM probe is the next workstream.

## Chosen first implementation prong: Sub-lever 1

User-selected 2026-05-20: downstream-override audit. Highest ROI (zero new ML),
clearest scope, surface area is a known cascade.

### Objective

For each of the 28 flip-target contacts where v2 scorer top-1 = GT, identify
which cascade stage between the scorer and persisted `actions_json` flipped
the player_track_id assignment to the wrong candidate. Decide whether a
targeted guardrail (e.g. "do not override scorer when scorer top-1 prob > θ")
can eliminate the override without regressing other cases.

### Approach

Instrument `classify_rally_actions` (and downstream attribution stages) to
emit a per-contact JSON trace recording `playerTrackId` at every stage
boundary. For the 51 affected rallies, run pipeline once with instrumentation
on, materialize per-contact stage history, and identify the override stage
per flip-target.

Cascade stages to instrument (per [[repair_cascade_audit_2026_05_17]]):

1. After contact detection (initial `playerTrackId` from `contact.player_candidates`)
2. After action-type classifier assigns action
3. After dynamic attribution scorer pick
4. After `repair_action_sequence` (rules 0-8)
5. After `viterbi_decode_actions`
6. After Pass 1 of `reattribute_players` (server exclusion)
7. After Pass 2 of `reattribute_players` (team-chain swap)
8. After MS-TCN++ sequence override (if applicable)
9. Final persisted `actions_json`

Output table:

| rally | contact_frame | action_type | stage_that_overrode | scorer_top1 | final_pick | gt_player |
|---|---|---|---|---|---|---|

### Decision tree from the audit

- **Single dominant override stage (>15/28 flip-targets):** propose a guardrail
  on that stage. Implementation + A/B in a follow-up plan.
- **Override is split across 2-3 stages:** evaluate per-stage cost/benefit. May
  yield a small guardrail per stage.
- **Override is diffuse (>4 stages):** flags a deeper cascade-design issue;
  escalate before adding guardrails (those would compose unpredictably per
  [[feedback_prefer_architecture_over_rules]]).
- **Scorer top-1 was correct but FINAL pick is also correct (i.e. no actual
  override occurred in production):** the violation in production output comes
  from upstream of the scorer's input (e.g. team_assignments wrong, causing
  the SAME player_id to resolve to a different team). Update the violation
  classification and re-decompose.

### Validation

Once a guardrail is identified, A/B against the v11 baseline on trusted-31
using `scripts/measure_attribution_trusted_29_2026_05_17.py` (or its
trusted-31 successor). Gate:

- Net non-regressive on attribution accuracy.
- Reduce baseline violation count in the affected buckets.
- Zero regressions on the 5 non-affected videos (mumu, keke, mame, veve,
  papa — the trusted-29 winners from v2 scorer A/B).

### Out of scope for this prong

- Sub-lever 2 (coherence reranker) — separate spec + plan once Sub-lever 1
  closes.
- Sub-lever 3 (position-coverage widening) — separate spec.
- WS-2 (GBM prev-action feature) — separate spec, can run parallel to
  Sub-lever 1 since it touches different code (`action_type_classifier.py`
  training, not `action_classifier.py` cascade).
- VLM probe — only justified if all four prongs land at ceiling and we want
  to push past ~92% attribution.

## Artifacts produced

- `analysis/scripts/probe_violation_oracle_decomp_2026_05_20.py` — oracle
  lever decomposition. Output: `analysis/reports/violation_oracle_decomp_2026_05_20/`.
- `analysis/scripts/probe_scorer_rank2_ceiling_2026_05_20.py` — B1 scorer
  rank ceiling. Output: `analysis/reports/scorer_rank2_ceiling_2026_05_20/`.
- This spec: `docs/superpowers/specs/2026-05-20-attribution-headroom-decomposition-design.md`.

## Related

- [[pipeline_architecture_review_2026_05_20]] — the queued architecture review
  that prompted this brainstorm. Should be updated post-converged-understanding
  to reflect: joint-decoder framing dropped, four independent levers identified.
- [[repair_cascade_audit_2026_05_17]] — prior architectural audit; its joint
  Viterbi NOT-WORTH-IT verdict was correct and load-bearing.
- [[attribution_ceiling_2026_05_14]] — visual-signal exhaustion verdict;
  bounds the four-prong ceiling.
- [[feedback_prefer_architecture_over_rules]] — durable rule that motivated
  this brainstorm-first-then-plan workflow.
