# Upstream-Pipeline Bottleneck Validation Probe — 2026-05-20

## TL;DR

After Sub-lever 1 NO-SHIP ([[attribution_sub_lever_1_no_ship_2026_05_20]]),
the four-prong campaign in [[attribution_headroom_decomposition_2026_05_20]]
needs a sanity check: the same "audit ceiling overshoots realistic recoverability"
pattern is likely to recur on Sub-levers 2/3. User has stated a competitor
achieves near-perfect attribution with classical ML, so the saturation
verdict from [[attribution_ceiling_2026_05_14]] is wrong as stated — it's
saturation against OUR current upstream pipeline quality, not against the
classical-ML signal level intrinsically.

This probe **rigorously validates which upstream pipeline layer is the
binding bottleneck** by measuring per-layer dual ceilings (oracle + realistic
intervention) on all ~296 wrong-attribution contacts on trusted-32. Output:
a ranked investment decision with confidence-weighted ceilings, surfacing
projection risk before any new investment.

## Background

### The Sub-lever 1 trap (avoid repeating)

The Sub-lever 1 audit measured "GT player at scorer rank-1 under
`expected_team=None`" on 122 B-only flip-targets and found 28 cases that
should be recoverable. The realistic intervention (`SCORER_CHAIN_FALLBACK`:
prefer higher-confidence pick) recovered only 4 (+0.26pp attribution,
Gate 2 -5 vs needed -20). The failure mode: **rank-1 ≠ confidence-leader.**
When the chain-aware scorer was wrong, its wrong pick had HIGHER confidence
than the no-chain pass's right pick (`team_matches_expected=1.0` produced
high prob on the wrong-team candidate). The audit measured the wrong
ceiling.

### Saturation verdict needs re-evaluation

[[attribution_ceiling_2026_05_14]] declared "feature engineering at the
geometric/proximity/pose signal level is empirically exhausted." Three
triangulations (learned-attribution +0.8pp, 3D projection +0.4pp,
Pose v2 -2.9pp) converged. But user reports a competitor achieves
near-perfect attribution with classical ML — so the saturation is against
OUR upstream pipeline (player tracker / candidate generation / contact
frame regression / ball tracking / GT scale), NOT against the classical-ML
signal level intrinsically.

The B1 ceiling probe already showed where: of 122 B-only flip-targets,
**40% fail upstream of the scorer** — 25% `scored_but_dropped` (player
tracker has no bbox within ±5 of contact), 15% `not_in_candidates` (GT
player never enters the candidate list). These were not the focus of
prior work; this probe corrects that.

## Scope and constraints

- **Classical ML only.** No VLM / vision embeddings. Per user constraint.
- **Substrate:** all ~296 contacts on trusted-32 (243 rallies with attribution
  GT) where pipeline `playerTrackId` ≠ GT `resolved_track_id`. Excludes
  unmatched cases (contact-detection FNs are a separate workstream).
- **Methodology:** dual ceiling per layer (oracle + realistic-intervention)
  to avoid the Sub-lever 1 projection-trap.
- **Output:** a ranked investment decision with confidence-weighted
  ceilings.

## Per-layer methodology

### L1 — Player-tracker contact-coverage

*Fail definition:* GT player has no bbox in `player_positions` within ±5
frames of GT contact frame (so `_find_pos` in scorer's `extract_features`
returns None and the candidate is dropped from scoring).

*Oracle ceiling:* for each wrong-attribution contact, scan the rally's
`player_positions` for ANY bbox for the GT player (any frame). If found,
substitute that bbox at the contact frame, re-run scorer with chain context
(production conditions), count cases where the re-scored pick = GT.

*Realistic intervention ceilings:*
- **R1.a — widen `_find_pos` tolerance from ±5 to ±10 frames.**
- **R1.b — widen to ±15 frames.** (Diminishing returns; may pollute other
  contacts.)
- **R1.c — interpolate bbox across short gaps** (≤10 frames between
  GT-player-tracked frames around contact, linear interpolation).
- **R1.d — detect ID-switch:** GT player tracked under a DIFFERENT
  `track_id` near contact; would within-rally repair recover it?

*Failure-mode categorization per contact:* off-screen / never-tracked /
short-gap / ID-switch / other.

### L2 — Candidate generation in `contact_detector`

*Fail definition:* GT player is not in `contact.player_candidates` list.

*Oracle ceiling:* for each L2-fail contact, force GT player into the
candidate list, re-run scorer with chain context, count cases where pick
= GT.

*Realistic intervention ceilings:*
- **R2.a — relax K-nearest filter** from current K to K+2 (read actual K
  from `contact_detector.py`).
- **R2.b — identify the eliminating rule per contact** (distance threshold,
  court side, on-screen, etc.) by tracing through the candidate-generation
  code path; measure recovery per relaxed rule individually.

*Output:* per contact, the rule that excluded GT player; per-rule recovery.

### L3 — Contact-frame regression accuracy

*Fail definition:* `|predicted_contact_frame − GT_contact_frame| > 0` for
wrong-attribution contacts.

*Oracle ceiling:* for each wrong-attribution contact, substitute the GT
contact frame as the contact frame, re-extract `CandidateFeatures` at GT
frame (positions + ball at GT frame), re-run scorer with chain context.
Count cases where pick = GT.

*Realistic intervention ceiling:* histogram of `|Δframe|` across all
trusted-32 contacts (correct + wrong). Correlate `|Δframe|` with
attribution error rate to test whether frame error is causal. No
retraining within this probe (out of scope; would be a follow-up).

*Output:* `|Δframe|` distribution; oracle recovery count; correlation
coefficient and quantile-by-quantile attribution error.

### L4 — Ball-tracking accuracy at contact

*Fail definition:* predicted `(ballX, ballY)` at contact frame differs
from GT `(ballX, ballY)` by Euclidean distance > 0 (any error).

*Prerequisite:* requires GT ball positions. Per memory, 43 rallies have
2D ball GT; overlap with trusted-32 (243 rallies) needs measurement upfront.
If overlap < 30 rallies, L4 oracle ceiling is reported with the caveat
"partial corpus, n=X rallies."

*Oracle ceiling:* for each wrong-attribution contact in the overlap, force
GT ball position into `extract_features` for the contact, re-score with
chain context. Count cases where pick = GT.

*Realistic intervention ceiling:* ball-position confidence at contact
frame, correlate with attribution error rate. WASB retraining out of
scope.

*Output:* overlap size; oracle recovery on overlap; confidence-vs-error
correlation.

### L5 — GT-scale learning curve

*No oracle/realistic split — this IS the realistic measurement.*

Train per-action scorer GBM on 25/50/75/100% of trusted GT (stratified by
action_type to preserve class balance). For each fraction, measure LOO CV
accuracy. Plot learning curve per action. Extrapolate to 2× and 5× current
GT size using a simple exponential / power-law fit.

*Decision rule:*
- If curve plateaus before 100% → more GT won't help; signal-limited.
- If still sloping at 100% (positive second derivative) → GT IS the
  bottleneck; labeling investment justified.
- If sloping per-action varies → identify which action types most benefit
  from more GT.

*Output:* learning curve plot + extrapolation table + per-action verdict.

### L6 — Team-chain accuracy

*Fail definition:* chain-derived `expected_team` for a contact ≠
GT-derived `expected_team`. GT-derived is computed by walking
`rally_action_ground_truth.resolved_track_id` and team_assignments through
the rally, marking each contact with the canonical team based on prior GT
contacts.

*Oracle ceiling:* for each wrong-attribution contact where GT-derived ≠
pipeline-derived expected_team, substitute the GT-derived expected_team
into the scorer's `team_matches_expected` feature, re-score, count cases
where pick = GT.

*Realistic intervention ceilings:*
- **R6.a — higher confidence threshold for chain propagation:** drop
  chain context (use `expected_team=0.5`) when chain certainty < X.
  Sweep X over {0.5, 0.6, 0.7, 0.8}.
- **R6.b — anchor chain to GT serve detection:** more conservative chain
  reseeding.
- **R6.c — chain-error origin decomposition:** for each chain-wrong
  contact, identify which prior contact's wrong attribution propagated
  forward. This reveals whether chain errors come from upstream
  attribution misses (L1/L2/L3 root cause) or from chain-propagation
  logic.

*Output:* chain vs GT-derived disagreement rate; oracle recovery;
chain-error origin attribution.

## Aggregator + decision framework

`analysis/scripts/aggregate_upstream_bottleneck_2026_05_20.py` consumes
the 6 per-layer JSONs and produces:

### Per-layer table

| Layer | Oracle ceiling | Realistic ceiling | Gap | Cost estimate | Confidence |
|---|---:|---:|---:|---|---|

### Multi-layer-fail Venn analysis

For each wrong-attribution contact, record which layers it fails at.
Categorize:
- **Single-layer-fail:** contact only fails at one layer. Layer's oracle
  ceiling counts this fully.
- **Multi-layer-fail:** contact fails at 2+ layers. Fixing only one layer
  may not recover it (confounded). Report multi-fail count per layer pair.

Decision rule penalty: a layer's effective ceiling is
`single_fail_count + 0.5 × multi_fail_count_with_each_other_layer`.

### Final ranking

`rank_score = realistic_ceiling × (1 − gap_ratio) / cost`

Where:
- `gap_ratio = (oracle − realistic) / oracle` (close to 0 = trustworthy,
  close to 1 = projection-trap candidate).
- `cost` ∈ {1 (config tweak), 3 (small classical-ML retrain), 10 (full
  pipeline component replacement), 100 (multi-week investment)}.

### Decision output (summary.md)

- **Top recommendation:** "Invest in L_X (realistic ceiling N, cost C,
  confidence high)."
- **If L5 still slopes:** "GT IS the bottleneck — invest in M more labels
  per action."
- **If all realistic ceilings <10 contacts:** "We are near the true
  classical-ML ceiling against current data; next signal class must shift
  (out of scope per user constraint)."
- **Confounding warnings** for multi-layer-fail contacts.

## Output deliverables

- `analysis/scripts/probe_upstream_L1_player_tracker_2026_05_20.py`
- `analysis/scripts/probe_upstream_L2_candidate_gen_2026_05_20.py`
- `analysis/scripts/probe_upstream_L3_contact_frame_2026_05_20.py`
- `analysis/scripts/probe_upstream_L4_ball_tracking_2026_05_20.py`
- `analysis/scripts/probe_upstream_L5_gt_scale_2026_05_20.py`
- `analysis/scripts/probe_upstream_L6_team_chain_2026_05_20.py`
- `analysis/scripts/aggregate_upstream_bottleneck_2026_05_20.py`
- `analysis/reports/upstream_bottleneck_2026_05_20/L{1..6}.json`
- `analysis/reports/upstream_bottleneck_2026_05_20/summary.md`
- `analysis/reports/upstream_bottleneck_2026_05_20/per_contact_failures.csv`

## Risk register

- **Confounded recoveries.** Many contacts fail at multiple layers; fixing
  only one may not recover them. Aggregator MUST surface this via the
  multi-layer Venn analysis and penalize confounded-only ceilings.
- **L4 partial corpus.** If ball-GT/trusted-32 overlap is small (<30
  rallies), L4 oracle ceiling is under-sampled. Report on partial subset
  only, flag in summary.
- **L5 small-sample classes.** BLOCK (n≈28 in trusted-32) won't have a
  usable learning curve. Skip BLOCK at low-fraction points; report only
  100% accuracy with caveat.
- **L6 GT-derived chain confound.** If we use pipeline `actions_json` to
  walk the chain, errors compound. Use `rally_action_ground_truth.resolved_track_id`
  directly to derive the canonical chain.
- **Substrate drift.** Trusted-32 DB state is currently post-v13 (NO-SHIP
  revert). Probes read live DB; record `actions_pipeline_version` of every
  rally consumed and warn on mixed-vintage data.
- **Aggregator-only false confidence.** The ranking is only as good as the
  per-layer measurements. If a layer's realistic intervention is the wrong
  choice (e.g., we widen the player-tracker tolerance when the actual fix
  is a better detector), the ranking misleads. Mitigation: per-layer
  realistic interventions are explicit + multiple; we report a range, not
  a single number.

## Out of scope

- Contact-detection FN (separate workstream; pre-attribution).
- WASB ball-tracker retraining (separate; affects multiple downstream
  components).
- Player-tracker model retrain (yolo11m, custom-trained ReID on volleyball
  crops) — would come AFTER this probe identifies L1 as a top lever.
- Sub-lever 2 (coherence reranker), Sub-lever 3 narrow widening, WS-2
  (GBM prev-action) from [[attribution_headroom_decomposition_2026_05_20]]
  — those should be revisited only AFTER this probe reranks the campaign.
- VLM probe per [[attribution_ceiling_2026_05_14]] — user-constrained out.

## Related

- [[attribution_sub_lever_1_no_ship_2026_05_20]] — the NO-SHIP that
  surfaced the projection-trap pattern. This probe's methodology
  (dual ceiling) directly addresses that failure mode.
- [[attribution_headroom_decomposition_2026_05_20]] — four-prong campaign
  this probe will rerank.
- [[attribution_ceiling_2026_05_14]] — prior "saturation" verdict to
  re-evaluate against upstream pipeline quality.
- [[trusted_attribution_corpus]] — 32-video supervision substrate.
- [[contact_fn_investigation_2026_05_17]] — contact pipeline this builds
  on.
