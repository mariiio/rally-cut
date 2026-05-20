# Chain-Asymmetry Diagnostic Probe — 2026-05-20

## TL;DR

[[upstream_bottleneck_findings_2026_05_20]] identified L6 team-chain
accuracy as the top-ranked attribution bottleneck (35 contacts recoverable
at cost 3, rank 11.67). Within that finding, a striking sub-pattern:
**of the 94 wrong-chain cases, 63 are pipe=A/gt=B vs 31 pipe=B/gt=A** —
the pipeline systematically over-assigns team A. Binomial test under H0
50/50: `p ≈ 0.001`. **The asymmetry is significant.**

This probe is a focused 1-day diagnostic to differentiate three
remaining hypotheses for the asymmetry, with explicit decision rules
that either lead to a small targeted fix or escalate to the full
chain-quality rewrite. No code changes within this workstream.

## Background

### Why a diagnostic before a rewrite

[[attribution_sub_lever_1_no_ship_2026_05_20]] documents the failure
mode of "build first, measure later": Sub-lever 1's audit predicted +28
violation recovery, the realistic intervention delivered +4. The
projection trap was that the oracle measured rank, but the realistic
intervention picked by confidence — different metrics.

For chain-quality interventions, the analogous risk is "fix the chain
walker, get a chain that's structurally different but accuracy doesn't
improve because the underlying bias is in the seed (serving team), not
in the walker." A diagnostic surfaces which layer to actually invest in.

### Chain-walker code reference

`analysis/rallycut/tracking/action_classifier.py:_compute_expected_teams`
(line 2968-3015). Key fragility points:

1. **Chain is seeded by the FIRST SERVE with `player_track_id >= 0`** (line
   2989-2992). If first serve is wrong-attributed → wrong seed team → ALL
   subsequent contacts have wrong expected_team.
2. **No re-anchoring** — chain walks forward, flipping on net-crossing
   actions (SERVE/ATTACK and possibly BLOCK). Never re-seeds on
   subsequent serves or attacks.
3. **Synthetic-serve handling** (line 3001-3007) re-uses original
   seed_team even when the synthetic serve is detected mid-rally, which
   may not match the actual mid-rally team.
4. **`_chain_integrity` is binary** (line 3018-3047): once broken (any
   UNKNOWN or non-seed synthetic action), permanently broken for that
   rally. No graded confidence tracking.

The 4-gate override predicate `_team_chain_override_allowed` (line 3050)
guards when chain context is allowed to override nearest-player
attribution. Sub-lever 1 found this fires "confidently wrong" frequently.

## Hypotheses

### H1 — Serving-team detection bias

The first serve in each rally is the chain seed. If first-serve
attribution is systematically wrong-team, the entire chain inherits the
wrong team. Asymmetry interpretation: pipeline systematically attributes
first serves to team-A players, even when GT says team B served.

**Test:** For each of 94 disagreement contacts, find the rally's first
serve (`actions` filter). Compare pipeline `playerTrackId` for the
first-serve to GT `resolved_track_id` for the GT-side first serve. Tally:

- `first_serve_wrong_count`: how many disagreement contacts come from
  rallies with wrong first-serve attribution?
- `first_serve_wrong_a_seeded`: how many of those got seeded as team A
  by the wrong serve?
- `first_serve_wrong_b_seeded`: how many got seeded as team B?

**Verdict criterion:** if ≥60% of disagreement contacts come from
wrong-first-serve rallies AND there's strong direction asymmetry in
the wrong seeds, H1 is supported.

### H2 — team_assignments labeling skew

`teamAssignments` is `{trackId: "A"/"B"}`. If the tracker over-assigns
PIDs to the "team A" range systematically, rallies with more team-A
PIDs would have more team-A wrong predictions.

**Test:** Per rally, count `teamAssignments` distribution. Stratify
across:
- **disagreement rallies** (those with ≥1 chain-disagreement contact)
- **agreeing rallies** (the 72 contacts with right chain are in
  rallies — count A vs B distribution there too)

Plus a fleet-wide baseline: across all trusted-32, what's the A/B
distribution?

**Verdict criterion:** if disagreement rallies have systematically
more A-PIDs than agreeing rallies AND fleet-baseline (e.g., 3+ A vs
1 B per rally), H2 is supported.

### H3 — Chain-walker init/transition bug

A code-path issue in `_compute_expected_teams` that produces
direction-asymmetric errors. Candidates:

- Synthetic-serve handling re-seeds incorrectly.
- UNKNOWN-action breaks happen disproportionately on team-B rallies.
- Net-crossing transition list misses an action type.

**Test:** For each disagreement contact, record:
- `first_serve_is_synthetic`: was the chain seed a synthetic serve?
- `unknown_actions_before`: how many UNKNOWN actions appear before this
  contact in the rally?
- `chain_integrity_at_contact`: True/False per `_chain_integrity`.
- `actions_in_chain_before`: position of this contact in the rally's
  action sequence (e.g., 1st, 2nd, ...).

**Verdict criterion:** if a specific precondition correlates with
disagreement direction (e.g., "synthetic-serve-seeded chains
disproportionately produce pipe=A errors"), H3 is supported.

### H4 (a priori rejected) — random distribution

Binomial test under H0 of 50/50 split: `P(63 of 94 or more extreme) ≈
0.001`. Rejected before probe runs.

## Probe structure

Single script: `analysis/scripts/probe_chain_asymmetry_2026_05_20.py`.

For each disagreement contact (the 94 cases from L6.json), compute:
- H1 evidence (first-serve attribution correctness + seeded team)
- H2 evidence (per-rally A/B distribution at this contact's rally)
- H3 evidence (synthetic-seed flag, UNKNOWN-count, chain-integrity)

Output per-contact attribution to **primary cause** when one hypothesis
has strong evidence and others don't. Track multi-cause cases honestly
in a separate column.

## Aggregation + decision

`analysis/reports/chain_asymmetry_2026_05_20/`:

- `per_disagreement.csv` — per-contact rows with all hypothesis evidence
  columns + primary_cause assignment.
- `per_hypothesis.json` — per-hypothesis aggregate counts + support
  fractions.
- `summary.md` — verdict + recommendation.

**Decision rules:**

| Outcome | Action |
|---|---|
| H1 dominates (≥60%) | Spec a fix for serving-team detection / first-serve attribution. Likely small. |
| H2 dominates (≥60%) | Investigate team_assignments labeling pipeline. Spec a fix. |
| H3 dominates (≥60%) | Spec a chain-walker code fix (likely small). |
| No single hypothesis ≥60% | Escalate to full chain-quality rewrite (probabilistic walker + re-anchoring + chain-quality classifier). Separate brainstorm cycle. |
| All hypotheses <30% | Inspect 10 disagreement contacts manually before re-scoping. May surface H5. |

## Out of scope

- Implementing any chain-walker fix. That's a separate plan once H1/H2/H3 is identified.
- Full chain-quality rewrite (probabilistic walker + classifier). Reserved for the case where no single hypothesis explains the asymmetry.
- L3 contact-frame regression workstream (separate, can run in parallel).
- Player-tracker improvements (L1 finding; deprioritized per ranking).
- VLM probe (out per user constraint).

## Risk register

- **Confounded hypotheses.** A disagreement contact may have evidence
  for multiple hypotheses (e.g., wrong first serve AND synthetic-seed).
  The per-contact CSV records ALL evidence; primary_cause assignment is
  a heuristic ("highest-magnitude single-axis evidence"). Multi-cause
  cases reported in summary.md.
- **Small wrong-first-serve subset.** If <30 of 94 have wrong first
  serves, H1 evidence is statistically weak even if 100% are A-seeded.
  Report with caveat.
- **Camera-side / formation bias as confound.** Memory notes
  `formation_semantic_flip` and side-switch logic (see
  [[side_switch_kuku_koko_diagnostic_2026_05_19]]). If team A always
  maps to the "near court" by camera convention and the asymmetry is
  driven by near-court detection bias, H2 will surface it but only via
  the team_assignments check. Note in summary if relevant.

## Artifact

- Spec: this file (commit TBD).
- Plan: written next via writing-plans skill, executed as a single-task
  spike (1 probe + 1 summary).
- Findings memo: `chain_asymmetry_findings_2026_05_20.md` (created when
  probe runs).

## Related

- [[upstream_bottleneck_findings_2026_05_20]] — found L6 as top
  bottleneck; this probe narrows the L6 finding to a specific cause.
- [[attribution_sub_lever_1_no_ship_2026_05_20]] — chain-context fallback
  NO-SHIP; documents the projection-trap that motivated diagnostic-first
  rigor.
- [[side_switch_kuku_koko_diagnostic_2026_05_19]] — relevant if H2's
  team_assignments labeling skew turns out to be camera-side-related.
- [[cross_team_prior_no_ship_2026_05_17]] — prior chain-related NO-SHIP
  ("chain-flipping 26% of seed serves wrong"). H1 here is essentially
  testing whether that 26% figure still holds on current pipeline.
