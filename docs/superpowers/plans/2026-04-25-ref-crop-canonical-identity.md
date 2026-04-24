# Ref-crop canonical identity — one source of truth for player pid

## Context

Today, "who is P1/P2/P3/P4" in RallyCut is decided by `match-players`' Hungarian assignment. Hungarian is non-deterministic at the margins: same-video re-runs can flip pid pair assignments (2↔4 is the observed common case). Every downstream consumer — editor display, action-GT labels, `actions_json.playerTrackId`, `match_stats`, team attribution, score-GT matching — treats this shifting value as a stable identity. Labelers press "2" trusting it means the same physical body forever; after a pipeline re-run the badge on the same body says "4".

The commit `3cf67c1` + `fa05965` landed a raw-`trackId` anchor on action-GT, which solves **data integrity** (labels still point at the correct physical body post-re-run). It did not solve **display consistency**: the badge number still floats because display resolves `trackId` through `appliedFullMapping`, which is recomputed every run.

Data anchoring is a necessary layer; display consistency is what lets a human trust what they label.

## Design principle

**An identifier is either human-anchored or it's not an identifier — it's a derived view that must carry a staleness flag and a single canonical re-derivation path, not be treated as truth by downstream consumers.**

Canonical player identity (pid 1-4) must be sourced from a human decision — the ref-crop uploads the user already does — not from an ML optimization. ML's role narrows to: given raw BoT-SORT tracks, score each against each ref crop and produce a deterministic `{raw_trackId → pid}` assignment. That assignment is the single source of truth; all consumers read from it.

Non-goals for this plan: changing the ML signals (HSV, DINOv2, position continuity stay). Changing the tracker. Rewriting the editor UX. Re-labeling existing GT (the raw `trackId` anchor protects it).

## Proposed direction

1. **Ref crops are the identity source.** Require exactly 4 ref crops per video (one per pid, user-labeled) before GT labeling is permitted. Soft-enforce on legacy videos that already have partial sets — flag in UI as "identity unstable, upload crops to stabilize" — but the new code path only activates with a full set.

2. **`Video.canonicalPidMapJson`** stores per-rally `{raw_trackId → pid}`. Written when the full crop set exists. Once written, it's the only read source for canonical pid — by editor, by GT rendering, by actions_json interpretation, by stats.

3. **`match-players` narrows its output** when crops are complete: per-track scoring against the crops replaces the cross-player Hungarian optimization for pid assignment. Same features feed the similarity; only the optimization target changes. Other outputs (`trackToPlayer` for legacy, `sideSwitchDetected`, `serverPlayerId`) keep working.

4. **Invalidation is explicit and minimal.** Ref-crop changes (upload, delete, re-label a crop's pid) invalidate `canonicalPidMapJson`. Nothing else invalidates it. Pipeline re-runs without crop changes are no-ops for identity.

5. **Rollout is gated by a pre-registered accuracy A/B.** Baseline `player_attr 57.66 / serve_attr 53.96 / court_side 66.56 / score 82.85` (per memory, post-cleanup LOO). New path must be within -1pp on all four, and within -2pp on the two primary metrics (`player_attr`, `serve_attr`). Ship only if green. Fresh LOO run required, not memory numbers.

6. **Fall-through for legacy videos** that don't have a full 4-crop set yet: current behavior stays, explicitly flagged "identity unstable" in the UI so the user knows GT labeling there is provisional.

## What this replaces / deprecates

- Dual-source pid lookups (`trackToPlayer` + `appliedFullMapping` + ad-hoc sort-order fallback). Collapse to one read helper reading `canonicalPidMapJson` with the two legacy fields as strictly legacy fallbacks.
- My `analysis/scripts/backfill_action_gt_trackid.py` usefulness post-ship: its trackId outputs are still valid, but running it on new videos should be unnecessary because the editor writes `trackId` directly and the canonical map renders it correctly from day one.
- The `canonical-pid-map-freeze` band-aid proposal from this session. Don't implement.

## Success criteria

Before ship:
- **Accuracy A/B passes** on the 9-fixture LOO (pre-registered above).
- **Determinism regression test** in CI: run `match-players` twice on a fixture with full crops, assert `canonicalPidMapJson` is bit-identical.
- **Invariant test**: given a video with full crops, for every raw trackId appearing in positions, `canonicalPidMapJson` has exactly one pid assignment, and that pid corresponds to the ref crop with highest similarity.
- **Editor behavior**: manual smoke — label 3 actions, run `match-players` twice, confirm P-badge number AND physical body both stay stable across renders.
- **Backward compat**: fixtures without full crops render with current behavior; no crash, no regression.

After ship:
- No new data-invalidation bugs reported in the two weeks following rollout.
- Subsequent GT labeling sessions don't produce "pid drifted" complaints.

## Risks and mitigations

- **Accuracy regression on match-tracker.** The A/B gate is the safety. If it fails, don't ship; investigate. Most likely root cause would be that Hungarian's cross-player optimization was compensating for crop-similarity noise on some borderline pair; per-track scoring would miss that compensation. Possible fix: keep Hungarian as a post-score refinement *only when all four crop-scores are near-tied*, preserving stability everywhere else.
- **Ref-crop quality**: if a user uploads a bad crop (wrong player, partially occluded, off-frame), per-track matching degrades. Mitigate with the existing crop-anomaly detector (already shipped, Phase 1.2 identity-layer per recent commits) — surface bad crops in the UI before permitting identity enforcement.
- **Videos mid-production** without full crops: current labeling flow doesn't require them. Enforcement is a new UX gate. Mitigate with the "identity unstable" flag + optional-but-nagging crop-upload prompt, not a hard block that breaks existing sessions.
- **Downstream consumers reading from old fields**: `actions_json.playerTrackId`, `match_stats.playerStats[i].playerId`, etc. still treat the old canonical pid as truth. In-scope for Tier 1 (separate follow-up). This plan does not fix those; it establishes the foundation.
- **Session fatigue / scope creep.** Execute in a fresh session, not during active debugging.

## Out of scope (Tier 1+ follow-ups, not now)

- Extending the raw-`trackId` anchor pattern to `actions_json` pipeline predictions.
- Team identity derivation from crop team-labels.
- JSON-schema validation on `matchAnalysisJson` / `actionsJson` / etc.
- Event-sourced change log.
- Editor UX redesign (thumbnail identity instead of numbers).
- Monitoring / alerting dashboards.

These are the right direction per the architecture audit, but each wants its own plan and its own A/B. Don't bundle.

## Starting point for the next session

- Read this plan and the commit trio on `main`: `3cf67c1`, `fa05965`, `737bcf0`.
- Capture the current LOO baseline fresh (don't trust memory numbers). Use `analysis/scripts/eval_loo_video.py` or equivalent.
- Decide between two implementation shapes before writing code:
  - **(a) In-place in `match_tracker.py`**: gate the Hungarian path behind "4 crops present?"; in the new branch, per-track independent crop scoring produces the assignment.
  - **(b) New dedicated module**: new `analysis/rallycut/tracking/canonical_identity.py` that owns `canonicalPidMap` derivation end-to-end; `match_tracker.py` just calls into it when crops are full.
  - (b) is cleaner architecturally; (a) is faster to land. Both are defensible.
- A/B-gate before shipping either. No ship without numbers.

## One-line summary

Source canonical pid from user-labeled ref crops; cache the derivation; read everywhere from that cache; don't recompute unless the user changes a crop.
