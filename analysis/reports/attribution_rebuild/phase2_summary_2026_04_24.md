# Phase 2 — Offline Chooser + match_tracker Team-Pair Fix

**Date:** 2026-04-24
**Status:** Stage-2 primitive fix SHIPPED. Offline chooser NO-GO (can't beat baseline).
**Scope:** 9 fixtures / 69 rallies / 468 GT actions.

## Outcomes

### ✅ Stage-2 team-pair bug FIXED

**Before:** `match_tracker.build_match_team_assignments` hardcoded `base_team = 0 if pid <= 2 else 1` — pairing was `{1,2}` vs `{3,4}` regardless of physical teams. Broke on 74% of rallies (Phase 1.3 measured 26% agreement with positional reality).

**After:** `_teams_from_positions` infers pairing from rally-start foot-y positions. Team 0 = 2 tids with highest rally-start median y (near), team 1 = other 2 (far). Legacy pid-ordering kept as fallback when positions unavailable.

**Result:** Phase 1.3 re-measured at **100% agreement** on all audit-eligible rallies. Primitive foundation correct.

### ⚠️ Offline chooser NO-GO

Built `scripts/phase2_run_chooser.py`: corrective overlay that keeps pipeline's current pick unless `(rank-1 candidate's rally-start team disagrees with ball_side) AND (rank-2 agrees)`.

Swept margin thresholds `[0, 0.05, 0.10, ..., 0.40]`:

| margin | correct | wrong | missing | abstain |
|---|---|---|---|---|
| baseline | 205 (43.8%) | 191 (40.8%) | 72 (15.4%) | 0 |
| 0.00 | 190 (40.6%) | 205 (43.8%) | 72 | 1 |
| 0.15 | 181 (38.7%) | 188 (40.2%) | 72 | 27 |
| 0.40 | 171 (36.5%) | 170 (36.3%) | 72 | 55 |

**No threshold beats baseline.** Correct drops more than wrong drops across the sweep.

## Why the chooser fails

### Of the 117 chooser_fixable actions from Phase 1.5:

- **34 have GT pid NOT in the candidate list** — unfixable by any chooser.
- **8 have GT at rank-1** — pipeline's own re-pick logic already broke these.
- **31 have GT at rank-2** — the potential override target.
- 44 have GT at rank 3-4 — too far for distance-based override.

### Primitive signal quality ceiling

- **Ball-side reliability**: 71% (Phase 1.4) — using ball_side as an override trigger produces ~29% false-overrides, flipping correct picks to wrong.
- **Positional team at contact frame**: noisy due to setters/defenders crossing the midline during play.
- **Rally-start team**: cleaner but doesn't capture mid-rally position states.

Net: override signal false-positive rate > true-positive rate on the cross_team_wrong subset. Any margin that gives enough precision costs too many correct picks from abstention.

## Why match_tracker fix didn't lift baseline

The team-pair fix corrects primitive labels but doesn't flow through to chooser picks because `reattribute_players` (analysis/rallycut/tracking/action_classifier.py:2825) uses **serve-chain anchoring** to derive expected teams:

```
serve's team → receive's team (opposite) → set's team (same) → attack's team (same) → ...
```

If the serve itself is misattributed (common on low-confidence first contacts), the whole chain's expected-team is wrong regardless of how correctly `teamAssignments` labels the players. The fix IS visible to downstream consumers (stats, analytics, UI coloring) but not to attribution accuracy on this corpus.

## Decision

1. **Ship match_tracker fix** — real primitive correction, no downstream regression.
2. **Close offline-chooser workstream** — baseline is the ceiling under current chooser architecture.
3. **Phase 3 complement rescue** is the next lever — operates at rally-level semantics (receive→set→attack chains), may capture detection-limit (71 missing) + primitive-fixable (43) cases that are structurally unreachable from Phase 2.
4. **Separate workstream for serve-contact recall** — if serve attribution were more robust, `reattribute_players`' serve-anchor chain would work on more rallies and amplify the team-pair fix.

## Artifacts

- `analysis/rallycut/tracking/match_tracker.py` — `_teams_from_positions` + `build_match_team_assignments` fix.
- `analysis/rallycut/cli/commands/reattribute_actions.py` — passes `rally_positions` to the builder.
- `analysis/scripts/phase2_run_chooser.py` — offline chooser harness (kept for future experiments; NO-GO record).
- `analysis/reports/phase1_3_team_side_2026_04_24.md` — regenerated, shows 100% post-fix.
- `analysis/reports/attribution_rebuild/baseline_2026_04_24.json` — re-locked with fixed team labels.
