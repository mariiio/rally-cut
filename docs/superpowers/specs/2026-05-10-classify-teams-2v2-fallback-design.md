# classify_teams 2v2-Fallback Producer Fix

**Date:** 2026-05-10
**Status:** Design — pending implementation plan
**Workstream context:** Sub-1.2.A. Producer-side fix that prevents new I-8 violations on future tracking runs by extending the existing median-index fallback in `classify_teams` to fire on any non-2v2 partition (not just the 0v4 imbalance case it currently catches). Companion to Sub-1.1.E's `cleanup-team-labels-by-majority` (which fixes legacy data); together they provide layered defense — producer enforces 2v2 shape, cleanup enforces team identity.

## Goal

Modify `analysis/rallycut/tracking/player_filter.py:721` so that `classify_teams` always emits a valid 2v2 partition when 4 primary tracks are present, regardless of how Y-based classification scored them. Stops new I-8 violations from accumulating on future tracking runs.

## Motivation

I-8 (added 2026-05-10, commit `5a6968b`) surfaced 112 fleet-wide rallies where the team partition is not 2A+2B. Root cause: `classify_teams` classifies each track independently by median-Y vs. court_split_y. When tracks are Y-ambiguous (sit near the split line), late-arriving (insufficient early-frame Y data), or occluded at rally start, their classification can land on the wrong side of the split — producing 1A+3B or 3A+1B partitions. These are structurally impossible in beach volleyball (which is definitionally 2v2).

The existing fallback at line 721 already handles the extreme 0v4 case (all tracks classified to the same side) by sorting tracks by median-Y and forcing a top-2 / bottom-2 split. The fix extends this fallback to fire on ANY non-2v2 partition, not just the 0v4 case.

## Scope

**In scope:**
- One-line conditional change in `classify_teams` at `player_filter.py:721`.
- Unit tests for the new trigger condition: 1A+3B → 2v2, 3A+1B → 2v2, 0A+4B → 2v2 (existing case still works), 2A+2B → unchanged (no false fix).
- A "no regression" check: existing tests continue to pass.

**Out of scope:**
- Improving the Y-signal source (still uses each track's first `window_frames` positions). Late arrivers and Y-ambiguous tracks may still get rank-wise misclassified; identity errors are downstream Sub-1.1.E cleanup territory.
- Adding bbox-area or color-based tiebreaks for ambiguous Y.
- Changing `precomputed_assignments` semantics or sources.
- Re-tracking existing data (legacy violations are addressed by Sub-1.1.E cleanup).
- Sub-2 / Sub-3 work.

## Architecture

### The change

In `classify_teams` at `analysis/rallycut/tracking/player_filter.py:721`:

```python
# Current
if len(team_assignments) >= 4 and (near_count == 0 or far_count == 0):

# Proposed
if len(team_assignments) == 4 and not (near_count == 2 and far_count == 2):
```

The body of the conditional (the median-index split that follows) is unchanged.

### Behavior summary

For a rally with exactly 4 primary tracks:

| Input partition | Old behavior | New behavior |
|---|---|---|
| 2A+2B (clean) | unchanged | **unchanged** |
| 1A+3B | unchanged (kept invalid) | **median-index split → 2v2** |
| 3A+1B | unchanged (kept invalid) | **median-index split → 2v2** |
| 0A+4B | median-index split → 2v2 | **median-index split → 2v2** (same) |
| 4A+0B | median-index split → 2v2 | **median-index split → 2v2** (same) |

For a rally with != 4 primary tracks: behavior unchanged. The condition tightens from `>= 4` to `== 4`, but the existing logic for fewer-than-4 tracks still routes through the per-track Y classification (no fallback, since 2v2 doesn't apply).

### Why this is the right shape vs. alternatives

**Shape vs. identity separation.** The producer fix's responsibility is to enforce structural correctness (always emit 2v2 for 4-track rallies). Team-identity correctness (which player is on which team) is downstream — Sub-1.1.E handles it via cross-rally majority. This separation keeps each layer focused and small.

**Why not also improve the Y signal?** The current `track_pos[:window_frames]` may be unrepresentative for late arrivers / occluded-at-start / off-frame servers. We could use full-track-history median, or add bbox/color tiebreaks. These changes are larger, riskier, and the cleanup CLI already absorbs identity errors. Defer until we see whether Approach 1 + cleanup is insufficient in practice.

### Real-world edge cases

- **Player occluded at rally start**: median-Y over their first 60 detected frames may be biased. Y-rank position may be wrong → identity error. Mitigated by cross-rally cleanup.
- **Off-frame server**: same — first 60 detected frames may not reflect rally-start position. Same mitigation.
- **Late arriver**: `[:window_frames]` is "first 60 of the track" not "first 60 of the rally". Median-Y still computed correctly over their available data; rank may not match teammate's actual side. Same mitigation.
- **Y-ambiguous tracks (near court_split_y)**: median-index split deterministically picks top-2 vs. bottom-2. Edge cases where two players sit on the boundary may be assigned to wrong teams. Same mitigation.

In all four cases, the producer fix guarantees shape (2v2). Cross-rally cleanup guarantees identity downstream.

## Testing

### Unit tests (new)

Add to existing test file for `player_filter.py` (or create one if absent), with synthetic `PlayerPosition` fixtures:

1. **`test_2v2_input_unchanged`**: 4 tracks classified by Y to a clean 2v2 → output equals input.
2. **`test_1A3B_forced_to_2v2`**: 4 tracks where Y-classification produces 1 near + 3 far → output is 2v2 (which 2 are A and which are B is determined by Y-rank, asserted accordingly).
3. **`test_3A1B_forced_to_2v2`**: mirror of above.
4. **`test_0A4B_forced_to_2v2`**: 4 tracks all classified to far → existing behavior preserved (regression check).
5. **`test_4A0B_forced_to_2v2`**: 4 tracks all classified to near → existing behavior preserved.
6. **`test_three_tracks_not_modified`**: 3-track input doesn't trigger fallback (because `len == 4` is required) — output is whatever per-track Y classification produced.

### No regression

Existing `classify_teams` tests (if any) must continue to pass. Run `cd analysis && uv run pytest tests/ -k "classify_teams or player_filter"` to confirm.

### End-to-end

After the fix lands and a video is re-tracked, the new tracking output should pass I-8 by construction. We don't add an integration test for this — it's covered by `audit-pid-invariants` in CI.

## Rollout sequence

1. Land the player_filter fix + unit tests (single commit).
2. Run regression suite (`cd analysis && uv run pytest tests/`).
3. Document in memory.
4. **Future tracking runs** will produce 2v2-clean output by construction. Legacy data is handled by Sub-1.1.E.

This fix takes effect ONLY on new tracking runs. Existing data in DB is unchanged. The composition with Sub-1.1.E remains correct: any new tracking that somehow still emits identity errors (per-rally Y bias) will be caught by I-8 audit and cleaned by Sub-1.1.E majority vote.

## Done criteria

- [ ] `analysis/rallycut/tracking/player_filter.py:721` updated with the new conditional.
- [ ] All new unit tests pass.
- [ ] Existing `classify_teams` / `player_filter` tests continue to pass.
- [ ] Lint + mypy clean.
- [ ] Memory entry recorded.

## Risks

1. **Hidden test that relies on the old behavior.** Some test may assert that a non-2v2 partition is preserved. TDD will surface this. If found, the test was wrong — non-2v2 is structurally invalid for 4-track rallies. Update the test to assert 2v2.

2. **Downstream consumer that depended on wrong shape.** Some module elsewhere may have been silently working around the old non-2v2 output. After the fix, it'd see only 2v2 — should be a strict improvement, but worth a quick fleet audit to confirm no I-1..I-7 regressions.

3. **Identity errors more visible.** Before the fix, identity errors hid behind structural errors (a 1A+3B partition is structurally invalid AND semantically meaningless). After the fix, all rallies are 2v2, so any identity errors stand out as "this player is labeled wrong" instead of "this rally has an invalid shape." This is desirable — visibility of correct vs. incorrect identity is the prerequisite for the cleanup CLI to do its job.

4. **No effect on legacy data.** As noted in scope. Spec calls this out explicitly so it's not a surprise.

## File-change summary

- Modified (1): `analysis/rallycut/tracking/player_filter.py:721` (1-line conditional change)
- New / modified tests (1 file): `analysis/tests/unit/test_player_filter.py` (or extend existing)
- **Total:** ~1-2 files, <60 LOC including tests.
