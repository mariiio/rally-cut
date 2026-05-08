# PID Invariant I-6: Re-align team_assignments after global_identity

**Date:** 2026-05-08
**Status:** Design — pending implementation plan
**Workstream context:** Sub-1.1.A. Closes the systematic I-6 violation surfaced by the audit at end of Sub-1 (`docs/superpowers/specs/2026-05-08-pid-leverage-audit-sub1-design.md`). Cleans the panel state so Sub-2 (team-coherence validator) builds on clean PIDs.

## Goal

Restore the invariant that `set(team_assignments.keys()) == {p.track_id for p in positions if p.track_id >= 0}` at the point where `PlayerTrackingResult` is persisted. After this fix, every primary track has a team label; no phantom (stale) keys remain.

## Motivation

The Sub-1 audit on the panel found 27 rallies where a primary track is in `primary_track_ids` but absent from `teamAssignments`, producing systematic I-6 violations across 4 distinct invariants (I-2/I-3/I-6/I-7 — all correlated through this gap).

Empirical root cause (verified on rally `8c49e480-407e-4118-8a9e-c4ed5172a7ce`):

- Pre-remap state: `positions` track IDs `[1, 2, 3, 6]`, `primaryTrackIds` `[1, 2, 3, 6]`, `teamAssignments` keys `['1', '2', '4']`.
- The `teamAssignments` dict already contains a phantom key (`4`, not in positions) and is missing two live tracks (`3`, `6`).

Looking at `analysis/rallycut/tracking/player_tracker.py:1733-1768`:

```python
team_assignments = classify_teams(positions, split_y, ...)         # line 1733
...
positions, global_result = optimize_global_identity(               # line 1761
    positions, team_assignments, color_store, ...
)
```

`classify_teams` builds a dict keyed on the pre-global track IDs. `optimize_global_identity` rewrites `track_id` on positions (segment reassignment to canonical anchors) but does **not** return updated `team_assignments`. The dict is then persisted with stale keys.

## Scope

**In scope:**
- One call-site change in `analysis/rallycut/tracking/player_tracker.py` immediately after `optimize_global_identity` returns: re-derive `team_assignments` from the post-global positions.
- One new unit test exercising the precomputed-passthrough behavior under simulated global-identity rewrite.
- Re-matching the 4 panel videos via `scripts/eval_cross_fixture.sh` to confirm the audit drops to zero.

**Out of scope:**
- Refactoring `optimize_global_identity` to own team_assignments updates (a larger architectural change with the same outcome).
- Any change to `remap-track-ids`, `match-players`, or the audit module.
- Sub-2 / Sub-3 work.

## The fix

In `player_tracker.py`, immediately after the existing `optimize_global_identity` block (line ~1768), add:

```python
positions, global_result = optimize_global_identity(
    positions, team_assignments, color_store, ...
)
# After global identity may rewrite track_ids on positions, re-derive
# team_assignments so its keys match the post-global track set.
# Restores PID-invariant I-6 (every primary track has a team label).
team_assignments = classify_teams(
    positions, split_y, precomputed_assignments=team_assignments,
)
```

### Why this works

`classify_teams` (defined in `analysis/rallycut/tracking/player_filter.py:660`) iterates over `track_positions` (built from the input positions list) and for each `track_id`:
- If the track_id is in `precomputed_assignments`, copy the label across.
- Otherwise, classify by median-Y on the early frames.

Passing the **old** `team_assignments` as `precomputed_assignments` produces:

- **Live, unchanged tracks**: present in both old dict and post-global positions → label preserved verbatim.
- **Stale phantom keys** (in old dict but no longer in positions): silently dropped — iteration is over post-global track_positions only.
- **Newly introduced track_ids** (from segment reassignment): not in old dict → classified by median-Y. This is the same fallback `classify_teams` uses for any novel track.

### Idempotence

If `optimize_global_identity` rewrote nothing (skip-reason path or zero-segment-reassignment), every post-global track_id is in the old dict, every label carries over via the precomputed branch, and the function returns a dict equal to its input. No behavior change in the no-rewrite case.

### Where the fix does NOT execute

The re-derive is gated by the same `if` that gates `optimize_global_identity` itself (`player_tracker.py:1746-1751`):

```python
if (
    not skip_global_identity
    and color_store is not None
    and color_store.has_data()
    and team_assignments
):
```

Rallies that skip global_identity entirely don't run the re-derive — preserves their behavior unchanged.

## Caveats (acknowledged, not blocking)

- **Median-Y fallback for late-arriving new tracks**: if `optimize_global_identity` introduces a track_id that lives entirely in the late portion of the rally, the median-Y classifier may misclassify (early frames don't exist for that track). This is the same risk `classify_teams` already carries for any novel track; the fix does not introduce new risk.
- **Architectural cleanliness**: the "purer" fix is to have `optimize_global_identity` own and return updated `team_assignments`. Rejected for this workstream because it requires changing the function's signature, propagating segment-to-team tracking through the assignment internals, and updating call sites. Out of scope for an invariant-restoration patch; can be revisited if subsequent work in this area pulls it in naturally.

## Architecture

### Files touched

- **Modified (1):** `analysis/rallycut/tracking/player_tracker.py` — single ~5-line addition at line ~1768.
- **New (1):** `analysis/tests/unit/test_classify_teams_post_global.py` — focused unit test covering the precomputed-passthrough behavior under simulated rewrite.

### Function contract (unchanged)

`classify_teams(positions, court_split_y, window_frames=60, precomputed_assignments=None) -> dict[int, int]`

The signature is untouched; this fix simply uses an existing parameter (`precomputed_assignments`) at a new call site.

## Testing

### Unit test (`analysis/tests/unit/test_classify_teams_post_global.py`)

Single test case:

1. Build a list of `PlayerPosition` entries spanning ~60 frames, with track_ids `{1, 2, 3, 6}` initially.
2. Define a pre-global team_assignments dict `{1: 0, 2: 1, 3: 0, 6: 1}`.
3. Simulate global_identity rewrite by relabeling all track_id=6 entries to track_id=4 in the position list (in-place mutation).
4. Call `classify_teams(post_global_positions, split_y=0.5, precomputed_assignments=pre_global_team_assignments)`.
5. Assert: returned dict has keys exactly `{1, 2, 3, 4}` (no key `6`), and tracks `1`, `2`, `3` retain their pre-global labels.

Track `4`'s label is determined by median-Y on its frames; the test should not over-assert on its value (the test exercises the *passthrough+fallback combination*, not median-Y semantics).

### Existing test suite

After the change, run `cd analysis && uv run pytest tests/ -k "team or filter or tracker"`. The precomputed branch is the existing path, so no regressions are expected.

### Integration verification

After landing the unit test and the player_tracker fix:

1. Run `bash analysis/scripts/eval_cross_fixture.sh`.
2. Expected: zero I-6 violations on all 4 panel videos. Other invariants (I-1, I-4 untouched; I-2/I-3/I-7 should also clear because re-matching regenerates `actions` consistent with the post-Task-12 silent-skip logic).
3. The script exits 0.

## Rollout sequence

1. Land player_tracker fix + unit test (single commit).
2. Run the eval script. Audit-clean confirmation comes from the script's `AUDIT_FAILED` exit gate.
3. Update `memory/MEMORY.md` (one-line entry under Player Tracking) noting the I-6 fix and the post-fix panel-clean baseline.

## Done criteria

- `cd analysis && uv run pytest tests/unit/test_classify_teams_post_global.py -v` passes.
- `cd analysis && uv run pytest tests/ -k "team or filter or tracker"` passes (no regressions).
- `bash analysis/scripts/eval_cross_fixture.sh` exits 0 on the 4 panel videos.
- `uv run rallycut audit-pid-invariants <each-panel-video>` exits 0.

## Risks

1. **Hidden coupling between `team_assignments` and intermediate stages between line 1768 and persistence.** Verified: the only intermediate consumers (downstream of global_identity, upstream of persist) read `team_assignments` for filter validation; correctness improves with a clean dict, never relies on phantom keys.
2. **Performance.** `classify_teams` is O(positions). For a 30s rally at 30fps with 4 tracks, ~3.6k entries — trivial.
3. **Determinism.** `classify_teams` has no randomness. Re-running on identical inputs yields identical outputs.

## File-change summary

- Modified (1): `analysis/rallycut/tracking/player_tracker.py` (~5 lines added)
- New (1): `analysis/tests/unit/test_classify_teams_post_global.py` (~50 lines)
- **Total:** 2 files, ~55 LOC.
