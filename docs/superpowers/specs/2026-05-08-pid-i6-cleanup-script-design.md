# PID Invariant I-6: One-shot teamAssignments cleanup CLI

**Date:** 2026-05-08
**Status:** Design — pending implementation plan
**Workstream context:** Sub-1.1.B. Companion to Sub-1.1.A's player_tracker fix. Sub-1.1.A prevents new I-6 violations by re-deriving `team_assignments` post-`optimize_global_identity`; this Sub-1.1.B spec covers the cleanup of legacy data already persisted before the fix landed.

## Goal

Provide a CLI command — `rallycut cleanup-team-assignments <video-id>` — that walks every rally of a given video and rebuilds the persisted `teamAssignments` dict using `classify_teams` with the existing dict as `precomputed_assignments`. Closes I-6 violations on legacy data without requiring expensive re-tracking.

## Motivation

Sub-1.1.A's fix (commit `3f0200b`) added a `classify_teams` call after `optimize_global_identity` returns, restoring the invariant that `team_assignments` keys match the post-global track set. The fix only takes effect for **new** tracking runs.

Existing panel data was tracked weeks ago, before the fix. Re-running `eval_cross_fixture.sh` (which runs `match-players` + `remap-track-ids`, but not player tracking itself) confirmed the I-6 violations persist:

- 5c756c41: 5 I-6 violations
- b5fb0594: 7 I-6 violations
- 854bb250: 2 I-6 violations
- 7d77980f: 13 I-6 violations
- **Total: 27 I-6 violations across the 4 panel videos.**

Re-tracking the panel via Modal/local CPU would clear these but is expensive (~minutes per video, ML compute). The cheaper alternative: re-run `classify_teams` on the already-persisted positions, with the existing `teamAssignments` as precomputed input. This is the same logic Sub-1.1.A's player_tracker fix uses; we apply it directly to persisted data.

## Scope

**In scope:**
- New CLI command `rallycut cleanup-team-assignments <video-id>` (Typer-based, follows existing pattern).
- Per-rally rebuild of `actions_json["teamAssignments"]` using `classify_teams` with precomputed-passthrough.
- Format conversion (`{str: "A"|"B"}` ↔ `{int: 0|1}`) on the way in and out.
- Skip rallies missing `court_split_y` or with empty positions.
- One DB transaction per rally for atomicity.
- Registration in `cli/main.py`.

**Out of scope:**
- I-3/I-7 phantom-action cleanup (different root cause; out per user scope decision).
- Any change to `player_tracker.py` (already fixed in Sub-1.1.A).
- Re-tracking of any video.
- Unit tests for the CLI itself — logic is already covered by Sub-1.1.A's `test_classify_teams_post_global.py` unit tests; integration verification is via `audit-pid-invariants`.
- Sub-2 / Sub-3 work.

## Architecture

### CLI command

`analysis/rallycut/cli/commands/cleanup_team_assignments.py` — Typer command with this signature:

```python
def cleanup_team_assignments_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-rally info"),
) -> None: ...
```

### Per-rally data flow

1. Load `PlayerTrack` row for the rally: `positions_json` (list, post-remap with canonical track_ids 1-4), `actions_json` (dict, contains `teamAssignments`), `court_split_y` (float).
2. If `court_split_y is None` or `len(positions_json) == 0`: skip with `[skip] rally <id>: missing court_split_y or empty positions`.
3. Reconstruct `PlayerPosition` instances from `positions_json` entries (only need `frame_number`, `track_id`, `y` for `classify_teams`; other fields can use safe defaults).
4. Convert legacy `teamAssignments` `{str: "A"|"B"}` → `{int: 0|1}`:
   - `"A"` → `0` (near), `"B"` → `1` (far). Matches `action_classifier.py:138` convention.
   - Skip entries with non-A/B values (corrupt).
5. Call `classify_teams(positions, court_split_y, precomputed_assignments=int_team_assignments)`.
6. Convert result `{int: 0|1}` → `{str: "A"|"B"}`:
   - `0` → `"A"`, `1` → `"B"`.
7. Mutate `actions_json["teamAssignments"]` to the new dict.
8. Write back to DB in a single transaction:
   ```sql
   UPDATE player_tracks SET actions_json = %s WHERE rally_id = %s
   ```

### Idempotence

Re-running the cleanup on already-clean data is a no-op:
- Every track_id in post-remap positions is in the existing teamAssignments
- `classify_teams` covers all via the precomputed branch
- The output dict is byte-identical to the input
- The DB write is the same value already there

### Format conversion convention

The "A"/"B" ↔ 0/1 mapping is established in `action_classifier.py:138`:
> Convention: team 0 (near court) = "A", team 1 (far court) = "B".

This spec follows that convention. If a future workstream changes the convention, both this CLI and `action_classifier.py` need updating together.

### Skip semantics

A rally is skipped (left untouched) when:
- `positions_json` is None or empty (rally never had successful tracking).
- `court_split_y` is None (court split couldn't be computed).
- `actions_json` is None or doesn't contain a `teamAssignments` key.

The command logs each skip with the reason. Skipped rallies do not count toward the cleanup count.

## Rollout sequence

1. Land the CLI command + `main.py` registration (single commit).
2. Run cleanup on all 4 panel videos:
   ```bash
   for v in 5c756c41-1cc1-4486-a95c-97398912cfbe \
            b5fb0594-d64f-4a0d-bad9-de8fc36414d0 \
            854bb250-3e91-47d2-944d-f62413e3cf45 \
            7d77980f-3006-40e0-adc0-db491a5bb659; do
     uv run rallycut cleanup-team-assignments "$v"
   done
   ```
3. Re-run audit per video; confirm I-6 → 0:
   ```bash
   uv run rallycut audit-pid-invariants <video-id>
   ```
4. Update memory entry to reflect the post-cleanup panel state (I-6 zero; I-3/I-7 residual on b5fb0594 + 7d77980f).
5. (Optional, future) Decide whether the residual I-3/I-7 violations warrant a separate cleanup workstream or are acceptable as Sub-2 input noise.

## Done criteria

- `uv run rallycut cleanup-team-assignments --help` shows the command.
- After running on all 4 panel videos, `uv run rallycut audit-pid-invariants <each-video>` reports zero I-6 violations.
- I-1, I-2, I-4, I-5 remain at their pre-cleanup values (zero on the panel).
- I-3 and I-7 remain at their pre-cleanup values (legacy phantom-action noise on b5fb0594 + 7d77980f, deliberately not addressed).

## Risks

1. **Wrong label for newly-classified tracks.** When `classify_teams` falls back to median-Y for a track not in the precomputed dict, the label is determined by the persisted `court_split_y` and the post-remap positions' Y values. If `court_split_y` was incorrect at tracking time (or has drifted), the median-Y classification may be wrong. **Mitigation:** this is the same input `optimize_global_identity → classify_teams` would have seen if the fix had been in place from the start, so the output matches what the fixed pipeline would produce. No new risk vs. re-tracking.
2. **DB write failure mid-rally.** If the transaction fails, the rally is left in its current state — no partial writes. The CLI should log the failure and continue with the next rally rather than abort the whole video.
3. **Convention drift.** The "A"/"B" ↔ 0/1 mapping is hardcoded; if `action_classifier.py:138` changes the convention, both modules need updating together. **Mitigation:** documented in the spec; small surface area.

## File-change summary

- New (1): `analysis/rallycut/cli/commands/cleanup_team_assignments.py` (~80 LOC)
- Modified (1): `analysis/rallycut/cli/main.py` (~2 lines: import + registration)
- **Total:** 2 files, ~82 LOC.
