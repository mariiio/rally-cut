# PID Stale Attribution Cleanup CLI

**Date:** 2026-05-09
**Status:** Design — pending implementation plan
**Workstream context:** Sub-1.1.C. Companion to Sub-1.1.A's player_tracker fix and Sub-1.1.B's `cleanup-team-assignments` CLI. After Sub-1.1.B cleared the fleet I-6 invariant, the residual 450 violations are dominated by I-2/I-3/I-7 — positions and actions referencing track IDs not in `primary_track_ids`. This spec covers the in-place cleanup of that legacy data.

## Goal

Provide a CLI command — `rallycut cleanup-stale-attribution <video-id>` — that walks every rally of a given video, filters `positions_json` to keep only entries whose `trackId` is in `primary_track_ids`, and filters `actions_json["actions"]` to keep only entries whose `playerTrackId` is in `primary_track_ids ∪ {-1}`. Closes I-2 (49 videos affected, 206 violations), I-3 (28 videos, 141), and I-7 (26 videos, 53) on legacy data without re-tracking. Includes a `--dry-run` flag for previewing the diff before mutating.

## Motivation

Fleet audit (2026-05-09) showed:

- I-2: 206 violations across 49 videos (positions_json contains non-primary tracks).
- I-3: 141 violations across 28 videos (actions attributed to non-primary tracks).
- I-7: 53 violations across 26 videos (post-mapping mapped_track_id outside {1..4} ∪ {-1}).

Total: 400 of 450 residual violations (the rest being I-1 and I-5 outliers requiring different treatment).

All three are correlated symptoms of the same legacy-data root cause: tracking pipelines that ran before Task 12's silent-skip fix in `compute_match_stats`, where the filter to primary tracks was incomplete or absent at storage time. Future tracking is clean; legacy data is dirty until cleaned. Re-tracking is the heavy alternative; this CLI is the lightweight one.

## Scope

**In scope:**
- New CLI command `rallycut cleanup-stale-attribution <video-id>` (Typer + Rich).
- Per-rally filtering of `positions_json` (keep only primary tracks) and `actions_json["actions"]` (keep only primary or `-1`).
- `--dry-run` flag that runs all filtering logic and prints the per-rally diff WITHOUT executing UPDATE statements or committing.
- Skip rallies missing `primary_track_ids` or with empty positions/actions.
- One DB transaction per rally.
- Registration in `cli/main.py`.

**Out of scope:**
- Modifying `contacts_json` (I-4 was clean across the fleet — no contact attribution leakage observed).
- Modifying `actions_json["teamAssignments"]` (already cleaned by Sub-1.1.B's `cleanup-team-assignments`).
- Modifying `match_analysis_json` on the videos table.
- Repairing `primary_track_ids` itself (I-1 case — primary set size ≠ 4; needs re-tracking, not cleanup).
- Investigating the 32 I-5 violations on `073cb11b` (separate workstream — likely match-players coverage gap).
- Unit tests for the CLI itself (logic is "load → filter → write back" with trivial predicates; `--dry-run` is the verification tool).
- Sub-2 / Sub-3 work.

## Architecture

### CLI command

`analysis/rallycut/cli/commands/cleanup_stale_attribution.py` — Typer command:

```python
def cleanup_stale_attribution_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without writing"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-rally info"),
) -> None: ...
```

### Per-rally data flow

1. Load `PlayerTrack` row for the rally: `positions_json` (list, post-remap with canonical track_ids), `actions_json` (dict containing `actions` array), `primary_track_ids` (list[int]).
2. Skip if `primary_track_ids` is empty or None: `[skip] rally <id>: no primary_track_ids` and continue.
3. **I-2 filter**: keep only `p` in `positions_json` where `int(p["trackId"]) in primary_track_ids`. Drop the rest.
4. **I-3/I-7 filter**: keep only `a` in `actions_json["actions"]` where `int(a["playerTrackId"]) in primary_track_ids ∪ {-1}`. Drop the rest.
5. If neither filter changed anything: `[noop] rally <id>: already clean` and continue.
6. If `--dry-run`: print `[DRY] rally <id>: would drop N positions, M actions` and continue without writing.
7. Otherwise (execute mode): write back via single transaction:
   ```sql
   UPDATE player_tracks SET positions_json = %s, actions_json = %s WHERE rally_id = %s
   ```
   Print `[fix] rally <id>: dropped N positions, M actions`.
   On failure: rollback, print `[err]`, continue.
8. End of run: print summary with totals (`updated`, `no-change`, `skipped`, optionally `dry-run` flag).

### Idempotence

Running on already-clean data is a no-op:
- All positions pass the primary filter.
- All actions pass the primary ∪ {-1} filter.
- Neither changed → no UPDATE.

### `--dry-run` semantics

- Runs the same loop, computes the same filtered output, prints the same per-rally line (with `[DRY]` prefix instead of `[fix]`).
- Does NOT call `cur.execute("UPDATE ...")`.
- Does NOT call `conn.commit()`.
- Final summary line includes a clear `[DRY RUN]` prefix.

This enables the recommended workflow: dry-run first, sanity-check totals against the audit's expected counts, then execute.

## Rollout sequence

1. Land the CLI command + `main.py` registration (single commit).
2. **Dry-run pass on the fleet** to preview totals:
   ```bash
   while read vid; do
     uv run rallycut cleanup-stale-attribution "$vid" --dry-run --quiet
   done < /tmp/audit_videos.txt
   ```
3. Sanity-check aggregate dry-run totals:
   - Total positions dropped should approximately equal I-2 fleet count (≈ 206 entries; could be more if multiple positions per offending track_id).
   - Total actions dropped should approximately equal I-3 fleet count (≈ 141 entries — one violation per action so close match).
   - If totals diverge significantly, stop and investigate before mutating.
4. **Real run on the fleet**:
   ```bash
   while read vid; do
     uv run rallycut cleanup-stale-attribution "$vid" --quiet
   done < /tmp/audit_videos.txt
   ```
5. **Re-audit fleet** (`scripts/fleet_pid_audit.py`). Expected residual: ~49 violations (16 I-1 + 33 I-5) on outlier videos. I-2/I-3/I-7 should be zero.
6. Update memory with post-cleanup fleet baseline (residual counts, outlier video list).

## Done criteria

- `uv run rallycut cleanup-stale-attribution --help` shows the command with `--dry-run` flag.
- Dry-run mode produces non-empty diff output and clearly DOES NOT mutate the DB (verifiable by re-running the audit before/after dry-run — should be byte-identical).
- After execute pass on all 70 audit-able videos, fleet audit shows:
  - I-2: 0
  - I-3: 0
  - I-7: 0
  - I-6: ≤ 1 (unchanged from pre-cleanup; one stubborn case on 635dcba2)
  - I-1: ~16 (unchanged)
  - I-5: ~33 (unchanged)
- Residual is exclusively the I-1 / I-5 outlier set, ~49 violations total.

## Risks

1. **User-visible stat shift.** Dropping phantom actions reduces total kill/attack/dig counts in stats. The dropped actions were attributed to invalid track IDs (e.g., refs, crowd, fragments) and produced wrong stats. Direction of correction is right; magnitude is bounded (only the actions that fail the filter). User-visible impact is small per video (~2-5 actions per affected rally) but real. The `--dry-run` step gives the user a chance to gauge impact before committing.

2. **Sequencing with future re-tracking.** If a video is later re-tracked (e.g., via Modal), the new tracking run will overwrite `positions_json` and `actions_json` with fresh, post-fix data. The cleanup's effect is wiped, which is correct — re-tracked data is the source of truth. No coupling concern.

3. **`-1` semantics for synthetic actions.** The filter keeps actions with `playerTrackId == -1` because synthetic serves use this sentinel. If the underlying convention ever changes (e.g., -1 → -2 for unattributed), this filter needs updating. Convention is documented in `action_classifier.py`.

4. **DB write failure mid-rally.** Per-rally transactions mean a failure leaves that rally untouched but doesn't abort the video. Standard pattern from `cleanup-team-assignments`.

## File-change summary

- New (1): `analysis/rallycut/cli/commands/cleanup_stale_attribution.py` (~100 LOC)
- Modified (1): `analysis/rallycut/cli/main.py` (~2 lines: import + `app.command`)
- **Total:** 2 files, ~102 LOC.
