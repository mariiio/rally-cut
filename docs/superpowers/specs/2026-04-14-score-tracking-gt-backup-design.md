# Score Tracking GT Backup to S3

**Date:** 2026-04-14
**Status:** Design approved, pending implementation

## Problem

Score tracking ground truth (`rallies.gt_serving_team` in PostgreSQL) is the
label set behind the `score_accuracy` metric (currently 88.6% end-to-end).
It exists only in the local PostgreSQL database. A DB reset, MinIO clear,
or laptop loss would destroy it. Other GT types (rally bounds, tracking,
actions, player-matching) already round-trip through S3 via
`rallycut train export-dataset` → `push` → `pull` → `restore`. Score GT
should ride the same rails.

## Goals

- Score GT survives DB resets and laptop loss.
- Future `train push --name <name>` snapshots include score GT automatically.
- `train restore` re-imports score GT without clobbering newer DB labels.
- Zero changes required for callers of `score_ground_truth.compute_score_metrics`.

## Non-Goals

- Backing up `gt_point_winner` (kept in DB, not consumed by current metric).
- Backfilling pre-existing S3 datasets — next push of each dataset adds it.
- A separate one-shot dump script — integrate into the existing pipeline only.
- `list-remote` reporting score-GT count.

## Architecture

Mirror the existing optional-GT pattern (`tracking_ground_truth.json`,
`action_ground_truth.json`, `player_matching_ground_truth.json`) end-to-end.
A new `score_ground_truth.json` artifact lives alongside them in each
dataset directory and in S3 under `datasets/<name>/`.

```
DB (rallies.gt_serving_team)
    │ export-dataset
    ▼
training_datasets/<name>/score_ground_truth.json
    │ push
    ▼
S3 .../datasets/<name>/score_ground_truth.json
    │ pull
    ▼
training_datasets/<name>/score_ground_truth.json
    │ restore (NULL-only upsert)
    ▼
DB (rallies.gt_serving_team)
```

## File Format

`score_ground_truth.json`:

```json
{
  "stats": {
    "total_rallies": 316,
    "total_videos": 63
  },
  "rallies": [
    {
      "rally_id": "fb8fd612-...",
      "video_id": "1a5da176-...",
      "content_hash": "abc123...",
      "gt_serving_team": "A"
    }
  ]
}
```

- Only rallies with non-NULL `gt_serving_team` are written.
- File is omitted entirely when zero matching rallies exist.
- `gt_serving_team` is `"A"` or `"B"` (matches DB convention).

## Components

### 1. Export — `analysis/rallycut/cli/commands/train.py`

Add a private helper:

```python
def _export_score_ground_truth(content_hashes: set[str]) -> dict | None:
    """Query rallies.gt_serving_team for rallies in the dataset's videos."""
```

- Single SQL: `SELECT r.id, r.video_id, v.content_hash, r.gt_serving_team
  FROM rallies r JOIN videos v ON v.id = r.video_id
  WHERE v.content_hash = ANY(%s) AND r.gt_serving_team IS NOT NULL`.
- Return `None` if no rows; else the dict shape above.

In `export_dataset()`, after the action-GT block, write
`score_ground_truth.json` when the helper returns non-None and print the
count line consistent with the others.

### 2. Push / Pull — `analysis/rallycut/training/backup.py`

- In `push_dataset`, append `score_ground_truth.json` to the
  `metadata_files` list if it exists locally.
- In `pull_dataset`, add a try/except `ClientError`-tolerant download
  block matching the existing optional GT downloads.
- No new dataclass fields, no schema changes — these files are pure
  pass-through metadata.

### 3. Restore — `analysis/rallycut/training/restore.py`

- Add `score_gt_restored: int = 0` to `RestoreResult`.
- Add `_restore_score_ground_truth(dataset_dir, dry_run, result)`:
  - Read `score_ground_truth.json`; skip if missing (matches existing
    optional GT handling).
  - For each entry, run:
    ```sql
    UPDATE rallies
       SET gt_serving_team = %s
     WHERE id = %s
       AND gt_serving_team IS NULL
    ```
  - Sum updated rowcounts into `result.score_gt_restored`.
  - Per-row failure (missing rally id) logs to `result.errors`, does not
    abort the restore.
- Wire into `restore_dataset_to_db` after the action-GT restore call.
- Mirror the file-presence check in the dry-run preview block.

**NULL-only update** is the key invariant: a re-run never overwrites a
label edited in the DB after the snapshot was taken. Acceptable
asymmetry — a deliberate NULL-out in the DB will be re-filled by the next
restore. That is a worthwhile trade for the safety of the common case.

### 4. CLI summary — `analysis/rallycut/cli/commands/train.py`

In the `restore` command's success block, after the existing
`Action GT restored` line, add:

```python
if result.score_gt_restored:
    rprint(f"  Score GT restored: [green]{result.score_gt_restored}[/green]")
```

## Error Handling

| Failure | Behavior |
| --- | --- |
| `score_ground_truth.json` missing on pull | Silent skip (matches tracking/action GT) |
| File missing on restore | Silent skip |
| Rally ID in JSON not in DB | Append per-row warning to `result.errors`, continue |
| `gt_serving_team` value invalid | Append per-row warning, continue |
| S3 upload error on push | Existing `metadata_files` upload loop already raises — no special handling |

## Testing

- Unit test: `_export_score_ground_truth` returns expected shape on a
  seeded rallies+videos fixture; returns `None` when no NULL-stripped
  rows remain.
- Unit test: `_restore_score_ground_truth` updates rallies with NULL
  `gt_serving_team` and leaves existing non-NULL values unchanged. A
  rally id missing from the DB appends an error string but does not
  raise.

No integration test against real S3 — push/pull are thin wrappers around
existing `metadata_files` plumbing already covered by the dataset-backup
flow.

## Out of Scope

- `gt_point_winner` (Q2 — not consumed today, can be added later by
  widening the JSON schema).
- Standalone one-shot script (Q1 — option B rejected).
- Restore behavior options A or C (Q3 — option B chosen: NULL-only).
- Updating `list-remote` to surface a Score-GT column.
- Backfilling already-pushed datasets without a re-push.

## Migration

Pre-existing S3 datasets gain a `score_ground_truth.json` only when
re-pushed. There is no migration script. The next routine push of each
named dataset (`beach_v3`, etc.) seeds S3.

## Open Questions

None — Q1/Q2/Q3 resolved during brainstorming.
