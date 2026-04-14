# Training-Dataset Backup Hardening

**Date:** 2026-04-14
**Status:** Design drafted, unscheduled

## Problem

An audit of `analysis/rallycut/training/backup.py` +
`analysis/rallycut/training/restore.py` turned up seven reliability
papercuts that each erode the "trust the backup" guarantee. None have
bitten yet, but several silently mask failures as success, which is the
worst failure mode for a disaster-recovery system.

The audit triggered after the `score_ground_truth.json` work, when the
original score-GT restore silently failed by matching on rally UUIDs
that change on restore — exactly the scenario the backup was built for.
That specific bug was fixed in commit `88dc2e3`. This spec captures the
broader class of issues that share the same "silent failure under load"
shape.

## Findings (by severity)

### Important — swallowed `ClientError` masks transport failures

`backup.py:374, 385, 396, 407` each catch `ClientError` to tolerate
optional files. A 403 (creds expired), 503 (throttling), or
NoSuchBucket is indistinguishable from 404 under the current logic and
produces a partial dataset that downstream restore treats as
"optional file not present."

**Fix:** inspect `e.response["Error"]["Code"]`. Swallow only `404` /
`NoSuchKey`. Re-raise everything else (or append to `result.errors`).

### Important — app-S3 upload runs inside DB transaction but is not transactional

`restore.py:218-222` calls `app_s3.upload_file(...)` inside the
`with conn.transaction():` block at `restore.py:158`. If the
transaction later raises (e.g. session insert hits a FK violation), the
DB rolls back but the MinIO/S3 objects remain orphaned at
`videos/{user_id}/{new_video_id}/...`.

**Fix:** collect `(local_path, s3_key)` tuples during the transaction
and upload after `conn.commit()`. Alternatively, track uploads and
compensate on rollback. The collect-then-upload pattern is simpler and
acceptable — an upload failure after commit just leaves the video
missing in S3 (fixable by re-running restore with
`--upload-to-app-s3`).

### Important — partial pull leaves a half-populated dataset dir

`backup.py:330-468` writes each metadata JSON and each symlink directly
into `output_dir/name/`. A failure mid-pull (network drop,
KeyboardInterrupt) leaves a dir that *looks* valid to subsequent
commands.

**Fix:** stage into `output_dir/.{name}.partial/`, then `os.replace` to
final name on success. On retry, detect the `.partial` dir and resume
or clobber.

### Important — no metadata integrity check

Videos are content-addressed (good), but `manifest.json`,
`ground_truth.json`, and the four optional GT files are trusted as
downloaded. Truncated download or S3 corruption silently propagates
into the DB.

**Fix:** upload each metadata file with an `ETag`/MD5 header, and on
pull verify `Content-MD5` matches. Alternatively, write a sibling
`.sha256` file per metadata file and verify in `pull_dataset`.

### Important — rallies silently skipped when video already in DB

`restore.py:240-244` only inserts rallies when the video was newly
inserted (`if old_video_id in inserted_video_ids`). If a video already
exists in the DB with *no* rallies (or with stale rallies), the GT
rallies in the dataset are never written, and the downstream GT
helpers then fail their `(content_hash, start_ms, end_ms)` lookup — now
with misleading "no matching rally" errors.

**Fix:** either upsert rallies (on `video_id + start_ms + end_ms`) or
surface a warning when `rallies` count in DB for an existing video
differs from the count in the dataset.

### Minor — DRY: 7-8 edit sites per new GT file

Adding the 5th optional GT type touched: the `_export_*` helper in
`train.py`, `export_dataset`, `push_dataset` metadata_files list,
`pull_dataset` download block, `restore_dataset_to_db` invocation,
`_preview_restore` display, CLI summary, module docstring.

**Fix:** registry pattern:

```python
OPTIONAL_GT_FILES: list[OptionalGTSpec] = [
    OptionalGTSpec(
        filename="tracking_ground_truth.json",
        export_fn=_export_tracking_ground_truth,
        restore_fn=_restore_tracking_gt,
        result_field="tracking_gt_restored",
        display_label="tracking GT",
    ),
    # ...
]
```

`push_dataset`, `pull_dataset`, `restore_dataset_to_db`, and
`_preview_restore` each become one loop over the registry.

### Minor — unvalidated `--name` in S3 key

`backup.py:205` interpolates the user-provided `name` into
`{prefix}/datasets/{name}/{filename}` with no allow-list.
`--name "../weights"` would write outside the datasets prefix.

**Fix:** validate against `^[A-Za-z0-9_.-]+$` at the Typer CLI
boundary.

## Priority

The first four (swallowed errors, non-transactional S3, partial pull,
metadata integrity) are the "silent failure" class that most erodes
trust in the backup. Fix those first.

The rally-insert-skip (#5) is adjacent to the UUID bug already fixed —
same family of "restore partially works, quietly." High priority for
the same reason.

DRY consolidation (#6) should precede the next optional-GT addition,
not follow it — the duplication is what lets UUID-style bugs sneak in.

`--name` validation (#7) is cheap and can piggyback on any of the above
PRs.

## Out of Scope

- Rewriting `_upload_one` / `_download_one` to stream rather than use
  `boto3.s3.transfer` (the transfer config already handles multipart).
- Introducing a separate manifest-signing key (over-engineering).
- Separating video storage from dataset metadata into different
  buckets.
- Backfilling or re-validating existing S3 datasets.

## Test Strategy

Each fix gets a focused unit test. The integration story is
"wipe DB, restore from S3, verify 620 rallies now have
`gt_serving_team` set." That is the one scenario this whole subsystem
exists to deliver and it is worth a single end-to-end integration test
that runs against a test database.

## Open Questions

- Keep `--upload-to-app-s3` orphans silent or surface them? (Leaning
  surface — the restore already has an `errors` list.)
- Introduce a `--force-overwrite` flag to re-fetch stale metadata on
  pull even when the local copy exists? (Probably yes — cheap.)
