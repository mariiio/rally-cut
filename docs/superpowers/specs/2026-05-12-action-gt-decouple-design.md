# Action GT Decoupling — Stable Identifier Snapshot

**Date:** 2026-05-12
**Status:** Spec — pending implementation plan

## Problem

`actionGroundTruthJson` lives on `PlayerTrack` (1-to-1 with `Rally`) as a JSON array of
`{frame, action, trackId?, playerTrackId?, ballX?, ballY?}`. The `trackId` is the raw
BoT-SORT id, stable only across `match-players` re-runs — not across a rally retrack,
model bump, or any path that calls `saveTrackingResult` again.

Lifecycle hazards observed in the current schema:

| Event                              | PlayerTrack row     | `actionGroundTruthJson`                                  |
| ---------------------------------- | ------------------- | -------------------------------------------------------- |
| Retrack (extend / manual / model)  | survives via upsert | **silently stranded** — `trackId`s point to dead namespace |
| Rally delete                       | cascaded away       | gone                                                     |
| Merge-with-gap                     | cascaded away       | gone (both parents)                                      |
| Merge-no-gap                       | dropped + recreated | concatenated by `rallySlicing.concatPlayerTracks`        |
| Split                              | dropped + 2 created | partitioned by `rallySlicing.partitionByFrame`           |
| Rally bounds reindex               | survives            | frame-shifted in place                                   |

The silent-stranding case is the worst because nothing screams. `remap_track_ids` is
the one place that rewrites GT to keep mappings consistent, and it is load-bearing in
a way that is not obvious from reading the codebase.

## Goal

A new table — `rally_action_ground_truth` — that stores each label as a self-contained
assertion anchored to **pixels** (snapshot bbox) rather than to a tracking-derived
identifier. The label survives any downstream tracking change; a re-resolver re-derives
the `trackId` attribution after each retrack. Failure to re-resolve is a visible state
(`UNRESOLVED`), not silent corruption.

The label represents a human claim about the physical world ("at frame F, the player in
this rectangle of pixels did action A"). The bbox is the only field anchored to the
physical world; every other identifier (trackId, playerTrackId, team) is derived
attribution downstream of tracking and so must be re-derivable, never load-bearing.

## Schema

```prisma
enum ActionLabel {
  SERVE
  RECEIVE
  SET
  ATTACK
  BLOCK
  DIG
}

enum ResolveSource {
  SNAPSHOT_EXACT     // re-resolved by trackId equality at this frame
  IOU_MATCH          // re-resolved by bbox IoU against current tracking
  NEAREST_CENTER     // re-resolved by center-point distance fallback
  MANUAL             // labeler reattached via UI
  UNRESOLVED         // no acceptable match — terminal-until-next-retrack
}

enum Team {
  A
  B
}

model RallyActionGroundTruth {
  id                 String           @id @default(uuid())
  rallyId            String           @map("rally_id")
  frame              Int                                       // rally-relative
  action             ActionLabel

  // Stable identifier snapshot — immutable after write.
  // Bbox is the pixel-anchored ground truth claim.
  snapshotBboxX1     Float?           @map("snapshot_bbox_x1") // normalized [0,1]
  snapshotBboxY1     Float?           @map("snapshot_bbox_y1")
  snapshotBboxX2     Float?           @map("snapshot_bbox_x2")
  snapshotBboxY2     Float?           @map("snapshot_bbox_y2")
  snapshotBallX      Float?           @map("snapshot_ball_x")
  snapshotBallY      Float?           @map("snapshot_ball_y")
  snapshotTeam       Team?            @map("snapshot_team")    // advisory, used as cross-check
  snapshotTrackId    Int?             @map("snapshot_track_id")// hint / audit trail (the trackId the labeler clicked)

  // Resolved attribution — cache, re-derived from snapshot against current PlayerTrack.
  resolvedTrackId    Int?             @map("resolved_track_id")
  resolvedAt         DateTime?        @map("resolved_at") @db.Timestamptz(6)
  resolvedSource     ResolveSource?   @map("resolved_source")

  createdAt          DateTime         @default(now()) @map("created_at")
  updatedAt          DateTime         @updatedAt @map("updated_at")
  createdBy          String?          @map("created_by")        // userId of the labeler (null on migrated rows)

  rally              Rally            @relation(fields: [rallyId], references: [id], onDelete: Cascade)

  @@unique([rallyId, frame, action])
  @@index([rallyId])
  @@index([rallyId, frame])
  @@map("rally_action_ground_truth")
}
```

Design notes:

- **Nullable bbox.** Migration from existing data can only snapshot a bbox when the
  player was tracked at the labeled frame. Where it cannot, `snapshotTrackId` is the
  only surviving hint and `resolvedSource = UNRESOLVED` until further notice.
- **No FK on PlayerTrack** by design — that is the decoupling.
- **`onDelete: Cascade` on Rally** is intentional and matches today's behavior. The
  merge-with-gap GT-loss case is closed at the write path (Section "Write path"),
  not at the schema layer.
- **Unique on `(rallyId, frame, action)`** allows the rare double-action-same-frame
  case (e.g., simultaneous block + dig is theoretically possible) while blocking
  accidental duplicates from double-submits.

## Write path

A single helper owns all writes. Two real callers, plus a re-resolver.

### Labeler UI — `saveActionGroundTruth(rallyId, labels, userId)`

The labeler clicks a rendered player box in the web UI. `PlayerOverlay` is already
drawing that bbox from `positionsJson`. The UI passes `{frame, action, trackId}`
to the API. For each label, the API:

1. Looks up `PlayerTrack.rawPositionsJson` and finds the bbox for `(frame, trackId)`
   → `snapshotBbox*`.
2. Looks up `PlayerTrack.ballPositionsJson` at `frame` → `snapshotBall*`.
3. Resolves team via `appliedFullMapping[trackId]` → canonical pid →
   `Video.teamAssignmentsJson` → `snapshotTeam`.
4. Sets `snapshotTrackId = trackId`, `resolvedTrackId = trackId`,
   `resolvedSource = SNAPSHOT_EXACT`, `resolvedAt = now()`, `createdBy = userId`.
5. `upsert` on `(rallyId, frame, action)`.

**Two paths through "no bbox to snapshot":**

- **PlayerTrack does not exist** (the current code rejects this with "Track players
  first"). The new path accepts the label with `snapshotBbox* = null`,
  `snapshotBall* = null`, `snapshotTeam = null`, `resolvedSource = UNRESOLVED`. The
  `snapshotTrackId` is whatever the UI passed (may be null).
- **PlayerTrack exists but `(frame, trackId)` has no entry in `rawPositionsJson`.**
  This is the offscreen-server case from `adaptive_candidate_window_v30_2026_05_11.md`.
  Same outcome: `snapshotBbox* = null`, `snapshotBall*` filled from the ball trace if
  the ball is tracked at that frame, `snapshotTeam` resolved if the trackId is in
  `primary_track_ids`, `resolvedSource = UNRESOLVED`.

The labels are real and must not be rejected. Both surface as `UNRESOLVED` in the
audit CLI.

### Re-resolver — invoked from `saveTrackingResult` (same transaction)

Hook point: extend the existing `prisma.$transaction` in
`api/src/services/playerTrackingService.ts:680-730`. After the PlayerTrack upsert
completes, before the transaction commits:

```
for row in SELECT * FROM rally_action_ground_truth WHERE rally_id = $1:
    if row.snapshotBbox is null:
        if positions_at(row.frame, row.snapshotTrackId) exists:
            resolvedTrackId = row.snapshotTrackId; resolvedSource = SNAPSHOT_EXACT
        else:
            resolvedTrackId = null;                resolvedSource = UNRESOLVED
    else:
        candidates = positions[row.frame]                       # current run
        best = argmax_c IoU(row.snapshotBbox, c.bbox)
        if IoU(best) >= 0.5:
            resolvedTrackId = best.trackId; resolvedSource = IOU_MATCH
        elif center_dist(row.snapshotBbox, best) <= 0.10:        # normalized
            resolvedTrackId = best.trackId; resolvedSource = NEAREST_CENTER
        else:
            resolvedTrackId = null;         resolvedSource = UNRESOLVED
    resolvedAt = now()
    UPDATE rally_action_ground_truth SET ... WHERE id = row.id
```

Thresholds: `IoU >= 0.5` (standard detection threshold), `center_dist <= 0.10`
(normalized — roughly one player width on a 1920×1080 frame). Both are pinned here
and reviewable when calibration data exists.

The resolver is best-effort and never throws. `UNRESOLVED` is a valid terminal state,
surfaced by the audit CLI (Section "Audit") and the labeling UI (ghost overlay +
reattach button).

### Other write paths

- **`rallySlicing.partitionByFrame` / `concatPlayerTracks`:** replace their JSON-array
  surgery on `actionGroundTruthJson` with row INSERT/UPDATE on the new table.
  - Split: partition rows by frame, shift second half's `frame` down by
    `secondStartFrame`.
  - Concat (merge-no-gap): copy B's rows into A's rally with `frame += a.frameCount`.
  - **Merge-with-gap:** explicitly copy both parents' rows into the new merged rally,
    *inside the same transaction* that creates the merged rally. Closes today's GT-loss
    cascade. Frame indices on the second parent's rows shift up by the duration of
    parent A; rows from the cut-gap frames are dropped (matches today's per-frame
    silence in the gap region).
- **`remap_track_ids`:** delete the `action_ground_truth_json` rewrite block entirely.
  Remap no longer touches GT — the snapshot is bbox-anchored, the resolver runs after
  every `saveTrackingResult` anyway.
- **Rally bounds reindex** (`playerTrackingService.ts:1602-1606`): becomes a SQL
  UPDATE on the GT rows for that rally, `frame -= deltaFrames`, then DELETE WHERE
  `frame < 0 OR frame >= newFrameCount`.

## Read path

### Web labeling UI — `GET /v1/rallies/:id/action-ground-truth`

Returns each row with:
```
{ frame, action,
  resolvedTrackId, resolvedSource,
  snapshotBboxX1, snapshotBboxY1, snapshotBboxX2, snapshotBboxY2,
  snapshotBallX, snapshotBallY,
  snapshotTeam,
  createdAt, updatedAt }
```

Rendered display pid is computed at render time via `appliedFullMapping[resolvedTrackId]`
with `trackToPlayer[resolvedTrackId]` fallback — same as today. When `resolvedTrackId`
is null, the UI shows the snapshot bbox as a ghost overlay with an "unresolved" badge
and a 1-click reattach action that calls a new `POST .../action-ground-truth/:id/reattach`
endpoint, setting `resolvedSource = MANUAL`.

### Training / eval — Python

`train.py`, `eval_action_detection.py`, `measure_attribution_*`,
`audit_action_gt_vs_click_gt.py`, `eval_visual_attribution.py` (full list in
`analysis/scripts/` from the grep at brainstorm time). The query pattern becomes:

```sql
SELECT frame, action, resolved_track_id,
       snapshot_bbox_x1, snapshot_bbox_y1, snapshot_bbox_x2, snapshot_bbox_y2,
       snapshot_ball_x, snapshot_ball_y, snapshot_team
FROM rally_action_ground_truth
WHERE rally_id = ANY($1)
  AND resolved_track_id IS NOT NULL
ORDER BY rally_id, frame
```

The `IS NOT NULL` filter is mandatory for training and accuracy eval — never train on
or evaluate against an unresolved label. Audit scripts read all rows including
unresolved.

### Backup / restore — `restore.py`

`restore.py` upserts onto rows of `rally_action_ground_truth` keyed by
`(rallyId, frame, action)` instead of writing a JSON blob on `player_tracks`. The
backup format becomes one row per label rather than one JSON array per rally.

## Migration

Single-source-of-truth cutover, no dual-read window.

### Step 1 — Prisma migration (additive)

Create `rally_action_ground_truth` + the three enums. Do not touch
`player_tracks.action_ground_truth_json` yet.

### Step 2 — Backfill (`analysis/rallycut/cli/commands/migrate_action_gt.py`)

Idempotent (`ON CONFLICT ... DO NOTHING`), rerunnable. For every `PlayerTrack` with a
non-null `actionGroundTruthJson`:

```
for label in pt.action_ground_truth_json:
    bbox = find_bbox(pt.raw_positions_json, label.frame, label.trackId)
    ball = find_ball(pt.ball_positions_json, label.frame)
    team = resolve_team(pt, label.trackId)
    INSERT INTO rally_action_ground_truth (...) VALUES (...)
    ON CONFLICT (rally_id, frame, action) DO NOTHING
```

Three outcomes per label, all written, with a summary report at the end:

| Outcome                                                    | bbox  | resolvedSource    |
| ---------------------------------------------------------- | ----- | ----------------- |
| Track present at frame                                     | found | `SNAPSHOT_EXACT`  |
| TrackId in `primary_track_ids` but no entry at this frame  | null  | `UNRESOLVED`      |
| TrackId not in `primary_track_ids` (offscreen-server etc.) | null  | `UNRESOLVED`      |

Expected on the panel (gigi/wawa/cece): the bulk are `SNAPSHOT_EXACT`; the ~5
absent-server cases from `adaptive_candidate_window_v30_2026_05_11.md` come through
as `UNRESOLVED`. Anything else is a migration anomaly worth investigating.

The backfill also writes a JSON dump of the pre-cutover state to
`backups/action_gt_pre_cutover_<ISO8601>.json` keyed by `rallyId` →
`actionGroundTruthJson`. This is the rollback artifact.

### Step 3 — Cutover commit

A single PR that:
- Switches all writers (API service + web client + Python scripts) to the new table.
- Switches all readers (training, eval, restore, web).
- Drops the column: `ALTER TABLE player_tracks DROP COLUMN action_ground_truth_json`.

### Pre-cutover guard

A CI check that fails the build if any reference to `actionGroundTruthJson` or
`action_ground_truth_json` remains in `src/`, `analysis/rallycut/`, `analysis/scripts/`
outside the migration script and the rollback restore path. This is a single `grep`
in the lint job.

### Rollback

`git revert` the cutover PR. Apply the backup dump via `restore.py` against the
old schema. The dropped column is the only destructive step and the backup is
captured before it.

## Audit

New CLI: `rallycut audit-action-gt --video-id <id>` (or `--all`).

Reports per video:
- Total labels in `rally_action_ground_truth`.
- Counts per `resolvedSource`.
- List of rallies where `UNRESOLVED` count exceeds a threshold (default ≥1).
- Cross-check: labels where `snapshotTeam` differs from
  `appliedFullMapping[resolvedTrackId]` → canonical pid → current team (i.e., a
  team-flip happened across the retrack — surface but do not auto-correct).

Sits next to `audit-pid-invariants`, `audit-coherence-invariants` in the audit suite.

## Invariants

1. Every successful save sets `resolvedAt` to `now()` and a non-null `resolvedSource`.
   Never null on a fresh write.
2. `saveTrackingResult` re-resolves every GT row in its rally within the same
   transaction. The resolver never throws; failure to find a match writes
   `UNRESOLVED`.
3. `onDelete: Cascade` from Rally; merge-with-gap explicitly copies parent GT rows
   into the new merged rally inside the rally-creation transaction.
4. `0 <= frame < playerTrack.frameCount` enforced on write when a PlayerTrack
   exists. When no PlayerTrack exists, frame is accepted as-is (no upper bound to
   validate against); the resolver will run when tracking arrives.
5. `(rallyId, frame, action)` uniqueness via Prisma `upsert`.
6. **Bbox quad atomicity:** all four `snapshotBbox*` fields are non-null together
   or all four null together. Enforced in application code at every write site;
   not a DB-level CHECK constraint (would clutter the migration).

## Testing

- **API unit:**
  - save-with-PlayerTrack writes bbox snapshot
  - save-without-PlayerTrack writes UNRESOLVED row
  - re-save on existing `(rally, frame, action)` updates in place
  - cascade delete via Rally
- **API integration:**
  - retrack flow re-resolves correctly via IoU
  - retrack with player off-screen at labeled frame yields UNRESOLVED
  - merge-with-gap preserves both parents' GT in the new rally
  - merge-no-gap, split, reindex all preserve GT correctly
- **Python integration:**
  - `train.py --use-gt` on panel fixtures produces an identical labeled training set
    pre/post migration (modulo `UNRESOLVED` filtering)
  - eval baselines on the GT panel match within `±0pp` PERMUTED PID accuracy
    (this is a non-regression gate — the schema change MUST NOT move measurements)
- **Migration:**
  - apply backfill against a snapshot of prod
  - `diff` the `(rallyId, frame, action)` set between the JSON-array source and the
    new table — every label survives or is explained by an UNRESOLVED row
- **Audit CLI:** integration test on a fixture with one known UNRESOLVED row.

## Scope explicitly excluded

- No changes to the action label schema itself (still `SERVE | RECEIVE | SET | ATTACK
  | BLOCK | DIG`).
- No new identity signals — bbox + team is the snapshot, nothing visual (no crops,
  no embeddings). Those are separate workstreams.
- No autoresolve of `UNRESOLVED` by inventing attribution. The labeler reattaches
  manually.
- No changes to the offscreen-server detection path itself; this spec just makes its
  GT labels representable without bbox.

## Open questions

- Is there demand to allow >1 GT label per `(rallyId, frame)` for *different* players
  doing different actions at the same frame? Current uniqueness `(rallyId, frame,
  action)` allows it (two players, two different actions, same frame). If we ever
  see two players doing the *same* action at the same frame the constraint would
  reject the second — flag now to revisit if a real case shows up.
