# Project B — Rally-Edit Propagation & Incremental Re-Analysis

**Status:** Draft
**Date:** 2026-04-15
**Preceded by:** Project A1 (quality overhaul, merged `2474263`), A2a (progressive UX, `81cd8af`), A2b (resilience, `cf1e187`), Project C (sport sanity + auto-rotate, `3954498` + `a701861`).
**Scope parent:** `memory/e2e_pipeline_redesign_2026_04_15.md`

## Context

Projects A and C shipped the orchestration, progressive UX, resilience, and sport-sanity surfaces for the match-analysis pipeline. Rally *edits* during or after analysis, however, still leave downstream data in inconsistent states:

- **Split rally** does not exist as an operation (no route, no UI, no service). Users who hit a detector over-merge have no tool but delete+re-record.
- **Delete rally** cascade-drops `PlayerTrack` but does not clear `Video.matchAnalysisJson`; surviving `matchAnalysisJson.rallies[]` entries reference a rally ID that no longer exists, and `matchStatsJson` keeps aggregating the deleted rally's contributions until the next full match-analysis run.
- **Extend rally** (A2b) marks `needsRetrack=true` and catch-up tracks — *unless* the rally is canonical-locked, in which case the flag is silently dropped because the retrack filter excludes locked rallies. The extended tail never gets tracked.
- **Every edit** triggers a full 6-stage match-analysis rerun through the A2a debounce. For edit storms of scalar fields (score/notes/servingTeam) or deletes, we run cross-rally Hungarian and appearance-profile rebuilds that change nothing in their output.

Project B closes these gaps with: a real split/merge operation that slices existing `PlayerTrack` data rather than re-tracking from scratch, uniform canonical-lock guardrails at every structural edit path, and edit-type-gated match-analysis recomputation that runs the minimum-sufficient stage set per edit storm.

Project B inherits Project A's groundwork and does not re-open its decisions: `qualityReportJson` is the video-level quality surface, `WebhookDelivery` is the idempotency substrate, `PlayerTrack.needsRetrack` is the retrack signal, the A2a 5-second edit-quiescence debounce is the trigger for match-analysis.

## Scope

### In scope

- `POST /v1/rallies/:id/split` — cut-with-gap, slices existing `PlayerTrack` arrays in-place.
- `POST /v1/rallies/merge` — concatenates two rallies; adjacent case stitches tracks in-place, gap case falls through to catch-up retrack.
- `POST /v1/rallies/:id/unlock` — clears the `canonicalLocked` flag; leaves ground-truth JSON intact.
- `DELETE /v1/rallies/:id` — adds optional `{confirmUnlock: true}` body parameter for locked rallies; clears its entry from `Video.matchAnalysisJson.rallies[]`.
- Canonical-lock guardrails enforced uniformly at every structural edit path (`extend`, `split`, `merge`, `delete`).
- New `Video.pendingAnalysisEditsJson` column accumulating `{rallyId, editKind}` markers during the debounce window.
- New `Video.matchAnalysisRanAt` column for staleness checks and telemetry baseline.
- Edit-type-gated `runMatchAnalysis` that skips stages 2 (`match-players`) and 3 (`repair-identities`) on edit storms whose contents do not invalidate cross-rally mappings.
- `--rally-ids` flag on stage 4 (`remap-track-ids`) and stage 5 (`reattribute-actions`) CLIs for per-rally processing.
- Stage-timing telemetry (`match_analysis.stage_timings` structured log) so we can later quantify the performance story.

### Out of scope

- Warm-start Hungarian / preserved appearance-profile caches for stage 2/3 (deferred — stage-timing telemetry will tell us whether it's worth the engineering).
- Rich client-side optimistic UI for split/merge (server-authoritative; client refetches after the response).
- Persistent edit-audit `RallyEdit` table (the JSON column is sufficient until a consumer demands history).
- Parent-child rally lineage (no `parentRallyId` FK; split deletes the parent row and creates two unrelated children).
- Score-tracker / stats-engine behavior changes (`matchStatsJson` aggregation logic untouched; only *when* it runs changes).

## Operation × PlayerTrack-State Matrix

Every edit that can affect tracking is gated by the rally's `PlayerTrack.status` / `needsRetrack` state. The matrix is enforced at the API layer via `assertTrackStateAllows(op)` helpers.

| Op | `null` | `COMPLETED` | `PROCESSING` | `FAILED` | `needsRetrack=true` |
|---|---|---|---|---|---|
| Update scalar (`score`, `servingTeam`, `notes`, `confidence`) | ok | ok | ok | ok | ok |
| Shorten bounds | ok | reindex | 409 | 409 | reindex, keep flag |
| Extend bounds | ok | set `needsRetrack` | 409 | 409 | keep flag |
| Split | ok (children PENDING, catch-up tracks them) | slice data | 409 | 409 | slice, keep flag on both children |
| Merge (no gap) | ok | concat data | 409 | 409 | concat, flag on result |
| Merge (with gap) | delete both PTs, new rally PENDING | delete both PTs, new rally PENDING | 409 | 409 | delete both PTs, new rally PENDING |
| Delete | ok | ok | cascade cancels | ok | ok |

## Canonical-Lock Overlay

Applied at the API gate *before* the state matrix — a locked rally's edits are rejected regardless of `PlayerTrack` state.

| Op on locked rally | Response |
|---|---|
| Scalar update | allowed (lock preserved) |
| Shorten bounds | allowed (lock preserved; `reindexTrackingData` handles GT frames naturally) |
| Extend bounds | 409 `LOCKED_RALLY_CANNOT_EXTEND` |
| Split | 409 `LOCKED_RALLY_CANNOT_SPLIT` |
| Merge (either input locked) | 409 `LOCKED_RALLY_CANNOT_MERGE` |
| Delete without `confirmUnlock` | 409 `LOCKED_RALLY_REQUIRES_CONFIRM` (payload: `{gtFrameCount: number}`) |
| Delete with `confirmUnlock: true` | proceed; emit structured audit log line |

**Unlock endpoint.** `POST /v1/rallies/:id/unlock` sets `Video.matchAnalysisJson.rallies[i].canonicalLocked = false` and no other mutation. `PlayerTrack.groundTruthJson` / `actionGroundTruthJson` are preserved for possible re-lock. Idempotent: unlocking an already-unlocked rally returns 200 with `wasLocked: false`.

**Retry semantics for locked rallies.** Locked rallies are never retracked; the A2b retrack filter (`filterOutCanonicalLockedRallies`) stays in force. If a locked rally somehow lacks a `PlayerTrack` (corrupt state), it tracks fresh, but match-analysis stage 2 skips locked entries during Hungarian and uses their stored `trackToPlayer` verbatim — the lock's mapping cannot be overwritten by a stage-2 rerun. Stages 4/5 fail loudly on locked rallies rather than silently skipping.

## Data Model Changes

### Prisma schema additions

```prisma
model Video {
  // ... existing fields ...

  /// Accumulates edit-type markers during the match-analysis debounce window.
  /// Cleared atomically by trigger-match-analysis on successful run.
  /// Shape: { entries: Array<{ rallyId: string, editKind: EditKind, at: string }> }
  pendingAnalysisEditsJson Json?

  /// Timestamp of the last successful match-analysis run.
  matchAnalysisRanAt DateTime?
}
```

### TypeScript types (persisted as strings inside the JSON column)

```ts
type EditKind =
  | 'scalar'    // score / servingTeam / notes / confidence (non-structural)
  | 'shorten'   // bounds changed, no new frames, track IDs unchanged
  | 'extend'    // bounds changed, new frames, needsRetrack=true
  | 'delete'
  | 'split'
  | 'merge'
  | 'create';   // new rally created; catch-up will track it
```

### What does not change

- `PlayerTrack` schema: all slicing/concat operates on existing JSON blobs.
- `Rally` schema: no `parentRallyId` or lineage fields.
- `canonicalLocked` representation: stays in `matchAnalysisJson.rallies[].canonicalLocked`.

### Migration

Single file adding two nullable columns. No backfill. Reversible. `pendingAnalysisEditsJson=null` is equivalent to an empty entries list; `matchAnalysisRanAt=null` just means the telemetry baseline starts fresh.

### Data integrity invariant

After any committed edit transaction, if `pendingAnalysisEditsJson` has entries, either (a) the A2a debounce is armed, or (b) a `trigger-match-analysis` call is in flight. No third state. `pendingAnalysisEditsJson` writes happen as the last mutation inside every edit transaction — rollback takes them with everything else.

## Split Operation

**Route:** `POST /v1/rallies/:id/split`

**Request body:**

```ts
{
  firstEndMs: number,
  secondStartMs: number,
}
```

**Validation (400/409):**

- `startMs < firstEndMs <= secondStartMs < endMs`.
- Both values within video duration.
- Rally exists, user has write permission on its video.
- `PlayerTrack` state not in forbid set per the matrix (→ 409).
- Canonical-lock check (→ 409).

**Algorithm** (`rallyService.splitRally(rallyId, userId, { firstEndMs, secondStartMs })`, entirely inside one `prisma.$transaction`):

1. Load rally + `PlayerTrack` + relevant `matchAnalysisJson.rallies[i]` entry.
2. Run guardrails (state matrix + lock). Throw typed `AppError` on fail.
3. Compute frame partition points:
   - `fps = playerTrack?.fps ?? 30`
   - `firstEndFrame = round((firstEndMs - startMs) / 1000 * fps)`
   - `secondStartFrame = round((secondStartMs - startMs) / 1000 * fps)`
4. Build child `Rally` records:
   - First: `{startMs, endMs: firstEndMs, scoreA: prevRally?.scoreA ?? 0, scoreB: prevRally?.scoreB ?? 0, servingTeam: inferFromPrev(prevRally), notes: parent.notes, confidence: parent.confidence}`
   - Second: `{startMs: secondStartMs, endMs, scoreA: parent.scoreA, scoreB: parent.scoreB, servingTeam: parent.servingTeam, notes: parent.notes, confidence: parent.confidence}`
   - `prevRally` = rally in the same video with the greatest `endMs` strictly less than parent's `startMs`.
5. If `PlayerTrack` exists, build two sliced payloads via `slicePlayerTrack(pt, firstEndFrame, secondStartFrame)` (pure helper in `rallySlicing.ts`):
   - Partition `positionsJson`, `rawPositionsJson`, `ballPositionsJson`, `contactsJson`, `actionsJson`, `groundTruthJson`, `actionGroundTruthJson` by `frame` / `frameNumber`:
     - First child keeps entries with frame `< firstEndFrame` (frames unchanged).
     - Second child keeps entries with frame `>= secondStartFrame` (frames shifted down by `secondStartFrame`).
     - The middle segment `[firstEndFrame, secondStartFrame)` is discarded.
   - Recompute per-child `frameCount`, `detectionRate`, `avgConfidence`, `avgPlayerCount`, `uniqueTrackCount` from the partitioned arrays.
   - `processingTimeMs`, `modelVersion`, `courtSplitY` inherit from parent.
   - `status` inherits `COMPLETED`; `needsRetrack` inherits parent's value.
6. Create both child `Rally` rows; create both child `PlayerTrack` rows when applicable.
7. Delete parent `Rally` row (cascades parent `PlayerTrack`).
8. Read-mutate-write `Video.matchAnalysisJson`: remove the parent's `rallies[i]` entry, insert two new entries for the children with `trackToPlayer` inherited verbatim from the parent, `canonicalLocked: false`, `serverPlayerId` duplicated.
9. Append `{rallyId, editKind: 'split', at: now}` for each child to `Video.pendingAnalysisEditsJson`.
10. Return `{firstRally, secondRally}`.

**Key design choice.** The parent's `trackToPlayer` mapping propagates to both children rather than being cleared. Slicing preserves raw track IDs (we partition arrays, not re-detect tracks), so the parent's Hungarian output is still correct as a starting point. Stages 4 (remap) and 5 (reattribute) re-run per-child under edit-type gating; stages 2 and 3 stay skipped. This is what makes split cheap.

## Merge Operation

**Route:** `POST /v1/rallies/merge`

**Request body:**

```ts
{
  rallyIds: [string, string],  // exactly two
}
```

**Validation (400/409):**

- Exactly two rally IDs.
- Both rallies exist, same video, user has write permission.
- After server sorting by `startMs`, the second's `startMs >= first's endMs` (overlap → 400 `RALLIES_OVERLAP`).
- Neither canonical-locked (→ 409).
- Both `PlayerTrack` states in the allowed set per the matrix (→ 409).

**Two sub-cases.**

**No gap (`rallyA.endMs === rallyB.startMs`).** Contiguous concat in place.

- New rally fields: `{startMs: A.startMs, endMs: B.endMs, scoreA: B.scoreA, scoreB: B.scoreB, servingTeam: B.servingTeam, notes: nonEmptyJoin(A.notes, B.notes), confidence: min(A.confidence, B.confidence)}`.
- If both parents had a `PlayerTrack`, build `concatPlayerTracks(A.pt, B.pt, frameOffset=A.frameCount)`:
  - A's entries unchanged; B's entries with `frameNumber` shifted up by `A.frameCount` in each of `positionsJson`, `rawPositionsJson`, `ballPositionsJson`, `contactsJson`, `actionsJson`, `groundTruthJson`, `actionGroundTruthJson`.
  - Recomputed metadata: `frameCount = A.frameCount + B.frameCount`, aggregate `detectionRate`, `avgConfidence`, `avgPlayerCount`, `uniqueTrackCount = |union of track-ID sets|`.
  - `courtSplitY`: inherit from A (tie-break; full rerun of stages 2+3 immediately follows so any inconsistency is caught).
  - `status = COMPLETED` if both were COMPLETED; `needsRetrack = A.needsRetrack || B.needsRetrack`.

**Gap (`rallyA.endMs < rallyB.startMs`).** Falls through to catch-up retrack — the new merged rally is created with `startMs=A.startMs, endMs=B.endMs` and no `PlayerTrack`. Both parents' `PlayerTrack`s are dropped (cascade on parent delete). A2a's catch-up pipeline will produce a clean contiguous track from scratch.

Rationale for the gap fall-through: the frame-indexing inside `PlayerTrack` arrays is "compact tracked frame" semantics (frame 0 = first frame of the tracked span). Preserving two partial tracks around an untracked middle would introduce a mismatch between `frameNumber` (compact) and `startMs/endMs` (continuous wall time) that breaks downstream consumers. Re-tracking the full span is cleaner and matches the A2a new-rally flow.

**edit-type gating.** Merge is always marked `editKind: 'merge'` and triggers full rerun, regardless of sub-case. Even the no-gap case collapses two parents' Hungarian columns into one that must be re-solved.

## Canonical-Lock Enforcement

**Guard helper:** `api/src/services/canonicalLockGuard.ts`

```ts
export async function isRallyLocked(tx, rallyId): Promise<boolean>
export async function assertNotLocked(tx, rallyId, opCode): Promise<void>
```

Every structural edit path calls `assertNotLocked(tx, rallyId, 'EXTEND' | 'SPLIT' | 'MERGE')` inside its transaction before mutation. Delete calls `isRallyLocked` and only throws if the request body lacks `{confirmUnlock: true}`.

**Extend-path integration** in `rallyService.updateRally`:

```ts
if (boundsExtended(old, next)) {
  await assertNotLocked(tx, rallyId, 'EXTEND');       // new: 409 if locked
  await markRetrackIfExtended(tx, rallyId, old, next); // existing A2b behavior
}
```

Scalar updates and shortens bypass the lock gate — they do not invalidate ground truth.

**Unlock endpoint body:**

```ts
// POST /v1/rallies/:id/unlock
// Response: { rallyId, wasLocked, unlockedAt }
```

Inside one transaction: read `Video.matchAnalysisJson`, locate the rally's entry, set `canonicalLocked = false` (idempotent), write back. Does not touch `PlayerTrack.groundTruthJson` / `actionGroundTruthJson`, `pendingAnalysisEditsJson`, or trigger any retrack.

**Delete-with-confirm path:** handler inspects `isRallyLocked`. If locked and `!body.confirmUnlock`, throws `LockedRallyRequiresConfirmError` with payload `{gtFrameCount}`. If locked and `confirmUnlock === true`, proceeds and emits `rally.locked.deleted {rallyId, videoId, userId, gtFrameCount}` structured log for post-hoc audit.

## Edit-Type-Gated Match-Analysis Recomputation

**Entry point unchanged:** `POST /v1/videos/:id/trigger-match-analysis` (from A2a). Internals restructured.

**`runMatchAnalysis(videoId, userId)` outline:**

```ts
async function runMatchAnalysis(videoId, userId) {
  const edits = await consumePendingEdits(videoId);
  const plan = planStages(edits);

  if (plan.fullRerun) {
    await runStages([1, 2, 3, 4, 5, 6], { allRallies: true });
  } else {
    await runStages([1], { allRallies: true });
    if (plan.runStage2) await runStages([2, 3], { allRallies: true });
    if (plan.changedRallyIds.length) {
      await runStages([4, 5], { rallyIds: plan.changedRallyIds });
    }
    await runStages([6], { allRallies: true });
  }

  await prisma.video.update({
    where: { id: videoId },
    data: { matchAnalysisRanAt: new Date() },
  });
  log.info('match_analysis.stage_timings', { videoId, plan, timingsMs });
}
```

**`planStages(edits)`** — pure function, trivially unit-testable:

- Any `extend | create | merge` edit → `fullRerun: true`.
- Otherwise: `runStage2: false`, `runStage3: false`, `changedRallyIds: dedupe(edits.filter(e => e.editKind in {shorten, delete, split}).map(e => e.rallyId))`.
- `scalar`-only storm → `changedRallyIds = []`; only stage 6 runs.
- `delete`-only storm → `changedRallyIds = []` (deleted rallies have no track to remap/reattribute); only stage 6 runs; deleted rallies drop from aggregation naturally.

**Per-rally CLI plumbing.** Stages 4 (`rallycut remap-track-ids`) and 5 (`rallycut reattribute-actions`) gain a `--rally-ids` flag. Without it → current behavior (all tracked rallies). With it → processes only the listed IDs. Flag plumbed through `matchAnalysisService.remapTrackIds(videoId, {rallyIds?})` and `reattributeActions(videoId, {rallyIds?})`.

**Safety of stage-2 skip** when `runStage2=false`. The existing `matchAnalysisJson.rallies[]` is used as-is:

- `scalar` edits don't touch tracks.
- `shorten` leaves raw track IDs unchanged; only frame indices shift via `reindexTrackingData`.
- `delete` removes a rally; surviving rallies' mappings are unaffected.
- `split` propagates parent's `trackToPlayer` to both children; mappings are correct in the stored JSON.

**`consumePendingEdits(videoId)` atomicity:** a single transaction reads the column, sets it to null, returns the old value. If a rally CRUD commit lands between our read and the start of stage work, the edit is queued for the *next* debounce cycle (the CRUD write's transaction commits its edit marker into a now-null column, which `trigger-match-analysis`'s next fire will consume). This is consistent with A2a's 5s debounce semantics.

**Backward compatibility:** `pendingAnalysisEditsJson=null` → `planStages({entries: []}).fullRerun = true`. Safe default — matches today's behavior.

**Contention guard:** A2a's `runningVideos` in-memory Set stays in place; one-analysis-at-a-time plus atomic consume means no interleaved plans on the same video.

## Error Handling

All new paths use typed `AppError` subclasses via the existing `errorHandler` middleware.

| Class | HTTP | Code | Raised by |
|---|---|---|---|
| `LockedRallyError` | 409 | `LOCKED_RALLY_CANNOT_EXTEND` / `_SPLIT` / `_MERGE` | canonical-lock guard |
| `LockedRallyRequiresConfirmError` | 409 | `LOCKED_RALLY_REQUIRES_CONFIRM` | delete path |
| `RallyTrackingStateError` | 409 | `RALLY_TRACKING_IN_PROGRESS` / `RALLY_TRACKING_FAILED` | split / merge |
| `RalliesOverlapError` | 400 | `RALLIES_OVERLAP` | merge |
| `SplitBoundsError` | 400 | `SPLIT_BOUNDS_INVALID` | split validation |

Every error's `details` field is structured (never a free-form string) so the frontend can render specific copy without string-matching.

## Transaction-Safety Invariants

1. `pendingAnalysisEditsJson` writes happen as the last mutation inside every edit transaction. Rollback takes them with everything else.
2. `markRetrackIfExtended` stays transaction-scoped (A2b pattern).
3. `consumePendingEdits` is a single-statement transaction; no overlap with long-running stage CLIs.
4. Split / merge parent-delete + children-create + `matchAnalysisJson` mutation + pending-edit write are all in one `prisma.$transaction`. All-or-nothing.

## Testing

### Unit (pure helpers, no DB)

- `slicePlayerTrack`: exact-frame split, split-at-frame-0 (degenerate), split-at-last-frame (degenerate), gap spanning entire middle, frame-rounding edge cases.
- `concatPlayerTracks`: adjacent concat, track-ID collision (documented as a `merge` edit-kind → full rerun handles it).
- `planStages(edits)`: every combination of edit kinds → assert `{fullRerun, runStage2, changedRallyIds}`.
- `assertNotLocked` / `isRallyLocked`: locked / unlocked / missing-entry cases.

### Integration (with Prisma + transactions)

- Split happy path on `COMPLETED` `PlayerTrack`: assert child rows, child `PlayerTrack`s, `matchAnalysisJson.rallies[]` replacement, `pendingAnalysisEditsJson` entries, parent deleted.
- Split rollback: force an inner step to throw → assert parent + children + `matchAnalysisJson` all reverted.
- Split on locked rally: 409, no DB mutation.
- Split on `PROCESSING`: 409.
- Split on `null` `PlayerTrack`: children created with `PlayerTrack=null`.
- Merge no-gap + merge gap (distinct branches).
- Merge with either input locked: 409.
- Delete locked without `confirmUnlock`: 409; with: succeeds, audit log emitted.
- Unlock idempotent on already-unlocked rally.
- Unlock then extend: succeeds (contrast: extend on locked rally fails).
- `trigger-match-analysis` with varied `pendingAnalysisEditsJson` → verify stage plan + CLI invocation pattern (mock CLIs, assert invocation args).
- `consumePendingEdits` race: concurrent rally edit lands during analysis run → appears in the next cycle, not lost.
- Stage-timing log lines have the expected shape.

### End-to-end (manual)

1. Upload a match; detection + tracking complete.
2. Split one rally — UI shows two rallies, tracking data visible on both within ~1s (no re-detect delay). Stats refresh within 5s.
3. Merge two adjacent rallies — contiguous tracking data visible; stats refresh.
4. Merge two non-adjacent rallies — merged rally shows tracking state, completes within catch-up window, stats refresh.
5. Attempt to extend a canonical-locked rally → 409 → UI modal "Unlock to continue?" → unlock → retry → succeeds.
6. Rapid edits: delete 2 rallies + shorten 1 + update a score within 5s → single match-analysis run fires, logs show `runStage2=false`.
7. Crash simulation: kill API mid-split → restart → rally state is either fully-parent or fully-split. No half-state.

### Regression guards

- No inline `res.status(409).json(...)` in new code (enforce typed errors).
- No top-level `prisma.` inside split / merge / unlock paths (enforce `tx.` threading).
- `canonicalLockGuard` imported by every route that mutates rally bounds or deletes.

**Calibration.** None. Project B is plumbing, not detection — "empirical lift" does not apply. The verification bar is correctness plus stage-timing telemetry.

## Implementation Sequence

Each commit unit leaves the system working. Stop points sit at commit boundaries.

1. **Schema migration** — `pendingAnalysisEditsJson`, `matchAnalysisRanAt`. Nullable, no backfill.
2. **Canonical-lock guard + unlock endpoint** — smallest standalone piece; no behavior change to existing flows.
3. **Edit-kind tracking in existing CRUD paths** — `createRally`, `updateRally` (scalar vs shorten vs extend classification), `deleteRally`. Still full-rerun match-analysis. Isolates the accumulate-edits mechanism.
4. **Stage-gated `runMatchAnalysis`** — `consumePendingEdits`, `planStages`, `--rally-ids` flag on stages 4/5 CLIs. Behavior change: scalar/shorten/delete storms skip stages 2/3. Ships `matchAnalysisRanAt` + stage-timing telemetry.
5. **Split endpoint + slicing helper + tests.**
6. **Merge endpoint + concat helper + tests** (both sub-cases).
7. **Client wiring** — `web/src/services/api.ts` + `editorStore.ts` actions for split / merge / unlock. `notifyRallyEdited` call on each new mutation. UI polish for the scrubber is deferred.

## Files

### New (≈8)

- `api/src/services/rallySlicing.ts` — pure slice/concat helpers.
- `api/src/services/canonicalLockGuard.ts` — `isRallyLocked`, `assertNotLocked`.
- `api/src/services/matchAnalysisPlanning.ts` — `planStages`, `consumePendingEdits`.
- `api/src/services/pendingAnalysisEdits.ts` — append helpers called from rally CRUD.
- `api/src/errors/rallyErrors.ts` — typed `AppError` subclasses.
- `api/prisma/migrations/20260416000000_project_b/migration.sql`.
- `api/tests/rallySlicing.test.ts`.
- `api/tests/matchAnalysisPlanning.test.ts`.

### Modified (≈8)

- `api/prisma/schema.prisma`.
- `api/src/services/rallyService.ts` — split + merge + delete-with-confirm + unlock.
- `api/src/services/batchTrackingService.ts` — retains A2b behavior; no structural change.
- `api/src/services/matchAnalysisService.ts` — replaces `runMatchAnalysis` body with stage-gated version; adds `--rally-ids` threading.
- `api/src/routes/rallies.ts` — new split / merge / unlock routes; delete body param.
- `analysis/rallycut/cli/commands/remap_track_ids.py`, `reattribute_actions.py` — add `--rally-ids`.
- `api/CLAUDE.md` — document Project B invariants.
- `web/src/services/api.ts`, `web/src/stores/editorStore.ts` — client wiring.

## Rollout

- No feature flag. New endpoints are additive; previously returned 404. Stage-gated match-analysis falls through to full-rerun when `pendingAnalysisEditsJson=null`, so first-run-post-deploy matches today's behavior.
- Single-pod deploy (existing topology).
- Post-deploy observability: watch `match_analysis.stage_timings` log for 24 hours; assemble the first real cost breakdown.

## Follow-Ups Not Addressed Here

- Warm-start Hungarian (only if stage 2 dominates real traffic once we have telemetry).
- Client-side split/merge scrubber UI polish.
- Persistent edit-audit table (if regulatory / GT-debug needs arise).

## Decision Log (for reference by future sessions)

- **Split is cut-with-gap, not cut-at-point.** The pattern matches the primary product motivation (detector includes dead time between two rallies as one rally). Cut-at-point falls out as the `firstEndMs === secondStartMs` degenerate case.
- **Field inheritance on split is hybrid (option d).** Scores + servingTeam follow end-state to second child; notes + confidence duplicate; `matchAnalysisJson.rallies[i]` cleared then repopulated with inherited `trackToPlayer`.
- **Canonical-locked extend/split/merge forbidden with 409.** Delete requires explicit `{confirmUnlock: true}`. Unlock preserves ground truth.
- **Edit-type-gated recomputation, not warm-start Hungarian.** Performance matters but engineering budget does not justify stage 2/3 incremental math until telemetry shows the cost.
- **No `parentRallyId` lineage, no `RallyEdit` audit table.** Schema weight without a query consumer.
- **PROCESSING / FAILED `PlayerTrack` → 409 on split/merge.** Forbid-and-retry beats a latent queued-edit state machine.
- **Gap merge falls through to catch-up retrack.** Preserving partial tracks around untracked middle frames would break `frameNumber` ↔ wall-time semantics for downstream consumers.
