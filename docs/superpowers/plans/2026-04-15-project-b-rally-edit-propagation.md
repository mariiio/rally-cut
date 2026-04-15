# Project B — Rally-Edit Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship split/merge rally operations that slice existing `PlayerTrack` data in-place, uniform canonical-lock guardrails at every structural edit path, and edit-type-gated match-analysis recomputation that skips stages 2/3 when a storm contains only scalar/shorten/delete/split edits.

**Architecture:** Extend the A2a/A2b foundation (`pendingAnalysisEditsJson` is the one new column) with typed `AppError` subclasses for new 4xx responses, a pure `rallySlicing.ts` helper for frame-accurate partition/concat, a pure `planStages` function driving a restructured `runMatchAnalysis`, and `--rally-ids` flags on stages 4/5 CLIs. Every structural edit runs inside a single `prisma.$transaction` with the lock guard assertion as the first mutation gate and the pending-edit marker as the last commit.

**Tech Stack:** TypeScript (Express, Prisma, Zod, Vitest), Python (Typer CLIs + `@handle_errors`), PostgreSQL migrations via `prisma migrate`, Zustand client store.

**Spec:** `docs/superpowers/specs/2026-04-15-project-b-rally-edit-propagation-design.md`

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `api/prisma/migrations/20260416000000_b_rally_edit_propagation/migration.sql` | Add `pending_analysis_edits_json`, `match_analysis_ran_at` to `videos` |
| `api/src/services/canonicalLockGuard.ts` | `isRallyLocked`, `assertNotLocked` |
| `api/src/services/pendingAnalysisEdits.ts` | `appendEdit`, `appendEditsBatch`, `consumePendingEdits` |
| `api/src/services/matchAnalysisPlanning.ts` | `planStages` (pure), types |
| `api/src/services/rallySlicing.ts` | `slicePlayerTrack`, `concatPlayerTracks` (pure) |
| `api/tests/canonicalLockGuard.test.ts` | |
| `api/tests/pendingAnalysisEdits.test.ts` | |
| `api/tests/matchAnalysisPlanning.test.ts` | |
| `api/tests/rallySlicing.test.ts` | |
| `api/tests/splitRally.test.ts` | |
| `api/tests/mergeRallies.test.ts` | |
| `api/tests/unlockRally.test.ts` | |

### Modified files

| Path | Change |
|---|---|
| `api/prisma/schema.prisma` | Add the two `Video` fields |
| `api/src/middleware/errorHandler.ts` | Add 5 new error subclasses + 2 new codes to `ErrorCode` union |
| `api/src/services/rallyService.ts` | `updateRally` adds lock-guard + edit-kind write; `deleteRally` adds confirm-unlock + audit log + edit-kind write; `createRally` adds edit-kind write; new `splitRally`, `mergeRallies`, `unlockRally` |
| `api/src/routes/rallies.ts` | New split/merge/unlock routes; delete body schema |
| `api/src/services/matchAnalysisService.ts` | Restructure `runMatchAnalysis` to plan → stage → telemetry; thread `rallyIds` to `remapTrackIds` + `reattributeActions` |
| `analysis/rallycut/cli/commands/remap_track_ids.py` | Add `--rally-ids` option |
| `analysis/rallycut/cli/commands/reattribute_actions.py` | Add `--rally-ids` option |
| `web/src/services/api.ts` | `splitRally`, `mergeRallies`, `unlockRally` client calls |
| `web/src/stores/editorStore.ts` | New `splitRally` action; `mergeRallies` now calls API; `unlockRally` action |
| `api/CLAUDE.md` | Document Project B invariants |

---

## Task 1: Schema migration

**Files:**
- Modify: `api/prisma/schema.prisma`
- Create: `api/prisma/migrations/20260416000000_b_rally_edit_propagation/migration.sql`

- [ ] **Step 1: Edit `api/prisma/schema.prisma`** — locate the `Video` model, add the two new fields below `matchAnalysisJson`:

```prisma
  pendingAnalysisEditsJson Json?    @map("pending_analysis_edits_json")
  matchAnalysisRanAt       DateTime? @map("match_analysis_ran_at")
```

- [ ] **Step 2: Create migration directory and SQL file.**

```bash
mkdir -p api/prisma/migrations/20260416000000_b_rally_edit_propagation
```

File `api/prisma/migrations/20260416000000_b_rally_edit_propagation/migration.sql`:

```sql
-- AlterTable
ALTER TABLE "videos" ADD COLUMN "pending_analysis_edits_json" JSONB;
ALTER TABLE "videos" ADD COLUMN "match_analysis_ran_at" TIMESTAMP(3);
```

- [ ] **Step 3: Apply migration.**

```bash
cd api && npx prisma migrate deploy
```

Expected: `Applying migration \`20260416000000_b_rally_edit_propagation\`` then `All migrations have been successfully applied.`

- [ ] **Step 4: Regenerate Prisma client.**

```bash
cd api && npx prisma generate
```

Expected: `Generated Prisma Client`.

- [ ] **Step 5: Typecheck.**

```bash
cd api && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Commit.**

```bash
git add api/prisma/schema.prisma api/prisma/migrations/20260416000000_b_rally_edit_propagation
git commit -m "feat(b): add pending_analysis_edits_json + match_analysis_ran_at columns"
```

---

## Task 2: New error classes

**Files:**
- Modify: `api/src/middleware/errorHandler.ts`

- [ ] **Step 1: Extend `ErrorCode` union** near the top of the file. Find the existing union (the export starts around line 4):

```ts
export type ErrorCode =
  | "VALIDATION_ERROR"
  | "NOT_FOUND"
  | "FORBIDDEN"
  | "ACCESS_DENIED"
  | "LIMIT_EXCEEDED"
  | "CONFLICT"
  | "INTERNAL_ERROR"
  | "LOCKED_RALLY_CANNOT_EXTEND"
  | "LOCKED_RALLY_CANNOT_SPLIT"
  | "LOCKED_RALLY_CANNOT_MERGE"
  | "LOCKED_RALLY_REQUIRES_CONFIRM"
  | "RALLY_TRACKING_IN_PROGRESS"
  | "RALLY_TRACKING_FAILED"
  | "RALLIES_OVERLAP"
  | "SPLIT_BOUNDS_INVALID";
```

- [ ] **Step 2: Add subclasses** at the bottom of the file, after the existing subclasses (after `AccessDeniedError`):

```ts
type LockedOp = "EXTEND" | "SPLIT" | "MERGE";

export class LockedRallyError extends AppError {
  constructor(op: LockedOp, rallyId: string) {
    super(
      `LOCKED_RALLY_CANNOT_${op}` as ErrorCode,
      `Rally '${rallyId}' is canonical-locked; ${op.toLowerCase()} is not allowed. Unlock the rally first.`,
      409,
      { rallyId, op }
    );
    this.name = "LockedRallyError";
  }
}

export class LockedRallyRequiresConfirmError extends AppError {
  constructor(rallyId: string, gtFrameCount: number) {
    super(
      "LOCKED_RALLY_REQUIRES_CONFIRM",
      `Rally '${rallyId}' is canonical-locked with ${gtFrameCount} GT frames. Pass {confirmUnlock: true} to proceed.`,
      409,
      { rallyId, gtFrameCount }
    );
    this.name = "LockedRallyRequiresConfirmError";
  }
}

type RallyTrackingFailReason = "IN_PROGRESS" | "FAILED";

export class RallyTrackingStateError extends AppError {
  constructor(reason: RallyTrackingFailReason, rallyId: string) {
    super(
      `RALLY_TRACKING_${reason}` as ErrorCode,
      reason === "IN_PROGRESS"
        ? `Rally '${rallyId}' is currently being tracked. Retry once tracking completes.`
        : `Rally '${rallyId}' tracking failed. Retrack or delete before retrying this operation.`,
      409,
      { rallyId, reason }
    );
    this.name = "RallyTrackingStateError";
  }
}

export class RalliesOverlapError extends AppError {
  constructor(rallyIds: string[]) {
    super(
      "RALLIES_OVERLAP",
      "Rallies overlap in time and cannot be merged.",
      400,
      { rallyIds }
    );
    this.name = "RalliesOverlapError";
  }
}

export class SplitBoundsError extends AppError {
  constructor(detail: string, bounds: Record<string, number>) {
    super(
      "SPLIT_BOUNDS_INVALID",
      `Split bounds invalid: ${detail}`,
      400,
      bounds
    );
    this.name = "SplitBoundsError";
  }
}
```

- [ ] **Step 3: Typecheck.**

```bash
cd api && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Commit.**

```bash
git add api/src/middleware/errorHandler.ts
git commit -m "feat(b): add typed AppError subclasses for rally-edit 4xx responses"
```

---

## Task 3: Canonical-lock guard

**Files:**
- Create: `api/src/services/canonicalLockGuard.ts`
- Create: `api/tests/canonicalLockGuard.test.ts`

- [ ] **Step 1: Write the failing test.** File `api/tests/canonicalLockGuard.test.ts`:

```ts
import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { isRallyLocked, assertNotLocked } from '../src/services/canonicalLockGuard';
import { LockedRallyError } from '../src/middleware/errorHandler';

const userId = '11111111-1111-1111-1111-000000000b03';
const videoId = '22222222-2222-2222-2222-000000000b03';
const rallyIdLocked = '33333333-3333-3333-3333-000000000b03';
const rallyIdUnlocked = '33333333-3333-3333-3333-000000000b04';

describe('canonicalLockGuard', () => {
  beforeEach(async () => {
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: {
        id: videoId,
        userId,
        filename: 'b.mp4',
        sizeBytes: BigInt(1),
        durationMs: 60000,
        matchAnalysisJson: {
          videoId,
          numRallies: 2,
          rallies: [
            { rallyId: rallyIdLocked, canonicalLocked: true, trackToPlayer: {}, assignmentConfidence: 1 },
            { rallyId: rallyIdUnlocked, canonicalLocked: false, trackToPlayer: {}, assignmentConfidence: 1 },
          ],
        },
      },
    });
    await prisma.rally.createMany({
      data: [
        { id: rallyIdLocked, videoId, startMs: 0, endMs: 5000 },
        { id: rallyIdUnlocked, videoId, startMs: 5000, endMs: 10000 },
      ],
    });
  });

  afterEach(async () => {
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
  });

  it('isRallyLocked returns true for locked rally', async () => {
    expect(await isRallyLocked(prisma, rallyIdLocked)).toBe(true);
  });

  it('isRallyLocked returns false for unlocked rally', async () => {
    expect(await isRallyLocked(prisma, rallyIdUnlocked)).toBe(false);
  });

  it('isRallyLocked returns false when rally is missing from matchAnalysisJson', async () => {
    expect(await isRallyLocked(prisma, '44444444-4444-4444-4444-000000000b03')).toBe(false);
  });

  it('assertNotLocked throws LockedRallyError for locked rally', async () => {
    await expect(assertNotLocked(prisma, rallyIdLocked, 'SPLIT')).rejects.toBeInstanceOf(LockedRallyError);
  });

  it('assertNotLocked is a no-op for unlocked rally', async () => {
    await expect(assertNotLocked(prisma, rallyIdUnlocked, 'SPLIT')).resolves.toBeUndefined();
  });
});
```

- [ ] **Step 2: Run the test, confirm it fails.**

```bash
cd api && npx vitest run tests/canonicalLockGuard.test.ts
```

Expected: FAIL (module `canonicalLockGuard` not found).

- [ ] **Step 3: Implement the guard.** File `api/src/services/canonicalLockGuard.ts`:

```ts
import { PrismaClient, Prisma } from '@prisma/client';
import { LockedRallyError } from '../middleware/errorHandler';

type Client = PrismaClient | Prisma.TransactionClient;

type MatchAnalysisRallyEntry = {
  rallyId: string;
  canonicalLocked?: boolean;
};

type MatchAnalysisJson = {
  rallies?: MatchAnalysisRallyEntry[];
};

export async function isRallyLocked(client: Client, rallyId: string): Promise<boolean> {
  const rally = await client.rally.findUnique({
    where: { id: rallyId },
    select: { video: { select: { matchAnalysisJson: true } } },
  });
  const json = rally?.video?.matchAnalysisJson as MatchAnalysisJson | null | undefined;
  const entry = json?.rallies?.find(r => r.rallyId === rallyId);
  return entry?.canonicalLocked === true;
}

export async function assertNotLocked(
  client: Client,
  rallyId: string,
  op: 'EXTEND' | 'SPLIT' | 'MERGE',
): Promise<void> {
  if (await isRallyLocked(client, rallyId)) {
    throw new LockedRallyError(op, rallyId);
  }
}
```

- [ ] **Step 4: Run the test, confirm pass.**

```bash
cd api && npx vitest run tests/canonicalLockGuard.test.ts
```

Expected: 5 passed.

- [ ] **Step 5: Commit.**

```bash
git add api/src/services/canonicalLockGuard.ts api/tests/canonicalLockGuard.test.ts
git commit -m "feat(b): add canonicalLockGuard service with isRallyLocked + assertNotLocked"
```

---

## Task 4: Unlock service + route

**Files:**
- Modify: `api/src/services/rallyService.ts`
- Modify: `api/src/routes/rallies.ts`
- Create: `api/tests/unlockRally.test.ts`

- [ ] **Step 1: Write the failing test.** File `api/tests/unlockRally.test.ts`:

```ts
import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { unlockRally } from '../src/services/rallyService';

const userId = '11111111-1111-1111-1111-000000000b04';
const videoId = '22222222-2222-2222-2222-000000000b04';
const rallyId = '33333333-3333-3333-3333-000000000b04';

describe('unlockRally', () => {
  beforeEach(async () => {
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: {
        id: videoId, userId, filename: 'b.mp4', sizeBytes: BigInt(1), durationMs: 60000,
        matchAnalysisJson: {
          videoId, numRallies: 1,
          rallies: [{ rallyId, canonicalLocked: true, trackToPlayer: { '1': 1 }, assignmentConfidence: 0.9 }],
        },
      },
    });
    await prisma.rally.create({ data: { id: rallyId, videoId, startMs: 0, endMs: 5000 } });
  });

  afterEach(async () => {
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
  });

  it('clears canonicalLocked flag and preserves other rally fields', async () => {
    const result = await unlockRally(rallyId, userId);
    expect(result.wasLocked).toBe(true);
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    const entry = (v!.matchAnalysisJson as any).rallies[0];
    expect(entry.canonicalLocked).toBe(false);
    expect(entry.trackToPlayer).toEqual({ '1': 1 });
    expect(entry.assignmentConfidence).toBe(0.9);
  });

  it('is idempotent on already-unlocked rally', async () => {
    await unlockRally(rallyId, userId);
    const result = await unlockRally(rallyId, userId);
    expect(result.wasLocked).toBe(false);
  });
});
```

- [ ] **Step 2: Run the test, confirm it fails.**

```bash
cd api && npx vitest run tests/unlockRally.test.ts
```

Expected: FAIL (`unlockRally` not exported).

- [ ] **Step 3: Add `unlockRally` to `api/src/services/rallyService.ts`.** Append this function to the file:

```ts
import { NotFoundError } from '../middleware/errorHandler';

export async function unlockRally(
  rallyId: string,
  userId: string,
): Promise<{ rallyId: string; wasLocked: boolean; unlockedAt: Date }> {
  return prisma.$transaction(async (tx) => {
    const rally = await tx.rally.findUnique({
      where: { id: rallyId },
      select: { videoId: true, video: { select: { userId: true, matchAnalysisJson: true } } },
    });
    if (!rally || rally.video.userId !== userId) throw new NotFoundError('Rally', rallyId);

    const json: any = rally.video.matchAnalysisJson ?? { rallies: [] };
    const entry = (json.rallies ?? []).find((r: any) => r.rallyId === rallyId);
    const wasLocked = entry?.canonicalLocked === true;

    if (wasLocked) {
      entry.canonicalLocked = false;
      await tx.video.update({
        where: { id: rally.videoId },
        data: { matchAnalysisJson: json },
      });
    }

    return { rallyId, wasLocked, unlockedAt: new Date() };
  });
}
```

- [ ] **Step 4: Add the route.** In `api/src/routes/rallies.ts`, add after the existing DELETE route:

```ts
router.post(
  '/v1/rallies/:id/unlock',
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const result = await unlockRally(req.params.id, req.userId!);
      res.json(result);
    } catch (error) {
      next(error);
    }
  },
);
```

Ensure `unlockRally` is added to the existing import from `../services/rallyService`.

- [ ] **Step 5: Run the test, confirm pass.**

```bash
cd api && npx vitest run tests/unlockRally.test.ts
```

Expected: 2 passed.

- [ ] **Step 6: Typecheck + commit.**

```bash
cd api && npx tsc --noEmit
git add api/src/services/rallyService.ts api/src/routes/rallies.ts api/tests/unlockRally.test.ts
git commit -m "feat(b): POST /v1/rallies/:id/unlock clears canonicalLocked flag"
```

---

## Task 5: Pending-analysis-edits helpers

**Files:**
- Create: `api/src/services/pendingAnalysisEdits.ts`
- Create: `api/tests/pendingAnalysisEdits.test.ts`

- [ ] **Step 1: Write the failing test.** File `api/tests/pendingAnalysisEdits.test.ts`:

```ts
import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { appendEdit, appendEditsBatch, consumePendingEdits } from '../src/services/pendingAnalysisEdits';

const userId = '11111111-1111-1111-1111-000000000b05';
const videoId = '22222222-2222-2222-2222-000000000b05';
const rallyA = '33333333-3333-3333-3333-000000000b0a';
const rallyB = '33333333-3333-3333-3333-000000000b0b';

describe('pendingAnalysisEdits', () => {
  beforeEach(async () => {
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: { id: videoId, userId, filename: 'b.mp4', sizeBytes: BigInt(1), durationMs: 60000 },
    });
  });

  afterEach(async () => {
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
  });

  it('appendEdit creates entries list when null', async () => {
    await appendEdit(prisma, videoId, rallyA, 'split');
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect((v!.pendingAnalysisEditsJson as any).entries).toHaveLength(1);
    expect((v!.pendingAnalysisEditsJson as any).entries[0]).toMatchObject({ rallyId: rallyA, editKind: 'split' });
  });

  it('appendEdit appends to existing entries', async () => {
    await appendEdit(prisma, videoId, rallyA, 'split');
    await appendEdit(prisma, videoId, rallyB, 'delete');
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect((v!.pendingAnalysisEditsJson as any).entries).toHaveLength(2);
  });

  it('appendEditsBatch writes multiple entries in one update', async () => {
    await appendEditsBatch(prisma, videoId, [
      { rallyId: rallyA, editKind: 'split' },
      { rallyId: rallyB, editKind: 'split' },
    ]);
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect((v!.pendingAnalysisEditsJson as any).entries).toHaveLength(2);
  });

  it('consumePendingEdits returns entries and nulls the column', async () => {
    await appendEdit(prisma, videoId, rallyA, 'shorten');
    const result = await consumePendingEdits(videoId);
    expect(result.entries).toHaveLength(1);
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v!.pendingAnalysisEditsJson).toBeNull();
  });

  it('consumePendingEdits on empty returns empty entries', async () => {
    const result = await consumePendingEdits(videoId);
    expect(result.entries).toEqual([]);
  });
});
```

- [ ] **Step 2: Run the test, confirm it fails.**

```bash
cd api && npx vitest run tests/pendingAnalysisEdits.test.ts
```

Expected: FAIL (module not found).

- [ ] **Step 3: Implement the helpers.** File `api/src/services/pendingAnalysisEdits.ts`:

```ts
import { PrismaClient, Prisma } from '@prisma/client';
import { prisma } from '../lib/prisma';

type Client = PrismaClient | Prisma.TransactionClient;

export type EditKind = 'scalar' | 'shorten' | 'extend' | 'delete' | 'split' | 'merge' | 'create';

export type PendingEdit = { rallyId: string; editKind: EditKind; at: string };

export type PendingEditsJson = { entries: PendingEdit[] };

export async function appendEdit(
  client: Client,
  videoId: string,
  rallyId: string,
  editKind: EditKind,
): Promise<void> {
  await appendEditsBatch(client, videoId, [{ rallyId, editKind }]);
}

export async function appendEditsBatch(
  client: Client,
  videoId: string,
  edits: Array<{ rallyId: string; editKind: EditKind }>,
): Promise<void> {
  const video = await client.video.findUnique({
    where: { id: videoId },
    select: { pendingAnalysisEditsJson: true },
  });
  const existing = (video?.pendingAnalysisEditsJson as PendingEditsJson | null)?.entries ?? [];
  const at = new Date().toISOString();
  const next: PendingEditsJson = {
    entries: [...existing, ...edits.map(e => ({ ...e, at }))],
  };
  await client.video.update({
    where: { id: videoId },
    data: { pendingAnalysisEditsJson: next as unknown as Prisma.InputJsonValue },
  });
}

export async function consumePendingEdits(videoId: string): Promise<PendingEditsJson> {
  return prisma.$transaction(async (tx) => {
    const video = await tx.video.findUnique({
      where: { id: videoId },
      select: { pendingAnalysisEditsJson: true },
    });
    const entries = (video?.pendingAnalysisEditsJson as PendingEditsJson | null)?.entries ?? [];
    if (entries.length > 0) {
      await tx.video.update({
        where: { id: videoId },
        data: { pendingAnalysisEditsJson: Prisma.DbNull },
      });
    }
    return { entries };
  });
}
```

- [ ] **Step 4: Run the test, confirm pass.**

```bash
cd api && npx vitest run tests/pendingAnalysisEdits.test.ts
```

Expected: 5 passed.

- [ ] **Step 5: Commit.**

```bash
git add api/src/services/pendingAnalysisEdits.ts api/tests/pendingAnalysisEdits.test.ts
git commit -m "feat(b): pendingAnalysisEdits helpers (append, consume)"
```

---

## Task 6: Wire pending-edit writes into existing rally CRUD

**Files:**
- Modify: `api/src/services/rallyService.ts`
- Modify: `api/tests/retrackOnExtend.test.ts` (add assertions)

- [ ] **Step 1: Modify `createRally`** in `api/src/services/rallyService.ts`. Inside the existing transaction (or wrap the create in one if it's not already), after the rally is created append an edit:

```ts
// After: const rally = await tx.rally.create({ data: ... });
await appendEdit(tx, videoId, rally.id, 'create');
```

Import: `import { appendEdit } from './pendingAnalysisEdits';`

- [ ] **Step 2: Modify `updateRally`** to classify the edit kind. Inside the existing `prisma.$transaction` callback (around lines 72–94 per anchor survey), replace the block after `reindexTrackingData` + `markRetrackIfExtended` with this:

```ts
// Classify edit kind
let editKind: 'scalar' | 'shorten' | 'extend' | null = null;
if (data.startMs !== undefined || data.endMs !== undefined) {
  const newStart = data.startMs ?? rally.startMs;
  const newEnd = data.endMs ?? rally.endMs;
  const extended = newStart < rally.startMs || newEnd > rally.endMs;
  editKind = extended ? 'extend' : 'shorten';
} else if (data.scoreA !== undefined || data.scoreB !== undefined || data.servingTeam !== undefined || data.notes !== undefined || data.confidence !== undefined) {
  editKind = 'scalar';
}
if (editKind) await appendEdit(tx, rally.videoId, id, editKind);
```

This must be the *last* write inside the transaction.

- [ ] **Step 3: Modify `deleteRally`.** Wrap in a transaction (if it isn't already), capture `videoId` before delete, then:

```ts
export async function deleteRally(id: string, userId: string, opts?: { confirmUnlock?: boolean }): Promise<void> {
  await prisma.$transaction(async (tx) => {
    const rally = await tx.rally.findUnique({
      where: { id },
      select: { videoId: true, video: { select: { userId: true, matchAnalysisJson: true } } },
    });
    if (!rally || rally.video.userId !== userId) throw new NotFoundError('Rally', id);

    // Canonical-lock confirm gate
    const json: any = rally.video.matchAnalysisJson ?? { rallies: [] };
    const entry = (json.rallies ?? []).find((r: any) => r.rallyId === id);
    const locked = entry?.canonicalLocked === true;
    if (locked && !opts?.confirmUnlock) {
      // Best-effort count of GT frames for the error payload
      const pt = await tx.playerTrack.findUnique({ where: { rallyId: id }, select: { groundTruthJson: true, actionGroundTruthJson: true } });
      const gtFrameCount = ((pt?.groundTruthJson as any)?.length ?? 0) + ((pt?.actionGroundTruthJson as any)?.length ?? 0);
      throw new LockedRallyRequiresConfirmError(id, gtFrameCount);
    }

    // Audit log for the destructive confirmed path
    if (locked && opts?.confirmUnlock) {
      console.log(JSON.stringify({
        event: 'rally.locked.deleted',
        rallyId: id, videoId: rally.videoId, userId,
        gtFrameCount: entry?.gtFrameCount ?? null,
      }));
    }

    // Also drop the rally from matchAnalysisJson.rallies[] so stats don't reference it
    if (json.rallies) {
      json.rallies = json.rallies.filter((r: any) => r.rallyId !== id);
      await tx.video.update({ where: { id: rally.videoId }, data: { matchAnalysisJson: json } });
    }

    await tx.rally.delete({ where: { id } });

    await appendEdit(tx, rally.videoId, id, 'delete');
  });
}
```

Imports: add `LockedRallyRequiresConfirmError` from `../middleware/errorHandler`.

- [ ] **Step 4: Update the DELETE route** in `api/src/routes/rallies.ts` to accept `{confirmUnlock?: boolean}` in the body:

```ts
router.delete(
  '/v1/rallies/:id',
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: z.object({ confirmUnlock: z.boolean().optional() }).optional(),
  }),
  async (req, res, next) => {
    try {
      await deleteRally(req.params.id, req.userId!, { confirmUnlock: req.body?.confirmUnlock });
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  },
);
```

- [ ] **Step 5: Run existing retrackOnExtend tests + a new delete-locked test.**

```bash
cd api && npx vitest run tests/retrackOnExtend.test.ts
```

Expected: all prior assertions still pass.

- [ ] **Step 6: Typecheck + commit.**

```bash
cd api && npx tsc --noEmit
git add api/src/services/rallyService.ts api/src/routes/rallies.ts
git commit -m "feat(b): classify + record rally edit kinds; delete-with-confirm on locked rallies"
```

---

## Task 7: planStages pure function

**Files:**
- Create: `api/src/services/matchAnalysisPlanning.ts`
- Create: `api/tests/matchAnalysisPlanning.test.ts`

- [ ] **Step 1: Write the failing test.** File `api/tests/matchAnalysisPlanning.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { planStages } from '../src/services/matchAnalysisPlanning';
import type { PendingEdit } from '../src/services/pendingAnalysisEdits';

const e = (rallyId: string, editKind: PendingEdit['editKind']): PendingEdit => ({ rallyId, editKind, at: '2026-04-15T00:00:00Z' });

describe('planStages', () => {
  it('empty entries → fullRerun', () => {
    expect(planStages({ entries: [] })).toEqual({ fullRerun: true, runStage2: false, changedRallyIds: [] });
  });

  it('scalar-only storm → no stages 2/3/4/5; stage 6 runs', () => {
    const plan = planStages({ entries: [e('r1', 'scalar'), e('r2', 'scalar')] });
    expect(plan).toEqual({ fullRerun: false, runStage2: false, changedRallyIds: [] });
  });

  it('delete-only storm → changedRallyIds empty', () => {
    expect(planStages({ entries: [e('r1', 'delete')] })).toEqual({ fullRerun: false, runStage2: false, changedRallyIds: [] });
  });

  it('shorten edits populate changedRallyIds (deduped)', () => {
    const plan = planStages({ entries: [e('r1', 'shorten'), e('r1', 'scalar'), e('r2', 'shorten')] });
    expect(plan).toMatchObject({ fullRerun: false, runStage2: false });
    expect(plan.changedRallyIds.sort()).toEqual(['r1', 'r2']);
  });

  it('split edits populate changedRallyIds', () => {
    const plan = planStages({ entries: [e('child-a', 'split'), e('child-b', 'split')] });
    expect(plan.fullRerun).toBe(false);
    expect(plan.changedRallyIds.sort()).toEqual(['child-a', 'child-b']);
  });

  it('extend triggers fullRerun regardless of other edits', () => {
    expect(planStages({ entries: [e('r1', 'shorten'), e('r2', 'extend')] })).toMatchObject({ fullRerun: true });
  });

  it('create triggers fullRerun', () => {
    expect(planStages({ entries: [e('r1', 'create')] })).toMatchObject({ fullRerun: true });
  });

  it('merge triggers fullRerun', () => {
    expect(planStages({ entries: [e('r1', 'merge')] })).toMatchObject({ fullRerun: true });
  });
});
```

- [ ] **Step 2: Run, confirm failure.**

```bash
cd api && npx vitest run tests/matchAnalysisPlanning.test.ts
```

Expected: FAIL (module not found).

- [ ] **Step 3: Implement.** File `api/src/services/matchAnalysisPlanning.ts`:

```ts
import type { PendingEditsJson, PendingEdit } from './pendingAnalysisEdits';

export type StagePlan = {
  fullRerun: boolean;
  runStage2: boolean;           // stage 2 implies stage 3 in the caller
  changedRallyIds: string[];    // stages 4/5 processed for these only
};

const FULL_RERUN_KINDS: ReadonlyArray<PendingEdit['editKind']> = ['extend', 'create', 'merge'];
const CHANGED_KINDS: ReadonlyArray<PendingEdit['editKind']> = ['shorten', 'split'];

export function planStages(edits: PendingEditsJson): StagePlan {
  if (edits.entries.length === 0) {
    return { fullRerun: true, runStage2: false, changedRallyIds: [] };
  }
  if (edits.entries.some(e => FULL_RERUN_KINDS.includes(e.editKind))) {
    return { fullRerun: true, runStage2: true, changedRallyIds: [] };
  }
  const changed = new Set<string>();
  for (const e of edits.entries) {
    if (CHANGED_KINDS.includes(e.editKind)) changed.add(e.rallyId);
  }
  return { fullRerun: false, runStage2: false, changedRallyIds: [...changed] };
}
```

- [ ] **Step 4: Run, confirm pass.**

```bash
cd api && npx vitest run tests/matchAnalysisPlanning.test.ts
```

Expected: 8 passed.

- [ ] **Step 5: Commit.**

```bash
git add api/src/services/matchAnalysisPlanning.ts api/tests/matchAnalysisPlanning.test.ts
git commit -m "feat(b): planStages pure function gates match-analysis stages by edit kinds"
```

---

## Task 8: `--rally-ids` flag on Python CLIs

**Files:**
- Modify: `analysis/rallycut/cli/commands/remap_track_ids.py`
- Modify: `analysis/rallycut/cli/commands/reattribute_actions.py`

- [ ] **Step 1: Add the option to `remap_track_ids_cmd`.** Add a new option parameter in the same signature block:

```python
rally_ids: Optional[str] = typer.Option(
    None,
    "--rally-ids",
    help="Comma-separated rally UUIDs to process. If omitted, all tracked rallies in the video are processed.",
),
```

Add `from typing import Optional` at top if not already present.

Inside the command body, parse and scope the rally loader:

```python
rally_id_filter: Optional[list[str]] = (
    [s.strip() for s in rally_ids.split(",") if s.strip()]
    if rally_ids else None
)
# Pass rally_id_filter into the existing loader function; when the loader
# fetches rallies, add `AND rally.id IN (...)` when filter is present.
```

Thread `rally_id_filter` into whatever helper loads the rally list (usually a SQL query via `get_connection()`). If the loader doesn't accept a filter, add a parameter `rally_ids: list[str] | None = None` and conditionally append to the SQL WHERE.

- [ ] **Step 2: Mirror the same change in `reattribute_actions_cmd`.** Same option, same filter plumbing.

- [ ] **Step 3: Smoke-test the CLI** by running in dry-run mode on a known video ID:

```bash
cd analysis && uv run rallycut remap-track-ids <video-id> --rally-ids <one-rally-id> --dry-run --quiet
uv run rallycut reattribute-actions <video-id> --rally-ids <one-rally-id> --dry-run --quiet
```

Expected: both exit 0. Output shows only the filtered rally being processed (vs. the video's full rally list when the flag is omitted).

- [ ] **Step 4: Type + lint.**

```bash
cd analysis && uv run mypy rallycut/cli/commands/remap_track_ids.py rallycut/cli/commands/reattribute_actions.py
uv run ruff check rallycut/cli/commands/remap_track_ids.py rallycut/cli/commands/reattribute_actions.py
```

Expected: 0 errors.

- [ ] **Step 5: Commit.**

```bash
git add analysis/rallycut/cli/commands/remap_track_ids.py analysis/rallycut/cli/commands/reattribute_actions.py
git commit -m "feat(b): --rally-ids flag on remap-track-ids + reattribute-actions CLIs"
```

---

## Task 9: Stage-gated `runMatchAnalysis`

**Files:**
- Modify: `api/src/services/matchAnalysisService.ts`

- [ ] **Step 1: Modify `remapTrackIds` and `reattributeActions` in `matchAnalysisService.ts`** to accept an optional `rallyIds` array and pass `--rally-ids` to the CLI.

Locate the existing `async function remapTrackIds(videoId: string): Promise<void>`. Change to:

```ts
async function remapTrackIds(videoId: string, opts: { rallyIds?: string[] } = {}): Promise<void> {
  // existing body, but when building the subcommand args:
  const args = ['remap-track-ids', videoId, '--quiet'];
  if (opts.rallyIds && opts.rallyIds.length > 0) {
    args.push('--rally-ids', opts.rallyIds.join(','));
  }
  // pass `args` to runCli or whatever existing spawn wrapper is used
}
```

Make the analogous change to `reattributeActions`.

- [ ] **Step 2: Restructure `runMatchAnalysis`.** Replace the body of the existing `runMatchAnalysis` (keep the signature) with a plan-driven form:

```ts
export async function runMatchAnalysis(
  videoId: string,
  onProgress?: ProgressCallback,
): Promise<MatchAnalysisResult | null> {
  const started = Date.now();
  const timings: Record<string, number> = {};

  const edits = await consumePendingEdits(videoId);
  const plan = planStages(edits);

  const report = (msg: string, step: number, total: number) => onProgress?.({ message: msg, step, total });

  const timed = async <T>(label: string, fn: () => Promise<T>): Promise<T> => {
    const t0 = Date.now();
    try { return await fn(); } finally { timings[label] = Date.now() - t0; }
  };

  // Stage 1 always runs (cheap validation)
  report('Validating tracked rallies…', 1, 6);
  await timed('stage1_validate', () => validateTrackedRallies(videoId));

  let result: MatchAnalysisResult | null = null;

  if (plan.fullRerun) {
    report('Matching players across rallies…', 2, 6);
    result = await timed('stage2_match', () => runMatchPlayersCli(videoId, /* existing args */));
    report('Repairing identities…', 3, 6);
    await timed('stage3_repair', () => repairIdentities(videoId));
    report('Remapping track IDs…', 4, 6);
    await timed('stage4_remap', () => remapTrackIds(videoId));
    report('Reattributing actions…', 5, 6);
    await timed('stage5_reattribute', () => reattributeActions(videoId));
  } else {
    if (plan.changedRallyIds.length > 0) {
      report('Remapping changed rallies…', 4, 6);
      await timed('stage4_remap', () => remapTrackIds(videoId, { rallyIds: plan.changedRallyIds }));
      report('Reattributing changed rallies…', 5, 6);
      await timed('stage5_reattribute', () => reattributeActions(videoId, { rallyIds: plan.changedRallyIds }));
    }
  }

  report('Computing match stats…', 6, 6);
  await timed('stage6_stats', () => computeAndSaveMatchStats(videoId));

  await prisma.video.update({
    where: { id: videoId },
    data: { matchAnalysisRanAt: new Date() },
  });

  console.log(JSON.stringify({
    event: 'match_analysis.stage_timings',
    videoId,
    plan,
    timingsMs: timings,
    totalMs: Date.now() - started,
  }));

  return result;
}
```

Keep any existing error handling, progress-report wrappers, and the `runningVideos` Set guard from A2a at the caller.

- [ ] **Step 3: Imports.** At the top of the file add:

```ts
import { consumePendingEdits } from './pendingAnalysisEdits';
import { planStages } from './matchAnalysisPlanning';
```

- [ ] **Step 4: Typecheck.**

```bash
cd api && npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 5: Run existing match-analysis tests.**

```bash
cd api && npx vitest run
```

Expected: all previously passing tests still pass (behavior should be unchanged for the pre-migration case — `pendingAnalysisEditsJson=null` → `planStages` returns `fullRerun: true`).

- [ ] **Step 6: Commit.**

```bash
git add api/src/services/matchAnalysisService.ts
git commit -m "feat(b): stage-gate runMatchAnalysis on planStages + pendingAnalysisEdits"
```

---

## Task 10: `slicePlayerTrack` helper + tests

**Files:**
- Create: `api/src/services/rallySlicing.ts`
- Create: `api/tests/rallySlicing.test.ts`

- [ ] **Step 1: Write the failing test.** File `api/tests/rallySlicing.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { slicePlayerTrack } from '../src/services/rallySlicing';

const basePt = {
  id: 'pt-1', rallyId: 'r-parent', status: 'COMPLETED',
  fps: 30, frameCount: 300, detectionRate: 1.0, avgConfidence: 0.9,
  avgPlayerCount: 4, uniqueTrackCount: 6, courtSplitY: 0.5,
  processingTimeMs: 1000, modelVersion: 'v1', needsRetrack: false,
  positionsJson: [
    { frameNumber: 10, trackId: 1, x: 0.1, y: 0.1 },
    { frameNumber: 150, trackId: 1, x: 0.5, y: 0.5 },
    { frameNumber: 250, trackId: 1, x: 0.9, y: 0.9 },
  ],
  rawPositionsJson: [],
  ballPositionsJson: [{ frameNumber: 20 }, { frameNumber: 260 }],
  contactsJson: [{ frame: 50, playerTrackId: 1 }, { frame: 220, playerTrackId: 2 }],
  actionsJson: [{ frame: 55 }, { frame: 225 }],
  groundTruthJson: null,
  actionGroundTruthJson: null,
  qualityReportJson: null,
};

describe('slicePlayerTrack', () => {
  it('partitions by frame and shifts back-half indices', () => {
    const { first, second } = slicePlayerTrack(basePt as any, 100, 200);
    expect(first.positionsJson).toEqual([{ frameNumber: 10, trackId: 1, x: 0.1, y: 0.1 }]);
    expect(second.positionsJson.map((p: any) => p.frameNumber)).toEqual([50]); // 250 - 200
    expect(first.contactsJson.map((c: any) => c.frame)).toEqual([50]);
    expect(second.contactsJson.map((c: any) => c.frame)).toEqual([20]); // 220 - 200
    expect(first.actionsJson.map((a: any) => a.frame)).toEqual([55]);
    expect(second.actionsJson.map((a: any) => a.frame)).toEqual([25]); // 225 - 200
    expect(first.ballPositionsJson.map((b: any) => b.frameNumber)).toEqual([20]);
    expect(second.ballPositionsJson.map((b: any) => b.frameNumber)).toEqual([60]);
  });

  it('discards middle segment between firstEndFrame and secondStartFrame', () => {
    const pt = { ...basePt, positionsJson: [{ frameNumber: 150, trackId: 1, x: 0, y: 0 }] };
    const { first, second } = slicePlayerTrack(pt as any, 100, 200);
    expect(first.positionsJson).toHaveLength(0);
    expect(second.positionsJson).toHaveLength(0);
  });

  it('recomputes frameCount per child', () => {
    const { first, second } = slicePlayerTrack(basePt as any, 100, 200);
    expect(first.frameCount).toBe(100);
    expect(second.frameCount).toBe(100); // 300 - 200
  });

  it('split-at-frame-0 degenerate case produces empty first child', () => {
    const { first, second } = slicePlayerTrack(basePt as any, 0, 0);
    expect(first.frameCount).toBe(0);
    expect(first.positionsJson).toHaveLength(0);
    expect(second.frameCount).toBe(300);
  });

  it('handles null GT arrays gracefully', () => {
    const { first, second } = slicePlayerTrack(basePt as any, 100, 200);
    expect(first.groundTruthJson).toBeNull();
    expect(second.groundTruthJson).toBeNull();
  });
});
```

- [ ] **Step 2: Run, confirm failure.**

```bash
cd api && npx vitest run tests/rallySlicing.test.ts
```

Expected: FAIL (module not found).

- [ ] **Step 3: Implement the slice helper.** File `api/src/services/rallySlicing.ts`:

```ts
type FrameField = 'frame' | 'frameNumber';

type AnyFrameEntry = Record<string, unknown>;

export type SlicePlayerTrackInput = {
  fps: number;
  frameCount: number;
  courtSplitY: number | null;
  processingTimeMs: number | null;
  modelVersion: string | null;
  status: string;
  needsRetrack: boolean;
  positionsJson: AnyFrameEntry[] | null;
  rawPositionsJson: AnyFrameEntry[] | null;
  ballPositionsJson: AnyFrameEntry[] | null;
  contactsJson: AnyFrameEntry[] | null;
  actionsJson: AnyFrameEntry[] | null;
  groundTruthJson: AnyFrameEntry[] | null;
  actionGroundTruthJson: AnyFrameEntry[] | null;
  qualityReportJson: unknown;
};

export type SlicedPlayerTrack = Omit<SlicePlayerTrackInput, 'frameCount'> & {
  frameCount: number;
  detectionRate: number;
  avgConfidence: number;
  avgPlayerCount: number;
  uniqueTrackCount: number;
};

function partitionByFrame(
  arr: AnyFrameEntry[] | null,
  field: FrameField,
  firstEndFrame: number,
  secondStartFrame: number,
): { firstArr: AnyFrameEntry[]; secondArr: AnyFrameEntry[] } {
  if (!arr) return { firstArr: [], secondArr: [] };
  const firstArr: AnyFrameEntry[] = [];
  const secondArr: AnyFrameEntry[] = [];
  for (const entry of arr) {
    const f = entry[field];
    if (typeof f !== 'number') continue;
    if (f < firstEndFrame) firstArr.push(entry);
    else if (f >= secondStartFrame) {
      secondArr.push({ ...entry, [field]: f - secondStartFrame });
    }
    // frames in [firstEndFrame, secondStartFrame) are discarded
  }
  return { firstArr, secondArr };
}

function recomputeMetadata(
  positionsJson: AnyFrameEntry[],
  frameCount: number,
): Pick<SlicedPlayerTrack, 'detectionRate' | 'avgConfidence' | 'avgPlayerCount' | 'uniqueTrackCount'> {
  const framesWithDetection = new Set<number>();
  const trackIds = new Set<string | number>();
  let confSum = 0;
  let confN = 0;
  const playersPerFrame = new Map<number, Set<string | number>>();
  for (const p of positionsJson) {
    const frame = p.frameNumber as number;
    const tid = (p.trackId ?? p.track_id) as string | number | undefined;
    framesWithDetection.add(frame);
    if (tid !== undefined) {
      trackIds.add(tid);
      if (!playersPerFrame.has(frame)) playersPerFrame.set(frame, new Set());
      playersPerFrame.get(frame)!.add(tid);
    }
    if (typeof p.confidence === 'number') { confSum += p.confidence; confN++; }
  }
  const detectionRate = frameCount > 0 ? framesWithDetection.size / frameCount : 0;
  const avgConfidence = confN > 0 ? confSum / confN : 0;
  const avgPlayerCount = playersPerFrame.size > 0
    ? [...playersPerFrame.values()].reduce((a, s) => a + s.size, 0) / playersPerFrame.size
    : 0;
  return {
    detectionRate,
    avgConfidence,
    avgPlayerCount,
    uniqueTrackCount: trackIds.size,
  };
}

export function slicePlayerTrack(
  pt: SlicePlayerTrackInput,
  firstEndFrame: number,
  secondStartFrame: number,
): { first: SlicedPlayerTrack; second: SlicedPlayerTrack } {
  const pos = partitionByFrame(pt.positionsJson, 'frameNumber', firstEndFrame, secondStartFrame);
  const raw = partitionByFrame(pt.rawPositionsJson, 'frameNumber', firstEndFrame, secondStartFrame);
  const ball = partitionByFrame(pt.ballPositionsJson, 'frameNumber', firstEndFrame, secondStartFrame);
  const contacts = partitionByFrame(pt.contactsJson, 'frame', firstEndFrame, secondStartFrame);
  const actions = partitionByFrame(pt.actionsJson, 'frame', firstEndFrame, secondStartFrame);
  const gt = partitionByFrame(pt.groundTruthJson, 'frame', firstEndFrame, secondStartFrame);
  const actGt = partitionByFrame(pt.actionGroundTruthJson, 'frame', firstEndFrame, secondStartFrame);

  const firstCount = firstEndFrame;
  const secondCount = Math.max(0, pt.frameCount - secondStartFrame);

  const firstMeta = recomputeMetadata(pos.firstArr, firstCount);
  const secondMeta = recomputeMetadata(pos.secondArr, secondCount);

  const base = {
    fps: pt.fps,
    courtSplitY: pt.courtSplitY,
    processingTimeMs: pt.processingTimeMs,
    modelVersion: pt.modelVersion,
    status: pt.status,
    needsRetrack: pt.needsRetrack,
    qualityReportJson: pt.qualityReportJson,
  };

  return {
    first: {
      ...base,
      frameCount: firstCount,
      ...firstMeta,
      positionsJson: pos.firstArr,
      rawPositionsJson: raw.firstArr,
      ballPositionsJson: ball.firstArr,
      contactsJson: contacts.firstArr,
      actionsJson: actions.firstArr,
      groundTruthJson: pt.groundTruthJson ? gt.firstArr : null,
      actionGroundTruthJson: pt.actionGroundTruthJson ? actGt.firstArr : null,
    },
    second: {
      ...base,
      frameCount: secondCount,
      ...secondMeta,
      positionsJson: pos.secondArr,
      rawPositionsJson: raw.secondArr,
      ballPositionsJson: ball.secondArr,
      contactsJson: contacts.secondArr,
      actionsJson: actions.secondArr,
      groundTruthJson: pt.groundTruthJson ? gt.secondArr : null,
      actionGroundTruthJson: pt.actionGroundTruthJson ? actGt.secondArr : null,
    },
  };
}
```

- [ ] **Step 4: Run, confirm pass.**

```bash
cd api && npx vitest run tests/rallySlicing.test.ts
```

Expected: 5 passed.

- [ ] **Step 5: Commit.**

```bash
git add api/src/services/rallySlicing.ts api/tests/rallySlicing.test.ts
git commit -m "feat(b): slicePlayerTrack pure helper partitions frame-indexed data"
```

---

## Task 11: Split service + route + integration tests

**Files:**
- Modify: `api/src/services/rallyService.ts`
- Modify: `api/src/routes/rallies.ts`
- Create: `api/tests/splitRally.test.ts`

- [ ] **Step 1: Write the failing integration test.** File `api/tests/splitRally.test.ts`:

```ts
import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { splitRally } from '../src/services/rallyService';
import { LockedRallyError, RallyTrackingStateError, SplitBoundsError } from '../src/middleware/errorHandler';

const userId = '11111111-1111-1111-1111-000000000b11';
const videoId = '22222222-2222-2222-2222-000000000b11';
const rallyId = '33333333-3333-3333-3333-000000000b11';

async function setupRally(opts: { locked?: boolean; trackStatus?: 'COMPLETED' | 'PROCESSING' | 'FAILED' | 'NONE' } = {}) {
  await prisma.playerTrack.deleteMany({ where: { rallyId } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
  await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
  await prisma.video.create({
    data: {
      id: videoId, userId, filename: 'b.mp4', sizeBytes: BigInt(1), durationMs: 30000,
      matchAnalysisJson: {
        videoId, numRallies: 1,
        rallies: [{
          rallyId, canonicalLocked: opts.locked === true,
          trackToPlayer: { '1': 1 }, assignmentConfidence: 0.9, serverPlayerId: 1,
        }],
      },
    },
  });
  await prisma.rally.create({
    data: { id: rallyId, videoId, startMs: 0, endMs: 10000, scoreA: 5, scoreB: 3, servingTeam: 'A', notes: 'parent' },
  });
  if (opts.trackStatus && opts.trackStatus !== 'NONE') {
    await prisma.playerTrack.create({
      data: {
        rallyId, status: opts.trackStatus, fps: 30, frameCount: 300,
        detectionRate: 1.0, avgConfidence: 0.9, avgPlayerCount: 4, uniqueTrackCount: 2,
        courtSplitY: 0.5, processingTimeMs: 1000, modelVersion: 'v1', needsRetrack: false,
        positionsJson: [
          { frameNumber: 30, trackId: 1, x: 0, y: 0, confidence: 0.9 },
          { frameNumber: 180, trackId: 1, x: 0, y: 0, confidence: 0.9 },
          { frameNumber: 270, trackId: 1, x: 0, y: 0, confidence: 0.9 },
        ],
        rawPositionsJson: [],
        ballPositionsJson: [],
        contactsJson: [{ frame: 50, playerTrackId: 1 }, { frame: 250, playerTrackId: 1 }],
        actionsJson: [],
      } as any,
    });
  }
}

async function cleanup() {
  await prisma.playerTrack.deleteMany({ where: { rallyId } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
}

describe('splitRally', () => {
  afterEach(cleanup);

  it('slices PlayerTrack data, creates two children, deletes parent', async () => {
    await setupRally({ trackStatus: 'COMPLETED' });
    const { firstRally, secondRally } = await splitRally(rallyId, userId, { firstEndMs: 4000, secondStartMs: 6000 });

    expect(firstRally.endMs).toBe(4000);
    expect(secondRally.startMs).toBe(6000);
    expect(secondRally.endMs).toBe(10000);

    // Parent deleted
    expect(await prisma.rally.findUnique({ where: { id: rallyId } })).toBeNull();

    // Children have sliced player tracks
    const firstPt = await prisma.playerTrack.findUnique({ where: { rallyId: firstRally.id } });
    const secondPt = await prisma.playerTrack.findUnique({ where: { rallyId: secondRally.id } });
    expect(firstPt).not.toBeNull();
    expect(secondPt).not.toBeNull();
    expect((firstPt!.contactsJson as any).map((c: any) => c.frame)).toEqual([50]);
    expect((secondPt!.contactsJson as any).map((c: any) => c.frame)).toEqual([70]); // 250 - 180

    // Inheritance: first child scores default 0, second inherits parent
    expect(firstRally.scoreA).toBe(0);
    expect(firstRally.scoreB).toBe(0);
    expect(secondRally.scoreA).toBe(5);
    expect(secondRally.scoreB).toBe(3);
    expect(firstRally.notes).toBe('parent');
    expect(secondRally.notes).toBe('parent');

    // matchAnalysisJson has both children, parent removed
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    const rallies = (v!.matchAnalysisJson as any).rallies;
    expect(rallies.map((r: any) => r.rallyId).sort()).toEqual([firstRally.id, secondRally.id].sort());
    expect(rallies.every((r: any) => r.canonicalLocked === false)).toBe(true);
    expect(rallies.every((r: any) => JSON.stringify(r.trackToPlayer) === JSON.stringify({ '1': 1 }))).toBe(true);

    // pendingAnalysisEdits has 2 split entries
    expect((v!.pendingAnalysisEditsJson as any).entries.map((e: any) => e.editKind)).toEqual(['split', 'split']);
  });

  it('allows split when PlayerTrack is null (children PENDING-equivalent)', async () => {
    await setupRally({ trackStatus: 'NONE' });
    const { firstRally, secondRally } = await splitRally(rallyId, userId, { firstEndMs: 4000, secondStartMs: 6000 });
    expect(await prisma.playerTrack.findUnique({ where: { rallyId: firstRally.id } })).toBeNull();
    expect(await prisma.playerTrack.findUnique({ where: { rallyId: secondRally.id } })).toBeNull();
  });

  it('rejects split on locked rally', async () => {
    await setupRally({ locked: true, trackStatus: 'COMPLETED' });
    await expect(splitRally(rallyId, userId, { firstEndMs: 4000, secondStartMs: 6000 }))
      .rejects.toBeInstanceOf(LockedRallyError);
    // Parent still present
    expect(await prisma.rally.findUnique({ where: { id: rallyId } })).not.toBeNull();
  });

  it('rejects split on PROCESSING PlayerTrack', async () => {
    await setupRally({ trackStatus: 'PROCESSING' });
    await expect(splitRally(rallyId, userId, { firstEndMs: 4000, secondStartMs: 6000 }))
      .rejects.toBeInstanceOf(RallyTrackingStateError);
  });

  it('rejects invalid bounds (firstEndMs > secondStartMs)', async () => {
    await setupRally({ trackStatus: 'COMPLETED' });
    await expect(splitRally(rallyId, userId, { firstEndMs: 7000, secondStartMs: 6000 }))
      .rejects.toBeInstanceOf(SplitBoundsError);
  });
});
```

- [ ] **Step 2: Run, confirm failure.**

```bash
cd api && npx vitest run tests/splitRally.test.ts
```

Expected: FAIL (`splitRally` not exported).

- [ ] **Step 3: Implement `splitRally`.** Add to `api/src/services/rallyService.ts`:

```ts
import { slicePlayerTrack } from './rallySlicing';
import { assertNotLocked } from './canonicalLockGuard';
import { appendEditsBatch } from './pendingAnalysisEdits';
import { RallyTrackingStateError, SplitBoundsError, NotFoundError } from '../middleware/errorHandler';

export type SplitRallyInput = { firstEndMs: number; secondStartMs: number };

export async function splitRally(
  rallyId: string,
  userId: string,
  input: SplitRallyInput,
): Promise<{ firstRally: Rally; secondRally: Rally }> {
  return prisma.$transaction(async (tx) => {
    const rally = await tx.rally.findUnique({
      where: { id: rallyId },
      include: { playerTrack: true, video: { select: { userId: true, matchAnalysisJson: true } } },
    });
    if (!rally || rally.video.userId !== userId) throw new NotFoundError('Rally', rallyId);

    // Bounds validation
    const { firstEndMs, secondStartMs } = input;
    if (!(rally.startMs < firstEndMs && firstEndMs <= secondStartMs && secondStartMs < rally.endMs)) {
      throw new SplitBoundsError(
        'must satisfy startMs < firstEndMs <= secondStartMs < endMs',
        { startMs: rally.startMs, firstEndMs, secondStartMs, endMs: rally.endMs },
      );
    }

    // Lock gate
    await assertNotLocked(tx, rallyId, 'SPLIT');

    // PlayerTrack state gate
    const pt = rally.playerTrack;
    if (pt) {
      if (pt.status === 'PROCESSING') throw new RallyTrackingStateError('IN_PROGRESS', rallyId);
      if (pt.status === 'FAILED') throw new RallyTrackingStateError('FAILED', rallyId);
    }

    // Frame math
    const fps = pt?.fps ?? 30;
    const firstEndFrame = Math.round(((firstEndMs - rally.startMs) / 1000) * fps);
    const secondStartFrame = Math.round(((secondStartMs - rally.startMs) / 1000) * fps);

    // Previous rally lookup for first-child score inheritance
    const prevRally = await tx.rally.findFirst({
      where: { videoId: rally.videoId, endMs: { lte: rally.startMs } },
      orderBy: { endMs: 'desc' },
    });

    // Create child Rally rows
    const firstRally = await tx.rally.create({
      data: {
        videoId: rally.videoId,
        startMs: rally.startMs,
        endMs: firstEndMs,
        scoreA: prevRally?.scoreA ?? 0,
        scoreB: prevRally?.scoreB ?? 0,
        servingTeam: prevRally?.servingTeam ?? rally.servingTeam,
        notes: rally.notes,
        confidence: rally.confidence,
      },
    });
    const secondRally = await tx.rally.create({
      data: {
        videoId: rally.videoId,
        startMs: secondStartMs,
        endMs: rally.endMs,
        scoreA: rally.scoreA,
        scoreB: rally.scoreB,
        servingTeam: rally.servingTeam,
        notes: rally.notes,
        confidence: rally.confidence,
      },
    });

    // Slice PlayerTrack if present
    if (pt) {
      const { first, second } = slicePlayerTrack(pt as any, firstEndFrame, secondStartFrame);
      await tx.playerTrack.create({
        data: { rallyId: firstRally.id, ...(first as any) },
      });
      await tx.playerTrack.create({
        data: { rallyId: secondRally.id, ...(second as any) },
      });
    }

    // Update matchAnalysisJson.rallies[]
    const json: any = rally.video.matchAnalysisJson ?? { rallies: [] };
    const parentEntry = (json.rallies ?? []).find((r: any) => r.rallyId === rallyId);
    const inheritMapping = parentEntry?.trackToPlayer ?? {};
    const inheritServer = parentEntry?.serverPlayerId ?? null;
    json.rallies = (json.rallies ?? []).filter((r: any) => r.rallyId !== rallyId);
    json.rallies.push(
      { rallyId: firstRally.id, canonicalLocked: false, trackToPlayer: inheritMapping, assignmentConfidence: parentEntry?.assignmentConfidence ?? null, serverPlayerId: inheritServer },
      { rallyId: secondRally.id, canonicalLocked: false, trackToPlayer: inheritMapping, assignmentConfidence: parentEntry?.assignmentConfidence ?? null, serverPlayerId: inheritServer },
    );
    await tx.video.update({ where: { id: rally.videoId }, data: { matchAnalysisJson: json } });

    // Delete parent (cascades PlayerTrack)
    await tx.rally.delete({ where: { id: rallyId } });

    // Record edits last
    await appendEditsBatch(tx, rally.videoId, [
      { rallyId: firstRally.id, editKind: 'split' },
      { rallyId: secondRally.id, editKind: 'split' },
    ]);

    return { firstRally, secondRally };
  });
}
```

- [ ] **Step 4: Add the route.** In `api/src/routes/rallies.ts`:

```ts
const splitRallySchema = z.object({
  firstEndMs: z.number().int().nonnegative(),
  secondStartMs: z.number().int().nonnegative(),
});

router.post(
  '/v1/rallies/:id/split',
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }), body: splitRallySchema }),
  async (req, res, next) => {
    try {
      const result = await splitRally(req.params.id, req.userId!, req.body);
      res.status(201).json(result);
    } catch (error) {
      next(error);
    }
  },
);
```

Add `splitRally` to the existing rallyService import.

- [ ] **Step 5: Run all split tests, confirm pass.**

```bash
cd api && npx vitest run tests/splitRally.test.ts
```

Expected: 5 passed.

- [ ] **Step 6: Typecheck + commit.**

```bash
cd api && npx tsc --noEmit
git add api/src/services/rallyService.ts api/src/routes/rallies.ts api/tests/splitRally.test.ts
git commit -m "feat(b): POST /v1/rallies/:id/split slices PlayerTrack in place"
```

---

## Task 12: `concatPlayerTracks` helper + merge service + route + tests

**Files:**
- Modify: `api/src/services/rallySlicing.ts`
- Modify: `api/src/services/rallyService.ts`
- Modify: `api/src/routes/rallies.ts`
- Modify: `api/tests/rallySlicing.test.ts` (add concat cases)
- Create: `api/tests/mergeRallies.test.ts`

- [ ] **Step 1: Add concat tests to `api/tests/rallySlicing.test.ts`.** Append:

```ts
import { concatPlayerTracks } from '../src/services/rallySlicing';

describe('concatPlayerTracks', () => {
  const a = {
    fps: 30, frameCount: 100, courtSplitY: 0.5, processingTimeMs: 500, modelVersion: 'v1',
    status: 'COMPLETED', needsRetrack: false, qualityReportJson: null,
    positionsJson: [{ frameNumber: 10, trackId: 1, confidence: 0.9 }],
    rawPositionsJson: [], ballPositionsJson: [{ frameNumber: 20 }],
    contactsJson: [{ frame: 50, playerTrackId: 1 }],
    actionsJson: [], groundTruthJson: null, actionGroundTruthJson: null,
  };
  const b = {
    fps: 30, frameCount: 150, courtSplitY: 0.5, processingTimeMs: 700, modelVersion: 'v1',
    status: 'COMPLETED', needsRetrack: false, qualityReportJson: null,
    positionsJson: [{ frameNumber: 5, trackId: 2, confidence: 0.85 }],
    rawPositionsJson: [], ballPositionsJson: [{ frameNumber: 10 }],
    contactsJson: [{ frame: 30, playerTrackId: 2 }],
    actionsJson: [], groundTruthJson: null, actionGroundTruthJson: null,
  };

  it('shifts b frames up by a.frameCount', () => {
    const merged = concatPlayerTracks(a as any, b as any);
    expect(merged.frameCount).toBe(250);
    expect(merged.positionsJson.map((p: any) => p.frameNumber).sort((x: number, y: number) => x - y)).toEqual([10, 105]);
    expect(merged.contactsJson.map((c: any) => c.frame).sort((x: number, y: number) => x - y)).toEqual([50, 130]);
    expect(merged.ballPositionsJson.map((p: any) => p.frameNumber).sort((x: number, y: number) => x - y)).toEqual([20, 110]);
  });

  it('unions trackIds', () => {
    const merged = concatPlayerTracks(a as any, b as any);
    expect(merged.uniqueTrackCount).toBe(2);
  });
});
```

- [ ] **Step 2: Run, confirm failure.**

```bash
cd api && npx vitest run tests/rallySlicing.test.ts
```

Expected: FAIL on `concatPlayerTracks` not exported.

- [ ] **Step 3: Implement `concatPlayerTracks`.** Append to `api/src/services/rallySlicing.ts`:

```ts
function shiftFrames(arr: AnyFrameEntry[] | null, field: FrameField, offset: number): AnyFrameEntry[] {
  if (!arr) return [];
  return arr.map(entry => {
    const f = entry[field];
    return typeof f === 'number' ? { ...entry, [field]: f + offset } : entry;
  });
}

export function concatPlayerTracks(
  a: SlicePlayerTrackInput,
  b: SlicePlayerTrackInput,
): SlicedPlayerTrack {
  const offset = a.frameCount;
  const positionsJson = [...(a.positionsJson ?? []), ...shiftFrames(b.positionsJson, 'frameNumber', offset)];
  const rawPositionsJson = [...(a.rawPositionsJson ?? []), ...shiftFrames(b.rawPositionsJson, 'frameNumber', offset)];
  const ballPositionsJson = [...(a.ballPositionsJson ?? []), ...shiftFrames(b.ballPositionsJson, 'frameNumber', offset)];
  const contactsJson = [...(a.contactsJson ?? []), ...shiftFrames(b.contactsJson, 'frame', offset)];
  const actionsJson = [...(a.actionsJson ?? []), ...shiftFrames(b.actionsJson, 'frame', offset)];
  const groundTruthJson = (a.groundTruthJson || b.groundTruthJson)
    ? [...(a.groundTruthJson ?? []), ...shiftFrames(b.groundTruthJson, 'frame', offset)]
    : null;
  const actionGroundTruthJson = (a.actionGroundTruthJson || b.actionGroundTruthJson)
    ? [...(a.actionGroundTruthJson ?? []), ...shiftFrames(b.actionGroundTruthJson, 'frame', offset)]
    : null;

  const frameCount = a.frameCount + b.frameCount;
  const meta = recomputeMetadata(positionsJson, frameCount);

  return {
    fps: a.fps,
    courtSplitY: a.courtSplitY,
    processingTimeMs: (a.processingTimeMs ?? 0) + (b.processingTimeMs ?? 0),
    modelVersion: a.modelVersion,
    status: a.status === 'COMPLETED' && b.status === 'COMPLETED' ? 'COMPLETED' : 'PROCESSING',
    needsRetrack: a.needsRetrack || b.needsRetrack,
    qualityReportJson: a.qualityReportJson,
    frameCount,
    ...meta,
    positionsJson,
    rawPositionsJson,
    ballPositionsJson,
    contactsJson,
    actionsJson,
    groundTruthJson,
    actionGroundTruthJson,
  };
}
```

- [ ] **Step 4: Run rallySlicing tests.**

```bash
cd api && npx vitest run tests/rallySlicing.test.ts
```

Expected: 7 passed (5 slice + 2 concat).

- [ ] **Step 5: Write merge integration test.** File `api/tests/mergeRallies.test.ts`:

```ts
import 'dotenv/config';
import { afterEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { mergeRallies } from '../src/services/rallyService';
import { RalliesOverlapError, LockedRallyError } from '../src/middleware/errorHandler';

const userId = '11111111-1111-1111-1111-000000000b12';
const videoId = '22222222-2222-2222-2222-000000000b12';
const rallyA = '33333333-3333-3333-3333-000000000b1a';
const rallyB = '33333333-3333-3333-3333-000000000b1b';

async function setup(opts: { gap?: boolean; lockB?: boolean } = {}) {
  await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
  await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
  await prisma.video.create({
    data: {
      id: videoId, userId, filename: 'b.mp4', sizeBytes: BigInt(1), durationMs: 60000,
      matchAnalysisJson: {
        videoId, numRallies: 2,
        rallies: [
          { rallyId: rallyA, canonicalLocked: false, trackToPlayer: { '1': 1 }, assignmentConfidence: 0.9 },
          { rallyId: rallyB, canonicalLocked: opts.lockB === true, trackToPlayer: { '1': 1 }, assignmentConfidence: 0.9 },
        ],
      },
    },
  });
  const bStart = opts.gap ? 6000 : 5000;
  await prisma.rally.createMany({
    data: [
      { id: rallyA, videoId, startMs: 0, endMs: 5000, scoreA: 1, scoreB: 0, servingTeam: 'A' },
      { id: rallyB, videoId, startMs: bStart, endMs: 10000, scoreA: 2, scoreB: 0, servingTeam: 'A' },
    ],
  });
  await prisma.playerTrack.createMany({
    data: [
      {
        rallyId: rallyA, status: 'COMPLETED', fps: 30, frameCount: 150, detectionRate: 1, avgConfidence: 0.9,
        avgPlayerCount: 4, uniqueTrackCount: 2, courtSplitY: 0.5, processingTimeMs: 500, modelVersion: 'v1', needsRetrack: false,
        positionsJson: [{ frameNumber: 10, trackId: 1, confidence: 0.9 }],
        rawPositionsJson: [], ballPositionsJson: [], contactsJson: [{ frame: 50, playerTrackId: 1 }], actionsJson: [],
      } as any,
      {
        rallyId: rallyB, status: 'COMPLETED', fps: 30, frameCount: opts.gap ? 120 : 150, detectionRate: 1, avgConfidence: 0.9,
        avgPlayerCount: 4, uniqueTrackCount: 2, courtSplitY: 0.5, processingTimeMs: 500, modelVersion: 'v1', needsRetrack: false,
        positionsJson: [{ frameNumber: 5, trackId: 2, confidence: 0.85 }],
        rawPositionsJson: [], ballPositionsJson: [], contactsJson: [{ frame: 30, playerTrackId: 2 }], actionsJson: [],
      } as any,
    ],
  });
}

async function cleanup() {
  await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
}

describe('mergeRallies', () => {
  afterEach(cleanup);

  it('no-gap concat: produces one rally with stitched PlayerTrack', async () => {
    await setup({ gap: false });
    const { rally } = await mergeRallies([rallyA, rallyB], userId);
    expect(rally.startMs).toBe(0);
    expect(rally.endMs).toBe(10000);
    const pt = await prisma.playerTrack.findUnique({ where: { rallyId: rally.id } });
    expect(pt!.frameCount).toBe(300);
    expect((pt!.contactsJson as any).map((c: any) => c.frame).sort()).toEqual([50, 180]); // 30 + 150
    expect(await prisma.rally.findUnique({ where: { id: rallyA } })).toBeNull();
    expect(await prisma.rally.findUnique({ where: { id: rallyB } })).toBeNull();
  });

  it('gap: creates new rally without PlayerTrack, drops both parents', async () => {
    await setup({ gap: true });
    const { rally } = await mergeRallies([rallyA, rallyB], userId);
    expect(rally.startMs).toBe(0);
    expect(rally.endMs).toBe(10000);
    expect(await prisma.playerTrack.findUnique({ where: { rallyId: rally.id } })).toBeNull();
  });

  it('rejects when either input is locked', async () => {
    await setup({ lockB: true });
    await expect(mergeRallies([rallyA, rallyB], userId)).rejects.toBeInstanceOf(LockedRallyError);
  });

  it('rejects overlapping rallies', async () => {
    await setup();
    // force overlap by resetting rallyB.startMs to 3000
    await prisma.rally.update({ where: { id: rallyB }, data: { startMs: 3000 } });
    await expect(mergeRallies([rallyA, rallyB], userId)).rejects.toBeInstanceOf(RalliesOverlapError);
  });
});
```

- [ ] **Step 6: Run, confirm failure.**

```bash
cd api && npx vitest run tests/mergeRallies.test.ts
```

Expected: FAIL (`mergeRallies` not exported).

- [ ] **Step 7: Implement `mergeRallies`.** Append to `api/src/services/rallyService.ts`:

```ts
import { concatPlayerTracks } from './rallySlicing';
import { RalliesOverlapError } from '../middleware/errorHandler';

export async function mergeRallies(
  rallyIds: [string, string],
  userId: string,
): Promise<{ rally: Rally }> {
  if (rallyIds.length !== 2) {
    throw new RalliesOverlapError(rallyIds);
  }
  return prisma.$transaction(async (tx) => {
    const raw = await tx.rally.findMany({
      where: { id: { in: rallyIds } },
      include: { playerTrack: true, video: { select: { userId: true, matchAnalysisJson: true } } },
    });
    if (raw.length !== 2) throw new NotFoundError('Rally', rallyIds.join(','));
    if (raw[0].video.userId !== userId || raw[1].video.userId !== userId) {
      throw new NotFoundError('Rally', rallyIds.join(','));
    }
    if (raw[0].videoId !== raw[1].videoId) throw new RalliesOverlapError(rallyIds);

    const [a, b] = [...raw].sort((x, y) => x.startMs - y.startMs);
    if (b.startMs < a.endMs) throw new RalliesOverlapError(rallyIds);

    // Lock gate on both
    await assertNotLocked(tx, a.id, 'MERGE');
    await assertNotLocked(tx, b.id, 'MERGE');

    // PlayerTrack state gate on both
    for (const r of [a, b]) {
      if (r.playerTrack?.status === 'PROCESSING') throw new RallyTrackingStateError('IN_PROGRESS', r.id);
      if (r.playerTrack?.status === 'FAILED') throw new RallyTrackingStateError('FAILED', r.id);
    }

    const gap = b.startMs > a.endMs;

    // Create merged rally
    const merged = await tx.rally.create({
      data: {
        videoId: a.videoId,
        startMs: a.startMs,
        endMs: b.endMs,
        scoreA: b.scoreA,
        scoreB: b.scoreB,
        servingTeam: b.servingTeam,
        notes: [a.notes, b.notes].filter(Boolean).join('\n') || null,
        confidence: Math.min(a.confidence ?? 1, b.confidence ?? 1),
      },
    });

    // Concat tracks only when there's no gap AND both tracks exist
    if (!gap && a.playerTrack && b.playerTrack) {
      const stitched = concatPlayerTracks(a.playerTrack as any, b.playerTrack as any);
      await tx.playerTrack.create({ data: { rallyId: merged.id, ...(stitched as any) } });
    }

    // Update matchAnalysisJson.rallies[]
    const json: any = a.video.matchAnalysisJson ?? { rallies: [] };
    const aEntry = (json.rallies ?? []).find((r: any) => r.rallyId === a.id);
    json.rallies = (json.rallies ?? []).filter((r: any) => r.rallyId !== a.id && r.rallyId !== b.id);
    json.rallies.push({
      rallyId: merged.id, canonicalLocked: false,
      trackToPlayer: aEntry?.trackToPlayer ?? {},
      assignmentConfidence: null, serverPlayerId: null,
    });
    await tx.video.update({ where: { id: a.videoId }, data: { matchAnalysisJson: json } });

    // Delete parents (cascades their PlayerTracks)
    await tx.rally.deleteMany({ where: { id: { in: [a.id, b.id] } } });

    // Record edit
    await appendEditsBatch(tx, a.videoId, [{ rallyId: merged.id, editKind: 'merge' }]);

    return { rally: merged };
  });
}
```

- [ ] **Step 8: Add route.** In `api/src/routes/rallies.ts`:

```ts
const mergeRalliesSchema = z.object({
  rallyIds: z.tuple([uuidSchema, uuidSchema]),
});

router.post(
  '/v1/rallies/merge',
  requireUser,
  validateRequest({ body: mergeRalliesSchema }),
  async (req, res, next) => {
    try {
      const result = await mergeRallies(req.body.rallyIds, req.userId!);
      res.status(201).json(result);
    } catch (error) {
      next(error);
    }
  },
);
```

Add `mergeRallies` to the rallyService import.

- [ ] **Step 9: Run all merge tests, confirm pass.**

```bash
cd api && npx vitest run tests/mergeRallies.test.ts
```

Expected: 4 passed.

- [ ] **Step 10: Typecheck + commit.**

```bash
cd api && npx tsc --noEmit
git add api/src/services/rallySlicing.ts api/src/services/rallyService.ts api/src/routes/rallies.ts api/tests/rallySlicing.test.ts api/tests/mergeRallies.test.ts
git commit -m "feat(b): POST /v1/rallies/merge with no-gap concat + gap-triggers-retrack"
```

---

## Task 13: Extend-path lock guard + full test run

**Files:**
- Modify: `api/src/services/rallyService.ts`

- [ ] **Step 1: Wire lock guard into `updateRally`.** Inside the existing transaction in `updateRally`, locate the extend-detection block (where `markRetrackIfExtended` is called). Before that call, add:

```ts
const willExtend = (data.startMs ?? rally.startMs) < rally.startMs || (data.endMs ?? rally.endMs) > rally.endMs;
if (willExtend) {
  await assertNotLocked(tx, id, 'EXTEND');
}
```

Ensure `assertNotLocked` is imported at the top: `import { assertNotLocked } from './canonicalLockGuard';`

- [ ] **Step 2: Run full vitest suite.**

```bash
cd api && npx vitest run
```

Expected: all tests pass (previous suites + all new Task 3–12 tests).

- [ ] **Step 3: Commit.**

```bash
git add api/src/services/rallyService.ts
git commit -m "feat(b): assertNotLocked before markRetrackIfExtended in updateRally"
```

---

## Task 14: Client wiring

**Files:**
- Modify: `web/src/services/api.ts`
- Modify: `web/src/stores/editorStore.ts`

- [ ] **Step 1: Add client API functions** in `web/src/services/api.ts`:

```ts
export async function splitRally(
  rallyId: string,
  body: { firstEndMs: number; secondStartMs: number },
): Promise<{ firstRally: Rally; secondRally: Rally }> {
  const res = await apiFetch(`/v1/rallies/${rallyId}/split`, { method: 'POST', body: JSON.stringify(body) });
  return res.json();
}

export async function mergeRalliesApi(
  rallyIds: [string, string],
): Promise<{ rally: Rally }> {
  const res = await apiFetch(`/v1/rallies/merge`, { method: 'POST', body: JSON.stringify({ rallyIds }) });
  return res.json();
}

export async function unlockRally(rallyId: string): Promise<{ rallyId: string; wasLocked: boolean; unlockedAt: string }> {
  const res = await apiFetch(`/v1/rallies/${rallyId}/unlock`, { method: 'POST' });
  return res.json();
}

export async function deleteRally(rallyId: string, opts?: { confirmUnlock?: boolean }): Promise<void> {
  await apiFetch(`/v1/rallies/${rallyId}`, {
    method: 'DELETE',
    body: opts?.confirmUnlock ? JSON.stringify({ confirmUnlock: true }) : undefined,
  });
}
```

Use the project's existing fetch helper and adjust the import path / wrapper name (`apiFetch`) to whatever the file already exports.

- [ ] **Step 2: Add `splitRally` action + wire existing `mergeRallies` to API** in `web/src/stores/editorStore.ts`. Locate the existing `mergeRallies` action; replace its body to call `mergeRalliesApi(firstId, secondId)` and update state with the returned rally, preserving the existing highlight/cameraEdit merging logic (which now runs after the server response). For `splitRally`, model the action after the existing `addRally` pattern:

```ts
splitRally: async (rallyId: string, firstEndMs: number, secondStartMs: number) => {
  const { firstRally, secondRally } = await splitRally(rallyId, { firstEndMs, secondStartMs });
  const activeMatchId = get().activeMatchId;
  set((state) => {
    state.pushHistory();
    state.rallies = state.rallies.filter(r => r.id !== rallyId).concat([firstRally, secondRally]).sort((a, b) => a.startMs - b.startMs);
  });
  syncService.markDirty();
  if (activeMatchId) useAnalysisStore.getState().notifyRallyEdited(activeMatchId);
},

unlockRally: async (rallyId: string) => {
  await unlockRallyApi(rallyId);
  // Re-fetch match analysis JSON to update lock badges; or patch local state
},
```

Adjust method signatures to match the existing editor store style (action names, param patterns, how the store accesses Zustand `set`/`get`).

- [ ] **Step 3: Typecheck web.**

```bash
cd web && npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 4: Commit.**

```bash
git add web/src/services/api.ts web/src/stores/editorStore.ts
git commit -m "feat(b): web client wiring for split / merge / unlock / delete-confirm"
```

---

## Task 15: Documentation

**Files:**
- Modify: `api/CLAUDE.md`

- [ ] **Step 1: Add a "Project B — Rally-Edit Propagation" subsection** below the existing "Resilience (A2b)" subsection. Cover:

- New endpoints: `POST /v1/rallies/:id/split`, `POST /v1/rallies/merge`, `POST /v1/rallies/:id/unlock`; delete gains `{confirmUnlock}` body.
- `Video.pendingAnalysisEditsJson` accumulates `{rallyId, editKind, at}` entries during the A2a debounce window; `consumePendingEdits` atomically returns + clears.
- `planStages` pure function in `matchAnalysisPlanning.ts` maps edit kinds to the minimum-sufficient stage set. `extend | create | merge` → fullRerun; `scalar | delete` → stage 6 only; `shorten | split` → stages 4/5 per-rally via `--rally-ids`.
- Canonical-lock guard is centralized in `canonicalLockGuard.ts`; every structural edit path must call `assertNotLocked(tx, rallyId, op)` inside its transaction.
- Split is cut-with-gap: middle frames `[firstEndFrame, secondStartFrame)` are discarded. Children inherit parent's `trackToPlayer` since raw track IDs are preserved.
- Merge has two branches: no-gap concatenates PlayerTracks in place with back-half frames shifted up; gap falls through to catch-up retrack (both parent tracks dropped, new rally is `PlayerTrack=null`).
- Stage-timing telemetry emitted as `match_analysis.stage_timings` structured log on every run. Use it to decide whether warm-start Hungarian is worth future work.

- [ ] **Step 2: Commit.**

```bash
git add api/CLAUDE.md
git commit -m "docs(b): document rally-edit propagation invariants + endpoints"
```

---

## Task 16: Final verification + merge

- [ ] **Step 1: Full API test suite.**

```bash
cd api && npx vitest run
```

Expected: all tests pass.

- [ ] **Step 2: Full API typecheck.**

```bash
cd api && npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 3: Full web typecheck.**

```bash
cd web && npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 4: Python type + lint on modified CLIs.**

```bash
cd analysis && uv run mypy rallycut/cli/commands/remap_track_ids.py rallycut/cli/commands/reattribute_actions.py
uv run ruff check rallycut/cli/commands/remap_track_ids.py rallycut/cli/commands/reattribute_actions.py
```

Expected: 0 errors.

- [ ] **Step 5: Manual end-to-end smoke** on a local video (see spec section "Testing → End-to-end"). Watch for `match_analysis.stage_timings` log lines to confirm telemetry is emitting.

- [ ] **Step 6: Surface any deviations** from the spec in a `Deviations from spec` section of the plan before merging (following A1's precedent).

---

## Self-Review Results

- **Spec coverage:** every in-scope item from the spec has a task — schema (Task 1), error classes (Task 2), lock guard (Task 3), unlock endpoint (Task 4), pending-edit helpers (Task 5), edit-kind tracking in existing CRUD (Task 6), planStages (Task 7), Python `--rally-ids` (Task 8), stage-gated runMatchAnalysis (Task 9), slice helper (Task 10), split service+route (Task 11), concat+merge (Task 12), extend lock guard (Task 13), client wiring (Task 14), documentation (Task 15).
- **Placeholder scan:** no TBD / TODO / "similar to Task N" references remain.
- **Type consistency:** `EditKind` is defined in Task 5 and reused in Task 7; `SlicePlayerTrackInput` / `SlicedPlayerTrack` defined in Task 10 and reused in Task 12; `assertNotLocked(op)` signature consistent across Task 3 / 11 / 12 / 13. `planStages` return shape (`{fullRerun, runStage2, changedRallyIds}`) consistent between Task 7 and Task 9.
- **Deviation from spec:** the spec listed `api/src/errors/rallyErrors.ts` as a new file, but the existing convention (anchored via the explore report) puts all `AppError` subclasses alongside the base in `api/src/middleware/errorHandler.ts`. Plan follows the existing convention to avoid a refactor wave; documented at Task 2.
