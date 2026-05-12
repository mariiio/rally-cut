# Action GT Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `PlayerTrack.actionGroundTruthJson` with a decoupled `rally_action_ground_truth` table whose rows carry a pixel-anchored bbox snapshot. Action GT survives retracks, merges, and model changes; a re-resolver re-derives `resolvedTrackId` after each tracking write.

**Architecture:** New table FK'd to `Rally`. Pure resolver function (bbox IoU → nearest-center → unresolved) wired into `saveTrackingResult` inside the existing transaction. New `actionGroundTruthService.ts` owns all writes; old helpers in `playerTrackingService.ts` are deleted. Backfill CLI snapshots bboxes from current `rawPositionsJson` where possible; rows that can't snapshot land in `UNRESOLVED` and are visible in a new `audit-action-gt` CLI.

**Tech Stack:** Prisma + PostgreSQL, Express + TypeScript (API), Python 3.11 + psycopg (analysis), Vitest (API tests), pytest (Python tests), Next.js + Zustand (web).

**Spec:** `docs/superpowers/specs/2026-05-12-action-gt-decouple-design.md`.

---

## File Structure

**Create (API):**
- `api/src/services/actionGroundTruthService.ts` — owns all GT writes/reads
- `api/src/services/actionGroundTruthResolver.ts` — pure resolver function
- `api/tests/actionGroundTruthResolver.test.ts` — resolver unit tests
- `api/tests/actionGroundTruthService.test.ts` — service integration tests
- `api/prisma/migrations/<ts>_action_gt_table/migration.sql` — schema migration
- `api/prisma/migrations/<ts>_drop_player_track_action_gt/migration.sql` — final cutover

**Modify (API):**
- `api/prisma/schema.prisma` — add `RallyActionGroundTruth` model + enums; drop `actionGroundTruthJson` column at cutover
- `api/src/services/playerTrackingService.ts` — delete inline GT helpers; hook resolver into `saveTrackingResult`
- `api/src/services/rallySlicing.ts` — drop `actionGroundTruthJson` from `SlicePlayerTrackInput`/`SlicedPlayerTrack`
- `api/src/services/mergeRalliesService.ts` — copy parent GT rows into merged rally
- `api/src/services/splitRallyService.ts` — partition GT rows by frame
- `api/src/services/matchAnalysisService.ts` — drop the GT copy at line ~757
- `api/src/routes/rallies.ts` — wire new service + reattach endpoint

**Create (Python):**
- `analysis/rallycut/cli/commands/migrate_action_gt.py` — backfill CLI
- `analysis/rallycut/cli/commands/audit_action_gt.py` — audit CLI
- `analysis/tests/integration/test_migrate_action_gt.py` — backfill tests
- `analysis/tests/integration/test_audit_action_gt.py` — audit tests

**Modify (Python):**
- `analysis/rallycut/cli/commands/train.py` — new query at lines 1637–1695
- `analysis/rallycut/training/restore.py` — write to new table at lines 511–568
- `analysis/rallycut/cli/commands/remap_track_ids.py` — delete GT rewrite at lines 690–915
- `analysis/scripts/eval_action_detection.py` — new query path
- `analysis/scripts/audit_action_gt_vs_click_gt.py` — read both resolved + unresolved
- `analysis/scripts/measure_attribution_*.py` — switch query
- `analysis/scripts/_smoke_snapshot_gt.py`, `analysis/scripts/check_gt_integrity.py` — switch query

**Modify (Web):**
- `web/src/services/api.ts` — `ActionGroundTruthLabel` gains `resolvedTrackId`, `resolvedSource`, `snapshotBbox*`, `snapshotBall*`, `snapshotTeam`
- `web/src/stores/playerTrackingStore.ts` — adapt save/load to new payload + add reattach
- `web/src/components/ActionLabelingMode.tsx` — pass bbox + ball context on save
- `web/src/components/ActionOverlay.tsx` — ghost-render UNRESOLVED rows
- `web/src/utils/gtLabelDisplay.ts` — use `resolvedTrackId` → `appliedFullMapping`

---

## Phase 1 — Schema

### Task 1: Add `RallyActionGroundTruth` model (additive, no callers)

**Files:**
- Modify: `api/prisma/schema.prisma`
- Create: `api/prisma/migrations/<ts>_action_gt_table/migration.sql` (auto-generated)

- [ ] **Step 1: Add enums + model to schema.prisma**

Append after the `PlayerTrack` model:

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
  SNAPSHOT_EXACT
  IOU_MATCH
  NEAREST_CENTER
  MANUAL
  UNRESOLVED
}

enum GtTeam {
  A
  B
}

model RallyActionGroundTruth {
  id              String         @id @default(uuid())
  rallyId         String         @map("rally_id")
  frame           Int
  action          ActionLabel

  snapshotBboxX1  Float?         @map("snapshot_bbox_x1")
  snapshotBboxY1  Float?         @map("snapshot_bbox_y1")
  snapshotBboxX2  Float?         @map("snapshot_bbox_x2")
  snapshotBboxY2  Float?         @map("snapshot_bbox_y2")
  snapshotBallX   Float?         @map("snapshot_ball_x")
  snapshotBallY   Float?         @map("snapshot_ball_y")
  snapshotTeam    GtTeam?        @map("snapshot_team")
  snapshotTrackId Int?           @map("snapshot_track_id")

  resolvedTrackId Int?           @map("resolved_track_id")
  resolvedAt      DateTime?      @map("resolved_at") @db.Timestamptz(6)
  resolvedSource  ResolveSource? @map("resolved_source")

  createdAt       DateTime       @default(now()) @map("created_at")
  updatedAt       DateTime       @updatedAt @map("updated_at")
  createdBy       String?        @map("created_by")

  rally           Rally          @relation(fields: [rallyId], references: [id], onDelete: Cascade)

  @@unique([rallyId, frame, action])
  @@index([rallyId])
  @@index([rallyId, frame])
  @@map("rally_action_ground_truth")
}
```

Also add the back-relation on `Rally`:

```prisma
model Rally {
  // ... existing fields ...
  actionGroundTruth RallyActionGroundTruth[]
}
```

- [ ] **Step 2: Generate migration**

Run: `cd api && npx prisma migrate dev --name action_gt_table`

Expected: new `api/prisma/migrations/<ts>_action_gt_table/migration.sql` created, applied to local DB.

- [ ] **Step 3: Verify with Prisma Studio**

Run: `cd api && npx prisma studio`

Expected: `RallyActionGroundTruth` table visible, empty, with all columns.

- [ ] **Step 4: Commit**

```bash
git add api/prisma/schema.prisma api/prisma/migrations/
git commit -m "feat(action-gt): add rally_action_ground_truth table"
```

---

## Phase 2 — Pure resolver

### Task 2: Resolver function + tests

**Files:**
- Create: `api/src/services/actionGroundTruthResolver.ts`
- Create: `api/tests/actionGroundTruthResolver.test.ts`

- [ ] **Step 1: Write the failing test file**

`api/tests/actionGroundTruthResolver.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { resolveGtRow } from '../src/services/actionGroundTruthResolver';

type Pos = { trackId: number; bbox: { x1: number; y1: number; x2: number; y2: number } };

const at = (trackId: number, x1: number, y1: number, x2: number, y2: number): Pos => ({
  trackId, bbox: { x1, y1, x2, y2 },
});

describe('resolveGtRow', () => {
  it('returns SNAPSHOT_EXACT when bbox is null but snapshotTrackId is present at frame', () => {
    const row = { snapshotBboxX1: null, snapshotBboxY1: null, snapshotBboxX2: null, snapshotBboxY2: null, snapshotTrackId: 7 };
    const positions = [at(7, 0.1, 0.1, 0.2, 0.3)];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: 7, resolvedSource: 'SNAPSHOT_EXACT' });
  });

  it('returns UNRESOLVED when bbox null and trackId absent from frame', () => {
    const row = { snapshotBboxX1: null, snapshotBboxY1: null, snapshotBboxX2: null, snapshotBboxY2: null, snapshotTrackId: 9 };
    const positions = [at(7, 0.1, 0.1, 0.2, 0.3)];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('returns IOU_MATCH when bbox overlaps >= 0.5 with a candidate', () => {
    const row = { snapshotBboxX1: 0.1, snapshotBboxY1: 0.1, snapshotBboxX2: 0.2, snapshotBboxY2: 0.3, snapshotTrackId: 99 };
    const positions = [at(7, 0.11, 0.11, 0.21, 0.31)]; // ~near-identical
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('IOU_MATCH');
    expect(out.resolvedTrackId).toBe(7);
  });

  it('returns NEAREST_CENTER when IoU is low but center within 0.10', () => {
    const row = { snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.12, snapshotBboxY2: 0.14, snapshotTrackId: 99 };
    // Candidate offset by ~0.03 in each axis: no IoU overlap, center within 0.10.
    const positions = [at(7, 0.13, 0.13, 0.15, 0.17)];
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('NEAREST_CENTER');
    expect(out.resolvedTrackId).toBe(7);
  });

  it('returns UNRESOLVED when no candidate meets either threshold', () => {
    const row = { snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.20, snapshotBboxY2: 0.30, snapshotTrackId: 99 };
    const positions = [at(7, 0.80, 0.80, 0.90, 0.95)];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('returns UNRESOLVED when positions are empty and bbox is null', () => {
    const row = { snapshotBboxX1: null, snapshotBboxY1: null, snapshotBboxX2: null, snapshotBboxY2: null, snapshotTrackId: 7 };
    expect(resolveGtRow(row, [])).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd api && npx vitest run tests/actionGroundTruthResolver.test.ts`
Expected: FAIL — `Cannot find module '../src/services/actionGroundTruthResolver'`.

- [ ] **Step 3: Implement the resolver**

`api/src/services/actionGroundTruthResolver.ts`:

```typescript
export const IOU_THRESHOLD = 0.5;
export const CENTER_DIST_THRESHOLD = 0.10; // normalized

export type ResolveSource = 'SNAPSHOT_EXACT' | 'IOU_MATCH' | 'NEAREST_CENTER' | 'MANUAL' | 'UNRESOLVED';

export interface GtRowInput {
  snapshotBboxX1: number | null;
  snapshotBboxY1: number | null;
  snapshotBboxX2: number | null;
  snapshotBboxY2: number | null;
  snapshotTrackId: number | null;
}

export interface Candidate {
  trackId: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

export interface ResolveResult {
  resolvedTrackId: number | null;
  resolvedSource: ResolveSource;
}

function iou(a: Candidate['bbox'], b: Candidate['bbox']): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

function center(b: Candidate['bbox']): [number, number] {
  return [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2];
}

export function resolveGtRow(row: GtRowInput, positions: Candidate[]): ResolveResult {
  const hasBbox =
    row.snapshotBboxX1 != null && row.snapshotBboxY1 != null &&
    row.snapshotBboxX2 != null && row.snapshotBboxY2 != null;

  if (!hasBbox) {
    if (row.snapshotTrackId != null && positions.some(p => p.trackId === row.snapshotTrackId)) {
      return { resolvedTrackId: row.snapshotTrackId, resolvedSource: 'SNAPSHOT_EXACT' };
    }
    return { resolvedTrackId: null, resolvedSource: 'UNRESOLVED' };
  }

  const snapshot = {
    x1: row.snapshotBboxX1!, y1: row.snapshotBboxY1!,
    x2: row.snapshotBboxX2!, y2: row.snapshotBboxY2!,
  };

  if (positions.length === 0) return { resolvedTrackId: null, resolvedSource: 'UNRESOLVED' };

  const ranked = positions
    .map(p => ({ trackId: p.trackId, iouVal: iou(snapshot, p.bbox), pos: p }))
    .sort((a, b) => b.iouVal - a.iouVal);
  if (ranked[0].iouVal >= IOU_THRESHOLD) {
    return { resolvedTrackId: ranked[0].trackId, resolvedSource: 'IOU_MATCH' };
  }

  const [sx, sy] = center(snapshot);
  const distRanked = positions
    .map(p => {
      const [cx, cy] = center(p.bbox);
      return { trackId: p.trackId, dist: Math.hypot(cx - sx, cy - sy) };
    })
    .sort((a, b) => a.dist - b.dist);
  if (distRanked[0].dist <= CENTER_DIST_THRESHOLD) {
    return { resolvedTrackId: distRanked[0].trackId, resolvedSource: 'NEAREST_CENTER' };
  }

  return { resolvedTrackId: null, resolvedSource: 'UNRESOLVED' };
}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd api && npx vitest run tests/actionGroundTruthResolver.test.ts`
Expected: PASS — 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add api/src/services/actionGroundTruthResolver.ts api/tests/actionGroundTruthResolver.test.ts
git commit -m "feat(action-gt): pure resolver with IoU + center-dist fallback"
```

---

## Phase 3 — API service

### Task 3: `actionGroundTruthService.ts` — save / get / reattach

**Files:**
- Create: `api/src/services/actionGroundTruthService.ts`
- Create: `api/tests/actionGroundTruthService.test.ts`

- [ ] **Step 1: Write the failing tests**

`api/tests/actionGroundTruthService.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { saveActionGroundTruth, getActionGroundTruth, reattachActionGroundTruth } from '../src/services/actionGroundTruthService';
import { setupTestUser, setupTestRallyWithTrack } from './fixtures/setup'; // Vitest fixture helpers — pattern matches existing tests

describe('actionGroundTruthService', () => {
  let userId: string;
  let rallyId: string;

  beforeEach(async () => {
    userId = (await setupTestUser()).id;
    rallyId = (await setupTestRallyWithTrack({
      userId,
      frameCount: 100,
      // One tracked player at frame 50: trackId 7, bbox (0.1, 0.1, 0.2, 0.3)
      rawPositions: [{ frameNumber: 50, trackId: 7, bbox: { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.3 }, confidence: 0.9 }],
      ballPositions: [{ frameNumber: 50, x: 0.5, y: 0.5 }],
      primaryTrackIds: [7],
      teamAssignments: { '7': 'A' },
    })).id;
  });

  it('saves a label with bbox snapshot when PlayerTrack has the trackId at frame', async () => {
    const { savedCount } = await saveActionGroundTruth(rallyId, userId, [
      { frame: 50, action: 'serve', trackId: 7 },
    ]);
    expect(savedCount).toBe(1);

    const rows = await prisma.rallyActionGroundTruth.findMany({ where: { rallyId } });
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      frame: 50,
      action: 'SERVE',
      snapshotBboxX1: 0.1, snapshotBboxY1: 0.1, snapshotBboxX2: 0.2, snapshotBboxY2: 0.3,
      snapshotBallX: 0.5, snapshotBallY: 0.5,
      snapshotTeam: 'A',
      snapshotTrackId: 7,
      resolvedTrackId: 7,
      resolvedSource: 'SNAPSHOT_EXACT',
      createdBy: userId,
    });
  });

  it('saves an UNRESOLVED row when trackId is not present at the labeled frame', async () => {
    await saveActionGroundTruth(rallyId, userId, [
      { frame: 50, action: 'serve', trackId: 99 }, // no such track
    ]);
    const rows = await prisma.rallyActionGroundTruth.findMany({ where: { rallyId } });
    expect(rows[0]).toMatchObject({
      snapshotBboxX1: null, snapshotBboxY1: null,
      snapshotTrackId: 99,
      resolvedTrackId: null,
      resolvedSource: 'UNRESOLVED',
    });
  });

  it('upserts on (rallyId, frame, action) — re-save updates in place', async () => {
    await saveActionGroundTruth(rallyId, userId, [{ frame: 50, action: 'serve', trackId: 7 }]);
    await saveActionGroundTruth(rallyId, userId, [{ frame: 50, action: 'serve', trackId: 7, ballX: 0.7, ballY: 0.7 }]);
    const rows = await prisma.rallyActionGroundTruth.findMany({ where: { rallyId } });
    expect(rows).toHaveLength(1);
  });

  it('cascade-deletes when Rally is deleted', async () => {
    await saveActionGroundTruth(rallyId, userId, [{ frame: 50, action: 'serve', trackId: 7 }]);
    await prisma.rally.delete({ where: { id: rallyId } });
    expect(await prisma.rallyActionGroundTruth.count({ where: { rallyId } })).toBe(0);
  });

  it('reattach sets resolvedSource=MANUAL and pinned trackId', async () => {
    const [{ id }] = (await saveActionGroundTruth(rallyId, userId, [{ frame: 50, action: 'serve', trackId: 99 }])).labels;
    await reattachActionGroundTruth(id, userId, 7);
    const row = await prisma.rallyActionGroundTruth.findUniqueOrThrow({ where: { id } });
    expect(row.resolvedTrackId).toBe(7);
    expect(row.resolvedSource).toBe('MANUAL');
  });

  it('rejects save when caller does not own the Video', async () => {
    const otherUser = (await setupTestUser()).id;
    await expect(saveActionGroundTruth(rallyId, otherUser, [{ frame: 50, action: 'serve', trackId: 7 }]))
      .rejects.toThrow(/permission/);
  });
});
```

> If `tests/fixtures/setup.ts` helpers do not yet expose `setupTestRallyWithTrack`, add them here in this step — mirror the patterns in `api/tests/saveTrackingResult.test.ts` (raw SQL + Prisma create).

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd api && npx vitest run tests/actionGroundTruthService.test.ts`
Expected: FAIL — service module missing.

- [ ] **Step 3: Implement the service**

`api/src/services/actionGroundTruthService.ts`:

```typescript
import { prisma } from '../lib/prisma';
import { Prisma, type ActionLabel, type ResolveSource, type GtTeam } from '@prisma/client';
import { ForbiddenError, NotFoundError, ValidationError } from '../middleware/errors';
import { resolveGtRow, type Candidate } from './actionGroundTruthResolver';

export interface ActionGtInput {
  frame: number;
  action: 'serve' | 'receive' | 'set' | 'attack' | 'block' | 'dig';
  trackId?: number;
  ballX?: number;
  ballY?: number;
}

const ACTION_TO_ENUM: Record<ActionGtInput['action'], ActionLabel> = {
  serve: 'SERVE', receive: 'RECEIVE', set: 'SET',
  attack: 'ATTACK', block: 'BLOCK', dig: 'DIG',
};

interface RawPos {
  frameNumber: number;
  trackId: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

interface BallPos { frameNumber: number; x: number; y: number; }

function findBbox(raw: RawPos[] | null, frame: number, trackId: number): RawPos['bbox'] | null {
  if (!raw) return null;
  const hit = raw.find(p => p.frameNumber === frame && p.trackId === trackId);
  return hit?.bbox ?? null;
}

function findBall(balls: BallPos[] | null, frame: number): { x: number; y: number } | null {
  if (!balls) return null;
  const hit = balls.find(b => b.frameNumber === frame);
  return hit ? { x: hit.x, y: hit.y } : null;
}

function resolveTeamFor(
  trackId: number,
  primaryTrackIds: number[] | null,
  teamAssignments: Record<string, 'A' | 'B'> | null,
): GtTeam | null {
  if (!primaryTrackIds || !teamAssignments) return null;
  if (!primaryTrackIds.includes(trackId)) return null;
  const t = teamAssignments[String(trackId)];
  return t === 'A' || t === 'B' ? t : null;
}

async function loadOwnedRally(rallyId: string, userId: string) {
  const rally = await prisma.rally.findUnique({
    where: { id: rallyId },
    include: { video: true, playerTrack: true },
  });
  if (!rally) throw new NotFoundError('Rally', rallyId);
  if (rally.video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to modify action ground truth for this rally');
  }
  return rally;
}

export async function saveActionGroundTruth(
  rallyId: string,
  userId: string,
  labels: ActionGtInput[],
): Promise<{ savedCount: number; labels: { id: string }[] }> {
  const rally = await loadOwnedRally(rallyId, userId);
  const pt = rally.playerTrack;

  const teamAssignments =
    (rally.video as unknown as { teamAssignmentsJson: Record<string, 'A' | 'B'> | null })
      .teamAssignmentsJson ?? null;
  const primaryTrackIds = (pt?.primaryTrackIds ?? null) as number[] | null;

  const written: { id: string }[] = [];
  for (const label of labels) {
    if (pt && (label.frame < 0 || (pt.frameCount != null && label.frame >= pt.frameCount))) {
      throw new ValidationError(`frame ${label.frame} out of range [0, ${pt.frameCount})`);
    }

    const bbox = pt && label.trackId != null
      ? findBbox(pt.rawPositionsJson as RawPos[] | null, label.frame, label.trackId)
      : null;
    const ball = pt
      ? findBall(pt.ballPositionsJson as BallPos[] | null, label.frame)
      : null;
    const team = label.trackId != null
      ? resolveTeamFor(label.trackId, primaryTrackIds, teamAssignments)
      : null;

    let resolvedTrackId: number | null = null;
    let resolvedSource: ResolveSource = 'UNRESOLVED';
    if (bbox && label.trackId != null) {
      resolvedTrackId = label.trackId;
      resolvedSource = 'SNAPSHOT_EXACT';
    }

    const upserted = await prisma.rallyActionGroundTruth.upsert({
      where: { rallyId_frame_action: { rallyId, frame: label.frame, action: ACTION_TO_ENUM[label.action] } },
      create: {
        rallyId,
        frame: label.frame,
        action: ACTION_TO_ENUM[label.action],
        snapshotBboxX1: bbox?.x1 ?? null,
        snapshotBboxY1: bbox?.y1 ?? null,
        snapshotBboxX2: bbox?.x2 ?? null,
        snapshotBboxY2: bbox?.y2 ?? null,
        snapshotBallX: label.ballX ?? ball?.x ?? null,
        snapshotBallY: label.ballY ?? ball?.y ?? null,
        snapshotTeam: team,
        snapshotTrackId: label.trackId ?? null,
        resolvedTrackId,
        resolvedAt: new Date(),
        resolvedSource,
        createdBy: userId,
      },
      update: {
        snapshotBboxX1: bbox?.x1 ?? null,
        snapshotBboxY1: bbox?.y1 ?? null,
        snapshotBboxX2: bbox?.x2 ?? null,
        snapshotBboxY2: bbox?.y2 ?? null,
        snapshotBallX: label.ballX ?? ball?.x ?? null,
        snapshotBallY: label.ballY ?? ball?.y ?? null,
        snapshotTeam: team,
        snapshotTrackId: label.trackId ?? null,
        resolvedTrackId,
        resolvedAt: new Date(),
        resolvedSource,
      },
    });
    written.push({ id: upserted.id });
  }
  return { savedCount: written.length, labels: written };
}

export async function getActionGroundTruth(rallyId: string, userId: string) {
  await loadOwnedRally(rallyId, userId);
  const rows = await prisma.rallyActionGroundTruth.findMany({
    where: { rallyId },
    orderBy: [{ frame: 'asc' }],
  });
  return { labels: rows };
}

export async function reattachActionGroundTruth(
  rowId: string,
  userId: string,
  newResolvedTrackId: number,
) {
  const row = await prisma.rallyActionGroundTruth.findUnique({
    where: { id: rowId },
    include: { rally: { include: { video: true } } },
  });
  if (!row) throw new NotFoundError('RallyActionGroundTruth', rowId);
  if (row.rally.video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to reattach this label');
  }
  await prisma.rallyActionGroundTruth.update({
    where: { id: rowId },
    data: {
      resolvedTrackId: newResolvedTrackId,
      resolvedSource: 'MANUAL',
      resolvedAt: new Date(),
    },
  });
}

export async function reresolveRallyGt(
  tx: Prisma.TransactionClient,
  rallyId: string,
  rawPositions: RawPos[] | null,
): Promise<void> {
  const rows = await tx.rallyActionGroundTruth.findMany({ where: { rallyId } });
  if (rows.length === 0) return;

  // Bucket positions by frame for O(rows × candidates_at_frame) work
  const byFrame = new Map<number, Candidate[]>();
  for (const p of rawPositions ?? []) {
    const arr = byFrame.get(p.frameNumber) ?? [];
    arr.push({ trackId: p.trackId, bbox: p.bbox });
    byFrame.set(p.frameNumber, arr);
  }

  const now = new Date();
  for (const row of rows) {
    const candidates = byFrame.get(row.frame) ?? [];
    const result = resolveGtRow(row, candidates);
    await tx.rallyActionGroundTruth.update({
      where: { id: row.id },
      data: {
        resolvedTrackId: result.resolvedTrackId,
        resolvedSource: result.resolvedSource,
        resolvedAt: now,
      },
    });
  }
}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd api && npx vitest run tests/actionGroundTruthService.test.ts`
Expected: PASS — 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add api/src/services/actionGroundTruthService.ts api/tests/actionGroundTruthService.test.ts api/tests/fixtures/
git commit -m "feat(action-gt): service with save/get/reattach + re-resolver helper"
```

---

### Task 4: Wire routes to new service + add reattach endpoint

**Files:**
- Modify: `api/src/routes/rallies.ts:241-274`

- [ ] **Step 1: Replace existing route handlers + add reattach**

In `api/src/routes/rallies.ts`, change the imports at the top to use the new service:

```typescript
// Replace:
// import { getActionGroundTruth, saveActionGroundTruth } from '../services/playerTrackingService.js';
// With:
import { getActionGroundTruth, saveActionGroundTruth, reattachActionGroundTruth } from '../services/actionGroundTruthService.js';
```

Replace the `GET` handler body at line 247–254 to return the new payload shape (no change needed in the handler itself — just verify the response is `res.json(result ?? { labels: [] })`).

Replace the `PUT` handler at 266–273 — already passes labels through; no change required.

Add a new route after line 274:

```typescript
router.post(
  "/v1/rallies/:rallyId/action-ground-truth/:rowId/reattach",
  requireUser,
  validateRequest({
    params: z.object({ rallyId: uuidSchema, rowId: uuidSchema }),
    body: z.object({ resolvedTrackId: z.number().int().min(0) }),
  }),
  async (req, res, next) => {
    try {
      await reattachActionGroundTruth(req.params.rowId, req.userId!, req.body.resolvedTrackId);
      res.json({ ok: true });
    } catch (error) {
      next(error);
    }
  }
);
```

- [ ] **Step 2: Delete the inline helpers in `playerTrackingService.ts`**

In `api/src/services/playerTrackingService.ts:1076-1155`, delete the `ActionGroundTruthLabel` interface and both `getActionGroundTruth` and `saveActionGroundTruth` functions. Imports elsewhere should already be retargeted (only `routes/rallies.ts` consumed them).

- [ ] **Step 3: Type-check + run touched test files**

```
cd api && npx tsc --noEmit
cd api && npx vitest run tests/actionGroundTruthService.test.ts
```

Expected: PASS — types clean, service tests still pass.

- [ ] **Step 4: Commit**

```bash
git add api/src/routes/rallies.ts api/src/services/playerTrackingService.ts
git commit -m "feat(action-gt): route to new service + reattach endpoint"
```

---

### Task 5: Hook re-resolver into `saveTrackingResult`

**Files:**
- Modify: `api/src/services/playerTrackingService.ts:680-730`
- Create: `api/tests/actionGroundTruthRetrack.test.ts`

- [ ] **Step 1: Write the failing integration test**

`api/tests/actionGroundTruthRetrack.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { saveTrackingResult } from '../src/services/playerTrackingService';
import { saveActionGroundTruth } from '../src/services/actionGroundTruthService';
import { setupTestUser, setupTestRallyWithTrack } from './fixtures/setup';

describe('saveTrackingResult re-resolves action GT', () => {
  let userId: string;
  let rallyId: string;
  let videoId: string;

  beforeEach(async () => {
    const u = await setupTestUser();
    userId = u.id;
    const r = await setupTestRallyWithTrack({
      userId, frameCount: 100,
      rawPositions: [{ frameNumber: 50, trackId: 7, bbox: { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.3 }, confidence: 0.9 }],
      ballPositions: [], primaryTrackIds: [7], teamAssignments: { '7': 'A' },
    });
    rallyId = r.id;
    videoId = r.videoId;
    await saveActionGroundTruth(rallyId, userId, [{ frame: 50, action: 'serve', trackId: 7 }]);
  });

  it('re-resolves to a new trackId via IoU match after retrack', async () => {
    await saveTrackingResult(rallyId, videoId, {
      frameCount: 100, fps: 30, detectionRate: 0.5, avgConfidence: 0.9, avgPlayerCount: 1,
      uniqueTrackCount: 1, courtSplitY: 0.5, primaryTrackIds: [11],
      positions: [], rawPositions: [
        { frameNumber: 50, trackId: 11, bbox: { x1: 0.11, y1: 0.11, x2: 0.21, y2: 0.31 }, confidence: 0.9 },
      ],
      ballPositions: [], contacts: {}, actions: {}, qualityReport: {},
    } as any, 0);

    const row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId } });
    expect(row.resolvedTrackId).toBe(11);
    expect(row.resolvedSource).toBe('IOU_MATCH');
    expect(row.snapshotTrackId).toBe(7); // snapshot unchanged
  });

  it('lands UNRESOLVED when player is no longer tracked at the frame', async () => {
    await saveTrackingResult(rallyId, videoId, {
      frameCount: 100, fps: 30, detectionRate: 0.5, avgConfidence: 0.9, avgPlayerCount: 1,
      uniqueTrackCount: 1, courtSplitY: 0.5, primaryTrackIds: [11],
      positions: [], rawPositions: [
        { frameNumber: 50, trackId: 11, bbox: { x1: 0.80, y1: 0.80, x2: 0.90, y2: 0.95 }, confidence: 0.9 },
      ],
      ballPositions: [], contacts: {}, actions: {}, qualityReport: {},
    } as any, 0);

    const row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId } });
    expect(row.resolvedTrackId).toBeNull();
    expect(row.resolvedSource).toBe('UNRESOLVED');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd api && npx vitest run tests/actionGroundTruthRetrack.test.ts`
Expected: FAIL — re-resolver not invoked.

- [ ] **Step 3: Modify `saveTrackingResult` to call `reresolveRallyGt`**

In `api/src/services/playerTrackingService.ts`, inside the `prisma.$transaction` at lines 680–730 (after the `playerTrack.upsert` and before `invalidateMatcherCachesForRally`):

```typescript
// Add import at top of file:
import { reresolveRallyGt } from './actionGroundTruthService.js';

// Inside the transaction, after the playerTrack.upsert, before invalidateMatcherCachesForRally:
await reresolveRallyGt(
  tx,
  rallyId,
  trackerResult.rawPositions as unknown as Array<{
    frameNumber: number;
    trackId: number;
    bbox: { x1: number; y1: number; x2: number; y2: number };
  }>,
);
```

- [ ] **Step 4: Run tests, verify pass**

```
cd api && npx vitest run tests/actionGroundTruthRetrack.test.ts
cd api && npx vitest run tests/saveTrackingResult.test.ts
```

Expected: PASS — both files green.

- [ ] **Step 5: Commit**

```bash
git add api/src/services/playerTrackingService.ts api/tests/actionGroundTruthRetrack.test.ts
git commit -m "feat(action-gt): re-resolve GT inside saveTrackingResult transaction"
```

---

### Task 6: Drop GT copy from `matchAnalysisService.ts`

**Files:**
- Modify: `api/src/services/matchAnalysisService.ts:755-780`

- [ ] **Step 1: Delete the manual GT propagation block**

In `matchAnalysisService.ts`, find the block currently at ~line 755:

```typescript
const actionGt = (playerTrack.actionGroundTruthJson ?? null) as Array<Record<string, unknown>> | null;
// ...
...(actionGt ? { actionGroundTruthJson: actionGt as unknown as Prisma.InputJsonValue } : {}),
```

Delete both lines. The new table owns its own lifecycle; match-analysis no longer touches GT.

- [ ] **Step 2: Type-check**

Run: `cd api && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 3: Run match-analysis tests**

Run: `cd api && npx vitest run tests/matchAnalysisPlanning.test.ts tests/matchAnalysisTrigger.test.ts`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add api/src/services/matchAnalysisService.ts
git commit -m "refactor(action-gt): drop manual GT propagation in match-analysis"
```

---

## Phase 4 — Slicing (split + merge + reindex)

### Task 7: Drop GT from `rallySlicing.ts`, move GT row ops into split/merge services

**Files:**
- Modify: `api/src/services/rallySlicing.ts`
- Modify: `api/src/services/splitRallyService.ts` (or wherever split lives — find via `grep "slicePlayerTrack"`)
- Modify: `api/src/services/mergeRalliesService.ts`
- Modify: `api/tests/rallySlicing.test.ts`, `api/tests/splitRally.test.ts`, `api/tests/mergeRallies.test.ts`

- [ ] **Step 1: Drop `actionGroundTruthJson` from slicing types**

In `rallySlicing.ts`:
- Remove `actionGroundTruthJson: AnyFrameEntry[] | null` from `SlicePlayerTrackInput` (line 20)
- Remove `actGt` usage in `slicePlayerTrack` (lines 156, 186, 198)
- Remove the `actionGroundTruthJson` branch in `concatPlayerTracks` (lines 107–109, 141)

(`groundTruthJson` for player tracking stays — that's a different field.)

- [ ] **Step 2: Update slicing tests**

In `api/tests/rallySlicing.test.ts`, remove any assertions on `actionGroundTruthJson` in the input/output of `concatPlayerTracks` and `slicePlayerTrack`. Replace input objects to no longer set that field.

- [ ] **Step 3: Run rallySlicing tests, verify still pass**

Run: `cd api && npx vitest run tests/rallySlicing.test.ts`
Expected: PASS.

- [ ] **Step 4: Add GT row ops to split service**

In the split service (the file that calls `slicePlayerTrack` and creates the two child rallies), after the children are persisted, add:

```typescript
import { Prisma } from '@prisma/client';

// Inside the same transaction that creates the children:
// firstChildId, secondChildId are known; firstEndFrame and secondStartFrame are the cut points.
await tx.$executeRaw`
  INSERT INTO rally_action_ground_truth (id, rally_id, frame, action, snapshot_bbox_x1, snapshot_bbox_y1, snapshot_bbox_x2, snapshot_bbox_y2, snapshot_ball_x, snapshot_ball_y, snapshot_team, snapshot_track_id, resolved_track_id, resolved_at, resolved_source, created_at, updated_at, created_by)
  SELECT gen_random_uuid(), ${firstChildId}::uuid, frame, action, snapshot_bbox_x1, snapshot_bbox_y1, snapshot_bbox_x2, snapshot_bbox_y2, snapshot_ball_x, snapshot_ball_y, snapshot_team, snapshot_track_id, resolved_track_id, resolved_at, resolved_source, created_at, NOW(), created_by
  FROM rally_action_ground_truth
  WHERE rally_id = ${parentRallyId}::uuid AND frame < ${firstEndFrame}
`;
await tx.$executeRaw`
  INSERT INTO rally_action_ground_truth (id, rally_id, frame, action, snapshot_bbox_x1, snapshot_bbox_y1, snapshot_bbox_x2, snapshot_bbox_y2, snapshot_ball_x, snapshot_ball_y, snapshot_team, snapshot_track_id, resolved_track_id, resolved_at, resolved_source, created_at, updated_at, created_by)
  SELECT gen_random_uuid(), ${secondChildId}::uuid, frame - ${secondStartFrame}, action, snapshot_bbox_x1, snapshot_bbox_y1, snapshot_bbox_x2, snapshot_bbox_y2, snapshot_ball_x, snapshot_ball_y, snapshot_team, snapshot_track_id, resolved_track_id, resolved_at, resolved_source, created_at, NOW(), created_by
  FROM rally_action_ground_truth
  WHERE rally_id = ${parentRallyId}::uuid AND frame >= ${secondStartFrame}
`;
// Frames in [firstEndFrame, secondStartFrame) are discarded by virtue of not matching either filter.
// Parent's GT rows cascade-delete with the parent rally.
```

- [ ] **Step 5: Extend split test**

In `api/tests/splitRally.test.ts`, add a case that:
1. Creates a rally with `PlayerTrack` and a GT label at frame 30 + another at frame 70.
2. Splits at `firstEndMs/secondStartMs` corresponding to frame boundary 40/50.
3. Asserts: first child has the frame-30 row (frame=30), second child has frame-70 row shifted to frame=20, parent's rows are gone (cascade).

- [ ] **Step 6: Run split tests, verify pass**

Run: `cd api && npx vitest run tests/splitRally.test.ts`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add api/src/services/rallySlicing.ts api/src/services/splitRallyService.ts api/tests/rallySlicing.test.ts api/tests/splitRally.test.ts
git commit -m "feat(action-gt): partition GT rows on rally split, drop JSON-array path"
```

---

### Task 8: Merge — copy parent GT rows into merged rally (closes gap-loss)

**Files:**
- Modify: `api/src/services/mergeRalliesService.ts`
- Modify: `api/tests/mergeRallies.test.ts`

- [ ] **Step 1: Write the failing test (merge-with-gap preserves GT)**

In `api/tests/mergeRallies.test.ts`, add:

```typescript
it('merge-with-gap copies both parents\' GT rows into the merged rally', async () => {
  const userId = (await setupTestUser()).id;
  // Two adjacent-but-not-touching rallies on the same video:
  const a = await setupTestRallyWithTrack({ userId, frameCount: 100, startMs: 0, endMs: 3333 /* 100 frames @ 30fps */ });
  const b = await setupTestRallyWithVideo(a.videoId, userId, { frameCount: 100, startMs: 5000, endMs: 8333 });
  await saveActionGroundTruth(a.id, userId, [{ frame: 30, action: 'serve', trackId: 7 }]);
  await saveActionGroundTruth(b.id, userId, [{ frame: 60, action: 'attack', trackId: 7 }]);

  const merged = await mergeRallies(a.videoId, userId, [a.id, b.id]);

  const rows = await prisma.rallyActionGroundTruth.findMany({ where: { rallyId: merged.id }, orderBy: { frame: 'asc' } });
  expect(rows).toHaveLength(2);
  expect(rows[0]).toMatchObject({ frame: 30, action: 'SERVE' });
  // B's frame=60 shifts up by (b.startMs - a.startMs) / fps * 1000 — i.e., gap frames between A.end and B.start.
  // For 30fps, gap is (5000 - 3333) ms = ~50 frames, so merged frame = 100 (A.frameCount) + 50 (gap) + 60 = 210
  // The merged rally spans A.start → B.end so duration is 8333ms = 250 frames; assert frame in expected bucket.
  expect(rows[1].frame).toBeGreaterThan(100);
  expect(rows[1].action).toBe('ATTACK');
});
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd api && npx vitest run tests/mergeRallies.test.ts -t "merge-with-gap copies"`
Expected: FAIL — merged rally has 0 GT rows (today's bug).

- [ ] **Step 3: Add GT copy inside merge transaction**

In `mergeRalliesService.ts`, inside the same `prisma.$transaction` that creates the merged rally, after the merged rally row is created but before parents are deleted:

```typescript
// For each parent, compute frame offset = (parent.startMs - mergedStartMs) / 1000 * fps
// Then copy GT rows with frame += offset.
const fps = 30; // or read from a parent's PlayerTrack.fps when available, fallback 30
for (const parent of [parentA, parentB]) {
  const offsetFrames = Math.round((parent.startMs - mergedRally.startMs) / 1000 * fps);
  await tx.$executeRaw`
    INSERT INTO rally_action_ground_truth (id, rally_id, frame, action, snapshot_bbox_x1, snapshot_bbox_y1, snapshot_bbox_x2, snapshot_bbox_y2, snapshot_ball_x, snapshot_ball_y, snapshot_team, snapshot_track_id, resolved_track_id, resolved_at, resolved_source, created_at, updated_at, created_by)
    SELECT gen_random_uuid(), ${mergedRally.id}::uuid, frame + ${offsetFrames}, action, snapshot_bbox_x1, snapshot_bbox_y1, snapshot_bbox_x2, snapshot_bbox_y2, snapshot_ball_x, snapshot_ball_y, snapshot_team, snapshot_track_id, resolved_track_id, resolved_at, resolved_source, created_at, NOW(), created_by
    FROM rally_action_ground_truth
    WHERE rally_id = ${parent.id}::uuid
  `;
}
// Parents' rows cascade-delete with the parents.
```

After the merged rally's `PlayerTrack` has been (re)tracked, the resolver will pass over all migrated rows.

- [ ] **Step 4: Run test, verify pass**

Run: `cd api && npx vitest run tests/mergeRallies.test.ts -t "merge-with-gap copies"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/src/services/mergeRalliesService.ts api/tests/mergeRallies.test.ts
git commit -m "feat(action-gt): copy parent GT rows into merged rally"
```

---

### Task 9: Rally bounds reindex updates GT row frames

**Files:**
- Modify: `api/src/services/playerTrackingService.ts:1602-1620`
- Modify: `api/tests/retrackOnExtend.test.ts`

- [ ] **Step 1: Add a test for the reindex case**

In `api/tests/retrackOnExtend.test.ts`, add:

```typescript
it('shifts GT frames when rally bounds change', async () => {
  const userId = (await setupTestUser()).id;
  const r = await setupTestRallyWithTrack({ userId, frameCount: 100, startMs: 1000, endMs: 4333 });
  await saveActionGroundTruth(r.id, userId, [{ frame: 50, action: 'serve', trackId: 7 }]);

  // Extend start earlier by 1s (30 frames at 30fps) → frame 50 should become 80
  await updateRallyBounds(r.id, { startMs: 0, endMs: 4333 });

  const row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId: r.id } });
  expect(row.frame).toBe(80);
});
```

- [ ] **Step 2: Replace the inline shift with a SQL update**

In `playerTrackingService.ts`, replace lines 1602–1606 (the `actionGtData` shift block) and the corresponding update at 1617–1619 with a direct SQL UPDATE inside the same transaction:

```typescript
// Inside the existing transaction for reindex:
await tx.$executeRaw`
  UPDATE rally_action_ground_truth
  SET frame = frame - ${deltaFrames}, updated_at = NOW()
  WHERE rally_id = ${rallyId}::uuid
`;
await tx.$executeRaw`
  DELETE FROM rally_action_ground_truth
  WHERE rally_id = ${rallyId}::uuid
    AND (frame < 0 OR frame >= ${newFrameCount})
`;
```

- [ ] **Step 3: Run test, verify pass**

Run: `cd api && npx vitest run tests/retrackOnExtend.test.ts`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add api/src/services/playerTrackingService.ts api/tests/retrackOnExtend.test.ts
git commit -m "feat(action-gt): shift GT frames in DB on rally bounds reindex"
```

---

## Phase 5 — Backfill

### Task 10: `migrate_action_gt.py` backfill CLI

**Files:**
- Create: `analysis/rallycut/cli/commands/migrate_action_gt.py`
- Create: `analysis/tests/integration/test_migrate_action_gt.py`

- [ ] **Step 1: Write the failing test**

`analysis/tests/integration/test_migrate_action_gt.py`:

```python
import json
import uuid
import pytest
import psycopg
from rallycut.cli.commands.migrate_action_gt import backfill_video

# Test uses the same Postgres test DB as the existing analysis integration suite.
# Helpers below mirror the patterns in test_detect_contacts_snapshot.py.

def make_video_with_gt(conn, *, action_gt, raw_positions, ball_positions, primary_track_ids, team_assignments):
    video_id = str(uuid.uuid4())
    rally_id = str(uuid.uuid4())
    pt_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("INSERT INTO videos (id, user_id, session_id, status, created_at, team_assignments_json) VALUES (%s, %s, %s, 'DETECTED', NOW(), %s::jsonb)",
                    (video_id, str(uuid.uuid4()), str(uuid.uuid4()), json.dumps(team_assignments)))
        cur.execute("INSERT INTO rallies (id, video_id, start_ms, end_ms, status, created_at, updated_at) VALUES (%s, %s, 0, 3333, 'CONFIRMED', NOW(), NOW())",
                    (rally_id, video_id))
        cur.execute("""INSERT INTO player_tracks (id, rally_id, status, frame_count, raw_positions_json, ball_positions_json, primary_track_ids, action_ground_truth_json)
                       VALUES (%s, %s, 'COMPLETED', 100, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)""",
                    (pt_id, rally_id, json.dumps(raw_positions), json.dumps(ball_positions),
                     json.dumps(primary_track_ids), json.dumps(action_gt)))
    return video_id, rally_id

def test_backfill_snapshot_exact(pg_conn):
    video_id, rally_id = make_video_with_gt(
        pg_conn,
        action_gt=[{"frame": 50, "action": "serve", "trackId": 7}],
        raw_positions=[{"frameNumber": 50, "trackId": 7, "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.3}, "confidence": 0.9}],
        ball_positions=[{"frameNumber": 50, "x": 0.5, "y": 0.5}],
        primary_track_ids=[7],
        team_assignments={"7": "A"},
    )
    report = backfill_video(pg_conn, video_id)
    assert report["snapshot_exact"] == 1
    assert report["unresolved"] == 0
    with pg_conn.cursor() as cur:
        cur.execute("SELECT snapshot_bbox_x1, snapshot_team, resolved_source FROM rally_action_ground_truth WHERE rally_id = %s", (rally_id,))
        row = cur.fetchone()
    assert row == (0.1, "A", "SNAPSHOT_EXACT")

def test_backfill_unresolved_when_no_position_at_frame(pg_conn):
    video_id, rally_id = make_video_with_gt(
        pg_conn,
        action_gt=[{"frame": 50, "action": "serve", "trackId": 7}],
        raw_positions=[],  # no positions at all
        ball_positions=[],
        primary_track_ids=[7],
        team_assignments={"7": "A"},
    )
    report = backfill_video(pg_conn, video_id)
    assert report["unresolved"] == 1
    with pg_conn.cursor() as cur:
        cur.execute("SELECT snapshot_bbox_x1, snapshot_track_id, resolved_source FROM rally_action_ground_truth WHERE rally_id = %s", (rally_id,))
        row = cur.fetchone()
    assert row == (None, 7, "UNRESOLVED")

def test_backfill_idempotent(pg_conn):
    video_id, _ = make_video_with_gt(
        pg_conn,
        action_gt=[{"frame": 50, "action": "serve", "trackId": 7}],
        raw_positions=[{"frameNumber": 50, "trackId": 7, "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.3}, "confidence": 0.9}],
        ball_positions=[],
        primary_track_ids=[7],
        team_assignments={"7": "A"},
    )
    r1 = backfill_video(pg_conn, video_id)
    r2 = backfill_video(pg_conn, video_id)
    assert r1["snapshot_exact"] == 1
    assert r2["already_present"] == 1  # second run skips via ON CONFLICT
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd analysis && pytest tests/integration/test_migrate_action_gt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rallycut.cli.commands.migrate_action_gt'`.

- [ ] **Step 3: Implement the backfill CLI**

`analysis/rallycut/cli/commands/migrate_action_gt.py`:

```python
"""Backfill rally_action_ground_truth from PlayerTrack.action_ground_truth_json.

For every PlayerTrack with a non-null action_ground_truth_json, write one row per
label into rally_action_ground_truth. Snapshot the bbox from raw_positions_json at
(frame, trackId) when present, otherwise leave bbox null and write UNRESOLVED.

Idempotent via ON CONFLICT (rally_id, frame, action) DO NOTHING. Also dumps a JSON
backup to backups/action_gt_pre_cutover_<ISO>.json before any writes.
"""
import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg

ACTION_MAP = {"serve": "SERVE", "receive": "RECEIVE", "set": "SET",
              "attack": "ATTACK", "block": "BLOCK", "dig": "DIG"}


def _bbox_at(raw_positions: list[dict], frame: int, track_id: int) -> dict | None:
    for p in raw_positions or []:
        if p.get("frameNumber") == frame and p.get("trackId") == track_id:
            return p.get("bbox")
    return None


def _ball_at(ball_positions: list[dict], frame: int) -> tuple[float, float] | None:
    for b in ball_positions or []:
        if b.get("frameNumber") == frame:
            return b.get("x"), b.get("y")
    return None


def _team_for(track_id: int, primary_track_ids: list[int], team_assignments: dict[str, str]) -> str | None:
    if not primary_track_ids or track_id not in primary_track_ids:
        return None
    t = (team_assignments or {}).get(str(track_id))
    return t if t in ("A", "B") else None


def backfill_video(conn: psycopg.Connection, video_id: str) -> dict[str, int]:
    """Backfill all rallies under a video. Returns a per-outcome counter."""
    report = {"snapshot_exact": 0, "unresolved": 0, "already_present": 0, "skipped_no_gt": 0}

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute("SELECT team_assignments_json FROM videos WHERE id = %s", (video_id,))
        video_row = cur.fetchone()
        if not video_row:
            return report
        team_assignments = video_row.get("team_assignments_json") or {}

        cur.execute("""
            SELECT pt.rally_id, pt.action_ground_truth_json AS gt,
                   pt.raw_positions_json AS raw, pt.ball_positions_json AS ball,
                   pt.primary_track_ids AS pti
              FROM player_tracks pt
              JOIN rallies r ON r.id = pt.rally_id
             WHERE r.video_id = %s AND pt.action_ground_truth_json IS NOT NULL
        """, (video_id,))
        tracks = cur.fetchall()

    for tr in tracks:
        rally_id = tr["rally_id"]
        labels = tr["gt"] or []
        raw = tr["raw"] or []
        ball = tr["ball"] or []
        pti = tr["pti"] or []
        if not labels:
            report["skipped_no_gt"] += 1
            continue
        for label in labels:
            frame = label.get("frame")
            action = label.get("action")
            track_id = label.get("trackId")
            if action not in ACTION_MAP or frame is None:
                continue
            bbox = _bbox_at(raw, frame, track_id) if track_id is not None else None
            ball_xy = _ball_at(ball, frame)
            team = _team_for(track_id, pti, team_assignments) if track_id is not None else None
            resolved_track_id = track_id if bbox is not None else None
            resolved_source = "SNAPSHOT_EXACT" if bbox is not None else "UNRESOLVED"

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rally_action_ground_truth
                      (id, rally_id, frame, action,
                       snapshot_bbox_x1, snapshot_bbox_y1, snapshot_bbox_x2, snapshot_bbox_y2,
                       snapshot_ball_x, snapshot_ball_y, snapshot_team, snapshot_track_id,
                       resolved_track_id, resolved_at, resolved_source,
                       created_at, updated_at, created_by)
                    VALUES (gen_random_uuid(), %s::uuid, %s, %s::"ActionLabel",
                            %s, %s, %s, %s,
                            %s, %s, %s::"GtTeam", %s,
                            %s, NOW(), %s::"ResolveSource",
                            NOW(), NOW(), NULL)
                    ON CONFLICT (rally_id, frame, action) DO NOTHING
                    RETURNING id
                """, (
                    rally_id, frame, ACTION_MAP[action],
                    bbox.get("x1") if bbox else None,
                    bbox.get("y1") if bbox else None,
                    bbox.get("x2") if bbox else None,
                    bbox.get("y2") if bbox else None,
                    ball_xy[0] if ball_xy else label.get("ballX"),
                    ball_xy[1] if ball_xy else label.get("ballY"),
                    team, track_id,
                    resolved_track_id, resolved_source,
                ))
                inserted = cur.fetchone()
                if inserted is None:
                    report["already_present"] += 1
                elif resolved_source == "SNAPSHOT_EXACT":
                    report["snapshot_exact"] += 1
                else:
                    report["unresolved"] += 1

        conn.commit()

    return report


def dump_backup(conn: psycopg.Connection, out_path: str) -> int:
    """Write a JSON file mapping rallyId -> actionGroundTruthJson before any column drop."""
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute("SELECT rally_id, action_ground_truth_json FROM player_tracks WHERE action_ground_truth_json IS NOT NULL")
        rows = cur.fetchall()
    out = {r["rally_id"]: r["action_ground_truth_json"] for r in rows}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    return len(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=os.environ["DATABASE_URL"])
    ap.add_argument("--video-id", help="Backfill a single video (omit for all)")
    ap.add_argument("--backup", action="store_true", help="Dump pre-cutover JSON backup")
    args = ap.parse_args()

    with psycopg.connect(args.db_url) as conn:
        if args.backup:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            n = dump_backup(conn, f"backups/action_gt_pre_cutover_{ts}.json")
            print(f"[backup] wrote {n} rallies to backups/action_gt_pre_cutover_{ts}.json", flush=True)

        if args.video_id:
            video_ids = [args.video_id]
        else:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT r.video_id FROM player_tracks pt
                      JOIN rallies r ON r.id = pt.rally_id
                     WHERE pt.action_ground_truth_json IS NOT NULL
                """)
                video_ids = [r[0] for r in cur.fetchall()]

        total = {"snapshot_exact": 0, "unresolved": 0, "already_present": 0, "skipped_no_gt": 0}
        for i, vid in enumerate(video_ids, 1):
            r = backfill_video(conn, vid)
            for k, v in r.items():
                total[k] += v
            print(f"[{i}/{len(video_ids)}] {vid}: {r}", flush=True)

        print(f"[done] totals: {total}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd analysis && pytest tests/integration/test_migrate_action_gt.py -v`
Expected: PASS — 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/cli/commands/migrate_action_gt.py analysis/tests/integration/test_migrate_action_gt.py
git commit -m "feat(action-gt): backfill CLI with bbox snapshot + JSON pre-cutover backup"
```

---

### Task 11: Run backfill in dev + spot-check the report

**Files:** none (operational task)

- [ ] **Step 1: Run backfill against local dev DB with backup**

Run:
```
cd analysis && python -m rallycut.cli.commands.migrate_action_gt --backup
```

Expected output: per-video line like `[7/63] <uuid>: {'snapshot_exact': 28, 'unresolved': 2, 'already_present': 0, 'skipped_no_gt': 0}` and a final `[done] totals: {...}` line. Backup file in `backups/action_gt_pre_cutover_<ts>.json`.

- [ ] **Step 2: Spot-check the report**

Expected on the panel (gigi/wawa/cece):
- `snapshot_exact` accounts for the bulk of labels.
- `unresolved` is small (≤5 — the absent-server cases from `adaptive_candidate_window_v30_2026_05_11.md`).
- `skipped_no_gt` is 0 (we filtered for `IS NOT NULL`).

If `unresolved` is unexpectedly high (>10% of total), STOP and investigate before proceeding. The expected absent-server count is documented in the memory file.

- [ ] **Step 3: Spot-check one rally end-to-end**

Pick the gigi/5b6f0474 rally referenced in the memory file (`adaptive_candidate_window_v30_2026_05_11.md`). Verify the synthetic-serve frame 48 came through as `UNRESOLVED` (no bbox at that frame, trackId not in `primary_track_ids`), and the other contacts came through as `SNAPSHOT_EXACT`.

```sql
SELECT frame, action, snapshot_bbox_x1 IS NULL AS bbox_null, resolved_source
  FROM rally_action_ground_truth
 WHERE rally_id = '<gigi/5b6f0474 rally uuid>'
 ORDER BY frame;
```

- [ ] **Step 4: No commit** (operational verification only)

---

## Phase 6 — Python readers

### Task 12: Switch `train.py` to the new table

**Files:**
- Modify: `analysis/rallycut/cli/commands/train.py:1637-1695`

- [ ] **Step 1: Replace the GT-loading query**

In `train.py`, find the function near line 1637 that queries `pt.action_ground_truth_json`. Replace the SQL with:

```python
def load_action_gt(conn, video_ids: list[str]) -> dict[str, list[dict]]:
    """Returns rallyId -> list of {frame, action, resolved_track_id, snapshot_ball_x, snapshot_ball_y}."""
    out: dict[str, list[dict]] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT r.id AS rally_id,
                   gt.frame, gt.action, gt.resolved_track_id,
                   gt.snapshot_ball_x, gt.snapshot_ball_y, gt.snapshot_team
              FROM rally_action_ground_truth gt
              JOIN rallies r ON r.id = gt.rally_id
             WHERE r.video_id = ANY(%s)
               AND gt.resolved_track_id IS NOT NULL
             ORDER BY r.id, gt.frame
        """, (video_ids,))
        for row in cur.fetchall():
            rid = row[0]
            out.setdefault(rid, []).append({
                "frame": row[1],
                "action": row[2].lower(),
                "trackId": row[3],
                "ballX": row[4],
                "ballY": row[5],
                "team": row[6],
            })
    return out
```

Update every call site in `train.py` that consumed the old JSON shape (`label["trackId"]`, `label["frame"]`, `label["action"]`, `label["ballX"]`, `label["ballY"]`) to use the new dict.

- [ ] **Step 2: Run training smoke test on panel videos**

```
cd analysis && python -u -m rallycut.cli.commands.train --smoke --videos gigi/5b6f0474,wawa/06c13117,cece/<id>
```

Expected: training set count matches pre-migration count (modulo `UNRESOLVED` filter). Print the count before commit so the regression is documented.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/cli/commands/train.py
git commit -m "refactor(action-gt): train.py reads from new rally_action_ground_truth table"
```

---

### Task 13: Switch `restore.py` to write to the new table

**Files:**
- Modify: `analysis/rallycut/training/restore.py:511-568`

- [ ] **Step 1: Replace the upsert block**

Find the block that does `INSERT INTO player_tracks (..., action_ground_truth_json, ...)` and `UPDATE player_tracks SET action_ground_truth_json = ...`. Replace it with row inserts on the new table:

```python
def restore_action_gt(conn, rally_id: str, labels: list[dict]) -> None:
    """labels: list of {frame, action, trackId, ballX, ballY} from a backup dump."""
    action_map = {"serve": "SERVE", "receive": "RECEIVE", "set": "SET",
                  "attack": "ATTACK", "block": "BLOCK", "dig": "DIG"}
    with conn.cursor() as cur:
        for label in labels:
            action = label.get("action")
            if action not in action_map or label.get("frame") is None:
                continue
            cur.execute("""
                INSERT INTO rally_action_ground_truth (
                    id, rally_id, frame, action,
                    snapshot_ball_x, snapshot_ball_y, snapshot_track_id,
                    resolved_track_id, resolved_at, resolved_source,
                    created_at, updated_at)
                VALUES (gen_random_uuid(), %s::uuid, %s, %s::"ActionLabel",
                        %s, %s, %s,
                        %s, NOW(), %s::"ResolveSource",
                        NOW(), NOW())
                ON CONFLICT (rally_id, frame, action) DO UPDATE SET
                    snapshot_ball_x = EXCLUDED.snapshot_ball_x,
                    snapshot_ball_y = EXCLUDED.snapshot_ball_y,
                    snapshot_track_id = EXCLUDED.snapshot_track_id,
                    resolved_track_id = EXCLUDED.resolved_track_id,
                    resolved_at = NOW(),
                    resolved_source = EXCLUDED.resolved_source
            """, (
                rally_id, label["frame"], action_map[action],
                label.get("ballX"), label.get("ballY"), label.get("trackId"),
                label.get("trackId"),  # restored labels treat trackId as resolved
                "SNAPSHOT_EXACT" if label.get("trackId") is not None else "UNRESOLVED",
            ))
        conn.commit()
```

Bboxes are not recovered from backups (the old backup format never had them); restore lands `SNAPSHOT_EXACT` with bbox-null but `resolvedTrackId = trackId`. The next `saveTrackingResult` will re-resolve and may move things to `IOU_MATCH` if the tracking is unchanged.

- [ ] **Step 2: Run a small restore round-trip in dev**

Pick one rally with action GT. Dump it, delete the rows, restore. Verify row count matches.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/training/restore.py
git commit -m "refactor(action-gt): restore.py writes to new table"
```

---

### Task 14: Switch eval and audit scripts

**Files:**
- Modify: `analysis/scripts/eval_action_detection.py`
- Modify: `analysis/scripts/audit_action_gt_vs_click_gt.py`
- Modify: `analysis/scripts/measure_attribution_fixture.py`
- Modify: `analysis/scripts/measure_attribution_fresh_gt.py`
- Modify: `analysis/scripts/measure_attribution_permuted.py`
- Modify: `analysis/scripts/measure_attribution_team_chain_ab.py`
- Modify: `analysis/scripts/measure_attribution_post_v13.py`
- Modify: `analysis/scripts/audit_action_sequence_anomalies.py`
- Modify: `analysis/scripts/check_gt_integrity.py`
- Modify: `analysis/scripts/_smoke_snapshot_gt.py`
- Modify: `analysis/scripts/diagnose_offscreen_server_candidates.py`
- Modify: `analysis/scripts/audit_fixture_gt.py`
- Modify: `analysis/scripts/audit_panel_gt_alignment.py`
- Modify: `analysis/scripts/diagnose_dig_serve_receive.py`
- Modify: `analysis/scripts/diagnose_dedup_pattern.py`
- Modify: `analysis/scripts/probe_block_mstcn_signal.py`
- Modify: `analysis/scripts/probe_pose_contact_proximity.py`
- Modify: `analysis/scripts/probe_db_state_drift.py`
- Modify: `analysis/scripts/repair_orphaned_gt.py`
- Modify: `analysis/scripts/render_attribution_clips.py`
- Modify: `analysis/scripts/render_attribution_errors.py`
- Modify: `analysis/scripts/render_attribution_errors_v2.py`
- Modify: `analysis/scripts/analyze_block_tp_vs_fp.py`
- Modify: `analysis/scripts/find_overpass_attacks.py`
- Modify: `analysis/scripts/sweep_block_rule_fp.py`
- Modify: `analysis/scripts/sweep_offscreen_gate_fp.py`
- Modify: `analysis/scripts/sweep_block_insert_fp.py`
- Modify: `analysis/scripts/feasibility_visual_attribution.py`
- Modify: `analysis/scripts/eval_visual_attribution.py`
- Modify: `analysis/scripts/measure_serve_prepend_clean_ab.py`
- Modify: `analysis/scripts/post_retrack_measurements.py`
- Modify: `analysis/scripts/diagnose_blocks_fleet.py`
- Modify: `analysis/scripts/measure_fleet_action_f1.py`
- Modify: `analysis/scripts/restore_canonical_drift_backups.py`
- Modify: `analysis/scripts/extract_contact_features.py`
- Modify: `analysis/scripts/regenerate_contact_candidates.py`
- Modify: `analysis/scripts/train_temporal_attribution.py`
- Modify: `analysis/scripts/calibrate_serve_burst_threshold.py`
- Modify: `analysis/scripts/analyze_within_track_bimodality.py`
- Modify: `analysis/scripts/probe_ball_motion_signal.py`
- Modify: `analysis/scripts/eval_court_side.py`
- Modify: `analysis/scripts/build_failure_review.py`
- Modify: `analysis/scripts/build_ab_verdict_ui.py`
- Modify: `analysis/scripts/audit_side_switch_gt.py`
- Modify: `analysis/scripts/backfill_action_gt_trackid.py`
- Modify: `analysis/scripts/resave_ball_for_action_gt.py`
- Modify: `analysis/scripts/batch_match_players.py`
- Modify: `analysis/scripts/backfill_ball_tracking.py`
- Modify: `analysis/scripts/probe_ball_blur.py`
- Modify: `analysis/scripts/probe_ball_blur_v2.py`
- Modify: `analysis/scripts/extract_blur_crops.py`
- Modify: `analysis/scripts/phase0_resolve_fixtures.py`
- Modify: `analysis/scripts/phase0_lock_baseline.py`
- Modify: `analysis/tests/integration/test_detect_contacts_snapshot.py`
- Create: `analysis/rallycut/training/action_gt_query.py` — shared loader

- [ ] **Step 1: Add a shared loader**

`analysis/rallycut/training/action_gt_query.py`:

```python
"""Single source of truth for reading rally_action_ground_truth in analysis scripts.

Every script that used to read `pt.action_ground_truth_json` should import from here.
"""
from __future__ import annotations
import psycopg


def load_for_rallies(conn: psycopg.Connection, rally_ids: list[str], *, include_unresolved: bool = False) -> dict[str, list[dict]]:
    """Returns rallyId -> list of label dicts."""
    where = "gt.rally_id = ANY(%s)"
    if not include_unresolved:
        where += " AND gt.resolved_track_id IS NOT NULL"
    out: dict[str, list[dict]] = {}
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT gt.rally_id, gt.frame, gt.action, gt.resolved_track_id,
                   gt.snapshot_ball_x, gt.snapshot_ball_y, gt.snapshot_team,
                   gt.snapshot_track_id, gt.snapshot_bbox_x1, gt.snapshot_bbox_y1,
                   gt.snapshot_bbox_x2, gt.snapshot_bbox_y2, gt.resolved_source
              FROM rally_action_ground_truth gt
             WHERE {where}
             ORDER BY gt.rally_id, gt.frame
        """, (rally_ids,))
        for row in cur.fetchall():
            rid = row[0]
            out.setdefault(rid, []).append({
                "frame": row[1],
                "action": row[2].lower(),
                "trackId": row[3],
                "ballX": row[4],
                "ballY": row[5],
                "team": row[6],
                "snapshotTrackId": row[7],
                "snapshotBbox": (row[8], row[9], row[10], row[11]) if row[8] is not None else None,
                "resolvedSource": row[12],
            })
    return out


def load_for_videos(conn: psycopg.Connection, video_ids: list[str], *, include_unresolved: bool = False) -> dict[str, list[dict]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM rallies WHERE video_id = ANY(%s)", (video_ids,))
        rally_ids = [r[0] for r in cur.fetchall()]
    return load_for_rallies(conn, rally_ids, include_unresolved=include_unresolved)
```

- [ ] **Step 2: Bulk-rewrite scripts**

For each script in the file list above, replace its inline `action_ground_truth_json` query with a call to `load_for_rallies` or `load_for_videos`. The dict shape returned matches the old JSON-array shape closely (`frame`, `action`, `trackId`, `ballX`, `ballY`) so most call sites can stay identical.

Scripts that audit GT integrity (`audit_action_gt_vs_click_gt.py`, `check_gt_integrity.py`, `audit_panel_gt_alignment.py`) call with `include_unresolved=True` and inspect `resolvedSource`.

Backfill helpers that no longer make sense after this migration (`backfill_action_gt_trackid.py`, `resave_ball_for_action_gt.py`) should be moved to `analysis/scripts/_archive_day123/` rather than deleted, mirroring the existing archive pattern.

- [ ] **Step 3: Run the smoke test that exists**

```
cd analysis && pytest tests/integration/test_detect_contacts_snapshot.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add analysis/rallycut/training/action_gt_query.py analysis/scripts/ analysis/tests/integration/test_detect_contacts_snapshot.py
git commit -m "refactor(action-gt): scripts read via load_for_rallies helper, drop JSON-array path"
```

---

### Task 15: Delete the GT rewrite block in `remap_track_ids.py`

**Files:**
- Modify: `analysis/rallycut/cli/commands/remap_track_ids.py:690-915`

- [ ] **Step 1: Delete the action_ground_truth_json rewrite**

In `remap_track_ids.py`, find the block that loads `pt.action_ground_truth_json` and computes `gt_labels` with remapped track IDs at lines ~694 and ~912. Delete:
- The `pt.action_ground_truth_json` field from the SELECT.
- The construction of remapped `gt_labels`.
- The `changes["action_ground_truth_json"] = json.dumps(gt_labels)` assignment.

The new model owns this: any change in track IDs after remap will be caught by the next `saveTrackingResult` triggering the re-resolver.

- [ ] **Step 2: Run remap-track-ids smoke against one rally**

```
cd analysis && python -u -m rallycut.cli.commands.remap_track_ids --rally-id <some-test-rally>
```

Expected: no errors. The rally's `rally_action_ground_truth` rows are untouched by remap.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/cli/commands/remap_track_ids.py
git commit -m "refactor(action-gt): remap-track-ids no longer rewrites GT (resolver owns it)"
```

---

## Phase 7 — Web

### Task 16: Update `web/src/services/api.ts` types

**Files:**
- Modify: `web/src/services/api.ts:1860-1872`

- [ ] **Step 1: Update the interface and the endpoints**

In `api.ts`:

```typescript
export type ResolveSource = 'SNAPSHOT_EXACT' | 'IOU_MATCH' | 'NEAREST_CENTER' | 'MANUAL' | 'UNRESOLVED';

export interface ActionGroundTruthLabel {
  id: string;
  frame: number;
  action: 'serve' | 'receive' | 'set' | 'attack' | 'block' | 'dig';

  // Stable snapshot
  snapshotBboxX1: number | null;
  snapshotBboxY1: number | null;
  snapshotBboxX2: number | null;
  snapshotBboxY2: number | null;
  snapshotBallX: number | null;
  snapshotBallY: number | null;
  snapshotTeam: 'A' | 'B' | null;
  snapshotTrackId: number | null;

  // Cached attribution
  resolvedTrackId: number | null;
  resolvedSource: ResolveSource;
  resolvedAt: string | null;
}

// Reattach endpoint:
export async function reattachActionLabel(rallyId: string, rowId: string, resolvedTrackId: number): Promise<void> {
  const res = await fetch(`${API_URL}/v1/rallies/${rallyId}/action-ground-truth/${rowId}/reattach`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Visitor-Id': visitorId() },
    body: JSON.stringify({ resolvedTrackId }),
  });
  if (!res.ok) throw new Error(`reattach failed: ${res.status}`);
}
```

The save payload sent to the API stays the same shape (`{frame, action, trackId?, ballX?, ballY?}`) — the server denormalizes. The response now contains the new fields.

- [ ] **Step 2: Type-check**

Run: `cd web && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add web/src/services/api.ts
git commit -m "feat(action-gt): web client types for new GT payload + reattach"
```

---

### Task 17: Update `playerTrackingStore.ts` + `gtLabelDisplay.ts`

**Files:**
- Modify: `web/src/stores/playerTrackingStore.ts`
- Modify: `web/src/utils/gtLabelDisplay.ts`

- [ ] **Step 1: Adapt the store to the new payload**

In `playerTrackingStore.ts`, the existing `actionGroundTruth: Record<string, ActionGroundTruthLabel[]>` keeps its shape but the entries now have the extra fields. Update `loadActionGroundTruth` and `saveActionGroundTruth` to pass through. Add a `reattachActionLabel(rallyId, rowId, resolvedTrackId)` action that calls the new endpoint and updates local state.

- [ ] **Step 2: Update `gtLabelDisplay.ts` to use `resolvedTrackId`**

The display pid is currently derived from `playerTrackId` or `trackId`. Replace with `resolvedTrackId` lookup against `appliedFullMapping`:

```typescript
import type { ActionGroundTruthLabel, MatchAnalysis } from '@/services/api';

export function displayPidFor(label: ActionGroundTruthLabel, match: MatchAnalysis | null): number | null {
  if (label.resolvedTrackId == null) return null; // UNRESOLVED → render as ghost, no pid badge
  const mapping = match?.appliedFullMapping ?? {};
  return mapping[String(label.resolvedTrackId)] ?? null;
}
```

- [ ] **Step 3: Type-check**

Run: `cd web && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add web/src/stores/playerTrackingStore.ts web/src/utils/gtLabelDisplay.ts
git commit -m "feat(action-gt): store + display use resolvedTrackId"
```

---

### Task 18: Ghost overlay + reattach UI for UNRESOLVED rows

**Files:**
- Modify: `web/src/components/ActionOverlay.tsx`
- Modify: `web/src/components/ActionLabelingMode.tsx`

- [ ] **Step 1: Render UNRESOLVED rows as ghost bboxes in `ActionOverlay`**

Where the overlay iterates GT labels and renders a colored badge per player, add a branch:

```tsx
{labels.map(label => {
  if (label.resolvedSource === 'UNRESOLVED' && label.snapshotBboxX1 != null) {
    return (
      <GhostBbox
        key={label.id}
        x1={label.snapshotBboxX1!}
        y1={label.snapshotBboxY1!}
        x2={label.snapshotBboxX2!}
        y2={label.snapshotBboxY2!}
        badge="UNRESOLVED"
        onClick={() => openReattach(label)}
      />
    );
  }
  return /* existing colored-bbox path */;
})}
```

`GhostBbox` is a thin styled component (dashed outline, neutral color) — add it inline if the project doesn't have one.

- [ ] **Step 2: Add a reattach prompt to `ActionLabelingMode`**

When a labeler clicks a tracked player while a `GhostBbox` is selected, call `reattachActionLabel(rallyId, label.id, clickedTrackId)`. Show a confirmation toast.

- [ ] **Step 3: Manually verify in dev**

```
make dev
```

Open a rally that has at least one `UNRESOLVED` row (gigi/5b6f0474 from the panel works after backfill). Verify:
- Ghost bbox renders.
- Clicking another player triggers reattach.
- After reattach, the label shows a normal colored badge.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/ActionOverlay.tsx web/src/components/ActionLabelingMode.tsx
git commit -m "feat(action-gt): ghost overlay + reattach UI for UNRESOLVED labels"
```

---

## Phase 8 — Audit CLI

### Task 19: `audit_action_gt.py`

**Files:**
- Create: `analysis/rallycut/cli/commands/audit_action_gt.py`
- Create: `analysis/tests/integration/test_audit_action_gt.py`

- [ ] **Step 1: Write the failing test**

`analysis/tests/integration/test_audit_action_gt.py`:

```python
import json, uuid
import pytest, psycopg
from rallycut.cli.commands.audit_action_gt import audit_video

def _seed(conn, video_id, rally_id, rows):
    """rows: list of (frame, action, resolved_source, resolved_track_id)"""
    with conn.cursor() as cur:
        for frame, action, source, rtid in rows:
            cur.execute("""
                INSERT INTO rally_action_ground_truth
                  (id, rally_id, frame, action, resolved_track_id, resolved_at, resolved_source, created_at, updated_at)
                VALUES (gen_random_uuid(), %s::uuid, %s, %s::"ActionLabel", %s, NOW(), %s::"ResolveSource", NOW(), NOW())
            """, (rally_id, frame, action, rtid, source))

def test_audit_counts_per_source(pg_conn, sample_video_with_rally):
    video_id, rally_id = sample_video_with_rally  # fixture mirroring make_video_with_gt
    _seed(pg_conn, video_id, rally_id, [
        (10, "SERVE", "SNAPSHOT_EXACT", 7),
        (20, "RECEIVE", "IOU_MATCH", 11),
        (30, "ATTACK", "UNRESOLVED", None),
    ])
    report = audit_video(pg_conn, video_id)
    assert report["counts"] == {"SNAPSHOT_EXACT": 1, "IOU_MATCH": 1, "NEAREST_CENTER": 0, "MANUAL": 0, "UNRESOLVED": 1}
    assert len(report["unresolved_rallies"]) == 1
```

- [ ] **Step 2: Run test, verify failure**

Run: `cd analysis && pytest tests/integration/test_audit_action_gt.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the audit CLI**

`analysis/rallycut/cli/commands/audit_action_gt.py`:

```python
"""Audit rally_action_ground_truth: count rows per resolved_source per video."""
import argparse, os, sys
from collections import Counter
import psycopg


def audit_video(conn: psycopg.Connection, video_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT gt.rally_id, gt.resolved_source
              FROM rally_action_ground_truth gt
              JOIN rallies r ON r.id = gt.rally_id
             WHERE r.video_id = %s
        """, (video_id,))
        rows = cur.fetchall()
    counts = Counter({s: 0 for s in ("SNAPSHOT_EXACT", "IOU_MATCH", "NEAREST_CENTER", "MANUAL", "UNRESOLVED")})
    per_rally = Counter()
    for rally_id, source in rows:
        counts[source] += 1
        if source == "UNRESOLVED":
            per_rally[rally_id] += 1
    return {
        "video_id": video_id,
        "counts": dict(counts),
        "unresolved_rallies": [{"rally_id": rid, "count": n} for rid, n in per_rally.items() if n > 0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=os.environ["DATABASE_URL"])
    ap.add_argument("--video-id")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not args.video_id and not args.all:
        ap.error("specify --video-id or --all")

    with psycopg.connect(args.db_url) as conn:
        if args.all:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT r.video_id FROM rally_action_ground_truth gt JOIN rallies r ON r.id = gt.rally_id")
                video_ids = [r[0] for r in cur.fetchall()]
        else:
            video_ids = [args.video_id]
        for i, vid in enumerate(video_ids, 1):
            r = audit_video(conn, vid)
            print(f"[{i}/{len(video_ids)}] {vid}: counts={r['counts']} unresolved_rallies={len(r['unresolved_rallies'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd analysis && pytest tests/integration/test_audit_action_gt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/cli/commands/audit_action_gt.py analysis/tests/integration/test_audit_action_gt.py
git commit -m "feat(action-gt): audit-action-gt CLI reports per-source counts per video"
```

---

## Phase 9 — Cutover

### Task 20: Pre-cutover CI guard (grep check)

**Files:**
- Modify: `.github/workflows/<existing-ci.yml>` or a project Makefile target — find via `grep -l "tsc --noEmit" .github/workflows/`

- [ ] **Step 1: Add a grep guard step**

Add a CI step:

```yaml
- name: No action_ground_truth_json references outside backups
  run: |
    if grep -rn "actionGroundTruthJson\|action_ground_truth_json" \
        api/src/ web/src/ analysis/rallycut/ analysis/scripts/ \
        --include="*.ts" --include="*.tsx" --include="*.py" \
        | grep -v "migrate_action_gt.py" \
        | grep -v "_archive_day123" \
        | grep -v "restore.py"  # keep the rollback path
    then
      echo "ERROR: stale references to action_ground_truth_json remain"
      exit 1
    fi
```

If the project uses a Makefile or pre-commit hook instead, add the same check there. The intent is: a build that still references the old column blocks the cutover commit.

- [ ] **Step 2: Run the grep manually**

Run the command above from the repo root and verify it exits 0 (no stale references after Tasks 12–15).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/<file>
git commit -m "ci(action-gt): block builds that still reference action_ground_truth_json"
```

---

### Task 21: Drop the `action_ground_truth_json` column

**Files:**
- Modify: `api/prisma/schema.prisma`
- Create: `api/prisma/migrations/<ts>_drop_player_track_action_gt/migration.sql`

- [ ] **Step 1: Re-run backfill + backup in dev**

```
cd analysis && python -m rallycut.cli.commands.migrate_action_gt --backup
```

Expected: idempotent — all rows already present, fresh backup written.

- [ ] **Step 2: Remove the field from the Prisma model**

In `api/prisma/schema.prisma`, delete line ~338:

```
actionGroundTruthJson     Json?             @map("action_ground_truth_json")
```

- [ ] **Step 3: Generate the migration**

Run: `cd api && npx prisma migrate dev --name drop_player_track_action_gt`

Expected: a DROP COLUMN migration is generated and applied locally.

- [ ] **Step 4: Verify the table no longer has the column**

```sql
\d player_tracks
```

Expected: no `action_ground_truth_json` column.

- [ ] **Step 5: Run all touched test suites end-to-end**

```
cd api && npx vitest run
cd analysis && pytest
cd web && npx tsc --noEmit
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add api/prisma/schema.prisma api/prisma/migrations/
git commit -m "feat(action-gt): drop player_tracks.action_ground_truth_json (cutover)"
```

---

### Task 22: Eval non-regression gate

**Files:** none (operational task — verify measurements)

- [ ] **Step 1: Run the panel attribution eval before and after**

Run the canonical attribution measurement against the panel videos (gigi/wawa/cece) and compare PERMUTED PID accuracy pre/post migration:

```
cd analysis && python -u scripts/measure_attribution_fixture.py --panel
```

Expected: per-video PERMUTED PID accuracy moves by 0 percentage points (the spec change MUST NOT move measurements — only the storage location of the labels).

If it does move, STOP. Verify the backfill report's `unresolved` count is in line with expectations from the spec's "Migration" section. Investigate any unexplained drift before merging.

- [ ] **Step 2: Run the fleet F1 eval**

```
cd analysis && python -u scripts/measure_fleet_action_f1.py
```

Expected: fleet F1 within ±0.001 of `0.927` (the documented baseline in `contact_detection_ceiling_2026_05_11.md`).

- [ ] **Step 3: Document the eval in a workstream report**

If both gates pass, save a one-page report to `analysis/reports/action_gt_decouple_<ts>.md`:

```markdown
# Action GT Decouple — Cutover Verification (<date>)

- Panel PERMUTED PID accuracy: pre=<X.X>, post=<Y.Y> (Δ=<...>)
- Fleet action F1: pre=<...>, post=<...> (Δ=<...>)
- Migration report: <snapshot_exact, iou_match, nearest_center, unresolved> counts
- Backup file: backups/action_gt_pre_cutover_<ts>.json
```

Commit the report.

```bash
git add analysis/reports/action_gt_decouple_*.md
git commit -m "report(action-gt): cutover verification — no regression on panel/fleet"
```

---

## Self-review

**Spec coverage:**

| Spec section            | Task(s)               |
| ----------------------- | --------------------- |
| Schema                  | 1                     |
| Write path — labeler UI | 3, 4                  |
| Write path — re-resolver| 2 (pure), 5 (wiring)  |
| Write path — slicing    | 7, 8, 9               |
| Write path — match-analysis cleanup | 6         |
| Read path — UI          | 16, 17, 18            |
| Read path — Python      | 12, 13, 14            |
| Read path — remap cleanup | 15                  |
| Migration — Prisma      | 1                     |
| Migration — backfill CLI| 10, 11                |
| Migration — cutover     | 20, 21                |
| Audit                   | 19                    |
| Testing — eval gate     | 22                    |

**Placeholder scan:** no "TBD"/"TODO"/"implement later"/"fill in details" strings. All steps contain actual code or commands. The two operational tasks (11, 22) are explicit verifications with stop-conditions.

**Type consistency:**
- `resolveGtRow` defined in Task 2, used in Task 3 — types match (`GtRowInput`, `Candidate`, `ResolveResult`).
- `reresolveRallyGt` defined in Task 3, called in Task 5 — signature matches `(tx, rallyId, rawPositions)`.
- `ActionGroundTruthLabel` shape on the wire is consistent across api.ts (Task 16), store (Task 17), and overlay (Task 18).
- Prisma enum names `ActionLabel`, `ResolveSource`, `GtTeam` consistent across schema (Task 1), service (Task 3), backfill (Task 10), audit (Task 19), restore (Task 13).
- The save-payload contract (UI → API) keeps the legacy `{frame, action, trackId?, ballX?, ballY?}` shape — server denormalizes. Both 16 and 17 confirm this.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-12-action-gt-decouple.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
