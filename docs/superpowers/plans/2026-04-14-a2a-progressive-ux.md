# Project A2a — Progressive UX & Debounced Match Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewire the analysis orchestrator so the user sees editable rallies the instant detection finishes, while tracking streams in the background; move match-analysis out of the Modal webhook path into a 5-second-idle debounce gated on batch completion; collapse the `AnalysisPhase` state machine to match the new flow; extend stale-job recovery to detection as well as tracking; block mid-batch rally creation with 409 (A2b lifts this).

**Architecture:**
- Shared `staleJobRecovery` helper deduplicates the 10-minute stale-timeout logic currently split between `qualityService.ts` and `batchTrackingService.ts`, and extends it to `RallyDetectionJob`.
- `handleTrackingBatchComplete` webhook stops auto-triggering `runMatchAnalysis`; a new fire-and-forget `POST /v1/videos/:id/trigger-match-analysis` endpoint owns the trigger.
- Frontend `analysisStore` owns the 5-second idle debounce, reset by rally CRUD events forwarded from `editorStore`; match-analysis fires exactly once after (batch complete ∧ idle ≥ 5s).
- `AnalysisPhase` collapses to `idle → preflight → preflight_gate → detecting → ready_tracking → match_analyzing → done | error`. `ready_tracking` is a non-blocking indicator — the editor UI is usable throughout.
- Mid-batch rally creation returns 409 `{ code: 'CREATE_DURING_TRACKING' }`; UPDATE/DELETE continue to work as today. A2b will replace 409 with the append-rally endpoint.

**Tech Stack:** TypeScript, Express 4, Prisma 6, PostgreSQL, Zustand 5, Vitest 2 (api-side only — web-side changes verified via `tsc --noEmit` + manual E2E per `web/CLAUDE.md`).

---

## Scope check

- **Out of scope (A2b, separate plan):** `WebhookDelivery` idempotency model, `BatchTrackingJob.appendRally()`, `POST /v1/batch-tracking/:jobId/append-rally`, `PlayerTrack.needsRetrack` flag, `markNeedsRetrack()`, node-cron sweeper. A2a blocks create-during-tracking with 409 instead of queueing; A2b replaces that with append-rally.
- **Out of scope (Project B):** split/extend rally semantics, match-analysis delta recomputation.
- **Out of scope (Project C):** auto-rotate tilted clips, stricter non-VB rejection.

---

## File structure

**New files:**
- `api/src/services/staleJobRecovery.ts` — shared stale-timeout helpers for `BatchTrackingJob` and `RallyDetectionJob`.
- `api/tests/staleJobRecovery.test.ts`
- `api/tests/matchAnalysisTrigger.test.ts`
- `api/tests/createDuringTracking.test.ts`
- `api/tests/trackingBatchComplete.test.ts` — proves the webhook no longer triggers match-analysis.

**Modified files:**
- `api/src/services/qualityService.ts` — replace inline stale-timeout block (lines 99–160) with helper call.
- `api/src/services/batchTrackingService.ts` — delete local `expireStaleJobs` (lines 30–54), import from helper.
- `api/src/services/modalTrackingService.ts` — delete the `runMatchAnalysis(video_id)` call inside `handleTrackingBatchComplete` (lines 166–174).
- `api/src/services/matchAnalysisService.ts` — export a `triggerMatchAnalysis(videoId, userId)` action (idempotent: no-op if already running, lock via a simple in-memory `Set`).
- `api/src/routes/videos.ts` — add `POST /v1/videos/:id/trigger-match-analysis` route; add 409 guard to rally-create path (find the exact route during Task 5).
- `api/src/routes/sessions.ts` (or wherever `sync-state` lives) — add matching 409 guard in the create-rally branch of the reconciliation loop.
- `web/src/stores/analysisStore.ts` — phase union, transitions, debounce, new API call.
- `web/src/stores/editorStore.ts` — emit rally-CRUD notifications to `analysisStore`.
- `web/src/services/api.ts` — add `triggerMatchAnalysis(videoId)` client call.
- `web/src/components/AnalysisPipeline.tsx` — UI labels for new phase names.
- `web/src/components/` — any other file that references `'quality_check'`, `'quality_warning'`, `'tracking'`, or `'completing'` (grep as part of Task 8).

**No schema migration in A2a.** All changes are code-only; persisted analysis-store version bumps to force-reset resumable pipelines across deploy.

---

## Task 1: Shared stale-job-recovery helper

**Files:**
- Create: `api/src/services/staleJobRecovery.ts`
- Create: `api/tests/staleJobRecovery.test.ts`
- Modify: `api/src/services/batchTrackingService.ts:30-54` (delete local function + import from helper)
- Modify: `api/src/services/qualityService.ts:99-160` (replace inline block with helper call)

- [ ] **Step 1: Write failing test**

Create `api/tests/staleJobRecovery.test.ts`:

```typescript
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { prisma } from '../src/lib/prisma';
import {
  expireStaleBatchTrackingJobs,
  STALE_PROGRESS_TIMEOUT_MS,
} from '../src/services/staleJobRecovery';

describe('expireStaleBatchTrackingJobs', () => {
  const videoId = 'vid-stale-test';
  const userId = 'user-stale-test';

  beforeEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { videoId } });
  });

  afterEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { videoId } });
    vi.useRealTimers();
  });

  it('marks PROCESSING jobs with no progress in >10min as FAILED', async () => {
    const stale = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS - 60_000);
    const job = await prisma.batchTrackingJob.create({
      data: {
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 3,
        completedRallies: 0,
        failedRallies: 0,
        lastProgressAt: stale,
        createdAt: stale,
      },
    });
    await expireStaleBatchTrackingJobs(videoId);
    const refreshed = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('FAILED');
    expect(refreshed.error).toMatch(/Timed out/i);
    expect(refreshed.completedAt).not.toBeNull();
  });

  it('leaves PROCESSING jobs with recent progress alone', async () => {
    const fresh = new Date(Date.now() - 60_000);
    const job = await prisma.batchTrackingJob.create({
      data: {
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 3,
        completedRallies: 1,
        failedRallies: 0,
        lastProgressAt: fresh,
        createdAt: fresh,
      },
    });
    await expireStaleBatchTrackingJobs(videoId);
    const refreshed = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('PROCESSING');
  });

  it('does not touch COMPLETED jobs even if lastProgressAt is ancient', async () => {
    const ancient = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS * 10);
    const job = await prisma.batchTrackingJob.create({
      data: {
        videoId,
        userId,
        status: 'COMPLETED',
        totalRallies: 3,
        completedRallies: 3,
        failedRallies: 0,
        lastProgressAt: ancient,
        createdAt: ancient,
        completedAt: ancient,
      },
    });
    await expireStaleBatchTrackingJobs(videoId);
    const refreshed = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('COMPLETED');
  });
});
```

- [ ] **Step 2: Run test — verify RED**

Run: `cd api && npm test -- staleJobRecovery`
Expected: FAIL — `Cannot find module '../src/services/staleJobRecovery'`.

- [ ] **Step 3: Create the helper**

Create `api/src/services/staleJobRecovery.ts`:

```typescript
import { prisma } from '../lib/prisma.js';

export const STALE_PROGRESS_TIMEOUT_MS = 10 * 60 * 1000;

/**
 * Mark stale PROCESSING/PENDING batch tracking jobs as FAILED.
 * Stale = lastProgressAt older than STALE_PROGRESS_TIMEOUT_MS.
 */
export async function expireStaleBatchTrackingJobs(videoId: string): Promise<number> {
  const cutoff = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS);
  const { count } = await prisma.batchTrackingJob.updateMany({
    where: {
      videoId,
      status: { in: ['PENDING', 'PROCESSING'] },
      lastProgressAt: { lt: cutoff },
    },
    data: {
      status: 'FAILED',
      completedAt: new Date(),
      error: 'Timed out — no progress for 10 minutes (server likely restarted)',
    },
  });
  if (count > 0) {
    console.log(`[STALE_JOB] Expired ${count} batch-tracking job(s) for video ${videoId}`);
  }
  return count;
}

/**
 * Mark stale PENDING/RUNNING detection jobs as FAILED.
 * Detection jobs don't have a lastProgressAt column — we treat createdAt as the progress floor.
 */
export async function expireStaleDetectionJobs(videoId: string): Promise<number> {
  const cutoff = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS);
  const { count } = await prisma.rallyDetectionJob.updateMany({
    where: {
      videoId,
      status: { in: ['PENDING', 'RUNNING'] },
      createdAt: { lt: cutoff },
    },
    data: {
      status: 'FAILED',
      completedAt: new Date(),
      errorMessage: 'Timed out — detection did not complete within 10 minutes',
    },
  });
  if (count > 0) {
    console.log(`[STALE_JOB] Expired ${count} detection job(s) for video ${videoId}`);
  }
  return count;
}
```

> **Note for executor:** If `RallyDetectionJob` lacks a `completedAt` or `errorMessage` field, adjust the `data` object to the actual column names — grep `prisma.rallyDetectionJob` usages in `api/src/services/` for the shape. Do **not** add a migration; if the field you want is truly absent, fall back to updating only `status` and log the reason via `console.warn`.

- [ ] **Step 4: Run test — verify GREEN**

Run: `cd api && npm test -- staleJobRecovery`
Expected: all three tests PASS.

- [ ] **Step 5: Wire helper into batchTrackingService**

Edit `api/src/services/batchTrackingService.ts` — **remove** lines 30–54 (the local `expireStaleJobs` function + its `STALE_PROGRESS_TIMEOUT_MS` constant). Replace the single call site (search for `await expireStaleJobs(videoId)`) with the import:

```typescript
import { expireStaleBatchTrackingJobs } from './staleJobRecovery.js';
// ... at the call site:
await expireStaleBatchTrackingJobs(videoId);
```

- [ ] **Step 6: Wire helper into qualityService**

Edit `api/src/services/qualityService.ts` — replace the inline block at lines 124–142 (the `if (batchJob && (batchJob.status === 'PROCESSING' ...` block) with:

```typescript
await expireStaleBatchTrackingJobs(videoId);
batchJob = await prisma.batchTrackingJob.findFirst({
  where: { videoId },
  orderBy: { createdAt: 'desc' },
});
```

Import at top: `import { expireStaleBatchTrackingJobs } from './staleJobRecovery.js';`

Also delete the local `STALE_JOB_TIMEOUT_MS` constant if it appears in this file.

- [ ] **Step 7: Typecheck + full test run**

Run: `cd api && npx tsc --noEmit && npm test`
Expected: zero type errors, all tests pass (including prior qualityService.test.ts and preflightPreview.test.ts).

- [ ] **Step 8: Commit**

```bash
git add api/src/services/staleJobRecovery.ts \
        api/tests/staleJobRecovery.test.ts \
        api/src/services/batchTrackingService.ts \
        api/src/services/qualityService.ts
git commit -m "refactor(api): extract shared staleJobRecovery helper"
```

---

## Task 2: Detection stale-timeout wiring

**Files:**
- Modify: `api/src/services/qualityService.ts` — call `expireStaleDetectionJobs(videoId)` inside `getAnalysisPipelineStatus` before reading detection status.
- Modify: `api/tests/staleJobRecovery.test.ts` — add a detection-job test case.

- [ ] **Step 1: Extend test file**

Append to `api/tests/staleJobRecovery.test.ts`:

```typescript
import { expireStaleDetectionJobs } from '../src/services/staleJobRecovery';

describe('expireStaleDetectionJobs', () => {
  const videoId = 'vid-det-stale-test';

  beforeEach(async () => {
    await prisma.rallyDetectionJob.deleteMany({ where: { videoId } });
  });
  afterEach(async () => {
    await prisma.rallyDetectionJob.deleteMany({ where: { videoId } });
  });

  it('marks RUNNING jobs older than 10min as FAILED', async () => {
    const stale = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS - 60_000);
    const job = await prisma.rallyDetectionJob.create({
      data: { videoId, status: 'RUNNING', createdAt: stale },
    });
    await expireStaleDetectionJobs(videoId);
    const refreshed = await prisma.rallyDetectionJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('FAILED');
  });

  it('leaves recent RUNNING jobs alone', async () => {
    const fresh = new Date(Date.now() - 30_000);
    const job = await prisma.rallyDetectionJob.create({
      data: { videoId, status: 'RUNNING', createdAt: fresh },
    });
    await expireStaleDetectionJobs(videoId);
    const refreshed = await prisma.rallyDetectionJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('RUNNING');
  });
});
```

> **Note for executor:** The exact required-field set for `rallyDetectionJob.create` varies by schema. If Prisma rejects the above, inspect `api/prisma/schema.prisma` for `RallyDetectionJob` and fill in required non-null fields (likely `modelVariant`, `userId`, etc.) before running.

- [ ] **Step 2: Run test — verify RED**

Run: `cd api && npm test -- staleJobRecovery`
Expected: new detection tests PASS already (helper exists from Task 1). If they fail because `getAnalysisPipelineStatus` isn't exercising them, that's fine — we're only testing the helper here.

- [ ] **Step 3: Wire into getAnalysisPipelineStatus**

In `api/src/services/qualityService.ts`, at the top of `getAnalysisPipelineStatus(videoId)` (before any detection-status read), add:

```typescript
await expireStaleDetectionJobs(videoId);
```

Import already added in Task 1 Step 6 — just extend the existing import:

```typescript
import { expireStaleBatchTrackingJobs, expireStaleDetectionJobs } from './staleJobRecovery.js';
```

- [ ] **Step 4: Run full api suite**

Run: `cd api && npx tsc --noEmit && npm test`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add api/src/services/qualityService.ts \
        api/src/services/staleJobRecovery.ts \
        api/tests/staleJobRecovery.test.ts
git commit -m "feat(api): expire stale detection jobs in analysis-pipeline-status"
```

---

## Task 3: Decouple match-analysis from batch-complete webhook

**Files:**
- Create: `api/tests/trackingBatchComplete.test.ts`
- Modify: `api/src/services/modalTrackingService.ts:166-174` — delete the `runMatchAnalysis` block.

- [ ] **Step 1: Write failing test**

Create `api/tests/trackingBatchComplete.test.ts`:

```typescript
import { afterEach, describe, expect, it, vi } from 'vitest';
import { handleTrackingBatchComplete } from '../src/services/modalTrackingService';
import * as matchAnalysisService from '../src/services/matchAnalysisService';
import { prisma } from '../src/lib/prisma';

describe('handleTrackingBatchComplete', () => {
  const videoId = 'vid-batch-test';
  const userId = 'user-batch-test';
  const jobId = 'job-batch-test';

  afterEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { id: jobId } });
    vi.restoreAllMocks();
  });

  it('does not invoke runMatchAnalysis when batch completes (A2a: debounced on client)', async () => {
    await prisma.batchTrackingJob.create({
      data: {
        id: jobId,
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 2,
        completedRallies: 1,
        failedRallies: 0,
      },
    });
    const runSpy = vi.spyOn(matchAnalysisService, 'runMatchAnalysis').mockResolvedValue(undefined);

    await handleTrackingBatchComplete({
      batch_job_id: jobId,
      video_id: videoId,
      status: 'completed',
      completed_rallies: 2,
      failed_rallies: 0,
    });

    expect(runSpy).not.toHaveBeenCalled();
    const updated = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: jobId } });
    expect(updated.status).toBe('COMPLETED');
    expect(updated.completedRallies).toBe(2);
  });
});
```

- [ ] **Step 2: Run — verify RED**

Run: `cd api && npm test -- trackingBatchComplete`
Expected: FAIL — `runMatchAnalysis` was called (current behavior).

- [ ] **Step 3: Delete auto-trigger**

Edit `api/src/services/modalTrackingService.ts` — delete lines 166–174 (the `if (completed_rallies > 0) { try { await runMatchAnalysis ... } catch ... }` block). Replace with a one-line comment:

```typescript
// Match analysis is now triggered client-side via POST /v1/videos/:id/trigger-match-analysis
// after a 5-second idle window following batch completion. See Project A2a.
```

Also remove the now-unused `runMatchAnalysis` import from this file.

- [ ] **Step 4: Run — verify GREEN**

Run: `cd api && npm test -- trackingBatchComplete`
Expected: PASS.

- [ ] **Step 5: Typecheck + full suite**

Run: `cd api && npx tsc --noEmit && npm test`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add api/src/services/modalTrackingService.ts api/tests/trackingBatchComplete.test.ts
git commit -m "refactor(api): remove auto match-analysis trigger from tracking-batch webhook"
```

---

## Task 4: Manual match-analysis trigger endpoint

**Files:**
- Modify: `api/src/services/matchAnalysisService.ts` — export `triggerMatchAnalysis(videoId, userId)` with an in-memory running-set guard.
- Modify: `api/src/routes/videos.ts` — register `POST /v1/videos/:id/trigger-match-analysis`.
- Create: `api/tests/matchAnalysisTrigger.test.ts`

- [ ] **Step 1: Write failing test**

Create `api/tests/matchAnalysisTrigger.test.ts`:

```typescript
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import request from 'supertest';
import app from '../src/index'; // adjust to actual app export path
import * as matchAnalysisService from '../src/services/matchAnalysisService';
import { prisma } from '../src/lib/prisma';

describe('POST /v1/videos/:id/trigger-match-analysis', () => {
  const videoId = 'vid-trigger-test';
  const userId = 'user-trigger-test';
  const visitorId = 'visitor-trigger-test';

  beforeEach(async () => {
    await prisma.user.upsert({
      where: { id: userId },
      update: {},
      create: { id: userId, email: `${userId}@test.local` },
    });
    await prisma.anonymousIdentity.upsert({
      where: { visitorId },
      update: { userId },
      create: { visitorId, userId },
    });
    await prisma.video.create({
      data: {
        id: videoId,
        userId,
        originalFilename: 'test.mp4',
        s3Key: `${videoId}.mp4`,
        status: 'DETECTED',
      },
    });
  });

  afterEach(async () => {
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.anonymousIdentity.deleteMany({ where: { visitorId } });
    await prisma.user.deleteMany({ where: { id: userId } });
    vi.restoreAllMocks();
  });

  it('returns 202 and invokes runMatchAnalysis fire-and-forget', async () => {
    const runSpy = vi
      .spyOn(matchAnalysisService, 'runMatchAnalysis')
      .mockResolvedValue(undefined);

    const res = await request(app)
      .post(`/v1/videos/${videoId}/trigger-match-analysis`)
      .set('X-Visitor-Id', visitorId);

    expect(res.status).toBe(202);
    expect(res.body).toMatchObject({ status: 'processing' });
    // Fire-and-forget — allow the next tick for the promise to schedule
    await new Promise((r) => setImmediate(r));
    expect(runSpy).toHaveBeenCalledWith(videoId);
  });

  it('returns 409 if an analysis is already running for this video', async () => {
    vi.spyOn(matchAnalysisService, 'runMatchAnalysis').mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 500)),
    );

    const first = await request(app)
      .post(`/v1/videos/${videoId}/trigger-match-analysis`)
      .set('X-Visitor-Id', visitorId);
    expect(first.status).toBe(202);

    const second = await request(app)
      .post(`/v1/videos/${videoId}/trigger-match-analysis`)
      .set('X-Visitor-Id', visitorId);
    expect(second.status).toBe(409);
    expect(second.body).toMatchObject({ code: 'MATCH_ANALYSIS_IN_PROGRESS' });
  });
});
```

> **Note for executor:** Exact `app` import and user-fixture helpers vary. Before writing the test, look at `api/tests/qualityService.test.ts` or `preflightPreview.test.ts` for the established supertest pattern and user-seed helper. Use whatever pattern those tests already use. The test above is illustrative — align field names and imports to the real setup before running RED.

- [ ] **Step 2: Run — verify RED**

Run: `cd api && npm test -- matchAnalysisTrigger`
Expected: FAIL — route 404s.

- [ ] **Step 3: Add service guard**

Edit `api/src/services/matchAnalysisService.ts`. Add at top of file (after imports):

```typescript
const runningVideos = new Set<string>();

export function isMatchAnalysisRunning(videoId: string): boolean {
  return runningVideos.has(videoId);
}

/**
 * Fire-and-forget trigger. Returns immediately after scheduling runMatchAnalysis.
 * If another call is already running for this videoId, returns false (caller should 409).
 */
export function triggerMatchAnalysis(videoId: string): boolean {
  if (runningVideos.has(videoId)) return false;
  runningVideos.add(videoId);
  // Fire-and-forget; swallow errors so the lock is always released.
  runMatchAnalysis(videoId)
    .catch((err) => {
      console.error(`[MATCH_ANALYSIS] ${videoId} failed:`, err);
    })
    .finally(() => {
      runningVideos.delete(videoId);
    });
  return true;
}
```

> **Note on in-memory lock:** This only protects against self-concurrency within a single API instance. If the API scales horizontally, two pods could race; A2b's `WebhookDelivery`/job model is the durable solution. A2a accepts this limitation because match-analysis is idempotent-ish (last write wins on `matchAnalysisJson`).

- [ ] **Step 4: Register route**

Find the rally/video mutation route cluster in `api/src/routes/videos.ts` (near `POST /v1/videos/:id/track-all-rallies`). Add:

```typescript
router.post('/videos/:id/trigger-match-analysis', requireUser, async (req, res, next) => {
  try {
    const { id: videoId } = req.params;
    const userId = req.userId!;
    const video = await prisma.video.findUnique({ where: { id: videoId } });
    if (!video) return res.status(404).json({ code: 'VIDEO_NOT_FOUND' });
    if (video.userId !== userId) return res.status(403).json({ code: 'FORBIDDEN' });

    const started = triggerMatchAnalysis(videoId);
    if (!started) {
      return res.status(409).json({ code: 'MATCH_ANALYSIS_IN_PROGRESS' });
    }
    return res.status(202).json({ status: 'processing' });
  } catch (err) {
    next(err);
  }
});
```

Import: `import { triggerMatchAnalysis } from '../services/matchAnalysisService.js';`

> **Note for executor:** Match the middleware names (`requireUser`, error-handler pattern) to whatever is already used by neighbouring routes in `videos.ts`. Do not introduce a new middleware convention.

- [ ] **Step 5: Run — verify GREEN**

Run: `cd api && npm test -- matchAnalysisTrigger`
Expected: both tests PASS.

- [ ] **Step 6: Full suite**

Run: `cd api && npx tsc --noEmit && npm test`

- [ ] **Step 7: Commit**

```bash
git add api/src/services/matchAnalysisService.ts \
        api/src/routes/videos.ts \
        api/tests/matchAnalysisTrigger.test.ts
git commit -m "feat(api): add POST /v1/videos/:id/trigger-match-analysis endpoint"
```

---

## Task 5: 409 guard for create-during-tracking

**Files:**
- Find: the rally-create entry points. Two paths exist: (a) an explicit `POST` route if any, (b) the create branch inside `POST /v1/sessions/:sessionId/sync-state`.
- Modify: both paths to return/raise 409 when the latest `BatchTrackingJob` for that video is `PROCESSING` or `PENDING`.
- Create: `api/tests/createDuringTracking.test.ts`

- [ ] **Step 1: Locate rally-create paths**

Run: `cd api && grep -rn "prisma.rally.create" src/`

Expected: one or more hits in `src/services/` or `src/routes/`. Also: `grep -rn "createRally" src/`. Note every call site; the 409 guard must be added to each that is reachable via the HTTP surface (not internal helpers).

- [ ] **Step 2: Write failing integration test**

Create `api/tests/createDuringTracking.test.ts`:

```typescript
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import request from 'supertest';
import app from '../src/index';
import { prisma } from '../src/lib/prisma';

describe('create rally during tracking → 409', () => {
  const videoId = 'vid-cdt-test';
  const userId = 'user-cdt-test';
  const visitorId = 'visitor-cdt-test';
  const sessionId = 'session-cdt-test';
  const jobId = 'job-cdt-test';

  beforeEach(async () => {
    // Seed user, session, video (use existing fixture helpers if any).
    // ...
    await prisma.batchTrackingJob.create({
      data: {
        id: jobId,
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 2,
        completedRallies: 0,
        failedRallies: 0,
      },
    });
  });

  afterEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { id: jobId } });
    // tear down fixtures...
  });

  it('rejects new rally in sync-state while batch job is PROCESSING', async () => {
    const res = await request(app)
      .post(`/v1/sessions/${sessionId}/sync-state`)
      .set('X-Visitor-Id', visitorId)
      .send({
        rallies: [{ id: `${videoId}_rally_new`, videoId, startMs: 1000, endMs: 4000, status: 'CONFIRMED' }],
        highlights: [],
      });
    expect(res.status).toBe(409);
    expect(res.body).toMatchObject({ code: 'CREATE_DURING_TRACKING' });
  });
});
```

> **Note for executor:** Fill in user/session fixture creation using the same helpers used elsewhere in `api/tests/`. If no helpers exist, copy the seed pattern from the first existing `*.test.ts` file that seeds a session.

- [ ] **Step 3: Run — verify RED**

Run: `cd api && npm test -- createDuringTracking`
Expected: FAIL.

- [ ] **Step 4: Add guard helper**

Append to `api/src/services/batchTrackingService.ts`:

```typescript
/**
 * Returns true if the latest BatchTrackingJob for this video is active (PENDING | PROCESSING).
 * A2a uses this to reject mid-batch rally creation with 409 CREATE_DURING_TRACKING.
 * A2b will replace the 409 with an append-rally enqueue.
 */
export async function isBatchTrackingActive(videoId: string): Promise<boolean> {
  const latest = await prisma.batchTrackingJob.findFirst({
    where: { videoId },
    orderBy: { createdAt: 'desc' },
    select: { status: true },
  });
  return latest?.status === 'PENDING' || latest?.status === 'PROCESSING';
}
```

- [ ] **Step 5: Apply guard at every rally-create call site**

For each HTTP-reachable call site located in Step 1, wrap the create with:

```typescript
import { isBatchTrackingActive } from '../services/batchTrackingService.js';

if (await isBatchTrackingActive(videoId)) {
  throw new ConflictError('CREATE_DURING_TRACKING', 'New rallies cannot be added while tracking is running. Please wait for tracking to finish.');
}
```

> **Note for executor:** Use whatever error-class pattern already lives in `api/src/middleware/` or `src/lib/errors.ts`. If a `ConflictError` class doesn't exist, raise an `Error` with `.status = 409` using the project's existing convention. Do not introduce a new error abstraction.

For the `sync-state` reconciliation loop specifically: the check must fire **only for newly-created rallies** (rallies in the request payload whose ID has no matching DB row). Updates and deletes are unaffected.

- [ ] **Step 6: Run — verify GREEN**

Run: `cd api && npm test -- createDuringTracking`
Expected: PASS.

- [ ] **Step 7: Full suite + typecheck**

Run: `cd api && npx tsc --noEmit && npm test`

- [ ] **Step 8: Commit**

```bash
git add api/src/services/batchTrackingService.ts \
        api/src/routes/ \
        api/src/services/ \
        api/tests/createDuringTracking.test.ts
git commit -m "feat(api): reject rally creation during active tracking with 409"
```

---

## Task 6: Collapse AnalysisPhase union

**Files:**
- Modify: `web/src/stores/analysisStore.ts`

- [ ] **Step 1: Rename phase union**

Edit `web/src/stores/analysisStore.ts:16-24`:

```typescript
export type AnalysisPhase =
  | 'idle'
  | 'preflight'
  | 'preflight_gate'
  | 'detecting'
  | 'ready_tracking'
  | 'match_analyzing'
  | 'done'
  | 'error';
```

- [ ] **Step 2: Replace literal references in this file**

In `web/src/stores/analysisStore.ts`, apply these semantic renames throughout:

| Old | New |
|-----|-----|
| `'quality_check'` | `'preflight'` |
| `'quality_warning'` | `'preflight_gate'` |
| `'tracking'` | `'ready_tracking'` |
| `'completing'` | `'match_analyzing'` |

Specifically, these lines need edits (spot-check after rename):
- Line 186: `phase: 'quality_check'` → `'preflight'`
- Line 205: `phase: 'quality_warning'` → `'preflight_gate'`
- Line 236: `pipeline.phase !== 'quality_warning'` → `'preflight_gate'`
- Line 243: `phase: 'detecting'` (unchanged)
- Line 287: `phase === 'tracking'` → `'ready_tracking'`
- Line 289: `phase === 'completing'` → `'match_analyzing'`
- Line 318, 324: `phase: 'tracking'` in `resumeIfNeeded` → `'ready_tracking'`
- Lines 346–347: partialize filter — update both literals.
- `advanceAfterDetection` (line 88): `phase: 'tracking'` → `'ready_tracking'`
- `pollTracking` (line 464): `pipeline.phase !== 'tracking'` → `'ready_tracking'`
- `pollTracking` (line 476): `phase: 'completing'` → `'match_analyzing'` **— but see Task 7**, this transition moves behind the debounce.
- `completeAnalysis` references: unchanged structurally but triggered by the debounce now.
- `resumeTracking` (line 592): `phase: 'completing'` → `'match_analyzing'`

- [ ] **Step 3: Bump persist version to force-reset in-progress pipelines**

In the `persist` options block (lines 332–351), change `version: 2` to `version: 3`. The existing `migrate: () => ({ pipelines: {} })` already resets on mismatch — no other change needed.

- [ ] **Step 4: Typecheck**

Run: `cd web && npx tsc --noEmit`
Expected: zero errors inside `analysisStore.ts`. Errors elsewhere that reference the old literal values are expected and handled in Task 8.

- [ ] **Step 5: Commit (WIP — will not build cleanly until Task 8)**

```bash
git add web/src/stores/analysisStore.ts
git commit -m "refactor(web): collapse AnalysisPhase union (wip: consumers still reference old literals)"
```

---

## Task 7: Client-side debounced match-analysis

**Files:**
- Modify: `web/src/services/api.ts` — add `triggerMatchAnalysis(videoId)` fetch.
- Modify: `web/src/stores/analysisStore.ts` — add `notifyRallyEdited(videoId)` action + internal debounce timer; change `pollTracking` so batch-complete arms the timer instead of jumping straight to `match_analyzing`.
- Modify: `web/src/stores/editorStore.ts` — call `analysisStore.notifyRallyEdited(videoId)` inside `createRally` / `updateRally` / `deleteRally`.

- [ ] **Step 1: Add API client method**

Append to `web/src/services/api.ts` (near other video endpoints):

```typescript
export async function triggerMatchAnalysis(videoId: string): Promise<void> {
  const res = await fetch(`${API_URL}/v1/videos/${videoId}/trigger-match-analysis`, {
    method: 'POST',
    headers: buildHeaders(),
  });
  if (!res.ok && res.status !== 409) {
    // 409 = already running, treat as no-op (another tab triggered it)
    throw new Error(`trigger-match-analysis failed: ${res.status}`);
  }
}
```

> Match the existing `API_URL` + `buildHeaders()` helpers in this file — name may differ.

- [ ] **Step 2: Wire debounce into analysisStore**

In `web/src/stores/analysisStore.ts`, add above the store definition:

```typescript
const MATCH_ANALYSIS_DEBOUNCE_MS = 5000;
const matchAnalysisDebounceTimers: Record<string, ReturnType<typeof setTimeout>> = {};

function clearMatchAnalysisDebounce(videoId: string) {
  if (matchAnalysisDebounceTimers[videoId]) {
    clearTimeout(matchAnalysisDebounceTimers[videoId]);
    delete matchAnalysisDebounceTimers[videoId];
  }
}
```

Extend the `AnalysisState` interface with:

```typescript
interface AnalysisState {
  pipelines: Record<string, AnalysisPipeline>;

  getPipeline: (videoId: string) => AnalysisPipeline;
  startAnalysis: (videoId: string) => Promise<void>;
  dismissWarnings: (videoId: string) => void;
  cancelAnalysis: (videoId: string) => void;
  resumeIfNeeded: (videoId: string) => Promise<void>;

  /** Called by editorStore on any rally CRUD. Resets the 5s debounce that
   *  eventually fires match-analysis once batch tracking is complete. */
  notifyRallyEdited: (videoId: string) => void;
}
```

Extend `AnalysisPipeline` with a new flag:

```typescript
export interface AnalysisPipeline {
  // ... existing fields
  batchTrackingComplete?: boolean;
}
```

Implement `notifyRallyEdited` inside the store body:

```typescript
notifyRallyEdited: (videoId: string) => {
  const pipeline = get().pipelines[videoId];
  if (!pipeline || pipeline.phase !== 'ready_tracking') return;
  // Only debounce once batch has finished — otherwise the edit just queues silently
  // until batch completes (batchTrackingComplete arms the initial timer).
  if (!pipeline.batchTrackingComplete) return;
  armMatchAnalysisDebounce(videoId, set, get);
},
```

Add the armer helper above the store:

```typescript
function armMatchAnalysisDebounce(videoId: string, set: SetFn, get: GetFn) {
  clearMatchAnalysisDebounce(videoId);
  const updatePipeline = makePipelineUpdater(videoId, set);
  matchAnalysisDebounceTimers[videoId] = setTimeout(async () => {
    delete matchAnalysisDebounceTimers[videoId];
    const current = get().pipelines[videoId];
    if (!current || current.phase !== 'ready_tracking' || !current.batchTrackingComplete) return;
    updatePipeline({
      phase: 'match_analyzing',
      progress: 92,
      stepMessage: 'Generating match stats...',
    });
    try {
      const { triggerMatchAnalysis } = await import('@/services/api');
      await triggerMatchAnalysis(videoId);
      await completeAnalysis(videoId, set, get);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Match analysis failed';
      updatePipeline({ phase: 'error', error: message, stepMessage: 'Match analysis failed' });
    }
  }, MATCH_ANALYSIS_DEBOUNCE_MS);
}
```

- [ ] **Step 3: Rewire pollTracking to arm the debounce instead of transitioning immediately**

In `pollTracking` (line ~457 onward), replace the `status.status === 'completed'` branch:

Old:
```typescript
if (status.status === 'completed') {
  clearPollTimer(videoId);
  updatePipeline({
    phase: 'completing',
    progress: 90,
    stepMessage: 'Generating match stats...',
    trackingProgress: { completed: status.completedRallies ?? 0, total: status.totalRallies ?? 0 },
  });
  await completeAnalysis(videoId, set, get);
}
```

New:
```typescript
if (status.status === 'completed') {
  clearPollTimer(videoId);
  updatePipeline({
    progress: 90,
    stepMessage: 'Tracking complete. Waiting for edits to settle...',
    trackingProgress: { completed: status.completedRallies ?? 0, total: status.totalRallies ?? 0 },
    batchTrackingComplete: true,
  });
  armMatchAnalysisDebounce(videoId, set, get);
}
```

- [ ] **Step 4: Rewire resumeTracking**

In `resumeTracking` (line ~580), the `status === 'completed'` branch currently transitions straight to `'completing'` + `completeAnalysis`. Replace with:

```typescript
if (status.status === 'completed') {
  updatePipeline({
    progress: 90,
    stepMessage: 'Tracking complete. Waiting for edits to settle...',
    batchTrackingComplete: true,
  });
  if (!isStale()) armMatchAnalysisDebounce(videoId, set, get);
}
```

- [ ] **Step 5: clearPollTimer should also clear the debounce**

In `cancelAnalysis` (line 253) and `clearPollTimer` (line 120): extend both to call `clearMatchAnalysisDebounce(videoId)`.

- [ ] **Step 6: Emit from editorStore**

Open `web/src/stores/editorStore.ts`. Find the three mutation actions (`createRally`, `updateRally`, `deleteRally` — line numbers ~150+). At the end of each, after the state update and `syncService.markDirty()`, add:

```typescript
import { useAnalysisStore } from './analysisStore';
// ... inside each action, at the end:
useAnalysisStore.getState().notifyRallyEdited(videoId);
```

> If `videoId` isn't in scope at that point, derive it from the rally: `const videoId = rally.id.split('_rally_')[0]` (based on the `{matchId}_rally_{n}` convention documented in `web/CLAUDE.md`). If `matchId !== videoId` in this project, trace `activeMatchId` → video mapping via the store.

- [ ] **Step 7: Typecheck**

Run: `cd web && npx tsc --noEmit`
Expected: errors only in components still using old phase literals (Task 8 fixes these).

- [ ] **Step 8: Commit**

```bash
git add web/src/services/api.ts \
        web/src/stores/analysisStore.ts \
        web/src/stores/editorStore.ts
git commit -m "feat(web): debounce match-analysis on rally edit after batch completes"
```

---

## Task 8: Update UI phase consumers

**Files:**
- Find: every file that references the old phase literals.
- Modify: each to use the new names + (where relevant) adjust copy for `ready_tracking` ("tracking in background, keep editing") vs `match_analyzing` ("generating match stats").

- [ ] **Step 1: Locate consumers**

Run:

```bash
cd web && grep -rn "'quality_check'\|'quality_warning'\|'tracking'\|'completing'" src/ --include='*.ts' --include='*.tsx'
```

Expected hits: `AnalysisPipeline.tsx` and possibly `RallyList.tsx`, `EditorLayout.tsx`, sidebar components.

- [ ] **Step 2: Apply renames**

For every hit, replace:

| Old literal | New literal |
|-------------|-------------|
| `'quality_check'` | `'preflight'` |
| `'quality_warning'` | `'preflight_gate'` |
| `'tracking'` | `'ready_tracking'` |
| `'completing'` | `'match_analyzing'` |

In `AnalysisPipeline.tsx`, ensure the copy for `ready_tracking` reads as non-blocking (e.g., "Tracking players in background — you can keep editing rallies."). For `match_analyzing`, retain "Generating match stats…".

- [ ] **Step 3: Verify rally list is not gated on phase**

In `RallyList.tsx` (and any other editor surfaces), confirm there's no `disabled={phase === 'tracking'}` or `phase !== 'done'` guard that prevents rally edits. If such a guard exists, remove it — rally edits are allowed during `ready_tracking` (that's the whole point).

Run:

```bash
cd web && grep -rn "phase\s*[!=]==" src/components/ --include='*.tsx'
```

Inspect each hit; remove any that block rally CRUD for non-terminal phases.

- [ ] **Step 4: Typecheck + lint**

Run: `cd web && npx tsc --noEmit && npm run lint`
Expected: zero errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/
git commit -m "refactor(web): rename phase literals in UI consumers"
```

---

## Task 9: Manual E2E verification

**No code changes.** This is a checklist executed against `make dev` running locally.

- [ ] **Step 1: Start dev stack**

Run: `make dev`
Wait for api (port 3001) + web (port 3000) to print "ready".

- [ ] **Step 2: Upload a clean volleyball clip**

Browse to `http://localhost:3000`. Upload an ordinary beach-VB match. Expected: preflight-preview passes; upload starts; video row appears.

- [ ] **Step 3: Click Analyze Match**

Expected UI flow:
1. Pipeline progress shows "Checking your video…" (phase = `preflight`)
2. If warnings exist, modal "N issues found" (phase = `preflight_gate`); click "Analyze Anyway".
3. "Looking for rallies…" (phase = `detecting`).
4. **As soon as rallies appear in the sidebar, they must be editable** — try adjusting a rally's start handle on the timeline; the edit must apply (phase = `ready_tracking`, tracking progress continues).
5. When batch tracking finishes, the UI shows "Tracking complete. Waiting for edits to settle…" for ~5 seconds.
6. After 5s idle, transition to "Generating match stats…" (phase = `match_analyzing`).
7. "Analysis complete! N rallies, M players" (phase = `done`).

- [ ] **Step 4: Verify debounce**

Redo Steps 2–3. This time, after tracking completes, immediately drag a rally handle. Expected: the 5s debounce timer resets; match-analysis does not fire until 5s of quiet. Do three edits spaced 2s apart — match-analysis fires ~5s after the third.

- [ ] **Step 5: Verify create-during-tracking 409**

During the `ready_tracking` phase (before batch completes), attempt to create a new rally via the timeline shortcut (Cmd/Ctrl+Enter). Expected: toast / error message "New rallies cannot be added while tracking is running." The existing rallies remain intact and tracking continues.

- [ ] **Step 6: Verify delete-during-tracking still works**

During `ready_tracking`, delete a rally. Expected: rally disappears; tracking continues on the remaining rallies; the eventual match-analysis operates on the reduced set.

- [ ] **Step 7: Verify stale-timeout on detection**

With dev tools, stop the API process mid-detection (Ctrl-C the `npm run dev` in api). Wait 10+ minutes, restart. Refresh the page. Expected: the pipeline surfaces "Timed out — detection did not complete within 10 minutes" rather than spinning forever.

- [ ] **Step 8: Report**

Record pass/fail for each step. If any step fails, file against this plan's deviations section rather than forcing a pass.

---

## Task 10: Open PR

- [ ] **Step 1: Rebase onto main + push branch**

```bash
git fetch origin
git rebase origin/main
git push -u origin feat/a2a-progressive-ux
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --title "feat(a2a): progressive UX & debounced match-analysis" --body "$(cat <<'EOF'
## Summary
- Collapse `AnalysisPhase` to `idle → preflight → preflight_gate → detecting → ready_tracking → match_analyzing → done | error`; `ready_tracking` is a non-blocking phase so the editor is usable immediately after detection.
- Move match-analysis trigger out of the Modal `tracking-batch-complete` webhook into a 5-second-idle debounce on the client, gated on `BatchTrackingJob.status === COMPLETED` and reset by rally CRUD events forwarded from `editorStore`.
- Extend the existing 10-minute stale-timeout helper to cover `RallyDetectionJob` as well as `BatchTrackingJob`, via a shared `staleJobRecovery` module.
- Reject mid-batch rally creation with `409 CREATE_DURING_TRACKING` (A2b will lift this via append-rally).

## Test plan
- [ ] `cd api && npm test` — all vitest suites pass (incl. new `staleJobRecovery`, `trackingBatchComplete`, `matchAnalysisTrigger`, `createDuringTracking`).
- [ ] `cd api && npx tsc --noEmit` — zero errors.
- [ ] `cd web && npx tsc --noEmit && npm run lint` — zero errors.
- [ ] Manual E2E checklist from plan Task 9 passes on `make dev`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Deviations

_Executor appends here when a task must diverge from the plan. Include task number, what changed, and why._

---
