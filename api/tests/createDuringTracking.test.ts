/**
 * Integration tests for rally CRUD while batch tracking is active.
 *
 * Post-Task-11: CREATE is no longer blocked with 409. New rallies are saved
 * without a PlayerTrack row and tagged for catch-up tracking (Task 12).
 *
 * Uses a real DB (same pattern as trackingBatchComplete.test.ts and
 * staleJobRecovery.test.ts). The sync-state route is POST /v1/sessions/:id/sync-state
 * and uses `ralliesPerVideo` (a map of videoId -> rally[]).
 *
 * Worker spawning is mocked so trackAllRallies({skipTracked:true}) creates
 * a real DB job but does not launch a child process.
 *
 * Fixture chain:
 *   User (PRO) → AnonymousIdentity (visitorId → userId)
 *              → Session (userId) → SessionVideo → Video
 *                                              → BatchTrackingJob (PROCESSING)
 */

// ---------------------------------------------------------------------------
// Mock child_process.spawn BEFORE any imports so spawnBatchWorker is a no-op.
// The mock returns an object shaped like ChildProcess (error handler + unref).
// ---------------------------------------------------------------------------
import { vi } from 'vitest';

vi.mock('child_process', async (importOriginal) => {
  const real = await importOriginal<typeof import('child_process')>();
  return {
    ...real,
    spawn: vi.fn().mockReturnValue({
      on: vi.fn(),
      unref: vi.fn(),
    }),
  };
});

// Also mock modalTrackingService in case MODAL_TRACKING_URL gets set.
vi.mock('../src/services/modalTrackingService.js', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/services/modalTrackingService.js')>();
  return {
    ...real,
    triggerModalBatchTracking: vi.fn().mockResolvedValue(undefined),
  };
});

import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import request from 'supertest';
import app from '../src/index';
import { prisma } from '../src/lib/prisma';
import { isBatchTrackingActive, trackAllRallies } from '../src/services/batchTrackingService';

const videoId = 'cd700000-0000-0000-0000-000000000001';
const userId = 'cd700000-0000-0000-0000-000000000002';
const visitorId = 'cd700000-0000-0000-0000-000000000003';
const sessionId = 'cd700000-0000-0000-0000-000000000004';
const jobId = 'cd700000-0000-0000-0000-000000000005';
const existingRallyId = 'cd700000-0000-0000-0000-000000000006';

describe('rally CRUD while tracking is active', () => {
  beforeEach(async () => {
    // Symmetric teardown first (handles test isolation if a previous run left data)
    await prisma.batchTrackingJob.deleteMany({ where: { videoId } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.sessionVideo.deleteMany({ where: { sessionId } });
    await prisma.session.deleteMany({ where: { id: sessionId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.anonymousIdentity.deleteMany({ where: { visitorId } });
    await prisma.user.deleteMany({ where: { id: userId } });

    // Seed: User (PRO — serverSyncEnabled: true)
    await prisma.user.create({
      data: {
        id: userId,
        tier: 'PRO',
      },
    });

    // AnonymousIdentity: visitor header → User.id
    await prisma.anonymousIdentity.create({
      data: {
        visitorId,
        userId,
      },
    });

    // Session owned by the user
    await prisma.session.create({
      data: {
        id: sessionId,
        userId,
        name: 'CDT Test Session',
      },
    });

    // Video (no FK to User required by schema)
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'cdt-test-video',
        filename: 'cdt-test.mp4',
        s3Key: 'test/cdt-test.mp4',
        contentHash: 'cdt-test-hash',
        userId,
      },
    });

    // Link video to session
    await prisma.sessionVideo.create({
      data: {
        sessionId,
        videoId,
      },
    });

    // Existing rally (used by UPDATE and DELETE tests)
    await prisma.rally.create({
      data: {
        id: existingRallyId,
        videoId,
        startMs: 0,
        endMs: 3000,
        order: 0,
      },
    });

    // BatchTrackingJob in PROCESSING for this video
    await prisma.batchTrackingJob.create({
      data: {
        id: jobId,
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 1,
        completedRallies: 0,
        failedRallies: 0,
      },
    });
  });

  afterEach(async () => {
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.batchTrackingJob.deleteMany({ where: { videoId } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.sessionVideo.deleteMany({ where: { sessionId } });
    await prisma.session.deleteMany({ where: { id: sessionId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.anonymousIdentity.deleteMany({ where: { visitorId } });
    await prisma.user.deleteMany({ where: { id: userId } });
    vi.clearAllMocks();
  });

  it('allows CREATE of a new rally while batch job is PROCESSING (no 409)', async () => {
    const res = await request(app)
      .post(`/v1/sessions/${sessionId}/sync-state`)
      .set('X-Visitor-Id', visitorId)
      .send({
        ralliesPerVideo: {
          // No `id` → treated as a new rally create
          [videoId]: [
            { startMs: 1000, endMs: 4000 },
          ],
        },
        highlights: [],
      });
    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ success: true });

    // The new rally should exist in the DB with no PlayerTrack
    const created = await prisma.rally.findFirst({
      where: { videoId, startMs: 1000 },
      include: { playerTrack: true },
    });
    expect(created).not.toBeNull();
    expect(created?.playerTrack).toBeNull();
  });

  it('allows UPDATE of an existing rally while batch job is PROCESSING', async () => {
    const res = await request(app)
      .post(`/v1/sessions/${sessionId}/sync-state`)
      .set('X-Visitor-Id', visitorId)
      .send({
        ralliesPerVideo: {
          [videoId]: [
            // Existing rally — has id matching DB row, updated boundaries
            { id: existingRallyId, startMs: 500, endMs: 3500 },
          ],
        },
        highlights: [],
      });

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ success: true });

    // Verify the rally was actually updated
    const updated = await prisma.rally.findUnique({ where: { id: existingRallyId } });
    expect(updated?.startMs).toBe(500);
    expect(updated?.endMs).toBe(3500);
  });

  it('allows DELETE of an existing rally while batch job is PROCESSING', async () => {
    const res = await request(app)
      .post(`/v1/sessions/${sessionId}/sync-state`)
      .set('X-Visitor-Id', visitorId)
      .send({
        ralliesPerVideo: {
          // Empty array for this video → the existing rally is omitted → deleted
          [videoId]: [],
        },
        highlights: [],
      });

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ success: true });

    // Verify the rally was actually deleted
    const deleted = await prisma.rally.findUnique({ where: { id: existingRallyId } });
    expect(deleted).toBeNull();
  });

  it('isBatchTrackingActive returns true for PENDING/PROCESSING, false for COMPLETED', async () => {
    expect(await isBatchTrackingActive(videoId)).toBe(true);

    // PENDING jobs are also considered active
    await prisma.batchTrackingJob.update({
      where: { id: jobId },
      data: { status: 'PENDING' },
    });
    expect(await isBatchTrackingActive(videoId)).toBe(true);

    await prisma.batchTrackingJob.update({
      where: { id: jobId },
      data: { status: 'COMPLETED', completedAt: new Date() },
    });
    expect(await isBatchTrackingActive(videoId)).toBe(false);
  });

  it('trackAllRallies with skipTracked=true returns {jobId:null, totalRallies:0} when all rallies are already tracked', async () => {
    // Seed: make the existing rally have a PlayerTrack row (already tracked)
    await prisma.playerTrack.create({
      data: { rallyId: existingRallyId, status: 'COMPLETED' },
    });
    const result = await trackAllRallies(videoId, userId, { skipTracked: true });
    expect(result.jobId).toBeNull();
    expect(result.totalRallies).toBe(0);
  });

  it('trackAllRallies with skipTracked=true returns a real jobId when there is an untracked rally', async () => {
    // The existing fixture rally (existingRallyId) has no PlayerTrack — it counts as untracked.
    // Complete the in-progress job so trackAllRallies doesn't reuse it (rally count differs after
    // the existing job was created with totalRallies=1 and our filter also finds 1, so it would
    // reuse — complete it to force a fresh job creation).
    await prisma.batchTrackingJob.update({
      where: { id: jobId },
      data: { status: 'COMPLETED', completedAt: new Date() },
    });
    const result = await trackAllRallies(videoId, userId, { skipTracked: true });
    expect(result.jobId).not.toBeNull();
    expect(result.totalRallies).toBeGreaterThanOrEqual(1);
  });
});
