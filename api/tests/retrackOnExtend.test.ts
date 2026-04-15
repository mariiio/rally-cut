/**
 * Integration tests for EXTEND-rally retrack detection (A2b Task 3).
 *
 * Tests:
 *  - markRetrackIfExtended: extend start, extend end, shorten (no-op), no PlayerTrack row
 *  - trackAllRallies skipTracked filter: picks up needsRetrack=true rows
 *
 * Uses a real DB (same pattern as createDuringTracking.test.ts).
 * Worker spawning is mocked so trackAllRallies creates a real DB job
 * but does not launch a child process.
 */

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

vi.mock('../src/services/modalTrackingService.js', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/services/modalTrackingService.js')>();
  return {
    ...real,
    triggerModalBatchTracking: vi.fn().mockResolvedValue(undefined),
  };
});

import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { markRetrackIfExtended, trackAllRallies } from '../src/services/batchTrackingService';

// ---------------------------------------------------------------------------
// Fixed UUIDs for test isolation
// ---------------------------------------------------------------------------
const videoId    = 'rroe0000-0000-0000-0000-000000000001';
const userId     = 'rroe0000-0000-0000-0000-000000000002';
const rallyId    = 'rroe0000-0000-0000-0000-000000000010';
const rallyId2   = 'rroe0000-0000-0000-0000-000000000011'; // no PlayerTrack
const rallyR1    = 'rroe0000-0000-0000-0000-000000000020'; // untracked (no PlayerTrack)
const rallyR2    = 'rroe0000-0000-0000-0000-000000000021'; // tracked, needsRetrack=false
const rallyR3    = 'rroe0000-0000-0000-0000-000000000022'; // tracked, needsRetrack=true
const jobId      = 'rroe0000-0000-0000-0000-000000000030';

// ---------------------------------------------------------------------------
// markRetrackIfExtended
// ---------------------------------------------------------------------------
describe('markRetrackIfExtended', () => {
  beforeEach(async () => {
    // Teardown first for isolation
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });

    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'retrack-test-video',
        filename: 'retrack.mp4',
        s3Key: 'test/retrack.mp4',
        contentHash: 'retrack-hash',
        userId,
      },
    });

    // Rally with a PlayerTrack (needsRetrack=false)
    await prisma.rally.create({
      data: { id: rallyId, videoId, startMs: 5000, endMs: 10000, order: 0 },
    });
    await prisma.playerTrack.create({
      data: { rallyId, status: 'COMPLETED', needsRetrack: false },
    });

    // Rally without a PlayerTrack (for no-op test)
    await prisma.rally.create({
      data: { id: rallyId2, videoId, startMs: 11000, endMs: 15000, order: 1 },
    });
  });

  afterEach(async () => {
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
  });

  it('marks needsRetrack=true when rally start is moved earlier', async () => {
    const result = await markRetrackIfExtended(
      prisma,
      rallyId,
      { startMs: 5000, endMs: 10000 },
      { startMs: 3000, endMs: 10000 },
    );
    expect(result).toBe(true);
    const pt = await prisma.playerTrack.findUniqueOrThrow({ where: { rallyId } });
    expect(pt.needsRetrack).toBe(true);
  });

  it('marks needsRetrack=true when rally end is moved later', async () => {
    const result = await markRetrackIfExtended(
      prisma,
      rallyId,
      { startMs: 5000, endMs: 10000 },
      { startMs: 5000, endMs: 12000 },
    );
    expect(result).toBe(true);
    const pt = await prisma.playerTrack.findUniqueOrThrow({ where: { rallyId } });
    expect(pt.needsRetrack).toBe(true);
  });

  it('marks needsRetrack=true when both start is earlier AND end is later', async () => {
    const result = await markRetrackIfExtended(
      prisma,
      rallyId,
      { startMs: 5000, endMs: 10000 },
      { startMs: 3000, endMs: 12000 },
    );
    expect(result).toBe(true);
    const pt = await prisma.playerTrack.findUniqueOrThrow({ where: { rallyId } });
    expect(pt.needsRetrack).toBe(true);
  });

  it('does NOT mark when rally is shortened (start later and end earlier)', async () => {
    const result = await markRetrackIfExtended(
      prisma,
      rallyId,
      { startMs: 5000, endMs: 10000 },
      { startMs: 6000, endMs: 9000 },
    );
    expect(result).toBe(false);
    const pt = await prisma.playerTrack.findUniqueOrThrow({ where: { rallyId } });
    expect(pt.needsRetrack).toBe(false);
  });

  it('does NOT mark when bounds are unchanged', async () => {
    const result = await markRetrackIfExtended(
      prisma,
      rallyId,
      { startMs: 5000, endMs: 10000 },
      { startMs: 5000, endMs: 10000 },
    );
    expect(result).toBe(false);
    const pt = await prisma.playerTrack.findUniqueOrThrow({ where: { rallyId } });
    expect(pt.needsRetrack).toBe(false);
  });

  it('no-op and returns false when there is no PlayerTrack row for the rally', async () => {
    const result = await markRetrackIfExtended(
      prisma,
      rallyId2,
      { startMs: 11000, endMs: 15000 },
      { startMs: 9000, endMs: 15000 },
    );
    // No PlayerTrack row → updateMany matches 0 rows → returns false
    expect(result).toBe(false);
    // Confirm no row was created
    const pt = await prisma.playerTrack.findUnique({ where: { rallyId: rallyId2 } });
    expect(pt).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// trackAllRallies skipTracked filter includes needsRetrack=true
// ---------------------------------------------------------------------------
describe('trackAllRallies skipTracked filter includes needsRetrack=true', () => {
  beforeEach(async () => {
    // Teardown
    await prisma.batchTrackingJob.deleteMany({ where: { videoId } });
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });

    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'retrack-filter-video',
        filename: 'retrack-filter.mp4',
        s3Key: 'test/retrack-filter.mp4',
        contentHash: 'retrack-filter-hash',
        userId,
      },
    });

    // R1: CONFIRMED, no PlayerTrack → should be picked up
    await prisma.rally.create({
      data: { id: rallyR1, videoId, startMs: 1000, endMs: 4000, order: 0, status: 'CONFIRMED' },
    });

    // R2: CONFIRMED, PlayerTrack with needsRetrack=false → should be skipped
    await prisma.rally.create({
      data: { id: rallyR2, videoId, startMs: 5000, endMs: 8000, order: 1, status: 'CONFIRMED' },
    });
    await prisma.playerTrack.create({
      data: { rallyId: rallyR2, status: 'COMPLETED', needsRetrack: false },
    });

    // R3: CONFIRMED, PlayerTrack with needsRetrack=true → should be picked up
    await prisma.rally.create({
      data: { id: rallyR3, videoId, startMs: 9000, endMs: 12000, order: 2, status: 'CONFIRMED' },
    });
    await prisma.playerTrack.create({
      data: { rallyId: rallyR3, status: 'COMPLETED', needsRetrack: true },
    });

    // Mark the pre-seeded job as COMPLETED so trackAllRallies creates a fresh one
    await prisma.batchTrackingJob.create({
      data: {
        id: jobId,
        videoId,
        userId,
        status: 'COMPLETED',
        completedAt: new Date(),
        totalRallies: 3,
        completedRallies: 3,
        failedRallies: 0,
      },
    });
  });

  afterEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { videoId } });
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
    vi.clearAllMocks();
  });

  it('picks up rallies without PlayerTrack AND rallies with needsRetrack=true, skips needsRetrack=false', async () => {
    const result = await trackAllRallies(videoId, userId, { skipTracked: true });

    // R1 (no track) + R3 (needsRetrack=true) → 2 rallies; R2 (needsRetrack=false) is skipped
    expect(result.totalRallies).toBe(2);
    expect(result.jobId).toEqual(expect.any(String));
  });
});
