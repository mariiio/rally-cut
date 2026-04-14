import 'dotenv/config';
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
    await prisma.video.deleteMany({ where: { id: videoId } });
    // Create a minimal Video fixture so BatchTrackingJob FK is satisfied.
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'stale-test-video',
        filename: 'stale-test.mp4',
        s3Key: 'test/stale-test.mp4',
        contentHash: 'stale-test-hash',
      },
    });
  });

  afterEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
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
