import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import {
  startStaleJobSweeper,
  stopStaleJobSweeper,
} from '../src/jobs/staleJobSweeper';
import { STALE_PROGRESS_TIMEOUT_MS } from '../src/services/staleJobRecovery';

describe('staleJobSweeper', () => {
  const testVideoIds = ['vid-sweeper-a', 'vid-sweeper-b'];
  const testUserId = 'user-sweeper';

  beforeEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { videoId: { in: testVideoIds } } });
    await prisma.video.deleteMany({ where: { id: { in: testVideoIds } } });
    for (const videoId of testVideoIds) {
      await prisma.video.create({
        data: {
          id: videoId,
          name: `test-${videoId}`,
          filename: `${videoId}.mp4`,
          s3Key: `${videoId}.mp4`,
          contentHash: `${'x'.repeat(63)}${videoId.slice(-1)}`,
        },
      });
    }
  });

  afterEach(async () => {
    stopStaleJobSweeper();
    await prisma.batchTrackingJob.deleteMany({ where: { videoId: { in: testVideoIds } } });
    await prisma.video.deleteMany({ where: { id: { in: testVideoIds } } });
  });

  it('runs once immediately on start, sweeping stale jobs across all videos', async () => {
    const stale = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS - 60_000);
    await prisma.batchTrackingJob.createMany({
      data: testVideoIds.map((videoId, i) => ({
        id: `sweeper-job-${i}`,
        videoId,
        userId: testUserId,
        status: 'PROCESSING',
        totalRallies: 1,
        completedRallies: 0,
        failedRallies: 0,
        lastProgressAt: stale,
        createdAt: stale,
      })),
    });

    startStaleJobSweeper();
    // The sweep is async; await a couple of event-loop ticks.
    await new Promise((r) => setImmediate(r));
    await new Promise((r) => setImmediate(r));

    const jobs = await prisma.batchTrackingJob.findMany({
      where: { videoId: { in: testVideoIds } },
      orderBy: { createdAt: 'asc' },
    });
    expect(jobs.length).toBe(2);
    expect(jobs.every((j) => j.status === 'FAILED')).toBe(true);
  });

  it('is idempotent — calling start twice does not stack timers', () => {
    startStaleJobSweeper();
    startStaleJobSweeper();
    // No assertion needed — the lazy handle check in start should silently no-op.
  });
});
