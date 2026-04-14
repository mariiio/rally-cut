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
 * Detection jobs have no `lastProgressAt` column, so `createdAt` is the
 * staleness floor — both never-started (PENDING) and silently-crashed
 * (RUNNING) jobs collapse onto the same cutoff. If RallyDetectionJob
 * ever gains a progress-update column, switch RUNNING to use that field.
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
