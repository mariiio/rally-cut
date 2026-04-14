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
