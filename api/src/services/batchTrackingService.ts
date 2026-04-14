/**
 * Batch tracking service — tracks all rallies in a video sequentially.
 *
 * Flow:
 * 1. POST /v1/videos/:id/track-all-rallies → creates BatchTrackingJob, returns 202
 * 2. Local: spawns detached worker process (survives API restarts from tsx watch)
 *    Modal: triggers remote batch tracking via webhook
 * 3. Each rally tracked sequentially (shares Python model warmup across rallies)
 * 4. Frontend polls GET /v1/videos/:id/batch-tracking-status for progress
 * 5. On completion, auto-triggers match analysis (cross-rally player matching)
 */

import { spawn } from 'child_process';
import { mkdirSync, openSync } from 'fs';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

import { env } from '../config/env.js';
import { prisma } from '../lib/prisma.js';
import { ForbiddenError, NotFoundError, ValidationError } from '../middleware/errorHandler.js';
import type { CalibrationCorner } from './playerTrackingService.js';
import { filterOutCanonicalLockedRallies } from './playerTrackingService.js';
import { triggerModalBatchTracking } from './modalTrackingService.js';
import { expireStaleBatchTrackingJobs } from './staleJobRecovery.js';

/**
 * Start batch tracking for all rallies in a video.
 * Returns immediately with job ID (fire-and-forget).
 */
export async function trackAllRallies(
  videoId: string,
  userId: string,
): Promise<{ jobId: string; totalRallies: number }> {
  // Fetch video with rallies
  const video = await prisma.video.findUnique({
    where: { id: videoId },
    include: {
      rallies: {
        where: { status: 'CONFIRMED' },
        orderBy: { startMs: 'asc' },
      },
    },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }

  if (video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to track players for this video');
  }

  if (video.rallies.length === 0) {
    throw new ValidationError('No confirmed rallies found for this video');
  }

  // F3b — skip retrack on canonicalLocked rallies. Their GT-anchored
  // trackToPlayer mapping would be silently invalidated by fresh raw IDs.
  const rallyIds = video.rallies.map((r) => r.id);
  const { unlocked: unlockedRallyIds, locked: lockedRallyIds } =
    await filterOutCanonicalLockedRallies(videoId, rallyIds);
  if (lockedRallyIds.length > 0) {
    console.log(
      `[BATCH_TRACK] Skipping ${lockedRallyIds.length}/${rallyIds.length} canonicalLocked rallies for video ${videoId}`,
    );
  }
  const unlockedSet = new Set(unlockedRallyIds);
  const ralliesToTrack = video.rallies.filter((r) => unlockedSet.has(r.id));
  if (ralliesToTrack.length === 0) {
    throw new ValidationError(
      'All rallies are canonicalLocked — nothing to track',
    );
  }

  // Prefer original quality for tracking — proxy (720p) degrades ball detection.
  // Falls back to proxy if original has been quality-downgraded.
  const videoKey = video.s3Key ?? video.proxyS3Key;
  if (!videoKey) {
    throw new ValidationError('Video has no accessible source');
  }

  // Expire any stale jobs before checking for in-progress ones
  await expireStaleBatchTrackingJobs(videoId);

  // Check for existing in-progress batch job (atomic check-and-create)
  const job = await prisma.$transaction(async (tx) => {
    const existingJob = await tx.batchTrackingJob.findFirst({
      where: {
        videoId,
        status: { in: ['PENDING', 'PROCESSING'] },
      },
    });

    if (existingJob) {
      // If rally count changed (user edited rallies then re-analyzed),
      // cancel the stale job and create a fresh one
      if (existingJob.totalRallies !== ralliesToTrack.length) {
        await tx.batchTrackingJob.update({
          where: { id: existingJob.id },
          data: {
            status: 'FAILED',
            completedAt: new Date(),
            error: 'Cancelled — rally count changed, re-analyzing',
          },
        });
      } else {
        return existingJob;
      }
    }

    return tx.batchTrackingJob.create({
      data: {
        videoId,
        userId,
        status: 'PENDING',
        totalRallies: ralliesToTrack.length,
      },
    });
  });

  // If we found an existing job, return it without starting a new one
  if (job.status !== 'PENDING') {
    return { jobId: job.id, totalRallies: job.totalRallies };
  }

  // Auto-load calibration from DB
  let calibrationCorners: CalibrationCorner[] | undefined;
  if (video.courtCalibrationJson) {
    const dbCorners = video.courtCalibrationJson as unknown as CalibrationCorner[];
    if (Array.isArray(dbCorners) && dbCorners.length === 4) {
      calibrationCorners = dbCorners;
      console.log(`[BATCH_TRACK] Using court calibration from database for video ${videoId}`);
    }
  }

  // Fire and forget — process in background
  if (env.MODAL_TRACKING_URL) {
    // Modal GPU path: trigger remote batch tracking
    console.log(`[BATCH_TRACK] Using Modal GPU for batch job ${job.id}`);
    await prisma.batchTrackingJob.update({
      where: { id: job.id },
      data: { status: 'PROCESSING', lastProgressAt: new Date() },
    });
    triggerModalBatchTracking({
      batchJobId: job.id,
      videoId,
      videoKey,
      rallies: ralliesToTrack.map((r) => ({ id: r.id, startMs: r.startMs, endMs: r.endMs })),
      calibrationCorners,
    }).catch(async (error) => {
      console.error(`[BATCH_TRACK] Modal trigger failed for job ${job.id}:`, error);
      try {
        await prisma.batchTrackingJob.update({
          where: { id: job.id },
          data: {
            status: 'FAILED',
            completedAt: new Date(),
            error: error instanceof Error ? error.message : String(error),
          },
        });
      } catch {
        // If even the DB update fails, we can't do anything
      }
    });
  } else {
    // Local CPU path: spawn detached worker process that survives API restarts
    spawnBatchWorker(job.id);
  }

  console.log(`[BATCH_TRACK] Started batch job ${job.id}: ${ralliesToTrack.length} rallies for video ${videoId}`);

  return { jobId: job.id, totalRallies: ralliesToTrack.length };
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Spawn the batch tracking worker as a detached child process.
 * The worker reads job config from DB and processes rallies independently.
 * Survives API restarts (tsx watch) because the process is detached + unref'd.
 */
function spawnBatchWorker(jobId: string): void {
  const workerScript = path.resolve(__dirname, '../scripts/batchTrack.ts');
  const tsxBin = path.resolve(__dirname, '../../node_modules/.bin/tsx');

  // Write worker logs to a file — can't pipe to the API process because
  // when tsx watch restarts the API, the pipe breaks and SIGPIPE kills
  // the worker (defeating the whole purpose of detaching).
  const logDir = path.join(os.tmpdir(), 'rallycut-batch-tracking');
  mkdirSync(logDir, { recursive: true });
  const logPath = path.join(logDir, `batch_${jobId}.log`);
  const logFd = openSync(logPath, 'a');

  const child = spawn(tsxBin, [workerScript, jobId], {
    cwd: path.resolve(__dirname, '../..'),
    stdio: ['ignore', logFd, logFd],
    detached: true,
    env: { ...process.env },
  });

  child.on('error', (error) => {
    console.error(`[BATCH_TRACK] Failed to spawn worker for job ${jobId}:`, error);
  });

  // Detach — API process can exit without killing the worker
  child.unref();
  console.log(`[BATCH_TRACK] Worker spawned for job ${jobId}, logs at ${logPath}`);
}

/**
 * Returns true if the latest BatchTrackingJob for this video is active
 * (PENDING or PROCESSING). A2a uses this to reject mid-batch rally CREATION
 * with a 409 CONFLICT — existing rallies can still be updated or deleted.
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

/**
 * Get batch tracking status for a video.
 */
export async function getBatchTrackingStatus(
  videoId: string,
  userId: string,
): Promise<{
  status: 'idle' | 'pending' | 'processing' | 'completed' | 'failed';
  jobId?: string;
  totalRallies?: number;
  completedRallies?: number;
  failedRallies?: number;
  currentRallyId?: string;
  error?: string;
  completedAt?: Date;
  rallyStatuses?: Array<{ rallyId: string; status: string }>;
}> {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }

  if (video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to view batch tracking status for this video');
  }

  // Expire stale jobs before reporting status
  await expireStaleBatchTrackingJobs(videoId);

  // Get the most recent batch job
  const job = await prisma.batchTrackingJob.findFirst({
    where: { videoId },
    orderBy: { createdAt: 'desc' },
  });

  if (!job) {
    return { status: 'idle' };
  }

  // Get per-rally tracking statuses
  const rallies = await prisma.rally.findMany({
    where: { videoId, status: 'CONFIRMED' },
    include: { playerTrack: { select: { status: true } } },
    orderBy: { startMs: 'asc' },
  });

  const rallyStatuses = rallies.map((r) => ({
    rallyId: r.id,
    status: r.playerTrack?.status ?? 'PENDING',
  }));

  return {
    status: job.status.toLowerCase() as 'pending' | 'processing' | 'completed' | 'failed',
    jobId: job.id,
    totalRallies: job.totalRallies,
    completedRallies: job.completedRallies,
    failedRallies: job.failedRallies,
    currentRallyId: job.currentRallyId ?? undefined,
    error: job.error ?? undefined,
    completedAt: job.completedAt ?? undefined,
    rallyStatuses,
  };
}
