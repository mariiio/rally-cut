/**
 * Batch tracking service — tracks all rallies in a video sequentially.
 *
 * Flow:
 * 1. POST /v1/videos/:id/track-all-rallies → creates BatchTrackingJob, returns 202
 * 2. Background: downloads video once, iterates rallies, extracts segments locally
 * 3. Each rally tracked sequentially (shares Python model warmup across rallies)
 * 4. Frontend polls GET /v1/videos/:id/batch-tracking-status for progress
 * 5. On completion, auto-triggers match analysis (cross-rally player matching)
 */

import fs from 'fs/promises';
import path from 'path';

import { env } from '../config/env.js';
import { prisma } from '../lib/prisma.js';
import { ForbiddenError, NotFoundError, ValidationError } from '../middleware/errorHandler.js';
import {
  downloadVideoToLocal,
  trackRallyFromLocalVideo,
  TEMP_DIR,
  type CalibrationCorner,
} from './playerTrackingService.js';
import { triggerModalBatchTracking } from './modalTrackingService.js';
import { runMatchAnalysis } from './matchAnalysisService.js';

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

  // Prefer original quality for tracking — proxy (720p) degrades ball detection.
  // Falls back to proxy if original has been quality-downgraded.
  const videoKey = video.s3Key ?? video.proxyS3Key;
  if (!videoKey) {
    throw new ValidationError('Video has no accessible source');
  }

  // Check for existing in-progress batch job (atomic check-and-create)
  const job = await prisma.$transaction(async (tx) => {
    const existingJob = await tx.batchTrackingJob.findFirst({
      where: {
        videoId,
        status: { in: ['PENDING', 'PROCESSING'] },
      },
    });

    if (existingJob) {
      return existingJob;
    }

    return tx.batchTrackingJob.create({
      data: {
        videoId,
        userId,
        status: 'PENDING',
        totalRallies: video.rallies.length,
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
      data: { status: 'PROCESSING' },
    });
    triggerModalBatchTracking({
      batchJobId: job.id,
      videoId,
      videoKey,
      rallies: video.rallies.map((r) => ({ id: r.id, startMs: r.startMs, endMs: r.endMs })),
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
    // Local CPU path: existing processBatchTracking() unchanged
    processBatchTracking(job.id, videoId, videoKey, video.rallies, calibrationCorners).catch(
      async (error) => {
        console.error(`[BATCH_TRACK] Fatal error in batch job ${job.id}:`, error);
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
      }
    );
  }

  console.log(`[BATCH_TRACK] Started batch job ${job.id}: ${video.rallies.length} rallies for video ${videoId}`);

  return { jobId: job.id, totalRallies: video.rallies.length };
}

/**
 * Background batch processing. Downloads video once, iterates rallies.
 */
async function processBatchTracking(
  jobId: string,
  videoId: string,
  videoKey: string,
  rallies: Array<{ id: string; startMs: number; endMs: number }>,
  calibrationCorners?: CalibrationCorner[],
): Promise<void> {
  await fs.mkdir(TEMP_DIR, { recursive: true });
  const localVideoPath = path.join(TEMP_DIR, `batch_${jobId}_full.mp4`);

  try {
    // Mark job as processing
    await prisma.batchTrackingJob.update({
      where: { id: jobId },
      data: { status: 'PROCESSING' },
    });

    // Download full video once
    console.log(`[BATCH_TRACK] Downloading video ${videoKey} to local...`);
    await downloadVideoToLocal(videoKey, localVideoPath);
    console.log(`[BATCH_TRACK] Video downloaded to ${localVideoPath}`);

    let completedCount = 0;
    let failedCount = 0;

    // Process each rally sequentially
    for (const rally of rallies) {
      // Update current rally
      await prisma.batchTrackingJob.update({
        where: { id: jobId },
        data: { currentRallyId: rally.id },
      });

      try {
        const result = await trackRallyFromLocalVideo(
          rally.id,
          videoId,
          localVideoPath,
          rally.startMs,
          rally.endMs,
          calibrationCorners,
        );

        if (result.status === 'completed') {
          completedCount++;
        } else {
          failedCount++;
        }
      } catch (error) {
        console.error(`[BATCH_TRACK] Rally ${rally.id} threw:`, error);
        failedCount++;
      }

      // Update progress
      await prisma.batchTrackingJob.update({
        where: { id: jobId },
        data: {
          completedRallies: completedCount,
          failedRallies: failedCount,
        },
      });
    }

    // Mark batch job as complete
    await prisma.batchTrackingJob.update({
      where: { id: jobId },
      data: {
        status: failedCount === rallies.length ? 'FAILED' : 'COMPLETED',
        completedAt: new Date(),
        currentRallyId: null,
        error: failedCount > 0 ? `${failedCount}/${rallies.length} rallies failed` : null,
      },
    });

    console.log(`[BATCH_TRACK] Job ${jobId} done: ${completedCount} completed, ${failedCount} failed`);

    // Auto-trigger match analysis if any rallies succeeded
    if (completedCount > 0) {
      try {
        await runMatchAnalysis(videoId);
      } catch (error) {
        console.error(`[BATCH_TRACK] Match analysis failed for video ${videoId}:`, error);
      }
    }
  } catch (error) {
    console.error(`[BATCH_TRACK] Job ${jobId} failed:`, error);
    await prisma.batchTrackingJob.update({
      where: { id: jobId },
      data: {
        status: 'FAILED',
        completedAt: new Date(),
        error: error instanceof Error ? error.message : String(error),
      },
    });
  } finally {
    // Clean up downloaded video
    await fs.unlink(localVideoPath).catch(() => {});
  }
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
