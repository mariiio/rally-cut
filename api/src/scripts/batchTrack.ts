/**
 * Standalone batch tracking worker — runs as a detached child process.
 *
 * Spawned by the API but survives API restarts (tsx watch).
 * Reads job config from BatchTrackingJob in DB, processes rallies
 * sequentially, updates progress in DB.
 *
 * Usage: tsx src/scripts/batchTrack.ts <jobId>
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

// Set up __dirname for ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load env before prisma
import { config } from 'dotenv';
config({ path: path.resolve(__dirname, '../../.env') });

import { PrismaClient } from '@prisma/client';
import {
  downloadVideoToLocal,
  trackRallyFromLocalVideo,
  TEMP_DIR,
  type CalibrationCorner,
} from '../services/playerTrackingService.js';
import { runMatchAnalysis } from '../services/matchAnalysisService.js';

const prisma = new PrismaClient();

async function main() {
  const jobId = process.argv[2];
  if (!jobId) {
    console.error('[BATCH_WORKER] Missing jobId argument');
    process.exit(1);
  }

  const job = await prisma.batchTrackingJob.findUnique({
    where: { id: jobId },
    include: {
      video: {
        include: {
          rallies: {
            where: { status: 'CONFIRMED' },
            orderBy: { startMs: 'asc' },
          },
        },
      },
    },
  });

  if (!job) {
    console.error(`[BATCH_WORKER] Job ${jobId} not found`);
    process.exit(1);
  }

  const video = job.video;
  const videoKey = video.s3Key ?? video.proxyS3Key;
  if (!videoKey) {
    await failJob(jobId, 'Video has no accessible source');
    process.exit(1);
  }

  const rallies = video.rallies.map((r) => ({
    id: r.id,
    startMs: r.startMs,
    endMs: r.endMs,
  }));

  // Load calibration from DB
  let calibrationCorners: CalibrationCorner[] | undefined;
  if (video.courtCalibrationJson) {
    const dbCorners = video.courtCalibrationJson as unknown as CalibrationCorner[];
    if (Array.isArray(dbCorners) && dbCorners.length === 4) {
      calibrationCorners = dbCorners;
      console.log(`[BATCH_WORKER] Using court calibration from database`);
    }
  }

  await fs.mkdir(TEMP_DIR, { recursive: true });
  const localVideoPath = path.join(TEMP_DIR, `batch_${jobId}_full.mp4`);

  try {
    // Mark job as processing
    await prisma.batchTrackingJob.update({
      where: { id: jobId },
      data: { status: 'PROCESSING', lastProgressAt: new Date() },
    });

    // Download full video once
    console.log(`[BATCH_WORKER] Downloading video ${videoKey} to local...`);
    await downloadVideoToLocal(videoKey, localVideoPath);
    console.log(`[BATCH_WORKER] Video downloaded`);

    // Reset any zombie PROCESSING player_tracks from crashed previous runs
    const { count: resetCount } = await prisma.playerTrack.updateMany({
      where: {
        rallyId: { in: rallies.map((r) => r.id) },
        status: 'PROCESSING',
      },
      data: { status: 'PENDING' },
    });
    if (resetCount > 0) {
      console.log(`[BATCH_WORKER] Reset ${resetCount} zombie PROCESSING player_track(s)`);
    }

    // Check which rallies are already completed (skip re-tracking)
    const completedTracks = await prisma.playerTrack.findMany({
      where: {
        rallyId: { in: rallies.map((r) => r.id) },
        status: 'COMPLETED',
      },
      select: { rallyId: true },
    });
    const alreadyCompleted = new Set(completedTracks.map((t) => t.rallyId));

    let completedCount = alreadyCompleted.size;
    let failedCount = 0;

    if (alreadyCompleted.size > 0) {
      console.log(`[BATCH_WORKER] Skipping ${alreadyCompleted.size} already-completed rallies`);
    }

    // Process each rally sequentially
    for (const rally of rallies) {
      if (alreadyCompleted.has(rally.id)) {
        continue;
      }

      // Update current rally + heartbeat
      await prisma.batchTrackingJob.update({
        where: { id: jobId },
        data: { currentRallyId: rally.id, lastProgressAt: new Date() },
      });

      try {
        const result = await trackRallyFromLocalVideo(
          rally.id,
          video.id,
          localVideoPath,
          rally.startMs,
          rally.endMs,
          video.fps ?? undefined,
          calibrationCorners,
        );

        if (result.status === 'completed') {
          completedCount++;
        } else {
          failedCount++;
        }
      } catch (error) {
        console.error(`[BATCH_WORKER] Rally ${rally.id} threw:`, error);
        failedCount++;
      }

      // Update progress + heartbeat
      await prisma.batchTrackingJob.update({
        where: { id: jobId },
        data: {
          completedRallies: completedCount,
          failedRallies: failedCount,
          lastProgressAt: new Date(),
        },
      });
    }

    // Mark batch job as complete
    const attemptedCount = rallies.length - alreadyCompleted.size;
    await prisma.batchTrackingJob.update({
      where: { id: jobId },
      data: {
        status: attemptedCount > 0 && failedCount === attemptedCount ? 'FAILED' : 'COMPLETED',
        completedAt: new Date(),
        currentRallyId: null,
        error: failedCount > 0 ? `${failedCount}/${rallies.length} rallies failed` : null,
      },
    });

    console.log(`[BATCH_WORKER] Job ${jobId} done: ${completedCount} completed, ${failedCount} failed`);

    // Auto-trigger match analysis if any rallies succeeded
    if (completedCount > 0) {
      try {
        await runMatchAnalysis(video.id);
      } catch (error) {
        console.error(`[BATCH_WORKER] Match analysis failed:`, error);
      }
    }
  } catch (error) {
    console.error(`[BATCH_WORKER] Job ${jobId} failed:`, error);
    await failJob(jobId, error instanceof Error ? error.message : String(error));
  } finally {
    await fs.unlink(localVideoPath).catch(() => {});
    await prisma.$disconnect();
  }
}

async function failJob(jobId: string, error: string) {
  await prisma.batchTrackingJob.update({
    where: { id: jobId },
    data: {
      status: 'FAILED',
      completedAt: new Date(),
      error,
    },
  });
}

main().catch((err) => {
  console.error('[BATCH_WORKER] Unhandled error:', err);
  process.exit(1);
});
