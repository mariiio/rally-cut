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

    // Track every rally the API handed us. Filtering "already-tracked"
    // rallies is the API layer's job:
    //   - `trackAllRallies(opts={})` (UI's "Retrack & analyze") selects ALL
    //     CONFIRMED rallies — user intent is to refresh tracking even when
    //     existing data is COMPLETED.
    //   - `trackAllRallies({skipTracked:true})` (catch-up path) selects only
    //     rallies with no PlayerTrack or `needsRetrack=true` — already-
    //     tracked rallies are not in the queue.
    // The previous defensive worker-side skip-on-COMPLETED short-circuited
    // the "Retrack & analyze" path entirely: every rally with prior tracking
    // was silently skipped and only post-processing (match-analysis) ran on
    // top of stale positions. Real-world result: corrupted primary_track_ids
    // accumulated and could never self-heal because tracking never re-ran.
    let completedCount = 0;
    let failedCount = 0;

    // Process each rally sequentially
    for (const rally of rallies) {
      // Check if job was cancelled (e.g. user re-analyzed with different rallies)
      const currentJob = await prisma.batchTrackingJob.findUnique({
        where: { id: jobId },
        select: { status: true },
      });
      if (!currentJob || currentJob.status === 'FAILED') {
        console.log(`[BATCH_WORKER] Job ${jobId} was cancelled, stopping`);
        break;
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
    const attemptedCount = rallies.length;
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

    // Match analysis is triggered client-side via the analysisStore's 5-s
    // debounce + catch-up flow once pollTracking observes batch completion.
    // This mirrors the Modal path (`tracking-batch-complete` webhook also
    // no longer auto-triggers). Auto-triggering here would bypass catch-up
    // for rallies created mid-batch and race with the client's SSE call.
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
