/**
 * Modal GPU batch tracking service.
 *
 * Handles:
 * 1. Triggering Modal batch tracking (triggerModalBatchTracking)
 * 2. Per-rally webhook (handleTrackingRallyComplete)
 * 3. Batch-complete webhook (handleTrackingBatchComplete)
 */

import { env } from '../config/env.js';
import { prisma } from '../lib/prisma.js';
import { saveTrackingResult, type PlayerTrackerOutput } from './playerTrackingService.js';
import { runMatchAnalysis } from './matchAnalysisService.js';

interface TriggerParams {
  batchJobId: string;
  videoId: string;
  videoKey: string;
  rallies: Array<{ id: string; startMs: number; endMs: number }>;
  calibrationCorners?: Array<{ x: number; y: number }>;
}

/**
 * POST to Modal tracking endpoint to start GPU batch tracking.
 */
export async function triggerModalBatchTracking(params: TriggerParams): Promise<void> {
  if (!env.MODAL_TRACKING_URL) {
    throw new Error('MODAL_TRACKING_URL is not configured');
  }

  const response = await fetch(env.MODAL_TRACKING_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      batch_job_id: params.batchJobId,
      video_id: params.videoId,
      video_key: params.videoKey,
      rallies: params.rallies.map((r) => ({
        id: r.id,
        start_ms: r.startMs,
        end_ms: r.endMs,
      })),
      calibration_corners: params.calibrationCorners ?? null,
      callback_url: `${env.API_BASE_URL}/v1/webhooks`,
      webhook_secret: env.MODAL_WEBHOOK_SECRET,
      s3_bucket: env.S3_BUCKET_NAME,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Modal tracking call failed: ${response.status} - ${text}`);
  }
}

interface RallyCompletePayload {
  batch_job_id: string;
  video_id: string;
  rally_id: string;
  status: 'completed' | 'failed';
  tracking_data?: Record<string, unknown>;
  error?: string;
}

/**
 * Handle per-rally tracking webhook from Modal.
 * Saves tracking result to DB and updates batch job progress.
 */
export async function handleTrackingRallyComplete(
  payload: RallyCompletePayload,
): Promise<{ success: boolean }> {
  const { batch_job_id, video_id, rally_id, status } = payload;

  console.log(
    `[MODAL_TRACK] Rally ${rally_id} ${status} (batch ${batch_job_id})`,
  );

  if (status === 'completed' && payload.tracking_data) {
    const data = payload.tracking_data;

    // Map CLI JSON output to PlayerTrackerOutput format
    const trackerResult: PlayerTrackerOutput = {
      positions: (data.positions as PlayerTrackerOutput['positions']) ?? [],
      rawPositions: data.rawPositions as PlayerTrackerOutput['positions'],
      frameCount: (data.frameCount as number) ?? 0,
      fps: (data.videoFps as number) ?? (data.fps as number) ?? 30,
      detectionRate: data.detectionRate as number ?? 0,
      avgConfidence: data.avgConfidence as number ?? 0,
      avgPlayerCount: data.avgPlayersPerFrame as number ?? data.avgPlayerCount as number ?? 0,
      uniqueTrackCount: data.uniqueTrackCount as number ?? 0,
      courtSplitY: data.courtSplitY as number | undefined,
      primaryTrackIds: data.primaryTrackIds as number[] | undefined,
      ballPositions: data.ballPositions as PlayerTrackerOutput['ballPositions'],
      contacts: data.contacts as PlayerTrackerOutput['contacts'],
      actions: data.actions as PlayerTrackerOutput['actions'],
      qualityReport: data.qualityReport as PlayerTrackerOutput['qualityReport'],
      courtDetection: data.courtDetection as PlayerTrackerOutput['courtDetection'],
    };

    // Use the shared save function (same DB write path as local tracking)
    await saveTrackingResult(rally_id, video_id, trackerResult, 0);

    // Increment completed count
    await prisma.batchTrackingJob.update({
      where: { id: batch_job_id },
      data: { completedRallies: { increment: 1 }, currentRallyId: rally_id },
    });
  } else {
    // Mark rally as failed
    await prisma.playerTrack.upsert({
      where: { rallyId: rally_id },
      create: {
        rallyId: rally_id,
        status: 'FAILED',
        error: payload.error ?? 'Unknown error',
      },
      update: {
        status: 'FAILED',
        error: payload.error ?? 'Unknown error',
      },
    });

    await prisma.batchTrackingJob.update({
      where: { id: batch_job_id },
      data: { failedRallies: { increment: 1 }, currentRallyId: rally_id },
    });
  }

  return { success: true };
}

interface BatchCompletePayload {
  batch_job_id: string;
  video_id: string;
  status: 'completed' | 'failed';
  completed_rallies: number;
  failed_rallies: number;
  error?: string;
}

/**
 * Handle batch-complete webhook from Modal.
 * Updates job status and triggers match analysis.
 */
export async function handleTrackingBatchComplete(
  payload: BatchCompletePayload,
): Promise<{ success: boolean }> {
  const { batch_job_id, video_id, status, completed_rallies, failed_rallies } = payload;

  console.log(
    `[MODAL_TRACK] Batch ${batch_job_id} ${status}: ${completed_rallies} completed, ${failed_rallies} failed`,
  );

  await prisma.batchTrackingJob.update({
    where: { id: batch_job_id },
    data: {
      status: status === 'completed' ? 'COMPLETED' : 'FAILED',
      completedAt: new Date(),
      currentRallyId: null,
      completedRallies: completed_rallies,
      failedRallies: failed_rallies,
      error: payload.error ?? null,
    },
  });

  // Trigger match analysis if any rallies succeeded
  if (completed_rallies > 0) {
    try {
      await runMatchAnalysis(video_id);
      console.log(`[MODAL_TRACK] Match analysis completed for video ${video_id}`);
    } catch (error) {
      console.error(`[MODAL_TRACK] Match analysis failed for video ${video_id}:`, error);
    }
  }

  return { success: true };
}
