import { ConfirmationStatus, Prisma } from "@prisma/client";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { spawn } from "child_process";
import fs from "fs/promises";
import os from "os";
import path from "path";

import { env } from "../config/env.js";
import { generateDownloadUrl, generateUploadUrl, deleteObject } from "../lib/s3.js";
import { prisma } from "../lib/prisma.js";
import { ForbiddenError, NotFoundError, ValidationError } from "../middleware/errorHandler.js";
import { getUserTier, getTierLimits } from "./tierService.js";

const lambdaClient = new LambdaClient({
  region: env.AWS_REGION,
});

// Check if we should use local FFmpeg (development mode)
function shouldUseLocalProcessing(): boolean {
  // Use local if no Lambda configured
  if (!env.EXPORT_LAMBDA_FUNCTION_NAME) {
    return true;
  }
  // Or if explicitly set
  if (process.env.USE_LOCAL_EXPORT === "true") {
    return true;
  }
  return false;
}

export interface TimestampMapping {
  rallyId: string;
  originalStartMs: number;
  originalEndMs: number;
  trimmedStartMs: number;
  trimmedEndMs: number;
}

interface RallyForConfirmation {
  id: string;
  startMs: number;
  endMs: number;
}

/**
 * Calculate timestamp mappings for rallies.
 * Maps original timestamps to trimmed (concatenated) timestamps.
 */
export function calculateTimestampMappings(
  rallies: RallyForConfirmation[]
): TimestampMapping[] {
  // Sort rallies by start time
  const sorted = [...rallies].sort((a, b) => a.startMs - b.startMs);
  let trimmedOffset = 0;

  return sorted.map((rally) => {
    const duration = rally.endMs - rally.startMs;
    const mapping: TimestampMapping = {
      rallyId: rally.id,
      originalStartMs: rally.startMs,
      originalEndMs: rally.endMs,
      trimmedStartMs: trimmedOffset,
      trimmedEndMs: trimmedOffset + duration,
    };
    trimmedOffset += duration;
    return mapping;
  });
}

export interface ConfirmationResult {
  confirmationId: string;
  status: ConfirmationStatus;
  progress: number;
  createdAt: Date;
}

/**
 * Initiate rally confirmation for a video.
 * Creates a trimmed video with only rally segments.
 */
export async function initiateConfirmation(
  videoId: string,
  userId: string
): Promise<ConfirmationResult> {
  // Check user tier
  const userTier = await getUserTier(userId);
  const limits = getTierLimits(userTier);

  if (userTier !== "PREMIUM") {
    throw new ForbiddenError("Rally confirmation requires Premium tier.");
  }

  // Get video with rallies
  const video = await prisma.video.findFirst({
    where: {
      id: videoId,
      userId,
      deletedAt: null,
    },
    include: {
      rallies: {
        orderBy: { startMs: "asc" },
      },
      confirmation: true,
    },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Check if there are rallies
  if (video.rallies.length === 0) {
    throw new ValidationError("Video must have at least one rally to confirm.");
  }

  // Check for existing pending/processing confirmation
  if (
    video.confirmation &&
    (video.confirmation.status === "PENDING" ||
      video.confirmation.status === "PROCESSING")
  ) {
    throw new ValidationError(
      "A confirmation is already in progress for this video."
    );
  }

  // Calculate timestamp mappings
  const mappings = calculateTimestampMappings(
    video.rallies.map((r) => ({
      id: r.id,
      startMs: r.startMs,
      endMs: r.endMs,
    }))
  );

  // Calculate trimmed duration
  const trimmedDurationMs = mappings.reduce(
    (sum, m) => sum + (m.trimmedEndMs - m.trimmedStartMs),
    0
  );

  // Create or update confirmation record
  const confirmation = await prisma.rallyConfirmation.upsert({
    where: { videoId },
    create: {
      videoId,
      status: ConfirmationStatus.PENDING,
      originalS3Key: video.s3Key,
      originalDurationMs: video.durationMs ?? 0,
      timestampMappings: mappings as unknown as Prisma.InputJsonValue,
      progress: 0,
    },
    update: {
      status: ConfirmationStatus.PENDING,
      originalS3Key: video.s3Key,
      originalDurationMs: video.durationMs ?? 0,
      timestampMappings: mappings as unknown as Prisma.InputJsonValue,
      trimmedS3Key: null,
      trimmedDurationMs: null,
      progress: 0,
      error: null,
      confirmedAt: null,
    },
  });

  console.log(
    `[CONFIRMATION] Created job ${confirmation.id} for video ${videoId} with ${mappings.length} rallies`
  );

  // Trigger processing (async, don't wait)
  const useLocal = shouldUseLocalProcessing();
  const triggerFn = useLocal ? triggerLocalProcessing : triggerLambdaProcessing;

  triggerFn(confirmation.id, video.s3Key, mappings).catch((error) => {
    console.error(
      `[CONFIRMATION] Failed to trigger processing for job ${confirmation.id}:`,
      error
    );
    prisma.rallyConfirmation
      .update({
        where: { id: confirmation.id },
        data: {
          status: ConfirmationStatus.FAILED,
          error: `Failed to start processing: ${String(error)}`,
        },
      })
      .catch(console.error);
  });

  return {
    confirmationId: confirmation.id,
    status: confirmation.status,
    progress: confirmation.progress,
    createdAt: confirmation.createdAt,
  };
}

/**
 * Get confirmation status for a video.
 */
export async function getConfirmationStatus(videoId: string, userId: string) {
  const video = await prisma.video.findFirst({
    where: {
      id: videoId,
      OR: [
        { userId },
        {
          sessionVideos: {
            some: {
              session: {
                share: { members: { some: { userId } } },
              },
            },
          },
        },
      ],
    },
    include: {
      confirmation: true,
    },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  if (!video.confirmation) {
    return { videoId, confirmation: null };
  }

  const conf = video.confirmation;
  return {
    videoId,
    confirmation: {
      id: conf.id,
      status: conf.status,
      progress: conf.progress,
      error: conf.error,
      confirmedAt: conf.confirmedAt,
      originalDurationMs: conf.originalDurationMs,
      trimmedDurationMs: conf.trimmedDurationMs,
      timestampMappings:
        conf.status === "CONFIRMED"
          ? (conf.timestampMappings as unknown as TimestampMapping[])
          : undefined,
    },
  };
}

/**
 * Restore original video, deleting the trimmed version.
 */
export async function restoreOriginal(
  videoId: string,
  userId: string
): Promise<{ success: boolean }> {
  // Check user tier
  const userTier = await getUserTier(userId);
  if (userTier !== "PREMIUM") {
    throw new ForbiddenError("Rally confirmation requires Premium tier.");
  }

  const video = await prisma.video.findFirst({
    where: {
      id: videoId,
      userId,
      deletedAt: null,
    },
    include: {
      confirmation: true,
      rallies: true,
    },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  if (!video.confirmation) {
    throw new ValidationError("No confirmation exists for this video.");
  }

  if (video.confirmation.status !== "CONFIRMED") {
    throw new ValidationError(
      "Can only restore a confirmed video. Current status: " +
        video.confirmation.status
    );
  }

  const confirmation = video.confirmation;
  const mappings = confirmation.timestampMappings as unknown as TimestampMapping[];

  // Use a transaction to revert timestamps and clean up
  await prisma.$transaction(async (tx) => {
    // Batch update rally timestamps to original values in a single query
    if (mappings.length > 0) {
      const ids = mappings.map((m) => m.rallyId);
      const startCase = mappings
        .map((m) => `WHEN '${m.rallyId}'::uuid THEN ${m.originalStartMs}`)
        .join(" ");
      const endCase = mappings
        .map((m) => `WHEN '${m.rallyId}'::uuid THEN ${m.originalEndMs}`)
        .join(" ");
      await tx.$executeRawUnsafe(
        `UPDATE "rallies" SET "start_ms" = CASE id ${startCase} END, "end_ms" = CASE id ${endCase} END WHERE id = ANY($1::uuid[])`,
        ids
      );
    }

    // Update video to use original s3Key and duration
    await tx.video.update({
      where: { id: videoId },
      data: {
        s3Key: confirmation.originalS3Key,
        durationMs: confirmation.originalDurationMs,
      },
    });

    // Delete the confirmation record
    await tx.rallyConfirmation.delete({
      where: { id: confirmation.id },
    });
  });

  // Delete trimmed video from S3 (outside transaction, non-blocking)
  // Uses structured logging for failed deletions to enable cleanup via log analysis
  if (confirmation.trimmedS3Key) {
    deleteObject(confirmation.trimmedS3Key).catch((error) => {
      // Log with structured data for potential automated cleanup
      console.error(
        JSON.stringify({
          event: "S3_DELETE_FAILED",
          type: "orphaned_trimmed_video",
          s3Key: confirmation.trimmedS3Key,
          videoId,
          confirmationId: confirmation.id,
          error: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        })
      );
    });
  }

  console.log(`[CONFIRMATION] Restored original video ${videoId}`);

  return { success: true };
}

interface ConfirmationCompletePayload {
  confirmation_id: string;
  status: "completed" | "failed";
  error_message?: string;
  output_s3_key?: string;
  duration_ms?: number;
}

/**
 * Handle confirmation completion webhook from Lambda.
 */
export async function handleConfirmationComplete(
  payload: ConfirmationCompletePayload
) {
  const confirmation = await prisma.rallyConfirmation.findUnique({
    where: { id: payload.confirmation_id },
    include: {
      video: {
        include: { rallies: true },
      },
    },
  });

  if (!confirmation) {
    throw new NotFoundError("RallyConfirmation", payload.confirmation_id);
  }

  // Use atomic update to prevent race conditions - only proceed if status is still processable
  const allowedStatuses: ConfirmationStatus[] = [ConfirmationStatus.PENDING, ConfirmationStatus.PROCESSING];
  if (!allowedStatuses.includes(confirmation.status)) {
    return { ignored: true, reason: "Job already processed" };
  }

  if (payload.status === "failed") {
    // Atomic update with status check to prevent race condition
    const result = await prisma.rallyConfirmation.updateMany({
      where: {
        id: confirmation.id,
        status: { in: allowedStatuses },
      },
      data: {
        status: ConfirmationStatus.FAILED,
        error: payload.error_message ?? "Unknown error",
      },
    });

    if (result.count === 0) {
      return { ignored: true, reason: "Job already processed" };
    }

    return { success: false, error: payload.error_message };
  }

  // Success - update timestamps and mark as confirmed
  const mappings = confirmation.timestampMappings as unknown as TimestampMapping[];

  const transactionResult = await prisma.$transaction(async (tx) => {
    // Atomic claim - update status only if still in allowed state
    const claimed = await tx.rallyConfirmation.updateMany({
      where: {
        id: confirmation.id,
        status: { in: allowedStatuses },
      },
      data: {
        status: ConfirmationStatus.CONFIRMED,
        progress: 100,
        trimmedS3Key: payload.output_s3_key,
        trimmedDurationMs: payload.duration_ms,
        confirmedAt: new Date(),
      },
    });

    // If no rows updated, another request already processed this
    if (claimed.count === 0) {
      return { ignored: true };
    }

    // Batch update rally timestamps to trimmed values in a single query
    if (mappings.length > 0) {
      const ids = mappings.map((m) => m.rallyId);
      const startCase = mappings
        .map((m) => `WHEN '${m.rallyId}'::uuid THEN ${m.trimmedStartMs}`)
        .join(" ");
      const endCase = mappings
        .map((m) => `WHEN '${m.rallyId}'::uuid THEN ${m.trimmedEndMs}`)
        .join(" ");
      await tx.$executeRawUnsafe(
        `UPDATE "rallies" SET "start_ms" = CASE id ${startCase} END, "end_ms" = CASE id ${endCase} END WHERE id = ANY($1::uuid[])`,
        ids
      );
    }

    // Update video to use trimmed s3Key and duration
    await tx.video.update({
      where: { id: confirmation.videoId },
      data: {
        s3Key: payload.output_s3_key ?? confirmation.video.s3Key,
        durationMs: payload.duration_ms ?? confirmation.trimmedDurationMs,
      },
    });

    return { success: true };
  });

  if (transactionResult.ignored) {
    return { ignored: true, reason: "Job already processed" };
  }

  console.log(
    `[CONFIRMATION] Job ${confirmation.id} completed successfully`
  );

  return { success: true };
}

/**
 * Update confirmation progress.
 */
export async function updateConfirmationProgress(
  confirmationId: string,
  progress: number
) {
  await prisma.rallyConfirmation.update({
    where: { id: confirmationId },
    data: {
      progress: Math.min(100, Math.max(0, progress)),
    },
  });
}

// ============================================================================
// Local Processing (Development)
// ============================================================================

/**
 * Run FFmpeg locally for development.
 * Downloads video from S3, extracts rally clips, concatenates, and uploads result.
 */
async function triggerLocalProcessing(
  confirmationId: string,
  videoS3Key: string,
  mappings: TimestampMapping[]
): Promise<void> {
  console.log(`[LOCAL CONFIRMATION] Starting job ${confirmationId}`);

  // Update status to processing
  await prisma.rallyConfirmation.update({
    where: { id: confirmationId },
    data: { status: ConfirmationStatus.PROCESSING },
  });

  // Create temp directory
  const tmpDir = path.join(os.tmpdir(), `rallycut-confirm-${confirmationId}`);
  await fs.mkdir(tmpDir, { recursive: true });

  try {
    // Download video from S3
    const downloadUrl = await generateDownloadUrl(videoS3Key);
    const videoPath = path.join(tmpDir, "input.mp4");

    console.log(`[LOCAL CONFIRMATION] Downloading video...`);
    await updateConfirmationProgress(confirmationId, 5);

    const response = await fetch(downloadUrl);
    if (!response.ok) {
      throw new Error(`Failed to download video: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    await fs.writeFile(videoPath, Buffer.from(buffer));

    await updateConfirmationProgress(confirmationId, 15);

    // Extract clips for each rally
    const clipPaths: string[] = [];

    for (let i = 0; i < mappings.length; i++) {
      const mapping = mappings[i];
      const progress = 15 + Math.round((i / mappings.length) * 60);
      await updateConfirmationProgress(confirmationId, progress);

      const clipPath = path.join(tmpDir, `clip_${i}.mp4`);
      const startSec = mapping.originalStartMs / 1000;
      const durationSec =
        (mapping.originalEndMs - mapping.originalStartMs) / 1000;

      console.log(
        `[LOCAL CONFIRMATION] Extracting clip ${i + 1}/${mappings.length}...`
      );

      // Fast copy (no re-encoding)
      await runFFmpeg([
        "-ss",
        String(startSec),
        "-i",
        videoPath,
        "-t",
        String(durationSec),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-y",
        clipPath,
      ]);

      clipPaths.push(clipPath);
    }

    await updateConfirmationProgress(confirmationId, 80);

    // Concatenate clips
    const outputPath = path.join(tmpDir, "output.mp4");

    if (clipPaths.length === 1) {
      // Just rename single clip
      await fs.rename(clipPaths[0], outputPath);
    } else {
      // Create concat file
      const concatPath = path.join(tmpDir, "concat.txt");
      const concatContent = clipPaths.map((p) => `file '${p}'`).join("\n");
      await fs.writeFile(concatPath, concatContent);

      console.log(
        `[LOCAL CONFIRMATION] Concatenating ${clipPaths.length} clips...`
      );
      await runFFmpeg([
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concatPath,
        "-c",
        "copy",
        "-y",
        outputPath,
      ]);
    }

    await updateConfirmationProgress(confirmationId, 90);

    // Get output duration
    const durationMs = await getVideoDuration(outputPath);

    // Upload to S3
    const outputKey = `confirmations/${confirmationId}/trimmed.mp4`;
    console.log(`[LOCAL CONFIRMATION] Uploading to ${outputKey}...`);

    const outputData = await fs.readFile(outputPath);
    const uploadUrl = await generateUploadUrl({
      key: outputKey,
      contentType: "video/mp4",
      contentLength: outputData.length,
    });

    const uploadResponse = await fetch(uploadUrl, {
      method: "PUT",
      headers: { "Content-Type": "video/mp4" },
      body: outputData,
    });

    if (!uploadResponse.ok) {
      throw new Error(`Failed to upload: ${uploadResponse.status}`);
    }

    // Mark as complete via webhook handler
    await handleConfirmationComplete({
      confirmation_id: confirmationId,
      status: "completed",
      output_s3_key: outputKey,
      duration_ms: durationMs,
    });

    console.log(`[LOCAL CONFIRMATION] Job ${confirmationId} completed`);
  } catch (error) {
    console.error(`[LOCAL CONFIRMATION] Job ${confirmationId} failed:`, error);
    await prisma.rallyConfirmation.update({
      where: { id: confirmationId },
      data: {
        status: ConfirmationStatus.FAILED,
        error: String(error),
      },
    });
  } finally {
    // Cleanup temp directory
    try {
      await fs.rm(tmpDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Run FFmpeg command and wait for completion.
 */
function runFFmpeg(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn("ffmpeg", args, { stdio: ["ignore", "pipe", "pipe"] });

    let stderr = "";
    proc.stderr?.on("data", (data: Buffer) => {
      stderr += data.toString();
    });

    proc.on("error", (err) => {
      reject(new Error(`FFmpeg failed to start: ${err.message}`));
    });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(
          new Error(`FFmpeg exited with code ${code}: ${stderr.slice(-500)}`)
        );
      }
    });
  });
}

/**
 * Get video duration using ffprobe.
 */
function getVideoDuration(videoPath: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const proc = spawn("ffprobe", [
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      videoPath,
    ]);

    let stdout = "";
    proc.stdout?.on("data", (data: Buffer) => {
      stdout += data.toString();
    });

    proc.on("error", (err) => {
      reject(new Error(`ffprobe failed: ${err.message}`));
    });

    proc.on("close", (code) => {
      if (code === 0) {
        const durationSec = parseFloat(stdout.trim());
        resolve(Math.round(durationSec * 1000));
      } else {
        reject(new Error(`ffprobe exited with code ${code}`));
      }
    });
  });
}

// ============================================================================
// Lambda Processing (Production)
// ============================================================================

/**
 * Trigger Lambda for video processing.
 */
async function triggerLambdaProcessing(
  confirmationId: string,
  videoS3Key: string,
  mappings: TimestampMapping[]
): Promise<void> {
  const functionName = env.EXPORT_LAMBDA_FUNCTION_NAME!;

  const payload = {
    confirmationId,
    videoS3Key,
    rallies: mappings.map((m) => ({
      startMs: m.originalStartMs,
      endMs: m.originalEndMs,
    })),
    callbackUrl: `${env.API_BASE_URL}/v1/webhooks/confirmation-complete`,
    webhookSecret: env.MODAL_WEBHOOK_SECRET,
    s3Bucket: env.S3_BUCKET_NAME,
  };

  const command = new InvokeCommand({
    FunctionName: functionName,
    InvocationType: "Event", // Async invocation
    Payload: Buffer.from(JSON.stringify(payload)),
  });

  await lambdaClient.send(command);

  // Update status to processing
  await prisma.rallyConfirmation.update({
    where: { id: confirmationId },
    data: { status: ConfirmationStatus.PROCESSING },
  });

  console.log(`[CONFIRMATION] Triggered Lambda for job ${confirmationId}`);
}
