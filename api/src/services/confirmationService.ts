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
  // Also pass proxyS3Key so we can trim both videos in parallel
  const useLocal = shouldUseLocalProcessing();
  const triggerFn = useLocal ? triggerLocalProcessing : triggerLambdaProcessing;

  triggerFn(confirmation.id, video.s3Key, mappings, video.proxyS3Key ?? undefined).catch((error) => {
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
    // Batch update rally timestamps to original values
    // Match by trimmed timestamps (current state) since rally IDs can change
    for (const m of mappings) {
      const rally = await tx.rally.findFirst({
        where: {
          videoId,
          startMs: m.trimmedStartMs,
          endMs: m.trimmedEndMs,
        },
      });

      if (rally) {
        await tx.rally.update({
          where: { id: rally.id },
          data: { startMs: m.originalStartMs, endMs: m.originalEndMs },
        });
      }
      // Rally not found - may have been deleted, skip
    }

    // NOTE: With proxy-only confirmation, video record was never modified.
    // We only need to delete the confirmation record.

    // Delete the confirmation record
    await tx.rallyConfirmation.delete({
      where: { id: confirmation.id },
    });
  });

  // Delete confirmation proxy from S3 (outside transaction, non-blocking)
  // Uses structured logging for failed deletions to enable cleanup via log analysis
  if (confirmation.proxyS3Key) {
    deleteObject(confirmation.proxyS3Key).catch((error) => {
      // Log with structured data for potential automated cleanup
      console.error(
        JSON.stringify({
          event: "S3_DELETE_FAILED",
          type: "orphaned_confirmation_proxy",
          s3Key: confirmation.proxyS3Key,
          videoId,
          confirmationId: confirmation.id,
          error: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        })
      );
    });
  }

  // Also delete trimmed video if it exists (from legacy confirmations)
  if (confirmation.trimmedS3Key) {
    deleteObject(confirmation.trimmedS3Key).catch((error) => {
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
  proxy_s3_key?: string;  // Trimmed proxy video (if available)
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
        proxyS3Key: payload.proxy_s3_key ?? null,  // Store trimmed proxy if available
        confirmedAt: new Date(),
      },
    });

    // If no rows updated, another request already processed this
    if (claimed.count === 0) {
      return { ignored: true };
    }

    // Batch update rally timestamps to trimmed values
    // Match by original timestamps instead of ID, since rally IDs can change
    // if a sync happens between initiation and completion
    for (const m of mappings) {
      const rally = await tx.rally.findFirst({
        where: {
          videoId: confirmation.videoId,
          startMs: m.originalStartMs,
          endMs: m.originalEndMs,
        },
      });

      if (rally) {
        await tx.rally.update({
          where: { id: rally.id },
          data: { startMs: m.trimmedStartMs, endMs: m.trimmedEndMs },
        });
      }
      // Rally not found - may have been deleted during processing, skip
    }

    // NOTE: With proxy-only confirmation, we do NOT modify the Video record.
    // - video.s3Key stays pointing to original (for exports)
    // - video.durationMs stays as original duration
    // - video.proxyS3Key stays as original proxy
    // The confirmation proxy is stored in RallyConfirmation.proxyS3Key
    // and the videoService returns it for editing when confirmed.

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
 * Generates ONLY a 720p proxy for editing (no full-quality trimmed video).
 * The original video stays unchanged - exports use it directly with reverse-mapped timestamps.
 */
async function triggerLocalProcessing(
  confirmationId: string,
  videoS3Key: string,
  mappings: TimestampMapping[],
  _proxyS3Key?: string  // Ignored - we generate fresh proxy from original
): Promise<void> {
  console.log(`[LOCAL CONFIRMATION] Starting job ${confirmationId} (proxy-only mode)`);

  // Update status to processing
  await prisma.rallyConfirmation.update({
    where: { id: confirmationId },
    data: { status: ConfirmationStatus.PROCESSING },
  });

  // Create temp directory
  const tmpDir = path.join(os.tmpdir(), `rallycut-confirm-${confirmationId}`);
  await fs.mkdir(tmpDir, { recursive: true });

  try {
    // Download original video from S3
    const videoPath = path.join(tmpDir, "input.mp4");

    console.log(`[LOCAL CONFIRMATION] Downloading video...`);
    await updateConfirmationProgress(confirmationId, 5);

    const downloadUrl = await generateDownloadUrl(videoS3Key);
    const response = await fetch(downloadUrl);
    if (!response.ok) {
      throw new Error(`Failed to download video: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    await fs.writeFile(videoPath, Buffer.from(buffer));

    await updateConfirmationProgress(confirmationId, 15);

    // Extract clips at 720p for each rally
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
        `[LOCAL CONFIRMATION] Extracting 720p clip ${i + 1}/${mappings.length}...`
      );

      // Extract at 720p (re-encode for proxy)
      await runFFmpeg([
        "-ss", String(startSec),
        "-i", videoPath,
        "-t", String(durationSec),
        "-vf", "scale=-2:720",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "28",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "96k",
        "-avoid_negative_ts", "make_zero",
        "-y", clipPath,
      ]);

      clipPaths.push(clipPath);
    }

    await updateConfirmationProgress(confirmationId, 80);

    // Concatenate clips into single proxy
    const outputPath = path.join(tmpDir, "proxy.mp4");

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

      // Use stream copy for concatenation (clips are already 720p)
      await runFFmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", concatPath,
        "-c", "copy",
        "-movflags", "+faststart",
        "-y", outputPath,
      ]);
    }

    await updateConfirmationProgress(confirmationId, 90);

    // Get output duration
    const durationMs = await getVideoDuration(outputPath);

    // Upload proxy to S3
    const proxyOutputKey = `confirmations/${confirmationId}/proxy.mp4`;

    console.log(`[LOCAL CONFIRMATION] Uploading proxy to ${proxyOutputKey}...`);

    const outputData = await fs.readFile(outputPath);
    const uploadUrl = await generateUploadUrl({
      key: proxyOutputKey,
      contentType: "video/mp4",
      contentLength: outputData.length,
    });

    const uploadResponse = await fetch(uploadUrl, {
      method: "PUT",
      headers: { "Content-Type": "video/mp4" },
      body: outputData,
    });

    if (!uploadResponse.ok) {
      throw new Error(`Failed to upload proxy: ${uploadResponse.status}`);
    }

    // Mark as complete via webhook handler
    // NOTE: output_s3_key is null (no full-quality trimmed video)
    // Only proxy_s3_key is set
    await handleConfirmationComplete({
      confirmation_id: confirmationId,
      status: "completed",
      output_s3_key: undefined,  // No full-quality trimmed video
      duration_ms: durationMs,
      proxy_s3_key: proxyOutputKey,
    });

    console.log(`[LOCAL CONFIRMATION] Job ${confirmationId} completed (proxy only)`);
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
 * Uses the same export Lambda with isConfirmation flag for proxy-only output.
 */
async function triggerLambdaProcessing(
  confirmationId: string,
  videoS3Key: string,
  mappings: TimestampMapping[],
  _proxyS3Key?: string  // Ignored - Lambda generates fresh proxy
): Promise<void> {
  const functionName = env.EXPORT_LAMBDA_FUNCTION_NAME!;

  // Send export-compatible payload with isConfirmation flag
  // Lambda will generate ONLY a 720p proxy (no full-quality trimmed video)
  const payload = {
    jobId: confirmationId,  // Lambda expects jobId
    tier: "PREMIUM",        // Confirmation is PREMIUM-only
    format: "mp4",
    isConfirmation: true,   // Enables proxy-only confirmation mode in Lambda
    proxyOnly: true,        // Explicitly request proxy-only output
    rallies: mappings.map((m) => ({
      videoS3Key,           // Same video for all rallies in confirmation
      startMs: m.originalStartMs,
      endMs: m.originalEndMs,
      exportQuality: "720p",  // Generate 720p proxy for editing
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

  console.log(`[CONFIRMATION] Triggered Lambda for job ${confirmationId} (proxy-only mode)`);
}
