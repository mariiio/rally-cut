import { ProcessingStatus } from "@prisma/client";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { spawn } from "child_process";
import fs from "fs/promises";
import os from "os";
import path from "path";

import { env } from "../config/env.js";
import {
  generateDownloadUrl,
  uploadProcessedVideo,
  uploadPoster,
} from "../lib/s3.js";
import { prisma } from "../lib/prisma.js";
import { NotFoundError } from "../middleware/errorHandler.js";
import { getUserTier } from "./tierService.js";

const lambdaClient = new LambdaClient({
  region: env.AWS_REGION,
});

// Check if we should use local FFmpeg (development mode)
function shouldUseLocalProcessing(): boolean {
  // Use local if no Lambda configured
  if (!env.PROCESSING_LAMBDA_FUNCTION_NAME) {
    return true;
  }
  // Or if explicitly set
  if (process.env.USE_LOCAL_PROCESSING === "true") {
    return true;
  }
  return false;
}

/**
 * Generate poster image immediately after upload.
 * This is called synchronously from confirmVideoUpload for fast UX.
 * ~2-3 seconds even for large videos (only reads 1 frame).
 */
export async function generatePosterImmediate(
  videoId: string,
  s3Key: string
): Promise<void> {
  console.log(`[POSTER] Generating poster for video ${videoId}`);

  const tmpDir = path.join(os.tmpdir(), `rallycut-poster-${videoId}`);
  await fs.mkdir(tmpDir, { recursive: true });

  try {
    const inputPath = path.join(tmpDir, "input.mp4");
    const posterPath = path.join(tmpDir, "poster.jpg");

    // Download video from S3
    const downloadUrl = await generateDownloadUrl(s3Key);
    const response = await fetch(downloadUrl);
    if (!response.ok) {
      throw new Error(`Failed to download video: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    await fs.writeFile(inputPath, Buffer.from(buffer));

    // Generate S3 key for poster
    const keyParts = s3Key.split("/");
    const filename = keyParts.pop()!;
    const folder = keyParts.join("/");
    const [name] = filename.split(/\.(?=[^.]+$)/);
    const baseKey = folder ? `${folder}/${name}` : name;
    const posterKey = `${baseKey}_poster.jpg`;

    // Try to extract frame at 1 second, fall back to 0 for short videos
    try {
      await runFFmpeg([
        "-ss", "1",
        "-i", inputPath,
        "-vframes", "1",
        "-vf", "scale=1280:-1",
        "-q:v", "2",
        "-y", posterPath,
      ]);
    } catch {
      // Video might be shorter than 1 second, try at 0
      console.log(`[POSTER] Video ${videoId} may be short, trying frame at 0s`);
      await runFFmpeg([
        "-ss", "0",
        "-i", inputPath,
        "-vframes", "1",
        "-vf", "scale=1280:-1",
        "-q:v", "2",
        "-y", posterPath,
      ]);
    }

    // Upload poster to S3
    const posterData = await fs.readFile(posterPath);
    await uploadPoster(posterKey, posterData);

    // Update database
    await prisma.video.update({
      where: { id: videoId },
      data: { posterS3Key: posterKey },
    });

    console.log(`[POSTER] Generated poster for video ${videoId}: ${posterKey}`);
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
 * Queue a video for optimization processing.
 * Called after upload confirmation.
 */
export async function queueVideoProcessing(
  videoId: string,
  userId: string
): Promise<void> {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Skip if already processed or processing
  if (video.processingStatus !== ProcessingStatus.PENDING) {
    console.log(
      `[PROCESSING] Video ${videoId} already ${video.processingStatus}, skipping`
    );
    return;
  }

  // Mark as queued and store original file size
  await prisma.video.update({
    where: { id: videoId },
    data: {
      processingStatus: ProcessingStatus.QUEUED,
      originalFileSizeBytes: video.fileSizeBytes,
    },
  });

  const userTier = await getUserTier(userId);
  const useLocal = shouldUseLocalProcessing();

  console.log(
    `[PROCESSING] Queuing video ${videoId} for optimization (${useLocal ? "LOCAL" : "LAMBDA"})`
  );

  const triggerFn = useLocal ? triggerLocalProcessing : triggerLambdaProcessing;

  // Trigger processing (async, don't wait)
  triggerFn(videoId, video.s3Key, userTier).catch((error) => {
    console.error(
      `[PROCESSING] Failed to trigger for video ${videoId}:`,
      error
    );
    prisma.video
      .update({
        where: { id: videoId },
        data: {
          processingStatus: ProcessingStatus.FAILED,
          processingError: String(error),
        },
      })
      .catch(console.error);
  });
}

/**
 * Handle processing completion webhook from Lambda.
 */
export interface ProcessingCompletePayload {
  video_id: string;
  status: "completed" | "failed" | "skipped";
  processed_s3_key?: string;
  processed_size_bytes?: number;
  poster_s3_key?: string;
  proxy_s3_key?: string;
  proxy_size_bytes?: number;
  was_optimized?: boolean;
  error_message?: string;
}

export async function handleProcessingComplete(
  payload: ProcessingCompletePayload
): Promise<{ success: boolean; message?: string }> {
  // Use atomic updateMany with status condition to prevent TOCTOU race conditions
  // Only update if status is still pending/queued/processing (not already finalized)
  const allowedStatuses = [
    ProcessingStatus.PENDING,
    ProcessingStatus.QUEUED,
    ProcessingStatus.PROCESSING,
  ];

  if (payload.status === "failed") {
    const result = await prisma.video.updateMany({
      where: {
        id: payload.video_id,
        processingStatus: { in: allowedStatuses },
      },
      data: {
        processingStatus: ProcessingStatus.FAILED,
        processingError: payload.error_message ?? "Unknown error",
      },
    });

    if (result.count === 0) {
      // Either video not found or already processed
      const video = await prisma.video.findUnique({
        where: { id: payload.video_id },
        select: { id: true },
      });
      if (!video) {
        throw new NotFoundError("Video", payload.video_id);
      }
      return { success: true, message: "Already processed" };
    }

    console.log(`[PROCESSING] Video ${payload.video_id} failed: ${payload.error_message}`);
    return { success: false, message: payload.error_message };
  }

  if (payload.status === "skipped") {
    const skippedData: {
      processingStatus: ProcessingStatus;
      processedAt: Date;
      posterS3Key?: string;
    } = {
      processingStatus: ProcessingStatus.SKIPPED,
      processedAt: new Date(),
    };

    // Still save poster even when skipped
    if (payload.poster_s3_key) {
      skippedData.posterS3Key = payload.poster_s3_key;
    }

    const result = await prisma.video.updateMany({
      where: {
        id: payload.video_id,
        processingStatus: { in: allowedStatuses },
      },
      data: skippedData,
    });

    if (result.count === 0) {
      const video = await prisma.video.findUnique({
        where: { id: payload.video_id },
        select: { id: true },
      });
      if (!video) {
        throw new NotFoundError("Video", payload.video_id);
      }
      return { success: true, message: "Already processed" };
    }

    console.log(`[PROCESSING] Video ${payload.video_id} skipped (already optimized)`);
    return { success: true, message: "Skipped" };
  }

  // Success case - update with optimized video info
  const updateData: {
    processingStatus: ProcessingStatus;
    processedAt: Date;
    processedS3Key?: string;
    s3Key?: string;
    fileSizeBytes?: bigint;
    posterS3Key?: string;
    proxyS3Key?: string;
    proxyFileSizeBytes?: bigint;
    proxyGeneratedAt?: Date;
  } = {
    processingStatus: ProcessingStatus.COMPLETED,
    processedAt: new Date(),
  };

  // Only update s3Key and processedS3Key if optimization actually occurred
  if (payload.was_optimized && payload.processed_s3_key) {
    updateData.processedS3Key = payload.processed_s3_key;
    updateData.s3Key = payload.processed_s3_key; // Point to optimized version
  }

  if (payload.processed_size_bytes) {
    updateData.fileSizeBytes = BigInt(payload.processed_size_bytes);
  }

  // Always update poster if provided
  if (payload.poster_s3_key) {
    updateData.posterS3Key = payload.poster_s3_key;
  }

  // Update proxy if provided (PREMIUM tier eager generation)
  if (payload.proxy_s3_key) {
    updateData.proxyS3Key = payload.proxy_s3_key;
    updateData.proxyGeneratedAt = new Date();
    if (payload.proxy_size_bytes) {
      updateData.proxyFileSizeBytes = BigInt(payload.proxy_size_bytes);
    }
  }

  const result = await prisma.video.updateMany({
    where: {
      id: payload.video_id,
      processingStatus: { in: allowedStatuses },
    },
    data: updateData,
  });

  if (result.count === 0) {
    const video = await prisma.video.findUnique({
      where: { id: payload.video_id },
      select: { id: true },
    });
    if (!video) {
      throw new NotFoundError("Video", payload.video_id);
    }
    return { success: true, message: "Already processed" };
  }

  console.log(
    `[PROCESSING] Video ${payload.video_id} completed` +
      (payload.was_optimized ? ` (optimized to ${payload.processed_s3_key})` : " (no optimization needed)")
  );
  return { success: true };
}

/**
 * Trigger Lambda for video processing.
 */
async function triggerLambdaProcessing(
  videoId: string,
  s3Key: string,
  tier: string
): Promise<void> {
  const functionName = env.PROCESSING_LAMBDA_FUNCTION_NAME!;

  const payload = {
    videoId,
    originalS3Key: s3Key,
    s3Bucket: env.S3_BUCKET_NAME,
    callbackUrl: `${env.API_BASE_URL}/v1/webhooks/processing-complete`,
    webhookSecret: env.MODAL_WEBHOOK_SECRET,
    tier,
  };

  const command = new InvokeCommand({
    FunctionName: functionName,
    InvocationType: "Event", // Async invocation
    Payload: Buffer.from(JSON.stringify(payload)),
  });

  await lambdaClient.send(command);

  // Update status to processing
  await prisma.video.update({
    where: { id: videoId },
    data: { processingStatus: ProcessingStatus.PROCESSING },
  });

  console.log(`[PROCESSING] Triggered Lambda for video ${videoId}`);
}

/**
 * Run FFmpeg locally for development.
 * Downloads video from S3, optimizes it, generates poster and proxy.
 */
async function triggerLocalProcessing(
  videoId: string,
  s3Key: string,
  tier: string
): Promise<void> {
  console.log(`[LOCAL PROCESSING] Starting video ${videoId} (tier: ${tier})`);

  // Update status to processing
  await prisma.video.update({
    where: { id: videoId },
    data: { processingStatus: ProcessingStatus.PROCESSING },
  });

  // Create temp directory
  const tmpDir = path.join(os.tmpdir(), `rallycut-processing-${videoId}`);
  await fs.mkdir(tmpDir, { recursive: true });

  // Track poster key so we can preserve it even if optimization fails
  let uploadedPosterKey: string | null = null;

  try {
    const inputPath = path.join(tmpDir, "input.mp4");
    const outputPath = path.join(tmpDir, "output.mp4");
    const posterPath = path.join(tmpDir, "poster.jpg");
    const proxyPath = path.join(tmpDir, "proxy.mp4");

    // Download video from S3
    console.log(`[LOCAL PROCESSING] Downloading ${s3Key}...`);
    const downloadUrl = await generateDownloadUrl(s3Key);
    const response = await fetch(downloadUrl);
    if (!response.ok) {
      throw new Error(`Failed to download video: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    await fs.writeFile(inputPath, Buffer.from(buffer));

    const originalSize = (await fs.stat(inputPath)).size;

    // Generate S3 key base (same folder as original)
    const keyParts = s3Key.split("/");
    const filename = keyParts.pop()!;
    const folder = keyParts.join("/");
    const [name] = filename.split(/\.(?=[^.]+$)/);
    const baseKey = folder ? `${folder}/${name}` : name;
    const posterKey = `${baseKey}_poster.jpg`;

    // Check if poster was already generated by generatePosterImmediate()
    const existingVideo = await prisma.video.findUnique({
      where: { id: videoId },
      select: { posterS3Key: true },
    });

    if (existingVideo?.posterS3Key) {
      uploadedPosterKey = existingVideo.posterS3Key;
      console.log(`[LOCAL PROCESSING] Poster already exists: ${uploadedPosterKey}`);
    } else {
      // Generate poster if not already done
      console.log(`[LOCAL PROCESSING] Generating poster...`);
      await runFFmpeg([
        "-ss", "1",
        "-i", inputPath,
        "-vframes", "1",
        "-vf", "scale=1280:-1",
        "-q:v", "2",
        "-y", posterPath,
      ]);
      const posterData = await fs.readFile(posterPath);
      await uploadPoster(posterKey, posterData);
      uploadedPosterKey = posterKey;
      console.log(`[LOCAL PROCESSING] Uploaded poster to ${posterKey}`);
    }

    // Check if optimization is needed
    const needsOptimization = await checkNeedsOptimization(inputPath);

    if (!needsOptimization) {
      console.log(`[LOCAL PROCESSING] Video ${videoId} already optimized, skipping video processing`);
      await handleProcessingComplete({
        video_id: videoId,
        status: "skipped",
        poster_s3_key: posterKey,
        was_optimized: false,
      });
      return;
    }

    // Optimize video with FFmpeg
    // Use -pix_fmt yuv420p to handle 10-bit/HDR videos (libx264 only supports 8-bit)
    console.log(`[LOCAL PROCESSING] Optimizing video ${videoId}...`);
    await runFFmpeg([
      "-i", inputPath,
      "-c:v", "libx264",
      "-pix_fmt", "yuv420p",
      "-crf", "23",
      "-preset", "fast",
      "-tune", "film",
      "-profile:v", "high",
      "-level", "4.1",
      "-movflags", "+faststart",
      "-c:a", "aac",
      "-b:a", "128k",
      "-ac", "2",
      "-y", outputPath,
    ]);

    const processedSize = (await fs.stat(outputPath)).size;
    const reduction = ((1 - processedSize / originalSize) * 100).toFixed(1);
    console.log(
      `[LOCAL PROCESSING] Optimized ${videoId}: ${originalSize} -> ${processedSize} bytes (${reduction}% reduction)`
    );

    // Upload optimized video
    const processedKey = `${baseKey}_optimized.mp4`;
    console.log(`[LOCAL PROCESSING] Uploading optimized video to ${processedKey}...`);
    const outputData = await fs.readFile(outputPath);
    await uploadProcessedVideo(processedKey, outputData);

    // Generate 720p proxy for fast editing (all tiers)
    console.log(`[LOCAL PROCESSING] Generating 720p proxy...`);
    await runFFmpeg([
      "-i", outputPath,
      "-vf", "scale=-2:720",
      "-c:v", "libx264",
      "-pix_fmt", "yuv420p",
      "-crf", "28",
      "-preset", "fast",
      "-movflags", "+faststart",
      "-c:a", "aac",
      "-b:a", "96k",
      "-y", proxyPath,
    ]);

    const proxyKey = `${baseKey}_proxy.mp4`;
    const proxyData = await fs.readFile(proxyPath);
    const proxySizeBytes = proxyData.length;
    await uploadProcessedVideo(proxyKey, proxyData);
    console.log(`[LOCAL PROCESSING] Uploaded proxy to ${proxyKey} (${(proxySizeBytes / 1024 / 1024).toFixed(1)} MB)`);

    // Update via webhook handler
    await handleProcessingComplete({
      video_id: videoId,
      status: "completed",
      processed_s3_key: processedKey,
      processed_size_bytes: processedSize,
      poster_s3_key: posterKey,
      proxy_s3_key: proxyKey,
      proxy_size_bytes: proxySizeBytes,
      was_optimized: true,
    });

    console.log(`[LOCAL PROCESSING] Video ${videoId} completed successfully`);
  } catch (error) {
    console.error(`[LOCAL PROCESSING] Video ${videoId} failed:`, error);
    // If poster was uploaded before failure, still save it
    if (uploadedPosterKey) {
      console.log(`[LOCAL PROCESSING] Preserving poster ${uploadedPosterKey} despite failure`);
      await prisma.video.update({
        where: { id: videoId },
        data: {
          processingStatus: ProcessingStatus.FAILED,
          processingError: String(error),
          posterS3Key: uploadedPosterKey,
        },
      });
    } else {
      await handleProcessingComplete({
        video_id: videoId,
        status: "failed",
        error_message: String(error),
      });
    }
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
 * Check if video needs optimization by looking at moov atom position and bitrate.
 */
async function checkNeedsOptimization(videoPath: string): Promise<boolean> {
  try {
    // Check bitrate with ffprobe
    const ffprobeResult = await runFFprobe([
      "-v", "error",
      "-show_entries", "format=bit_rate",
      "-of", "json",
      videoPath,
    ]);

    const info = JSON.parse(ffprobeResult);
    const bitRate = parseInt(info?.format?.bit_rate ?? "0", 10);

    // If bitrate > 8 Mbps, optimize to reduce size
    if (bitRate > 8_000_000) {
      console.log(`[PROCESSING] High bitrate detected (${(bitRate / 1_000_000).toFixed(1)} Mbps)`);
      return true;
    }

    // Check moov atom position (first 64KB)
    const fileHandle = await fs.open(videoPath, "r");
    const buffer = Buffer.alloc(65536);
    await fileHandle.read(buffer, 0, 65536, 0);
    await fileHandle.close();

    const moovPos = buffer.indexOf("moov");
    if (moovPos === -1 || moovPos > 32768) {
      console.log(`[PROCESSING] moov atom not at start (pos: ${moovPos})`);
      return true;
    }

    return false;
  } catch (error) {
    console.log(`[PROCESSING] Error checking video, will optimize: ${error}`);
    return true;
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
        reject(new Error(`FFmpeg exited with code ${code}: ${stderr.slice(-500)}`));
      }
    });
  });
}

/**
 * Run FFprobe command and return output.
 */
function runFFprobe(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn("ffprobe", args, { stdio: ["ignore", "pipe", "pipe"] });

    let stdout = "";
    let stderr = "";
    proc.stdout?.on("data", (data: Buffer) => {
      stdout += data.toString();
    });
    proc.stderr?.on("data", (data: Buffer) => {
      stderr += data.toString();
    });

    proc.on("error", (err) => {
      reject(new Error(`FFprobe failed to start: ${err.message}`));
    });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`FFprobe exited with code ${code}: ${stderr.slice(-500)}`));
      }
    });
  });
}
