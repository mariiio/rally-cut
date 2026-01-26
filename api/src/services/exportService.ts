import { ConfirmationStatus, ExportStatus } from "@prisma/client";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { spawn } from "child_process";
import fs from "fs/promises";
import os from "os";
import path from "path";

import { env } from "../config/env.js";
import { generateDownloadUrl, generateUploadUrl } from "../lib/s3.js";
import { prisma } from "../lib/prisma.js";
import { ForbiddenError, NotFoundError } from "../middleware/errorHandler.js";
import { getUserTier, getTierLimits, type UserTier } from "./tierService.js";
import type { TimestampMapping } from "./confirmationService.js";

const lambdaClient = new LambdaClient({
  region: env.AWS_REGION,
});

type ExportQuality = "original" | "720p";

/**
 * Determine export quality for a video based on tier, upload age, and user preference.
 * FREE users get original quality exports for their grace period (originalQualityDays), then 720p.
 * PRO and ELITE users get original quality based on their tier's originalQualityDays.
 * User can request lower quality (720p) but cannot request original if not allowed by tier.
 */
function getExportQuality(
  tier: UserTier,
  videoCreatedAt: Date,
  userRequestedQuality?: ExportQuality
): ExportQuality {
  const limits = getTierLimits(tier);

  // If user explicitly requested 720p, honor that
  if (userRequestedQuality === "720p") {
    return "720p";
  }

  // If tier has no original quality limit (null), always allow original
  if (limits.originalQualityDays === null) return "original";

  // Check if within grace period
  const daysSinceUpload = (Date.now() - videoCreatedAt.getTime()) / (24 * 60 * 60 * 1000);
  const withinGracePeriod = daysSinceUpload <= limits.originalQualityDays;

  // FREE tier cannot get original quality after grace period
  if (userRequestedQuality === "original" && !limits.lambdaExportEnabled && !withinGracePeriod) {
    return "720p";
  }

  return withinGracePeriod ? "original" : "720p";
}

// Check if we should use local FFmpeg (development mode)
function shouldUseLocalExport(): boolean {
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

// Camera keyframe for FFmpeg filter generation
interface CameraKeyframe {
  timeOffset: number;  // 0.0-1.0 within rally
  positionX: number;   // 0.0-1.0
  positionY: number;   // 0.0-1.0
  zoom: number;        // 1.0-3.0
  rotation: number;    // degrees, typically -30 to +30
  easing: "LINEAR" | "EASE_IN" | "EASE_OUT" | "EASE_IN_OUT";
}

interface CameraEdit {
  aspectRatio: "ORIGINAL" | "VERTICAL";  // VERTICAL reserved for future use
  keyframes: CameraKeyframe[];
}

interface CreateExportJobInput {
  sessionId: string;
  // tier is NOT in input - determined by backend from user
  config: {
    format: "mp4" | "webm";
    quality?: ExportQuality;  // User requested quality
  };
  rallies: Array<{
    videoId: string;
    videoS3Key: string;
    startMs: number;
    endMs: number;
    camera?: CameraEdit;
  }>;
}

// Internal interface with tier and exportQuality for trigger functions
interface ExportTriggerInput {
  sessionId: string;
  tier: UserTier;
  config: {
    format: "mp4" | "webm";
  };
  rallies: Array<{
    videoId: string;
    videoS3Key: string;
    startMs: number;
    endMs: number;
    camera?: CameraEdit;
    exportQuality: ExportQuality; // Per-rally quality based on video age
  }>;
}

/**
 * Generate FFmpeg video filter for camera panning/zooming.
 * Uses crop filter with expressions to animate position based on keyframes.
 *
 * @param camera - Camera edit configuration
 * @param durationSec - Rally duration in seconds
 * @param inputWidth - Source video width (default 1920)
 * @param inputHeight - Source video height (default 1080)
 * @param targetHeight - Optional target output height (e.g., 720 for 720p)
 * @returns FFmpeg -vf filter string
 */
function generateCameraFilter(
  camera: CameraEdit,
  durationSec: number,
  inputWidth = 1920,
  inputHeight = 1080,
  targetHeight?: number
): string {
  // Use time-based expressions (t) instead of frame-based (n) to be FPS-independent

  // Calculate output dimensions based on aspect ratio
  let outputWidth: number;
  let outputHeight: number;

  if (camera.aspectRatio === "VERTICAL") {
    // 9:16 vertical from 16:9 source
    outputHeight = inputHeight;
    outputWidth = Math.round(outputHeight * 9 / 16);
  } else {
    // Keep original aspect ratio
    outputWidth = inputWidth;
    outputHeight = inputHeight;
  }

  const keyframes = camera.keyframes;

  // If no keyframes, use static center crop
  if (!keyframes || keyframes.length === 0) {
    const x = Math.round((inputWidth - outputWidth) / 2);
    const y = Math.round((inputHeight - outputHeight) / 2);
    const scaleExpr = targetHeight ? `,scale=-2:${targetHeight}` : '';
    return `crop=${outputWidth}:${outputHeight}:${x}:${y}${scaleExpr}`;
  }

  // Sort keyframes by time
  const sortedKeyframes = [...keyframes].sort((a, b) => a.timeOffset - b.timeOffset);

  // For single keyframe, use static position with zoom and rotation
  if (sortedKeyframes.length === 1) {
    const kf = sortedKeyframes[0];
    const zoom = Math.max(1, Math.min(3, kf.zoom));
    const rotation = kf.rotation ?? 0;
    const croppedW = Math.round(outputWidth / zoom);
    const croppedH = Math.round(outputHeight / zoom);
    const maxX = inputWidth - croppedW;
    const maxY = inputHeight - croppedH;
    const x = Math.round(kf.positionX * maxX);
    const y = Math.round(kf.positionY * maxY);
    const scaleExpr = targetHeight
      ? `scale=-2:${targetHeight}`
      : `scale=${outputWidth}:${outputHeight}`;
    let filter = `crop=${croppedW}:${croppedH}:${x}:${y},${scaleExpr}`;
    // Add rotation filter if non-zero (angle in radians, fill with transparent black)
    if (Math.abs(rotation) > 0.01) {
      const angleRad = rotation * Math.PI / 180;
      filter += `,rotate=${angleRad}:ow=iw:oh=ih:c=black@0`;
    }
    return filter;
  }

  // Build FFmpeg expression for animated crop
  // We'll generate a piecewise expression using if() functions

  // Helper to generate easing expression (reserved for future animated crop)
  const _easingExpr = (t: string, easing: string): string => {
    switch (easing) {
      case "EASE_IN":
        return `(${t}*${t})`;
      case "EASE_OUT":
        return `(1-(1-${t})*(1-${t}))`;
      case "EASE_IN_OUT":
        return `(${t}<0.5?2*${t}*${t}:1-(-2*${t}+2)*(-2*${t}+2)/2)`;
      default: // LINEAR
        return t;
    }
  };

  // Build zoom, x, y expressions as piecewise using time (t) instead of frame number (n)
  // This makes the filter FPS-independent
  const buildPiecewiseExpr = (
    keyframes: CameraKeyframe[],
    getValue: (kf: CameraKeyframe) => number,
    durationSec: number
  ): string => {
    if (keyframes.length === 1) {
      return String(getValue(keyframes[0]));
    }

    // Check if all values are the same (constant) - no interpolation needed
    const values = keyframes.map(getValue);
    const allSame = values.every(v => Math.abs(v - values[0]) < 0.0001);
    if (allSame) {
      return String(values[0]);
    }

    // Build piecewise expression with proper segment boundaries
    // Each segment: if(lt(t, endTime), segmentExpr, nextSegment)
    // Using time (t) in seconds for FPS independence
    const segments: Array<{ endTime: number; expr: string }> = [];

    for (let i = 0; i < keyframes.length - 1; i++) {
      const kf1 = keyframes[i];
      const kf2 = keyframes[i + 1];
      // Convert timeOffset (0-1) to actual time in seconds
      const time1 = kf1.timeOffset * durationSec;
      const time2 = kf2.timeOffset * durationSec;
      const v1 = getValue(kf1);
      const v2 = getValue(kf2);

      // If values are the same, use constant for this segment
      if (Math.abs(v2 - v1) < 0.0001) {
        segments.push({ endTime: time2, expr: String(v1) });
        continue;
      }

      const timeDiff = Math.max(0.001, time2 - time1);
      // Linear interpolation: progress = (t - start) / (end - start), clamped to [0,1]
      const progressExpr = `min(1,max(0,(t-${time1.toFixed(4)})/${timeDiff.toFixed(4)}))`;
      const interpExpr = `(${v1}+${progressExpr}*(${v2}-${v1}))`;

      segments.push({ endTime: time2, expr: interpExpr });
    }

    if (segments.length === 0) {
      return String(values[0]);
    }

    // Build final expression from inside out:
    // if(lt(t,t1),expr1,if(lt(t,t2),expr2,...,lastValue))
    const lastValue = getValue(keyframes[keyframes.length - 1]);
    let expr = String(lastValue);
    for (let i = segments.length - 1; i >= 0; i--) {
      const seg = segments[i];
      expr = `if(lt(t,${seg.endTime.toFixed(4)}),${seg.expr},${expr})`;
    }

    return expr;
  };

  const zoomExpr = buildPiecewiseExpr(sortedKeyframes, kf => Math.max(1, Math.min(3, kf.zoom)), durationSec);
  const pxExpr = buildPiecewiseExpr(sortedKeyframes, kf => kf.positionX, durationSec);
  const pyExpr = buildPiecewiseExpr(sortedKeyframes, kf => kf.positionY, durationSec);
  const rotExpr = buildPiecewiseExpr(sortedKeyframes, kf => kf.rotation ?? 0, durationSec);

  // Check if any keyframe has non-zero rotation
  const hasRotation = sortedKeyframes.some(kf => Math.abs(kf.rotation ?? 0) > 0.01);

  // Calculate base crop dimensions (at zoom=1) for output aspect ratio
  const baseW = outputWidth;
  const baseH = outputHeight;

  // X position: normalize positionX to available pan range
  // At zoom Z, crop size is baseW/Z x baseH/Z
  // Available X range: 0 to (inputW - baseW/Z)
  // X = positionX * (inputW - baseW/zoom)
  const xExprFinal = `floor((${pxExpr})*(${inputWidth}-${baseW}/(${zoomExpr})))`;
  const yExprFinal = `floor((${pyExpr})*(${inputHeight}-${baseH}/(${zoomExpr})))`;
  const wExprFinal = `floor(${baseW}/(${zoomExpr}))`;
  const hExprFinal = `floor(${baseH}/(${zoomExpr}))`;

  // Use crop with expressions, then scale to output size
  // If targetHeight specified, scale to that (maintaining aspect ratio)
  const scaleExpr = targetHeight
    ? `scale=-2:${targetHeight}`
    : `scale=${outputWidth}:${outputHeight}`;
  let filter = `crop=w='${wExprFinal}':h='${hExprFinal}':x='${xExprFinal}':y='${yExprFinal}',${scaleExpr}`;

  // Add rotation filter if any keyframe has rotation
  // Convert degrees to radians: angle * PI / 180
  if (hasRotation) {
    const rotRadExpr = `(${rotExpr})*PI/180`;
    filter += `,rotate='${rotRadExpr}':ow=iw:oh=ih:c=black@0`;
  }

  return filter;
}

export async function createExportJob(
  userId: string,
  input: CreateExportJobInput
) {
  console.log('[EXPORT] Received export request:', JSON.stringify({
    config: input.config,
    ralliesCount: input.rallies.length,
    rallies: input.rallies.map(r => ({
      videoId: r.videoId,
      hasCamera: !!r.camera,
      cameraKeyframes: r.camera?.keyframes?.length ?? 0,
    })),
  }, null, 2));

  // Get user's actual tier - don't trust frontend tier parameter
  const userTier = await getUserTier(userId);
  const limits = getTierLimits(userTier);

  // Force tier to match user's actual tier (prevent frontend override)
  const effectiveTier = userTier;

  // Check if user can use Lambda export
  const useLocal = shouldUseLocalExport();
  if (!useLocal && !limits.lambdaExportEnabled) {
    // FREE users cannot use Lambda export - they must use browser export
    throw new ForbiddenError(
      "Server-side export requires a paid tier (Pro or Elite). Please use browser export instead."
    );
  }

  // Verify session exists and user has access
  const session = await prisma.session.findFirst({
    where: {
      id: input.sessionId,
      deletedAt: null,
      OR: [
        { userId },
        { share: { members: { some: { userId } } } },
      ],
    },
  });

  if (!session) {
    throw new NotFoundError("Session", input.sessionId);
  }

  // Look up video createdAt and confirmations for each rally
  const videoIds = [...new Set(input.rallies.map((r) => r.videoId))];
  const videos = await prisma.video.findMany({
    where: { id: { in: videoIds } },
    select: { id: true, createdAt: true },
  });
  const videoCreatedAtMap = new Map(videos.map((v) => [v.id, v.createdAt]));

  // Look up confirmations for confirmed videos (for timestamp reverse-mapping)
  const confirmations = await prisma.rallyConfirmation.findMany({
    where: {
      videoId: { in: videoIds },
      status: ConfirmationStatus.CONFIRMED,
    },
    select: {
      videoId: true,
      originalS3Key: true,
      timestampMappings: true,
    },
  });
  const confirmationMap = new Map(confirmations.map((c) => [c.videoId, c]));

  // Add exportQuality and reverse-map timestamps for confirmed videos
  const userRequestedQuality = input.config.quality;
  console.log(`[EXPORT] User requested quality: ${userRequestedQuality}, tier: ${effectiveTier}`);

  const ralliesWithQuality = input.rallies.map((rally) => {
    const videoCreatedAt = videoCreatedAtMap.get(rally.videoId);
    const exportQuality = videoCreatedAt
      ? getExportQuality(effectiveTier, videoCreatedAt, userRequestedQuality)
      : "720p"; // Default to 720p if video not found

    console.log(`[EXPORT] Rally quality: requested=${userRequestedQuality}, effective=${exportQuality}, hasCamera=${!!rally.camera}, keyframes=${rally.camera?.keyframes?.length ?? 0}`);

    // Check if this video is confirmed
    const confirmation = confirmationMap.get(rally.videoId);
    if (confirmation) {
      // Reverse-map timestamps: find the mapping that matches this rally's trimmed timestamps
      const mappings = confirmation.timestampMappings as unknown as TimestampMapping[];
      const mapping = mappings.find(
        (m) => m.trimmedStartMs === rally.startMs && m.trimmedEndMs === rally.endMs
      );

      if (mapping) {
        // Use original video and original timestamps for export
        console.log(
          `[EXPORT] Reverse-mapping confirmed video ${rally.videoId}: ` +
          `${rally.startMs}-${rally.endMs} -> ${mapping.originalStartMs}-${mapping.originalEndMs}`
        );
        return {
          ...rally,
          videoS3Key: confirmation.originalS3Key,  // Use original video
          startMs: mapping.originalStartMs,         // Use original timestamps
          endMs: mapping.originalEndMs,
          exportQuality,
        };
      }
      // Mapping not found - rally might have been modified, use as-is
      console.warn(
        `[EXPORT] No mapping found for confirmed rally ${rally.startMs}-${rally.endMs} on video ${rally.videoId}`
      );
    }

    return { ...rally, exportQuality };
  });

  // Create the export job with user's actual tier
  const job = await prisma.exportJob.create({
    data: {
      sessionId: input.sessionId,
      userId,
      tier: effectiveTier,
      status: ExportStatus.PENDING,
      config: input.config,
      rallies: input.rallies as unknown as Parameters<typeof prisma.exportJob.create>[0]['data']['rallies'],
    },
  });

  console.log(`[EXPORT] Using ${useLocal ? "LOCAL" : "LAMBDA"} export for job ${job.id} (tier: ${effectiveTier})`);

  const triggerFn = useLocal ? triggerLocalExport : triggerExportLambda;

  // Build export input with tier and per-rally exportQuality
  const exportInput: ExportTriggerInput = {
    sessionId: input.sessionId,
    tier: effectiveTier,
    config: input.config,
    rallies: ralliesWithQuality,
  };

  // Trigger export (async, don't wait)
  triggerFn(job.id, exportInput).catch((error) => {
    console.error(`[EXPORT] Failed to trigger export for job ${job.id}:`, error);
    // Update job status to failed
    prisma.exportJob
      .update({
        where: { id: job.id },
        data: {
          status: ExportStatus.FAILED,
          error: `Failed to start export: ${String(error)}`,
        },
      })
      .catch(console.error);
  });

  return {
    id: job.id,
    status: job.status,
    progress: job.progress,
    createdAt: job.createdAt,
  };
}

/**
 * Run FFmpeg locally for development.
 * Downloads videos from S3, processes them, and uploads the result.
 */
async function triggerLocalExport(
  jobId: string,
  input: ExportTriggerInput
): Promise<void> {
  console.log(`[LOCAL EXPORT] Starting job ${jobId}`);

  // Update status to processing
  await prisma.exportJob.update({
    where: { id: jobId },
    data: { status: ExportStatus.PROCESSING },
  });

  // Create temp directory
  const tmpDir = path.join(os.tmpdir(), `rallycut-export-${jobId}`);
  await fs.mkdir(tmpDir, { recursive: true });

  try {
    const clipPaths: string[] = [];

    // Pre-download all unique videos in parallel for better performance
    const uniqueS3Keys = [...new Set(input.rallies.map((r) => r.videoS3Key))];
    const videoPathMap = new Map<string, string>();

    console.log(`[LOCAL EXPORT] Downloading ${uniqueS3Keys.length} unique videos in parallel...`);
    await updateExportProgress(jobId, 5);

    // Download videos in parallel with concurrency limit of 3
    const downloadVideo = async (s3Key: string, index: number): Promise<void> => {
      const downloadUrl = await generateDownloadUrl(s3Key);
      const videoPath = path.join(tmpDir, `video_${index}.mp4`);

      const response = await fetch(downloadUrl);
      if (!response.ok) {
        throw new Error(`Failed to download video ${s3Key}: ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      await fs.writeFile(videoPath, Buffer.from(buffer));
      videoPathMap.set(s3Key, videoPath);
    };

    // Parallel download with concurrency limit
    const CONCURRENCY_LIMIT = 3;
    for (let i = 0; i < uniqueS3Keys.length; i += CONCURRENCY_LIMIT) {
      const batch = uniqueS3Keys.slice(i, i + CONCURRENCY_LIMIT);
      await Promise.all(batch.map((key, idx) => downloadVideo(key, i + idx)));
      const downloadProgress = Math.round(((i + batch.length) / uniqueS3Keys.length) * 30);
      await updateExportProgress(jobId, 5 + downloadProgress);
    }

    console.log(`[LOCAL EXPORT] All videos downloaded, extracting ${input.rallies.length} clips...`);

    // Process each rally clip sequentially (FFmpeg is CPU-intensive)
    for (let i = 0; i < input.rallies.length; i++) {
      const rally = input.rallies[i];
      const progress = 35 + Math.round((i / input.rallies.length) * 45);
      await updateExportProgress(jobId, progress);

      const videoPath = videoPathMap.get(rally.videoS3Key)!;
      const clipPath = path.join(tmpDir, `clip_${i}.mp4`);

      // Extract clip with FFmpeg
      const startSec = rally.startMs / 1000;
      const durationSec = (rally.endMs - rally.startMs) / 1000;

      // Check if this rally has camera edits
      const hasCamera = rally.camera?.keyframes && rally.camera.keyframes.length > 0;

      console.log(`[LOCAL EXPORT] Extracting clip ${i + 1}/${input.rallies.length}${hasCamera ? ' with camera effects' : ''}...`);

      // Use per-rally exportQuality (original or 720p based on video age)
      const quality = rally.exportQuality;

      if (hasCamera) {
        // Camera edits require re-encoding
        // Pass target height for 720p quality to avoid double scaling
        const targetHeight = quality === "720p" ? 720 : undefined;
        const vf = generateCameraFilter(rally.camera!, durationSec, 1920, 1080, targetHeight);
        console.log(`[LOCAL EXPORT] Camera filter (${quality}): ${vf}`);

        await runFFmpeg([
          "-ss", String(startSec),
          "-i", videoPath,
          "-t", String(durationSec),
          "-vf", vf,
          "-c:v", "libx264",
          "-preset", "fast",
          "-crf", "23",
          "-c:a", "aac",
          "-b:a", "128k",
          "-avoid_negative_ts", "make_zero",
          "-y", clipPath,
        ]);
      } else if (quality === "original") {
        // Fast copy for original quality (PREMIUM or FREE within grace period)
        await runFFmpeg([
          "-ss", String(startSec),
          "-i", videoPath,
          "-t", String(durationSec),
          "-c", "copy",
          "-avoid_negative_ts", "make_zero",
          "-y", clipPath,
        ]);
      } else {
        // Re-encode to 720p (FREE after grace period, no watermark in local dev)
        await runFFmpeg([
          "-ss", String(startSec),
          "-i", videoPath,
          "-t", String(durationSec),
          "-vf", "scale=-2:720",
          "-c:v", "libx264",
          "-preset", "fast",
          "-crf", "23",
          "-c:a", "aac",
          "-b:a", "128k",
          "-avoid_negative_ts", "make_zero",
          "-y", clipPath,
        ]);
      }

      clipPaths.push(clipPath);
    }

    // Clean up downloaded videos
    for (const videoPath of videoPathMap.values()) {
      await fs.unlink(videoPath).catch((err) => {
        console.error(`[export] Failed to clean up ${videoPath}:`, err);
      });
    }

    await updateExportProgress(jobId, 85);

    // Concatenate clips
    const outputPath = path.join(tmpDir, `output.${input.config.format}`);

    if (clipPaths.length === 1) {
      // Just rename single clip
      await fs.rename(clipPaths[0], outputPath);
    } else {
      // Create concat file
      const concatPath = path.join(tmpDir, "concat.txt");
      const concatContent = clipPaths.map((p) => `file '${p}'`).join("\n");
      await fs.writeFile(concatPath, concatContent);

      console.log(`[LOCAL EXPORT] Concatenating ${clipPaths.length} clips...`);
      await runFFmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", concatPath,
        "-c", "copy",
        "-y", outputPath,
      ]);
    }

    await updateExportProgress(jobId, 95);

    // Upload to S3
    const outputKey = `exports/${jobId}/output.${input.config.format}`;
    console.log(`[LOCAL EXPORT] Uploading to ${outputKey}...`);

    const outputData = await fs.readFile(outputPath);
    const uploadUrl = await generateUploadUrl({
      key: outputKey,
      contentType: `video/${input.config.format}`,
      contentLength: outputData.length,
    });

    const uploadResponse = await fetch(uploadUrl, {
      method: "PUT",
      headers: { "Content-Type": `video/${input.config.format}` },
      body: outputData,
    });

    if (!uploadResponse.ok) {
      throw new Error(`Failed to upload: ${uploadResponse.status}`);
    }

    // Mark as complete
    await prisma.exportJob.update({
      where: { id: jobId },
      data: {
        status: ExportStatus.COMPLETED,
        progress: 100,
        outputKey,
      },
    });

    console.log(`[LOCAL EXPORT] Job ${jobId} completed successfully`);
  } catch (error) {
    console.error(`[LOCAL EXPORT] Job ${jobId} failed:`, error);
    await prisma.exportJob.update({
      where: { id: jobId },
      data: {
        status: ExportStatus.FAILED,
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
 * Run FFmpeg command and wait for completion
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

async function triggerExportLambda(jobId: string, input: ExportTriggerInput) {
  const functionName = env.EXPORT_LAMBDA_FUNCTION_NAME!;

  const payload = {
    jobId,
    tier: input.tier,
    format: input.config.format,
    rallies: input.rallies.map((r) => ({
      videoS3Key: r.videoS3Key,
      startMs: r.startMs,
      endMs: r.endMs,
      exportQuality: r.exportQuality, // Per-rally quality based on video age
      // Include camera edit if present
      ...(r.camera?.keyframes?.length && {
        camera: {
          aspectRatio: r.camera.aspectRatio,
          keyframes: r.camera.keyframes,
        },
      }),
    })),
    callbackUrl: `${env.API_BASE_URL}/v1/webhooks/export-complete`,
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
  await prisma.exportJob.update({
    where: { id: jobId },
    data: { status: ExportStatus.PROCESSING },
  });

  console.log(`[EXPORT] Triggered Lambda for job ${jobId}`);
}

export async function getExportJob(jobId: string, userId: string) {
  const job = await prisma.exportJob.findFirst({
    where: {
      id: jobId,
      userId,
    },
  });

  if (!job) {
    return null;
  }

  return {
    id: job.id,
    status: job.status,
    tier: job.tier,
    progress: job.progress,
    error: job.error,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
  };
}

export async function getExportDownloadUrl(jobId: string, userId: string) {
  const job = await prisma.exportJob.findFirst({
    where: {
      id: jobId,
      userId,
    },
  });

  if (!job) {
    return null;
  }

  if (job.status !== ExportStatus.COMPLETED || !job.outputKey) {
    return {
      id: job.id,
      status: job.status,
      downloadUrl: null,
    };
  }

  const downloadUrl = await generateDownloadUrl(job.outputKey);

  return {
    id: job.id,
    status: job.status,
    downloadUrl,
  };
}

export async function updateExportProgress(jobId: string, progress: number) {
  await prisma.exportJob.update({
    where: { id: jobId },
    data: {
      progress: Math.min(100, Math.max(0, progress)),
    },
  });
}

interface ExportCompletePayload {
  job_id: string;
  status: "completed" | "failed";
  error_message?: string;
  output_s3_key?: string;
}

export async function handleExportComplete(payload: ExportCompletePayload) {
  const job = await prisma.exportJob.findUnique({
    where: { id: payload.job_id },
  });

  if (!job) {
    throw new NotFoundError("ExportJob", payload.job_id);
  }

  if (job.status === ExportStatus.COMPLETED || job.status === ExportStatus.FAILED) {
    return { ignored: true, reason: "Job already processed" };
  }

  if (payload.status === "failed") {
    await prisma.exportJob.update({
      where: { id: job.id },
      data: {
        status: ExportStatus.FAILED,
        error: payload.error_message ?? "Unknown error",
      },
    });

    return { success: false, error: payload.error_message };
  }

  await prisma.exportJob.update({
    where: { id: job.id },
    data: {
      status: ExportStatus.COMPLETED,
      progress: 100,
      outputKey: payload.output_s3_key,
    },
  });

  return { success: true };
}
