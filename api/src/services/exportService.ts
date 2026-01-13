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
import { getUserTier, getTierLimits, TIER_LIMITS, type UserTier } from "./tierService.js";
import type { TimestampMapping } from "./confirmationService.js";

const lambdaClient = new LambdaClient({
  region: env.AWS_REGION,
});

type ExportQuality = "original" | "720p";

/**
 * Determine export quality for a video based on tier and upload age.
 * FREE users get original quality exports for their grace period (originalQualityDays), then 720p.
 * PRO and ELITE users get original quality based on their tier's originalQualityDays.
 */
function getExportQuality(tier: UserTier, videoCreatedAt: Date): ExportQuality {
  const limits = getTierLimits(tier);

  // If tier has no original quality limit (null), always use original
  if (limits.originalQualityDays === null) return "original";

  const daysSinceUpload = (Date.now() - videoCreatedAt.getTime()) / (24 * 60 * 60 * 1000);
  return daysSinceUpload <= limits.originalQualityDays ? "original" : "720p";
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
  easing: "LINEAR" | "EASE_IN" | "EASE_OUT" | "EASE_IN_OUT";
}

interface CameraEdit {
  enabled: boolean;
  aspectRatio: "ORIGINAL" | "VERTICAL";
  keyframes: CameraKeyframe[];
}

interface CreateExportJobInput {
  sessionId: string;
  // tier is NOT in input - determined by backend from user
  config: {
    format: "mp4" | "webm";
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
 * @returns FFmpeg -vf filter string
 */
function generateCameraFilter(
  camera: CameraEdit,
  durationSec: number,
  inputWidth = 1920,
  inputHeight = 1080
): string {
  const fps = 30;
  const totalFrames = Math.ceil(durationSec * fps);

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
    return `crop=${outputWidth}:${outputHeight}:${x}:${y}`;
  }

  // Sort keyframes by time
  const sortedKeyframes = [...keyframes].sort((a, b) => a.timeOffset - b.timeOffset);

  // For single keyframe, use static position with zoom
  if (sortedKeyframes.length === 1) {
    const kf = sortedKeyframes[0];
    const zoom = Math.max(1, Math.min(3, kf.zoom));
    const croppedW = Math.round(outputWidth / zoom);
    const croppedH = Math.round(outputHeight / zoom);
    const maxX = inputWidth - croppedW;
    const maxY = inputHeight - croppedH;
    const x = Math.round(kf.positionX * maxX);
    const y = Math.round(kf.positionY * maxY);
    return `crop=${croppedW}:${croppedH}:${x}:${y},scale=${outputWidth}:${outputHeight}`;
  }

  // Build FFmpeg expression for animated crop
  // We'll generate a piecewise expression using if() functions

  // Helper to generate easing expression
  const easingExpr = (t: string, easing: string): string => {
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

  // Generate expressions for x, y positions and zoom
  // FFmpeg uses 'n' for frame number in expressions
  const segments: string[] = [];

  for (let i = 0; i < sortedKeyframes.length - 1; i++) {
    const kf1 = sortedKeyframes[i];
    const kf2 = sortedKeyframes[i + 1];

    const frame1 = Math.round(kf1.timeOffset * totalFrames);
    const frame2 = Math.round(kf2.timeOffset * totalFrames);

    // Linear interpolation with easing
    const tExpr = `((n-${frame1})/${Math.max(1, frame2 - frame1)})`;
    const easedT = easingExpr(tExpr, kf2.easing);

    // Interpolate zoom
    const z1 = Math.max(1, Math.min(3, kf1.zoom));
    const z2 = Math.max(1, Math.min(3, kf2.zoom));
    const zoomExpr = `(${z1}+${easedT}*(${z2}-${z1}))`;

    // Interpolate position (0-1 normalized)
    const px1 = kf1.positionX;
    const px2 = kf2.positionX;
    const py1 = kf1.positionY;
    const py2 = kf2.positionY;
    const pxExpr = `(${px1}+${easedT}*(${px2}-${px1}))`;
    const pyExpr = `(${py1}+${easedT}*(${py2}-${py1}))`;

    // Width/height expressions accounting for zoom
    const wExpr = `floor(${outputWidth}/${zoomExpr})`;
    const hExpr = `floor(${outputHeight}/${zoomExpr})`;

    // X/Y position expressions (position is relative to valid crop area)
    const xExpr = `floor(${pxExpr}*(${inputWidth}-${wExpr}))`;
    const yExpr = `floor(${pyExpr}*(${inputHeight}-${hExpr}))`;

    segments.push({
      frame1,
      frame2,
      wExpr,
      hExpr,
      xExpr,
      yExpr,
    } as unknown as string); // We'll process this differently
  }

  // For simplicity with complex expressions, we'll use a simpler approach:
  // Generate a static filter based on first keyframe if expressions get too complex
  // FFmpeg expressions have limits on complexity

  // Simplified approach: Use zoompan filter which handles this better
  // Format: zoompan=z='zoom_expr':x='x_expr':y='y_expr':d=frames:s=WxH:fps=fps

  // Build zoom, x, y expressions as piecewise
  const buildPiecewiseExpr = (
    keyframes: CameraKeyframe[],
    getValue: (kf: CameraKeyframe) => number,
    totalFrames: number
  ): string => {
    if (keyframes.length === 1) {
      return String(getValue(keyframes[0]));
    }

    let expr = "";
    for (let i = keyframes.length - 2; i >= 0; i--) {
      const kf1 = keyframes[i];
      const kf2 = keyframes[i + 1];
      const frame1 = Math.round(kf1.timeOffset * totalFrames);
      const frame2 = Math.round(kf2.timeOffset * totalFrames);
      const v1 = getValue(kf1);
      const v2 = getValue(kf2);

      const tExpr = `(on-${frame1})/${Math.max(1, frame2 - frame1)}`;
      const easedT = easingExpr(tExpr, kf2.easing);
      const interpExpr = `${v1}+${easedT}*(${v2}-${v1})`;

      if (i === keyframes.length - 2) {
        // Last segment (innermost)
        expr = `if(lt(on,${frame1}),${v1},${interpExpr})`;
      } else {
        // Wrap with condition for this segment
        expr = `if(lt(on,${frame1}),${v1},if(lt(on,${frame2}),${interpExpr},${expr}))`;
      }
    }
    return expr || String(getValue(keyframes[0]));
  };

  const zoomExpr = buildPiecewiseExpr(sortedKeyframes, kf => Math.max(1, Math.min(3, kf.zoom)), totalFrames);
  const pxExpr = buildPiecewiseExpr(sortedKeyframes, kf => kf.positionX, totalFrames);
  const pyExpr = buildPiecewiseExpr(sortedKeyframes, kf => kf.positionY, totalFrames);

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
  return `crop=w='${wExprFinal}':h='${hExprFinal}':x='${xExprFinal}':y='${yExprFinal}',scale=${outputWidth}:${outputHeight}`;
}

export async function createExportJob(
  userId: string,
  input: CreateExportJobInput
) {
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
  const ralliesWithQuality = input.rallies.map((rally) => {
    const videoCreatedAt = videoCreatedAtMap.get(rally.videoId);
    const exportQuality = videoCreatedAt
      ? getExportQuality(effectiveTier, videoCreatedAt)
      : "720p"; // Default to 720p if video not found

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
      tier: effectiveTier as UserTier,
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
      const hasCamera = rally.camera?.enabled && rally.camera?.keyframes?.length > 0;

      console.log(`[LOCAL EXPORT] Extracting clip ${i + 1}/${input.rallies.length}${hasCamera ? ' with camera effects' : ''}...`);

      // Use per-rally exportQuality (original or 720p based on video age)
      const quality = rally.exportQuality;

      if (hasCamera) {
        // Camera edits require re-encoding
        const cameraFilter = generateCameraFilter(rally.camera!, durationSec);
        // If 720p quality, add scale filter after camera filter
        const vf = quality === "720p"
          ? `${cameraFilter},scale=-2:720`
          : cameraFilter;
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
      // Include camera edit if present and enabled
      ...(r.camera?.enabled && {
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
