import { ExportStatus, ExportTier } from "@prisma/client";
import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { spawn } from "child_process";
import fs from "fs/promises";
import os from "os";
import path from "path";

import { env } from "../config/env.js";
import { generateDownloadUrl, generateUploadUrl } from "../lib/s3.js";
import { prisma } from "../lib/prisma.js";
import { NotFoundError } from "../middleware/errorHandler.js";

const lambdaClient = new LambdaClient({
  region: env.AWS_REGION,
});

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

interface CreateExportJobInput {
  sessionId: string;
  tier: "FREE" | "PREMIUM";
  config: {
    format: "mp4" | "webm";
  };
  rallies: Array<{
    videoId: string;
    videoS3Key: string;
    startMs: number;
    endMs: number;
  }>;
}

export async function createExportJob(
  userId: string,
  input: CreateExportJobInput
) {
  // Verify session exists and user has access
  const session = await prisma.session.findFirst({
    where: {
      id: input.sessionId,
      OR: [
        { userId },
        { share: { members: { some: { userId } } } },
      ],
    },
  });

  if (!session) {
    throw new NotFoundError("Session", input.sessionId);
  }

  // Create the export job
  const job = await prisma.exportJob.create({
    data: {
      sessionId: input.sessionId,
      userId,
      tier: input.tier as ExportTier,
      status: ExportStatus.PENDING,
      config: input.config,
      rallies: input.rallies,
    },
  });

  // Choose between local (dev) or Lambda (prod)
  const useLocal = shouldUseLocalExport();
  console.log(`[EXPORT] Using ${useLocal ? "LOCAL" : "LAMBDA"} export for job ${job.id}`);

  const triggerFn = useLocal ? triggerLocalExport : triggerExportLambda;

  // Trigger export (async, don't wait)
  triggerFn(job.id, input).catch((error) => {
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
  input: CreateExportJobInput
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

    // Process each rally
    for (let i = 0; i < input.rallies.length; i++) {
      const rally = input.rallies[i];
      const progress = Math.round((i / input.rallies.length) * 80);
      await updateExportProgress(jobId, progress);

      // Download video from S3 using presigned URL
      const downloadUrl = await generateDownloadUrl(rally.videoS3Key);
      const videoPath = path.join(tmpDir, `input_${i}.mp4`);
      const clipPath = path.join(tmpDir, `clip_${i}.mp4`);

      console.log(`[LOCAL EXPORT] Downloading ${rally.videoS3Key}...`);
      const response = await fetch(downloadUrl);
      if (!response.ok) {
        throw new Error(`Failed to download video: ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      await fs.writeFile(videoPath, Buffer.from(buffer));

      // Extract clip with FFmpeg
      const startSec = rally.startMs / 1000;
      const durationSec = (rally.endMs - rally.startMs) / 1000;

      console.log(`[LOCAL EXPORT] Extracting clip ${i + 1}/${input.rallies.length}...`);

      if (input.tier === "PREMIUM") {
        // Fast copy for premium
        await runFFmpeg([
          "-ss", String(startSec),
          "-i", videoPath,
          "-t", String(durationSec),
          "-c", "copy",
          "-avoid_negative_ts", "make_zero",
          "-y", clipPath,
        ]);
      } else {
        // Re-encode to 720p for free tier (no watermark in local dev)
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

      // Clean up input to save space
      await fs.unlink(videoPath);
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

async function triggerExportLambda(jobId: string, input: CreateExportJobInput) {
  const functionName = env.EXPORT_LAMBDA_FUNCTION_NAME!;

  const payload = {
    jobId,
    tier: input.tier,
    format: input.config.format,
    rallies: input.rallies.map((r) => ({
      videoS3Key: r.videoS3Key,
      startMs: r.startMs,
      endMs: r.endMs,
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
