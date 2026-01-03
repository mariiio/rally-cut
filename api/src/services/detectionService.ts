import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

import { env } from "../config/env.js";
import { triggerModalDetection } from "../lib/modal.js";
import { prisma } from "../lib/prisma.js";
import {
  ConflictError,
  ForbiddenError,
  LimitExceededError,
  NotFoundError,
  ValidationError,
} from "../middleware/errorHandler.js";
import {
  getUserTier,
  getTierLimits,
  checkAndReserveDetectionQuota,
} from "./tierService.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Check if we should use local detection (development mode without Modal)
function shouldUseLocalDetection(): boolean {
  // Explicitly disabled = use Modal
  if (process.env.USE_LOCAL_DETECTION === "false") {
    return false;
  }
  // Explicitly enabled = use local
  if (process.env.USE_LOCAL_DETECTION === "true") {
    return true;
  }
  // In development, default to local detection
  // (Modal can't callback to localhost without a tunnel)
  return env.NODE_ENV === "development";
}

// Trigger local detection using Python subprocess
async function triggerLocalDetection(params: {
  jobId: string;
  videoS3Key: string;
  callbackUrl: string;
}): Promise<void> {
  // Path to the analysis project (relative to api/)
  const analysisDir = path.resolve(__dirname, "../../../analysis");
  const pythonScript = "rallycut.service.local_runner";

  console.log(`[LOCAL] Starting detection for job ${params.jobId}`);
  console.log(`[LOCAL] Video: ${params.videoS3Key}`);
  console.log(`[LOCAL] Callback: ${params.callbackUrl}`);

  // Spawn the Python process in the background
  const child = spawn(
    "uv",
    [
      "run",
      "python",
      "-m",
      pythonScript,
      "--job-id",
      params.jobId,
      "--video-path",
      params.videoS3Key,
      "--callback-url",
      params.callbackUrl,
      "--webhook-secret",
      env.MODAL_WEBHOOK_SECRET,
      "--s3-bucket",
      env.S3_BUCKET_NAME,
    ],
    {
      cwd: analysisDir,
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        AWS_ACCESS_KEY_ID: env.AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY: env.AWS_SECRET_ACCESS_KEY,
        AWS_REGION: env.AWS_REGION,
      },
    }
  );

  // Log output from the Python process
  child.stdout?.on("data", (data: Buffer) => {
    console.log(`[LOCAL:${params.jobId.slice(0, 8)}] ${data.toString().trim()}`);
  });

  child.stderr?.on("data", (data: Buffer) => {
    console.error(
      `[LOCAL:${params.jobId.slice(0, 8)}] ${data.toString().trim()}`
    );
  });

  child.on("error", async (error) => {
    console.error(`[LOCAL] Failed to start detection: ${error.message}`);
    // Update job status to failed
    try {
      await prisma.$transaction([
        prisma.rallyDetectionJob.update({
          where: { id: params.jobId },
          data: {
            status: "FAILED",
            errorMessage: `Failed to start: ${error.message}`,
            completedAt: new Date(),
          },
        }),
        prisma.video.updateMany({
          where: {
            rallies: { none: {} },
            detectionJobs: { some: { id: params.jobId } },
          },
          data: { status: "ERROR" },
        }),
      ]);
    } catch (e) {
      console.error(`[LOCAL] Failed to update job status: ${e}`);
    }
  });

  child.on("exit", async (code) => {
    console.log(`[LOCAL] Detection process exited with code ${code}`);
    // If process exited with error and job is still RUNNING, mark as failed
    if (code !== 0) {
      try {
        const job = await prisma.rallyDetectionJob.findUnique({
          where: { id: params.jobId },
        });
        if (job && job.status === "RUNNING") {
          await prisma.$transaction([
            prisma.rallyDetectionJob.update({
              where: { id: params.jobId },
              data: {
                status: "FAILED",
                errorMessage: `Process exited with code ${code}`,
                completedAt: new Date(),
              },
            }),
            prisma.video.updateMany({
              where: {
                detectionJobs: { some: { id: params.jobId } },
              },
              data: { status: "ERROR" },
            }),
          ]);
        }
      } catch (e) {
        console.error(`[LOCAL] Failed to update job status: ${e}`);
      }
    }
  });

  // Unref so the parent process can exit independently
  child.unref();

  // Return immediately - the subprocess will call the webhook when done
  return Promise.resolve();
}

export async function triggerRallyDetection(videoId: string, userId: string) {
  // userId is now required - enforced by requireUser middleware

  const tier = await getUserTier(userId);
  const limits = getTierLimits(tier);

  // First, validate the video exists and is in correct state
  // We do this BEFORE reserving quota to avoid wasting a slot on invalid requests
  const where = { id: videoId, userId, deletedAt: null };

  const video = await prisma.video.findFirst({
    where,
  });

  if (video === null) {
    throw new NotFoundError("Video", videoId);
  }

  if (video.status !== "UPLOADED") {
    throw new ConflictError(
      `Video must be in UPLOADED status to trigger detection (current: ${video.status})`
    );
  }

  // Check duration against tier limit
  if (video.durationMs !== null && video.durationMs > limits.maxVideoDurationMs) {
    const maxMinutes = limits.maxVideoDurationMs / 60000;
    const videoMinutes = Math.round(video.durationMs / 60000);
    throw new LimitExceededError(
      `Video duration (${videoMinutes} min) exceeds ${maxMinutes} minute limit for ${tier} tier`,
      {
        field: "durationMs",
        value: video.durationMs,
        limit: limits.maxVideoDurationMs,
      }
    );
  }

  // Atomically check and reserve a detection quota slot
  // This prevents race conditions where concurrent requests bypass quota limits
  const quota = await checkAndReserveDetectionQuota(userId, limits);
  if (!quota.allowed) {
    throw new LimitExceededError(
      `Monthly detection limit reached (${quota.used}/${quota.limit}). Upgrade to Premium for more detections.`,
      {
        field: "detectionsUsed",
        value: quota.used,
        limit: quota.limit,
      }
    );
  }

  // Quota is now reserved - proceed with detection

  // Check for existing job with same contentHash (completed, pending, or running)
  // This prevents duplicate ML processing for the same content
  const existingJob = await prisma.rallyDetectionJob.findFirst({
    where: {
      contentHash: video.contentHash,
      status: { in: ["COMPLETED", "PENDING", "RUNNING"] },
    },
    include: {
      video: {
        include: {
          rallies: true,
        },
      },
    },
    orderBy: { createdAt: "desc" },
  });

  if (existingJob !== null) {
    if (existingJob.status === "COMPLETED") {
      // Reuse cached results from completed job
      await prisma.$transaction(async (tx) => {
        const rallies = existingJob.video.rallies.map((r, index) => ({
          videoId,
          startMs: r.startMs,
          endMs: r.endMs,
          confidence: r.confidence,
          order: index,
        }));

        await tx.rally.createMany({ data: rallies });
        await tx.video.update({
          where: { id: videoId },
          data: { status: "DETECTED" },
        });
      });

      // Quota was already reserved atomically above - no separate increment needed
      return { jobId: existingJob.id, status: "completed", cached: true };
    }

    // Job is PENDING or RUNNING - link this video to the existing job
    // Update video status to DETECTING and return the existing job
    await prisma.video.update({
      where: { id: videoId },
      data: { status: "DETECTING" },
    });

    return {
      jobId: existingJob.id,
      status: existingJob.status.toLowerCase() as "pending" | "running",
      cached: false,
      shared: true, // Indicates this video is sharing detection with another
    };
  }

  const job = await prisma.$transaction(async (tx) => {
    const createdJob = await tx.rallyDetectionJob.create({
      data: {
        videoId,
        contentHash: video.contentHash,
        status: "PENDING",
      },
    });

    await tx.video.update({
      where: { id: videoId },
      data: { status: "DETECTING" },
    });

    return createdJob;
  });

  const callbackUrl = `${env.CORS_ORIGIN.replace("localhost:3000", "localhost:3001")}/v1/webhooks/detection-complete`;

  // Choose between local detection (dev) or Modal (prod)
  const triggerFn = shouldUseLocalDetection()
    ? triggerLocalDetection
    : triggerModalDetection;

  const useLocal = shouldUseLocalDetection();
  console.log(
    `[DETECTION] Using ${useLocal ? "LOCAL" : "MODAL"} detection for job ${job.id}`
  );

  await triggerFn({
    jobId: job.id,
    videoS3Key: video.s3Key,
    callbackUrl,
  }).catch(async (error) => {
    await prisma.$transaction([
      prisma.rallyDetectionJob.update({
        where: { id: job.id },
        data: { status: "FAILED", errorMessage: String(error) },
      }),
      prisma.video.update({
        where: { id: videoId },
        data: { status: "ERROR" },
      }),
    ]);
    throw error;
  });

  await prisma.rallyDetectionJob.update({
    where: { id: job.id },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  // Quota was already reserved atomically above - no separate increment needed
  return { jobId: job.id, status: "pending", cached: false };
}

export async function getDetectionStatus(videoId: string) {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
  });

  if (video === null) {
    throw new NotFoundError("Video", videoId);
  }

  const job = await prisma.rallyDetectionJob.findFirst({
    where: { videoId },
    orderBy: { createdAt: "desc" },
  });

  if (job === null) {
    return {
      videoId,
      status: video.status,
      job: null,
    };
  }

  return {
    videoId,
    status: video.status,
    job: {
      id: job.id,
      status: job.status,
      progress: job.progress ?? 0,
      progressMessage: job.progressMessage,
      startedAt: job.startedAt,
      completedAt: job.completedAt,
      errorMessage: job.errorMessage,
    },
  };
}

export async function updateDetectionProgress(
  jobId: string,
  progress: number,
  message: string
) {
  await prisma.rallyDetectionJob.update({
    where: { id: jobId },
    data: {
      progress: Math.min(100, Math.max(0, progress)),
      progressMessage: message,
    },
  });
}

interface DetectionResult {
  start_ms: number;
  end_ms: number;
  confidence?: number;
}

interface DetectionPayload {
  job_id: string;
  status: "completed" | "failed";
  error_message?: string;
  rallies?: DetectionResult[];
  result_s3_key?: string;
}

// Check if a rally was user-modified (has user-specific data)
function isUserModifiedRally(rally: {
  scoreA: number | null;
  scoreB: number | null;
  servingTeam: string | null;
  notes: string | null;
  confidence: number | null;
}): boolean {
  // Rally is user-modified if it has score, serving team, notes, or no ML confidence
  return (
    rally.scoreA !== null ||
    rally.scoreB !== null ||
    rally.servingTeam !== null ||
    (rally.notes !== null && rally.notes.length > 0) ||
    rally.confidence === null
  );
}

// Check if two time ranges overlap significantly (>50% of the shorter one)
function ralliesOverlap(
  a: { startMs: number; endMs: number },
  b: { startMs: number; endMs: number },
  overlapThreshold = 0.5
): boolean {
  const overlapStart = Math.max(a.startMs, b.startMs);
  const overlapEnd = Math.min(a.endMs, b.endMs);
  const overlapDuration = Math.max(0, overlapEnd - overlapStart);

  const aDuration = a.endMs - a.startMs;
  const bDuration = b.endMs - b.startMs;
  const shorterDuration = Math.min(aDuration, bDuration);

  return shorterDuration > 0 && overlapDuration / shorterDuration >= overlapThreshold;
}

export async function handleDetectionComplete(payload: DetectionPayload) {
  const job = await prisma.rallyDetectionJob.findUnique({
    where: { id: payload.job_id },
    include: { video: true },
  });

  if (job === null) {
    throw new NotFoundError("RallyDetectionJob", payload.job_id);
  }

  if (job.status === "COMPLETED" || job.status === "FAILED") {
    return { ignored: true, reason: "Job already processed" };
  }

  if (payload.status === "failed") {
    await prisma.$transaction([
      prisma.rallyDetectionJob.update({
        where: { id: job.id },
        data: {
          status: "FAILED",
          errorMessage: payload.error_message ?? "Unknown error",
          completedAt: new Date(),
        },
      }),
      prisma.video.update({
        where: { id: job.videoId },
        data: { status: "ERROR" },
      }),
    ]);

    return { success: false, error: payload.error_message };
  }

  const mlRallies = payload.rallies ?? [];

  // Fetch existing rallies for this video
  const existingRallies = await prisma.rally.findMany({
    where: { videoId: job.videoId },
    orderBy: { startMs: "asc" },
  });

  // Separate user-modified rallies from ML-generated ones
  const userRallies = existingRallies.filter(isUserModifiedRally);
  const mlOnlyRallies = existingRallies.filter((r) => !isUserModifiedRally(r));

  // Filter new ML rallies to exclude those that overlap with user rallies
  const newMlRallies = mlRallies.filter((mlRally) => {
    const mlRange = { startMs: mlRally.start_ms, endMs: mlRally.end_ms };
    // Skip if this ML rally overlaps with any user-modified rally
    return !userRallies.some((userRally) => ralliesOverlap(mlRange, userRally));
  });

  await prisma.$transaction(async (tx) => {
    // Delete old ML-only rallies (they'll be replaced with new ML results)
    if (mlOnlyRallies.length > 0) {
      await tx.rally.deleteMany({
        where: {
          id: { in: mlOnlyRallies.map((r) => r.id) },
        },
      });
    }

    // Create new ML rallies (non-overlapping with user rallies)
    if (newMlRallies.length > 0) {
      await tx.rally.createMany({
        data: newMlRallies.map((r, index) => ({
          videoId: job.videoId,
          startMs: r.start_ms,
          endMs: r.end_ms,
          confidence: r.confidence,
          order: userRallies.length + index,
        })),
      });
    }

    // Reorder all rallies by start time
    const allRallies = await tx.rally.findMany({
      where: { videoId: job.videoId },
      orderBy: { startMs: "asc" },
    });

    // Collect rallies that need order updates
    const updates = allRallies
      .map((rally, i) => ({ id: rally.id, order: i }))
      .filter((u, i) => allRallies[i].order !== u.order);

    // Batch update all order changes in a single query
    if (updates.length > 0) {
      const ids = updates.map((u) => u.id);
      const orderCase = updates
        .map((u) => `WHEN '${u.id}' THEN ${u.order}`)
        .join(" ");
      await tx.$executeRawUnsafe(
        `UPDATE "Rally" SET "order" = CASE id ${orderCase} END WHERE id = ANY($1::uuid[])`,
        ids
      );
    }

    await tx.rallyDetectionJob.update({
      where: { id: job.id },
      data: {
        status: "COMPLETED",
        resultS3Key: payload.result_s3_key,
        completedAt: new Date(),
      },
    });

    await tx.video.update({
      where: { id: job.videoId },
      data: { status: "DETECTED" },
    });
  });

  return {
    success: true,
    ralliesCreated: newMlRallies.length,
    userRalliesPreserved: userRallies.length,
    mlRalliesReplaced: mlOnlyRallies.length,
  };
}
