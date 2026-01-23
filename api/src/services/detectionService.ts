import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

import { ProcessingStatus, RejectionReason, Video } from "@prisma/client";

import { env } from "../config/env.js";
import { triggerModalDetection } from "../lib/modal.js";
import { prisma } from "../lib/prisma.js";
import {
  ConflictError,
  ForbiddenError,
  LimitExceededError,
  NotFoundError,
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
  modelVariant?: "indoor" | "beach";
}): Promise<void> {
  // Path to the analysis project (relative to api/)
  const analysisDir = path.resolve(__dirname, "../../../analysis");
  const pythonScript = "rallycut.service.local_runner";

  console.log(`[LOCAL] Starting detection for job ${params.jobId}`);
  console.log(`[LOCAL] Video: ${params.videoS3Key}`);
  console.log(`[LOCAL] Model: ${params.modelVariant || "indoor"}`);
  console.log(`[LOCAL] Callback: ${params.callbackUrl}`);

  // Build arguments
  const args = [
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
  ];

  // Add model variant if specified
  if (params.modelVariant) {
    args.push("--model", params.modelVariant);
  }

  // Spawn the Python process in the background
  const child = spawn("uv", args, {
      cwd: analysisDir,
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        AWS_ACCESS_KEY_ID: env.AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY: env.AWS_SECRET_ACCESS_KEY,
        AWS_REGION: env.AWS_REGION,
        // Pass S3_ENDPOINT for MinIO/local development
        ...(env.S3_ENDPOINT && { S3_ENDPOINT: env.S3_ENDPOINT }),
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

// Check if video proxy is ready for detection
// Returns the video - uses proxy if available, otherwise falls back to original
function getVideoForDetection(video: Video): Video {
  // Proxy is ready - use it for faster detection
  if (video.proxyS3Key) {
    console.log(`[DETECTION] Using proxy for video ${video.id}`);
    return video;
  }

  // Processing failed - warn but allow detection with original
  if (video.processingStatus === ProcessingStatus.FAILED) {
    console.log(
      `[DETECTION] Processing failed for ${video.id}, using original video`
    );
    return video;
  }

  // Proxy not ready yet - fall back to original video
  // This allows detection to start immediately without waiting for proxy
  console.log(
    `[DETECTION] Proxy not ready (status: ${video.processingStatus}), using original for ${video.id}`
  );
  return video;
}

export async function triggerRallyDetection(
  videoId: string,
  userId: string,
  modelVariant?: "indoor" | "beach"
) {
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

  // Check if video has confirmed rallies - detection is not allowed on confirmed videos
  const confirmation = await prisma.rallyConfirmation.findFirst({
    where: {
      videoId,
      status: "CONFIRMED",
    },
  });

  if (confirmation !== null) {
    throw new ForbiddenError(
      "Cannot run detection on a video with confirmed rallies. Restore the original video first."
    );
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

  // Get the video source for detection (proxy if available, otherwise original)
  const detectionVideo = getVideoForDetection(video);

  // Atomically check and reserve a detection quota slot
  // This prevents race conditions where concurrent requests bypass quota limits
  const quota = await checkAndReserveDetectionQuota(userId, limits);
  if (!quota.allowed) {
    const upgradeHint =
      tier === "FREE" ? " Upgrade to Pro for more detections." :
      tier === "PRO" ? " Upgrade to Elite for more detections." : "";
    throw new LimitExceededError(
      `Monthly detection limit reached (${quota.used}/${quota.limit}).${upgradeHint}`,
      {
        field: "detectionsUsed",
        value: quota.used,
        limit: quota.limit,
      }
    );
  }

  // Quota is now reserved - proceed with detection

  // Atomically check for existing job and create new one if needed
  // Uses serializable isolation to prevent race conditions where concurrent
  // requests both pass the "no existing job" check and create duplicates
  const { job, existingJob, cached } = await prisma.$transaction(
    async (tx) => {
      // Check for existing job with same contentHash (completed, pending, or running)
      // This prevents duplicate ML processing for the same content
      const existing = await tx.rallyDetectionJob.findFirst({
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

      if (existing !== null) {
        if (existing.status === "COMPLETED") {
          // Reuse cached results from completed job
          const rallies = existing.video.rallies.map((r, index) => ({
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

          return { job: null, existingJob: existing, cached: true };
        }

        // Job is PENDING or RUNNING - link this video to the existing job
        await tx.video.update({
          where: { id: videoId },
          data: { status: "DETECTING" },
        });

        return { job: null, existingJob: existing, cached: false };
      }

      // No existing job - create new one atomically within same transaction
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

      return { job: createdJob, existingJob: null, cached: false };
    },
    {
      isolationLevel: "Serializable",
    }
  );

  // Handle existing job cases (returned from transaction)
  if (existingJob !== null) {
    if (cached) {
      return { jobId: existingJob.id, status: "completed", cached: true };
    }
    return {
      jobId: existingJob.id,
      status: existingJob.status.toLowerCase() as "pending" | "running",
      cached: false,
      shared: true,
    };
  }

  // New job was created - job is guaranteed non-null here (early return handled existingJob case)
  const createdJob = job;

  const callbackUrl = `${env.CORS_ORIGIN.replace("localhost:3000", "localhost:3001")}/v1/webhooks/detection-complete`;

  // Choose between local detection (dev) or Modal (prod)
  const triggerFn = shouldUseLocalDetection()
    ? triggerLocalDetection
    : triggerModalDetection;

  const useLocal = shouldUseLocalDetection();
  const videoKey = detectionVideo.proxyS3Key ?? detectionVideo.s3Key;
  console.log(
    `[DETECTION] Using ${useLocal ? "LOCAL" : "MODAL"} detection for job ${createdJob.id}`
  );
  console.log(`[DETECTION] Video key: ${videoKey}`);
  console.log(`[DETECTION] Model variant: ${modelVariant || "indoor"}`);

  await triggerFn({
    jobId: createdJob.id,
    videoS3Key: videoKey,
    callbackUrl,
    modelVariant,
  }).catch(async (error) => {
    await prisma.$transaction([
      prisma.rallyDetectionJob.update({
        where: { id: createdJob.id },
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
    where: { id: createdJob.id },
    data: { status: "RUNNING", startedAt: new Date() },
  });

  // Quota was already reserved atomically above - no separate increment needed
  return { jobId: createdJob.id, status: "pending", cached: false };
}

export async function getDetectionStatus(videoId: string, userId: string) {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
  });

  if (video === null) {
    throw new NotFoundError("Video", videoId);
  }

  if (video.userId !== userId) {
    throw new ForbiddenError("You do not have permission to access this video's detection status");
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

interface SuggestedResult {
  start_ms: number;
  end_ms: number;
  confidence: number;
  rejection_reason: "insufficient_windows" | "too_short" | "sparse_density";
}

interface DetectionPayload {
  job_id: string;
  status: "completed" | "failed";
  error_message?: string;
  rallies?: DetectionResult[];
  suggested_rallies?: SuggestedResult[];
  result_s3_key?: string;
}

// Map rejection_reason string to Prisma enum
function mapRejectionReason(reason: SuggestedResult["rejection_reason"]): RejectionReason {
  const map: Record<SuggestedResult["rejection_reason"], RejectionReason> = {
    insufficient_windows: "INSUFFICIENT_WINDOWS",
    too_short: "TOO_SHORT",
    sparse_density: "SPARSE_DENSITY",
  };
  return map[reason];
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

  // Clamp rally timestamps to video duration (safety net for ML results)
  const durationMs = job.video.durationMs ?? Infinity;
  const clampedMlRallies = newMlRallies.map((r) => ({
    ...r,
    start_ms: Math.max(0, Math.min(r.start_ms, durationMs)),
    end_ms: Math.max(0, Math.min(r.end_ms, durationMs)),
  }));

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
    if (clampedMlRallies.length > 0) {
      await tx.rally.createMany({
        data: clampedMlRallies.map((r, index) => ({
          videoId: job.videoId,
          startMs: r.start_ms,
          endMs: r.end_ms,
          confidence: r.confidence,
          status: "CONFIRMED",
          order: userRallies.length + index,
        })),
      });
    }

    // Process suggested rallies (segments that almost passed detection)
    const suggestedRallies = payload.suggested_rallies ?? [];

    // Filter suggestions that overlap with existing/new confirmed rallies
    const confirmedRanges = [
      ...userRallies.map((r) => ({ startMs: r.startMs, endMs: r.endMs })),
      ...clampedMlRallies.map((r) => ({ startMs: r.start_ms, endMs: r.end_ms })),
    ];

    const newSuggestions = suggestedRallies.filter((s) => {
      const range = { startMs: s.start_ms, endMs: s.end_ms };
      return !confirmedRanges.some((existing) => ralliesOverlap(range, existing, 0.3));
    });

    // Clamp suggested rally timestamps to video duration
    const clampedSuggestions = newSuggestions.map((r) => ({
      ...r,
      start_ms: Math.max(0, Math.min(r.start_ms, durationMs)),
      end_ms: Math.max(0, Math.min(r.end_ms, durationMs)),
    }));

    // Create suggested rallies
    if (clampedSuggestions.length > 0) {
      await tx.rally.createMany({
        data: clampedSuggestions.map((r, index) => ({
          videoId: job.videoId,
          startMs: r.start_ms,
          endMs: r.end_ms,
          confidence: r.confidence,
          status: "SUGGESTED",
          rejectionReason: mapRejectionReason(r.rejection_reason),
          order: userRallies.length + clampedMlRallies.length + index,
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

    // Batch update all order changes
    for (const u of updates) {
      await tx.rally.update({
        where: { id: u.id },
        data: { order: u.order },
      });
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

  const suggestedRallies = payload.suggested_rallies ?? [];
  const confirmedRanges = [
    ...userRallies.map((r) => ({ startMs: r.startMs, endMs: r.endMs })),
    ...newMlRallies.map((r) => ({ startMs: r.start_ms, endMs: r.end_ms })),
  ];
  const createdSuggestions = suggestedRallies.filter((s) => {
    const range = { startMs: s.start_ms, endMs: s.end_ms };
    return !confirmedRanges.some((existing) => ralliesOverlap(range, existing, 0.3));
  });

  return {
    success: true,
    ralliesCreated: newMlRallies.length,
    suggestedRalliesCreated: createdSuggestions.length,
    userRalliesPreserved: userRallies.length,
    mlRalliesReplaced: mlOnlyRallies.length,
  };
}
