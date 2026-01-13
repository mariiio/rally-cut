import { prisma } from "../lib/prisma.js";
import {
  generateUploadUrl,
  getVideoS3Key,
  initiateMultipartUpload as s3InitiateMultipart,
  generatePartUploadUrl,
  completeMultipartUpload as s3CompleteMultipart,
  abortMultipartUpload as s3AbortMultipart,
  deleteObject,
} from "../lib/s3.js";
import {
  ConflictError,
  LimitExceededError,
  NotFoundError,
  ValidationError,
} from "../middleware/errorHandler.js";
import type {
  ConfirmUploadInput,
  RequestUploadUrlInput,
  UpdateVideoInput,
} from "../schemas/video.js";
import { getOrCreateAllVideosSession } from "./sessionService.js";
import {
  getUserTier,
  getTierLimits,
  checkUploadQuota,
  checkAndReserveUploadQuota,
  TIER_LIMITS,
} from "./tierService.js";
import { queueVideoProcessing, generatePosterImmediate } from "./processingService.js";

const MAX_VIDEOS_PER_SESSION = 5;

// Type for Prisma transaction client
type PrismaTransaction = Parameters<Parameters<typeof prisma.$transaction>[0]>[0];

// Helper to convert BigInt fields to strings for JSON serialization
function serializeBigInts<T>(obj: T): T {
  return JSON.parse(
    JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value
    )
  );
}

// Shared validation for upload requests
interface UploadValidationParams {
  userId: string;
  fileSize: number;
  durationMs?: number;
  contentHash: string;
}

interface UploadValidationResult {
  tier: Awaited<ReturnType<typeof getUserTier>>;
  limits: ReturnType<typeof getTierLimits>;
  expiresAt: Date | null;
}

async function validateUploadRequest(
  params: UploadValidationParams
): Promise<UploadValidationResult> {
  const tier = await getUserTier(params.userId);
  const limits = getTierLimits(tier);

  // Log tier info for debugging upload issues
  console.log(`[UPLOAD] User ${params.userId} has tier ${tier}, file size: ${Math.round(params.fileSize / (1024 * 1024))} MB, limit: ${Math.round(limits.maxFileSizeBytes / (1024 * 1024))} MB`);

  // Validate file size against tier limit
  if (params.fileSize > limits.maxFileSizeBytes) {
    const fileSizeMB = Math.round(params.fileSize / (1024 * 1024));
    const limitMB = Math.round(limits.maxFileSizeBytes / (1024 * 1024));
    const limitGB = limitMB >= 1024 ? `${limitMB / 1024} GB` : `${limitMB} MB`;
    const eliteLimitGB = TIER_LIMITS.ELITE.maxFileSizeBytes / (1024 * 1024 * 1024);
    const upgradeHint = tier === "FREE"
      ? ` Upgrade to Pro or Elite for larger uploads (up to ${eliteLimitGB} GB).`
      : tier === "PRO"
        ? ` Upgrade to Elite for ${eliteLimitGB} GB uploads.`
        : " Consider compressing the video or splitting it into parts.";
    throw new LimitExceededError(
      `File size (${fileSizeMB} MB) exceeds ${limitGB} limit for ${tier} tier.${upgradeHint}`,
      { fileSize: params.fileSize, limit: limits.maxFileSizeBytes, tier }
    );
  }

  // Validate duration against tier limit (if provided)
  if (params.durationMs && params.durationMs > limits.maxVideoDurationMs) {
    const durationMin = Math.round(params.durationMs / 60000);
    const limitMin = Math.round(limits.maxVideoDurationMs / 60000);
    const eliteLimitMin = Math.round(TIER_LIMITS.ELITE.maxVideoDurationMs / 60000);
    const upgradeHint = tier === "FREE"
      ? ` Upgrade to Pro or Elite for longer videos (up to ${eliteLimitMin} min).`
      : tier === "PRO"
        ? ` Upgrade to Elite for ${eliteLimitMin} minute videos.`
        : "";
    throw new LimitExceededError(
      `Video duration (${durationMin} min) exceeds ${limitMin} minute limit for ${tier} tier.${upgradeHint}`,
      { duration: params.durationMs, limit: limits.maxVideoDurationMs }
    );
  }

  // Check upload quota (quick fail for UX - actual atomic reservation happens at confirm)
  const uploadQuota = await checkUploadQuota(params.userId);
  if (!uploadQuota.allowed) {
    throw new LimitExceededError(
      `Monthly upload limit reached (${uploadQuota.used}/${uploadQuota.limit}). Upgrade to a higher tier for more uploads, or wait until next month.`,
      { used: uploadQuota.used, limit: uploadQuota.limit }
    );
  }

  // Note: Duplicate content hash check is done atomically in createVideoUploadUrl/requestUploadUrl
  // to prevent TOCTOU race conditions

  // No time-based expiration - videos kept until 2 months of user inactivity
  return { tier, limits, expiresAt: null };
}

// Helper to find or create a pending video for upload
// Reuses PENDING videos to allow retries after failed uploads
interface FindOrCreatePendingVideoParams {
  userId: string;
  contentHash: string;
  filename: string;
  fileSize: number;
  durationMs?: number;
  expiresAt: Date | null;
}

interface FindOrCreatePendingVideoResult {
  video: {
    id: string;
    s3Key: string;
  };
  reused: boolean;
  alreadyExists: boolean; // True if video was already uploaded (not pending)
}

async function findOrCreatePendingVideo(
  tx: PrismaTransaction,
  params: FindOrCreatePendingVideoParams
): Promise<FindOrCreatePendingVideoResult> {
  // Check for existing video with same content hash
  const existingVideo = await tx.video.findFirst({
    where: {
      userId: params.userId,
      contentHash: params.contentHash,
      deletedAt: null,
    },
  });

  if (existingVideo) {
    // Reuse PENDING videos (allows retrying failed uploads)
    if (existingVideo.status === "PENDING") {
      // Update metadata in case file size or duration changed
      // Keep original s3Key and filename to avoid orphaned S3 files
      await tx.video.update({
        where: { id: existingVideo.id },
        data: {
          fileSizeBytes: params.fileSize,
          durationMs: params.durationMs,
          expiresAt: params.expiresAt,
        },
      });
      return {
        video: { id: existingVideo.id, s3Key: existingVideo.s3Key },
        reused: true,
        alreadyExists: false,
      };
    }

    // Completed videos - return existing video so frontend can skip upload
    return {
      video: { id: existingVideo.id, s3Key: existingVideo.s3Key },
      reused: true,
      alreadyExists: true,
    };
  }

  // No existing video - create new one
  const videoId = crypto.randomUUID();
  const s3Key = getVideoS3Key(params.userId, videoId, params.filename);

  const video = await tx.video.create({
    data: {
      id: videoId,
      userId: params.userId,
      name: params.filename.replace(/\.[^.]+$/, ""),
      filename: params.filename,
      s3Key,
      contentHash: params.contentHash,
      fileSizeBytes: params.fileSize,
      durationMs: params.durationMs,
      expiresAt: params.expiresAt,
    },
  });

  return { video: { id: video.id, s3Key: video.s3Key }, reused: false, alreadyExists: false };
}

// ============================================================================
// Video Library (User-scoped)
// ============================================================================

export interface VideoListParams {
  search?: string;
  page?: number;
  limit?: number;
}

export async function listUserVideos(userId: string, params: VideoListParams = {}) {
  const page = params.page ?? 1;
  const limit = params.limit ?? 50;
  const skip = (page - 1) * limit;

  const where = {
    userId,
    deletedAt: null,
    ...(params.search && {
      name: { contains: params.search, mode: "insensitive" as const },
    }),
  };

  const [videos, total] = await Promise.all([
    prisma.video.findMany({
      where,
      orderBy: { createdAt: "desc" },
      skip,
      take: limit,
      include: {
        // Only fetch session id, name, type - no need for separate _count
        sessionVideos: {
          select: {
            session: {
              select: { id: true, name: true, type: true },
            },
          },
        },
      },
    }),
    prisma.video.count({ where }),
  ]);

  return {
    data: videos.map((v) => ({
      ...serializeBigInts(v),
      sessionCount: v.sessionVideos.length, // Use array length instead of extra _count query
      sessions: v.sessionVideos.map((sv) => ({
        id: sv.session.id,
        name: sv.session.name,
        type: sv.session.type,
      })),
    })),
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit),
    },
  };
}

// ============================================================================
// Video Upload (User-scoped, optionally linked to session)
// ============================================================================

export interface CreateVideoUploadParams {
  userId: string;
  filename: string;
  contentType: string;
  contentHash: string;
  fileSize: number;
  durationMs?: number;
}

export async function createVideoUploadUrl(params: CreateVideoUploadParams) {
  // Validate upload request (size, duration, quota - NOT duplicates, that's atomic below)
  const { expiresAt } = await validateUploadRequest({
    userId: params.userId,
    fileSize: params.fileSize,
    durationMs: params.durationMs,
    contentHash: params.contentHash,
  });

  // Use transaction to atomically find/create video
  // Reuses PENDING videos to allow retries after failed uploads
  const { video, alreadyExists } = await prisma.$transaction(async (tx) => {
    return findOrCreatePendingVideo(tx, {
      userId: params.userId,
      contentHash: params.contentHash,
      filename: params.filename,
      fileSize: params.fileSize,
      durationMs: params.durationMs,
      expiresAt,
    });
  });

  // If video already exists, skip upload URL generation
  if (alreadyExists) {
    return {
      uploadUrl: null,
      videoId: video.id,
      s3Key: video.s3Key,
      alreadyExists: true,
    };
  }

  const uploadUrl = await generateUploadUrl({
    key: video.s3Key,
    contentType: params.contentType,
    contentLength: params.fileSize,
  });

  return {
    uploadUrl,
    videoId: video.id,
    s3Key: video.s3Key,
    alreadyExists: false,
  };
}

// ============================================================================
// Multipart Upload (for large files, 100MB+)
// ============================================================================

// Calculate optimal part size (5MB min, scale up for large files)
function calculatePartSize(fileSize: number): number {
  const MIN_PART_SIZE = 5 * 1024 * 1024; // 5MB minimum (S3 requirement)
  const MAX_PARTS = 10000; // S3 max parts limit
  const TARGET_PARTS = 100; // Aim for ~100 parts for balance of parallelism vs overhead

  let partSize = Math.ceil(fileSize / TARGET_PARTS);
  partSize = Math.max(partSize, MIN_PART_SIZE);

  // Ensure we don't exceed max parts
  if (Math.ceil(fileSize / partSize) > MAX_PARTS) {
    partSize = Math.ceil(fileSize / MAX_PARTS);
  }

  return partSize;
}

export interface InitiateMultipartParams {
  userId: string;
  filename: string;
  contentType: string;
  contentHash: string;
  fileSize: number;
  durationMs?: number;
}

export async function initiateMultipartUpload(params: InitiateMultipartParams) {
  // Validate upload request (size, duration, quota)
  const { expiresAt } = await validateUploadRequest({
    userId: params.userId,
    fileSize: params.fileSize,
    durationMs: params.durationMs,
    contentHash: params.contentHash,
  });

  // Find or create PENDING video (reuse logic from single upload)
  const { video, alreadyExists } = await prisma.$transaction(async (tx) => {
    return findOrCreatePendingVideo(tx, {
      userId: params.userId,
      contentHash: params.contentHash,
      filename: params.filename,
      fileSize: params.fileSize,
      durationMs: params.durationMs,
      expiresAt,
    });
  });

  // If video already exists, skip multipart upload entirely
  if (alreadyExists) {
    return {
      videoId: video.id,
      s3Key: video.s3Key,
      uploadId: null,
      partSize: 0,
      partUrls: [],
      alreadyExists: true,
    };
  }

  // Initiate S3 multipart upload
  const uploadId = await s3InitiateMultipart(video.s3Key, params.contentType);

  // Calculate parts and generate presigned URLs
  const partSize = calculatePartSize(params.fileSize);
  const partCount = Math.ceil(params.fileSize / partSize);

  const partUrls = await Promise.all(
    Array.from({ length: partCount }, (_, i) =>
      generatePartUploadUrl(video.s3Key, uploadId, i + 1)
    )
  );

  return {
    videoId: video.id,
    s3Key: video.s3Key,
    uploadId,
    partSize,
    partUrls,
    alreadyExists: false,
  };
}

export async function completeMultipartUpload(
  videoId: string,
  userId: string,
  uploadId: string,
  parts: { partNumber: number; etag: string }[]
) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, status: "PENDING" },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Complete the multipart upload in S3
  await s3CompleteMultipart(
    video.s3Key,
    uploadId,
    parts.map((p) => ({ PartNumber: p.partNumber, ETag: p.etag }))
  );
}

export async function abortMultipartUpload(
  videoId: string,
  userId: string,
  uploadId: string
) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Abort the multipart upload and cleanup parts in S3
  await s3AbortMultipart(video.s3Key, uploadId);
}

export async function confirmVideoUpload(
  videoId: string,
  userId: string,
  data: { durationMs?: number; width?: number; height?: number }
) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Atomically check and reserve upload quota (prevents race conditions)
  const tier = await getUserTier(userId);
  const limits = getTierLimits(tier);
  const quotaResult = await checkAndReserveUploadQuota(userId, limits);

  if (!quotaResult.allowed) {
    throw new LimitExceededError(
      `Monthly upload limit reached (${quotaResult.used}/${quotaResult.limit}). Upgrade for more uploads, or wait until next month.`,
      { used: quotaResult.used, limit: quotaResult.limit }
    );
  }

  const updated = await prisma.video.update({
    where: { id: videoId },
    data: {
      status: "UPLOADED",
      durationMs: data.durationMs,
      width: data.width,
      height: data.height,
    },
  });

  // Auto-add to "All Videos" session
  const allVideosSession = await getOrCreateAllVideosSession(userId);

  // Check if not already in the session
  const existingLink = await prisma.sessionVideo.findUnique({
    where: {
      sessionId_videoId: { sessionId: allVideosSession.id, videoId },
    },
  });

  if (!existingLink) {
    // Use MAX(order) + 1 in a transaction to prevent order collisions
    await prisma.$transaction(async (tx) => {
      const maxOrder = await tx.sessionVideo.aggregate({
        where: { sessionId: allVideosSession.id },
        _max: { order: true },
      });
      await tx.sessionVideo.create({
        data: {
          sessionId: allVideosSession.id,
          videoId,
          order: (maxOrder._max.order ?? -1) + 1,
        },
      });
    });
  }

  // Generate poster immediately for fast UX (async, ~2-3 seconds)
  generatePosterImmediate(videoId, updated.s3Key).catch((error) => {
    console.error(`[UPLOAD] Failed to generate poster for ${videoId}:`, error);
    // Non-fatal: processing pipeline will regenerate if missing
  });

  // Queue video for optimization processing (async, non-blocking)
  // Video is immediately playable; processing happens in background
  queueVideoProcessing(videoId, userId).catch((error) => {
    console.error(`[UPLOAD] Failed to queue processing for video ${videoId}:`, error);
  });

  return serializeBigInts(updated);
}

// ============================================================================
// Video Update & Delete
// ============================================================================

export async function updateVideo(
  videoId: string,
  userId: string,
  data: UpdateVideoInput
) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  const updated = await prisma.video.update({
    where: { id: videoId },
    data,
  });

  return serializeBigInts(updated);
}

export async function softDeleteVideo(videoId: string, userId: string) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  await prisma.video.update({
    where: { id: videoId },
    data: { deletedAt: new Date() },
  });

  return { success: true };
}

/**
 * Permanently delete a video and all associated data.
 * - Deletes database record (cascades: rallies, camera edits, keyframes, highlight rallies, session videos)
 * - Deletes S3 files (original, optimized, poster, proxy, trimmed)
 */
export async function hardDeleteVideo(videoId: string, userId: string) {
  // Fetch video with confirmation (for trimmedS3Key)
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
    include: {
      confirmation: { select: { trimmedS3Key: true } },
    },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Collect S3 keys to delete
  const s3KeysToDelete = [
    video.s3Key,
    video.originalS3Key,
    video.posterS3Key,
    video.proxyS3Key,
    video.processedS3Key,
    video.confirmation?.trimmedS3Key,
  ].filter((key): key is string => !!key);

  // Delete from database (cascades handle related records)
  await prisma.video.delete({
    where: { id: videoId },
  });

  // Delete S3 files (non-blocking, log failures)
  for (const key of s3KeysToDelete) {
    deleteObject(key).catch((error) => {
      console.error(
        JSON.stringify({
          event: "S3_DELETE_FAILED",
          type: "video_hard_delete",
          videoId,
          s3Key: key,
          error: error instanceof Error ? error.message : String(error),
        })
      );
    });
  }

  return { success: true };
}

export async function restoreVideo(videoId: string, userId: string) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: { not: null } },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  await prisma.video.update({
    where: { id: videoId },
    data: { deletedAt: null },
  });

  return { success: true };
}

// ============================================================================
// Session-Video Junction
// ============================================================================

export async function addVideoToSession(
  sessionId: string,
  videoId: string,
  userId: string,
  order?: number
) {
  // Verify session belongs to user
  const session = await prisma.session.findFirst({
    where: { id: sessionId, userId },
    include: {
      _count: { select: { sessionVideos: true } },
    },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  // Prevent manual additions to ALL_VIDEOS session
  if (session.type === "ALL_VIDEOS") {
    throw new ValidationError(
      "Cannot manually add videos to 'All Videos' session"
    );
  }

  if (session._count.sessionVideos >= MAX_VIDEOS_PER_SESSION) {
    throw new LimitExceededError(
      `Maximum of ${MAX_VIDEOS_PER_SESSION} videos per session`,
      { current: session._count.sessionVideos, limit: MAX_VIDEOS_PER_SESSION }
    );
  }

  // Verify video belongs to user and not deleted
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Check if already linked
  const existing = await prisma.sessionVideo.findUnique({
    where: {
      sessionId_videoId: { sessionId, videoId },
    },
  });

  if (existing) {
    throw new ConflictError("Video already in session", { sessionId, videoId });
  }

  // Use MAX(order) + 1 in a transaction to prevent order collisions
  const sessionVideo = await prisma.$transaction(async (tx) => {
    const maxOrder = await tx.sessionVideo.aggregate({
      where: { sessionId },
      _max: { order: true },
    });
    return tx.sessionVideo.create({
      data: {
        sessionId,
        videoId,
        order: order ?? (maxOrder._max.order ?? -1) + 1,
      },
    });
  });

  return sessionVideo;
}

export async function removeVideoFromSession(
  sessionId: string,
  videoId: string,
  userId: string
) {
  // Verify session belongs to user
  const session = await prisma.session.findFirst({
    where: { id: sessionId, userId },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  // Prevent manual removals from ALL_VIDEOS session
  if (session.type === "ALL_VIDEOS") {
    throw new ValidationError(
      "Cannot manually remove videos from 'All Videos' session"
    );
  }

  const sessionVideo = await prisma.sessionVideo.findUnique({
    where: {
      sessionId_videoId: { sessionId, videoId },
    },
  });

  if (!sessionVideo) {
    throw new NotFoundError("SessionVideo", `${sessionId}/${videoId}`);
  }

  // Use transaction to clean up highlights properly
  await prisma.$transaction(async (tx) => {
    // 1. Delete the session-video junction
    await tx.sessionVideo.delete({
      where: { id: sessionVideo.id },
    });

    // 2. Remove highlight-rally associations for rallies from this video
    await tx.highlightRally.deleteMany({
      where: {
        highlight: { sessionId },
        rally: { videoId },
      },
    });

    // 3. Delete highlights that are now empty (have no rallies)
    const emptyHighlights = await tx.highlight.findMany({
      where: {
        sessionId,
        highlightRallies: { none: {} },
      },
      select: { id: true },
    });

    if (emptyHighlights.length > 0) {
      await tx.highlight.deleteMany({
        where: {
          id: { in: emptyHighlights.map((h) => h.id) },
        },
      });
    }
  });

  return { success: true };
}

export async function reorderSessionVideos(
  sessionId: string,
  userId: string,
  videos: { videoId: string; order: number }[]
) {
  // Verify session belongs to user
  const session = await prisma.session.findFirst({
    where: { id: sessionId, userId },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  await prisma.$transaction(
    videos.map(({ videoId, order }) =>
      prisma.sessionVideo.updateMany({
        where: { sessionId, videoId },
        data: { order },
      })
    )
  );

  return { success: true };
}

// ============================================================================
// Legacy/Compatibility functions for session upload flow
// ============================================================================

/**
 * Request upload URL for a video that will be added to a session.
 * Creates Video + SessionVideo in one transaction.
 */
export async function requestUploadUrl(
  sessionId: string,
  userId: string,
  data: RequestUploadUrlInput
) {
  const session = await prisma.session.findFirst({
    where: { id: sessionId, userId },
    include: {
      _count: { select: { sessionVideos: true } },
    },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session._count.sessionVideos >= MAX_VIDEOS_PER_SESSION) {
    throw new LimitExceededError(
      `Maximum of ${MAX_VIDEOS_PER_SESSION} videos per session`,
      { current: session._count.sessionVideos, limit: MAX_VIDEOS_PER_SESSION }
    );
  }

  // Validate upload request (size, duration, quota, duplicates)
  const { expiresAt } = await validateUploadRequest({
    userId,
    fileSize: data.fileSize,
    durationMs: data.durationMs,
    contentHash: data.contentHash,
  });

  // Create/reuse video and link to session in transaction
  // Reuses PENDING videos to allow retries after failed uploads
  const { video } = await prisma.$transaction(async (tx) => {
    const result = await findOrCreatePendingVideo(tx, {
      userId,
      contentHash: data.contentHash,
      filename: data.filename,
      fileSize: data.fileSize,
      durationMs: data.durationMs,
      expiresAt,
    });

    // Handle SessionVideo junction
    // If reusing a video, it might already be linked to this session
    const existingSessionVideo = await tx.sessionVideo.findUnique({
      where: { sessionId_videoId: { sessionId, videoId: result.video.id } },
    });

    if (!existingSessionVideo) {
      // Link video to session with next order value
      const maxOrder = await tx.sessionVideo.aggregate({
        where: { sessionId },
        _max: { order: true },
      });

      await tx.sessionVideo.create({
        data: {
          sessionId,
          videoId: result.video.id,
          order: (maxOrder._max.order ?? -1) + 1,
        },
      });
    }

    return result;
  });

  const uploadUrl = await generateUploadUrl({
    key: video.s3Key,
    contentType: data.contentType,
    contentLength: data.fileSize,
  });

  return {
    uploadUrl,
    videoId: video.id,
    s3Key: video.s3Key,
  };
}

/**
 * Confirm upload for a video (session context for backwards compat).
 */
export async function confirmUpload(
  sessionId: string,
  userId: string,
  data: ConfirmUploadInput
) {
  // Verify video is in this session and belongs to user
  const sessionVideo = await prisma.sessionVideo.findFirst({
    where: {
      sessionId,
      videoId: data.videoId,
      session: { userId },
    },
    include: { video: true },
  });

  if (!sessionVideo) {
    throw new NotFoundError("Video", data.videoId);
  }

  // Atomically check and reserve upload quota (prevents race conditions)
  const tier = await getUserTier(userId);
  const limits = getTierLimits(tier);
  const quotaResult = await checkAndReserveUploadQuota(userId, limits);

  if (!quotaResult.allowed) {
    throw new LimitExceededError(
      `Monthly upload limit reached (${quotaResult.used}/${quotaResult.limit}). Upgrade for more uploads, or wait until next month.`,
      { used: quotaResult.used, limit: quotaResult.limit }
    );
  }

  const updated = await prisma.video.update({
    where: { id: data.videoId },
    data: {
      status: "UPLOADED",
      durationMs: data.durationMs,
      width: data.width,
      height: data.height,
    },
  });

  // Auto-add to "All Videos" session
  const allVideosSession = await getOrCreateAllVideosSession(userId);

  // Check if not already in the session
  const existingLink = await prisma.sessionVideo.findUnique({
    where: {
      sessionId_videoId: { sessionId: allVideosSession.id, videoId: data.videoId },
    },
  });

  if (!existingLink) {
    // Use MAX(order) + 1 in a transaction to prevent order collisions
    await prisma.$transaction(async (tx) => {
      const maxOrder = await tx.sessionVideo.aggregate({
        where: { sessionId: allVideosSession.id },
        _max: { order: true },
      });
      await tx.sessionVideo.create({
        data: {
          sessionId: allVideosSession.id,
          videoId: data.videoId,
          order: (maxOrder._max.order ?? -1) + 1,
        },
      });
    });
  }

  // Generate poster immediately for fast UX (async, ~2-3 seconds)
  generatePosterImmediate(data.videoId, updated.s3Key).catch((error) => {
    console.error(`[UPLOAD] Failed to generate poster for ${data.videoId}:`, error);
    // Non-fatal: processing pipeline will regenerate if missing
  });

  // Queue video for optimization processing (async, non-blocking)
  queueVideoProcessing(data.videoId, userId).catch((error) => {
    console.error(`[UPLOAD] Failed to queue processing for video ${data.videoId}:`, error);
  });

  return serializeBigInts(updated);
}

/**
 * Delete video - for session context, just removes from session.
 * For library context, does soft delete.
 */
export async function deleteVideo(videoId: string, userId: string) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Soft delete
  await prisma.video.update({
    where: { id: videoId },
    data: { deletedAt: new Date() },
  });

  return video.s3Key;
}

export async function getVideoById(id: string) {
  const video = await prisma.video.findUnique({
    where: { id },
    include: {
      rallies: {
        orderBy: { order: "asc" },
      },
    },
  });

  if (!video) {
    throw new NotFoundError("Video", id);
  }

  return serializeBigInts(video);
}

// ============================================================================
// Single Video Editor
// ============================================================================

/**
 * Get a video with all data needed for the single-video editor.
 * Returns the video with rallies, camera edits, and filtered highlights.
 */
export async function getVideoForEditor(videoId: string, userId: string) {
  // Fetch video with rallies, camera edits, and confirmation status
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
    include: {
      rallies: {
        orderBy: { order: "asc" },
        include: {
          cameraEdit: {
            include: {
              keyframes: {
                orderBy: { timeOffset: "asc" },
              },
            },
          },
        },
      },
      confirmation: true,
    },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Check if video is confirmed
  const isConfirmed = video.confirmation?.status === "CONFIRMED";
  // For confirmed videos, use the confirmation proxy for editing
  // and the trimmed duration for the player timeline
  const confirmationProxy = isConfirmed ? video.confirmation?.proxyS3Key : null;
  const effectiveDuration = isConfirmed
    ? video.confirmation?.trimmedDurationMs ?? video.durationMs
    : video.durationMs;

  // Get the ALL_VIDEOS session for this user (for syncing edits)
  const allVideosSession = await getOrCreateAllVideosSession(userId);

  // Get rally IDs for this video
  const videoRallyIds = video.rallies.map((r) => r.id);

  // Fetch highlights from ALL_VIDEOS session that contain rallies from this video
  const highlights = await prisma.highlight.findMany({
    where: {
      sessionId: allVideosSession.id,
      highlightRallies: {
        some: {
          rallyId: { in: videoRallyIds },
        },
      },
    },
    orderBy: { createdAt: "asc" },
    include: {
      highlightRallies: {
        orderBy: { order: "asc" },
        include: {
          rally: true,
        },
      },
      createdByUser: {
        select: {
          id: true,
          name: true,
        },
      },
    },
  });

  return serializeBigInts({
    video: {
      id: video.id,
      name: video.name,
      filename: video.filename,
      s3Key: video.s3Key,  // Original video (unchanged, used for exports)
      posterS3Key: video.posterS3Key,
      // For confirmed videos, use confirmation proxy for editing
      // For non-confirmed, use regular proxy/processed
      proxyS3Key: confirmationProxy ?? video.proxyS3Key,
      processedS3Key: isConfirmed ? null : video.processedS3Key,
      // For confirmed videos, return trimmed duration for player timeline
      durationMs: effectiveDuration,
      width: video.width,
      height: video.height,
      rallies: video.rallies,
    },
    allVideosSessionId: allVideosSession.id,
    highlights,
  });
}
