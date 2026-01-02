import { prisma } from "../lib/prisma.js";
import { generateUploadUrl, getVideoS3Key } from "../lib/s3.js";
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

const MAX_VIDEOS_PER_SESSION = 5;

// Helper to convert BigInt fields to strings for JSON serialization
function serializeBigInts<T>(obj: T): T {
  return JSON.parse(
    JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value
    )
  );
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
        _count: {
          select: { sessionVideos: true },
        },
        sessionVideos: {
          include: {
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
      sessionCount: v._count.sessionVideos,
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
  // Check for duplicate content hash for this user
  const existingVideo = await prisma.video.findFirst({
    where: {
      userId: params.userId,
      contentHash: params.contentHash,
      deletedAt: null,
    },
  });

  if (existingVideo) {
    throw new ConflictError("Video with this content already exists", {
      existingVideoId: existingVideo.id,
    });
  }

  const videoId = crypto.randomUUID();
  const s3Key = getVideoS3Key(params.userId, videoId, params.filename);

  const video = await prisma.video.create({
    data: {
      id: videoId,
      userId: params.userId,
      name: params.filename.replace(/\.[^.]+$/, ""),
      filename: params.filename,
      s3Key,
      contentHash: params.contentHash,
      fileSizeBytes: params.fileSize,
      durationMs: params.durationMs,
    },
  });

  const uploadUrl = await generateUploadUrl({
    key: s3Key,
    contentType: params.contentType,
    contentLength: params.fileSize,
  });

  return {
    uploadUrl,
    videoId: video.id,
    s3Key,
  };
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
    const videoCount = await prisma.sessionVideo.count({
      where: { sessionId: allVideosSession.id },
    });

    await prisma.sessionVideo.create({
      data: {
        sessionId: allVideosSession.id,
        videoId,
        order: videoCount,
      },
    });
  }

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

  const sessionVideo = await prisma.sessionVideo.create({
    data: {
      sessionId,
      videoId,
      order: order ?? session._count.sessionVideos,
    },
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

  await prisma.sessionVideo.delete({
    where: { id: sessionVideo.id },
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

  // Check for duplicate content hash for this user
  const existingVideo = await prisma.video.findFirst({
    where: {
      userId,
      contentHash: data.contentHash,
      deletedAt: null,
    },
  });

  if (existingVideo) {
    throw new ConflictError("Video with this content already exists", {
      existingVideoId: existingVideo.id,
    });
  }

  const videoId = crypto.randomUUID();
  const s3Key = getVideoS3Key(userId, videoId, data.filename);

  // Create video and link to session in transaction
  const result = await prisma.$transaction(async (tx) => {
    const video = await tx.video.create({
      data: {
        id: videoId,
        userId,
        name: data.filename.replace(/\.[^.]+$/, ""),
        filename: data.filename,
        s3Key,
        contentHash: data.contentHash,
        fileSizeBytes: data.fileSize,
        durationMs: data.durationMs,
      },
    });

    await tx.sessionVideo.create({
      data: {
        sessionId,
        videoId: video.id,
        order: session._count.sessionVideos,
      },
    });

    return video;
  });

  const uploadUrl = await generateUploadUrl({
    key: s3Key,
    contentType: data.contentType,
    contentLength: data.fileSize,
  });

  return {
    uploadUrl,
    videoId: result.id,
    s3Key,
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
    const videoCount = await prisma.sessionVideo.count({
      where: { sessionId: allVideosSession.id },
    });

    await prisma.sessionVideo.create({
      data: {
        sessionId: allVideosSession.id,
        videoId: data.videoId,
        order: videoCount,
      },
    });
  }

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
