import { prisma } from "../lib/prisma.js";
import { generateUploadUrl, getVideoS3Key } from "../lib/s3.js";
import {
  ConflictError,
  LimitExceededError,
  NotFoundError,
} from "../middleware/errorHandler.js";
import type {
  ConfirmUploadInput,
  RequestUploadUrlInput,
  UpdateVideoInput,
} from "../schemas/video.js";

const MAX_VIDEOS_PER_SESSION = 5;

export async function requestUploadUrl(
  sessionId: string,
  data: RequestUploadUrlInput
) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: {
      _count: { select: { videos: true } },
    },
  });

  if (session === null) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session._count.videos >= MAX_VIDEOS_PER_SESSION) {
    throw new LimitExceededError(
      `Maximum of ${MAX_VIDEOS_PER_SESSION} videos per session`,
      { current: session._count.videos, limit: MAX_VIDEOS_PER_SESSION }
    );
  }

  const existingVideo = await prisma.video.findFirst({
    where: {
      sessionId,
      contentHash: data.contentHash,
    },
  });

  if (existingVideo !== null) {
    throw new ConflictError("Video with this content already exists", {
      existingVideoId: existingVideo.id,
    });
  }

  const videoId = crypto.randomUUID();
  const s3Key = getVideoS3Key(sessionId, videoId, data.filename);

  const video = await prisma.video.create({
    data: {
      id: videoId,
      sessionId,
      name: data.filename.replace(/\.[^.]+$/, ""),
      filename: data.filename,
      s3Key,
      contentHash: data.contentHash,
      fileSizeBytes: data.fileSize,
      durationMs: data.durationMs,
      order: session._count.videos,
    },
  });

  const uploadUrl = await generateUploadUrl({
    key: s3Key,
    contentType: data.contentType,
    contentLength: data.fileSize,
  });

  return {
    uploadUrl,
    videoId: video.id,
    s3Key,
  };
}

export async function confirmUpload(
  sessionId: string,
  data: ConfirmUploadInput
) {
  const video = await prisma.video.findFirst({
    where: {
      id: data.videoId,
      sessionId,
    },
  });

  if (video === null) {
    throw new NotFoundError("Video", data.videoId);
  }

  return prisma.video.update({
    where: { id: data.videoId },
    data: {
      status: "UPLOADED",
      durationMs: data.durationMs,
      width: data.width,
      height: data.height,
    },
  });
}

export async function updateVideo(id: string, data: UpdateVideoInput) {
  const video = await prisma.video.findUnique({
    where: { id },
  });

  if (video === null) {
    throw new NotFoundError("Video", id);
  }

  return prisma.video.update({
    where: { id },
    data,
  });
}

export async function deleteVideo(id: string) {
  const video = await prisma.video.findUnique({
    where: { id },
  });

  if (video === null) {
    throw new NotFoundError("Video", id);
  }

  await prisma.video.delete({
    where: { id },
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

  if (video === null) {
    throw new NotFoundError("Video", id);
  }

  return video;
}
