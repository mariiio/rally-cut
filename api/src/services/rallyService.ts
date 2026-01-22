import { prisma } from "../lib/prisma.js";
import { ForbiddenError, NotFoundError } from "../middleware/errorHandler.js";
import type { CreateRallyInput, UpdateRallyInput } from "../schemas/rally.js";

export async function listRallies(videoId: string, userId: string) {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
  });

  if (video === null) {
    throw new NotFoundError("Video", videoId);
  }

  if (video.userId !== userId) {
    throw new ForbiddenError("You do not have permission to access this video's rallies");
  }

  return prisma.rally.findMany({
    where: { videoId },
    orderBy: { order: "asc" },
  });
}

export async function createRally(videoId: string, userId: string, data: CreateRallyInput) {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
  });

  if (video === null) {
    throw new NotFoundError("Video", videoId);
  }

  if (video.userId !== userId) {
    throw new ForbiddenError("You do not have permission to create rallies for this video");
  }

  // Use MAX(order) + 1 in a transaction to prevent order collisions
  return prisma.$transaction(async (tx) => {
    const maxOrder = await tx.rally.aggregate({
      where: { videoId },
      _max: { order: true },
    });

    return tx.rally.create({
      data: {
        videoId,
        startMs: data.startMs,
        endMs: data.endMs,
        confidence: data.confidence,
        scoreA: data.scoreA,
        scoreB: data.scoreB,
        servingTeam: data.servingTeam,
        notes: data.notes,
        order: (maxOrder._max.order ?? -1) + 1,
      },
    });
  });
}

export async function updateRally(id: string, userId: string, data: UpdateRallyInput) {
  const rally = await prisma.rally.findUnique({
    where: { id },
    include: { video: true },
  });

  if (rally === null) {
    throw new NotFoundError("Rally", id);
  }

  if (rally.video.userId !== userId) {
    throw new ForbiddenError("You do not have permission to update this rally");
  }

  return prisma.rally.update({
    where: { id },
    data,
  });
}

export async function deleteRally(id: string, userId: string) {
  const rally = await prisma.rally.findUnique({
    where: { id },
    include: { video: true },
  });

  if (rally === null) {
    throw new NotFoundError("Rally", id);
  }

  if (rally.video.userId !== userId) {
    throw new ForbiddenError("You do not have permission to delete this rally");
  }

  await prisma.rally.delete({
    where: { id },
  });
}
