import { prisma } from "../lib/prisma.js";
import { NotFoundError } from "../middleware/errorHandler.js";
import type { CreateRallyInput, UpdateRallyInput } from "../schemas/rally.js";

export async function listRallies(videoId: string) {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
  });

  if (video === null) {
    throw new NotFoundError("Video", videoId);
  }

  return prisma.rally.findMany({
    where: { videoId },
    orderBy: { order: "asc" },
  });
}

export async function createRally(videoId: string, data: CreateRallyInput) {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
  });

  if (video === null) {
    throw new NotFoundError("Video", videoId);
  }

  const maxOrder = await prisma.rally.aggregate({
    where: { videoId },
    _max: { order: true },
  });

  return prisma.rally.create({
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
}

export async function updateRally(id: string, data: UpdateRallyInput) {
  const rally = await prisma.rally.findUnique({
    where: { id },
  });

  if (rally === null) {
    throw new NotFoundError("Rally", id);
  }

  return prisma.rally.update({
    where: { id },
    data,
  });
}

export async function deleteRally(id: string) {
  const rally = await prisma.rally.findUnique({
    where: { id },
  });

  if (rally === null) {
    throw new NotFoundError("Rally", id);
  }

  await prisma.rally.delete({
    where: { id },
  });
}
