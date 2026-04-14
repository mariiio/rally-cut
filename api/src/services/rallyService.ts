import { Prisma } from "@prisma/client";
import { prisma } from "../lib/prisma.js";
import { ConflictError, ForbiddenError, NotFoundError } from "../middleware/errorHandler.js";
import type { CreateRallyInput, UpdateRallyInput } from "../schemas/rally.js";
import { isBatchTrackingActive } from "./batchTrackingService.js";
import { reindexTrackingData } from "./playerTrackingService.js";
import { canAccessVideoRallies } from "./shareService.js";

export async function listRallies(videoId: string, userId: string) {
  // canAccessVideoRallies throws NotFoundError if video doesn't exist
  const hasAccess = await canAccessVideoRallies(videoId, userId, false);
  if (!hasAccess) {
    throw new ForbiddenError("You do not have permission to access this video's rallies");
  }

  return prisma.rally.findMany({
    where: { videoId },
    orderBy: { order: "asc" },
  });
}

export async function createRally(videoId: string, userId: string, data: CreateRallyInput) {
  // canAccessVideoRallies throws NotFoundError if video doesn't exist
  const hasAccess = await canAccessVideoRallies(videoId, userId, true);
  if (!hasAccess) {
    throw new ForbiddenError("You do not have permission to create rallies for this video");
  }

  if (await isBatchTrackingActive(videoId)) {
    throw new ConflictError(
      'New rallies cannot be added while tracking is running. Please wait for tracking to finish.',
      { reason: 'CREATE_DURING_TRACKING' },
    );
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
    select: { videoId: true, startMs: true, endMs: true },
  });

  if (rally === null) {
    throw new NotFoundError("Rally", id);
  }

  // canAccessVideoRallies throws NotFoundError if video doesn't exist
  const hasAccess = await canAccessVideoRallies(rally.videoId, userId, true);
  if (!hasAccess) {
    throw new ForbiddenError("You do not have permission to update this rally");
  }

  const boundaryChanged =
    (data.startMs !== undefined && data.startMs !== rally.startMs) ||
    (data.endMs !== undefined && data.endMs !== rally.endMs);

  if (boundaryChanged) {
    return prisma.$transaction(async (tx) => {
      const updated = await tx.rally.update({ where: { id }, data });
      const reindexed = await reindexTrackingData(
        tx, id,
        rally.startMs, rally.endMs,
        data.startMs ?? rally.startMs, data.endMs ?? rally.endMs,
      );
      if (reindexed) {
        await tx.video.update({
          where: { id: rally.videoId },
          data: { matchAnalysisJson: Prisma.DbNull, matchStatsJson: Prisma.DbNull },
        });
      }
      return updated;
    });
  }

  return prisma.rally.update({
    where: { id },
    data,
  });
}

export async function deleteRally(id: string, userId: string) {
  const rally = await prisma.rally.findUnique({
    where: { id },
    select: { videoId: true },
  });

  if (rally === null) {
    throw new NotFoundError("Rally", id);
  }

  // canAccessVideoRallies throws NotFoundError if video doesn't exist
  const hasAccess = await canAccessVideoRallies(rally.videoId, userId, true);
  if (!hasAccess) {
    throw new ForbiddenError("You do not have permission to delete this rally");
  }

  await prisma.rally.delete({
    where: { id },
  });
}
