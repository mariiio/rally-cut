import { Prisma } from "@prisma/client";
import { prisma } from "../lib/prisma.js";
import { ForbiddenError, LockedRallyRequiresConfirmError, NotFoundError } from "../middleware/errorHandler.js";
import type { CreateRallyInput, UpdateRallyInput } from "../schemas/rally.js";
import { reindexTrackingData } from "./playerTrackingService.js";
import { markRetrackIfExtended } from "./batchTrackingService.js";
import { canAccessVideoRallies } from "./shareService.js";
import { appendEdit } from "./pendingAnalysisEdits.js";

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

  // Use MAX(order) + 1 in a transaction to prevent order collisions
  return prisma.$transaction(async (tx) => {
    const maxOrder = await tx.rally.aggregate({
      where: { videoId },
      _max: { order: true },
    });

    const rally = await tx.rally.create({
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

    await appendEdit(tx, videoId, rally.id, 'create');

    return rally;
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
      // Mark for retrack if bounds were extended (inside tx — rolled back atomically on failure)
      await markRetrackIfExtended(
        tx,
        id,
        { startMs: rally.startMs, endMs: rally.endMs },
        { startMs: data.startMs ?? rally.startMs, endMs: data.endMs ?? rally.endMs },
      );

      // Classify edit kind
      const newStart = data.startMs ?? rally.startMs;
      const newEnd = data.endMs ?? rally.endMs;
      const extended = newStart < rally.startMs || newEnd > rally.endMs;
      const editKind: 'shorten' | 'extend' = extended ? 'extend' : 'shorten';
      await appendEdit(tx, rally.videoId, id, editKind);

      return updated;
    });
  }

  return prisma.$transaction(async (tx) => {
    const updated = await tx.rally.update({ where: { id }, data });

    // Classify edit kind for scalar fields
    let editKind: 'scalar' | null = null;
    if (data.scoreA !== undefined || data.scoreB !== undefined || data.servingTeam !== undefined || data.notes !== undefined) {
      editKind = 'scalar';
    }
    if (editKind) await appendEdit(tx, rally.videoId, id, editKind);

    return updated;
  });
}

export async function deleteRally(id: string, userId: string, opts?: { confirmUnlock?: boolean }): Promise<void> {
  await prisma.$transaction(async (tx) => {
    const rally = await tx.rally.findUnique({
      where: { id },
      select: { videoId: true, video: { select: { userId: true, matchAnalysisJson: true } } },
    });
    if (!rally || rally.video.userId !== userId) throw new NotFoundError('Rally', id);

    // Canonical-lock confirm gate
    const json: any = rally.video.matchAnalysisJson ?? { rallies: [] };
    const entry = (json.rallies ?? []).find((r: any) => r.rallyId === id);
    const locked = entry?.canonicalLocked === true;
    if (locked && !opts?.confirmUnlock) {
      // Best-effort count of GT frames for the error payload
      const pt = await tx.playerTrack.findUnique({ where: { rallyId: id }, select: { groundTruthJson: true, actionGroundTruthJson: true } });
      const gtFrameCount = ((pt?.groundTruthJson as any)?.length ?? 0) + ((pt?.actionGroundTruthJson as any)?.length ?? 0);
      throw new LockedRallyRequiresConfirmError(id, gtFrameCount);
    }

    // Audit log for the destructive confirmed path
    if (locked && opts?.confirmUnlock) {
      console.log(JSON.stringify({
        event: 'rally.locked.deleted',
        rallyId: id, videoId: rally.videoId, userId,
        gtFrameCount: entry?.gtFrameCount ?? null,
      }));
    }

    // Also drop the rally from matchAnalysisJson.rallies[] so stats don't reference it
    if (json.rallies) {
      json.rallies = json.rallies.filter((r: any) => r.rallyId !== id);
      await tx.video.update({ where: { id: rally.videoId }, data: { matchAnalysisJson: json } });
    }

    await tx.rally.delete({ where: { id } });

    await appendEdit(tx, rally.videoId, id, 'delete');
  });
}

export async function unlockRally(
  rallyId: string,
  userId: string,
): Promise<{ rallyId: string; wasLocked: boolean; unlockedAt: Date }> {
  return prisma.$transaction(async (tx) => {
    const rally = await tx.rally.findUnique({
      where: { id: rallyId },
      select: { videoId: true, video: { select: { userId: true, matchAnalysisJson: true } } },
    });
    if (!rally || rally.video.userId !== userId) throw new NotFoundError('Rally', rallyId);

    const json: any = rally.video.matchAnalysisJson ?? { rallies: [] };
    const entry = (json.rallies ?? []).find((r: any) => r.rallyId === rallyId);
    const wasLocked = entry?.canonicalLocked === true;

    if (wasLocked) {
      entry.canonicalLocked = false;
      await tx.video.update({
        where: { id: rally.videoId },
        data: { matchAnalysisJson: json },
      });
    }

    return { rallyId, wasLocked, unlockedAt: new Date() };
  });
}
