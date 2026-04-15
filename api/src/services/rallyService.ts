import { Prisma } from "@prisma/client";
import { prisma } from "../lib/prisma.js";
import { ForbiddenError, LockedRallyRequiresConfirmError, NotFoundError, RalliesOverlapError, RallyTrackingStateError, SplitBoundsError } from "../middleware/errorHandler.js";
import type { CreateRallyInput, UpdateRallyInput } from "../schemas/rally.js";
import { reindexTrackingData } from "./playerTrackingService.js";
import { markRetrackIfExtended } from "./batchTrackingService.js";
import { canAccessVideoRallies } from "./shareService.js";
import { appendEdit, appendEditsBatch } from "./pendingAnalysisEdits.js";
import { slicePlayerTrack, concatPlayerTracks } from "./rallySlicing.js";
import { assertNotLocked } from "./canonicalLockGuard.js";

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
      // Guard: extending bounds invalidates GT — reject if rally is locked
      const willExtend = (data.startMs ?? rally.startMs) < rally.startMs || (data.endMs ?? rally.endMs) > rally.endMs;
      if (willExtend) {
        await assertNotLocked(tx, id, 'EXTEND');
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

export type SplitRallyInput = { firstEndMs: number; secondStartMs: number };

export async function splitRally(
  rallyId: string,
  userId: string,
  input: SplitRallyInput,
): Promise<{ firstRally: Awaited<ReturnType<typeof prisma.rally.create>>; secondRally: Awaited<ReturnType<typeof prisma.rally.create>> }> {
  return prisma.$transaction(async (tx) => {
    const rally = await tx.rally.findUnique({
      where: { id: rallyId },
      include: { playerTrack: true, video: { select: { userId: true, matchAnalysisJson: true } } },
    });
    if (!rally || rally.video.userId !== userId) throw new NotFoundError('Rally', rallyId);

    // Bounds validation
    const { firstEndMs, secondStartMs } = input;
    if (!(rally.startMs < firstEndMs && firstEndMs <= secondStartMs && secondStartMs < rally.endMs)) {
      throw new SplitBoundsError(
        'must satisfy startMs < firstEndMs <= secondStartMs < endMs',
        { startMs: rally.startMs, firstEndMs, secondStartMs, endMs: rally.endMs },
      );
    }

    // Lock gate
    await assertNotLocked(tx, rallyId, 'SPLIT');

    // PlayerTrack state gate
    const pt = rally.playerTrack;
    if (pt) {
      if (pt.status === 'PROCESSING') throw new RallyTrackingStateError('IN_PROGRESS', rallyId);
      if (pt.status === 'FAILED') throw new RallyTrackingStateError('FAILED', rallyId);
    }

    // Frame math
    const fps = pt?.fps ?? 30;
    const firstEndFrame = Math.round(((firstEndMs - rally.startMs) / 1000) * fps);
    const secondStartFrame = Math.round(((secondStartMs - rally.startMs) / 1000) * fps);

    // Previous rally lookup for first-child score inheritance
    const prevRally = await tx.rally.findFirst({
      where: { videoId: rally.videoId, endMs: { lte: rally.startMs } },
      orderBy: { endMs: 'desc' },
    });

    // Create child Rally rows
    const firstRally = await tx.rally.create({
      data: {
        videoId: rally.videoId,
        startMs: rally.startMs,
        endMs: firstEndMs,
        scoreA: prevRally?.scoreA ?? 0,
        scoreB: prevRally?.scoreB ?? 0,
        servingTeam: prevRally?.servingTeam ?? rally.servingTeam,
        notes: rally.notes,
        confidence: rally.confidence,
      },
    });
    const secondRally = await tx.rally.create({
      data: {
        videoId: rally.videoId,
        startMs: secondStartMs,
        endMs: rally.endMs,
        scoreA: rally.scoreA,
        scoreB: rally.scoreB,
        servingTeam: rally.servingTeam,
        notes: rally.notes,
        confidence: rally.confidence,
      },
    });

    // Slice PlayerTrack if present
    if (pt) {
      const { first, second } = slicePlayerTrack(pt as any, firstEndFrame, secondStartFrame);
      await tx.playerTrack.create({
        data: { rallyId: firstRally.id, ...(first as any) },
      });
      await tx.playerTrack.create({
        data: { rallyId: secondRally.id, ...(second as any) },
      });
    }

    // Update matchAnalysisJson.rallies[]
    const json: any = rally.video.matchAnalysisJson ?? { rallies: [] };
    const parentEntry = (json.rallies ?? []).find((r: any) => r.rallyId === rallyId);
    const inheritMapping = parentEntry?.trackToPlayer ?? {};
    const inheritServer = parentEntry?.serverPlayerId ?? null;
    json.rallies = (json.rallies ?? []).filter((r: any) => r.rallyId !== rallyId);
    json.rallies.push(
      { rallyId: firstRally.id, canonicalLocked: false, trackToPlayer: inheritMapping, assignmentConfidence: parentEntry?.assignmentConfidence ?? null, serverPlayerId: inheritServer },
      { rallyId: secondRally.id, canonicalLocked: false, trackToPlayer: inheritMapping, assignmentConfidence: parentEntry?.assignmentConfidence ?? null, serverPlayerId: inheritServer },
    );
    await tx.video.update({ where: { id: rally.videoId }, data: { matchAnalysisJson: json } });

    // Delete parent (cascades PlayerTrack)
    await tx.rally.delete({ where: { id: rallyId } });

    // Record edits last
    await appendEditsBatch(tx, rally.videoId, [
      { rallyId: firstRally.id, editKind: 'split' },
      { rallyId: secondRally.id, editKind: 'split' },
    ]);

    return { firstRally, secondRally };
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

export async function mergeRallies(
  rallyIds: [string, string],
  userId: string,
): Promise<{ rally: Awaited<ReturnType<typeof prisma.rally.create>> }> {
  return prisma.$transaction(async (tx) => {
    const raw = await tx.rally.findMany({
      where: { id: { in: rallyIds } },
      include: { playerTrack: true, video: { select: { userId: true, matchAnalysisJson: true } } },
    });
    if (raw.length !== 2) throw new NotFoundError('Rally', rallyIds.join(','));
    if (raw[0].video.userId !== userId || raw[1].video.userId !== userId) {
      throw new NotFoundError('Rally', rallyIds.join(','));
    }
    if (raw[0].videoId !== raw[1].videoId) throw new RalliesOverlapError(rallyIds);

    const [a, b] = [...raw].sort((x, y) => x.startMs - y.startMs);
    if (b.startMs < a.endMs) throw new RalliesOverlapError(rallyIds);

    // Lock gate on both
    await assertNotLocked(tx, a.id, 'MERGE');
    await assertNotLocked(tx, b.id, 'MERGE');

    // PlayerTrack state gate on both
    for (const r of [a, b]) {
      if (r.playerTrack?.status === 'PROCESSING') throw new RallyTrackingStateError('IN_PROGRESS', r.id);
      if (r.playerTrack?.status === 'FAILED') throw new RallyTrackingStateError('FAILED', r.id);
    }

    const gap = b.startMs > a.endMs;

    // Create merged rally
    const merged = await tx.rally.create({
      data: {
        videoId: a.videoId,
        startMs: a.startMs,
        endMs: b.endMs,
        scoreA: b.scoreA,
        scoreB: b.scoreB,
        servingTeam: b.servingTeam,
        notes: [a.notes, b.notes].filter(Boolean).join('\n') || null,
        confidence: Math.min(a.confidence ?? 1, b.confidence ?? 1),
      },
    });

    // Concat tracks only when there's no gap AND both tracks exist
    if (!gap && a.playerTrack && b.playerTrack) {
      const stitched = concatPlayerTracks(a.playerTrack as any, b.playerTrack as any);
      await tx.playerTrack.create({ data: { rallyId: merged.id, ...(stitched as any) } });
    }

    // Update matchAnalysisJson.rallies[]
    const json: any = a.video.matchAnalysisJson ?? { rallies: [] };
    const aEntry = (json.rallies ?? []).find((r: any) => r.rallyId === a.id);
    json.rallies = (json.rallies ?? []).filter((r: any) => r.rallyId !== a.id && r.rallyId !== b.id);
    json.rallies.push({
      rallyId: merged.id, canonicalLocked: false,
      trackToPlayer: aEntry?.trackToPlayer ?? {},
      assignmentConfidence: null, serverPlayerId: null,
    });
    await tx.video.update({ where: { id: a.videoId }, data: { matchAnalysisJson: json } });

    // Delete parents (cascades their PlayerTracks)
    await tx.rally.deleteMany({ where: { id: { in: [a.id, b.id] } } });

    // Record edit
    await appendEditsBatch(tx, a.videoId, [{ rallyId: merged.id, editKind: 'merge' }]);

    return { rally: merged };
  });
}
