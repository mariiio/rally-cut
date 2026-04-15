import { PrismaClient, Prisma } from '@prisma/client';
import { prisma } from '../lib/prisma.js';

type Client = PrismaClient | Prisma.TransactionClient;

export type EditKind = 'scalar' | 'shorten' | 'extend' | 'delete' | 'split' | 'merge' | 'create';

export type PendingEdit = { rallyId: string; editKind: EditKind; at: string };

export type PendingEditsJson = { entries: PendingEdit[] };

export async function appendEdit(
  client: Client,
  videoId: string,
  rallyId: string,
  editKind: EditKind,
): Promise<void> {
  await appendEditsBatch(client, videoId, [{ rallyId, editKind }]);
}

export async function appendEditsBatch(
  client: Client,
  videoId: string,
  edits: Array<{ rallyId: string; editKind: EditKind }>,
): Promise<void> {
  const video = await client.video.findUnique({
    where: { id: videoId },
    select: { pendingAnalysisEditsJson: true },
  });
  const existing = (video?.pendingAnalysisEditsJson as PendingEditsJson | null)?.entries ?? [];
  const at = new Date().toISOString();
  const next: PendingEditsJson = {
    entries: [...existing, ...edits.map(e => ({ ...e, at }))],
  };
  await client.video.update({
    where: { id: videoId },
    data: { pendingAnalysisEditsJson: next as unknown as Prisma.InputJsonValue },
  });
}

export async function consumePendingEdits(videoId: string): Promise<PendingEditsJson> {
  return prisma.$transaction(async (tx) => {
    const video = await tx.video.findUnique({
      where: { id: videoId },
      select: { pendingAnalysisEditsJson: true },
    });
    const entries = (video?.pendingAnalysisEditsJson as PendingEditsJson | null)?.entries ?? [];
    if (entries.length > 0) {
      await tx.video.update({
        where: { id: videoId },
        data: { pendingAnalysisEditsJson: Prisma.DbNull },
      });
    }
    return { entries };
  });
}
