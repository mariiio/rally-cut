import { PrismaClient, Prisma } from '@prisma/client';
import { LockedRallyError } from '../middleware/errorHandler.js';

type Client = PrismaClient | Prisma.TransactionClient;

type MatchAnalysisRallyEntry = {
  rallyId: string;
  canonicalLocked?: boolean;
};

type MatchAnalysisJson = {
  rallies?: MatchAnalysisRallyEntry[];
};

export async function isRallyLocked(client: Client, rallyId: string): Promise<boolean> {
  const rally = await client.rally.findUnique({
    where: { id: rallyId },
    select: { video: { select: { matchAnalysisJson: true } } },
  });
  const json = rally?.video?.matchAnalysisJson as MatchAnalysisJson | null | undefined;
  const entry = json?.rallies?.find(r => r.rallyId === rallyId);
  return entry?.canonicalLocked === true;
}

export async function assertNotLocked(
  client: Client,
  rallyId: string,
  op: 'EXTEND' | 'SPLIT' | 'MERGE',
): Promise<void> {
  if (await isRallyLocked(client, rallyId)) {
    throw new LockedRallyError(op, rallyId);
  }
}
