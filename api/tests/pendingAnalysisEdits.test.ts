import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { appendEdit, appendEditsBatch, consumePendingEdits } from '../src/services/pendingAnalysisEdits';

const userId = '11111111-1111-1111-1111-000000000b05';
const videoId = '22222222-2222-2222-2222-000000000b05';
const rallyA = '33333333-3333-3333-3333-000000000b0a';
const rallyB = '33333333-3333-3333-3333-000000000b0b';

describe('pendingAnalysisEdits', () => {
  beforeEach(async () => {
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: { id: videoId, userId, name: 'b', filename: 'b.mp4', s3Key: 'test/b.mp4', contentHash: 'deadbeef', durationMs: 60000 },
    });
  });

  afterEach(async () => {
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
  });

  it('appendEdit creates entries list when null', async () => {
    await appendEdit(prisma, videoId, rallyA, 'split');
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect((v!.pendingAnalysisEditsJson as any).entries).toHaveLength(1);
    expect((v!.pendingAnalysisEditsJson as any).entries[0]).toMatchObject({ rallyId: rallyA, editKind: 'split' });
  });

  it('appendEdit appends to existing entries', async () => {
    await appendEdit(prisma, videoId, rallyA, 'split');
    await appendEdit(prisma, videoId, rallyB, 'delete');
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect((v!.pendingAnalysisEditsJson as any).entries).toHaveLength(2);
  });

  it('appendEditsBatch writes multiple entries in one update', async () => {
    await appendEditsBatch(prisma, videoId, [
      { rallyId: rallyA, editKind: 'split' },
      { rallyId: rallyB, editKind: 'split' },
    ]);
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect((v!.pendingAnalysisEditsJson as any).entries).toHaveLength(2);
  });

  it('consumePendingEdits returns entries and nulls the column', async () => {
    await appendEdit(prisma, videoId, rallyA, 'shorten');
    const result = await consumePendingEdits(videoId);
    expect(result.entries).toHaveLength(1);
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v!.pendingAnalysisEditsJson).toBeNull();
  });

  it('consumePendingEdits on empty returns empty entries', async () => {
    const result = await consumePendingEdits(videoId);
    expect(result.entries).toEqual([]);
  });
});
