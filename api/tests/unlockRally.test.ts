import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { unlockRally } from '../src/services/rallyService';

const userId = '11111111-1111-1111-1111-000000000b04';
const videoId = '22222222-2222-2222-2222-000000000b04';
const rallyId = '33333333-3333-3333-3333-000000000b04';

describe('unlockRally', () => {
  beforeEach(async () => {
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: {
        id: videoId,
        userId,
        name: 'b',
        filename: 'b.mp4',
        s3Key: 'test/b.mp4',
        contentHash: 'deadbeef',
        durationMs: 60000,
        matchAnalysisJson: {
          videoId,
          numRallies: 1,
          rallies: [{ rallyId, canonicalLocked: true, trackToPlayer: { '1': 1 }, assignmentConfidence: 0.9 }],
        },
      },
    });
    await prisma.rally.create({ data: { id: rallyId, videoId, startMs: 0, endMs: 5000 } });
  });

  afterEach(async () => {
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
  });

  it('clears canonicalLocked flag and preserves other rally fields', async () => {
    const result = await unlockRally(rallyId, userId);
    expect(result.wasLocked).toBe(true);
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    const entry = (v!.matchAnalysisJson as any).rallies[0];
    expect(entry.canonicalLocked).toBe(false);
    expect(entry.trackToPlayer).toEqual({ '1': 1 });
    expect(entry.assignmentConfidence).toBe(0.9);
  });

  it('is idempotent on already-unlocked rally', async () => {
    await unlockRally(rallyId, userId);
    const result = await unlockRally(rallyId, userId);
    expect(result.wasLocked).toBe(false);
  });
});
