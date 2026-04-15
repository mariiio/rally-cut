import 'dotenv/config';
import { afterEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { mergeRallies } from '../src/services/rallyService';
import { RalliesOverlapError, LockedRallyError } from '../src/middleware/errorHandler';

const userId = '11111111-1111-1111-1111-000000000b12';
const videoId = '22222222-2222-2222-2222-000000000b12';
const rallyA = '33333333-3333-3333-3333-000000000b1a';
const rallyB = '33333333-3333-3333-3333-000000000b1b';

async function setup(opts: { gap?: boolean; lockB?: boolean } = {}) {
  await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
  await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
  await prisma.video.create({
    data: {
      id: videoId, userId, name: 'b', filename: 'b.mp4', s3Key: 'test/b.mp4', contentHash: 'deadbeef-b12',
      durationMs: 60000,
      matchAnalysisJson: {
        videoId, numRallies: 2,
        rallies: [
          { rallyId: rallyA, canonicalLocked: false, trackToPlayer: { '1': 1 }, assignmentConfidence: 0.9 },
          { rallyId: rallyB, canonicalLocked: opts.lockB === true, trackToPlayer: { '1': 1 }, assignmentConfidence: 0.9 },
        ],
      },
    },
  });
  const bStart = opts.gap ? 6000 : 5000;
  await prisma.rally.createMany({
    data: [
      { id: rallyA, videoId, startMs: 0, endMs: 5000, scoreA: 1, scoreB: 0, servingTeam: 'A' },
      { id: rallyB, videoId, startMs: bStart, endMs: 10000, scoreA: 2, scoreB: 0, servingTeam: 'A' },
    ],
  });
  await prisma.playerTrack.createMany({
    data: [
      {
        rallyId: rallyA, status: 'COMPLETED', fps: 30, frameCount: 150, detectionRate: 1, avgConfidence: 0.9,
        avgPlayerCount: 4, uniqueTrackCount: 2, courtSplitY: 0.5, processingTimeMs: 500, modelVersion: 'v1', needsRetrack: false,
        positionsJson: [{ frameNumber: 10, trackId: 1, confidence: 0.9 }],
        rawPositionsJson: [], ballPositionsJson: [], contactsJson: [{ frame: 50, playerTrackId: 1 }], actionsJson: [],
      } as any,
      {
        rallyId: rallyB, status: 'COMPLETED', fps: 30, frameCount: opts.gap ? 120 : 150, detectionRate: 1, avgConfidence: 0.9,
        avgPlayerCount: 4, uniqueTrackCount: 2, courtSplitY: 0.5, processingTimeMs: 500, modelVersion: 'v1', needsRetrack: false,
        positionsJson: [{ frameNumber: 5, trackId: 2, confidence: 0.85 }],
        rawPositionsJson: [], ballPositionsJson: [], contactsJson: [{ frame: 30, playerTrackId: 2 }], actionsJson: [],
      } as any,
    ],
  });
}

async function cleanup() {
  await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
}

describe('mergeRallies', () => {
  afterEach(cleanup);

  it('no-gap concat: produces one rally with stitched PlayerTrack', async () => {
    await setup({ gap: false });
    const { rally } = await mergeRallies([rallyA, rallyB], userId);
    expect(rally.startMs).toBe(0);
    expect(rally.endMs).toBe(10000);
    const pt = await prisma.playerTrack.findUnique({ where: { rallyId: rally.id } });
    expect(pt!.frameCount).toBe(300);
    expect((pt!.contactsJson as any).map((c: any) => c.frame).sort((x: number, y: number) => x - y)).toEqual([50, 180]); // 30 + 150
    expect(await prisma.rally.findUnique({ where: { id: rallyA } })).toBeNull();
    expect(await prisma.rally.findUnique({ where: { id: rallyB } })).toBeNull();
  });

  it('gap: creates new rally without PlayerTrack, drops both parents', async () => {
    await setup({ gap: true });
    const { rally } = await mergeRallies([rallyA, rallyB], userId);
    expect(rally.startMs).toBe(0);
    expect(rally.endMs).toBe(10000);
    expect(await prisma.playerTrack.findUnique({ where: { rallyId: rally.id } })).toBeNull();
  });

  it('rejects when either input is locked', async () => {
    await setup({ lockB: true });
    await expect(mergeRallies([rallyA, rallyB], userId)).rejects.toBeInstanceOf(LockedRallyError);
  });

  it('rejects overlapping rallies', async () => {
    await setup();
    // force overlap by resetting rallyB.startMs to 3000
    await prisma.rally.update({ where: { id: rallyB }, data: { startMs: 3000 } });
    await expect(mergeRallies([rallyA, rallyB], userId)).rejects.toBeInstanceOf(RalliesOverlapError);
  });
});
