import 'dotenv/config';
import { afterEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { splitRally } from '../src/services/rallyService';
import { LockedRallyError, RallyTrackingStateError, SplitBoundsError } from '../src/middleware/errorHandler';

const userId = '11111111-1111-1111-1111-000000000b11';
const videoId = '22222222-2222-2222-2222-000000000b11';
const rallyId = '33333333-3333-3333-3333-000000000b11';

async function setupRally(opts: { locked?: boolean; trackStatus?: 'COMPLETED' | 'PROCESSING' | 'FAILED' | 'NONE' } = {}) {
  await prisma.playerTrack.deleteMany({ where: { rallyId } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
  await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
  await prisma.video.create({
    data: {
      id: videoId, userId, name: 'b', filename: 'b.mp4', s3Key: 'test/b.mp4', contentHash: 'deadbeef11',
      durationMs: 30000,
      matchAnalysisJson: {
        videoId, numRallies: 1,
        rallies: [{
          rallyId, canonicalLocked: opts.locked === true,
          trackToPlayer: { '1': 1 }, assignmentConfidence: 0.9, serverPlayerId: 1,
        }],
      },
    },
  });
  await prisma.rally.create({
    data: { id: rallyId, videoId, startMs: 0, endMs: 10000, scoreA: 5, scoreB: 3, servingTeam: 'A', notes: 'parent' },
  });
  if (opts.trackStatus && opts.trackStatus !== 'NONE') {
    await prisma.playerTrack.create({
      data: {
        rallyId, status: opts.trackStatus, fps: 30, frameCount: 300,
        detectionRate: 1.0, avgConfidence: 0.9, avgPlayerCount: 4, uniqueTrackCount: 2,
        courtSplitY: 0.5, processingTimeMs: 1000, modelVersion: 'v1', needsRetrack: false,
        positionsJson: [
          { frameNumber: 30, trackId: 1, x: 0, y: 0, confidence: 0.9 },
          { frameNumber: 180, trackId: 1, x: 0, y: 0, confidence: 0.9 },
          { frameNumber: 270, trackId: 1, x: 0, y: 0, confidence: 0.9 },
        ],
        rawPositionsJson: [],
        ballPositionsJson: [],
        contactsJson: [{ frame: 50, playerTrackId: 1 }, { frame: 250, playerTrackId: 1 }],
        actionsJson: [],
      } as any,
    });
  }
}

async function cleanup() {
  await prisma.playerTrack.deleteMany({ where: { rallyId } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
}

describe('splitRally', () => {
  afterEach(cleanup);

  it('slices PlayerTrack data, creates two children, deletes parent', async () => {
    await setupRally({ trackStatus: 'COMPLETED' });
    const { firstRally, secondRally } = await splitRally(rallyId, userId, { firstEndMs: 4000, secondStartMs: 6000 });

    expect(firstRally.endMs).toBe(4000);
    expect(secondRally.startMs).toBe(6000);
    expect(secondRally.endMs).toBe(10000);

    // Parent deleted
    expect(await prisma.rally.findUnique({ where: { id: rallyId } })).toBeNull();

    // Children have sliced player tracks
    const firstPt = await prisma.playerTrack.findUnique({ where: { rallyId: firstRally.id } });
    const secondPt = await prisma.playerTrack.findUnique({ where: { rallyId: secondRally.id } });
    expect(firstPt).not.toBeNull();
    expect(secondPt).not.toBeNull();
    expect((firstPt!.contactsJson as any).map((c: any) => c.frame)).toEqual([50]);
    expect((secondPt!.contactsJson as any).map((c: any) => c.frame)).toEqual([70]); // 250 - 180

    // Inheritance: first child scores default 0, second inherits parent
    expect(firstRally.scoreA).toBe(0);
    expect(firstRally.scoreB).toBe(0);
    expect(secondRally.scoreA).toBe(5);
    expect(secondRally.scoreB).toBe(3);
    expect(firstRally.notes).toBe('parent');
    expect(secondRally.notes).toBe('parent');

    // matchAnalysisJson has both children, parent removed
    const v = await prisma.video.findUnique({ where: { id: videoId } });
    const rallies = (v!.matchAnalysisJson as any).rallies;
    expect(rallies.map((r: any) => r.rallyId).sort()).toEqual([firstRally.id, secondRally.id].sort());
    expect(rallies.every((r: any) => r.canonicalLocked === false)).toBe(true);
    expect(rallies.every((r: any) => JSON.stringify(r.trackToPlayer) === JSON.stringify({ '1': 1 }))).toBe(true);

    // pendingAnalysisEdits has 2 split entries
    expect((v!.pendingAnalysisEditsJson as any).entries.map((e: any) => e.editKind)).toEqual(['split', 'split']);
  });

  it('allows split when PlayerTrack is null (children PENDING-equivalent)', async () => {
    await setupRally({ trackStatus: 'NONE' });
    const { firstRally, secondRally } = await splitRally(rallyId, userId, { firstEndMs: 4000, secondStartMs: 6000 });
    expect(await prisma.playerTrack.findUnique({ where: { rallyId: firstRally.id } })).toBeNull();
    expect(await prisma.playerTrack.findUnique({ where: { rallyId: secondRally.id } })).toBeNull();
  });

  it('rejects split on locked rally', async () => {
    await setupRally({ locked: true, trackStatus: 'COMPLETED' });
    await expect(splitRally(rallyId, userId, { firstEndMs: 4000, secondStartMs: 6000 }))
      .rejects.toBeInstanceOf(LockedRallyError);
    // Parent still present
    expect(await prisma.rally.findUnique({ where: { id: rallyId } })).not.toBeNull();
  });

  it('rejects split on PROCESSING PlayerTrack', async () => {
    await setupRally({ trackStatus: 'PROCESSING' });
    await expect(splitRally(rallyId, userId, { firstEndMs: 4000, secondStartMs: 6000 }))
      .rejects.toBeInstanceOf(RallyTrackingStateError);
  });

  it('rejects invalid bounds (firstEndMs > secondStartMs)', async () => {
    await setupRally({ trackStatus: 'COMPLETED' });
    await expect(splitRally(rallyId, userId, { firstEndMs: 7000, secondStartMs: 6000 }))
      .rejects.toBeInstanceOf(SplitBoundsError);
  });
});
