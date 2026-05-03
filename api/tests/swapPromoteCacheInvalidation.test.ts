/**
 * Integration test: swapPlayerTracks and promoteRawTrack invalidate the
 * same matcher caches as saveTrackingResult. These two endpoints
 * (`POST /v1/rallies/:id/player-track/{swap,promote}`) modify
 * positions_json with per-frame track-ID changes. Without invalidation,
 * the next match-players + remap-track-ids cycle reads stale snapshots
 * and silently undoes the user's manual edit (Pattern E re-emergence).
 *
 * Each test:
 *  - Pre-seeds the rally with stale matcher caches.
 *  - Runs swap or promote.
 *  - Asserts the caches for THIS rally were stripped, OTHER rallies' caches
 *    were not, and Video.canonicalPidMapJson was nulled.
 */
import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { promoteRawTrack, swapPlayerTracks } from '../src/services/playerTrackingService';

const videoId  = 'sp700000-0000-0000-0000-000000000001';
const userId   = 'sp700000-0000-0000-0000-000000000002';
const rallyId  = 'sp700000-0000-0000-0000-000000000010';
const otherRid = 'sp700000-0000-0000-0000-000000000011';

const stalePositions = [
  { frameNumber: 0, trackId: 1, x: 0.1, y: 0.2, width: 50, height: 100, confidence: 0.9 },
  { frameNumber: 0, trackId: 2, x: 0.3, y: 0.4, width: 50, height: 100, confidence: 0.9 },
  { frameNumber: 0, trackId: 3, x: 0.5, y: 0.6, width: 50, height: 100, confidence: 0.9 },
  { frameNumber: 1, trackId: 1, x: 0.1, y: 0.2, width: 50, height: 100, confidence: 0.9 },
  { frameNumber: 1, trackId: 2, x: 0.3, y: 0.4, width: 50, height: 100, confidence: 0.9 },
  { frameNumber: 1, trackId: 3, x: 0.5, y: 0.6, width: 50, height: 100, confidence: 0.9 },
];

const staleRawPositions = [
  ...stalePositions,
  { frameNumber: 0, trackId: 4, x: 0.7, y: 0.8, width: 50, height: 100, confidence: 0.9 },
  { frameNumber: 1, trackId: 4, x: 0.7, y: 0.8, width: 50, height: 100, confidence: 0.9 },
];

async function seed() {
  await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });

  await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
  await prisma.video.create({
    data: {
      id: videoId,
      name: 'sp-test',
      filename: 'sp.mp4',
      s3Key: 'test/sp.mp4',
      contentHash: 'sp-hash',
      userId,
      matchAnalysisJson: {
        rallies: [
          {
            rallyId,
            trackToPlayer: { '1': 1, '2': 2, '3': 3 },
            appliedFullMapping: { '1': 2, '2': 1, '3': 3 },
            remapApplied: true,
            assignmentAnchor: {
              trackStatsHash: 'stale-hash',
              assignment: { '1': 2, '2': 1, '3': 3 },
              confidence: 0.8,
              matcherVersion: 'v3',
            },
          },
          {
            rallyId: otherRid,
            trackToPlayer: { '1': 1, '2': 2, '3': 3, '4': 4 },
            appliedFullMapping: { '1': 1, '2': 2, '3': 3, '4': 4 },
            remapApplied: true,
            assignmentAnchor: {
              trackStatsHash: 'other-hash',
              assignment: { '1': 1, '2': 2, '3': 3, '4': 4 },
              confidence: 0.9,
              matcherVersion: 'v3',
            },
          },
        ],
      },
      canonicalPidMapJson: {
        version: 1,
        sourceRefCropsSha: 'stale-sha',
        rallies: { [rallyId]: { '1': 2, '2': 1, '3': 3 } },
      },
    },
  });
  await prisma.rally.create({
    data: { id: rallyId, videoId, startMs: 0, endMs: 10000, order: 0 },
  });
  await prisma.rally.create({
    data: { id: otherRid, videoId, startMs: 11000, endMs: 20000, order: 1 },
  });
  await prisma.playerTrack.create({
    data: {
      rallyId,
      status: 'COMPLETED',
      positionsJson: stalePositions as unknown as object[],
      rawPositionsJson: staleRawPositions as unknown as object[],
      primaryTrackIds: [1, 2, 3],
      preRemapStateJson: { positions: stalePositions, primaryTrackIds: [1, 2, 3] },
    },
  });
}

async function teardown() {
  await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
  await prisma.rally.deleteMany({ where: { videoId } });
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
}

async function assertCachesInvalidated() {
  const pt = await prisma.playerTrack.findUnique({ where: { rallyId } });
  expect(pt?.preRemapStateJson).toBeNull();

  const v = await prisma.video.findUnique({ where: { id: videoId } });
  const ma = v?.matchAnalysisJson as { rallies: Array<Record<string, unknown>> };
  const thisEntry  = ma.rallies.find(r => r.rallyId === rallyId)!;
  const otherEntry = ma.rallies.find(r => r.rallyId === otherRid)!;

  expect(thisEntry.appliedFullMapping).toBeUndefined();
  expect(thisEntry.remapApplied).toBeUndefined();
  expect(thisEntry.assignmentAnchor).toBeUndefined();

  // Other rally's caches must remain.
  expect(otherEntry.appliedFullMapping).toEqual({ '1': 1, '2': 2, '3': 3, '4': 4 });
  expect(otherEntry.remapApplied).toBe(true);
  expect(otherEntry.assignmentAnchor).toBeTruthy();

  expect(v?.canonicalPidMapJson).toBeNull();
}

describe('swapPlayerTracks — Pattern E cache invalidation', () => {
  beforeEach(seed);
  afterEach(teardown);

  it('invalidates matcher caches for the rally after swap', async () => {
    await swapPlayerTracks(rallyId, userId, 1, 2, 1);
    await assertCachesInvalidated();

    // Sanity check: positions were actually swapped.
    const pt = await prisma.playerTrack.findUnique({ where: { rallyId } });
    const positions = pt?.positionsJson as Array<{ frameNumber: number; trackId: number }>;
    const f1 = positions.filter(p => p.frameNumber === 1).map(p => p.trackId).sort();
    expect(f1).toEqual([1, 2, 3]);  // ID set unchanged, but per-frame mapping swapped.
  });
});

describe('promoteRawTrack — Pattern E cache invalidation', () => {
  beforeEach(seed);
  afterEach(teardown);

  it('invalidates matcher caches for the rally after promote', async () => {
    await promoteRawTrack(rallyId, userId, 1, 4, 1);
    await assertCachesInvalidated();

    // Sanity check: track 1 frames >=1 came from raw track 4.
    const pt = await prisma.playerTrack.findUnique({ where: { rallyId } });
    const positions = pt?.positionsJson as Array<{ frameNumber: number; trackId: number; x: number }>;
    const f1Track1 = positions.find(p => p.frameNumber === 1 && p.trackId === 1);
    expect(f1Track1?.x).toBeCloseTo(0.7);  // raw track 4's x.
  });
});
