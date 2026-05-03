/**
 * Integration test: saveTrackingResult clears stale matcher caches.
 *
 * Track IDs change every time BoT-SORT runs, so any matcher state keyed
 * on the old track IDs is stale after retrack. saveTrackingResult must
 * invalidate that state atomically with the new tracking write, otherwise
 * the next match-players + remap-track-ids cycle can re-introduce the
 * "Pattern E" corruption shape (see analysis memory entry
 * `pattern_e_corruption_2026_05_03.md`).
 *
 * Specifically, on every save we must clear:
 *   - PlayerTrack.preRemapStateJson         (stale snapshot)
 *   - Video.matchAnalysisJson.rallies[].appliedFullMapping for THIS rally
 *   - Video.matchAnalysisJson.rallies[].remapApplied         for THIS rally
 *   - Video.matchAnalysisJson.rallies[].assignmentAnchor     for THIS rally
 *   - Video.canonicalPidMapJson                              (video-wide)
 *
 * Other rallies' entries on Video.matchAnalysisJson must be left untouched.
 */
import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { saveTrackingResult, type PlayerTrackerOutput } from '../src/services/playerTrackingService';

const videoId   = 'st700000-0000-0000-0000-000000000001';
const userId    = 'st700000-0000-0000-0000-000000000002';
const rallyId   = 'st700000-0000-0000-0000-000000000010';
const otherRid  = 'st700000-0000-0000-0000-000000000011';

const stubResult: PlayerTrackerOutput = {
  frameCount: 100,
  fps: 30,
  detectionRate: 1.0,
  avgConfidence: 0.9,
  avgPlayerCount: 4,
  uniqueTrackCount: 4,
  courtSplitY: 0.5,
  primaryTrackIds: [1, 2, 3, 5],
  positions: [],
  rawPositions: [],
  ballPositions: [],
  contacts: { contacts: [] },
  actions: { actions: [], teamAssignments: {} },
  qualityReport: { issues: [] },
};

describe('saveTrackingResult — Pattern E cache invalidation', () => {
  beforeEach(async () => {
    // Teardown first for isolation
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });

    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'st-test',
        filename: 'st.mp4',
        s3Key: 'test/st.mp4',
        contentHash: 'st-hash',
        userId,
        // Pre-existing match analysis with stale caches for both rallies.
        matchAnalysisJson: {
          rallies: [
            {
              rallyId,
              trackToPlayer: { '1': 2, '2': 1, '3': 4, '5': 3 },
              appliedFullMapping: { '1': 2, '2': 1, '3': 4, '5': 3 },
              remapApplied: true,
              assignmentAnchor: {
                trackStatsHash: 'stale-hash',
                assignment: { '1': 2, '2': 1, '3': 4, '5': 3 },
                confidence: 0.8,
                matcherVersion: 'v3',
              },
            },
            {
              rallyId: otherRid,
              trackToPlayer: { '1': 1, '2': 2, '3': 3, '5': 4 },
              appliedFullMapping: { '1': 1, '2': 2, '3': 3, '5': 4 },
              remapApplied: true,
              assignmentAnchor: {
                trackStatsHash: 'other-hash',
                assignment: { '1': 1, '2': 2, '3': 3, '5': 4 },
                confidence: 0.9,
                matcherVersion: 'v3',
              },
            },
          ],
        },
        canonicalPidMapJson: {
          version: 1,
          sourceRefCropsSha: 'stale-sha',
          rallies: { [rallyId]: { '1': 2, '2': 1, '3': 4, '5': 3 } },
        },
      },
    });
    await prisma.rally.create({
      data: { id: rallyId, videoId, startMs: 0, endMs: 10000, order: 0 },
    });
    await prisma.rally.create({
      data: { id: otherRid, videoId, startMs: 11000, endMs: 20000, order: 1 },
    });
    // Pre-seed a PlayerTrack with a stale preRemapStateJson snapshot.
    await prisma.playerTrack.create({
      data: {
        rallyId,
        status: 'COMPLETED',
        needsRetrack: true,
        preRemapStateJson: { positions: [{ trackId: 5 }], primaryTrackIds: [1, 2, 3, 5] },
      },
    });
  });

  afterEach(async () => {
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
  });

  it('clears preRemapStateJson on the affected PlayerTrack', async () => {
    await saveTrackingResult(rallyId, videoId, stubResult, 100);

    const pt = await prisma.playerTrack.findUnique({ where: { rallyId } });
    expect(pt).toBeTruthy();
    expect(pt?.preRemapStateJson).toBeNull();
    expect(pt?.needsRetrack).toBe(false);
  });

  it('strips appliedFullMapping/remapApplied/assignmentAnchor for THIS rally only', async () => {
    await saveTrackingResult(rallyId, videoId, stubResult, 100);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    const ma = v?.matchAnalysisJson as { rallies: Array<Record<string, unknown>> };
    const thisEntry  = ma.rallies.find(r => r.rallyId === rallyId)!;
    const otherEntry = ma.rallies.find(r => r.rallyId === otherRid)!;

    expect(thisEntry.appliedFullMapping).toBeUndefined();
    expect(thisEntry.remapApplied).toBeUndefined();
    expect(thisEntry.assignmentAnchor).toBeUndefined();
    // trackToPlayer is not stripped — that's match-players' output, not a stale cache.
    expect(thisEntry.trackToPlayer).toEqual({ '1': 2, '2': 1, '3': 4, '5': 3 });

    // Other rally's caches must NOT be touched.
    expect(otherEntry.appliedFullMapping).toEqual({ '1': 1, '2': 2, '3': 3, '5': 4 });
    expect(otherEntry.remapApplied).toBe(true);
    expect(otherEntry.assignmentAnchor).toBeTruthy();
  });

  it('NULLs canonicalPidMapJson on the video', async () => {
    await saveTrackingResult(rallyId, videoId, stubResult, 100);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.canonicalPidMapJson).toBeNull();
  });
});
