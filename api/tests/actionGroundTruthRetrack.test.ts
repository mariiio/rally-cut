import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { saveTrackingResult, type PlayerTrackerOutput } from '../src/services/playerTrackingService';
import { saveActionGroundTruth } from '../src/services/actionGroundTruthService';

const videoId = 'gt500000-0000-0000-0000-000000000001';
const userId  = 'gt500000-0000-0000-0000-000000000002';
const rallyId = 'gt500000-0000-0000-0000-000000000010';

describe('saveTrackingResult re-resolves action GT', () => {
  beforeEach(async () => {
    await prisma.rallyActionGroundTruth.deleteMany({ where: { rallyId } });
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });

    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'gt-retrack-test',
        filename: 'gt.mp4',
        s3Key: 'test/gt.mp4',
        contentHash: 'gt-hash',
        userId,
      },
    });
    await prisma.rally.create({
      data: { id: rallyId, videoId, startMs: 0, endMs: 3333, order: 0 },
    });
    await prisma.playerTrack.create({
      data: {
        rallyId,
        status: 'COMPLETED',
        frameCount: 100,
        fps: 30,
        primaryTrackIds: [7] as unknown as object,
        rawPositionsJson: [
          { frameNumber: 50, trackId: 7, x: 0.1, y: 0.1, width: 0.1, height: 0.2, confidence: 0.9 },
        ] as unknown as object,
        positionsJson: [
          { frameNumber: 50, trackId: 7, x: 0.1, y: 0.1, width: 0.1, height: 0.2, confidence: 0.9 },
        ] as unknown as object,
        ballPositionsJson: [] as unknown as object,
        actionsJson: { teamAssignments: { '7': 'A' } } as unknown as object,
      },
    });

    // Seed one GT label that snapshots trackId 7's bbox.
    await saveActionGroundTruth(rallyId, userId, [{ frame: 50, action: 'serve', trackId: 7 }]);
  });

  afterEach(async () => {
    await prisma.rallyActionGroundTruth.deleteMany({ where: { rallyId } });
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
  });

  it('re-resolves to a new trackId via IoU match after retrack', async () => {
    const result: PlayerTrackerOutput = {
      frameCount: 100, fps: 30, detectionRate: 0.5, avgConfidence: 0.9, avgPlayerCount: 1,
      uniqueTrackCount: 1, courtSplitY: 0.5,
      primaryTrackIds: [11],
      positions: [],
      rawPositions: [
        // trackId 11 at frame 50 with a bbox that heavily overlaps trackId 7's snapshot
        // snapshot was x=0.1 y=0.1 w=0.1 h=0.2 → bbox (0.1,0.1,0.2,0.3)
        // new position: x=0.11 y=0.11 w=0.10 h=0.20 → bbox (0.11,0.11,0.21,0.31) — high IoU
        { frameNumber: 50, trackId: 11, x: 0.11, y: 0.11, width: 0.10, height: 0.20, confidence: 0.9 },
      ] as unknown as PlayerTrackerOutput['rawPositions'],
      ballPositions: [],
      contacts: { contacts: [] } as unknown as PlayerTrackerOutput['contacts'],
      actions: { actions: [], teamAssignments: {} } as unknown as PlayerTrackerOutput['actions'],
      qualityReport: { issues: [] } as unknown as PlayerTrackerOutput['qualityReport'],
    };
    await saveTrackingResult(rallyId, videoId, result, 0);

    const row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId } });
    expect(row.resolvedTrackId).toBe(11);
    expect(row.resolvedSource).toBe('IOU_MATCH');
    expect(row.snapshotTrackId).toBe(7);
  });

  it('lands UNRESOLVED when player is no longer tracked at the frame', async () => {
    const result: PlayerTrackerOutput = {
      frameCount: 100, fps: 30, detectionRate: 0.5, avgConfidence: 0.9, avgPlayerCount: 1,
      uniqueTrackCount: 1, courtSplitY: 0.5,
      primaryTrackIds: [11],
      positions: [],
      rawPositions: [
        // trackId 11 at frame 50 but far away from trackId 7's snapshot bbox
        { frameNumber: 50, trackId: 11, x: 0.80, y: 0.80, width: 0.10, height: 0.15, confidence: 0.9 },
      ] as unknown as PlayerTrackerOutput['rawPositions'],
      ballPositions: [],
      contacts: { contacts: [] } as unknown as PlayerTrackerOutput['contacts'],
      actions: { actions: [], teamAssignments: {} } as unknown as PlayerTrackerOutput['actions'],
      qualityReport: { issues: [] } as unknown as PlayerTrackerOutput['qualityReport'],
    };
    await saveTrackingResult(rallyId, videoId, result, 0);

    const row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId } });
    expect(row.resolvedTrackId).toBeNull();
    expect(row.resolvedSource).toBe('UNRESOLVED');
  });
});
