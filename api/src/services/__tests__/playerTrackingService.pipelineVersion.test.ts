/**
 * Integration test: saveTrackingResult persists contactsPipelineVersion
 * and actionsPipelineVersion columns.
 */
import 'dotenv/config';
import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import { prisma } from '../../lib/prisma.js';
import { saveTrackingResult } from '../playerTrackingService.js';

const VIDEO_ID = 'pv-test00-0000-0000-0000-000000000001';
const RALLY_ID = 'pv-test00-0000-0000-0000-000000000010';

describe('saveTrackingResult — pipeline version columns', () => {
  beforeAll(async () => {
    // Teardown first for isolation
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId: VIDEO_ID } } });
    await prisma.rally.deleteMany({ where: { videoId: VIDEO_ID } });
    await prisma.video.deleteMany({ where: { id: VIDEO_ID } });

    await prisma.video.create({
      data: {
        id: VIDEO_ID,
        name: 'pv-test',
        filename: 'pv-test.mp4',
        s3Key: `test/pipeline-version-${Date.now()}.mp4`,
        contentHash: `pv-test-${Date.now()}`,
        status: 'UPLOADED',
      } as any,
    });
    await prisma.rally.create({
      data: {
        id: RALLY_ID,
        videoId: VIDEO_ID,
        startMs: 0,
        endMs: 1000,
      } as any,
    });
  });

  afterAll(async () => {
    try {
      await prisma.playerTrack.deleteMany({ where: { rally: { videoId: VIDEO_ID } } });
      await prisma.rally.deleteMany({ where: { videoId: VIDEO_ID } });
      await prisma.video.delete({ where: { id: VIDEO_ID } });
    } catch {
      // ignore — already cleaned
    }
    await prisma.$disconnect();
  });

  it('writes contactsPipelineVersion and actionsPipelineVersion when content is present', async () => {
    await saveTrackingResult(
      RALLY_ID,
      VIDEO_ID,
      {
        frameCount: 100,
        fps: 30,
        detectionRate: 0.9,
        avgConfidence: 0.8,
        avgPlayerCount: 4,
        uniqueTrackCount: 4,
        courtSplitY: 0.5,
        primaryTrackIds: [1, 2, 3, 4],
        positions: [],
        rawPositions: [],
        ballPositions: [],
        contacts: { numContacts: 0, contacts: [] } as any,
        actions: { numContacts: 0, actions: [], teamAssignments: {} } as any,
        contactsPipelineVersion: 'v1',
        actionsPipelineVersion: 'v1',
        qualityReport: undefined,
      } as any,
      1234,
    );

    const row = await prisma.playerTrack.findUnique({ where: { rallyId: RALLY_ID } });
    expect(row?.contactsPipelineVersion).toBe('v1');
    expect(row?.actionsPipelineVersion).toBe('v1');
  });

  it('leaves columns null when contacts/actions are null/undefined', async () => {
    await saveTrackingResult(
      RALLY_ID,
      VIDEO_ID,
      {
        frameCount: 100,
        fps: 30,
        detectionRate: 0.9,
        avgConfidence: 0.8,
        avgPlayerCount: 4,
        uniqueTrackCount: 4,
        courtSplitY: 0.5,
        primaryTrackIds: [1, 2, 3, 4],
        positions: [],
        rawPositions: [],
        ballPositions: [],
        contacts: undefined,
        actions: undefined,
        contactsPipelineVersion: undefined,
        actionsPipelineVersion: undefined,
        qualityReport: undefined,
      } as any,
      1234,
    );

    const row = await prisma.playerTrack.findUnique({ where: { rallyId: RALLY_ID } });
    expect(row?.contactsPipelineVersion).toBeNull();
    expect(row?.actionsPipelineVersion).toBeNull();
  });
});
