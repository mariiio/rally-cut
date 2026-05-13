/**
 * Integration tests for the court auto-calibration write helpers.
 *
 * Two semantics live in courtCalibration.ts:
 *
 *   - refreshCourtAutoCalibration (preflight)  — save when confident, clear when not
 *   - backfillCourtCalibration    (tracker)    — fill only when missing
 *
 * The unit-level `areCornersReasonable` checks live alongside the merge
 * tests in qualityService.test.ts; this file exercises the DB-write paths.
 */
import 'dotenv/config';
import { afterAll, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import {
  backfillCourtCalibration,
  refreshCourtAutoCalibration,
  type CourtDetection,
} from '../src/services/courtCalibration';

const videoId = 'cc700000-0000-0000-0000-000000000001';
const userId  = 'cc700000-0000-0000-0000-000000000002';

const goodDetection: CourtDetection = {
  corners: [
    { x: 0.2, y: 0.8 },
    { x: 0.8, y: 0.8 },
    { x: 0.7, y: 0.4 },
    { x: 0.3, y: 0.4 },
  ],
  confidence: 0.87,
};

const lowConfidenceDetection: CourtDetection = {
  corners: goodDetection.corners,
  confidence: 0.55,
};

const offscreenDetection: CourtDetection = {
  corners: [
    { x: -0.5, y: 0.8 }, // way off
    { x: 0.8, y: 0.8 },
    { x: 0.7, y: 0.4 },
    { x: 0.3, y: 0.4 },
  ],
  confidence: 0.9,
};

async function teardown() {
  await prisma.video.deleteMany({ where: { id: videoId } });
  await prisma.user.deleteMany({ where: { id: userId } });
}

async function seedVideo(opts: {
  courtCalibrationJson?: object;
  courtCalibrationSource?: 'auto' | 'manual';
} = {}) {
  await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
  await prisma.video.create({
    data: {
      id: videoId,
      name: 'cc-test',
      filename: 'cc.mp4',
      s3Key: 'test/cc.mp4',
      contentHash: 'cc-hash',
      userId,
      ...(opts.courtCalibrationJson !== undefined && {
        courtCalibrationJson: opts.courtCalibrationJson as never,
      }),
      ...(opts.courtCalibrationSource !== undefined && {
        courtCalibrationSource: opts.courtCalibrationSource,
      }),
    },
  });
}

beforeEach(teardown);
afterAll(teardown);

describe('refreshCourtAutoCalibration (preflight semantics)', () => {
  it('saves a confident detection with source=auto when calibration is missing', async () => {
    await seedVideo();
    await refreshCourtAutoCalibration(videoId, goodDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationSource).toBe('auto');
    expect(v?.courtCalibrationJson).toEqual(goodDetection.corners);
  });

  it('overwrites a prior auto calibration with a fresh confident detection', async () => {
    await seedVideo({
      courtCalibrationJson: [{ x: 0.1, y: 0.9 }, { x: 0.9, y: 0.9 }, { x: 0.85, y: 0.4 }, { x: 0.15, y: 0.4 }],
      courtCalibrationSource: 'auto',
    });
    await refreshCourtAutoCalibration(videoId, goodDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toEqual(goodDetection.corners);
    expect(v?.courtCalibrationSource).toBe('auto');
  });

  it('never overwrites a manual calibration even with a confident detection', async () => {
    const manualCorners = [
      { x: 0.11, y: 0.91 },
      { x: 0.91, y: 0.91 },
      { x: 0.81, y: 0.41 },
      { x: 0.19, y: 0.41 },
    ];
    await seedVideo({ courtCalibrationJson: manualCorners, courtCalibrationSource: 'manual' });
    await refreshCourtAutoCalibration(videoId, goodDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toEqual(manualCorners);
    expect(v?.courtCalibrationSource).toBe('manual');
  });

  it('clears stale auto-calibration when a fresh detection drops below confidence', async () => {
    await seedVideo({
      courtCalibrationJson: goodDetection.corners,
      courtCalibrationSource: 'auto',
    });
    await refreshCourtAutoCalibration(videoId, lowConfidenceDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toBeNull();
    expect(v?.courtCalibrationSource).toBeNull();
  });

  it('clears stale auto-calibration when corners go off-screen', async () => {
    await seedVideo({
      courtCalibrationJson: goodDetection.corners,
      courtCalibrationSource: 'auto',
    });
    await refreshCourtAutoCalibration(videoId, offscreenDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toBeNull();
  });

  it('leaves manual calibration alone when detection is below the bar', async () => {
    const manualCorners = goodDetection.corners;
    await seedVideo({ courtCalibrationJson: manualCorners, courtCalibrationSource: 'manual' });
    await refreshCourtAutoCalibration(videoId, null);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toEqual(manualCorners);
    expect(v?.courtCalibrationSource).toBe('manual');
  });
});

describe('backfillCourtCalibration (tracker fill-only semantics)', () => {
  it('fills in a confident detection when calibration is missing', async () => {
    await seedVideo();
    await backfillCourtCalibration(videoId, goodDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationSource).toBe('auto');
    expect(v?.courtCalibrationJson).toEqual(goodDetection.corners);
  });

  it('NEVER overwrites an existing auto calibration', async () => {
    const existingCorners = [
      { x: 0.21, y: 0.81 },
      { x: 0.81, y: 0.81 },
      { x: 0.71, y: 0.41 },
      { x: 0.31, y: 0.41 },
    ];
    await seedVideo({ courtCalibrationJson: existingCorners, courtCalibrationSource: 'auto' });
    await backfillCourtCalibration(videoId, goodDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toEqual(existingCorners);
    expect(v?.courtCalibrationSource).toBe('auto');
  });

  it('NEVER overwrites a manual calibration', async () => {
    const manualCorners = [
      { x: 0.22, y: 0.82 },
      { x: 0.82, y: 0.82 },
      { x: 0.72, y: 0.42 },
      { x: 0.32, y: 0.42 },
    ];
    await seedVideo({ courtCalibrationJson: manualCorners, courtCalibrationSource: 'manual' });
    await backfillCourtCalibration(videoId, goodDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toEqual(manualCorners);
    expect(v?.courtCalibrationSource).toBe('manual');
  });

  it('does NOT save a low-confidence detection', async () => {
    await seedVideo();
    await backfillCourtCalibration(videoId, lowConfidenceDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toBeNull();
    expect(v?.courtCalibrationSource).toBeNull();
  });

  it('does NOT save an off-screen detection', async () => {
    await seedVideo();
    await backfillCourtCalibration(videoId, offscreenDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toBeNull();
  });

  it('NEVER clears an existing calibration (unlike refresh)', async () => {
    await seedVideo({
      courtCalibrationJson: goodDetection.corners,
      courtCalibrationSource: 'auto',
    });
    await backfillCourtCalibration(videoId, lowConfidenceDetection);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toEqual(goodDetection.corners);
    expect(v?.courtCalibrationSource).toBe('auto');
  });

  it('is a no-op when detection is null', async () => {
    await seedVideo();
    await backfillCourtCalibration(videoId, null);

    const v = await prisma.video.findUnique({ where: { id: videoId } });
    expect(v?.courtCalibrationJson).toBeNull();
  });
});
