import { describe, expect, it } from 'vitest';
import { slicePlayerTrack, concatPlayerTracks } from '../src/services/rallySlicing.js';

const basePt = {
  id: 'pt-1', rallyId: 'r-parent', status: 'COMPLETED',
  fps: 30, frameCount: 300, detectionRate: 1.0, avgConfidence: 0.9,
  avgPlayerCount: 4, uniqueTrackCount: 6, courtSplitY: 0.5,
  processingTimeMs: 1000, modelVersion: 'v1', needsRetrack: false, primaryTrackIds: null,
  positionsJson: [
    { frameNumber: 10, trackId: 1, x: 0.1, y: 0.1 },
    { frameNumber: 150, trackId: 1, x: 0.5, y: 0.5 },
    { frameNumber: 250, trackId: 1, x: 0.9, y: 0.9 },
  ],
  rawPositionsJson: [],
  ballPositionsJson: [{ frameNumber: 20 }, { frameNumber: 260 }],
  contactsJson: [{ frame: 50, playerTrackId: 1 }, { frame: 220, playerTrackId: 2 }],
  actionsJson: [{ frame: 55 }, { frame: 225 }],
  groundTruthJson: null,
  actionGroundTruthJson: null,
  qualityReportJson: null,
};

describe('slicePlayerTrack', () => {
  it('partitions by frame and shifts back-half indices', () => {
    const { first, second } = slicePlayerTrack(basePt as any, 100, 200);
    expect(first.positionsJson).toEqual([{ frameNumber: 10, trackId: 1, x: 0.1, y: 0.1 }]);
    expect(second.positionsJson.map((p: any) => p.frameNumber)).toEqual([50]); // 250 - 200
    expect(first.contactsJson.map((c: any) => c.frame)).toEqual([50]);
    expect(second.contactsJson.map((c: any) => c.frame)).toEqual([20]); // 220 - 200
    expect(first.actionsJson.map((a: any) => a.frame)).toEqual([55]);
    expect(second.actionsJson.map((a: any) => a.frame)).toEqual([25]); // 225 - 200
    expect(first.ballPositionsJson.map((b: any) => b.frameNumber)).toEqual([20]);
    expect(second.ballPositionsJson.map((b: any) => b.frameNumber)).toEqual([60]);
  });

  it('discards middle segment between firstEndFrame and secondStartFrame', () => {
    const pt = { ...basePt, positionsJson: [{ frameNumber: 150, trackId: 1, x: 0, y: 0 }] };
    const { first, second } = slicePlayerTrack(pt as any, 100, 200);
    expect(first.positionsJson).toHaveLength(0);
    expect(second.positionsJson).toHaveLength(0);
  });

  it('recomputes frameCount per child', () => {
    const { first, second } = slicePlayerTrack(basePt as any, 100, 200);
    expect(first.frameCount).toBe(100);
    expect(second.frameCount).toBe(100); // 300 - 200
  });

  it('split-at-frame-0 degenerate case produces empty first child', () => {
    const { first, second } = slicePlayerTrack(basePt as any, 0, 0);
    expect(first.frameCount).toBe(0);
    expect(first.positionsJson).toHaveLength(0);
    expect(second.frameCount).toBe(300);
  });

  it('handles null GT arrays gracefully', () => {
    const { first, second } = slicePlayerTrack(basePt as any, 100, 200);
    expect(first.groundTruthJson).toBeNull();
    expect(second.groundTruthJson).toBeNull();
  });

  it('propagates primaryTrackIds verbatim to both children', () => {
    const pt = { ...basePt, primaryTrackIds: [1, 2] };
    const { first, second } = slicePlayerTrack(pt as any, 100, 200);
    expect(first.primaryTrackIds).toEqual([1, 2]);
    expect(second.primaryTrackIds).toEqual([1, 2]);
  });
});

describe('concatPlayerTracks', () => {
  const a = {
    fps: 30, frameCount: 100, courtSplitY: 0.5, processingTimeMs: 500, modelVersion: 'v1',
    status: 'COMPLETED', needsRetrack: false, primaryTrackIds: null, qualityReportJson: null,
    positionsJson: [{ frameNumber: 10, trackId: 1, confidence: 0.9 }],
    rawPositionsJson: [], ballPositionsJson: [{ frameNumber: 20 }],
    contactsJson: [{ frame: 50, playerTrackId: 1 }],
    actionsJson: [], groundTruthJson: null, actionGroundTruthJson: null,
  };
  const b = {
    fps: 30, frameCount: 150, courtSplitY: 0.5, processingTimeMs: 700, modelVersion: 'v1',
    status: 'COMPLETED', needsRetrack: false, primaryTrackIds: null, qualityReportJson: null,
    positionsJson: [{ frameNumber: 5, trackId: 2, confidence: 0.85 }],
    rawPositionsJson: [], ballPositionsJson: [{ frameNumber: 10 }],
    contactsJson: [{ frame: 30, playerTrackId: 2 }],
    actionsJson: [], groundTruthJson: null, actionGroundTruthJson: null,
  };

  it('shifts b frames up by a.frameCount', () => {
    const merged = concatPlayerTracks(a as any, b as any);
    expect(merged.frameCount).toBe(250);
    expect(merged.positionsJson.map((p: any) => p.frameNumber).sort((x: number, y: number) => x - y)).toEqual([10, 105]);
    expect(merged.contactsJson.map((c: any) => c.frame).sort((x: number, y: number) => x - y)).toEqual([50, 130]);
    expect(merged.ballPositionsJson.map((p: any) => p.frameNumber).sort((x: number, y: number) => x - y)).toEqual([20, 110]);
  });

  it('unions trackIds', () => {
    const merged = concatPlayerTracks(a as any, b as any);
    expect(merged.uniqueTrackCount).toBe(2);
  });

  it('merges primaryTrackIds: union of both arrays; falls back to non-null side', () => {
    const aWithIds = { ...a, primaryTrackIds: [1] };
    const bWithIds = { ...b, primaryTrackIds: [2, 3] };
    const merged = concatPlayerTracks(aWithIds as any, bWithIds as any);
    expect((merged.primaryTrackIds as number[]).sort()).toEqual([1, 2, 3]);

    const aNull = { ...a, primaryTrackIds: null };
    const bWithIds2 = { ...b, primaryTrackIds: [2] };
    const merged2 = concatPlayerTracks(aNull as any, bWithIds2 as any);
    expect(merged2.primaryTrackIds).toEqual([2]);
  });
});
