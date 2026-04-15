import { describe, expect, it } from 'vitest';
import { slicePlayerTrack } from '../src/services/rallySlicing.js';

const basePt = {
  id: 'pt-1', rallyId: 'r-parent', status: 'COMPLETED',
  fps: 30, frameCount: 300, detectionRate: 1.0, avgConfidence: 0.9,
  avgPlayerCount: 4, uniqueTrackCount: 6, courtSplitY: 0.5,
  processingTimeMs: 1000, modelVersion: 'v1', needsRetrack: false,
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
});
