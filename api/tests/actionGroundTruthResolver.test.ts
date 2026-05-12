import { describe, it, expect } from 'vitest';
import { resolveGtRow } from '../src/services/actionGroundTruthResolver';

type Pos = { trackId: number; bbox: { x1: number; y1: number; x2: number; y2: number } };

const at = (trackId: number, x1: number, y1: number, x2: number, y2: number): Pos => ({
  trackId, bbox: { x1, y1, x2, y2 },
});

describe('resolveGtRow', () => {
  it('returns SNAPSHOT_EXACT when bbox is null but snapshotTrackId is present at frame', () => {
    const row = { snapshotBboxX1: null, snapshotBboxY1: null, snapshotBboxX2: null, snapshotBboxY2: null, snapshotTrackId: 7 };
    const positions = [at(7, 0.1, 0.1, 0.2, 0.3)];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: 7, resolvedSource: 'SNAPSHOT_EXACT' });
  });

  it('returns UNRESOLVED when bbox null and trackId absent from frame', () => {
    const row = { snapshotBboxX1: null, snapshotBboxY1: null, snapshotBboxX2: null, snapshotBboxY2: null, snapshotTrackId: 9 };
    const positions = [at(7, 0.1, 0.1, 0.2, 0.3)];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('returns IOU_MATCH when bbox overlaps >= 0.5 with a candidate', () => {
    const row = { snapshotBboxX1: 0.1, snapshotBboxY1: 0.1, snapshotBboxX2: 0.2, snapshotBboxY2: 0.3, snapshotTrackId: 99 };
    const positions = [at(7, 0.11, 0.11, 0.21, 0.31)];
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('IOU_MATCH');
    expect(out.resolvedTrackId).toBe(7);
  });

  it('returns NEAREST_CENTER when IoU is low but center within 0.10', () => {
    const row = { snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.12, snapshotBboxY2: 0.14, snapshotTrackId: 99 };
    const positions = [at(7, 0.13, 0.13, 0.15, 0.17)];
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('NEAREST_CENTER');
    expect(out.resolvedTrackId).toBe(7);
  });

  it('returns UNRESOLVED when no candidate meets either threshold', () => {
    const row = { snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.20, snapshotBboxY2: 0.30, snapshotTrackId: 99 };
    const positions = [at(7, 0.80, 0.80, 0.90, 0.95)];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('returns UNRESOLVED when positions are empty and bbox is null', () => {
    const row = { snapshotBboxX1: null, snapshotBboxY1: null, snapshotBboxX2: null, snapshotBboxY2: null, snapshotTrackId: 7 };
    expect(resolveGtRow(row, [])).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('returns UNRESOLVED when positions are empty and bbox is present', () => {
    const row = { snapshotBboxX1: 0.1, snapshotBboxY1: 0.1, snapshotBboxX2: 0.2, snapshotBboxY2: 0.3, snapshotTrackId: 7 };
    expect(resolveGtRow(row, [])).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });
});
