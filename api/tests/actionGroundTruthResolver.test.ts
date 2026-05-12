import { describe, it, expect } from 'vitest';
import { resolveGtRow } from '../src/services/actionGroundTruthResolver';

type Pos = { trackId: number; bbox: { x1: number; y1: number; x2: number; y2: number }; embedding: Float32Array | null; team: 'A' | 'B' | null };

const at = (
  trackId: number,
  x1: number, y1: number, x2: number, y2: number,
  embedding: Float32Array | null = null,
  team: 'A' | 'B' | null = null,
): Pos => ({
  trackId, bbox: { x1, y1, x2, y2 }, embedding, team,
});

function makeEmbedding(seed: number): Float32Array {
  const e = new Float32Array(128);
  for (let i = 0; i < 128; i++) e[i] = Math.sin(seed * 0.1 + i * 0.01);
  return e;
}

describe('resolveGtRow', () => {
  it('returns SNAPSHOT_EXACT when bbox is null but snapshotTrackId is present at frame', () => {
    const row = { snapshotBboxX1: null, snapshotBboxY1: null, snapshotBboxX2: null, snapshotBboxY2: null, snapshotTrackId: 7, snapshotReidEmbedding: null, snapshotTeam: null };
    const positions = [at(7, 0.1, 0.1, 0.2, 0.3)];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: 7, resolvedSource: 'SNAPSHOT_EXACT' });
  });

  it('returns UNRESOLVED when bbox null and trackId absent from frame', () => {
    const row = { snapshotBboxX1: null, snapshotBboxY1: null, snapshotBboxX2: null, snapshotBboxY2: null, snapshotTrackId: 9, snapshotReidEmbedding: null, snapshotTeam: null };
    const positions = [at(7, 0.1, 0.1, 0.2, 0.3)];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('returns IOU_MATCH when bbox overlaps >= 0.5 with a candidate', () => {
    const row = { snapshotBboxX1: 0.1, snapshotBboxY1: 0.1, snapshotBboxX2: 0.2, snapshotBboxY2: 0.3, snapshotTrackId: 99, snapshotReidEmbedding: null, snapshotTeam: null };
    const positions = [at(7, 0.11, 0.11, 0.21, 0.31)];
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('IOU_MATCH');
    expect(out.resolvedTrackId).toBe(7);
  });

  it('returns NEAREST_CENTER when IoU is low but center within 0.10', () => {
    const row = { snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.12, snapshotBboxY2: 0.14, snapshotTrackId: 99, snapshotReidEmbedding: null, snapshotTeam: null };
    const positions = [at(7, 0.13, 0.13, 0.15, 0.17)];
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('NEAREST_CENTER');
    expect(out.resolvedTrackId).toBe(7);
  });

  it('returns UNRESOLVED when no candidate meets either threshold', () => {
    const row = { snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.20, snapshotBboxY2: 0.30, snapshotTrackId: 99, snapshotReidEmbedding: null, snapshotTeam: null };
    const positions = [at(7, 0.80, 0.80, 0.90, 0.95)];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('returns UNRESOLVED when positions are empty and bbox is null', () => {
    const row = { snapshotBboxX1: null, snapshotBboxY1: null, snapshotBboxX2: null, snapshotBboxY2: null, snapshotTrackId: 7, snapshotReidEmbedding: null, snapshotTeam: null };
    expect(resolveGtRow(row, [])).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('returns UNRESOLVED when positions are empty and bbox is present', () => {
    const row = { snapshotBboxX1: 0.1, snapshotBboxY1: 0.1, snapshotBboxX2: 0.2, snapshotBboxY2: 0.3, snapshotTrackId: 7, snapshotReidEmbedding: null, snapshotTeam: null };
    expect(resolveGtRow(row, [])).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('returns REID_MATCH when IoU fails but embedding cosine >= 0.5', () => {
    const snapshotEmb = makeEmbedding(7);
    const row = {
      snapshotBboxX1: 0.1, snapshotBboxY1: 0.1, snapshotBboxX2: 0.2, snapshotBboxY2: 0.3,
      snapshotTrackId: 99, snapshotReidEmbedding: snapshotEmb, snapshotTeam: null,
    };
    // Candidate has a non-overlapping bbox (IoU=0, center-dist > 0.10) but identical embedding.
    const positions = [at(7, 0.80, 0.80, 0.90, 0.95, snapshotEmb)];
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('REID_MATCH');
    expect(out.resolvedTrackId).toBe(7);
  });

  it('REID tier is skipped when snapshot embedding is null', () => {
    const row = {
      snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.20, snapshotBboxY2: 0.30,
      snapshotTrackId: 99, snapshotReidEmbedding: null, snapshotTeam: null,
    };
    // Candidate has a perfect embedding match but null on the row's side — REID can't fire.
    const positions = [at(7, 0.80, 0.80, 0.90, 0.95, makeEmbedding(7))];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('NEAREST_CENTER rejects wrong-team candidate when snapshotTeam is set', () => {
    // Two candidates near the snapshot bbox; team A is slightly farther but the
    // snapshot is team A. Without the cross-check, geometric proximity would pick
    // the team-B candidate. With the cross-check, picks team A.
    const row = {
      snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.12, snapshotBboxY2: 0.14,
      snapshotTrackId: 99, snapshotReidEmbedding: null, snapshotTeam: 'A' as const,
    };
    const positions = [
      at(7, 0.125, 0.125, 0.145, 0.165, null, 'B'),   // closer to snapshot center
      at(8, 0.135, 0.135, 0.155, 0.175, null, 'A'),   // slightly farther
    ];
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('NEAREST_CENTER');
    expect(out.resolvedTrackId).toBe(8);
  });

  it('NEAREST_CENTER returns UNRESOLVED when no candidate matches the snapshot team', () => {
    const row = {
      snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.12, snapshotBboxY2: 0.14,
      snapshotTrackId: 99, snapshotReidEmbedding: null, snapshotTeam: 'A' as const,
    };
    // Only candidate available is team B → must NOT resolve to it.
    const positions = [at(7, 0.125, 0.125, 0.145, 0.165, null, 'B')];
    expect(resolveGtRow(row, positions)).toEqual({ resolvedTrackId: null, resolvedSource: 'UNRESOLVED' });
  });

  it('NEAREST_CENTER falls back to unfiltered pool when no candidate has a team annotation', () => {
    // Snapshot has a team, but the new tracking didn't supply team annotations
    // (legacy callers, or pre-match-analysis state). Cross-check must not crash —
    // it falls back to the unfiltered geometric match.
    const row = {
      snapshotBboxX1: 0.10, snapshotBboxY1: 0.10, snapshotBboxX2: 0.12, snapshotBboxY2: 0.14,
      snapshotTrackId: 99, snapshotReidEmbedding: null, snapshotTeam: 'A' as const,
    };
    const positions = [at(7, 0.13, 0.13, 0.15, 0.17, null, null)];
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('NEAREST_CENTER');
    expect(out.resolvedTrackId).toBe(7);
  });

  it('IoU tier still wins when both IoU and REID would match', () => {
    const emb = makeEmbedding(7);
    const row = {
      snapshotBboxX1: 0.1, snapshotBboxY1: 0.1, snapshotBboxX2: 0.2, snapshotBboxY2: 0.3,
      snapshotTrackId: 99, snapshotReidEmbedding: emb, snapshotTeam: null,
    };
    // High IoU overlap AND matching embedding → IoU tier wins (cheaper).
    const positions = [at(7, 0.11, 0.11, 0.21, 0.31, emb)];
    const out = resolveGtRow(row, positions);
    expect(out.resolvedSource).toBe('IOU_MATCH');
    expect(out.resolvedTrackId).toBe(7);
  });
});
