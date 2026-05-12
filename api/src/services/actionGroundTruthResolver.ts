import type { ResolveSource } from '@prisma/client';
export type { ResolveSource };

export const IOU_THRESHOLD = 0.5;
export const CENTER_DIST_THRESHOLD = 0.10;
export const REID_COSINE_THRESHOLD = 0.5;

export interface GtRowInput {
  snapshotBboxX1: number | null;
  snapshotBboxY1: number | null;
  snapshotBboxX2: number | null;
  snapshotBboxY2: number | null;
  snapshotTrackId: number | null;
  snapshotReidEmbedding: Float32Array | null;  // 128-dim, null when not yet captured
  // Advisory cross-check for the geometric-fallback tier. When non-null, the
  // NEAREST_CENTER tier rejects candidates with a different team. Prevents
  // the "two players clustered near the snapshot bbox" cross-team misattribution.
  // Has no effect on SNAPSHOT_EXACT, IOU_MATCH, REID_MATCH tiers, which the
  // bbox/embedding evidence already distinguishes.
  snapshotTeam: 'A' | 'B' | null;
}

export interface Candidate {
  trackId: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
  embedding: Float32Array | null;  // optional appearance vector from tracker
  team: 'A' | 'B' | null;          // optional team annotation from teamAssignments
}

export interface ResolveResult {
  resolvedTrackId: number | null;
  resolvedSource: ResolveSource;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / Math.sqrt(na * nb);
}

function iou(a: Candidate['bbox'], b: Candidate['bbox']): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

function center(b: Candidate['bbox']): [number, number] {
  return [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2];
}

export function resolveGtRow(row: GtRowInput, positions: Candidate[]): ResolveResult {
  const hasBbox =
    row.snapshotBboxX1 != null && row.snapshotBboxY1 != null &&
    row.snapshotBboxX2 != null && row.snapshotBboxY2 != null;

  if (!hasBbox) {
    if (row.snapshotTrackId != null && positions.some(p => p.trackId === row.snapshotTrackId)) {
      return { resolvedTrackId: row.snapshotTrackId, resolvedSource: 'SNAPSHOT_EXACT' };
    }
    return { resolvedTrackId: null, resolvedSource: 'UNRESOLVED' };
  }

  const snapshot = {
    x1: row.snapshotBboxX1!, y1: row.snapshotBboxY1!,
    x2: row.snapshotBboxX2!, y2: row.snapshotBboxY2!,
  };

  if (positions.length === 0) return { resolvedTrackId: null, resolvedSource: 'UNRESOLVED' };

  const ranked = positions
    .map(p => ({ trackId: p.trackId, iouVal: iou(snapshot, p.bbox) }))
    .sort((a, b) => b.iouVal - a.iouVal);
  if (ranked[0].iouVal >= IOU_THRESHOLD) {
    return { resolvedTrackId: ranked[0].trackId, resolvedSource: 'IOU_MATCH' };
  }

  if (row.snapshotReidEmbedding !== null) {
    const reidRanked = positions
      .filter(p => p.embedding !== null)
      .map(p => ({ trackId: p.trackId, sim: cosineSimilarity(row.snapshotReidEmbedding!, p.embedding!) }))
      .sort((a, b) => b.sim - a.sim);
    if (reidRanked.length > 0 && reidRanked[0].sim >= REID_COSINE_THRESHOLD) {
      return { resolvedTrackId: reidRanked[0].trackId, resolvedSource: 'REID_MATCH' };
    }
  }

  const [sx, sy] = center(snapshot);
  // Team cross-check: when the snapshot has a team annotation AND at least one
  // candidate is annotated, restrict the NEAREST_CENTER candidate pool to the
  // matching team. If no candidate has a team annotation, fall back to the
  // unfiltered pool (preserves backward compat for callers that don't pass
  // teamAssignments). This tier is the most error-prone — the bbox/embedding
  // tiers above already rejected — so it's where cross-team noise leaks in.
  let centerPool = positions;
  if (row.snapshotTeam !== null) {
    const anyAnnotated = positions.some(p => p.team !== null);
    if (anyAnnotated) {
      centerPool = positions.filter(p => p.team === row.snapshotTeam);
    }
  }
  if (centerPool.length === 0) {
    return { resolvedTrackId: null, resolvedSource: 'UNRESOLVED' };
  }
  const distRanked = centerPool
    .map(p => {
      const [cx, cy] = center(p.bbox);
      return { trackId: p.trackId, dist: Math.hypot(cx - sx, cy - sy) };
    })
    .sort((a, b) => a.dist - b.dist);
  if (distRanked[0].dist <= CENTER_DIST_THRESHOLD) {
    return { resolvedTrackId: distRanked[0].trackId, resolvedSource: 'NEAREST_CENTER' };
  }

  return { resolvedTrackId: null, resolvedSource: 'UNRESOLVED' };
}
