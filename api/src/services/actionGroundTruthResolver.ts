export const IOU_THRESHOLD = 0.5;
export const CENTER_DIST_THRESHOLD = 0.10;

export type ResolveSource = 'SNAPSHOT_EXACT' | 'IOU_MATCH' | 'NEAREST_CENTER' | 'MANUAL' | 'UNRESOLVED';

export interface GtRowInput {
  snapshotBboxX1: number | null;
  snapshotBboxY1: number | null;
  snapshotBboxX2: number | null;
  snapshotBboxY2: number | null;
  snapshotTrackId: number | null;
}

export interface Candidate {
  trackId: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
}

export interface ResolveResult {
  resolvedTrackId: number | null;
  resolvedSource: ResolveSource;
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

  const [sx, sy] = center(snapshot);
  const distRanked = positions
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
