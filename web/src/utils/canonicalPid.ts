import type { MatchAnalysis } from '@/services/api';

/**
 * Per-rally `{ rawTrackId: pid }` slice from `Video.canonicalPidMapJson`.
 * Pre-extracted by callers so this module stays decoupled from the wider
 * MatchAnalysis container shape.
 */
export type CanonicalRallyMap = Record<string, number>;

/**
 * Resolve a raw BoT-SORT track id to its canonical display pid (1-4).
 *
 * Source priority (per docs/superpowers/plans/2026-04-25-ref-crop-canonical-identity.md):
 *   1. ``canonicalRallyMap[trackId]`` — ref-crop sourced. Bit-deterministic
 *      across re-runs; the contract requires editor + stats render the
 *      same pid for the same physical body even after a retrack.
 *   2. ``appliedFullMapping[trackId]`` — legacy Hungarian output. Used
 *      when no canonical map exists for the rally, or for videos without
 *      the full 4-pid ref-crop set.
 *   3. ``playerNumberMap.get(trackId)`` — local sort-order over visible
 *      tracks. Last-resort display.
 *
 * Returns ``null`` when nothing resolves; callers decide how to render.
 */
export function resolveCanonicalPid(
  trackId: number,
  canonicalRallyMap: CanonicalRallyMap | undefined,
  appliedFullMapping: Record<string, number> | undefined,
  playerNumberMap: Map<number, number> | undefined,
): number | null {
  const fromCanonical = canonicalRallyMap?.[String(trackId)];
  if (fromCanonical !== undefined && Number.isFinite(fromCanonical)) {
    return fromCanonical;
  }
  const fromApplied = appliedFullMapping?.[String(trackId)];
  if (fromApplied !== undefined && Number.isFinite(fromApplied)) {
    return fromApplied;
  }
  const fromSortOrder = playerNumberMap?.get(trackId);
  if (fromSortOrder !== undefined) {
    return fromSortOrder;
  }
  return null;
}

/**
 * Inverse of `resolveCanonicalPid` — given a display pid, find the raw
 * track id the editor should store on a GT row. Used by the labeling
 * write path: when the user presses "Player N", the GT must anchor to a
 * raw BoT-SORT id (which is stable across retracks), not the display pid.
 */
export function pidToTrackId(
  displayPid: number,
  canonicalRallyMap: CanonicalRallyMap | undefined,
  appliedFullMapping: Record<string, number> | undefined,
  playerNumberMap: Map<number, number> | undefined,
): number | null {
  if (canonicalRallyMap) {
    for (const [tid, pid] of Object.entries(canonicalRallyMap)) {
      if (pid === displayPid) return Number(tid);
    }
  }
  if (appliedFullMapping) {
    for (const [tid, pid] of Object.entries(appliedFullMapping)) {
      if (pid === displayPid) return Number(tid);
    }
  }
  if (playerNumberMap) {
    for (const [tid, num] of playerNumberMap.entries()) {
      if (num === displayPid) return tid;
    }
  }
  return null;
}

/**
 * Convenience: extract the canonical map slice for a rally from a
 * MatchAnalysis container. Returns ``undefined`` when the video lacks a
 * canonical map or the rally isn't covered.
 */
export function canonicalRallyMapFor(
  matchAnalysis: MatchAnalysis | undefined,
  rallyId: string | null | undefined,
): CanonicalRallyMap | undefined {
  if (!matchAnalysis?.canonicalPidMap || !rallyId) return undefined;
  return matchAnalysis.canonicalPidMap.rallies?.[rallyId];
}
