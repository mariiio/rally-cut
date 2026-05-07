/**
 * Resolve a raw BoT-SORT track id to its canonical display pid (1-4).
 *
 * Source priority:
 *   1. ``appliedFullMapping[trackId]`` — Hungarian output from match-players.
 *      Pre-remap anchor for resolving action-GT `trackId` to a display pid.
 *   2. ``playerNumberMap.get(trackId)`` — local sort-order over visible
 *      tracks. Last-resort display.
 *
 * Returns ``null`` when nothing resolves; callers decide how to render.
 */
export function resolveCanonicalPid(
  trackId: number,
  appliedFullMapping: Record<string, number> | undefined,
  playerNumberMap: Map<number, number> | undefined,
): number | null {
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
  appliedFullMapping: Record<string, number> | undefined,
  playerNumberMap: Map<number, number> | undefined,
): number | null {
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
