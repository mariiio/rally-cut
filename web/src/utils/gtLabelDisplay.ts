import type { ActionGroundTruthLabel, MatchAnalysis } from '@/services/api';

/**
 * Resolve the display pid (1-4) for a GT label.
 *
 * Priority:
 *   1. `trackId` (raw BoT-SORT id) → look up in the rally's
 *      `appliedFullMapping` (raw → canonical pid). If post-remap positions are
 *      what the overlay sees, then `playerNumberMap` collapses to identity
 *      over canonical ids — either path yields the same number.
 *   2. `trackId` present but no mapping → try `playerNumberMap` (local sorted
 *      index over visible tracks) as a best-effort display.
 *   3. Legacy `playerTrackId` → `playerNumberMap`.
 *   4. Nothing resolvable → `null` (caller renders raw value).
 */
export function resolveGtDisplayPid(
  gt: ActionGroundTruthLabel,
  appliedFullMapping: Record<string, number> | undefined,
  playerNumberMap: Map<number, number> | undefined,
): number | null {
  if (gt.trackId !== undefined) {
    const canonical = appliedFullMapping?.[String(gt.trackId)];
    if (canonical !== undefined && Number.isFinite(canonical)) return canonical;
    const localNum = playerNumberMap?.get(gt.trackId);
    if (localNum !== undefined) return localNum;
  }
  if (gt.playerTrackId !== undefined && gt.playerTrackId >= 0) {
    const localNum = playerNumberMap?.get(gt.playerTrackId);
    if (localNum !== undefined) return localNum;
    // Legacy rows stored the canonical pid directly in `playerTrackId`; when
    // no local map covers it we return it as-is and let the caller decide
    // how to render. Non-canonical values (<1 or >4) are intentionally
    // dropped above via the `>= 0` guard plus the caller-side `anchor >= 0`
    // gate so we never render `P-1`.
    return gt.playerTrackId;
  }
  return null;
}

/** Raw anchor id for a GT label (trackId first, playerTrackId as legacy fallback). */
export function gtAnchorId(gt: ActionGroundTruthLabel): number | null {
  if (gt.trackId !== undefined) return gt.trackId;
  if (gt.playerTrackId !== undefined) return gt.playerTrackId;
  return null;
}

/** Look up the current rally entry from a cached match analysis. */
export function rallyMatchEntry(
  analysis: MatchAnalysis | undefined,
  backendRallyId: string | null,
): MatchAnalysis['rallies'][number] | undefined {
  if (!analysis || !backendRallyId) return undefined;
  return analysis.rallies.find((r) => r.rallyId === backendRallyId);
}
