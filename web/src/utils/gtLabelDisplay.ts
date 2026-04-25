import type { ActionGroundTruthLabel, MatchAnalysis } from '@/services/api';
import { type CanonicalRallyMap, resolveCanonicalPid } from '@/utils/canonicalPid';

/**
 * Resolve the display pid (1-4) for a GT label.
 *
 * Delegates the trackId → pid lookup to ``resolveCanonicalPid`` so the
 * canonicalPidMap (ref-crop-sourced) takes precedence over
 * ``appliedFullMapping`` (legacy Hungarian) which takes precedence over
 * sort-order. Adds the GT-specific fallback to legacy ``playerTrackId``
 * for rows predating the ``trackId`` anchor (commit 3cf67c1).
 */
export function resolveGtDisplayPid(
  gt: ActionGroundTruthLabel,
  canonicalRallyMap: CanonicalRallyMap | undefined,
  appliedFullMapping: Record<string, number> | undefined,
  playerNumberMap: Map<number, number> | undefined,
): number | null {
  if (gt.trackId !== undefined) {
    const resolved = resolveCanonicalPid(
      gt.trackId, canonicalRallyMap, appliedFullMapping, playerNumberMap,
    );
    if (resolved !== null) return resolved;
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
