import type { ActionGroundTruthLabel, MatchAnalysis } from '@/services/api';
import { resolveCanonicalPid } from '@/utils/canonicalPid';

/**
 * Resolve the display pid (1-4) for a GT label.
 *
 * Contract: `resolvedTrackId` is *canonical* (post-remap) when the resolver
 * has run (`resolvedSource` is one of SNAPSHOT_EXACT / IOU_MATCH /
 * REID_MATCH / NEAREST_CENTER / MANUAL). After the 2026-05-13 resolver
 * canonicalization (see `api/src/services/matchAnalysisService.ts` stage4b),
 * the resolver runs against `positionsJson` (canonical, post-remap), so a
 * non-null `resolvedTrackId` is directly usable for display — no AFM hop.
 *
 * Fall back to `snapshotTrackId` (the trackId the user originally clicked,
 * a raw BoT-SORT id) when the resolver couldn't match. The fallback is
 * still routed through `appliedFullMapping` because the snapshot is by
 * definition raw. Without the fallback, labels for players not tracked
 * at the exact labeled frame would silently disappear from the UI even
 * though the user's choice is preserved in the DB.
 */
export function resolveGtDisplayPid(
  gt: ActionGroundTruthLabel,
  appliedFullMapping: Record<string, number> | undefined,
  playerNumberMap: Map<number, number> | undefined,
): number | null {
  // Resolver-attributed → canonical, use directly.
  if (gt.resolvedTrackId != null && gt.resolvedSource != null && gt.resolvedSource !== 'UNRESOLVED') {
    return gt.resolvedTrackId;
  }
  // Unresolved or pre-resolver legacy row → fall back to snapshot, route
  // through AFM (which maps raw → canonical).
  const anchor = gt.resolvedTrackId ?? gt.snapshotTrackId;
  if (anchor == null) return null;
  const resolved = resolveCanonicalPid(
    anchor, appliedFullMapping, playerNumberMap,
  );
  return resolved ?? anchor;
}

/** Anchor id for a GT label — prefer resolved, fall back to snapshot. */
export function gtAnchorId(gt: ActionGroundTruthLabel): number | null {
  return gt.resolvedTrackId ?? gt.snapshotTrackId ?? null;
}

/** Look up the current rally entry from a cached match analysis. */
export function rallyMatchEntry(
  analysis: MatchAnalysis | undefined,
  backendRallyId: string | null,
): MatchAnalysis['rallies'][number] | undefined {
  if (!analysis || !backendRallyId) return undefined;
  return analysis.rallies.find((r) => r.rallyId === backendRallyId);
}
