import type { ActionGroundTruthLabel, MatchAnalysis } from '@/services/api';
import { resolveCanonicalPid } from '@/utils/canonicalPid';

/**
 * Resolve the display pid (1-4) for a GT label.
 *
 * After Task 5's resolver runs, resolvedTrackId is the post-remap canonical
 * pid (when the resolver wrote SNAPSHOT_EXACT/MANUAL/NEAREST_CENTER from
 * positionsJson) OR the raw BoT-SORT id (when it wrote IOU_MATCH from
 * rawPositionsJson). For display we route through appliedFullMapping
 * (raw → canonical pid 1-4); if the lookup misses, resolvedTrackId
 * is already canonical so return it directly.
 */
export function resolveGtDisplayPid(
  gt: ActionGroundTruthLabel,
  appliedFullMapping: Record<string, number> | undefined,
  playerNumberMap: Map<number, number> | undefined,
): number | null {
  if (gt.resolvedTrackId == null) return null;
  const resolved = resolveCanonicalPid(
    gt.resolvedTrackId, appliedFullMapping, playerNumberMap,
  );
  return resolved ?? gt.resolvedTrackId;
}

/** Anchor id for a GT label — uses resolvedTrackId (current attribution). */
export function gtAnchorId(gt: ActionGroundTruthLabel): number | null {
  return gt.resolvedTrackId ?? null;
}

/** Look up the current rally entry from a cached match analysis. */
export function rallyMatchEntry(
  analysis: MatchAnalysis | undefined,
  backendRallyId: string | null,
): MatchAnalysis['rallies'][number] | undefined {
  if (!analysis || !backendRallyId) return undefined;
  return analysis.rallies.find((r) => r.rallyId === backendRallyId);
}
