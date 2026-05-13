import type { ActionGroundTruthLabel, MatchAnalysis } from '@/services/api';
import { resolveCanonicalPid } from '@/utils/canonicalPid';

/**
 * Resolve the display pid (1-4) for a GT label.
 *
 * Prefer `resolvedTrackId` (post-resolver attribution against current
 * tracking); fall back to `snapshotTrackId` (the trackId the user
 * originally clicked) when the resolver couldn't match. Without the
 * fallback, labels for players not tracked at the exact labeled frame
 * silently disappear from the UI even though the user's choice is
 * preserved in the DB.
 *
 * After Task 5's resolver runs, resolvedTrackId is the post-remap canonical
 * pid (when the resolver wrote SNAPSHOT_EXACT/MANUAL/NEAREST_CENTER from
 * positionsJson) OR the raw BoT-SORT id (when it wrote IOU_MATCH from
 * rawPositionsJson). For display we route through appliedFullMapping
 * (raw → canonical pid 1-4); if the lookup misses, the anchor is already
 * canonical so return it directly.
 */
export function resolveGtDisplayPid(
  gt: ActionGroundTruthLabel,
  appliedFullMapping: Record<string, number> | undefined,
  playerNumberMap: Map<number, number> | undefined,
): number | null {
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
