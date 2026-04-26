import type { ActionsData } from '@/services/api';

/**
 * Build the effective per-track team mapping for a rally, applying any
 * persisted high-confidence corrections from `correctionsApplied`.
 *
 * Server-side, `_try_fix_serve_receive` (compute_match_stats) detects
 * volleyball-rule violations (serve and receive on the same team) and
 * computes a per-track team override. The override is persisted as an
 * audit list (`correctionsApplied`) plus per-action `teamCorrected` shadow
 * fields — but the rally-level `teamAssignments` is left untouched as the
 * raw-model-output source-of-truth.
 *
 * UI consumers that color players by team (e.g. PlayerOverlay) should use
 * the corrected mapping. Returns the raw `teamAssignments` when no
 * corrections apply, or a copy with the overrides merged in.
 */
export function effectiveTeamAssignments(
  actions: ActionsData | undefined,
): Record<string, string> | undefined {
  const raw = actions?.teamAssignments;
  const corrections = actions?.correctionsApplied;
  if (!corrections || corrections.length === 0) return raw;

  const merged: Record<string, string> = { ...(raw ?? {}) };
  for (const c of corrections) {
    merged[String(c.playerTrackId)] = c.correctedTeam;
  }
  return merged;
}
