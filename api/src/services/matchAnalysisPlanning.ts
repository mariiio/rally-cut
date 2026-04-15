import type { PendingEditsJson, PendingEdit } from './pendingAnalysisEdits.js';

export type StagePlan = {
  fullRerun: boolean;
  changedRallyIds: string[];    // stages 4/5 processed for these only
};

const FULL_RERUN_KINDS: ReadonlyArray<PendingEdit['editKind']> = ['extend', 'create', 'merge'];
const CHANGED_KINDS: ReadonlyArray<PendingEdit['editKind']> = ['shorten', 'split'];

export function planStages(edits: PendingEditsJson): StagePlan {
  if (edits.entries.length === 0) {
    return { fullRerun: true, changedRallyIds: [] };
  }
  if (edits.entries.some(e => FULL_RERUN_KINDS.includes(e.editKind))) {
    return { fullRerun: true, changedRallyIds: [] };
  }
  const changed = new Set<string>();
  for (const e of edits.entries) {
    if (CHANGED_KINDS.includes(e.editKind)) changed.add(e.rallyId);
  }
  return { fullRerun: false, changedRallyIds: [...changed] };
}
