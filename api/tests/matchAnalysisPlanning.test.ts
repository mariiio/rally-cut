import { describe, expect, it } from 'vitest';
import { planStages } from '../src/services/matchAnalysisPlanning';
import type { PendingEdit } from '../src/services/pendingAnalysisEdits';

const e = (rallyId: string, editKind: PendingEdit['editKind']): PendingEdit => ({ rallyId, editKind, at: '2026-04-15T00:00:00Z' });

describe('planStages', () => {
  it('empty entries → fullRerun', () => {
    expect(planStages({ entries: [] })).toEqual({ fullRerun: true, runStage2: false, changedRallyIds: [] });
  });

  it('scalar-only storm → no stages 2/3/4/5; stage 6 runs', () => {
    const plan = planStages({ entries: [e('r1', 'scalar'), e('r2', 'scalar')] });
    expect(plan).toEqual({ fullRerun: false, runStage2: false, changedRallyIds: [] });
  });

  it('delete-only storm → changedRallyIds empty', () => {
    expect(planStages({ entries: [e('r1', 'delete')] })).toEqual({ fullRerun: false, runStage2: false, changedRallyIds: [] });
  });

  it('shorten edits populate changedRallyIds (deduped)', () => {
    const plan = planStages({ entries: [e('r1', 'shorten'), e('r1', 'scalar'), e('r2', 'shorten')] });
    expect(plan).toMatchObject({ fullRerun: false, runStage2: false });
    expect(plan.changedRallyIds.sort()).toEqual(['r1', 'r2']);
  });

  it('split edits populate changedRallyIds', () => {
    const plan = planStages({ entries: [e('child-a', 'split'), e('child-b', 'split')] });
    expect(plan.fullRerun).toBe(false);
    expect(plan.changedRallyIds.sort()).toEqual(['child-a', 'child-b']);
  });

  it('extend triggers fullRerun regardless of other edits', () => {
    expect(planStages({ entries: [e('r1', 'shorten'), e('r2', 'extend')] })).toMatchObject({ fullRerun: true });
  });

  it('create triggers fullRerun', () => {
    expect(planStages({ entries: [e('r1', 'create')] })).toMatchObject({ fullRerun: true });
  });

  it('merge triggers fullRerun', () => {
    expect(planStages({ entries: [e('r1', 'merge')] })).toMatchObject({ fullRerun: true });
  });
});
