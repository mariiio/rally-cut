import { describe, expect, it } from 'vitest';
import { planStages, shouldForceFullRerun } from '../src/services/matchAnalysisPlanning';
import type { PendingEdit } from '../src/services/pendingAnalysisEdits';

const e = (rallyId: string, editKind: PendingEdit['editKind']): PendingEdit => ({ rallyId, editKind, at: '2026-04-15T00:00:00Z' });

describe('planStages', () => {
  it('empty entries → fullRerun', () => {
    expect(planStages({ entries: [] })).toEqual({ fullRerun: true, changedRallyIds: [] });
  });

  it('scalar-only storm → no stages 2/3/4/5; stage 6 runs', () => {
    const plan = planStages({ entries: [e('r1', 'scalar'), e('r2', 'scalar')] });
    expect(plan).toEqual({ fullRerun: false, changedRallyIds: [] });
  });

  it('delete-only storm → changedRallyIds empty', () => {
    expect(planStages({ entries: [e('r1', 'delete')] })).toEqual({ fullRerun: false, changedRallyIds: [] });
  });

  it('shorten edits populate changedRallyIds (deduped)', () => {
    const plan = planStages({ entries: [e('r1', 'shorten'), e('r1', 'scalar'), e('r2', 'shorten')] });
    expect(plan).toMatchObject({ fullRerun: false });
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

  it('refCrop triggers fullRerun (canonical pid map needs full rebuild)', () => {
    expect(planStages({ entries: [e('', 'refCrop')] })).toMatchObject({ fullRerun: true });
  });

  it('refCrop alongside scalars still triggers fullRerun', () => {
    expect(planStages({ entries: [e('r1', 'scalar'), e('', 'refCrop')] })).toMatchObject({ fullRerun: true });
  });
});

describe('shouldForceFullRerun', () => {
  it('forces fullRerun when plan is partial AND matchAnalysisJson is missing', () => {
    // Real-world trigger: tracking finishes, scalar/delete edits accumulated
    // during tracking → planStages returns partial → but no prior matchAnalysisJson
    // exists, so partial-rerun would skip match-players and player IDs stay
    // inconsistent across rallies.
    const partial = planStages({ entries: [e('r1', 'scalar')] });
    expect(shouldForceFullRerun(partial, /* hasMatchAnalysisJson */ false)).toBe(true);
  });

  it('does NOT force fullRerun when matchAnalysisJson already exists', () => {
    // Normal partial-rerun case: prior matchAnalysisJson exists, only scalar
    // edits since — stages 4/5 sufficient.
    const partial = planStages({ entries: [e('r1', 'scalar')] });
    expect(shouldForceFullRerun(partial, /* hasMatchAnalysisJson */ true)).toBe(false);
  });

  it('does NOT force when plan is already fullRerun', () => {
    const full = planStages({ entries: [e('r1', 'create')] });
    expect(shouldForceFullRerun(full, /* hasMatchAnalysisJson */ false)).toBe(false);
    expect(shouldForceFullRerun(full, /* hasMatchAnalysisJson */ true)).toBe(false);
  });

  it('does NOT force when plan is fullRerun from empty edits', () => {
    const full = planStages({ entries: [] });
    expect(shouldForceFullRerun(full, /* hasMatchAnalysisJson */ false)).toBe(false);
  });
});
