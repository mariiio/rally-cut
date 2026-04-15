import { describe, it, expect } from 'vitest';
import { mergeQualityReports, AutoFix, QualityReport } from '../src/services/qualityReport.js';

describe('mergeQualityReports — autoFixes', () => {
  it('concatenates autoFixes arrays across reports', () => {
    const merged = mergeQualityReports([
      { version: 2, issues: [], autoFixes: [{ id: 'auto_straightened', message: 'A', appliedAt: '2026-04-15T00:00:00Z' }] },
      { version: 2, issues: [], autoFixes: [{ id: 'other_fix', message: 'B', appliedAt: '2026-04-15T00:00:01Z' }] },
    ]);
    expect(merged.autoFixes).toHaveLength(2);
    expect(merged.autoFixes?.map((f) => f.id).sort()).toEqual(['auto_straightened', 'other_fix']);
  });

  it('dedupes by id, keeping the first occurrence', () => {
    const merged = mergeQualityReports([
      { version: 2, issues: [], autoFixes: [{ id: 'auto_straightened', message: 'FIRST', appliedAt: '2026-04-15T00:00:00Z' }] },
      { version: 2, issues: [], autoFixes: [{ id: 'auto_straightened', message: 'SECOND', appliedAt: '2026-04-15T00:00:01Z' }] },
    ]);
    expect(merged.autoFixes).toHaveLength(1);
    expect(merged.autoFixes?.[0].message).toBe('FIRST');
  });

  it('returns undefined autoFixes when all inputs are empty', () => {
    const merged = mergeQualityReports([{ version: 2, issues: [] }]);
    expect(merged.autoFixes).toBeUndefined();
  });
});

describe('mergeQualityReports — new scalar fields', () => {
  it('keeps first non-null autoRotated', () => {
    const merged = mergeQualityReports([
      { version: 2, issues: [], autoRotated: true },
      { version: 2, issues: [], autoRotated: false },
    ]);
    expect(merged.autoRotated).toBe(true);
  });

  it('keeps first non-null tiltDeg', () => {
    const merged = mergeQualityReports([
      { version: 2, issues: [], tiltDeg: 7.3 },
      { version: 2, issues: [], tiltDeg: 0 },
    ]);
    expect(merged.tiltDeg).toBe(7.3);
  });
});
