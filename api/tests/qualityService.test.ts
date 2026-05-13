import { describe, it, expect } from 'vitest';
import { mergeQualityReports, pickTopIssues, type Issue } from '../src/services/qualityReport.js';
import { areCornersReasonable } from '../src/services/qualityService.js';

describe('mergeQualityReports', () => {
  it('combines upload + preflight issues and sorts by tier then severity', () => {
    const upload = {
      version: 2 as const,
      issues: [
        { id: 'too_dark', tier: 'gate' as const, severity: 0.4, message: '', source: 'upload' as const, detectedAt: '', data: {} },
      ],
    };
    const preflight = {
      version: 2 as const,
      issues: [
        { id: 'wrong_angle_or_not_volleyball', tier: 'block' as const, severity: 1.0, message: '', source: 'preflight' as const, detectedAt: '', data: {} },
        { id: 'camera_too_far', tier: 'gate' as const, severity: 0.9, message: '', source: 'preflight' as const, detectedAt: '', data: {} },
      ],
    };
    const merged = mergeQualityReports([upload, preflight]);
    expect(merged.issues.map((i) => i.id)).toEqual(['wrong_angle_or_not_volleyball', 'camera_too_far', 'too_dark']);
  });

  it('prefers non-null brightness/resolution/preflight across reports', () => {
    const r1 = { version: 2 as const, issues: [], brightness: null, resolution: null };
    const r2 = { version: 2 as const, issues: [], brightness: 0.42, resolution: { width: 1920, height: 1080 } };
    const merged = mergeQualityReports([r1, r2]);
    expect(merged.brightness).toBe(0.42);
    expect(merged.resolution).toEqual({ width: 1920, height: 1080 });
  });

  it('preserves an existing court detection when a fresh report omits it', () => {
    const existing = {
      version: 2 as const,
      issues: [],
      court: {
        corners: [
          { x: 0.2, y: 0.8 },
          { x: 0.8, y: 0.8 },
          { x: 0.7, y: 0.4 },
          { x: 0.3, y: 0.4 },
        ],
        confidence: 0.85,
      },
    };
    const fresh = { version: 2 as const, issues: [] };
    const merged = mergeQualityReports([existing, fresh]);
    expect(merged.court?.confidence).toBe(0.85);
  });

  it('overwrites an existing court detection with a fresh one (last-wins)', () => {
    const existing = {
      version: 2 as const,
      issues: [],
      court: {
        corners: [
          { x: 0.1, y: 0.9 },
          { x: 0.9, y: 0.9 },
          { x: 0.85, y: 0.4 },
          { x: 0.15, y: 0.4 },
        ],
        confidence: 0.55,
      },
    };
    const fresh = {
      version: 2 as const,
      issues: [],
      court: {
        corners: [
          { x: 0.2, y: 0.8 },
          { x: 0.8, y: 0.8 },
          { x: 0.7, y: 0.4 },
          { x: 0.3, y: 0.4 },
        ],
        confidence: 0.91,
      },
    };
    const merged = mergeQualityReports([existing, fresh]);
    expect(merged.court?.confidence).toBe(0.91);
  });
});

describe('areCornersReasonable', () => {
  const ok = [
    { x: 0.2, y: 0.8 },
    { x: 0.8, y: 0.8 },
    { x: 0.7, y: 0.4 },
    { x: 0.3, y: 0.4 },
  ];

  it('accepts on-screen corners', () => {
    expect(areCornersReasonable(ok)).toBe(true);
  });

  it('accepts modestly off-screen corners (low camera angle)', () => {
    expect(
      areCornersReasonable([
        { x: -0.1, y: 1.1 },
        { x: 1.1, y: 1.1 },
        { x: 0.85, y: 0.4 },
        { x: 0.15, y: 0.4 },
      ]),
    ).toBe(true);
  });

  it('rejects corners far off the frame', () => {
    expect(
      areCornersReasonable([
        { x: -0.5, y: 0.8 },
        ok[1],
        ok[2],
        ok[3],
      ]),
    ).toBe(false);
  });

  it('rejects wrong-length input', () => {
    expect(areCornersReasonable(ok.slice(0, 3))).toBe(false);
    expect(areCornersReasonable([...ok, { x: 0.5, y: 0.5 }])).toBe(false);
  });
});

describe('pickTopIssues', () => {
  it('caps display to 3 items by (tier, -severity, id)', () => {
    const issues: Issue[] = [
      { id: 'a', tier: 'advisory', severity: 0.9, message: '', source: 'preflight', detectedAt: '', data: {} },
      { id: 'b', tier: 'block', severity: 0.2, message: '', source: 'preflight', detectedAt: '', data: {} },
      { id: 'c', tier: 'gate', severity: 0.8, message: '', source: 'preflight', detectedAt: '', data: {} },
      { id: 'd', tier: 'gate', severity: 0.1, message: '', source: 'preflight', detectedAt: '', data: {} },
    ];
    expect(pickTopIssues(issues).map((i) => i.id)).toEqual(['b', 'c', 'd']);
  });
});
