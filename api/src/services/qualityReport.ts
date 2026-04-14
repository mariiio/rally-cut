/**
 * Pure types and functions for QualityReport manipulation.
 * No side-effectful imports — safe to import in tests without DB/S3/env.
 */

export type Tier = 'block' | 'gate' | 'advisory';

export interface Issue {
  id: string;
  tier: Tier;
  severity: number;
  message: string;
  source: 'preview' | 'upload' | 'preflight' | 'tracking';
  detectedAt: string;
  data?: Record<string, number>;
}

export interface QualityReport {
  version: 2;
  issues: Issue[];
  preflight?: { ranAt: string; sampleSeconds: number; durationMs: number } | null;
  brightness?: number | null;
  resolution?: { width: number; height: number } | null;
}

const TIER_ORDER: Record<Tier, number> = { block: 0, gate: 1, advisory: 2 };

export function pickTopIssues(issues: Issue[], max = 3): Issue[] {
  return [...issues]
    .sort(
      (a, b) =>
        TIER_ORDER[a.tier] - TIER_ORDER[b.tier] ||
        b.severity - a.severity ||
        a.id.localeCompare(b.id),
    )
    .slice(0, max);
}

export function mergeQualityReports(reports: Array<Partial<QualityReport>>): QualityReport {
  const allIssues: Issue[] = reports.flatMap((r) => r.issues ?? []);
  const brightness = reports.map((r) => r.brightness).find((v) => v != null) ?? null;
  const resolution = reports.map((r) => r.resolution).find((v) => v != null) ?? null;
  const preflight = reports.map((r) => r.preflight).find((v) => v != null) ?? null;
  return {
    version: 2,
    issues: pickTopIssues(allIssues),
    preflight,
    brightness,
    resolution,
  };
}
