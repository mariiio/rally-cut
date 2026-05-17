import { vi } from 'vitest';

// Mock env module so processingService can be imported without real env vars.
vi.mock('../src/config/env.js', () => ({
  env: {
    AWS_REGION: 'us-east-1',
    S3_BUCKET_NAME: 'test-bucket',
    MODAL_WEBHOOK_SECRET: 'test-secret',
    API_BASE_URL: 'http://localhost:3001',
    CORS_ORIGIN: 'http://localhost:3000',
  },
}));

import { describe, expect, it } from 'vitest';
import {
  buildRotateFilterChain,
  shouldAutoRotate,
  shouldSkipOptimization,
} from '../src/services/processingService.js';

describe('shouldAutoRotate', () => {
  const baseGood = { linesScored: 15 };

  it('fires when |tilt| is in (5, 8] with enough supporting lines', () => {
    expect(shouldAutoRotate({ ...baseGood, tiltDeg: 7 })).toBe(true);
  });
  it('fires on negative tilt within the band', () => {
    expect(shouldAutoRotate({ ...baseGood, tiltDeg: -7 })).toBe(true);
  });
  it('fires exactly at the 8° upper cap', () => {
    expect(shouldAutoRotate({ ...baseGood, tiltDeg: 8 })).toBe(true);
  });
  it('does not fire when |tilt| is 5 (strict lower >)', () => {
    expect(shouldAutoRotate({ ...baseGood, tiltDeg: 5 })).toBe(false);
  });
  it('does not fire when |tilt| is -5 (strict lower >)', () => {
    expect(shouldAutoRotate({ ...baseGood, tiltDeg: -5 })).toBe(false);
  });
  it('does not fire when |tilt| exceeds 8° cap (perspective territory)', () => {
    expect(shouldAutoRotate({ ...baseGood, tiltDeg: 9 })).toBe(false);
  });
  it('rejects the 952e1bf8 false-positive case (20° with 973 lines)', () => {
    expect(shouldAutoRotate({ tiltDeg: -19.97, linesScored: 973 })).toBe(false);
  });
  it('does not fire when linesScored is below 10', () => {
    expect(shouldAutoRotate({ ...baseGood, linesScored: 8, tiltDeg: 7 })).toBe(false);
  });
  it('fires when linesScored is exactly 10 (inclusive lower bound)', () => {
    expect(shouldAutoRotate({ ...baseGood, linesScored: 10, tiltDeg: 7 })).toBe(true);
  });
  it('does not fire when already autoRotated', () => {
    expect(shouldAutoRotate({ ...baseGood, autoRotated: true, tiltDeg: 7 })).toBe(false);
  });
  it('does not fire on empty input', () => {
    expect(shouldAutoRotate({})).toBe(false);
  });
});

describe('buildRotateFilterChain', () => {
  function parseScaleFactor(scaleStr: string): number {
    // "scale=trunc(iw*1.151.../2)*2:trunc(ih*1.151.../2)*2"
    const m = scaleStr.match(/iw\*([0-9.]+)\//);
    if (!m) throw new Error(`Could not parse scale factor from ${scaleStr}`);
    return Number(m[1]);
  }

  it('emits a [scale, rotate] pair that crops back to source dimensions', () => {
    const chain = buildRotateFilterChain(0.1, 1920, 1080);
    expect(chain).toHaveLength(2);
    expect(chain[0]).toMatch(/^scale=trunc\(iw\*/);
    expect(chain[1]).toMatch(/^rotate=0\.1:ow=1920:oh=1080:c=black$/);
  });

  it('scale factor matches M = |cos θ| + |sin θ| * (long/short)', () => {
    const rad = (7 * Math.PI) / 180;
    const expected = Math.cos(rad) + Math.sin(rad) * (1920 / 1080);
    const M = parseScaleFactor(buildRotateFilterChain(rad, 1920, 1080)[0]);
    expect(M).toBeCloseTo(expected, 6);
  });

  it('is sign-independent (clockwise and counter-clockwise zoom identically)', () => {
    const M_pos = parseScaleFactor(buildRotateFilterChain(0.12, 1920, 1080)[0]);
    const M_neg = parseScaleFactor(buildRotateFilterChain(-0.12, 1920, 1080)[0]);
    expect(M_pos).toBeCloseTo(M_neg, 10);
  });

  it('is aspect-orientation symmetric (landscape and portrait zoom identically)', () => {
    const M_land = parseScaleFactor(buildRotateFilterChain(0.12, 1920, 1080)[0]);
    const M_port = parseScaleFactor(buildRotateFilterChain(0.12, 1080, 1920)[0]);
    expect(M_land).toBeCloseTo(M_port, 10);
  });

  it('M ≈ 1.24 at the 8° upper cap (1920×1080)', () => {
    const rad = (8 * Math.PI) / 180;
    const M = parseScaleFactor(buildRotateFilterChain(rad, 1920, 1080)[0]);
    expect(M).toBeGreaterThan(1.23);
    expect(M).toBeLessThan(1.25);
  });

  it('throws on invalid dimensions', () => {
    expect(() => buildRotateFilterChain(0.1, 0, 1080)).toThrow(/invalid dimensions/);
    expect(() => buildRotateFilterChain(0.1, 1920, -1)).toThrow(/invalid dimensions/);
    expect(() => buildRotateFilterChain(0.1, NaN, 1080)).toThrow(/invalid dimensions/);
  });
});

describe('shouldSkipOptimization', () => {
  it('skips when neither bitrate/moov nor tilt demand a re-encode', () => {
    expect(shouldSkipOptimization({ needsOptimization: false, wantsRotate: false })).toBe(true);
  });
  it('runs when bitrate/moov demand optimization (legacy path)', () => {
    expect(shouldSkipOptimization({ needsOptimization: true, wantsRotate: false })).toBe(false);
  });
  it('runs when only rotation is needed — closes the pre-2026-05-13 bypass', () => {
    expect(shouldSkipOptimization({ needsOptimization: false, wantsRotate: true })).toBe(false);
  });
  it('runs when both signals demand a re-encode', () => {
    expect(shouldSkipOptimization({ needsOptimization: true, wantsRotate: true })).toBe(false);
  });
});
