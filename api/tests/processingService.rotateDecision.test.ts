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
import { shouldAutoRotate, shouldSkipOptimization } from '../src/services/processingService.js';

describe('shouldAutoRotate', () => {
  it('fires when |tilt| > 5 and linesScored >= 3', () => {
    expect(shouldAutoRotate({ tiltDeg: 7, linesScored: 5 })).toBe(true);
  });
  it('fires on negative tilt (|tilt| > 5)', () => {
    expect(shouldAutoRotate({ tiltDeg: -8, linesScored: 10 })).toBe(true);
  });
  it('does not fire when |tilt| is 5 (strict >)', () => {
    expect(shouldAutoRotate({ tiltDeg: 5, linesScored: 20 })).toBe(false);
  });
  it('does not fire when |tilt| is -5 (strict >)', () => {
    expect(shouldAutoRotate({ tiltDeg: -5, linesScored: 20 })).toBe(false);
  });
  it('does not fire when linesScored is 2 (needs >= 3)', () => {
    expect(shouldAutoRotate({ tiltDeg: 10, linesScored: 2 })).toBe(false);
  });
  it('fires when linesScored is exactly 3 (inclusive lower bound)', () => {
    expect(shouldAutoRotate({ tiltDeg: 10, linesScored: 3 })).toBe(true);
  });
  it('does not fire when already autoRotated', () => {
    expect(shouldAutoRotate({ autoRotated: true, tiltDeg: 10, linesScored: 20 })).toBe(false);
  });
  it('does not fire on null fields', () => {
    expect(shouldAutoRotate({})).toBe(false);
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
