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
import { shouldAutoRotate } from '../src/services/processingService.js';

describe('shouldAutoRotate', () => {
  it('fires when tilt > 5 and confidence > 0.6', () => {
    expect(shouldAutoRotate({ tiltDeg: 7, courtConfidence: 0.7 })).toBe(true);
  });
  it('does not fire when tilt is 5 (strict >)', () => {
    expect(shouldAutoRotate({ tiltDeg: 5, courtConfidence: 0.9 })).toBe(false);
  });
  it('does not fire when confidence is 0.6 (strict >)', () => {
    expect(shouldAutoRotate({ tiltDeg: 10, courtConfidence: 0.6 })).toBe(false);
  });
  it('does not fire when already autoRotated', () => {
    expect(shouldAutoRotate({ autoRotated: true, tiltDeg: 10, courtConfidence: 0.9 })).toBe(false);
  });
  it('does not fire on null fields', () => {
    expect(shouldAutoRotate({})).toBe(false);
  });
});
