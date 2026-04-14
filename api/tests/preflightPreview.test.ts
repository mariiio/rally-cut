/**
 * Tests for POST /v1/videos/preflight-preview
 *
 * Full integration (supertest + real subprocess + real DB) requires the
 * analysis venv, keypoint model weights, and a live DATABASE_URL — not
 * available in the unit-test environment. These tests therefore mock
 * runPreviewChecks and verify the HTTP handler contract instead.
 *
 * End-to-end verification (real frames → real CLI → real assertion) is
 * covered by the Task 18 E2E suite.
 */
import { describe, it, expect, vi, beforeAll } from 'vitest';
import request from 'supertest';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR = path.join(__dirname, 'fixtures');

// ---------------------------------------------------------------------------
// Mock qualityService BEFORE the app is imported so the route picks it up.
// ---------------------------------------------------------------------------
vi.mock('../src/services/qualityService.js', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/services/qualityService.js')>();
  return {
    ...real,
    runPreviewChecks: vi.fn(),
  };
});

// Mock Prisma and env so the app can be imported without a real DB.
vi.mock('../src/lib/prisma.js', () => ({
  prisma: {
    user: { findFirst: vi.fn(), create: vi.fn(), upsert: vi.fn() },
    anonymousIdentity: { findUnique: vi.fn(), upsert: vi.fn() },
  },
}));

// Set required env vars before env.ts parses them.
process.env['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/test';
process.env['AWS_REGION'] = 'us-east-1';
process.env['S3_BUCKET_NAME'] = 'test-bucket';
process.env['MODAL_WEBHOOK_SECRET'] = 'test-secret';
process.env['CORS_ORIGIN'] = 'http://localhost:3000';

let app: Express.Application;
let runPreviewChecks: ReturnType<typeof vi.fn>;

beforeAll(async () => {
  const appModule = await import('../src/index.js');
  app = appModule.default as Express.Application;
  const qualityModule = await import('../src/services/qualityService.js');
  runPreviewChecks = qualityModule.runPreviewChecks as ReturnType<typeof vi.fn>;
});

const VISITOR_ID = '00000000-0000-0000-0000-000000000001';

// Mock resolveUser middleware: attach userId to req so requireUser passes.
vi.mock('../src/middleware/resolveUser.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/middleware/resolveUser.js')>();
  return {
    ...actual,
    resolveUser: (req: { userId: string }, _res: unknown, next: () => void) => {
      req.userId = VISITOR_ID;
      next();
    },
    requireUser: (_req: unknown, _res: unknown, next: () => void) => next(),
    requireAuthenticated: (_req: unknown, _res: unknown, next: () => void) => next(),
  };
});

describe('POST /v1/videos/preflight-preview', () => {
  it('returns pass=true for a beach-volleyball thumbnail (mocked)', async () => {
    runPreviewChecks.mockResolvedValueOnce({ pass: true, issues: [] });

    const frame = fs.readFileSync(path.join(FIXTURE_DIR, 'beach_vb_frame.jpg'));
    const res = await request(app)
      .post('/v1/videos/preflight-preview')
      .set('X-Visitor-Id', VISITOR_ID)
      .attach('frames', frame, 'f1.jpg')
      .field('width', '1920')
      .field('height', '1080')
      .field('durationS', '300');

    expect(res.status).toBe(200);
    expect(res.body.pass).toBe(true);
    expect(res.body.issues).toEqual([]);
  });

  it('returns pass=false with a block issue for a non-VB thumbnail (mocked)', async () => {
    runPreviewChecks.mockResolvedValueOnce({
      pass: false,
      issues: [
        {
          id: 'wrong_angle_or_not_volleyball',
          tier: 'block',
          severity: 1.0,
          message: "We couldn't find a beach volleyball court in this video.",
          source: 'preview',
          detectedAt: new Date().toISOString(),
          data: { courtConfidence: 0.1 },
        },
      ],
    });

    const frame = fs.readFileSync(path.join(FIXTURE_DIR, 'non_vb_frame.jpg'));
    const res = await request(app)
      .post('/v1/videos/preflight-preview')
      .set('X-Visitor-Id', VISITOR_ID)
      .attach('frames', frame, 'f1.jpg')
      .field('width', '1920')
      .field('height', '1080')
      .field('durationS', '300');

    expect(res.status).toBe(200);
    expect(res.body.pass).toBe(false);
    expect(res.body.issues.some((i: { tier: string }) => i.tier === 'block')).toBe(true);
  });

  it('returns 400 when no frames are uploaded', async () => {
    const res = await request(app)
      .post('/v1/videos/preflight-preview')
      .set('X-Visitor-Id', VISITOR_ID)
      .field('width', '1920')
      .field('height', '1080')
      .field('durationS', '300');

    expect(res.status).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('returns 400 when width/height/durationS are missing', async () => {
    const frame = fs.readFileSync(path.join(FIXTURE_DIR, 'non_vb_frame.jpg'));
    const res = await request(app)
      .post('/v1/videos/preflight-preview')
      .set('X-Visitor-Id', VISITOR_ID)
      .attach('frames', frame, 'f1.jpg');

    expect(res.status).toBe(400);
    expect(res.body.error).toBeDefined();
  });
});
