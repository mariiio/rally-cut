/**
 * Tests for POST /v1/videos/:id/trigger-match-analysis
 *
 * Tests the HTTP handler contract: 202, 409, 404, 403.
 *
 * matchAnalysisService is mocked at the module level so the route receives
 * a controlled triggerMatchAnalysis. This mirrors how preflightPreview.test.ts
 * mocks qualityService before importing the app.
 *
 * Separately, the guard logic (runningVideos set) is tested as a unit.
 */
import { describe, it, expect, vi, beforeAll, afterEach } from 'vitest';
import request from 'supertest';

const VISITOR_ID = '00000000-0000-0000-0000-000000000002';
const OTHER_USER_ID = '00000000-0000-0000-0000-000000000099';
const VIDEO_ID = '11111111-1111-1111-1111-111111111111';

// ---------------------------------------------------------------------------
// Mock matchAnalysisService BEFORE the app is imported.
// triggerMatchAnalysis is the boundary the route calls — we control its
// return value per-test via mockReturnValue.
// ---------------------------------------------------------------------------
vi.mock('../src/services/matchAnalysisService.js', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/services/matchAnalysisService.js')>();
  return {
    ...real,
    triggerMatchAnalysis: vi.fn().mockReturnValue(true),
    runMatchAnalysis: vi.fn().mockResolvedValue(undefined),
  };
});

// ---------------------------------------------------------------------------
// Mock prisma BEFORE the app is imported.
// ---------------------------------------------------------------------------
vi.mock('../src/lib/prisma.js', () => ({
  prisma: {
    video: { findUnique: vi.fn() },
    user: { findFirst: vi.fn(), create: vi.fn(), upsert: vi.fn() },
    anonymousIdentity: { findUnique: vi.fn(), upsert: vi.fn() },
  },
}));

// Mock resolveUser middleware: inject VISITOR_ID so requireUser passes.
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

// Set required env vars before env.ts parses them.
process.env['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/test';
process.env['AWS_REGION'] = 'us-east-1';
process.env['S3_BUCKET_NAME'] = 'test-bucket';
process.env['MODAL_WEBHOOK_SECRET'] = 'test-secret';
process.env['CORS_ORIGIN'] = 'http://localhost:3000';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let app: any;
let prismaVideoFindUnique: ReturnType<typeof vi.fn>;
let triggerMatchAnalysisMock: ReturnType<typeof vi.fn>;

beforeAll(async () => {
  const appModule = await import('../src/index.js');
  app = appModule.default;
  const { prisma } = await import('../src/lib/prisma.js');
  prismaVideoFindUnique = prisma.video.findUnique as ReturnType<typeof vi.fn>;
  const svc = await import('../src/services/matchAnalysisService.js');
  triggerMatchAnalysisMock = svc.triggerMatchAnalysis as ReturnType<typeof vi.fn>;
});

afterEach(() => {
  vi.clearAllMocks();
  // Restore triggerMatchAnalysis default: returns true (started)
  triggerMatchAnalysisMock?.mockReturnValue(true);
});

describe('POST /v1/videos/:id/trigger-match-analysis', () => {
  it('returns 202 and invokes triggerMatchAnalysis when video is owned', async () => {
    prismaVideoFindUnique.mockResolvedValue({ id: VIDEO_ID, userId: VISITOR_ID });

    const res = await request(app)
      .post(`/v1/videos/${VIDEO_ID}/trigger-match-analysis`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(202);
    expect(res.body).toMatchObject({ status: 'processing' });
    expect(triggerMatchAnalysisMock).toHaveBeenCalledWith(VIDEO_ID);
  });

  it('returns 409 when triggerMatchAnalysis reports a run is already in flight', async () => {
    prismaVideoFindUnique.mockResolvedValue({ id: VIDEO_ID, userId: VISITOR_ID });
    // Simulate the in-memory guard already armed.
    triggerMatchAnalysisMock.mockReturnValue(false);

    const res = await request(app)
      .post(`/v1/videos/${VIDEO_ID}/trigger-match-analysis`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(409);
    expect(res.body).toMatchObject({
      error: { code: 'CONFLICT', details: { reason: 'MATCH_ANALYSIS_IN_PROGRESS' } },
    });
  });

  it('returns 404 when the video does not exist', async () => {
    prismaVideoFindUnique.mockResolvedValue(null);

    const res = await request(app)
      .post(`/v1/videos/${VIDEO_ID}/trigger-match-analysis`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(404);
    expect(res.body).toMatchObject({ error: { code: 'NOT_FOUND' } });
  });

  it('returns 403 when the caller does not own the video', async () => {
    prismaVideoFindUnique.mockResolvedValue({ id: VIDEO_ID, userId: OTHER_USER_ID });

    const res = await request(app)
      .post(`/v1/videos/${VIDEO_ID}/trigger-match-analysis`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(403);
    expect(res.body).toMatchObject({ error: { code: 'FORBIDDEN' } });
  });
});

// ---------------------------------------------------------------------------
// Unit tests for the in-memory guard inside triggerMatchAnalysis.
// These import the real (non-mocked) service directly so the runningVideos
// Set behaviour can be observed end-to-end without HTTP overhead.
// ---------------------------------------------------------------------------
describe('triggerMatchAnalysis — in-memory guard (unit)', () => {
  it('returns true on first call and false if called again before completion', async () => {
    // Import the real service (bypasses vi.mock by using the unmocked module).
    // We mock runMatchAnalysis inside the real module via spyOn — this works
    // because we import the service directly and spy on its runMatchAnalysis
    // binding BEFORE calling triggerMatchAnalysis.
    //
    // Since both live in the same module, we test the guard by observing the
    // boolean return values rather than through vi.spyOn on runMatchAnalysis.
    const realSvc = await vi.importActual<typeof import('../src/services/matchAnalysisService.js')>(
      '../src/services/matchAnalysisService.js',
    );

    const videoId = 'guard-test-' + Math.random().toString(36).slice(2);

    // First call: should start the job (returns true)
    // But since we can't intercept the internal runMatchAnalysis call here,
    // we observe the guard state via isMatchAnalysisRunning after the call
    // finishes (which it does immediately since prisma mock throws and .finally runs).
    // Instead, spy via the exported reference trick: call triggerMatchAnalysis
    // and check isMatchAnalysisRunning immediately after (before event loop yields).

    const first = realSvc.triggerMatchAnalysis(videoId);
    expect(first).toBe(true);
    // Immediately: guard is still armed (runMatchAnalysis hasn't resolved yet)
    expect(realSvc.isMatchAnalysisRunning(videoId)).toBe(true);

    const second = realSvc.triggerMatchAnalysis(videoId);
    expect(second).toBe(false);

    // Let the promise resolve (real runMatchAnalysis will fail, .finally clears set)
    await new Promise((r) => setTimeout(r, 100));
    expect(realSvc.isMatchAnalysisRunning(videoId)).toBe(false);

  });
});
