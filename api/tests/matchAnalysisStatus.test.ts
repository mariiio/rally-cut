/**
 * Tests for:
 *   - GET  /v1/videos/:id/match-analysis-status — derives status from
 *     matchAnalysisStartedAt / matchAnalysisRanAt / matchAnalysisError.
 *   - POST /v1/videos/:id/run-match-analysis — gates on the in-memory
 *     runningVideos lock; second concurrent caller gets 409.
 *
 * matchAnalysisService is mocked so the SSE route receives a controlled
 * `withMatchAnalysisLock` implementation. Mirrors the pattern in
 * matchAnalysisTrigger.test.ts.
 */
import { describe, it, expect, vi, beforeAll, afterEach } from 'vitest';
import request from 'supertest';

const VISITOR_ID = '00000000-0000-0000-0000-000000000002';
const OTHER_USER_ID = '00000000-0000-0000-0000-000000000099';
const VIDEO_ID = '22222222-2222-2222-2222-222222222222';

// ---------------------------------------------------------------------------
// Mock matchAnalysisService BEFORE the app is imported.
// withMatchAnalysisLock controls the 409 path; getMatchAnalysisStatus
// controls what the status route returns.
// ---------------------------------------------------------------------------
vi.mock('../src/services/matchAnalysisService.js', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/services/matchAnalysisService.js')>();
  return {
    ...real,
    withMatchAnalysisLock: vi.fn(),
    getMatchAnalysisStatus: vi.fn(),
    runMatchAnalysis: vi.fn().mockResolvedValue({ videoId: VIDEO_ID, numRallies: 0, rallies: [] }),
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
let withLockMock: ReturnType<typeof vi.fn>;
let getStatusMock: ReturnType<typeof vi.fn>;

beforeAll(async () => {
  const appModule = await import('../src/index.js');
  app = appModule.default;
  const { prisma } = await import('../src/lib/prisma.js');
  prismaVideoFindUnique = prisma.video.findUnique as ReturnType<typeof vi.fn>;
  const svc = await import('../src/services/matchAnalysisService.js');
  withLockMock = svc.withMatchAnalysisLock as ReturnType<typeof vi.fn>;
  getStatusMock = svc.getMatchAnalysisStatus as ReturnType<typeof vi.fn>;
});

afterEach(() => {
  vi.clearAllMocks();
});

// ---------------------------------------------------------------------------
// GET /v1/videos/:id/match-analysis-status
// ---------------------------------------------------------------------------
describe('GET /v1/videos/:id/match-analysis-status', () => {
  it('reports `idle` when the video has never been analyzed', async () => {
    getStatusMock.mockResolvedValue({
      status: 'idle', startedAt: null, ranAt: null, error: null,
    });

    const res = await request(app)
      .get(`/v1/videos/${VIDEO_ID}/match-analysis-status`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ status: 'idle' });
  });

  it('reports `processing` when startedAt > ranAt', async () => {
    getStatusMock.mockResolvedValue({
      status: 'processing',
      startedAt: '2026-05-13T20:00:00.000Z',
      ranAt: '2026-05-13T19:50:00.000Z',
      error: null,
    });

    const res = await request(app)
      .get(`/v1/videos/${VIDEO_ID}/match-analysis-status`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(200);
    expect(res.body.status).toBe('processing');
  });

  it('reports `completed` when ranAt >= startedAt and no error', async () => {
    getStatusMock.mockResolvedValue({
      status: 'completed',
      startedAt: '2026-05-13T20:00:00.000Z',
      ranAt: '2026-05-13T20:01:30.000Z',
      error: null,
    });

    const res = await request(app)
      .get(`/v1/videos/${VIDEO_ID}/match-analysis-status`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(200);
    expect(res.body.status).toBe('completed');
    expect(res.body.error).toBeNull();
  });

  it('reports `failed` with error message', async () => {
    getStatusMock.mockResolvedValue({
      status: 'failed',
      startedAt: '2026-05-13T20:00:00.000Z',
      ranAt: '2026-05-13T20:02:00.000Z',
      error: 'match-players timed out after 15 minutes',
    });

    const res = await request(app)
      .get(`/v1/videos/${VIDEO_ID}/match-analysis-status`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(200);
    expect(res.body.status).toBe('failed');
    expect(res.body.error).toMatch(/match-players timed out/);
  });

  it('returns 404 when the video does not exist', async () => {
    const { NotFoundError } = await import('../src/middleware/errorHandler.js');
    getStatusMock.mockRejectedValue(new NotFoundError('Video', VIDEO_ID));

    const res = await request(app)
      .get(`/v1/videos/${VIDEO_ID}/match-analysis-status`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(404);
  });

  it('returns 403 when the caller does not own the video', async () => {
    const { ForbiddenError } = await import('../src/middleware/errorHandler.js');
    getStatusMock.mockRejectedValue(new ForbiddenError('You do not have permission to view match analysis status for this video'));

    const res = await request(app)
      .get(`/v1/videos/${VIDEO_ID}/match-analysis-status`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(403);
  });
});

// ---------------------------------------------------------------------------
// POST /v1/videos/:id/run-match-analysis — concurrency gate
// ---------------------------------------------------------------------------
describe('POST /v1/videos/:id/run-match-analysis', () => {
  it('returns 409 when withMatchAnalysisLock signals an in-flight run', async () => {
    prismaVideoFindUnique.mockResolvedValue({ id: VIDEO_ID, userId: VISITOR_ID });
    // Simulate the lock already held by another caller.
    withLockMock.mockResolvedValue(null);

    const res = await request(app)
      .post(`/v1/videos/${VIDEO_ID}/run-match-analysis`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(409);
    expect(res.body).toMatchObject({
      error: { code: 'CONFLICT', details: { reason: 'MATCH_ANALYSIS_IN_PROGRESS' } },
    });
  });

  it('returns 404 when the video does not exist', async () => {
    prismaVideoFindUnique.mockResolvedValue(null);

    const res = await request(app)
      .post(`/v1/videos/${VIDEO_ID}/run-match-analysis`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(404);
  });

  it('returns 403 when the caller does not own the video', async () => {
    prismaVideoFindUnique.mockResolvedValue({ id: VIDEO_ID, userId: OTHER_USER_ID });

    const res = await request(app)
      .post(`/v1/videos/${VIDEO_ID}/run-match-analysis`)
      .set('X-Visitor-Id', VISITOR_ID);

    expect(res.status).toBe(403);
  });
});

// ---------------------------------------------------------------------------
// withMatchAnalysisLock — unit
// ---------------------------------------------------------------------------
describe('withMatchAnalysisLock — in-memory guard (unit)', () => {
  it('returns the wrapped result on first call and null on a concurrent second call', async () => {
    const realSvc = await vi.importActual<typeof import('../src/services/matchAnalysisService.js')>(
      '../src/services/matchAnalysisService.js',
    );

    const videoId = 'lock-test-' + Math.random().toString(36).slice(2);

    let release: (() => void) = () => {};
    const firstPromise = realSvc.withMatchAnalysisLock(videoId, () => new Promise<string>((resolve) => {
      release = () => resolve('first-done');
    }));

    // While the first call is in flight, the second call should fail-fast.
    const secondResult = await realSvc.withMatchAnalysisLock(videoId, async () => 'should-not-run');
    expect(secondResult).toBeNull();
    expect(realSvc.isMatchAnalysisRunning(videoId)).toBe(true);

    release();
    const firstResult = await firstPromise;
    expect(firstResult).toBe('first-done');
    expect(realSvc.isMatchAnalysisRunning(videoId)).toBe(false);
  });

  it('releases the lock if the wrapped fn throws', async () => {
    const realSvc = await vi.importActual<typeof import('../src/services/matchAnalysisService.js')>(
      '../src/services/matchAnalysisService.js',
    );

    const videoId = 'lock-throw-' + Math.random().toString(36).slice(2);

    await expect(realSvc.withMatchAnalysisLock(videoId, async () => {
      throw new Error('boom');
    })).rejects.toThrow('boom');

    expect(realSvc.isMatchAnalysisRunning(videoId)).toBe(false);
  });
});
