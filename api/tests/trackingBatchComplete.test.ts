import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import request from 'supertest';
import { handleTrackingBatchComplete } from '../src/services/modalTrackingService';
import * as matchAnalysisService from '../src/services/matchAnalysisService';
import { prisma } from '../src/lib/prisma';
import app from '../src/index';

describe('handleTrackingBatchComplete', () => {
  const videoId = 'vid-batch-test';
  const userId = 'user-batch-test';
  const jobId = 'job-batch-test';

  beforeEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { id: jobId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    // Create a minimal Video fixture so BatchTrackingJob FK is satisfied.
    // Pattern mirrors staleJobRecovery.test.ts — no User row required.
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'batch-test-video',
        filename: 'batch-test.mp4',
        s3Key: 'test/batch-test.mp4',
        contentHash: 'batch-test-hash',
      },
    });
    await prisma.batchTrackingJob.create({
      data: {
        id: jobId,
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 2,
        completedRallies: 1,
        failedRallies: 0,
      },
    });
  });

  afterEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { id: jobId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    vi.restoreAllMocks();
  });

  it('does NOT invoke runMatchAnalysis when batch completes (A2a: debounced on client)', async () => {
    const runSpy = vi
      .spyOn(matchAnalysisService, 'runMatchAnalysis')
      .mockResolvedValue(undefined as never);

    await handleTrackingBatchComplete({
      batch_job_id: jobId,
      video_id: videoId,
      status: 'completed',
      completed_rallies: 2,
      failed_rallies: 0,
    });

    expect(runSpy).not.toHaveBeenCalled();
    const updated = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: jobId } });
    expect(updated.status).toBe('COMPLETED');
    expect(updated.completedRallies).toBe(2);
    expect(updated.failedRallies).toBe(0);
  });

  it('still records failure status when batch fails', async () => {
    await handleTrackingBatchComplete({
      batch_job_id: jobId,
      video_id: videoId,
      status: 'failed',
      completed_rallies: 0,
      failed_rallies: 2,
      error: 'test failure',
    });

    const updated = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: jobId } });
    expect(updated.status).toBe('FAILED');
    expect(updated.error).toBe('test failure');
  });
});

describe('POST /v1/webhooks/tracking-batch-complete — idempotency', () => {
  // UUIDs are required by the Zod schema that validates incoming webhook bodies.
  const videoId = 'cc000000-dead-beef-0000-000000000001';
  const userId = 'user-dedup-test';
  const jobId = 'cc000000-dead-beef-0000-000000000002';
  const webhookPath = '/v1/webhooks/tracking-batch-complete';
  const webhookSecret = process.env['MODAL_WEBHOOK_SECRET'] ?? 'test-secret';

  beforeEach(async () => {
    await prisma.webhookDelivery.deleteMany({ where: { webhookPath } });
    await prisma.batchTrackingJob.deleteMany({ where: { id: jobId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'dedup-test-video',
        filename: 'dedup-test.mp4',
        s3Key: 'test/dedup-test.mp4',
        contentHash: 'dedup-test-hash',
      },
    });
    await prisma.batchTrackingJob.create({
      data: {
        id: jobId,
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 2,
        completedRallies: 0,
        failedRallies: 0,
      },
    });
  });

  afterEach(async () => {
    await prisma.webhookDelivery.deleteMany({ where: { webhookPath } });
    await prisma.batchTrackingJob.deleteMany({ where: { id: jobId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    vi.restoreAllMocks();
  });

  it('POST tracking-batch-complete dedups identical deliveries at route level', async () => {
    vi.spyOn(matchAnalysisService, 'runMatchAnalysis').mockResolvedValue(undefined as never);

    const payload = {
      batch_job_id: jobId,
      video_id: videoId,
      status: 'completed',
      completed_rallies: 2,
      failed_rallies: 0,
    };

    // First delivery — actually processes.
    const first = await request(app)
      .post(webhookPath)
      .set('x-webhook-secret', webhookSecret)
      .send(payload);
    expect(first.status).toBe(200);
    expect(first.body).not.toHaveProperty('deduplicated');

    const afterFirst = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: jobId } });
    expect(afterFirst.status).toBe('COMPLETED');

    // Second POST, identical payload — should dedup at the route level.
    const second = await request(app)
      .post(webhookPath)
      .set('x-webhook-secret', webhookSecret)
      .send(payload);
    expect(second.status).toBe(200);
    expect(second.body).toMatchObject({ deduplicated: true });

    // DB state must remain COMPLETED — the second call did no mutations.
    const afterSecond = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: jobId } });
    expect(afterSecond.status).toBe('COMPLETED');
  });
});
