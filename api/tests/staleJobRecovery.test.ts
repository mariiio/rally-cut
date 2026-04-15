import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { prisma } from '../src/lib/prisma';
import {
  expireStaleBatchTrackingJobs,
  expireStaleDetectionJobs,
  STALE_PROGRESS_TIMEOUT_MS,
} from '../src/services/staleJobRecovery';

describe('expireStaleBatchTrackingJobs', () => {
  const videoId = 'vid-stale-test';
  const userId = 'user-stale-test';

  beforeEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    // Create a minimal Video fixture so BatchTrackingJob FK is satisfied.
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'stale-test-video',
        filename: 'stale-test.mp4',
        s3Key: 'test/stale-test.mp4',
        contentHash: 'stale-test-hash',
      },
    });
  });

  afterEach(async () => {
    await prisma.batchTrackingJob.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    vi.useRealTimers();
  });

  it('marks PROCESSING jobs with no progress in >10min as FAILED', async () => {
    const stale = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS - 60_000);
    const job = await prisma.batchTrackingJob.create({
      data: {
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 3,
        completedRallies: 0,
        failedRallies: 0,
        lastProgressAt: stale,
        createdAt: stale,
      },
    });
    await expireStaleBatchTrackingJobs(videoId);
    const refreshed = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('FAILED');
    expect(refreshed.error).toMatch(/Timed out/i);
    expect(refreshed.completedAt).not.toBeNull();
  });

  it('leaves PROCESSING jobs with recent progress alone', async () => {
    const fresh = new Date(Date.now() - 60_000);
    const job = await prisma.batchTrackingJob.create({
      data: {
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 3,
        completedRallies: 1,
        failedRallies: 0,
        lastProgressAt: fresh,
        createdAt: fresh,
      },
    });
    await expireStaleBatchTrackingJobs(videoId);
    const refreshed = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('PROCESSING');
  });

  it('does not touch COMPLETED jobs even if lastProgressAt is ancient', async () => {
    const ancient = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS * 10);
    const job = await prisma.batchTrackingJob.create({
      data: {
        videoId,
        userId,
        status: 'COMPLETED',
        totalRallies: 3,
        completedRallies: 3,
        failedRallies: 0,
        lastProgressAt: ancient,
        createdAt: ancient,
        completedAt: ancient,
      },
    });
    await expireStaleBatchTrackingJobs(videoId);
    const refreshed = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('COMPLETED');
  });

  it('sweeps across multiple videos when called with no videoId', async () => {
    const extraVideoId = 'vid-stale-test-2';
    await prisma.batchTrackingJob.deleteMany({ where: { videoId: extraVideoId } });
    await prisma.video.deleteMany({ where: { id: extraVideoId } });
    await prisma.video.create({
      data: {
        id: extraVideoId,
        name: 'stale-test-video-2',
        filename: 'stale-test-2.mp4',
        s3Key: `${extraVideoId}.mp4`,
        contentHash: 'stale-test-hash-2',
      },
    });

    const stale = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS - 60_000);
    const jobA = await prisma.batchTrackingJob.create({
      data: {
        videoId,
        userId,
        status: 'PROCESSING',
        totalRallies: 1,
        completedRallies: 0,
        failedRallies: 0,
        lastProgressAt: stale,
        createdAt: stale,
      },
    });
    const jobB = await prisma.batchTrackingJob.create({
      data: {
        videoId: extraVideoId,
        userId,
        status: 'PENDING',
        totalRallies: 1,
        completedRallies: 0,
        failedRallies: 0,
        lastProgressAt: stale,
        createdAt: stale,
      },
    });

    // Call with no videoId — should sweep both videos
    const count = await expireStaleBatchTrackingJobs();
    expect(count).toBeGreaterThanOrEqual(2);

    const refreshedA = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: jobA.id } });
    const refreshedB = await prisma.batchTrackingJob.findUniqueOrThrow({ where: { id: jobB.id } });
    expect(refreshedA.status).toBe('FAILED');
    expect(refreshedB.status).toBe('FAILED');

    // Cleanup extra fixtures
    await prisma.batchTrackingJob.deleteMany({ where: { videoId: extraVideoId } });
    await prisma.video.deleteMany({ where: { id: extraVideoId } });
  });
});

describe('expireStaleDetectionJobs', () => {
  const videoId = 'vid-det-stale-test';

  beforeEach(async () => {
    await prisma.rallyDetectionJob.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'det-stale-test-video',
        filename: 'det-stale-test.mp4',
        s3Key: `${videoId}.mp4`,
        contentHash: 'det-stale-test-hash',
      },
    });
  });

  afterEach(async () => {
    await prisma.rallyDetectionJob.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    vi.useRealTimers();
  });

  it('marks RUNNING jobs older than 10min as FAILED', async () => {
    const stale = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS - 60_000);
    const job = await prisma.rallyDetectionJob.create({
      data: {
        videoId,
        contentHash: 'x'.repeat(64),
        status: 'RUNNING',
        createdAt: stale,
      },
    });
    await expireStaleDetectionJobs(videoId);
    const refreshed = await prisma.rallyDetectionJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('FAILED');
    expect(refreshed.errorMessage).toMatch(/Timed out/i);
    expect(refreshed.completedAt).not.toBeNull();
  });

  it('marks PENDING jobs older than 10min as FAILED', async () => {
    const stale = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS - 60_000);
    const job = await prisma.rallyDetectionJob.create({
      data: {
        videoId,
        contentHash: 'y'.repeat(64),
        status: 'PENDING',
        createdAt: stale,
      },
    });
    await expireStaleDetectionJobs(videoId);
    const refreshed = await prisma.rallyDetectionJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('FAILED');
  });

  it('leaves recent RUNNING jobs alone', async () => {
    const fresh = new Date(Date.now() - 30_000);
    const job = await prisma.rallyDetectionJob.create({
      data: {
        videoId,
        contentHash: 'z'.repeat(64),
        status: 'RUNNING',
        createdAt: fresh,
      },
    });
    await expireStaleDetectionJobs(videoId);
    const refreshed = await prisma.rallyDetectionJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('RUNNING');
  });

  it('does not touch COMPLETED jobs even if createdAt is ancient', async () => {
    const ancient = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS * 10);
    const job = await prisma.rallyDetectionJob.create({
      data: {
        videoId,
        contentHash: 'w'.repeat(64),
        status: 'COMPLETED',
        createdAt: ancient,
        completedAt: ancient,
      },
    });
    await expireStaleDetectionJobs(videoId);
    const refreshed = await prisma.rallyDetectionJob.findUniqueOrThrow({ where: { id: job.id } });
    expect(refreshed.status).toBe('COMPLETED');
  });

  it('sweeps across multiple videos when called with no videoId', async () => {
    const extraVideoId = 'vid-det-stale-test-2';
    await prisma.rallyDetectionJob.deleteMany({ where: { videoId: extraVideoId } });
    await prisma.video.deleteMany({ where: { id: extraVideoId } });
    await prisma.video.create({
      data: {
        id: extraVideoId,
        name: 'det-stale-test-video-2',
        filename: 'det-stale-test-2.mp4',
        s3Key: `${extraVideoId}.mp4`,
        contentHash: 'det-stale-test-hash-2',
      },
    });

    const stale = new Date(Date.now() - STALE_PROGRESS_TIMEOUT_MS - 60_000);
    const jobA = await prisma.rallyDetectionJob.create({
      data: {
        videoId,
        contentHash: 'a'.repeat(64),
        status: 'RUNNING',
        createdAt: stale,
      },
    });
    const jobB = await prisma.rallyDetectionJob.create({
      data: {
        videoId: extraVideoId,
        contentHash: 'b'.repeat(64),
        status: 'PENDING',
        createdAt: stale,
      },
    });

    // Call with no videoId — should sweep both videos
    const count = await expireStaleDetectionJobs();
    expect(count).toBeGreaterThanOrEqual(2);

    const refreshedA = await prisma.rallyDetectionJob.findUniqueOrThrow({ where: { id: jobA.id } });
    const refreshedB = await prisma.rallyDetectionJob.findUniqueOrThrow({ where: { id: jobB.id } });
    expect(refreshedA.status).toBe('FAILED');
    expect(refreshedB.status).toBe('FAILED');

    // Cleanup extra fixtures
    await prisma.rallyDetectionJob.deleteMany({ where: { videoId: extraVideoId } });
    await prisma.video.deleteMany({ where: { id: extraVideoId } });
  });
});
