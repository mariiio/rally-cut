/**
 * Quality report service.
 *
 * `runUploadChecks` runs on `POST /v1/videos/:id/confirm` (fast, metadata + frame sample).
 * `runPreflightChecks` runs on `POST /v1/videos/:id/assess-quality` just before detection;
 * spawns `rallycut preflight` which performs heavier checks (court keypoints, YOLO, etc.).
 * Both write to `Video.qualityReportJson`; the banner picks the top 3.
 */
import { spawn } from 'child_process';
import { createWriteStream } from 'fs';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { fileURLToPath } from 'url';

import { Prisma, VideoStatus } from '@prisma/client';
import { prisma } from '../lib/prisma.js';
import { generateDownloadUrl } from '../lib/s3.js';
import { NotFoundError } from '../middleware/errorHandler.js';
import {
  mergeQualityReports,
  pickTopIssues,
  type Issue,
  type QualityReport,
  type Tier,
} from './qualityReport.js';

// Re-export pure types and functions so callers can import from this module.
export { mergeQualityReports, pickTopIssues };
export type { Issue, QualityReport, Tier };

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ANALYSIS_DIR = path.resolve(__dirname, '../../../analysis');
const TEMP_DIR = path.join(os.tmpdir(), 'rallycut-quality');
const PREFLIGHT_TIMEOUT_MS = 120_000;

export async function runPreflightChecks(videoId: string, userId: string): Promise<QualityReport> {
  const video = await prisma.video.findFirst({ where: { id: videoId, userId, deletedAt: null } });
  if (!video) throw new NotFoundError('Video', videoId);

  await fs.mkdir(TEMP_DIR, { recursive: true });
  const suffix = `_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const ext = path.extname(video.filename || '.mp4');
  const localPath = path.join(TEMP_DIR, `preflight_${videoId}${suffix}${ext}`);

  try {
    await downloadFromS3(video.s3Key, localPath);
    const report = await runPreflightCli(localPath);

    const existing = (video.qualityReportJson as Partial<QualityReport> | null) ?? {};
    const merged = mergeQualityReports([existing, report]);
    await prisma.video.update({
      where: { id: videoId },
      data: { qualityReportJson: merged as unknown as Prisma.InputJsonValue },
    });

    // Sync Video.status with the block-issue state:
    //   - block issue present → mark REJECTED (disables Analyze button)
    //   - no block issue AND status was REJECTED from a previous run → revert to UPLOADED
    //     so the user can proceed after fixing the underlying issue (e.g. replacing the file)
    //   - otherwise leave status alone (preserves DETECTING/DETECTED)
    const hasBlock = merged.issues.some((i) => i.tier === 'block');
    if (hasBlock && video.status !== VideoStatus.REJECTED) {
      await prisma.video.update({
        where: { id: videoId },
        data: { status: VideoStatus.REJECTED },
      });
    } else if (!hasBlock && video.status === VideoStatus.REJECTED) {
      await prisma.video.update({
        where: { id: videoId },
        data: { status: VideoStatus.UPLOADED },
      });
    }

    return merged;
  } finally {
    await fs.unlink(localPath).catch(() => {});
  }
}

export async function saveUploadReport(videoId: string, report: Partial<QualityReport>) {
  const existing = await prisma.video.findUnique({ where: { id: videoId } });
  const prior = (existing?.qualityReportJson as Partial<QualityReport> | null) ?? {};
  const merged = mergeQualityReports([prior, report]);
  await prisma.video.update({
    where: { id: videoId },
    data: { qualityReportJson: merged as unknown as Prisma.InputJsonValue },
  });
}

/**
 * Get unified pipeline status for the AnalysisPipeline UI. Shape preserved
 * from the old service so current callers don't break; `quality.warnings` is
 * derived from `qualityReportJson.issues[].message`.
 */
export async function getAnalysisPipelineStatus(videoId: string, userId: string) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
    include: { rallies: { where: { status: 'CONFIRMED' }, select: { id: true } } },
  });
  if (!video) throw new NotFoundError('Video', videoId);

  const qr = video.qualityReportJson as QualityReport | null;
  const quality = qr
    ? {
        expectedQuality: qr.issues[0] ? 1 - qr.issues[0].severity : 1,
        warnings: qr.issues.map((i) => i.message),
        courtDetected: video.courtCalibrationJson !== null,
      }
    : null;

  const detectionStatus =
    video.status === 'DETECTED'
      ? 'completed'
      : video.status === 'DETECTING'
        ? 'processing'
        : video.status === 'UPLOADED'
          ? 'idle'
          : video.status;

  const STALE_JOB_TIMEOUT_MS = 10 * 60 * 1000;
  let batchJob = await prisma.batchTrackingJob.findFirst({
    where: { videoId },
    orderBy: { createdAt: 'desc' },
  });
  if (
    batchJob &&
    (batchJob.status === 'PROCESSING' || batchJob.status === 'PENDING') &&
    Date.now() - batchJob.lastProgressAt.getTime() > STALE_JOB_TIMEOUT_MS
  ) {
    batchJob = await prisma.batchTrackingJob.update({
      where: { id: batchJob.id },
      data: {
        status: 'FAILED',
        completedAt: new Date(),
        error: 'Timed out — job was interrupted or never completed',
      },
    });
  }

  const trackedRallies = await prisma.rally.count({
    where: { videoId, status: 'CONFIRMED', playerTrack: { status: 'COMPLETED' } },
  });

  return {
    quality,
    detection: { status: detectionStatus, ralliesFound: video.rallies.length },
    tracking: {
      status: batchJob?.status?.toLowerCase() ?? 'idle',
      completed: batchJob?.completedRallies ?? trackedRallies,
      total: batchJob?.totalRallies ?? video.rallies.length,
      failed: batchJob?.failedRallies ?? 0,
    },
    matchAnalysis: { available: video.matchAnalysisJson !== null },
    matchStats: { available: video.matchStatsJson !== null },
  };
}

// ============================================================================
// Player matching GT (preserved verbatim)
// ============================================================================

/**
 * Save player matching ground truth labels.
 */
// Player matching GT payload. Each rally carries a list of bbox-keyed
// labels — see analysis/rallycut/evaluation/gt_loader.py for the format
// spec and the IoU-based resolver used at load time.
type PlayerMatchingGtLabel = {
  playerId: number;
  frame: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
};

type PlayerMatchingGtRally = { labels: PlayerMatchingGtLabel[] };

export async function savePlayerMatchingGt(
  videoId: string,
  userId: string,
  gt: {
    rallies: Record<string, PlayerMatchingGtRally>;
    sideSwitches: number[];
    excludedRallies?: string[];
  },
) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }

  await prisma.video.update({
    where: { id: videoId },
    data: {
      playerMatchingGtJson: {
        ...gt,
        savedAt: new Date().toISOString(),
      } as unknown as Prisma.InputJsonValue,
    },
  });

  return { success: true };
}

/**
 * Get player matching ground truth labels.
 */
export async function getPlayerMatchingGt(videoId: string, userId: string) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }

  return video.playerMatchingGtJson as Record<string, unknown> | null;
}

// ============================================================================
// Helpers
// ============================================================================

function runPreflightCli(videoPath: string): Promise<QualityReport> {
  return new Promise((resolve, reject) => {
    const args = ['run', 'rallycut', 'preflight', videoPath, '--json', '--quiet'];
    const child = spawn('uv', args, {
      cwd: ANALYSIS_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env },
    });
    let stdout = '';
    let stderr = '';
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try {
        child.kill();
      } catch {
        /* ignore */
      }
      reject(new Error('Preflight timed out'));
    }, PREFLIGHT_TIMEOUT_MS);
    child.stdout?.on('data', (d: Buffer) => {
      stdout += d.toString();
    });
    child.stderr?.on('data', (d: Buffer) => {
      stderr += d.toString();
      const line = d.toString().trim();
      if (line) console.log(`[PREFLIGHT:PY] ${line}`);
    });
    child.on('error', (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`Preflight failed to start: ${err.message}`));
    });
    child.on('exit', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code !== 0) {
        return reject(
          new Error(`Preflight exited ${code}: ${(stderr || stdout).slice(-500)}`),
        );
      }
      try {
        const m = stdout.match(/\{[\s\S]*\}/);
        if (!m) return reject(new Error('No JSON from preflight'));
        resolve(JSON.parse(m[0]) as QualityReport);
      } catch (e) {
        reject(new Error(`Preflight parse failed: ${e}`));
      }
    });
  });
}

// ============================================================================
// Preview checks (pre-upload, no video row required)
// ============================================================================

export interface PreviewInput {
  frames: Buffer[];
  width: number;
  height: number;
  durationS: number;
}

export interface PreviewResult {
  pass: boolean;
  issues: Issue[];
}

export async function runPreviewChecks(input: PreviewInput): Promise<PreviewResult> {
  await fs.mkdir(TEMP_DIR, { recursive: true });
  const dir = path.join(
    TEMP_DIR,
    `preview_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
  );
  await fs.mkdir(dir, { recursive: true });
  try {
    await Promise.all(
      input.frames.map((buf, i) => fs.writeFile(path.join(dir, `frame_${i}.jpg`), buf)),
    );
    const issues = await runPreviewCli(dir, input);
    return { pass: !issues.some((i) => i.tier === 'block'), issues };
  } finally {
    await fs.rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}

function runPreviewCli(
  dir: string,
  meta: { width: number; height: number; durationS: number },
): Promise<Issue[]> {
  return new Promise((resolve, reject) => {
    const args = [
      'run',
      'rallycut',
      'preview-check',
      dir,
      '--width',
      String(meta.width),
      '--height',
      String(meta.height),
      '--duration-s',
      String(meta.durationS),
      '--json',
      '--quiet',
    ];
    const child = spawn('uv', args, {
      cwd: ANALYSIS_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env },
    });
    let stdout = '';
    let stderr = '';
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try {
        child.kill();
      } catch {
        /* ignore */
      }
      reject(new Error('Preview timed out'));
    }, 20_000);
    child.stdout?.on('data', (d: Buffer) => {
      stdout += d.toString();
    });
    child.stderr?.on('data', (d: Buffer) => {
      stderr += d.toString();
    });
    child.on('error', (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`preview-check failed to start: ${err.message}`));
    });
    child.on('exit', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code !== 0) {
        return reject(new Error(`preview-check exited ${code}: ${stderr.slice(-500)}`));
      }
      try {
        const m = stdout.match(/\{[\s\S]*\}/);
        if (!m) return reject(new Error('No JSON from preview-check'));
        const parsed = JSON.parse(m[0]);
        resolve((parsed.issues ?? []) as Issue[]);
      } catch (e) {
        reject(e);
      }
    });
  });
}

async function downloadFromS3(s3Key: string, destPath: string): Promise<void> {
  const url = await generateDownloadUrl(s3Key);
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`S3 download failed: ${response.status}`);
  await pipeline(Readable.fromWeb(response.body as never), createWriteStream(destPath));
}
