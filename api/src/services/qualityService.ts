/**
 * Quality assessment service — video quality check + court auto-detection.
 *
 * Runs `rallycut assess-quality` and `rallycut detect-court` CLI commands
 * in parallel. Returns quality warnings + court detection result.
 * Auto-saves court calibration if confidence > 0.7.
 */

import { spawn } from 'child_process';
import { createWriteStream } from 'fs';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { fileURLToPath } from 'url';

import { Prisma } from '@prisma/client';
import { prisma } from '../lib/prisma.js';
import { generateDownloadUrl } from '../lib/s3.js';
import { NotFoundError } from '../middleware/errorHandler.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ANALYSIS_DIR = path.resolve(__dirname, '../../../analysis');
const TEMP_DIR = path.join(os.tmpdir(), 'rallycut-quality');

interface QualityAssessmentResult {
  expectedQuality: number;
  warnings: string[];
  cameraDistance: {
    avgBboxHeight: number;
    category: string;
  };
  sceneComplexity: {
    avgPeople: number;
    category: string;
  };
  heightVariance: number;
  resolution: { width: number; height: number };
  videoInfo: {
    width: number;
    height: number;
    fps: number;
    totalFrames: number;
  };
}

interface CourtDetectionResult {
  corners: Array<{ x: number; y: number }>;
  confidence: number;
  detected_lines: Array<{
    label: string;
    p1: { x: number; y: number };
    p2: { x: number; y: number };
    support: number;
    angle_deg: number;
  }>;
  warnings: string[];
}

export interface AssessQualityResponse {
  quality: {
    expectedQuality: number;
    warnings: string[];
    cameraDistance: string;
    sceneComplexity: string;
    resolution: { width: number; height: number };
  };
  courtDetection: {
    detected: boolean;
    confidence: number;
    autoSaved: boolean;
    corners?: Array<{ x: number; y: number }>;
  };
}

/**
 * Run quality assessment and court detection for a video.
 * Returns quality warnings + court detection result.
 * Auto-saves court calibration if confidence > 0.7.
 */
export async function assessVideoQuality(
  videoId: string,
  userId: string,
): Promise<AssessQualityResponse> {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }

  // Download video to local temp file for CLI analysis.
  // Quality assessment uses proxy (720p is fine for resolution/scene checks).
  // Court detection uses original — proxy resolution loses keypoint detail,
  // dropping confidence below the auto-save threshold.
  await fs.mkdir(TEMP_DIR, { recursive: true });
  const suffix = `_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const ext = path.extname(video.filename || '.mp4');
  const qualityVideoPath = path.join(TEMP_DIR, `quality_${videoId}${suffix}${ext}`);

  const proxyKey = video.proxyS3Key ?? video.s3Key;
  const originalKey = video.s3Key;
  const needsSeparateOriginal = originalKey && originalKey !== proxyKey;
  const courtVideoPath = needsSeparateOriginal
    ? path.join(TEMP_DIR, `court_${videoId}${suffix}${ext}`)
    : qualityVideoPath;

  try {
    const downloads = [downloadFromS3(proxyKey, qualityVideoPath)];
    if (needsSeparateOriginal) {
      downloads.push(downloadFromS3(originalKey, courtVideoPath));
    }
    await Promise.all(downloads);
  } catch (downloadErr) {
    throw new Error(`Failed to download video for quality check: ${downloadErr}`);
  }

  // Run both CLI commands in parallel
  let qualityResult: PromiseSettledResult<QualityAssessmentResult>;
  let courtResult: PromiseSettledResult<CourtDetectionResult>;

  try {
    [qualityResult, courtResult] = await Promise.allSettled([
      runQualityAssessmentCli(qualityVideoPath),
      runCourtDetectionCli(courtVideoPath),
    ]);
  } finally {
    // Cleanup temp files
    await fs.unlink(qualityVideoPath).catch(() => {});
    if (needsSeparateOriginal) {
      await fs.unlink(courtVideoPath).catch(() => {});
    }
  }

  const quality = qualityResult.status === 'fulfilled' ? qualityResult.value : null;
  const court = courtResult.status === 'fulfilled' ? courtResult.value : null;

  if (qualityResult.status === 'rejected') {
    console.error('[QUALITY] Quality assessment failed:', qualityResult.reason);
  }
  if (courtResult.status === 'rejected') {
    console.error('[QUALITY] Court detection failed:', courtResult.reason);
  }

  // Build high-confidence warnings only
  const warnings: string[] = [];
  if (quality) {
    if (quality.cameraDistance.category === 'far' && quality.cameraDistance.avgBboxHeight < 0.12) {
      warnings.push('Camera is very far from the court — player tracking may be less accurate');
    }
    if (quality.sceneComplexity.avgPeople > 8) {
      warnings.push('Crowded scene detected — spectators may interfere with tracking');
    }
    if (quality.resolution?.width < 1280) {
      warnings.push('Low video resolution — recording in 1080p or higher gives better results');
    }
    if (quality.sceneComplexity.avgPeople < 2) {
      warnings.push('Very few people detected — this may not be a volleyball match');
    }
  }

  // Auto-save court calibration if confident, but never overwrite manual calibration.
  // Uses transaction to prevent TOCTOU race between read and write.
  let courtAutoSaved = false;
  const cornersUsable = court && court.corners.length === 4 && areCornersReasonable(court.corners);
  if (cornersUsable && court.confidence > 0.7) {
    courtAutoSaved = await prisma.$transaction(async (tx) => {
      const currentVideo = await tx.video.findUnique({
        where: { id: videoId },
        select: { courtCalibrationJson: true, courtCalibrationSource: true },
      });
      if (currentVideo?.courtCalibrationSource === 'manual') {
        console.log(`[QUALITY] Skipped auto-save for video ${videoId} — manual calibration exists`);
        return false;
      }
      await tx.video.update({
        where: { id: videoId },
        data: {
          courtCalibrationJson: court.corners as unknown as Prisma.InputJsonValue,
          courtCalibrationSource: 'auto',
        },
      });
      console.log(`[QUALITY] Auto-saved court calibration for video ${videoId} (confidence: ${court.confidence.toFixed(2)})`);
      return true;
    });
  } else if (court) {
    // Detection ran but didn't meet the quality bar — clear any stale auto
    // calibration that a previous (less strict) run may have saved.
    // Manual calibrations are always preserved.
    const cleared = await prisma.$transaction(async (tx) => {
      const currentVideo = await tx.video.findUnique({
        where: { id: videoId },
        select: { courtCalibrationJson: true, courtCalibrationSource: true },
      });
      if (!currentVideo?.courtCalibrationJson || currentVideo.courtCalibrationSource === 'manual') {
        return false;
      }
      await tx.video.update({
        where: { id: videoId },
        data: { courtCalibrationJson: Prisma.DbNull, courtCalibrationSource: null },
      });
      console.log(
        `[QUALITY] Cleared stale auto-calibration for video ${videoId} ` +
          `(new detection confidence: ${court.confidence.toFixed(2)})`,
      );
      return true;
    });
    if (cleared) {
      console.log(`[QUALITY] Video ${videoId} will use per-rally auto-detection during tracking`);
    }
  }

  // Save quality assessment to characteristicsJson (merge with existing)
  if (quality) {
    const existing = (video.characteristicsJson as Record<string, unknown>) ?? {};
    await prisma.video.update({
      where: { id: videoId },
      data: {
        characteristicsJson: {
          ...existing,
          qualityAssessment: {
            expectedQuality: quality.expectedQuality,
            warnings,
            cameraDistance: quality.cameraDistance,
            sceneComplexity: quality.sceneComplexity,
            resolution: quality.resolution,
          },
        } as unknown as Prisma.InputJsonValue,
      },
    });
  }

  return {
    quality: {
      expectedQuality: quality?.expectedQuality ?? 0.5,
      warnings,
      cameraDistance: quality?.cameraDistance.category ?? 'unknown',
      sceneComplexity: quality?.sceneComplexity.category ?? 'unknown',
      resolution: quality?.resolution ?? { width: 0, height: 0 },
    },
    courtDetection: {
      detected: (court?.confidence ?? 0) > 0.7,
      confidence: court?.confidence ?? 0,
      autoSaved: courtAutoSaved,
      corners: courtAutoSaved && court ? court.corners : undefined,
    },
  };
}

/**
 * Get unified pipeline status by reading existing DB state.
 */
export async function getAnalysisPipelineStatus(
  videoId: string,
  userId: string,
) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
    include: {
      rallies: {
        where: { status: 'CONFIRMED' },
        select: { id: true },
      },
    },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }

  // Quality from characteristicsJson
  const characteristics = video.characteristicsJson as Record<string, unknown> | null;
  const qualityAssessment = characteristics?.qualityAssessment as {
    expectedQuality: number;
    warnings: string[];
  } | undefined;

  // Detection status
  const detectionStatus = video.status === 'DETECTED'
    ? 'completed'
    : video.status === 'DETECTING'
      ? 'processing'
      : video.status === 'UPLOADED'
        ? 'idle'
        : video.status;

  // Batch tracking status (treat stale PROCESSING jobs as failed)
  const STALE_JOB_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes without progress
  let batchJob = await prisma.batchTrackingJob.findFirst({
    where: { videoId },
    orderBy: { createdAt: 'desc' },
  });
  if (
    batchJob &&
    (batchJob.status === 'PROCESSING' || batchJob.status === 'PENDING') &&
    Date.now() - batchJob.lastProgressAt.getTime() > STALE_JOB_TIMEOUT_MS
  ) {
    // Auto-expire stale job
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
    where: {
      videoId,
      status: 'CONFIRMED',
      playerTrack: { status: 'COMPLETED' },
    },
  });

  // Match analysis and stats
  const hasMatchAnalysis = video.matchAnalysisJson !== null;
  const hasMatchStats = video.matchStatsJson !== null;

  return {
    quality: qualityAssessment
      ? {
          expectedQuality: qualityAssessment.expectedQuality,
          warnings: qualityAssessment.warnings,
          courtDetected: video.courtCalibrationJson !== null,
        }
      : null,
    detection: {
      status: detectionStatus,
      ralliesFound: video.rallies.length,
    },
    tracking: {
      status: batchJob?.status?.toLowerCase() ?? 'idle',
      completed: batchJob?.completedRallies ?? trackedRallies,
      total: batchJob?.totalRallies ?? video.rallies.length,
      failed: batchJob?.failedRallies ?? 0,
    },
    matchAnalysis: { available: hasMatchAnalysis },
    matchStats: { available: hasMatchStats },
  };
}


/**
 * Save player matching ground truth labels.
 */
export async function savePlayerMatchingGt(
  videoId: string,
  userId: string,
  gt: { rallies: Record<string, Record<string, number>>; sideSwitches: number[] },
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

/**
 * Reject degenerate corners before auto-saving. Defense-in-depth: even if
 * confidence is honest, don't persist corners that are wildly off-screen
 * (e.g. (-0.2, 1.17)) — they produce near-full-frame ROIs that provide
 * no filtering value.
 */
function areCornersReasonable(corners: Array<{ x: number; y: number }>): boolean {
  // All corners must be within a generous margin of the frame.
  // Legitimate off-screen corners (low camera angle) are at most ~0.15
  // beyond the frame edge; anything beyond 0.3 is degenerate.
  const MAX_MARGIN = 0.3;
  for (const c of corners) {
    if (c.x < -MAX_MARGIN || c.x > 1 + MAX_MARGIN || c.y < -MAX_MARGIN || c.y > 1 + MAX_MARGIN) {
      console.log(
        `[QUALITY] Corner (${c.x.toFixed(3)}, ${c.y.toFixed(3)}) too far off-screen — skipping auto-save`,
      );
      return false;
    }
  }
  return true;
}

// ============================================================================
// CLI runners
// ============================================================================

function runQualityAssessmentCli(videoPath: string): Promise<QualityAssessmentResult> {
  return new Promise((resolve, reject) => {
    const args = ['run', 'rallycut', 'assess-quality', videoPath, '--json', '--quiet'];

    console.log(`[QUALITY] Running: uv ${args.join(' ')}`);

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
      try { child.kill(); } catch { /* ignore */ }
      reject(new Error('Quality assessment timed out'));
    }, 60000);

    child.stdout?.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    child.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
      const line = data.toString().trim();
      if (line) console.log(`[QUALITY:PY] ${line}`);
    });

    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`Quality assessment failed to start: ${error.message}`));
    });

    child.on('exit', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`Quality assessment exited with code ${code}: ${(stderr || stdout).slice(-500)}`));
        return;
      }
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          reject(new Error(`No JSON found in quality assessment output: ${stdout.slice(0, 200)}`));
          return;
        }
        const result = JSON.parse(jsonMatch[0]);
        resolve(result);
      } catch (e) {
        reject(new Error(`Failed to parse quality assessment output: ${e}`));
      }
    });
  });
}

function runCourtDetectionCli(videoPath: string): Promise<CourtDetectionResult> {
  return new Promise((resolve, reject) => {
    const args = ['run', 'rallycut', 'detect-court', videoPath, '--json', '--quiet'];

    console.log(`[QUALITY] Running: uv ${args.join(' ')}`);

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
      try { child.kill(); } catch { /* ignore */ }
      reject(new Error('Court detection timed out'));
    }, 60000);

    child.stdout?.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    child.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
      const line = data.toString().trim();
      if (line) console.log(`[QUALITY:COURT:PY] ${line}`);
    });

    child.on('error', (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`Court detection failed to start: ${error.message}`));
    });

    child.on('exit', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`Court detection exited with code ${code}: ${(stderr || stdout).slice(-500)}`));
        return;
      }
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          reject(new Error(`No JSON found in court detection output: ${stdout.slice(0, 200)}`));
          return;
        }
        const result = JSON.parse(jsonMatch[0]);
        resolve(result);
      } catch (e) {
        reject(new Error(`Failed to parse court detection output: ${e}`));
      }
    });
  });
}

/**
 * Download a file from S3 to local disk using streaming.
 */
async function downloadFromS3(s3Key: string, destPath: string): Promise<void> {
  const downloadUrl = await generateDownloadUrl(s3Key);
  const response = await fetch(downloadUrl);
  if (!response.ok) {
    throw new Error(`Failed to download from S3: ${response.status}`);
  }
  if (!response.body) {
    throw new Error('No response body from S3');
  }
  const writeStream = createWriteStream(destPath);
  await pipeline(Readable.fromWeb(response.body as never), writeStream);
}
