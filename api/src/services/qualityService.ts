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

  // Download video to local temp file for CLI analysis
  await fs.mkdir(TEMP_DIR, { recursive: true });
  const suffix = `_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const videoPath = path.join(TEMP_DIR, `quality_${videoId}${suffix}${path.extname(video.filename || '.mp4')}`);

  try {
    await downloadFromS3(video.proxyS3Key ?? video.s3Key, videoPath);
  } catch (downloadErr) {
    throw new Error(`Failed to download video for quality check: ${downloadErr}`);
  }

  // Run both CLI commands in parallel
  let qualityResult: PromiseSettledResult<QualityAssessmentResult>;
  let courtResult: PromiseSettledResult<CourtDetectionResult>;

  try {
    [qualityResult, courtResult] = await Promise.allSettled([
      runQualityAssessmentCli(videoPath),
      runCourtDetectionCli(videoPath),
    ]);
  } finally {
    // Cleanup temp file
    await fs.unlink(videoPath).catch(() => {});
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
    if (quality.resolution.width < 1280) {
      warnings.push('Low video resolution — recording in 1080p or higher gives better results');
    }
    if (quality.sceneComplexity.avgPeople < 2) {
      warnings.push('Very few people detected — this may not be a volleyball match');
    }
  }

  // Auto-save court calibration if confident
  let courtAutoSaved = false;
  if (court && court.confidence > 0.7 && court.corners.length === 4) {
    await prisma.video.update({
      where: { id: videoId },
      data: {
        courtCalibrationJson: court.corners as unknown as Prisma.InputJsonValue,
      },
    });
    courtAutoSaved = true;
    console.log(`[QUALITY] Auto-saved court calibration for video ${videoId} (confidence: ${court.confidence.toFixed(2)})`);
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

  // Batch tracking status
  const batchJob = await prisma.batchTrackingJob.findFirst({
    where: { videoId },
    orderBy: { createdAt: 'desc' },
  });

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
 * Save player names to matchAnalysisJson.
 */
export async function savePlayerNames(
  videoId: string,
  userId: string,
  names: Record<string, string>,
) {
  const video = await prisma.video.findFirst({
    where: { id: videoId, userId, deletedAt: null },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }

  const existing = (video.matchAnalysisJson as Record<string, unknown>) ?? {};
  await prisma.video.update({
    where: { id: videoId },
    data: {
      matchAnalysisJson: {
        ...existing,
        playerNames: names,
      } as unknown as Prisma.InputJsonValue,
    },
  });

  return { success: true };
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
        const result = JSON.parse(stdout.trim());
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
        const result = JSON.parse(stdout.trim());
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
