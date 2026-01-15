/**
 * Ball tracking service for automatic camera keyframe generation.
 *
 * Orchestrates ball tracking pipeline:
 * 1. Extract rally video segment
 * 2. Run Python ball tracker (ONNX inference)
 * 3. Convert positions to camera keyframes
 */

import { spawn } from 'child_process';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

import type { AspectRatio, BallTrackStatus, CameraEasing } from '@prisma/client';

import { env } from '../config/env.js';
import { generateDownloadUrl } from '../lib/s3.js';
import { prisma } from '../lib/prisma.js';
import { ConflictError, NotFoundError, ValidationError } from '../middleware/errorHandler.js';
import { convertBallTrackingToKeyframes, type BallPosition, type CameraKeyframe, VERTICAL_CONFIG } from '../utils/ballToKeyframes.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Temp directory for video segments
const TEMP_DIR = path.join(os.tmpdir(), 'rallycut-ball-tracking');

// Max duration for synchronous processing (30 seconds)
const SYNC_MAX_DURATION_MS = 30000;

interface TrackBallOptions {
  aspectRatio?: AspectRatio;
  generateKeyframes?: boolean;
  zoom?: number; // For 16:9 with zoom > 1, enables ball-following camera
}

interface TrackBallResult {
  status: 'completed' | 'processing' | 'failed';
  ballTrack?: {
    id: string;
    detectionRate: number;
    frameCount: number;
  };
  keyframes?: Array<{
    timeOffset: number;
    positionX: number;
    positionY: number;
    zoom: number;
    easing: CameraEasing;
  }>;
  quality?: {
    coverage: number;
    averageConfidence: number;
    isUsable: boolean;
    recommendation: string;
  };
  error?: string;
}

/**
 * Track ball positions in a rally and optionally generate camera keyframes.
 */
export async function trackBallForRally(
  rallyId: string,
  options: TrackBallOptions = {}
): Promise<TrackBallResult> {
  const { aspectRatio = 'ORIGINAL', generateKeyframes = true, zoom } = options;

  // Build custom config if zoom is provided (for 16:9 with zoom > 1)
  const customConfig = zoom ? { baseZoom: zoom } : undefined;

  // Fetch rally with video
  const rally = await prisma.rally.findUnique({
    where: { id: rallyId },
    include: {
      video: true,
      ballTrack: true,
    },
  });

  if (!rally) {
    throw new NotFoundError('Rally', rallyId);
  }

  // Check for existing in-progress tracking - but allow retry if stuck for >60 seconds
  // (processing a 30-second rally should take at most ~30 seconds)
  if (rally.ballTrack?.status === 'PROCESSING') {
    const stuckThresholdMs = 60 * 1000; // 60 seconds
    const createdAt = rally.ballTrack.createdAt.getTime();
    const isStuck = Date.now() - createdAt > stuckThresholdMs;

    if (!isStuck) {
      // Genuine in-progress tracking, return status
      return {
        status: 'processing',
        ballTrack: {
          id: rally.ballTrack.id,
          detectionRate: 0,
          frameCount: 0,
        },
      };
    }
    // Stuck record - will be reset by upsert below
    console.log(`[BALL_TRACK] Resetting stuck PROCESSING record for rally ${rallyId}`);
  }

  // If tracking already completed, return existing keyframes without re-running
  if (rally.ballTrack?.status === 'COMPLETED' && rally.ballTrack.positionsJson) {
    const durationMs = rally.endMs - rally.startMs;
    const positions = rally.ballTrack.positionsJson as unknown as BallPosition[];
    // Estimate fps from frame count and duration (default to 30 if not available)
    const fps = rally.ballTrack.frameCount && durationMs > 0
      ? (rally.ballTrack.frameCount / (durationMs / 1000))
      : 30;

    console.log(`[BALL_TRACK] *** Regenerating keyframes from ${positions.length} stored positions (aspectRatio=${aspectRatio}) ***`);
    const result = convertBallTrackingToKeyframes(positions, durationMs, fps, aspectRatio, customConfig);

    console.log(`[BALL_TRACK] Returning ${result.keyframes.length} keyframes for rally ${rallyId}`);

    return {
      status: 'completed',
      ballTrack: {
        id: rally.ballTrack.id,
        detectionRate: rally.ballTrack.detectionRate ?? 0,
        frameCount: rally.ballTrack.frameCount ?? 0,
      },
      keyframes: generateKeyframes ? result.keyframes : [],
      quality: result.quality,
    };
  }

  const durationMs = rally.endMs - rally.startMs;

  // Validate rally duration
  if (durationMs <= 0) {
    throw new ValidationError('Rally has invalid duration', {
      startMs: rally.startMs,
      endMs: rally.endMs,
    });
  }

  // Limit to 30 seconds max for performance
  const MAX_DURATION_MS = 30000;
  if (durationMs > MAX_DURATION_MS) {
    throw new ValidationError('Rally too long for auto-tracking (max 30 seconds)', {
      durationMs,
      maxDurationMs: MAX_DURATION_MS,
    });
  }

  // Create or update ball track record (reset timestamps on retry)
  const ballTrack = await prisma.ballTrack.upsert({
    where: { rallyId },
    create: { rallyId, status: 'PROCESSING' },
    update: { status: 'PROCESSING', error: null, completedAt: null, createdAt: new Date() },
  });

  try {
    // For long rallies, we would spawn async - but for now, process synchronously
    if (durationMs > SYNC_MAX_DURATION_MS) {
      console.log(`[BALL_TRACK] Rally ${rallyId} is long (${durationMs}ms), processing synchronously anyway`);
    }

    const result = await runBallTrackingSync(rally, ballTrack.id, aspectRatio, generateKeyframes, customConfig);
    return result;
  } catch (error) {
    // Update ball track status to failed
    await prisma.ballTrack.update({
      where: { id: ballTrack.id },
      data: {
        status: 'FAILED',
        error: error instanceof Error ? error.message : String(error),
      },
    });

    throw error;
  }
}

/**
 * Run ball tracking synchronously.
 */
async function runBallTrackingSync(
  rally: {
    id: string;
    startMs: number;
    endMs: number;
    video: {
      id: string;
      proxyS3Key: string | null;
      s3Key: string;
      width: number | null;
      height: number | null;
    };
  },
  ballTrackId: string,
  aspectRatio: AspectRatio,
  generateKeyframes: boolean,
  customConfig?: { baseZoom: number }
): Promise<TrackBallResult> {
  const startTime = Date.now();
  const durationMs = rally.endMs - rally.startMs;

  // Ensure temp directory exists
  await fs.mkdir(TEMP_DIR, { recursive: true });

  const segmentPath = path.join(TEMP_DIR, `${ballTrackId}.mp4`);
  const outputPath = path.join(TEMP_DIR, `${ballTrackId}.json`);

  try {
    // Get video URL (prefer proxy for faster download)
    const videoKey = rally.video.proxyS3Key ?? rally.video.s3Key;
    const videoUrl = await generateDownloadUrl(videoKey);

    console.log(`[BALL_TRACK] Extracting rally segment for ${rally.id} (aspectRatio: ${aspectRatio})`);
    console.log(`[BALL_TRACK] Video: ${videoKey}`);
    console.log(`[BALL_TRACK] Time range: ${rally.startMs}ms - ${rally.endMs}ms`);

    // Extract rally segment using FFmpeg
    await extractVideoSegment(
      videoUrl,
      rally.startMs / 1000,
      durationMs / 1000,
      segmentPath
    );

    // Verify video segment exists
    const segmentStats = await fs.stat(segmentPath);
    console.log(`[BALL_TRACK] Running ball tracker on ${segmentPath} (${segmentStats.size} bytes)`);

    // Run Python ball tracker
    const trackerOutput = await runBallTracker(segmentPath, outputPath);
    const { positions, fps, frameCount: totalFrames } = trackerOutput;

    console.log(`[BALL_TRACK] Detection complete: ${positions.length} raw positions, fps=${fps}, totalFrames=${totalFrames}`);

    // Generate keyframes (this also deduplicates and calculates quality)
    let keyframes: CameraKeyframe[] = [];
    let quality = {
      coverage: 0,
      averageConfidence: 0,
      isUsable: false,
      recommendation: 'No positions detected',
    };

    if (positions.length > 0) {
      const result = convertBallTrackingToKeyframes(positions, durationMs, fps, aspectRatio, customConfig);
      keyframes = generateKeyframes ? result.keyframes : [];
      quality = result.quality;
      console.log(`[BALL_TRACK] Quality: coverage=${(quality.coverage * 100).toFixed(1)}%, avgConf=${quality.averageConfidence.toFixed(2)}, usable=${quality.isUsable}, keyframes=${result.keyframes.length}`);
    }

    // Use quality.coverage for detection rate (already deduplicated)
    const detectionRate = quality.coverage;

    const processingTimeMs = Date.now() - startTime;

    // Update ball track record
    await prisma.ballTrack.update({
      where: { id: ballTrackId },
      data: {
        status: 'COMPLETED',
        frameCount: totalFrames,
        detectionRate,
        positionsJson: positions as unknown as object[],
        processingTimeMs,
        modelVersion: 'VballNetV1b_seq9_grayscale_best.onnx',
        completedAt: new Date(),
      },
    });

    return {
      status: 'completed',
      ballTrack: {
        id: ballTrackId,
        detectionRate,
        frameCount: totalFrames,
      },
      keyframes,
      quality,
    };
  } finally {
    // Cleanup temp files
    await fs.unlink(segmentPath).catch(() => {});
    await fs.unlink(outputPath).catch(() => {});
  }
}

/**
 * Extract a video segment using FFmpeg with frame-accurate seeking.
 *
 * Uses -ss AFTER -i for accurate seeking (decodes from start).
 * Re-encodes video to ensure frame 0 = exact start time.
 * Slower than fast-seek but necessary for accurate ball tracking sync.
 */
async function extractVideoSegment(
  videoUrl: string,
  startSeconds: number,
  durationSeconds: number,
  outputPath: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    // Frame-accurate extraction:
    // -ss AFTER -i = accurate seek (decodes from start to find exact frame)
    // Re-encode with ultrafast preset for speed
    // This ensures frame 0 of output = exactly startSeconds of input
    const args = [
      '-i', videoUrl,
      '-ss', startSeconds.toString(),
      '-t', durationSeconds.toString(),
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-crf', '23',
      '-c:a', 'aac',
      '-b:a', '128k',
      '-y', // Overwrite
      outputPath,
    ];

    const ffmpeg = spawn('ffmpeg', args, {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stderr = '';
    ffmpeg.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    ffmpeg.on('error', (error) => {
      reject(new Error(`FFmpeg failed to start: ${error.message}`));
    });

    ffmpeg.on('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`FFmpeg exited with code ${code}: ${stderr.slice(-500)}`));
      }
    });
  });
}

interface BallTrackerOutput {
  positions: BallPosition[];
  fps: number;
  frameCount: number;
}

/**
 * Run the Python ball tracker CLI.
 */
async function runBallTracker(
  videoPath: string,
  outputPath: string
): Promise<BallTrackerOutput> {
  return new Promise((resolve, reject) => {
    const analysisDir = path.resolve(__dirname, '../../../analysis');

    const args = [
      'run',
      'rallycut',
      'track-ball',
      videoPath,
      '--output', outputPath,
      '--quiet',
    ];

    console.log(`[BALL_TRACK] Running: uv ${args.join(' ')}`);
    console.log(`[BALL_TRACK] CWD: ${analysisDir}`);

    const child = spawn('uv', args, {
      cwd: analysisDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        // Pass AWS credentials for model download if needed
        AWS_ACCESS_KEY_ID: env.AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY: env.AWS_SECRET_ACCESS_KEY,
        AWS_REGION: env.AWS_REGION,
      },
    });

    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', (data: Buffer) => {
      stdout += data.toString();
      // Log stdout for debugging
      const line = data.toString().trim();
      if (line) {
        console.log(`[BALL_TRACK:PY:OUT] ${line}`);
      }
    });

    child.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
      // Log progress
      const line = data.toString().trim();
      if (line) {
        console.log(`[BALL_TRACK:PY] ${line}`);
      }
    });

    child.on('error', (error) => {
      reject(new Error(`Ball tracker failed to start: ${error.message}`));
    });

    child.on('exit', async (code) => {
      if (code !== 0) {
        const output = stderr || stdout; // Python exceptions may go to stdout
        reject(new Error(`Ball tracker exited with code ${code}: ${output.slice(-1000)}`));
        return;
      }

      try {
        // Read output JSON
        const jsonContent = await fs.readFile(outputPath, 'utf-8');
        const result = JSON.parse(jsonContent);

        // Parse positions
        const positions: BallPosition[] = (result.positions || []).map((p: {
          frameNumber: number;
          x: number;
          y: number;
          confidence: number;
        }) => ({
          frameNumber: p.frameNumber,
          x: p.x,
          y: p.y,
          confidence: p.confidence,
        }));

        // Get fps and frame count from tracker output
        const fps = result.videoFps || 30;
        const frameCount = result.frameCount || positions.length;

        console.log(`[BALL_TRACK] Tracker output: fps=${fps}, frameCount=${frameCount}, positions=${positions.length}`);

        resolve({ positions, fps, frameCount });
      } catch (parseError) {
        reject(new Error(`Failed to parse ball tracker output: ${parseError}`));
      }
    });
  });
}

/**
 * Get ball tracking status for a rally.
 */
export async function getBallTrackStatus(rallyId: string, options: { includePositions?: boolean } = {}): Promise<{
  rallyId: string;
  ballTrack: {
    id: string;
    status: BallTrackStatus;
    detectionRate: number | null;
    frameCount: number | null;
    processingTimeMs: number | null;
    error: string | null;
    positions?: Array<{ frameNumber: number; x: number; y: number; confidence: number }>;
  } | null;
}> {
  const rally = await prisma.rally.findUnique({
    where: { id: rallyId },
    include: { ballTrack: true },
  });

  if (!rally) {
    throw new NotFoundError('Rally', rallyId);
  }

  if (!rally.ballTrack) {
    return { rallyId, ballTrack: null };
  }

  const result: {
    id: string;
    status: BallTrackStatus;
    detectionRate: number | null;
    frameCount: number | null;
    processingTimeMs: number | null;
    error: string | null;
    positions?: Array<{ frameNumber: number; x: number; y: number; confidence: number }>;
  } = {
    id: rally.ballTrack.id,
    status: rally.ballTrack.status,
    detectionRate: rally.ballTrack.detectionRate,
    frameCount: rally.ballTrack.frameCount,
    processingTimeMs: rally.ballTrack.processingTimeMs,
    error: rally.ballTrack.error,
  };

  // Include positions if requested (with offset correction for debug overlay)
  if (options.includePositions && rally.ballTrack.positionsJson) {
    const rawPositions = rally.ballTrack.positionsJson as Array<{
      frameNumber: number;
      x: number;
      y: number;
      confidence: number;
    }>;

    // Apply offset corrections so debug overlay matches actual ball position
    const xOffset = VERTICAL_CONFIG.xOffsetCorrection ?? 0;
    const yOffset = VERTICAL_CONFIG.yOffsetCorrection ?? 0;

    console.log(`[BALL_TRACK] Applying offset to ${rawPositions.length} positions: xOffset=${xOffset}, yOffset=${yOffset}`);

    result.positions = rawPositions.map(p => ({
      frameNumber: p.frameNumber,
      x: Math.min(1, Math.max(0, p.x + xOffset)),
      y: Math.min(1, Math.max(0, p.y + yOffset)),
      confidence: p.confidence,
    }));
  }

  return { rallyId, ballTrack: result };
}
