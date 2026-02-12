/**
 * Player tracking service for debug visualization.
 *
 * Runs YOLOv8n + ByteTrack to detect and track players in rally segments.
 * Returns raw tracking data for frontend overlay visualization.
 */

import { spawn } from 'child_process';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

import { env } from '../config/env.js';
import { generateDownloadUrl } from '../lib/s3.js';
import { prisma } from '../lib/prisma.js';
import { ForbiddenError, NotFoundError, ValidationError } from '../middleware/errorHandler.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Temp directory for video segments
const TEMP_DIR = path.join(os.tmpdir(), 'rallycut-player-tracking');

// Max duration for synchronous processing (30 seconds)
const MAX_DURATION_MS = 30000;

// Position format for frontend overlay
interface PlayerPosition {
  frameNumber: number;
  trackId: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

// Ball phase detection
interface BallPhase {
  phase: string;  // "serve", "attack", "defense", "transition", "unknown"
  frameStart: number;
  frameEnd: number;
  velocity: number;
  ballX: number;
  ballY: number;
}

// Server detection info
interface ServerInfo {
  trackId: number;
  confidence: number;
  serveFrame: number;
  serveVelocity: number;
  isNearCourt: boolean;
}

// Ball position for trajectory overlay
interface BallPosition {
  frameNumber: number;
  x: number;
  y: number;
  confidence: number;
}

export interface TrackPlayersResult {
  status: 'completed' | 'failed';
  frameCount?: number;
  fps?: number;  // Actual fps from tracked video
  detectionRate?: number;
  avgConfidence?: number;
  avgPlayerCount?: number;
  uniqueTrackCount?: number;
  processingTimeMs?: number;
  error?: string;
  // Court split Y for debug overlay (horizontal line, camera behind baseline)
  courtSplitY?: number;
  primaryTrackIds?: number[];
  // Positions included for immediate display
  positions?: PlayerPosition[];
  // Ball phase detection
  ballPhases?: BallPhase[];
  serverInfo?: ServerInfo;
  // Ball positions for trajectory overlay
  ballPositions?: BallPosition[];
}

export interface GetPlayerTrackResult {
  status: 'completed' | 'failed' | 'not_found';
  frameCount?: number;
  fps?: number;
  detectionRate?: number;
  avgConfidence?: number;
  avgPlayerCount?: number;
  uniqueTrackCount?: number;
  courtSplitY?: number;
  primaryTrackIds?: number[];
  positions?: PlayerPosition[];
  ballPhases?: BallPhase[];
  serverInfo?: ServerInfo;
  ballPositions?: BallPosition[];
  error?: string;
}

/**
 * Extract video segment using FFmpeg.
 */
async function extractVideoSegment(
  videoUrl: string,
  startSeconds: number,
  durationSeconds: number,
  outputPath: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const args = [
      '-ss', startSeconds.toString(), // Input-level seeking (fast, uses HTTP range requests)
      '-i', videoUrl,
      '-t', durationSeconds.toString(),
      '-c:v', 'libx264',
      '-preset', 'fast',
      '-crf', '18', // Visually lossless — preserves detail for VballNet ball detection
      '-an', // No audio needed for player tracking
      '-y',
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

interface PlayerTrackerOutput {
  positions: PlayerPosition[];
  rawPositions?: PlayerPosition[];  // Raw positions before filtering (for param tuning)
  frameCount: number;
  fps: number;
  detectionRate: number;
  avgConfidence: number;
  avgPlayerCount: number;
  uniqueTrackCount: number;
  courtSplitY?: number;
  primaryTrackIds?: number[];
  ballPhases?: BallPhase[];
  serverInfo?: ServerInfo;
  ballPositions?: BallPosition[];
}

/**
 * Run the Python player tracker CLI.
 */
async function runPlayerTracker(
  videoPath: string,
  outputPath: string,
  calibrationCorners?: CalibrationCorner[],
): Promise<PlayerTrackerOutput> {
  return new Promise((resolve, reject) => {
    const analysisDir = path.resolve(__dirname, '../../../analysis');

    const args = [
      'run',
      'rallycut',
      'track-players',
      videoPath,
      '--output', outputPath,
      // Note: Not using --quiet so we can see filter logs
    ];

    // Pass calibration corners if available
    if (calibrationCorners && calibrationCorners.length === 4) {
      args.push('--calibration', JSON.stringify(calibrationCorners));
    }

    console.log(`[PLAYER_TRACK] Running: uv ${args.join(' ')}`);
    console.log(`[PLAYER_TRACK] CWD: ${analysisDir}`);

    const child = spawn('uv', args, {
      cwd: analysisDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        AWS_ACCESS_KEY_ID: env.AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY: env.AWS_SECRET_ACCESS_KEY,
        AWS_REGION: env.AWS_REGION,
      },
    });

    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', (data: Buffer) => {
      stdout += data.toString();
      const line = data.toString().trim();
      if (line) {
        console.log(`[PLAYER_TRACK:PY:OUT] ${line}`);
      }
    });

    child.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
      const line = data.toString().trim();
      if (line) {
        console.log(`[PLAYER_TRACK:PY] ${line}`);
      }
    });

    child.on('error', (error) => {
      reject(new Error(`Player tracker failed to start: ${error.message}`));
    });

    child.on('exit', async (code) => {
      if (code !== 0) {
        const output = stderr || stdout;
        reject(new Error(`Player tracker exited with code ${code}: ${output.slice(-1000)}`));
        return;
      }

      try {
        // Check if output file exists
        try {
          await fs.access(outputPath);
        } catch {
          const output = stderr || stdout || 'No output';
          reject(new Error(`Player tracker completed but no output file. Stderr: ${output.slice(-500)}`));
          return;
        }

        // Read output JSON
        const jsonContent = await fs.readFile(outputPath, 'utf-8');
        const result = JSON.parse(jsonContent);

        // Flat positions list with frameNumber, trackId, x, y, width, height, confidence
        const positions: PlayerPosition[] = (result.positions || []).map((p: PlayerPosition) => ({
          frameNumber: p.frameNumber,
          trackId: p.trackId,
          x: p.x,
          y: p.y,
          width: p.width,
          height: p.height,
          confidence: p.confidence,
        }));

        // Calculate stats
        const frameCount = result.frameCount || 0;
        const fps = result.videoFps || 30;
        const framesWithDetections = new Set(positions.map(p => p.frameNumber)).size;
        const detectionRate = result.detectionRate ?? (framesWithDetections / Math.max(1, frameCount));
        const avgConfidence = positions.length > 0
          ? positions.reduce((sum, p) => sum + p.confidence, 0) / positions.length
          : 0;
        const avgPlayerCount = result.avgPlayersPerFrame ?? (framesWithDetections > 0
          ? positions.length / framesWithDetections
          : 0);
        const uniqueTrackCount = result.uniqueTrackCount ?? new Set(positions.map(p => p.trackId).filter(id => id >= 0)).size;

        // Court split for debug overlay
        const courtSplitY = result.courtSplitY as number | undefined;
        const primaryTrackIds = result.primaryTrackIds as number[] | undefined;

        // Filter method used
        const filterMethod = result.filterMethod as string | undefined;

        // Ball phase detection
        const ballPhases: BallPhase[] | undefined = result.ballPhases?.map((bp: BallPhase) => ({
          phase: bp.phase,
          frameStart: bp.frameStart,
          frameEnd: bp.frameEnd,
          velocity: bp.velocity,
          ballX: bp.ballX,
          ballY: bp.ballY,
        }));

        // Server detection
        const serverInfo: ServerInfo | undefined = result.serverInfo ? {
          trackId: result.serverInfo.trackId,
          confidence: result.serverInfo.confidence,
          serveFrame: result.serverInfo.serveFrame,
          serveVelocity: result.serverInfo.serveVelocity,
          isNearCourt: result.serverInfo.isNearCourt,
        } : undefined;

        // Ball positions for trajectory overlay
        const ballPositions: BallPosition[] | undefined = result.ballPositions?.map((bp: BallPosition) => ({
          frameNumber: bp.frameNumber,
          x: bp.x,
          y: bp.y,
          confidence: bp.confidence,
        }));

        // Raw positions before filtering (for parameter tuning)
        const rawPositions: PlayerPosition[] | undefined = result.rawPositions?.map((p: PlayerPosition) => ({
          frameNumber: p.frameNumber,
          trackId: p.trackId,
          x: p.x,
          y: p.y,
          width: p.width,
          height: p.height,
          confidence: p.confidence,
        }));

        console.log(`[PLAYER_TRACK] Output: frameCount=${frameCount}, detectionRate=${detectionRate.toFixed(2)}, avgPlayers=${avgPlayerCount.toFixed(1)}, tracks=${uniqueTrackCount}, positions=${positions.length}, rawPositions=${rawPositions?.length ?? 0}, ballPositions=${ballPositions?.length ?? 0}`);
        console.log(`[PLAYER_TRACK] Filter method: ${filterMethod ?? 'none (filtering may have failed)'}`);
        if (courtSplitY !== undefined) {
          console.log(`[PLAYER_TRACK] Court split enabled: y=${courtSplitY.toFixed(3)}, primary tracks: ${primaryTrackIds?.join(', ') ?? 'none'}`);
        } else {
          console.log(`[PLAYER_TRACK] Warning: No court split detected - two-team filtering may not be working`);
        }
        if (ballPhases?.length) {
          const phaseCounts: Record<string, number> = {};
          for (const bp of ballPhases) {
            phaseCounts[bp.phase] = (phaseCounts[bp.phase] || 0) + 1;
          }
          console.log(`[PLAYER_TRACK] Ball phases: ${JSON.stringify(phaseCounts)}`);
        }
        if (serverInfo && serverInfo.trackId >= 0) {
          console.log(`[PLAYER_TRACK] Server detected: track #${serverInfo.trackId} (confidence: ${(serverInfo.confidence * 100).toFixed(0)}%)`);
        }

        resolve({
          positions,
          rawPositions,
          frameCount,
          fps,
          detectionRate,
          avgConfidence,
          avgPlayerCount,
          uniqueTrackCount,
          courtSplitY,
          primaryTrackIds,
          ballPhases,
          serverInfo,
          ballPositions,
        });
      } catch (parseError) {
        reject(new Error(`Failed to parse player tracker output: ${parseError}`));
      }
    });
  });
}

/**
 * Get existing player tracking data for a rally.
 */
export async function getPlayerTrack(
  rallyId: string,
  userId: string,
): Promise<GetPlayerTrackResult> {
  // Fetch rally with video and player track
  const rally = await prisma.rally.findUnique({
    where: { id: rallyId },
    include: {
      video: true,
      playerTrack: true,
    },
  });

  if (!rally) {
    throw new NotFoundError('Rally', rallyId);
  }

  if (rally.video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to view player tracking for this rally');
  }

  if (!rally.playerTrack || rally.playerTrack.status !== 'COMPLETED') {
    return { status: 'not_found' };
  }

  const track = rally.playerTrack;
  return {
    status: 'completed',
    frameCount: track.frameCount ?? undefined,
    fps: track.fps ?? undefined,
    detectionRate: track.detectionRate ?? undefined,
    avgConfidence: track.avgConfidence ?? undefined,
    avgPlayerCount: track.avgPlayerCount ?? undefined,
    uniqueTrackCount: track.uniqueTrackCount ?? undefined,
    courtSplitY: track.courtSplitY ?? undefined,
    primaryTrackIds: (track.primaryTrackIds as number[] | null) ?? undefined,
    positions: (track.positionsJson as PlayerPosition[] | null) ?? undefined,
    ballPhases: (track.ballPhasesJson as BallPhase[] | null) ?? undefined,
    serverInfo: (track.serverInfoJson as ServerInfo | null) ?? undefined,
    ballPositions: (track.ballPositionsJson as BallPosition[] | null) ?? undefined,
  };
}

// Calibration corner from frontend
interface CalibrationCorner {
  x: number;
  y: number;
}

/**
 * Track players in a rally video segment.
 */
export async function trackPlayersForRally(
  rallyId: string,
  userId: string,
  calibrationCorners?: CalibrationCorner[],
): Promise<TrackPlayersResult> {
  const startTime = Date.now();

  // Fetch rally with video
  const rally = await prisma.rally.findUnique({
    where: { id: rallyId },
    include: { video: true },
  });

  if (!rally) {
    throw new NotFoundError('Rally', rallyId);
  }

  if (rally.video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to track players for this rally');
  }

  // Check duration limit
  const durationMs = rally.endMs - rally.startMs;
  if (durationMs > MAX_DURATION_MS) {
    throw new ValidationError(`Rally duration (${(durationMs / 1000).toFixed(1)}s) exceeds maximum of ${MAX_DURATION_MS / 1000}s for player tracking`);
  }

  // Ensure temp directory exists
  await fs.mkdir(TEMP_DIR, { recursive: true });

  const segmentPath = path.join(TEMP_DIR, `rally_${rallyId}_segment.mp4`);
  const outputPath = path.join(TEMP_DIR, `rally_${rallyId}_players.json`);

  try {
    // Prefer original quality — proxy video (720p) degrades VballNet ball
    // detection significantly due to lower resolution and re-encoding artifacts.
    // Falls back to proxy if original has been quality-downgraded.
    const videoKey = rally.video.s3Key ?? rally.video.proxyS3Key;
    if (!videoKey) {
      throw new ValidationError('Video has no accessible source');
    }

    const videoUrl = await generateDownloadUrl(videoKey);
    const startSeconds = rally.startMs / 1000;
    const durationSeconds = durationMs / 1000;

    console.log(`[PLAYER_TRACK] Extracting segment: ${startSeconds}s - ${rally.endMs / 1000}s (${durationSeconds}s)`);

    // Extract video segment
    await extractVideoSegment(videoUrl, startSeconds, durationSeconds, segmentPath);

    // Run player tracker (with court filtering enabled by default)
    const trackerResult = await runPlayerTracker(segmentPath, outputPath, calibrationCorners);

    const processingTimeMs = Date.now() - startTime;
    console.log(`[PLAYER_TRACK] Completed in ${processingTimeMs}ms`);

    // Save to database (cast arrays to JSON for Prisma)
    await prisma.playerTrack.upsert({
      where: { rallyId },
      create: {
        rallyId,
        status: 'COMPLETED',
        frameCount: trackerResult.frameCount,
        fps: trackerResult.fps,
        detectionRate: trackerResult.detectionRate,
        avgConfidence: trackerResult.avgConfidence,
        avgPlayerCount: trackerResult.avgPlayerCount,
        uniqueTrackCount: trackerResult.uniqueTrackCount,
        courtSplitY: trackerResult.courtSplitY,
        primaryTrackIds: trackerResult.primaryTrackIds as unknown as number[],
        positionsJson: trackerResult.positions as unknown as object[],
        rawPositionsJson: trackerResult.rawPositions as unknown as object[],
        ballPhasesJson: trackerResult.ballPhases as unknown as object[],
        serverInfoJson: trackerResult.serverInfo as unknown as object,
        ballPositionsJson: trackerResult.ballPositions as unknown as object[],
        processingTimeMs,
        modelVersion: 'yolov8n',
        completedAt: new Date(),
      },
      update: {
        status: 'COMPLETED',
        frameCount: trackerResult.frameCount,
        fps: trackerResult.fps,
        detectionRate: trackerResult.detectionRate,
        avgConfidence: trackerResult.avgConfidence,
        avgPlayerCount: trackerResult.avgPlayerCount,
        uniqueTrackCount: trackerResult.uniqueTrackCount,
        courtSplitY: trackerResult.courtSplitY,
        primaryTrackIds: trackerResult.primaryTrackIds as unknown as number[],
        positionsJson: trackerResult.positions as unknown as object[],
        rawPositionsJson: trackerResult.rawPositions as unknown as object[],
        ballPhasesJson: trackerResult.ballPhases as unknown as object[],
        serverInfoJson: trackerResult.serverInfo as unknown as object,
        ballPositionsJson: trackerResult.ballPositions as unknown as object[],
        processingTimeMs,
        modelVersion: 'yolov8n',
        completedAt: new Date(),
        error: null,
      },
    });

    return {
      status: 'completed',
      frameCount: trackerResult.frameCount,
      fps: trackerResult.fps,
      detectionRate: trackerResult.detectionRate,
      avgConfidence: trackerResult.avgConfidence,
      avgPlayerCount: trackerResult.avgPlayerCount,
      uniqueTrackCount: trackerResult.uniqueTrackCount,
      processingTimeMs,
      courtSplitY: trackerResult.courtSplitY,
      primaryTrackIds: trackerResult.primaryTrackIds,
      positions: trackerResult.positions,
      ballPhases: trackerResult.ballPhases,
      serverInfo: trackerResult.serverInfo,
      ballPositions: trackerResult.ballPositions,
    };
  } catch (error) {
    console.error(`[PLAYER_TRACK] Error:`, error);

    // Save error to database
    await prisma.playerTrack.upsert({
      where: { rallyId },
      create: {
        rallyId,
        status: 'FAILED',
        error: error instanceof Error ? error.message : String(error),
      },
      update: {
        status: 'FAILED',
        error: error instanceof Error ? error.message : String(error),
      },
    });

    return {
      status: 'failed',
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    // Clean up temp files
    try {
      await fs.unlink(segmentPath).catch(() => {});
      await fs.unlink(outputPath).catch(() => {});
    } catch {
      // Ignore cleanup errors
    }
  }
}
