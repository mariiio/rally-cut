/**
 * Player tracking service.
 *
 * Runs YOLO11s + BoT-SORT to detect and track players in rally segments.
 * Supports both single-rally sync tracking and batch async processing.
 */

import { spawn } from 'child_process';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

import { env } from '../config/env.js';
import { generateDownloadUrl } from '../lib/s3.js';
import { Prisma } from '@prisma/client';
import { prisma } from '../lib/prisma.js';
import { ForbiddenError, NotFoundError, ValidationError } from '../middleware/errorHandler.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Temp directory for video segments
const TEMP_DIR = path.join(os.tmpdir(), 'rallycut-player-tracking');

// Max duration for synchronous single-rally processing (30 seconds)
const MAX_DURATION_MS = 30000;

// Max duration for batch processing (60 seconds per rally — more lenient for batch)
const MAX_BATCH_RALLY_DURATION_MS = 60000;

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

// Ball position for trajectory overlay
interface BallPosition {
  frameNumber: number;
  x: number;
  y: number;
  confidence: number;
}

// Contact detection from ball trajectory
interface ContactInfo {
  frame: number;
  ballX: number;
  ballY: number;
  velocity: number;
  directionChangeDeg: number;
  playerTrackId: number;
  playerDistance: number;
  courtSide: string;
  isAtNet: boolean;
  isValidated: boolean;
}

interface ContactsData {
  numContacts: number;
  netY: number;
  rallyStartFrame: number;
  contacts: ContactInfo[];
}

// Action classification from contact sequence
interface ActionInfo {
  action: string;  // "serve", "receive", "set", "attack", "block", "dig", "unknown"
  frame: number;
  ballX: number;
  ballY: number;
  velocity: number;
  playerTrackId: number;
  courtSide: string;
  confidence: number;
  team: string;  // "A" (near court), "B" (far court), or "unknown"
}

interface ActionsData {
  rallyId: string;
  numContacts: number;
  actionSequence: string[];
  actions: ActionInfo[];
  teamAssignments?: Record<string, string>;  // trackId → "A"|"B"
  servingTeam?: string;  // "A" or "B"
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
  // Ball positions for trajectory overlay
  ballPositions?: BallPosition[];
  // Contact detection and action classification
  contacts?: ContactsData;
  actions?: ActionsData;
  qualityReport?: QualityReport;
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
  ballPositions?: BallPosition[];
  contacts?: ContactsData;
  actions?: ActionsData;
  qualityReport?: QualityReport;
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
      '-c', 'copy',  // Stream copy — no re-encoding (~2s vs ~2min)
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

interface QualityReport {
  ballDetectionRate: number;
  ballTrajectorySpread: number;
  avgDetectionsPerFrame: number;
  primaryTrackCount: number;
  trackCreationRate: number;
  trackDestructionRate: number;
  avgTrackLifespanFrames: number;
  idSwitchCount: number;
  colorSplitCount: number;
  uniqueRawTrackCount: number;
  calibrationRecommended: boolean;
  courtDetected: boolean;
  courtConfidence: number;
  trackabilityScore: number;
  suggestions: string[];
}

interface CourtDetectionInsights {
  detected: boolean;
  confidence: number;
  linesFound: number;
  cameraHeight: string;
  lineVisibility: string;
  fittingMethod: string;
  recordingTips: string[];
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
  ballPositions?: BallPosition[];
  contacts?: ContactsData;
  actions?: ActionsData;
  qualityReport?: QualityReport;
  courtDetection?: CourtDetectionInsights;
}

/**
 * Compute adaptive timeout for player tracking based on rally duration.
 * Processing time scales with frame count: YOLO ~2-6 FPS + WASB ~4-33 FPS + post-processing.
 * Returns timeout in milliseconds.
 */
function computeTrackingTimeout(durationSeconds: number): number {
  const BASE_MS = 120_000;                // 2min for model loading + post-processing
  const PER_VIDEO_SECOND_MS = 30_000;     // ~30s per 1s video (60fps stride=1 at ~2-3 FPS YOLO throughput)
  const MIN_TIMEOUT_MS = 5 * 60 * 1000;   // 5min floor (model loading dominates short rallies)

  return Math.max(MIN_TIMEOUT_MS, BASE_MS + durationSeconds * PER_VIDEO_SECOND_MS);
}

/**
 * Run the Python player tracker CLI.
 */
async function runPlayerTracker(
  videoPath: string,
  outputPath: string,
  durationSeconds: number,
  videoFps?: number,
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
      '--actions',  // Enable contact detection + action classification
      // Note: Not using --quiet so we can see filter logs
    ];

    // Auto-stride for high-FPS videos: 60fps at stride=2 ≈ 30fps quality,
    // halves processing time with negligible accuracy loss (IDsw 41→37)
    if (videoFps && videoFps >= 50) {
      args.push('--stride', '2');
    }

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

    // Adaptive timeout: scales with rally duration to prevent false timeouts on long rallies,
    // while still catching actual hangs (e.g., YOLO GPU lock)
    const timeoutMs = computeTrackingTimeout(durationSeconds);
    const timeout = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error(`Player tracker timed out after ${Math.round(timeoutMs / 1000)}s (rally duration: ${durationSeconds.toFixed(1)}s)`));
    }, timeoutMs);
    console.log(`[PLAYER_TRACK] Timeout: ${Math.round(timeoutMs / 1000)}s for ${durationSeconds.toFixed(1)}s rally`);

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
      clearTimeout(timeout);
      reject(new Error(`Player tracker failed to start: ${error.message}`));
    });

    child.on('exit', async (code) => {
      clearTimeout(timeout);
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
        let result;
        try {
          result = JSON.parse(jsonContent);
        } catch {
          reject(new Error(`Player tracker output is not valid JSON: ${jsonContent.slice(0, 200)}`));
          return;
        }

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
        const fps = result.videoFps ?? 30;
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

        // Contact detection and action classification (from --actions flag)
        const contacts: ContactsData | undefined = result.contacts ? {
          numContacts: result.contacts.numContacts,
          netY: result.contacts.netY,
          rallyStartFrame: result.contacts.rallyStartFrame,
          contacts: (result.contacts.contacts || []).map((c: ContactInfo) => ({
            frame: c.frame,
            ballX: c.ballX,
            ballY: c.ballY,
            velocity: c.velocity,
            directionChangeDeg: c.directionChangeDeg,
            playerTrackId: c.playerTrackId,
            playerDistance: c.playerDistance,
            courtSide: c.courtSide,
            isAtNet: c.isAtNet,
            isValidated: c.isValidated,
          })),
        } : undefined;

        const actions: ActionsData | undefined = result.actions ? {
          rallyId: result.actions.rallyId || '',
          numContacts: result.actions.numContacts,
          actionSequence: result.actions.actionSequence || [],
          actions: (result.actions.actions || []).map((a: ActionInfo) => ({
            action: a.action,
            frame: a.frame,
            ballX: a.ballX,
            ballY: a.ballY,
            velocity: a.velocity,
            playerTrackId: a.playerTrackId,
            courtSide: a.courtSide,
            confidence: a.confidence,
            team: a.team || 'unknown',
          })),
          teamAssignments: result.actions.teamAssignments,
          servingTeam: result.actions.servingTeam,
        } : undefined;

        console.log(`[PLAYER_TRACK] Output: frameCount=${frameCount}, detectionRate=${detectionRate.toFixed(2)}, avgPlayers=${avgPlayerCount.toFixed(1)}, tracks=${uniqueTrackCount}, positions=${positions.length}, rawPositions=${rawPositions?.length ?? 0}, ballPositions=${ballPositions?.length ?? 0}`);
        console.log(`[PLAYER_TRACK] Filter method: ${filterMethod ?? 'none (filtering may have failed)'}`);
        if (courtSplitY !== undefined) {
          console.log(`[PLAYER_TRACK] Court split enabled: y=${courtSplitY.toFixed(3)}, primary tracks: ${primaryTrackIds?.join(', ') ?? 'none'}`);
        } else {
          console.log(`[PLAYER_TRACK] Warning: No court split detected - two-team filtering may not be working`);
        }
        if (actions?.actions.length) {
          console.log(`[PLAYER_TRACK] Actions: ${actions.actionSequence.join(' → ')} (${actions.numContacts} contacts)`);
        }

        const qualityReport = result.qualityReport as QualityReport | undefined;
        if (qualityReport) {
          console.log(`[PLAYER_TRACK] Quality: score=${qualityReport.trackabilityScore.toFixed(2)}, calibration=${qualityReport.calibrationRecommended ? 'recommended' : 'not needed'}`);
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
          ballPositions,
          contacts,
          actions,
          qualityReport,
        });
      } catch (parseError) {
        reject(new Error(`Failed to parse player tracker output: ${parseError}`));
      }
    });
  });
}

/**
 * Extract video segment from a LOCAL file (no S3 download — instant seek).
 * Used by batch tracking to avoid re-downloading the full video per rally.
 */
export async function extractVideoSegmentFromLocal(
  localVideoPath: string,
  startSeconds: number,
  durationSeconds: number,
  outputPath: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const args = [
      '-ss', startSeconds.toString(),
      '-i', localVideoPath,
      '-t', durationSeconds.toString(),
      '-c', 'copy',  // Stream copy — no re-encoding
      '-an',
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

/**
 * Save tracking results to the database.
 * Shared by single-rally sync tracking and batch tracking.
 */
export async function saveTrackingResult(
  rallyId: string,
  videoId: string,
  trackerResult: PlayerTrackerOutput,
  processingTimeMs: number,
): Promise<void> {
  // Save to database
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
      ballPositionsJson: trackerResult.ballPositions as unknown as object[],
      contactsJson: trackerResult.contacts as unknown as object,
      actionsJson: trackerResult.actions as unknown as object,
      qualityReportJson: trackerResult.qualityReport as unknown as object,
      processingTimeMs,
      modelVersion: 'yolo11s',
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
      ballPositionsJson: trackerResult.ballPositions as unknown as object[],
      contactsJson: trackerResult.contacts as unknown as object,
      actionsJson: trackerResult.actions as unknown as object,
      qualityReportJson: trackerResult.qualityReport as unknown as object,
      processingTimeMs,
      modelVersion: 'yolo11s',
      completedAt: new Date(),
      error: null,
    },
  });

  // Auto-populate Rally.servingTeam from detected serve team
  const detectedServingTeam = trackerResult.actions?.servingTeam;
  if (detectedServingTeam === 'A' || detectedServingTeam === 'B') {
    try {
      await prisma.rally.update({
        where: { id: rallyId },
        data: { servingTeam: detectedServingTeam },
      });
      console.log(`[PLAYER_TRACK] Set servingTeam=${detectedServingTeam} for rally ${rallyId}`);
    } catch (err) {
      console.log(`[PLAYER_TRACK] Failed to update servingTeam:`, err);
    }
  }

  // Update video characteristics from first completed tracking
  try {
    const primaryIds = new Set(trackerResult.primaryTrackIds ?? []);
    const heights = (trackerResult.positions as Array<{ trackId: number; height: number }>)
      .filter(p => primaryIds.has(p.trackId))
      .map(p => p.height);

    let cameraDistance: { avgBboxHeight: number; category: 'close' | 'medium' | 'far' } | undefined;
    if (heights.length > 0) {
      const sorted = [...heights].sort((a, b) => a - b);
      const avgBboxHeight = sorted[Math.floor(sorted.length / 2)];
      const category = avgBboxHeight > 0.35 ? 'close' : avgBboxHeight < 0.20 ? 'far' : 'medium';
      cameraDistance = { avgBboxHeight: Math.round(avgBboxHeight * 1000) / 1000, category };
    }

    const rawPositions = trackerResult.rawPositions ?? trackerResult.positions;
    const rawFrameCounts: Record<number, number> = {};
    for (const p of rawPositions as Array<{ frameNumber: number }>) {
      rawFrameCounts[p.frameNumber] = (rawFrameCounts[p.frameNumber] ?? 0) + 1;
    }
    const rawFrameValues = Object.values(rawFrameCounts);
    const avgPeople = rawFrameValues.length > 0
      ? rawFrameValues.reduce((a, b) => a + b, 0) / rawFrameValues.length
      : 0;
    const sceneComplexity = {
      avgPeople: Math.round(avgPeople * 10) / 10,
      category: (avgPeople > 6 ? 'complex' : 'simple') as 'simple' | 'complex',
    };

    const courtDetection = trackerResult.courtDetection;

    const video = await prisma.video.findUnique({
      where: { id: videoId },
      select: { characteristicsJson: true },
    });
    const existing = (video?.characteristicsJson as Record<string, unknown>) ?? {};
    const characteristics = {
      ...existing,
      ...(cameraDistance && { cameraDistance }),
      sceneComplexity,
      ...(courtDetection && { courtDetection }),
      version: 1,
    };
    await prisma.video.update({
      where: { id: videoId },
      data: {
        characteristicsJson: characteristics as unknown as Prisma.InputJsonValue,
      },
    });
    console.log(`[PLAYER_TRACK] Updated video characteristics for ${videoId}`);
  } catch (err) {
    console.log(`[PLAYER_TRACK] Failed to update video characteristics:`, err);
  }
}

/**
 * Track a single rally from a local video file.
 * Used by batch tracking — skips S3 download, extracts segment locally.
 */
export async function trackRallyFromLocalVideo(
  rallyId: string,
  videoId: string,
  localVideoPath: string,
  startMs: number,
  endMs: number,
  videoFps?: number,
  calibrationCorners?: CalibrationCorner[],
): Promise<TrackPlayersResult> {
  const startTime = Date.now();
  const durationMs = endMs - startMs;

  if (durationMs > MAX_BATCH_RALLY_DURATION_MS) {
    console.log(`[BATCH_TRACK] Skipping rally ${rallyId}: duration ${(durationMs / 1000).toFixed(1)}s exceeds limit`);
    await prisma.playerTrack.upsert({
      where: { rallyId },
      create: { rallyId, status: 'FAILED', error: `Duration exceeds ${MAX_BATCH_RALLY_DURATION_MS / 1000}s limit` },
      update: { status: 'FAILED', error: `Duration exceeds ${MAX_BATCH_RALLY_DURATION_MS / 1000}s limit` },
    });
    return { status: 'failed', error: `Duration exceeds limit` };
  }

  await fs.mkdir(TEMP_DIR, { recursive: true });
  const segmentPath = path.join(TEMP_DIR, `rally_${rallyId}_segment.mp4`);
  const outputPath = path.join(TEMP_DIR, `rally_${rallyId}_players.json`);

  try {
    // Mark as processing
    await prisma.playerTrack.upsert({
      where: { rallyId },
      create: { rallyId, status: 'PROCESSING' },
      update: { status: 'PROCESSING', error: null },
    });

    const startSeconds = startMs / 1000;
    const durationSeconds = durationMs / 1000;

    console.log(`[BATCH_TRACK] Extracting segment for rally ${rallyId}: ${startSeconds}s - ${endMs / 1000}s`);
    await extractVideoSegmentFromLocal(localVideoPath, startSeconds, durationSeconds, segmentPath);

    const trackerResult = await runPlayerTracker(segmentPath, outputPath, durationSeconds, videoFps, calibrationCorners);
    const processingTimeMs = Date.now() - startTime;

    await saveTrackingResult(rallyId, videoId, trackerResult, processingTimeMs);

    console.log(`[BATCH_TRACK] Rally ${rallyId} completed in ${processingTimeMs}ms`);

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
      ballPositions: trackerResult.ballPositions,
      contacts: trackerResult.contacts,
      actions: trackerResult.actions,
      qualityReport: trackerResult.qualityReport,
    };
  } catch (error) {
    console.error(`[BATCH_TRACK] Rally ${rallyId} failed:`, error);
    await prisma.playerTrack.upsert({
      where: { rallyId },
      create: { rallyId, status: 'FAILED', error: error instanceof Error ? error.message : String(error) },
      update: { status: 'FAILED', error: error instanceof Error ? error.message : String(error) },
    });
    return { status: 'failed', error: error instanceof Error ? error.message : String(error) };
  } finally {
    await fs.unlink(segmentPath).catch(() => {});
    await fs.unlink(outputPath).catch(() => {});
  }
}

/**
 * Download a full video from S3 to a local path.
 * Used by batch tracking to download once and extract segments locally.
 */
export async function downloadVideoToLocal(
  s3Key: string,
  localPath: string,
): Promise<void> {
  const videoUrl = await generateDownloadUrl(s3Key);
  return new Promise((resolve, reject) => {
    const args = [
      '-i', videoUrl,
      '-c', 'copy',  // No re-encoding, just download
      '-y',
      localPath,
    ];

    const ffmpeg = spawn('ffmpeg', args, {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stderr = '';
    ffmpeg.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    ffmpeg.on('error', (error) => {
      reject(new Error(`FFmpeg download failed to start: ${error.message}`));
    });

    ffmpeg.on('exit', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`FFmpeg download exited with code ${code}: ${stderr.slice(-500)}`));
      }
    });
  });
}

// Re-export types and constants used by batch tracking service
export { TEMP_DIR, MAX_BATCH_RALLY_DURATION_MS };
export type { CalibrationCorner, PlayerTrackerOutput };

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
    ballPositions: (track.ballPositionsJson as BallPosition[] | null) ?? undefined,
    contacts: (track.contactsJson as ContactsData | null) ?? undefined,
    actions: (track.actionsJson as ActionsData | null) ?? undefined,
    qualityReport: (track.qualityReportJson as QualityReport | null) ?? undefined,
  };
}

/**
 * Swap two player tracks from a given frame onward.
 * For all positions with frameNumber >= fromFrame, swaps trackId A ↔ B.
 */
export async function swapPlayerTracks(
  rallyId: string,
  userId: string,
  trackA: number,
  trackB: number,
  fromFrame: number,
): Promise<{ swappedCount: number }> {
  if (trackA === trackB) {
    throw new ValidationError('trackA and trackB must be different');
  }

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
    throw new ForbiddenError('You do not have permission to modify player tracking for this rally');
  }

  if (!rally.playerTrack || rally.playerTrack.status !== 'COMPLETED') {
    throw new ValidationError('No completed player tracking data found for this rally');
  }

  const positions = rally.playerTrack.positionsJson as PlayerPosition[] | null;
  if (!positions || positions.length === 0) {
    throw new ValidationError('No player positions found in tracking data');
  }

  let swappedCount = 0;
  for (const pos of positions) {
    if (pos.frameNumber >= fromFrame) {
      if (pos.trackId === trackA) {
        pos.trackId = trackB;
        swappedCount++;
      } else if (pos.trackId === trackB) {
        pos.trackId = trackA;
        swappedCount++;
      }
    }
  }

  await prisma.playerTrack.update({
    where: { rallyId },
    data: { positionsJson: positions as unknown as object[] },
  });

  console.log(`[PLAYER_TRACK] Swapped tracks ${trackA} ↔ ${trackB} from frame ${fromFrame}: ${swappedCount} positions updated`);

  return { swappedCount };
}

// Action ground truth label format
interface ActionGroundTruthLabel {
  frame: number;
  action: string;  // "serve", "receive", "set", "attack", "block", "dig"
  playerTrackId: number;
  ballX?: number;
  ballY?: number;
}

/**
 * Get action ground truth labels for a rally.
 */
export async function getActionGroundTruth(
  rallyId: string,
  userId: string,
): Promise<{ labels: ActionGroundTruthLabel[] } | null> {
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
    throw new ForbiddenError('You do not have permission to view action ground truth for this rally');
  }

  if (!rally.playerTrack) {
    return null;
  }

  const labels = rally.playerTrack.actionGroundTruthJson as ActionGroundTruthLabel[] | null;
  return labels ? { labels } : null;
}

/**
 * Save action ground truth labels for a rally.
 */
export async function saveActionGroundTruth(
  rallyId: string,
  userId: string,
  labels: ActionGroundTruthLabel[],
): Promise<{ savedCount: number }> {
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
    throw new ForbiddenError('You do not have permission to modify action ground truth for this rally');
  }

  if (!rally.playerTrack) {
    throw new ValidationError('No player tracking data found for this rally. Track players first.');
  }

  // Sort labels by frame
  const sorted = [...labels].sort((a, b) => a.frame - b.frame);

  await prisma.playerTrack.update({
    where: { rallyId },
    data: { actionGroundTruthJson: sorted as unknown as object[] },
  });

  console.log(`[PLAYER_TRACK] Saved ${sorted.length} action ground truth labels for rally ${rallyId}`);

  return { savedCount: sorted.length };
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

  // Auto-load calibration from DB when not provided by frontend
  // (same as batch tracking does in batchTrackingService.ts)
  if (!calibrationCorners) {
    const dbCorners = rally.video.courtCalibrationJson as unknown as CalibrationCorner[];
    if (Array.isArray(dbCorners) && dbCorners.length === 4) {
      calibrationCorners = dbCorners;
      console.log(`[PLAYER_TRACK] Using court calibration from database for video ${rally.video.id}`);
    }
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
    // Prefer original quality — proxy (720p) degrades ball detection.
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
    const videoFps = rally.video.fps ?? undefined;
    const trackerResult = await runPlayerTracker(segmentPath, outputPath, durationSeconds, videoFps, calibrationCorners);

    const processingTimeMs = Date.now() - startTime;
    console.log(`[PLAYER_TRACK] Completed in ${processingTimeMs}ms`);

    // Save to database and update video characteristics
    await saveTrackingResult(rallyId, rally.video.id, trackerResult, processingTimeMs);

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
      ballPositions: trackerResult.ballPositions,
      contacts: trackerResult.contacts,
      actions: trackerResult.actions,
      qualityReport: trackerResult.qualityReport,
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
