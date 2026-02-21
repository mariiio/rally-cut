/**
 * Match analysis service — cross-rally player identity + match statistics.
 *
 * Runs `rallycut match-players <video-id>` CLI to perform Hungarian-algorithm
 * appearance matching across all tracked rallies. Then runs
 * `rallycut compute-match-stats <video-id>` for efficiency metrics,
 * outcome attribution, and score progression.
 *
 * Persists player ID mappings in Video.matchAnalysisJson and
 * match statistics in Video.matchStatsJson.
 */

import { spawn } from 'child_process';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

import { env } from '../config/env.js';
import { Prisma } from '@prisma/client';
import { prisma } from '../lib/prisma.js';
import { ForbiddenError, NotFoundError } from '../middleware/errorHandler.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEMP_DIR = path.join(os.tmpdir(), 'rallycut-match-analysis');

interface MatchAnalysisResult {
  videoId: string;
  numRallies: number;
  rallies: Array<{
    rallyId: string;
    rallyIndex: number;
    startMs: number;
    endMs: number;
    trackToPlayer: Record<string, number>;
    assignmentConfidence: number;
    sideSwitchDetected: boolean;
    serverPlayerId: number | null;
  }>;
}

export interface MatchStatsResult {
  totalRallies: number;
  totalContacts: number;
  avgRallyDurationS: number;
  longestRallyDurationS: number;
  avgContactsPerRally: number;
  sideOutRate: number;
  playerStats: Array<{
    trackId: number;
    team: string;
    serves: number;
    receives: number;
    sets: number;
    attacks: number;
    blocks: number;
    digs: number;
    kills: number;
    attackErrors: number;
    aces: number;
    serveErrors: number;
    killPct: number;
    attackEfficiency: number;
    totalActions: number;
    totalDistancePx: number;
    totalDistanceM?: number;
    numFrames: number;
    courtSide: string;
  }>;
  teamStats?: Array<{
    team: string;
    playerIds: number[];
    serves: number;
    receives: number;
    sets: number;
    attacks: number;
    blocks: number;
    digs: number;
    kills: number;
    attackErrors: number;
    aces: number;
    serveErrors: number;
    killPct: number;
    attackEfficiency: number;
    totalActions: number;
    pointsWon: number;
    sideOutPct: number;
    serveWinPct: number;
  }>;
  rallyStats: Array<{
    rallyId: string;
    durationSeconds: number;
    numContacts: number;
    actionSequence: string[];
    terminalAction?: string;
    terminalPlayerTrackId?: number;
    pointWinner?: string;
    servingSide: string;
  }>;
  scoreProgression?: Array<{
    rallyId: string;
    scoreA: number;
    scoreB: number;
    servingTeam: string;
    pointWinner: string;
  }>;
  finalScoreA?: number;
  finalScoreB?: number;
}

/**
 * Run cross-rally player matching for a video.
 * Called automatically after batch tracking completes.
 */
export async function runMatchAnalysis(videoId: string): Promise<MatchAnalysisResult | null> {
  // Check that video has tracked rallies
  const trackedRallies = await prisma.rally.findMany({
    where: {
      videoId,
      status: 'CONFIRMED',
      playerTrack: { status: 'COMPLETED' },
    },
    orderBy: { startMs: 'asc' },
  });

  if (trackedRallies.length < 2) {
    console.log(`[MATCH_ANALYSIS] Skipping video ${videoId}: only ${trackedRallies.length} tracked rallies (need 2+)`);
    return null;
  }

  console.log(`[MATCH_ANALYSIS] Running cross-rally matching for video ${videoId} (${trackedRallies.length} rallies)`);

  await fs.mkdir(TEMP_DIR, { recursive: true });
  const outputPath = path.join(TEMP_DIR, `match_${videoId}.json`);

  try {
    const result = await runMatchPlayersCli(videoId, outputPath);

    // Persist to database
    await prisma.video.update({
      where: { id: videoId },
      data: {
        matchAnalysisJson: result as unknown as Prisma.InputJsonValue,
      },
    });

    console.log(`[MATCH_ANALYSIS] Saved match analysis for video ${videoId}: ${result.numRallies} rallies matched`);

    // Also compute match stats (best-effort — don't fail the whole pipeline)
    try {
      await computeAndSaveMatchStats(videoId);
    } catch (statsError) {
      console.error(`[MATCH_ANALYSIS] Stats computation failed (non-fatal):`, statsError);
    }

    return result;
  } catch (error) {
    console.error(`[MATCH_ANALYSIS] Failed for video ${videoId}:`, error);
    return null;
  } finally {
    await fs.unlink(outputPath).catch(() => {});
  }
}

/**
 * Compute and persist match statistics for a video.
 */
async function computeAndSaveMatchStats(videoId: string): Promise<void> {
  const outputPath = path.join(TEMP_DIR, `stats_${videoId}.json`);

  try {
    const stats = await runComputeMatchStatsCli(videoId, outputPath);

    await prisma.video.update({
      where: { id: videoId },
      data: {
        matchStatsJson: stats as unknown as Prisma.InputJsonValue,
      },
    });

    console.log(`[MATCH_STATS] Saved stats for video ${videoId}: ${stats.totalRallies} rallies, ${stats.totalContacts} contacts`);
  } finally {
    await fs.unlink(outputPath).catch(() => {});
  }
}

/**
 * Run the `rallycut match-players` CLI command.
 */
async function runMatchPlayersCli(
  videoId: string,
  outputPath: string,
): Promise<MatchAnalysisResult> {
  return runCli<MatchAnalysisResult>(
    ['match-players', videoId, '--output', outputPath, '--quiet'],
    outputPath,
    'MATCH_ANALYSIS',
  );
}

/**
 * Run the `rallycut compute-match-stats` CLI command.
 */
async function runComputeMatchStatsCli(
  videoId: string,
  outputPath: string,
): Promise<MatchStatsResult> {
  return runCli<MatchStatsResult>(
    ['compute-match-stats', videoId, '--output', outputPath, '--quiet'],
    outputPath,
    'MATCH_STATS',
  );
}

/**
 * Generic CLI runner for analysis commands.
 */
function runCli<T>(
  subcommandArgs: string[],
  outputPath: string,
  logPrefix: string,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const analysisDir = path.resolve(__dirname, '../../../analysis');

    const args = ['run', 'rallycut', ...subcommandArgs];

    console.log(`[${logPrefix}] Running: uv ${args.join(' ')}`);

    const child = spawn('uv', args, {
      cwd: analysisDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        DATABASE_URL: env.DATABASE_URL,
      },
    });

    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', (data: Buffer) => {
      stdout += data.toString();
      const line = data.toString().trim();
      if (line) console.log(`[${logPrefix}:PY:OUT] ${line}`);
    });

    child.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
      const line = data.toString().trim();
      if (line) console.log(`[${logPrefix}:PY] ${line}`);
    });

    child.on('error', (error) => {
      reject(new Error(`${logPrefix} failed to start: ${error.message}`));
    });

    child.on('exit', async (code) => {
      if (code !== 0) {
        reject(new Error(`${logPrefix} exited with code ${code}: ${(stderr || stdout).slice(-1000)}`));
        return;
      }

      try {
        const jsonContent = await fs.readFile(outputPath, 'utf-8');
        const result = JSON.parse(jsonContent) as T;
        resolve(result);
      } catch (parseError) {
        reject(new Error(`Failed to parse ${logPrefix} output: ${parseError}`));
      }
    });
  });
}

/**
 * Get match analysis for a video.
 */
export async function getMatchAnalysis(
  videoId: string,
  userId: string,
): Promise<MatchAnalysisResult | null> {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }

  if (video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to view match analysis for this video');
  }

  return (video.matchAnalysisJson as unknown as MatchAnalysisResult) ?? null;
}

/**
 * Get match statistics for a video.
 */
export async function getMatchStats(
  videoId: string,
  userId: string,
): Promise<MatchStatsResult | null> {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }

  if (video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to view match stats for this video');
  }

  return (video.matchStatsJson as unknown as MatchStatsResult) ?? null;
}
