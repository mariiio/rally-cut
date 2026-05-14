/**
 * Match analysis service — cross-rally player identity + match statistics.
 *
 * Runs `rallycut match-players <video-id>` CLI to perform Hungarian-algorithm
 * appearance matching across all tracked rallies. Then runs identity repair,
 * track ID remapping, action re-attribution, and match stats computation.
 *
 * Persists player ID mappings in Video.matchAnalysisJson and
 * match statistics in Video.matchStatsJson.
 */

import { spawn } from 'child_process';
import { randomUUID } from 'crypto';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

import { env } from '../config/env.js';
import { Prisma } from '@prisma/client';
import { prisma } from '../lib/prisma.js';
import { ForbiddenError, NotFoundError } from '../middleware/errorHandler.js';
import { consumePendingEdits } from './pendingAnalysisEdits.js';
import { planStages, shouldForceFullRerun } from './matchAnalysisPlanning.js';
import { reresolveRallyGt, reresolveVideoGtAgainstCanonical } from './actionGroundTruthService.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEMP_DIR = path.join(os.tmpdir(), 'rallycut-match-analysis');

interface TeamTemplateData {
  teamLabel: string;
  playerIds: number[];
  confidence: number;
}

// Video-wide pid→team mapping derived by per-pid majority vote across
// rallies. Written by compute-match-stats CLI (see
// analysis/rallycut/statistics/match_stats.py: compute_merged_team_identity).
// `valid: false` means the vote could not commit (no 2-and-2 partition or
// any per-pid confidence below 0.5). Consumers should fall back to
// per-rally team labels in that case rather than commit a poisoned identity.
export interface MergedTeamIdentity {
  version: 1;
  pidTeams: Record<string, "A" | "B">;
  perPidConfidence: Record<string, number>;
  partition: number[][];  // [[teamA_pids], [teamB_pids]] when valid, else []
  valid: boolean;
  totalRalliesVoting: number;
  source: "majority_vote";
}

// Per-rally cache key written by match-players blind path so the next
// run can pin this rally and skip MatchSolver re-decision when the
// pre-solve structural fingerprint is unchanged. See
// analysis/rallycut/tracking/match_tracker.py (ANCHOR_MIN_CONFIDENCE,
// MATCHER_VERSION). Read here only to keep the type honest — TS never
// mutates this; it round-trips through Prisma.
interface AssignmentAnchor {
  trackStatsHash: string;
  matcherVersion: string;
  assignment: Record<string, number>;
  confidence: number;
}

// Within-rally split fragment, written by match-players when phase-1
// repair detects an appearance-based ID switch inside a rally. Frames
// `[fStart, fEnd)` of `parentTrackId` belong to `pid` instead of the
// parent track's matcher PID. Source of truth:
// analysis/rallycut/tracking/_within_rally_id_switch.py.
interface SubTrack {
  syntheticTrackId: number;
  parentTrackId: number;
  segmentIndex: number;
  fStart: number;
  fEnd: number;
  pid: number;
  margin: number;
}

interface MatchAnalysisResult {
  videoId: string;
  numRallies: number;
  rallies: Array<{
    rallyId: string;
    rallyIndex: number;
    startMs: number;
    endMs: number;
    trackToPlayer: Record<string, number>;
    // Raw BoT-SORT trackId → canonical pid, written by remap-track-ids
    // (stage 4). Unlike `trackToPlayer` — which is collapsed to identity
    // post-remap — this preserves the original Hungarian mapping and is the
    // anchor the editor uses to resolve action-GT `trackId` to a display pid.
    appliedFullMapping?: Record<string, number>;
    remapApplied?: boolean;
    assignmentConfidence: number;
    sideSwitchDetected: boolean;
    serverPlayerId: number | null;
    assignmentAnchor?: AssignmentAnchor;
    subTracks?: SubTrack[];
  }>;
  playerProfiles?: Record<string, Record<string, unknown>>;
  teamTemplates?: Record<string, TeamTemplateData>;
  // Persisted by compute-match-stats (stage 6) as a single source of
  // truth for video-wide team identity. Replaces the legacy practice of
  // re-deriving team labels from per-rally `teamAssignments` via
  // last-rally-wins aggregation.
  mergedTeamIdentity?: MergedTeamIdentity;
  // Per-rally appearance scratchpad written by match-players (Phase 0,
  // see analysis/rallycut/tracking/match_tracker.py:scratchpad_to_dict).
  // Used by relabel-with-crops to replay Pass 2 with new frozen profiles
  // without re-extracting appearance from video. Opaque to TS.
  rallyScratchpad?: Record<string, unknown>;
}

export interface MatchStatsResult {
  totalRallies: number;
  totalContacts: number;
  avgRallyDurationS: number;
  longestRallyDurationS: number;
  avgContactsPerRally: number;
  sideOutRate: number;
  playerStats: Array<{
    playerId: number;
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
    terminalPlayerId?: number;
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

// ---------------------------------------------------------------------------
// Fire-and-forget trigger with in-memory running-set guard
// ---------------------------------------------------------------------------

const runningVideos = new Set<string>();

export function isMatchAnalysisRunning(videoId: string): boolean {
  return runningVideos.has(videoId);
}

/**
 * Run `fn` while holding the per-video match-analysis lock. Returns null
 * when the lock is already held (caller should respond 409 / fall back to
 * status polling). The lock is process-local — see triggerMatchAnalysis for
 * the multi-pod caveat.
 */
export async function withMatchAnalysisLock<T>(
  videoId: string,
  fn: () => Promise<T>,
): Promise<T | null> {
  if (runningVideos.has(videoId)) return null;
  runningVideos.add(videoId);
  try {
    return await fn();
  } finally {
    runningVideos.delete(videoId);
  }
}

/**
 * Fire-and-forget trigger. Returns true if scheduling succeeded, false if
 * a run is already in flight for this videoId (caller should respond 409).
 *
 * The running-set is process-local — if the API scales horizontally, two pods
 * could race. A2b's WebhookDelivery / job-model work will replace this with a
 * durable lock; for A2a we accept the limitation because runMatchAnalysis is
 * effectively idempotent (last write wins on Video.matchAnalysisJson).
 */
export function triggerMatchAnalysis(videoId: string): boolean {
  if (runningVideos.has(videoId)) return false;
  runningVideos.add(videoId);
  runMatchAnalysis(videoId)
    .catch((err) => {
      console.error(`[MATCH_ANALYSIS] ${videoId} failed:`, err);
    })
    .finally(() => {
      runningVideos.delete(videoId);
    });
  return true;
}

/**
 * Run cross-rally player matching for a video.
 * Called automatically after batch tracking completes.
 *
 * Prefer `triggerMatchAnalysis(videoId)` for client-initiated fire-and-forget flows.
 * Use this synchronous variant only for ops scripts, the legacy `/run-match-analysis`
 * SSE route, and test harnesses that need to await completion directly.
 *
 */
export type ProgressCallback = (step: string, index: number, total: number) => void;

export async function runMatchAnalysis(
  videoId: string,
  onProgress?: ProgressCallback,
): Promise<MatchAnalysisResult | null> {
  const started = Date.now();
  const timings: Record<string, number> = {};

  let result: MatchAnalysisResult | null = null;
  let fatalError: string | null = null;
  let plan: ReturnType<typeof planStages> | null = null;

  try {
    // Mark this run as in-flight. The paired matchAnalysisRanAt write in
    // the finally block below is what flips the UI status from `processing`
    // to `completed` / `failed`. Kept inside the try so a transient DB
    // failure here is recorded by the finally rather than thrown to callers
    // (the SSE route can't return a proper error after flushing headers).
    await prisma.video.update({
      where: { id: videoId },
      data: { matchAnalysisStartedAt: new Date(), matchAnalysisError: null },
    });

    const edits = await consumePendingEdits(videoId);
    plan = planStages(edits);

    // Override: partial-rerun assumes an existing matchAnalysisJson to update
    // against. If it's null (first-ever analysis, or cleared by a recent edit),
    // partial-rerun would skip match-players (stage 2) + repair-identities
    // (stage 3) and player IDs stay inconsistent across rallies. Force full.
    if (!plan.fullRerun) {
      const v = await prisma.video.findUnique({
        where: { id: videoId },
        select: { matchAnalysisJson: true },
      });
      if (shouldForceFullRerun(plan, v?.matchAnalysisJson != null)) {
        console.warn(
          `[MATCH_ANALYSIS] Forcing fullRerun for video ${videoId}: matchAnalysisJson is missing (partial-rerun would skip match-players)`,
        );
        plan = { fullRerun: true, changedRallyIds: [] };
      }
    }

    const report = onProgress ?? (() => {});

    const timed = async <T>(label: string, fn: () => Promise<T>): Promise<T> => {
      const t0 = Date.now();
      try { return await fn(); } finally { timings[label] = Date.now() - t0; }
    };

    // Stage 1: Validate tracked rallies — demote ball-pass false positives (always runs)
    report('Validating tracked rallies...', 1, 6);
    try {
      await timed('stage1_validate', () => validateTrackedRallies(videoId));
    } catch (valError) {
      console.error(`[MATCH_ANALYSIS] Rally validation failed (non-fatal):`, valError);
    }

    if (plan.fullRerun) {
      // Check that video has tracked rallies (required for match-players CLI)
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
      // UUID suffix so concurrent runs (multi-tab race) don't trample the same file.
      const outputPath = path.join(TEMP_DIR, `match_${videoId}_${randomUUID()}.json`);

      try {
        // Stage 2: Match players across rallies
        report('Matching players across rallies...', 2, 6);
        result = await timed(
          'stage2_match',
          () => runMatchPlayersCli(videoId, outputPath),
        );

        // Persist to database
        await prisma.video.update({
          where: { id: videoId },
          data: {
            matchAnalysisJson: result as unknown as Prisma.InputJsonValue,
          },
        });

        console.log(`[MATCH_ANALYSIS] Saved match analysis for video ${videoId}: ${result.numRallies} rallies matched`);

        // Stage 3: Repair within-rally identity switches using match-level profiles
        // (best-effort — don't fail the whole pipeline)
        report('Repairing identity switches...', 3, 6);
        try {
          await timed('stage3_repair', () => repairIdentities(videoId));
        } catch (repairError) {
          console.error(`[MATCH_ANALYSIS] Identity repair failed (non-fatal):`, repairError);
        }

        // Stage 4: Remap per-rally track IDs to consistent player IDs (1-4)
        // (best-effort — don't fail the whole pipeline)
        report('Remapping track IDs...', 4, 6);
        try {
          await timed('stage4_remap', () => remapTrackIds(videoId));
        } catch (remapError) {
          console.error(`[MATCH_ANALYSIS] Track ID remapping failed (non-fatal):`, remapError);
        }

        // Stage 4b: Re-resolve action GT rows against canonical positions
        // (post-remap). saveTrackingResult resolves at raw-id time, but
        // remap-track-ids rewrites the trackIds afterward — without this
        // step, `resolved_track_id` carries stale raw ids that don't
        // appear in positionsJson, contaminating any downstream consumer
        // that joins on the canonical id space. (best-effort)
        try {
          await timed('stage4b_reresolve_gt', async () => {
            const stats = await reresolveVideoGtAgainstCanonical(videoId);
            console.log(
              `[MATCH_ANALYSIS] Re-resolved GT for video ${videoId}: ` +
              `${stats.ralliesProcessed} rallies, ${stats.rowsUpdated} rows`
            );
          });
        } catch (reresolveError) {
          console.error(`[MATCH_ANALYSIS] GT re-resolve failed (non-fatal):`, reresolveError);
        }

        // Stage 5: Re-attribute player actions using match-level team identity
        // (best-effort — don't fail the whole pipeline)
        report('Classifying actions...', 5, 6);
        try {
          await timed('stage5_reattribute', () => reattributeActions(videoId));
        } catch (reattribError) {
          console.error(`[MATCH_ANALYSIS] Action re-attribution failed (non-fatal):`, reattribError);
        }
      } catch (error) {
        fatalError = error instanceof Error ? error.message : String(error);
        console.error(`[MATCH_ANALYSIS] Failed for video ${videoId}:`, error);
      } finally {
        await fs.unlink(outputPath).catch(() => {});
      }
    } else {
      // Partial rerun: only re-process changed rallies (stages 4 + 5).
      // Pin changedRallyIds locally so the closures below don't trip the
      // "plan is possibly null" narrowing loss.
      const changedRallyIds = plan.changedRallyIds;
      if (changedRallyIds.length > 0) {
        report('Remapping changed rallies...', 4, 6);
        try {
          await timed('stage4_remap', () => remapTrackIds(videoId, { rallyIds: changedRallyIds }));
        } catch (remapError) {
          console.error(`[MATCH_ANALYSIS] Track ID remapping failed (non-fatal):`, remapError);
        }

        // Stage 4b: Re-resolve action GT for changed rallies against canonical
        // positions. Walks all rallies (cheap relative to remap; per-rally
        // filter happens inside `reresolveVideoGtAgainstCanonical` via the
        // per-rally tx).
        try {
          await timed('stage4b_reresolve_gt', async () => {
            const stats = await reresolveVideoGtAgainstCanonical(videoId);
            console.log(
              `[MATCH_ANALYSIS] Re-resolved GT for video ${videoId}: ` +
              `${stats.ralliesProcessed} rallies, ${stats.rowsUpdated} rows`
            );
          });
        } catch (reresolveError) {
          console.error(`[MATCH_ANALYSIS] GT re-resolve failed (non-fatal):`, reresolveError);
        }

        report('Reattributing changed rallies...', 5, 6);
        try {
          await timed('stage5_reattribute', () => reattributeActions(videoId, { rallyIds: changedRallyIds }));
        } catch (reattribError) {
          console.error(`[MATCH_ANALYSIS] Action re-attribution failed (non-fatal):`, reattribError);
        }
      }
    }

    // Stage 6: Compute match stats (best-effort — don't fail the whole pipeline)
    report('Computing match stats...', 6, 6);
    try {
      await timed('stage6_stats', () => computeAndSaveMatchStats(videoId));
    } catch (statsError) {
      console.error(`[MATCH_ANALYSIS] Stats computation failed (non-fatal):`, statsError);
    }
  } catch (err) {
    fatalError = err instanceof Error ? err.message : String(err);
    console.error(`[MATCH_ANALYSIS] Unexpected failure for video ${videoId}:`, err);
  } finally {
    // Wrapped so a failing write here (e.g., video row deleted mid-run)
    // doesn't mask the original error or leave the lock dangling.
    try {
      await prisma.video.update({
        where: { id: videoId },
        data: {
          matchAnalysisRanAt: new Date(),
          matchAnalysisError: fatalError,
        },
      });
    } catch (cleanupErr) {
      console.error(`[MATCH_ANALYSIS] Failed to record completion for video ${videoId}:`, cleanupErr);
    }

    console.log(JSON.stringify({
      event: 'match_analysis.stage_timings',
      videoId,
      plan,
      timingsMs: timings,
      totalMs: Date.now() - started,
      error: fatalError,
    }));
  }

  return result;
}

/**
 * Validate tracked rallies and demote ball-pass false positives.
 *
 * A "disqualification score" accumulates from failing volleyball-semantic criteria.
 * Rally demoted to SUGGESTED (rejectionReason=BALL_PASS) only when score >= 4,
 * requiring multiple signals to agree.
 *
 * Safety gates (skip validation entirely):
 * - Ball detection rate < 15% → tracking data unreliable
 * - User-modified rally (has scoreA/scoreB/notes/servingTeam)
 * - Rally already SUGGESTED
 */
async function validateTrackedRallies(videoId: string): Promise<void> {
  const rallies = await prisma.rally.findMany({
    where: {
      videoId,
      status: 'CONFIRMED',
      playerTrack: { status: 'COMPLETED' },
    },
    include: { playerTrack: true },
    orderBy: { startMs: 'asc' },
  });

  const toDemote: Array<{ id: string; score: number; contactCount: number; hasServe: boolean; durationS: number }> = [];

  for (const rally of rallies) {
    // Safety gate: skip user-modified rallies
    if (rally.scoreA != null || rally.scoreB != null || rally.notes != null || rally.servingTeam != null) {
      continue;
    }

    const pt = rally.playerTrack;
    if (!pt) continue;

    // Safety gate: skip if ball detection rate is too low (tracking unreliable)
    const qualityReport = pt.qualityReportJson as Record<string, unknown> | null;
    const detectionRate = qualityReport?.detectionRate as number | undefined
      ?? pt.detectionRate as number | undefined;
    if (detectionRate != null && detectionRate < 0.15) {
      continue;
    }

    // Extract signals from tracking data
    const contactsData = pt.contactsJson as { contacts?: Array<{ isValidated?: boolean }> } | null;
    const validatedContacts = (contactsData?.contacts ?? []).filter(c => c.isValidated !== false);
    const contactCount = validatedContacts.length;

    const actionsData = pt.actionsJson as {
      actions?: Array<{ action?: string }>;
      actionSequence?: string[];
    } | null;
    const actionSequence = actionsData?.actionSequence ?? [];
    const hasServe = actionSequence.some(a => a === 'serve')
      || (actionsData?.actions ?? []).some(a => a.action === 'serve');

    const durationS = (rally.endMs - rally.startMs) / 1000;

    // Compute disqualification score
    let score = 0;
    if (contactCount === 0) score += 3;
    else if (contactCount === 1) score += 1;
    if (!hasServe) score += 2;
    if (durationS < 6 && contactCount < 2) score += 1;

    if (score >= 4) {
      toDemote.push({ id: rally.id, score, contactCount, hasServe, durationS });
    }
  }

  if (toDemote.length > 0) {
    // Batch update in a single transaction
    await prisma.$transaction(
      toDemote.map(r =>
        prisma.rally.update({
          where: { id: r.id },
          data: { status: 'SUGGESTED', rejectionReason: 'BALL_PASS' },
        }),
      ),
    );

    for (const r of toDemote) {
      console.log(
        `[RALLY_VALIDATION] Demoted rally ${r.id} (score=${r.score}, contacts=${r.contactCount}, serve=${r.hasServe}, dur=${r.durationS.toFixed(1)}s)`,
      );
    }
    console.log(`[RALLY_VALIDATION] Demoted ${toDemote.length}/${rallies.length} rallies as ball passes for video ${videoId}`);
  } else {
    console.log(`[RALLY_VALIDATION] All ${rallies.length} rallies passed validation for video ${videoId}`);
  }
}

/**
 * Detect and fix within-rally identity switches using match-level player profiles.
 * Compares temporal windows of each track against profiles to find appearance shifts,
 * then swaps track IDs in the affected time range if confident.
 */
async function repairIdentities(videoId: string): Promise<void> {
  const logPrefix = 'REPAIR_IDENTITIES';
  const analysisDir = path.resolve(__dirname, '../../../analysis');
  const args = ['run', 'rallycut', 'repair-identities', videoId, '--quiet'];

  console.log(`[${logPrefix}] Running: uv ${args.join(' ')}`);

  return new Promise((resolve, reject) => {
    let settled = false;
    const settle = (fn: () => void) => {
      if (!settled) { settled = true; fn(); }
    };

    const child = spawn('uv', args, {
      cwd: analysisDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, DATABASE_URL: env.DATABASE_URL },
    });

    // 5-minute timeout
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      settle(() => reject(new Error(`${logPrefix} timed out after 15 minutes`)));
    }, 15 * 60 * 1000);

    let stderr = '';
    child.stdout?.on('data', (data: Buffer) => {
      const line = data.toString().trim();
      if (line) console.log(`[${logPrefix}:PY:OUT] ${line}`);
    });
    child.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
      const line = data.toString().trim();
      if (line) console.log(`[${logPrefix}:PY] ${line}`);
    });
    child.on('error', (error) => {
      clearTimeout(timer);
      settle(() => reject(new Error(`${logPrefix} failed to start: ${error.message}`)));
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code != null && code === 0) {
        console.log(`[${logPrefix}] Completed for video ${videoId}`);
        settle(() => resolve());
      } else {
        settle(() => reject(new Error(
          `${logPrefix} exited with code ${code}: ${(stderr || '').slice(-1000)}`,
        )));
      }
    });
  });
}

/**
 * Remap per-rally track IDs to consistent match-level player IDs (1-4).
 * Updates positions_json, contacts_json, actions_json in player_tracks.
 */
async function remapTrackIds(videoId: string, opts: { rallyIds?: string[] } = {}): Promise<void> {
  const logPrefix = 'REMAP_TRACKS';
  const analysisDir = path.resolve(__dirname, '../../../analysis');
  const args = ['run', 'rallycut', 'remap-track-ids', videoId, '--quiet'];
  if (opts.rallyIds && opts.rallyIds.length > 0) {
    args.push('--rally-ids', opts.rallyIds.join(','));
  }

  console.log(`[${logPrefix}] Running: uv ${args.join(' ')}`);

  return new Promise((resolve, reject) => {
    let settled = false;
    const settle = (fn: () => void) => {
      if (!settled) { settled = true; fn(); }
    };

    const child = spawn('uv', args, {
      cwd: analysisDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, DATABASE_URL: env.DATABASE_URL },
    });

    // 5-minute timeout
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      settle(() => reject(new Error(`${logPrefix} timed out after 15 minutes`)));
    }, 15 * 60 * 1000);

    let stderr = '';
    child.stdout?.on('data', (data: Buffer) => {
      const line = data.toString().trim();
      if (line) console.log(`[${logPrefix}:PY:OUT] ${line}`);
    });
    child.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
      const line = data.toString().trim();
      if (line) console.log(`[${logPrefix}:PY] ${line}`);
    });
    child.on('error', (error) => {
      clearTimeout(timer);
      settle(() => reject(new Error(`${logPrefix} failed to start: ${error.message}`)));
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code != null && code === 0) {
        console.log(`[${logPrefix}] Completed for video ${videoId}`);
        settle(() => resolve());
      } else {
        settle(() => reject(new Error(
          `${logPrefix} exited with code ${code}: ${(stderr || '').slice(-1000)}`,
        )));
      }
    });
  });
}

/**
 * Re-match and remap track IDs for a single rally after re-tracking.
 *
 * Runs match-players (video-wide, fast) to produce a fresh trackToPlayer mapping
 * for the re-tracked rally's new raw track IDs, then remaps just this rally's
 * positions, contacts, actions, and primaryTrackIds.
 *
 * Skips if no match analysis exists yet (first-time tracking before batch analysis).
 * Repair-identities is not re-run — it already ran during the full match analysis
 * and the re-tracked rally has fresh detections that don't need appearance repair.
 *
 * Returns true if remapping was applied, false if skipped.
 */
export async function remapSingleRally(videoId: string, rallyId: string): Promise<boolean> {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
    select: { matchAnalysisJson: true },
  });

  if (!video?.matchAnalysisJson) {
    return false;
  }

  // Re-run match-players to get fresh mapping for the re-tracked rally's new raw track IDs.
  // This is video-wide but fast (DB reads + appearance comparison, no GPU).
  console.log(`[REMAP_SINGLE] Re-running match-players for video ${videoId}`);
  await fs.mkdir(TEMP_DIR, { recursive: true });
  const outputPath = path.join(TEMP_DIR, `rematch_${videoId}.json`);

  try {
    // Run match-players with existing profiles as frozen anchors.
    // This classifies the re-tracked rally against established player profiles
    // instead of rebuilding from scratch — prevents ambiguous assignments
    // that cause player teleports on single-rally retrack.
    await runMatchPlayersCli(videoId, outputPath, true);

    // Read the fresh match analysis from DB (camelCase, written by CLI)
    const updatedVideo = await prisma.video.findUnique({
      where: { id: videoId },
      select: { matchAnalysisJson: true },
    });
    const matchResult = updatedVideo?.matchAnalysisJson as unknown as MatchAnalysisResult | null;
    if (!matchResult?.rallies) {
      console.log(`[REMAP_SINGLE] No match analysis in DB after match-players, skipping remap`);
      return false;
    }

    // Find the rally's mapping from the fresh result
    const rallyEntry = matchResult.rallies.find((r) => r.rallyId === rallyId);
    if (!rallyEntry) {
      const matchRallyIds = matchResult.rallies.map((r) => r.rallyId);
      console.log(`[REMAP_SINGLE] Rally ${rallyId} not in match result (${matchResult.numRallies} rallies: ${matchRallyIds.join(', ')}), skipping remap`);
      return false;
    }

    const rawTrackToPlayer = rallyEntry.trackToPlayer;
    const trackToPlayer = new Map<number, number>();
    for (const [k, v] of Object.entries(rawTrackToPlayer)) {
      trackToPlayer.set(Number(k), v);
    }

    if (trackToPlayer.size === 0) {
      return false;
    }

    // Now remap just this rally (inline, no CLI)
    return await applyRemapToRally(videoId, rallyId, trackToPlayer, matchResult);
  } finally {
    await fs.unlink(outputPath).catch(() => {});
  }
}

/**
 * Apply a trackToPlayer mapping to a single rally's DB data.
 */
async function applyRemapToRally(
  videoId: string,
  rallyId: string,
  trackToPlayer: Map<number, number>,
  matchResult: MatchAnalysisResult,
): Promise<boolean> {

  // Load the player track
  const playerTrack = await prisma.playerTrack.findUnique({
    where: { rallyId },
  });
  if (!playerTrack || playerTrack.status !== 'COMPLETED') {
    return false;
  }

  // Collect all track IDs present in positions
  const positions = (playerTrack.positionsJson ?? []) as Array<Record<string, unknown>>;
  const allTrackIds = new Set<number>();
  for (const p of positions) {
    const tid = p.trackId as number | undefined;
    if (tid != null) allTrackIds.add(tid);
  }
  const primaryIds = (playerTrack.primaryTrackIds ?? []) as number[];
  for (const pid of primaryIds) allTrackIds.add(pid);

  // Build collision-safe full mapping (mapped → player IDs, unmapped → 101+ if collision)
  const usedIds = new Set(trackToPlayer.values());
  const mapping = new Map<number, number>();
  for (const [tid, pid] of trackToPlayer) {
    mapping.set(tid, pid);
  }
  let nextShifted = 101;
  for (const tid of [...allTrackIds].sort((a, b) => a - b)) {
    if (mapping.has(tid)) continue;
    if (usedIds.has(tid)) {
      while (allTrackIds.has(nextShifted) || usedIds.has(nextShifted)) nextShifted++;
      mapping.set(tid, nextShifted++);
    } else {
      mapping.set(tid, tid);
    }
  }

  // Check if mapping is all identity (nothing to remap)
  let hasChanges = false;
  for (const [k, v] of mapping) {
    if (k !== v) { hasChanges = true; break; }
  }
  if (!hasChanges) {
    console.log(`[REMAP_SINGLE] Rally ${rallyId.slice(0, 8)}: already using player IDs`);
    return false;
  }

  // Apply mapping to positions
  let remapCount = 0;
  for (const p of positions) {
    const oldId = p.trackId as number | undefined;
    if (oldId != null && mapping.has(oldId) && mapping.get(oldId) !== oldId) {
      p.trackId = mapping.get(oldId);
      remapCount++;
    }
  }

  // Apply mapping to contacts
  const contactsData = (playerTrack.contactsJson ?? {}) as Record<string, unknown>;
  const contacts = (contactsData.contacts ?? []) as Array<Record<string, unknown>>;
  for (const c of contacts) {
    const oldId = c.playerTrackId as number | undefined;
    if (oldId != null && mapping.has(oldId) && mapping.get(oldId) !== oldId) {
      c.playerTrackId = mapping.get(oldId);
      remapCount++;
    }
    const candidates = (c.playerCandidates ?? []) as Array<unknown[]>;
    for (const cand of candidates) {
      if (Array.isArray(cand) && cand.length >= 1) {
        const candId = cand[0] as number;
        if (mapping.has(candId)) cand[0] = mapping.get(candId);
      }
    }
  }

  // Apply mapping to actions
  const actionsData = (playerTrack.actionsJson ?? {}) as Record<string, unknown>;
  const actions = (actionsData.actions ?? []) as Array<Record<string, unknown>>;
  for (const a of actions) {
    const oldId = a.playerTrackId as number | undefined;
    if (oldId != null && mapping.has(oldId) && mapping.get(oldId) !== oldId) {
      a.playerTrackId = mapping.get(oldId);
      remapCount++;
    }
  }
  // Remap teamAssignments keys
  const oldTa = actionsData.teamAssignments as Record<string, string> | undefined;
  if (oldTa) {
    const newTa: Record<string, string> = {};
    for (const [tidStr, team] of Object.entries(oldTa)) {
      const tid = Number(tidStr);
      const newTid = mapping.get(tid) ?? tid;
      newTa[String(newTid)] = team;
    }
    actionsData.teamAssignments = newTa;
  }

  // Remap primaryTrackIds
  const newPrimaryIds = primaryIds.map((tid) => mapping.get(tid) ?? tid);

  // Update DB + re-resolve action GT in a single transaction. The GT
  // re-resolve uses the just-canonicalized positionsJson so
  // `resolved_track_id` lands in canonical id space (the contract the
  // web display layer + analysis eval consumers depend on).
  const canonicalTeamAssignments: Record<string, 'A' | 'B'> | undefined =
    actionsData.teamAssignments as Record<string, 'A' | 'B'> | undefined;
  await prisma.$transaction(async (tx) => {
    await tx.playerTrack.update({
      where: { rallyId },
      data: {
        positionsJson: positions as unknown as Prisma.InputJsonValue,
        contactsJson: contactsData as unknown as Prisma.InputJsonValue,
        actionsJson: actionsData as unknown as Prisma.InputJsonValue,
        primaryTrackIds: newPrimaryIds as unknown as Prisma.InputJsonValue,
      },
    });
    await reresolveRallyGt(
      tx,
      rallyId,
      positions as unknown as Array<{
        frameNumber: number; trackId: number;
        x: number; y: number; width: number; height: number;
        confidence?: number; embedding?: number[] | null;
      }>,
      canonicalTeamAssignments ?? null,
    );
  });

  // Update matchAnalysisJson: store appliedFullMapping + set trackToPlayer to identity
  const rallyEntry = matchResult.rallies.find((r) => r.rallyId === rallyId);
  if (rallyEntry) {
    const mappingObj: Record<string, number> = {};
    for (const [k, v] of mapping) mappingObj[String(k)] = v;
    (rallyEntry as Record<string, unknown>).appliedFullMapping = mappingObj;
    (rallyEntry as Record<string, unknown>).remapApplied = true;
    // Set trackToPlayer to identity (downstream consumers expect post-remap identity)
    const identityTtp: Record<string, number> = {};
    for (const v of trackToPlayer.values()) identityTtp[String(v)] = v;
    rallyEntry.trackToPlayer = identityTtp;

    await prisma.video.update({
      where: { id: videoId },
      data: { matchAnalysisJson: matchResult as unknown as Prisma.InputJsonValue },
    });
  }

  const mappingStr = [...trackToPlayer.entries()]
    .filter(([k, v]) => k !== v)
    .map(([k, v]) => `T${k}→P${v}`)
    .join(', ');
  console.log(`[REMAP_SINGLE] Rally ${rallyId.slice(0, 8)}: ${remapCount} remapped (${mappingStr})`);

  return true;
}

/**
 * Re-attribute player actions using match-level team assignments.
 * Updates actions_json in player_tracks where the team signal improves attribution.
 */
async function reattributeActions(videoId: string, opts: { rallyIds?: string[] } = {}): Promise<void> {
  const logPrefix = 'REATTRIBUTE';
  const analysisDir = path.resolve(__dirname, '../../../analysis');
  const args = ['run', 'rallycut', 'reattribute-actions', videoId, '--quiet'];
  if (opts.rallyIds && opts.rallyIds.length > 0) {
    args.push('--rally-ids', opts.rallyIds.join(','));
  }

  console.log(`[${logPrefix}] Running: uv ${args.join(' ')}`);

  return new Promise((resolve, reject) => {
    let settled = false;
    const settle = (fn: () => void) => {
      if (!settled) { settled = true; fn(); }
    };

    const child = spawn('uv', args, {
      cwd: analysisDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, DATABASE_URL: env.DATABASE_URL },
    });

    // 5-minute timeout
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      settle(() => reject(new Error(`${logPrefix} timed out after 15 minutes`)));
    }, 15 * 60 * 1000);

    let stderr = '';
    child.stdout?.on('data', (data: Buffer) => {
      const line = data.toString().trim();
      if (line) console.log(`[${logPrefix}:PY:OUT] ${line}`);
    });
    child.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
      const line = data.toString().trim();
      if (line) console.log(`[${logPrefix}:PY] ${line}`);
    });
    child.on('error', (error) => {
      clearTimeout(timer);
      settle(() => reject(new Error(`${logPrefix} failed to start: ${error.message}`)));
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code != null && code === 0) {
        console.log(`[${logPrefix}] Completed for video ${videoId}`);
        settle(() => resolve());
      } else {
        settle(() => reject(new Error(
          `${logPrefix} exited with code ${code}: ${(stderr || '').slice(-1000)}`,
        )));
      }
    });
  });
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
  useExistingProfiles?: boolean,
): Promise<MatchAnalysisResult> {
  const args = ['match-players', videoId, '--output', outputPath, '--quiet'];
  if (useExistingProfiles) {
    args.push('--use-existing-profiles');
  }
  return runCli<MatchAnalysisResult>(args, outputPath, 'MATCH_ANALYSIS');
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
    let settled = false;
    const settle = (fn: () => void) => {
      if (!settled) { settled = true; fn(); }
    };

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

    // 5-minute timeout
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      settle(() => reject(new Error(`${logPrefix} timed out after 15 minutes`)));
    }, 15 * 60 * 1000);

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
      clearTimeout(timer);
      settle(() => reject(new Error(`${logPrefix} failed to start: ${error.message}`)));
    });

    child.on('close', async (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        settle(() => reject(new Error(`${logPrefix} exited with code ${code}: ${(stderr || stdout).slice(-1000)}`)));
        return;
      }

      try {
        const jsonContent = await fs.readFile(outputPath, 'utf-8');
        const result = JSON.parse(jsonContent) as T;
        settle(() => resolve(result));
      } catch (parseError) {
        settle(() => reject(new Error(`Failed to parse ${logPrefix} output: ${parseError}`)));
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
 * Lifecycle status for the match-analysis pipeline, used by the editor to
 * decide whether to keep the spinner up or show the "Analysis complete!" toast.
 *
 * - idle      → no run has ever been recorded
 * - processing → matchAnalysisStartedAt > matchAnalysisRanAt (or ranAt is null)
 * - failed    → ran ≥ started AND matchAnalysisError is set
 * - completed → ran ≥ started AND no error
 */
export type MatchAnalysisStatus = 'idle' | 'processing' | 'completed' | 'failed';

export interface MatchAnalysisStatusResponse {
  status: MatchAnalysisStatus;
  startedAt: string | null;
  ranAt: string | null;
  error: string | null;
}

export async function getMatchAnalysisStatus(
  videoId: string,
  userId: string,
): Promise<MatchAnalysisStatusResponse> {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
    select: {
      userId: true,
      matchAnalysisStartedAt: true,
      matchAnalysisRanAt: true,
      matchAnalysisError: true,
    },
  });

  if (!video) {
    throw new NotFoundError('Video', videoId);
  }
  if (video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to view match analysis status for this video');
  }

  const { matchAnalysisStartedAt: startedAt, matchAnalysisRanAt: ranAt, matchAnalysisError: error } = video;

  let status: MatchAnalysisStatus;
  if (!startedAt && !ranAt) {
    status = 'idle';
  } else if (startedAt && (!ranAt || ranAt < startedAt)) {
    status = 'processing';
  } else if (error) {
    status = 'failed';
  } else {
    status = 'completed';
  }

  return {
    status,
    startedAt: startedAt ? startedAt.toISOString() : null,
    ranAt: ranAt ? ranAt.toISOString() : null,
    error,
  };
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

  const stats = (video.matchStatsJson as unknown as MatchStatsResult) ?? null;
  if (stats) {
    // Normalize old DB rows that used trackId → playerId
    for (const p of stats.playerStats ?? []) {
      if (!('playerId' in p) && 'trackId' in p) {
        (p as any).playerId = (p as any).trackId;
        delete (p as any).trackId;
      }
    }
    for (const r of stats.rallyStats ?? []) {
      if (!('terminalPlayerId' in r) && 'terminalPlayerTrackId' in r) {
        (r as any).terminalPlayerId = (r as any).terminalPlayerTrackId;
        delete (r as any).terminalPlayerTrackId;
      }
    }
  }
  return stats;
}
