/**
 * Reference-crops service — invokes the analysis/ Python CLIs that back the
 * user-crop-guided identity workflow.
 *
 * - `validateReferenceCrops`: runs `rallycut validate-reference-crops`,
 *   returns the quality-validator JSON. Used by the pre-flight gate that
 *   blocks the "Re-run Matching" button until the selected crops can
 *   produce reliable prototypes.
 * - `suggestReferenceCrops`: runs `rallycut suggest-reference-crops`,
 *   returns ranked candidate metadata per player slot. Used by the dialog
 *   to auto-suggest diverse crops.
 * - `preAssignCandidates`: uses the current match analysis' trackToPlayer
 *   mapping to pre-fill the dialog — no Python subprocess needed.
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

import { env } from '../config/env.js';
import { prisma } from '../lib/prisma.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Shape of `ValidationResult.to_dict()` from
// analysis/rallycut/tracking/crop_guided_identity.py
export interface ValidationIssue {
  code: string;
  message: string;
  playerId: number | null;
}

export interface ValidationPayload {
  videoId: string;
  pass: boolean;
  issues: ValidationIssue[];
  cropCounts: Record<string, number>;
}

export interface CandidateCrop {
  rallyId: string;
  trackId: number;
  frameMs: number;
  bbox: { x: number; y: number; w: number; h: number };
  detectionConfidence: number;
}

export interface SuggestionPayload {
  videoId: string;
  candidates: Record<string, CandidateCrop[]>;
}

export interface PreAssignedCandidate extends CandidateCrop {
  /** Slot the Python side thinks this candidate belongs to (1-4). */
  suggestedPlayerId: number;
  /** Confidence of the pre-assignment in [0, 1]. */
  confidence: number;
}

async function runPythonJson<T>(
  subcommandArgs: string[],
  logPrefix: string,
): Promise<T> {
  const analysisDir = path.resolve(__dirname, '../../../analysis');
  const args = ['run', 'rallycut', ...subcommandArgs];

  console.log(`[${logPrefix}] Running: uv ${args.join(' ')}`);

  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const settle = (fn: () => void) => {
      if (!settled) { settled = true; fn(); }
    };

    const child = spawn('uv', args, {
      cwd: analysisDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, DATABASE_URL: env.DATABASE_URL },
    });

    // 2-minute timeout — these are short DB reads + DINOv2 feature extraction.
    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      settle(() => reject(new Error(`${logPrefix} timed out after 2 minutes`)));
    }, 2 * 60 * 1000);

    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', (data: Buffer) => { stdout += data.toString(); });
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
      if (code !== 0) {
        settle(() => reject(new Error(
          `${logPrefix} exited with code ${code}: ${(stderr || stdout).slice(-500)}`
        )));
        return;
      }
      try {
        const parsed = JSON.parse(stdout) as T;
        settle(() => resolve(parsed));
      } catch (err) {
        settle(() => reject(new Error(
          `${logPrefix} produced invalid JSON: ${(err as Error).message}`
        )));
      }
    });
  });
}

/**
 * Validate that the video's currently-assigned reference crops will produce
 * reliable DINOv2 prototypes. Returns structured per-player issues — the UI
 * surfaces these inline, and the "Re-run Matching" button stays disabled
 * until `pass: true`.
 */
export async function validateReferenceCrops(
  videoId: string,
  expectedPlayers = 4,
): Promise<ValidationPayload> {
  return runPythonJson<ValidationPayload>(
    ['validate-reference-crops', videoId, '--expected-players', String(expectedPlayers)],
    'VALIDATE_CROPS',
  );
}

/**
 * Emit ranked candidate reference crops per player slot. Requires that the
 * video has already run match-analysis at least once (so we have a
 * trackToPlayer mapping to group candidates by player).
 */
export async function suggestReferenceCrops(
  videoId: string,
  numCandidates = 6,
): Promise<SuggestionPayload> {
  return runPythonJson<SuggestionPayload>(
    ['suggest-reference-crops', videoId, '--num-candidates', String(numCandidates)],
    'SUGGEST_CROPS',
  );
}

/**
 * Pre-assign candidates to player slots using the current match-analysis
 * trackToPlayer mapping. Short-circuits to an empty payload when no mapping
 * exists yet — the dialog falls back to manual assignment in that case.
 */
export async function preAssignCandidates(
  videoId: string,
  candidates: CandidateCrop[],
): Promise<PreAssignedCandidate[]> {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
    select: { matchAnalysisJson: true },
  });
  const matchAnalysis = video?.matchAnalysisJson as
    | { rallies?: Array<{ rallyId: string; trackToPlayer?: Record<string, number> }> }
    | null;
  if (!matchAnalysis?.rallies) {
    return candidates.map(c => ({
      ...c,
      suggestedPlayerId: 0,
      confidence: 0,
    }));
  }

  const trackToPlayerByRally = new Map<string, Map<number, number>>();
  for (const entry of matchAnalysis.rallies) {
    if (!entry.trackToPlayer) continue;
    const map = new Map<number, number>();
    for (const [k, v] of Object.entries(entry.trackToPlayer)) {
      map.set(Number(k), Number(v));
    }
    trackToPlayerByRally.set(entry.rallyId, map);
  }

  return candidates.map(c => {
    const map = trackToPlayerByRally.get(c.rallyId);
    const suggested = map?.get(c.trackId) ?? 0;
    // Confidence is a simple boolean — the underlying match-players
    // assignment already has its own confidence, but at pre-assign time the
    // dialog only needs to know "do we have a suggestion" vs "unknown".
    const confidence = suggested > 0 ? 1.0 : 0.0;
    return {
      ...c,
      suggestedPlayerId: suggested,
      confidence,
    };
  });
}
