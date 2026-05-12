/**
 * Action ground truth service.
 *
 * Provides save / get / reattach operations for RallyActionGroundTruth rows,
 * plus a re-resolve helper used by saveTrackingResult.
 *
 * Snapshot semantics on save:
 *  - PlayerTrack present AND trackId has a position at frame:
 *      snapshot bbox + ball + team, resolvedSource = SNAPSHOT_EXACT, resolvedTrackId = trackId.
 *  - PlayerTrack present but trackId has no position at that frame:
 *      bbox null, ball from ballPositionsJson if available, team null, resolvedSource = UNRESOLVED.
 *  - PlayerTrack absent:
 *      everything null except trackId hint, resolvedSource = UNRESOLVED.
 *
 * Frame bounds: when PlayerTrack exists, assert 0 <= frame < frameCount; throw ValidationError.
 * Permissions: only the owner of the Video can save / get / reattach. Throw ForbiddenError.
 */

import { ActionLabel, ServingTeam } from '@prisma/client';
import { prisma, type PrismaTransaction } from '../lib/prisma.js';
import { ForbiddenError, NotFoundError, ValidationError } from '../middleware/errorHandler.js';
import { resolveGtRow, type Candidate, type ResolveSource } from './actionGroundTruthResolver.js';

function bufferToFloat32(buf: Buffer | null): Float32Array | null {
  if (!buf) return null;
  // Prisma returns bytea as Buffer; reinterpret raw bytes as 128 × float32 LE.
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return new Float32Array(ab);
}

function float32ToBuffer(arr: Float32Array | null): Uint8Array<ArrayBuffer> | null {
  if (!arr) return null;
  // Copy into a plain ArrayBuffer to satisfy Prisma's Bytes type (no SharedArrayBuffer).
  const copy = arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength) as ArrayBuffer;
  return new Uint8Array(copy);
}

/** Shape of entries inside rawPositionsJson (tracker-native format, written by R3+). */
interface RawPos {
  frameNumber: number;
  trackId: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence?: number;
  embedding?: number[];
}

/**
 * Extract the OSNet embedding for (frame, trackId) from rawPositionsJson.
 * Returns null when the record is absent or has no embedding field.
 */
function findEmbedding(raw: RawPos[] | null, frame: number, trackId: number): Float32Array | null {
  if (!raw) return null;
  const hit = raw.find(p => p.frameNumber === frame && p.trackId === trackId);
  const emb = hit?.embedding;
  if (!emb || !Array.isArray(emb) || emb.length === 0) return null;
  return new Float32Array(emb);
}

/** One label entry as provided by the caller (lowercase action string). */
export interface ActionGtLabel {
  frame: number;
  action: string;  // lowercase: "serve" | "receive" | "set" | "attack" | "block" | "dig"
  trackId?: number | null;
  ballX?: number | null;
  ballY?: number | null;
}

/** Shape of entries inside positionsJson */
interface PlayerPosition {
  frameNumber: number;
  trackId: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

/** Shape of entries inside ballPositionsJson */
interface BallPosition {
  frameNumber: number;
  x: number;
  y: number;
  confidence: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Map lowercase action names to Prisma ActionLabel enum values. */
const ACTION_MAP: Record<string, ActionLabel> = {
  serve:   'SERVE',
  receive: 'RECEIVE',
  set:     'SET',
  attack:  'ATTACK',
  block:   'BLOCK',
  dig:     'DIG',
};

function toActionLabel(action: string): ActionLabel {
  const label = ACTION_MAP[action.toLowerCase()];
  if (label === undefined) {
    throw new ValidationError(`Unknown action label: '${action}'`);
  }
  return label;
}

/**
 * Extract teamAssignments from PlayerTrack.actionsJson.
 * Returns null if the field is absent or malformed.
 */
function readTeamAssignments(actionsJson: unknown): Record<string, 'A' | 'B'> | null {
  if (actionsJson == null || typeof actionsJson !== 'object') return null;
  const ta = (actionsJson as { teamAssignments?: Record<string, string> }).teamAssignments;
  if (!ta) return null;
  const out: Record<string, 'A' | 'B'> = {};
  for (const [k, v] of Object.entries(ta)) {
    if (v === 'A' || v === 'B') out[k] = v;
  }
  return out;
}

/** Convert a PlayerPosition to bbox coordinates (x1, y1, x2, y2). */
function positionToBbox(p: PlayerPosition): { x1: number; y1: number; x2: number; y2: number } {
  return {
    x1: p.x,
    y1: p.y,
    x2: p.x + p.width,
    y2: p.y + p.height,
  };
}

// ---------------------------------------------------------------------------
// saveActionGroundTruth
// ---------------------------------------------------------------------------

export async function saveActionGroundTruth(
  rallyId: string,
  userId: string,
  labels: ActionGtLabel[],
): Promise<{ savedCount: number; labels: Array<{ id: string }> }> {
  // 1. Load rally with its video (for ownership) and playerTrack (for snapshot).
  const rally = await prisma.rally.findUnique({
    where: { id: rallyId },
    include: {
      video: { select: { userId: true } },
      playerTrack: {
        select: {
          frameCount: true,
          positionsJson: true,
          rawPositionsJson: true,
          ballPositionsJson: true,
          actionsJson: true,
        },
      },
    },
  });

  if (!rally) {
    throw new NotFoundError('Rally', rallyId);
  }

  // Ownership check
  if (rally.video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to label this rally');
  }

  const pt = rally.playerTrack;
  const frameCount = pt?.frameCount ?? null;

  // Parse positions for snapshot lookup (cast through unknown for Prisma Json types)
  const positions = (pt?.positionsJson ?? []) as unknown as PlayerPosition[];
  const rawPositions = (pt?.rawPositionsJson ?? null) as unknown as RawPos[] | null;
  const ballPositions = (pt?.ballPositionsJson ?? []) as unknown as BallPosition[];
  const teamAssignments = readTeamAssignments(pt?.actionsJson);

  const savedLabels = await prisma.$transaction(async (tx) => {
    const results: Array<{ id: string }> = [];

    for (const label of labels) {
      const actionEnum = toActionLabel(label.action);

      // Frame bounds validation (only when PlayerTrack exists)
      if (frameCount !== null) {
        if (label.frame < 0 || label.frame >= frameCount) {
          throw new ValidationError(
            `Frame ${label.frame} out of bounds [0, ${frameCount}) for rally ${rallyId}`,
          );
        }
      }

      const trackId = label.trackId ?? null;

      // Find the position for this trackId at this frame
      const posAtFrame = trackId !== null
        ? positions.find(p => p.trackId === trackId && p.frameNumber === label.frame)
        : undefined;

      // Find ball at this frame
      const ballAtFrame = ballPositions.find(b => b.frameNumber === label.frame);

      let snapshotBboxX1: number | null = null;
      let snapshotBboxY1: number | null = null;
      let snapshotBboxX2: number | null = null;
      let snapshotBboxY2: number | null = null;
      let snapshotBallX: number | null = null;
      let snapshotBallY: number | null = null;
      let snapshotTeam: ServingTeam | null = null;
      let resolvedSource: ResolveSource;
      let resolvedTrackId: number | null = null;

      if (pt && posAtFrame) {
        // SNAPSHOT_EXACT: PlayerTrack present AND trackId has a position at frame
        const bbox = positionToBbox(posAtFrame);
        snapshotBboxX1 = bbox.x1;
        snapshotBboxY1 = bbox.y1;
        snapshotBboxX2 = bbox.x2;
        snapshotBboxY2 = bbox.y2;
        snapshotBallX = ballAtFrame?.x ?? (label.ballX ?? null);
        snapshotBallY = ballAtFrame?.y ?? (label.ballY ?? null);
        const teamChar = teamAssignments?.[String(trackId)];
        snapshotTeam = (teamChar === 'A' ? 'A' : teamChar === 'B' ? 'B' : null) as ServingTeam | null;
        resolvedSource = 'SNAPSHOT_EXACT';
        resolvedTrackId = trackId;
      } else {
        // UNRESOLVED: PlayerTrack absent OR trackId not present at frame
        // Still record ball if available
        snapshotBallX = ballAtFrame?.x ?? (label.ballX ?? null);
        snapshotBallY = ballAtFrame?.y ?? (label.ballY ?? null);
        resolvedSource = 'UNRESOLVED';
        resolvedTrackId = null;
      }

      // Extract OSNet embedding from rawPositionsJson (R3+ tracker writes embedding on each position).
      const embeddingArr = pt && trackId !== null
        ? findEmbedding(rawPositions, label.frame, trackId)
        : null;
      const snapshotReidEmbedding = embeddingArr ? float32ToBuffer(embeddingArr) : null;

      // Upsert by (rallyId, frame, action)
      const upserted = await tx.rallyActionGroundTruth.upsert({
        where: {
          rallyId_frame_action: {
            rallyId,
            frame: label.frame,
            action: actionEnum,
          },
        },
        create: {
          rallyId,
          frame: label.frame,
          action: actionEnum,
          snapshotBboxX1,
          snapshotBboxY1,
          snapshotBboxX2,
          snapshotBboxY2,
          snapshotBallX,
          snapshotBallY,
          snapshotTeam,
          snapshotTrackId: trackId,
          snapshotReidEmbedding,
          resolvedSource,
          resolvedTrackId,
          resolvedAt: resolvedSource !== 'UNRESOLVED' ? new Date() : null,
          createdBy: userId,
        },
        update: {
          snapshotBboxX1,
          snapshotBboxY1,
          snapshotBboxX2,
          snapshotBboxY2,
          snapshotBallX,
          snapshotBallY,
          snapshotTeam,
          snapshotTrackId: trackId,
          snapshotReidEmbedding,
          resolvedSource,
          resolvedTrackId,
          resolvedAt: resolvedSource !== 'UNRESOLVED' ? new Date() : null,
        },
        select: { id: true },
      });

      results.push({ id: upserted.id });
    }

    return results;
  });

  return { savedCount: savedLabels.length, labels: savedLabels };
}

// ---------------------------------------------------------------------------
// getActionGroundTruth
// ---------------------------------------------------------------------------

const ACTION_TO_LOWERCASE: Record<ActionLabel, string> = {
  SERVE: 'serve',
  RECEIVE: 'receive',
  SET: 'set',
  ATTACK: 'attack',
  BLOCK: 'block',
  DIG: 'dig',
};

export interface ActionGroundTruthResponse {
  id: string;
  frame: number;
  action: string;
  snapshotBboxX1: number | null;
  snapshotBboxY1: number | null;
  snapshotBboxX2: number | null;
  snapshotBboxY2: number | null;
  snapshotBallX: number | null;
  snapshotBallY: number | null;
  snapshotTeam: string | null;
  snapshotTrackId: number | null;
  resolvedTrackId: number | null;
  resolvedSource: string | null;
  resolvedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export async function getActionGroundTruth(
  rallyId: string,
  userId: string,
): Promise<{ labels: ActionGroundTruthResponse[] }> {
  const rally = await prisma.rally.findUnique({
    where: { id: rallyId },
    include: { video: { select: { userId: true } } },
  });

  if (!rally) {
    throw new NotFoundError('Rally', rallyId);
  }

  if (rally.video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to view labels for this rally');
  }

  const rows = await prisma.rallyActionGroundTruth.findMany({
    where: { rallyId },
    orderBy: { frame: 'asc' },
  });

  // Wire format: lowercase action for the legacy web contract; omit the
  // OSNet embedding (server-only — the web has no use for the raw bytes).
  const labels: ActionGroundTruthResponse[] = rows.map((r) => ({
    id: r.id,
    frame: r.frame,
    action: ACTION_TO_LOWERCASE[r.action],
    snapshotBboxX1: r.snapshotBboxX1,
    snapshotBboxY1: r.snapshotBboxY1,
    snapshotBboxX2: r.snapshotBboxX2,
    snapshotBboxY2: r.snapshotBboxY2,
    snapshotBallX: r.snapshotBallX,
    snapshotBallY: r.snapshotBallY,
    snapshotTeam: r.snapshotTeam,
    snapshotTrackId: r.snapshotTrackId,
    resolvedTrackId: r.resolvedTrackId,
    resolvedSource: r.resolvedSource,
    resolvedAt: r.resolvedAt ? r.resolvedAt.toISOString() : null,
    createdAt: r.createdAt.toISOString(),
    updatedAt: r.updatedAt.toISOString(),
  }));

  return { labels };
}

// ---------------------------------------------------------------------------
// reattachActionGroundTruth
// ---------------------------------------------------------------------------

export async function reattachActionGroundTruth(
  rowId: string,
  userId: string,
  newResolvedTrackId: number,
): Promise<void> {
  // Load the row and its rally's video for permission check
  const row = await prisma.rallyActionGroundTruth.findUnique({
    where: { id: rowId },
    include: {
      rally: {
        include: { video: { select: { userId: true } } },
      },
    },
  });

  if (!row) {
    throw new NotFoundError('RallyActionGroundTruth', rowId);
  }

  if (row.rally.video.userId !== userId) {
    throw new ForbiddenError('You do not have permission to reattach this label');
  }

  await prisma.rallyActionGroundTruth.update({
    where: { id: rowId },
    data: {
      resolvedSource: 'MANUAL',
      resolvedTrackId: newResolvedTrackId,
      resolvedAt: new Date(),
    },
  });
}

// ---------------------------------------------------------------------------
// reresolveRallyGt
// ---------------------------------------------------------------------------

/**
 * Re-resolve all GT rows for a rally against new tracking data.
 *
 * Called by saveTrackingResult after writing a new PlayerTrack.
 * Iterates each GT row, calls resolveGtRow to find the best matching
 * current track, and writes the resolved fields.
 *
 * Rows with resolvedSource=MANUAL are left untouched.
 *
 * @param tx      Prisma transaction client
 * @param rallyId  Rally whose GT rows to re-resolve
 * @param rawPositions  Array of {frameNumber, trackId, x, y, width, height}
 *                      from the new PlayerTrack (raw positions).
 */
export async function reresolveRallyGt(
  tx: PrismaTransaction,
  rallyId: string,
  rawPositions: Array<{
    frameNumber: number;
    trackId: number;
    x: number;
    y: number;
    width: number;
    height: number;
    confidence?: number;
    embedding?: number[] | null;     // 128 floats from the tracker, if available
  }>,
): Promise<void> {
  const rows = await tx.rallyActionGroundTruth.findMany({
    where: { rallyId },
  });

  if (rows.length === 0) return;

  // Build a frame → Candidate[] index for fast lookup
  const byFrame = new Map<number, Candidate[]>();
  for (const p of rawPositions) {
    const list = byFrame.get(p.frameNumber) ?? [];
    const candidate: Candidate = {
      trackId: p.trackId,
      bbox: {
        x1: p.x,
        y1: p.y,
        x2: p.x + p.width,
        y2: p.y + p.height,
      },
      embedding: p.embedding ? new Float32Array(p.embedding) : null,
    };
    list.push(candidate);
    byFrame.set(p.frameNumber, list);
  }

  for (const row of rows) {
    // Don't overwrite manual pins
    if (row.resolvedSource === 'MANUAL') continue;

    // Lookup candidates at the exact labeled frame. If empty, fall back to
    // ±1 frame — covers high-fps videos where the tracker subsamples (e.g.,
    // 60fps source, tracker outputs at even frames only, label happened to
    // land on an odd frame). Player position barely moves in 1/60s, so the
    // bbox IoU still matches strongly. Without this fallback the row lands
    // in UNRESOLVED purely because of the frame-index parity.
    let candidates = byFrame.get(row.frame) ?? [];
    if (candidates.length === 0) {
      candidates = [
        ...(byFrame.get(row.frame - 1) ?? []),
        ...(byFrame.get(row.frame + 1) ?? []),
      ];
    }
    const { resolvedTrackId, resolvedSource } = resolveGtRow(
      {
        snapshotBboxX1: row.snapshotBboxX1,
        snapshotBboxY1: row.snapshotBboxY1,
        snapshotBboxX2: row.snapshotBboxX2,
        snapshotBboxY2: row.snapshotBboxY2,
        snapshotTrackId: row.snapshotTrackId,
        snapshotReidEmbedding: bufferToFloat32(row.snapshotReidEmbedding as Buffer | null),
      },
      candidates,
    );

    await tx.rallyActionGroundTruth.update({
      where: { id: row.id },
      data: {
        resolvedTrackId,
        resolvedSource,
        resolvedAt: resolvedTrackId !== null ? new Date() : row.resolvedAt,
      },
    });
  }
}
