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

  // Defensive dedup of the incoming batch. Same semantics as the web client
  // (web/src/stores/playerTrackingStore.ts addActionLabel):
  //   - Same frame, any action → keep latest only.
  //   - Same action AND same trackId within ±SAME_ACTION_WINDOW frames → keep latest.
  //   - Different action OR different player → both kept.
  // "Latest" here means the later position in the incoming labels array, which
  // is the order the client built them in. Stale state from non-client API
  // consumers also gets caught here.
  const SAME_ACTION_WINDOW = 3;
  const dedupedLabels: ActionGtLabel[] = [];
  for (const label of labels) {
    const keepers: ActionGtLabel[] = [];
    for (const existing of dedupedLabels) {
      if (existing.frame === label.frame) continue; // replaced by current
      if (existing.action !== label.action) { keepers.push(existing); continue; }
      const within = Math.abs(existing.frame - label.frame) <= SAME_ACTION_WINDOW;
      if (!within) { keepers.push(existing); continue; }
      const existingTid = existing.trackId ?? null;
      const newTid = label.trackId ?? null;
      if (existingTid !== null && newTid !== null && existingTid !== newTid) {
        keepers.push(existing); continue;
      }
      // Same action, within window, same/unknown trackId → drop earlier.
    }
    keepers.push(label);
    dedupedLabels.length = 0;
    dedupedLabels.push(...keepers);
  }

  const savedLabels = await prisma.$transaction(async (tx) => {
    const results: Array<{ id: string }> = [];

    // Replace semantic: the client always sends the COMPLETE intended state
    // for this rally (loadActionGroundTruth replaces the in-memory list with
    // server state; all mutations update that list; save sends the full list).
    // So labels that exist on the server but are NOT in this payload were
    // explicitly removed by the user and should be deleted.
    //
    // Safety guards:
    //  - Skip the delete entirely if the incoming payload is empty AND the
    //    server has rows (suggests the client never loaded; deleting would
    //    be destructive). The upsert loop is still a no-op in this case.
    //  - Never delete MANUAL pins (labeler explicitly attached them; treated
    //    as a stronger signal than the live in-memory list, which doesn't
    //    distinguish manual from optimistic).
    const existingRows = await tx.rallyActionGroundTruth.findMany({
      where: { rallyId },
      select: { id: true, frame: true, action: true, resolvedSource: true },
    });
    if (existingRows.length > 0 && dedupedLabels.length === 0) {
      console.warn(
        `[saveActionGroundTruth] Skipping replace-delete on rally ${rallyId}: ` +
        `server has ${existingRows.length} rows but client sent 0. ` +
        `This usually means loadActionGroundTruth never ran; refusing to wipe.`,
      );
    } else {
      const keysInPayload = new Set(
        dedupedLabels.map(l => `${l.frame}:${toActionLabel(l.action)}`),
      );
      // Delete any server row not in the incoming payload — including MANUAL.
      // MANUAL is a signal to the auto-resolver (reresolveVideoGtAgainstCanonical
      // skips it), NOT a signal to the explicit-delete path. When the user
      // removes a label in the UI and re-saves, the missing row is gone
      // regardless of how it was previously attributed. Without this, the
      // offscreen-player case (which lands MANUAL on save) would create
      // un-deletable labels.
      const toDelete = existingRows.filter(r =>
        !keysInPayload.has(`${r.frame}:${r.action}`),
      );
      if (toDelete.length > 0) {
        await tx.rallyActionGroundTruth.deleteMany({
          where: { id: { in: toDelete.map(r => r.id) } },
        });
        console.log(
          `[saveActionGroundTruth] Deleted ${toDelete.length} stale labels on rally ${rallyId} ` +
          `(client list omitted them): ${toDelete.map(r => `${r.frame}:${r.action}`).join(', ')}`,
        );
      }
    }

    for (const label of dedupedLabels) {
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

      // Find the position for this trackId at this frame.
      // Prefer positionsJson (canonical post-remap, trackIds 1-4) because the
      // labeler sends canonical pid directly (post-2026-05-14 fix in
      // ActionLabelingMode). Fall back to rawPositionsJson for two cases:
      //   (a) pre-remap rallies where positionsJson still has raw ids; the
      //       labeler's "canonical" pid happens to coincide with a raw id.
      //   (b) legacy clients still sending raw BoT-SORT ids.
      // Keeping positionsJson first means resolved_track_id stores the
      // canonical pid end-to-end, matching what reresolveVideoGtAgainstCanonical
      // writes and what gtLabelDisplay reads directly — no AFM hop, no
      // half-canonical confusion.
      // Lookup order:
      //  1. positionsJson at exact frame (canonical post-remap, normal case).
      //  2. positionsJson at frame±1 (covers 60fps stride-2 parity flips where
      //     the labeler steps to an odd frame the tracker doesn't write to).
      //  3. rawPositionsJson at exact frame (legacy/raw client fallback).
      // Each step only fires if the previous one missed.
      let posAtFrame: PlayerPosition | undefined;
      if (trackId !== null) {
        posAtFrame = positions.find(p => p.trackId === trackId && p.frameNumber === label.frame);
        if (!posAtFrame) {
          posAtFrame = positions.find(p =>
            p.trackId === trackId && Math.abs(p.frameNumber - label.frame) === 1,
          );
        }
        if (!posAtFrame && rawPositions) {
          const rawHit = rawPositions.find(p => p.trackId === trackId && p.frameNumber === label.frame);
          posAtFrame = rawHit as unknown as PlayerPosition | undefined;
        }
      }

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
        // SNAPSHOT_EXACT: PlayerTrack present AND trackId has a position at
        // frame (or ±1 via stride-2 fallback). Capture bbox + team.
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
      } else if (trackId !== null) {
        // MANUAL: user gave an explicit trackId but the player isn't tracked
        // at this frame (off-screen player, e.g., the server before they enter
        // the frame). Preserve the user's intent: resolved_track_id = trackId
        // so the display renders the right pid, and resolvedSource = MANUAL
        // so the auto-resolver doesn't override on the next match-analysis.
        // No bbox snapshot is possible here; ReID re-anchoring on retrack
        // also won't work for these. That's OK — the label is a user
        // assertion, not a tracking-derived attribution.
        snapshotBallX = ballAtFrame?.x ?? (label.ballX ?? null);
        snapshotBallY = ballAtFrame?.y ?? (label.ballY ?? null);
        resolvedSource = 'MANUAL';
        resolvedTrackId = trackId;
      } else {
        // UNRESOLVED: no trackId given (user labeled an action but didn't
        // assign a player; or auto-detect found nothing). Render as ghost.
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
          // Snapshot bbox/ball/team always reflect the current save's lookup.
          // If the user re-saved at the same (frame, action), they presumably
          // re-clicked or re-verified — overwriting the snapshot with the
          // freshest data is correct, including writing null when the lookup
          // missed (e.g., re-save against a rally that's since been retracked
          // and now lacks the player at this frame).
          snapshotBboxX1,
          snapshotBboxY1,
          snapshotBboxX2,
          snapshotBboxY2,
          snapshotBallX,
          snapshotBallY,
          snapshotTeam,
          // Resolver fields: when the user provided a trackId, that's an
          // explicit assertion — overwrite any stale auto-resolved value.
          // When trackId is null (no explicit player), preserve the existing
          // resolver decision (don't wipe a SNAPSHOT_EXACT to UNRESOLVED
          // just because a partial re-save came in).
          snapshotTrackId: trackId !== null ? trackId : undefined,
          snapshotReidEmbedding: snapshotReidEmbedding ?? undefined,
          resolvedSource: trackId !== null ? resolvedSource : undefined,
          resolvedTrackId: trackId !== null ? resolvedTrackId : undefined,
          resolvedAt: trackId !== null ? new Date() : undefined,
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
 * Re-resolve all GT rows for a rally against current tracking positions.
 *
 * The `positions` array is whatever the caller wants to match against —
 * `positionsJson` (canonical, post-remap) is the production choice so
 * `resolved_track_id` always lands in the canonical id space that
 * downstream consumers query (`{1, 2, 3, 4, 101+}` after remap). Passing
 * `rawPositionsJson` works too but produces stale raw ids after
 * `remap-track-ids` rewrites `positionsJson`, which contaminates any
 * consumer that joins on canonical ids.
 *
 * Iterates each GT row, calls resolveGtRow to find the best matching
 * current track, and writes the resolved fields.
 *
 * Rows with resolvedSource=MANUAL are left untouched.
 *
 * @param tx       Prisma transaction client
 * @param rallyId  Rally whose GT rows to re-resolve
 * @param positions  Array of {frameNumber, trackId, x, y, width, height}
 *                   from the new PlayerTrack. Prefer canonical positions
 *                   (post-remap) so resolved_track_id is consumer-ready.
 */
export async function reresolveRallyGt(
  tx: PrismaTransaction,
  rallyId: string,
  positions: Array<{
    frameNumber: number;
    trackId: number;
    x: number;
    y: number;
    width: number;
    height: number;
    confidence?: number;
    embedding?: number[] | null;     // 128 floats from the tracker, if available
  }>,
  // Optional trackId → team map. When provided, threaded into each Candidate.team
  // so the resolver's NEAREST_CENTER tier can reject wrong-team candidates.
  teamAssignments?: Record<string, 'A' | 'B'> | null,
): Promise<void> {
  const rows = await tx.rallyActionGroundTruth.findMany({
    where: { rallyId },
  });

  if (rows.length === 0) return;

  // Build a frame → Candidate[] index for fast lookup
  const byFrame = new Map<number, Candidate[]>();
  for (const p of positions) {
    const list = byFrame.get(p.frameNumber) ?? [];
    const team = teamAssignments ? (teamAssignments[String(p.trackId)] ?? null) : null;
    const candidate: Candidate = {
      trackId: p.trackId,
      bbox: {
        x1: p.x,
        y1: p.y,
        x2: p.x + p.width,
        y2: p.y + p.height,
      },
      embedding: p.embedding ? new Float32Array(p.embedding) : null,
      team,
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
        snapshotTeam: row.snapshotTeam,
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

// ---------------------------------------------------------------------------
// reresolveVideoGtAgainstCanonical
// ---------------------------------------------------------------------------

/**
 * Re-resolve every GT row across every rally of a video against the
 * **canonical** positions (post-`remap-track-ids`). Intended to be invoked
 * by `matchAnalysisService` after the remap stage so `resolved_track_id`
 * lands in the canonical id space (`{1, 2, 3, 4, 101+}`) that downstream
 * eval and analytics consumers query.
 *
 * The save-time hook inside `saveTrackingResult` runs the resolver against
 * `rawPositionsJson` (pre-remap), which is correct at that moment. But by
 * the time match-analysis remaps the per-rally trackIds, those raw ids no
 * longer match `positionsJson` — leaving `resolved_track_id` stale. This
 * helper closes that gap.
 *
 * Returns per-rally counts for observability.
 */
export async function reresolveVideoGtAgainstCanonical(
  videoId: string,
): Promise<{ ralliesProcessed: number; rowsUpdated: number }> {
  const rallies = await prisma.rally.findMany({
    where: { videoId },
    include: {
      playerTrack: {
        select: { positionsJson: true, actionsJson: true },
      },
    },
  });

  let ralliesProcessed = 0;
  let rowsUpdated = 0;

  for (const rally of rallies) {
    if (!rally.playerTrack?.positionsJson) continue;
    const canonicalPositions = rally.playerTrack.positionsJson as unknown as Array<{
      frameNumber: number;
      trackId: number;
      x: number;
      y: number;
      width: number;
      height: number;
      confidence?: number;
      embedding?: number[] | null;
    }>;
    if (!Array.isArray(canonicalPositions) || canonicalPositions.length === 0) continue;

    const teamAssignments = readTeamAssignments(rally.playerTrack.actionsJson);

    await prisma.$transaction(async (tx: PrismaTransaction) => {
      const before = await tx.rallyActionGroundTruth.count({
        where: { rallyId: rally.id, NOT: { resolvedSource: 'MANUAL' } },
      });
      await reresolveRallyGt(tx, rally.id, canonicalPositions, teamAssignments);
      rowsUpdated += before;
    });
    ralliesProcessed++;
  }

  return { ralliesProcessed, rowsUpdated };
}
