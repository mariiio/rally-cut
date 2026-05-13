/**
 * Re-resolve `rally_action_ground_truth` rows against current canonical
 * tracking (post-remap `positionsJson`).
 *
 * Why: prior to the 2026-05-13 fix, `saveTrackingResult` resolved GT rows
 * against raw positions (pre-remap). When `remap-track-ids` subsequently
 * rewrote per-rally track ids, `resolved_track_id` was left pointing at
 * stale raw ids that no longer existed in `positionsJson`. Downstream eval
 * scripts joining on canonical ids ended up reading wrong-player labels.
 *
 * Usage:
 *   # All videos, dry-run (default — prints the diff, writes nothing)
 *   npx tsx src/scripts/reresolveVideoGt.ts --all
 *
 *   # Single video, dry-run
 *   npx tsx src/scripts/reresolveVideoGt.ts --video <uuid>
 *
 *   # Apply
 *   npx tsx src/scripts/reresolveVideoGt.ts --all --apply
 *
 *   # Legacy env-var path still works (single video, applies immediately)
 *   VIDEO_ID=<uuid> npx tsx src/scripts/reresolveVideoGt.ts
 */
import { prisma } from '../lib/prisma.js';
import { resolveGtRow, type Candidate } from '../services/actionGroundTruthResolver.js';

interface Args {
  all: boolean;
  videoId: string | null;
  apply: boolean;
}

function parseArgs(argv: string[]): Args {
  const out: Args = { all: false, videoId: null, apply: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--all') out.all = true;
    else if (a === '--apply') out.apply = true;
    else if (a === '--video' || a === '--video-id') out.videoId = argv[++i] ?? null;
    else if (a === '--dry-run') out.apply = false;
  }
  // Back-compat: VIDEO_ID env var maps to --video <uuid> and implies --apply.
  if (!out.all && !out.videoId) {
    const envVid = process.env.VIDEO_ID;
    if (envVid) {
      out.videoId = envVid;
      out.apply = true;
    }
  }
  return out;
}

type ResolveSource = 'SNAPSHOT_EXACT' | 'IOU_MATCH' | 'REID_MATCH'
  | 'NEAREST_CENTER' | 'MANUAL' | 'UNRESOLVED';

interface Counts {
  SNAPSHOT_EXACT: number;
  IOU_MATCH: number;
  REID_MATCH: number;
  NEAREST_CENTER: number;
  MANUAL: number;
  UNRESOLVED: number;
}

function emptyCounts(): Counts {
  return {
    SNAPSHOT_EXACT: 0, IOU_MATCH: 0, REID_MATCH: 0,
    NEAREST_CENTER: 0, MANUAL: 0, UNRESOLVED: 0,
  };
}

function bufferToFloat32(buf: Buffer | null): Float32Array | null {
  if (!buf) return null;
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return new Float32Array(ab);
}

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

interface ResolvedDelta {
  rowId: string;
  rallyId: string;
  frame: number;
  oldTid: number | null;
  oldSrc: ResolveSource;
  newTid: number | null;
  newSrc: ResolveSource;
}

async function processVideo(videoId: string, apply: boolean): Promise<{
  before: Counts;
  after: Counts;
  deltas: ResolvedDelta[];
  ralliesScanned: number;
  ralliesWithCanonical: number;
}> {
  const before = emptyCounts();
  const after = emptyCounts();
  const deltas: ResolvedDelta[] = [];

  const rallies = await prisma.rally.findMany({
    where: { videoId },
    include: {
      playerTrack: {
        select: { positionsJson: true, actionsJson: true },
      },
    },
  });

  let ralliesWithCanonical = 0;

  for (const rally of rallies) {
    const rows = await prisma.rallyActionGroundTruth.findMany({
      where: { rallyId: rally.id },
    });
    if (rows.length === 0) continue;

    // Accumulate "before" counts for all rows in this rally.
    for (const r of rows) {
      before[r.resolvedSource as ResolveSource]++;
    }

    const canonicalPositions = rally.playerTrack?.positionsJson as unknown as Array<{
      frameNumber: number; trackId: number;
      x: number; y: number; width: number; height: number;
      embedding?: number[] | null;
    }> | null;

    if (!Array.isArray(canonicalPositions) || canonicalPositions.length === 0) {
      // Can't resolve — count as unchanged.
      for (const r of rows) after[r.resolvedSource as ResolveSource]++;
      continue;
    }

    ralliesWithCanonical++;
    const teamAssignments = readTeamAssignments(rally.playerTrack?.actionsJson);

    // Build frame → candidate index.
    const byFrame = new Map<number, Candidate[]>();
    for (const p of canonicalPositions) {
      const list = byFrame.get(p.frameNumber) ?? [];
      const team = teamAssignments ? (teamAssignments[String(p.trackId)] ?? null) : null;
      list.push({
        trackId: p.trackId,
        bbox: { x1: p.x, y1: p.y, x2: p.x + p.width, y2: p.y + p.height },
        embedding: p.embedding ? new Float32Array(p.embedding) : null,
        team,
      });
      byFrame.set(p.frameNumber, list);
    }

    const updates: Array<{
      rowId: string; newTid: number | null; newSrc: ResolveSource;
    }> = [];

    for (const row of rows) {
      // Preserve MANUAL pins.
      if (row.resolvedSource === 'MANUAL') {
        after.MANUAL++;
        continue;
      }
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
      after[resolvedSource as ResolveSource]++;

      const changed = row.resolvedTrackId !== resolvedTrackId
        || row.resolvedSource !== resolvedSource;
      if (changed) {
        deltas.push({
          rowId: row.id,
          rallyId: rally.id,
          frame: row.frame,
          oldTid: row.resolvedTrackId,
          oldSrc: row.resolvedSource as ResolveSource,
          newTid: resolvedTrackId,
          newSrc: resolvedSource as ResolveSource,
        });
        updates.push({ rowId: row.id, newTid: resolvedTrackId, newSrc: resolvedSource as ResolveSource });
      }
    }

    if (apply && updates.length > 0) {
      await prisma.$transaction(async (tx) => {
        for (const u of updates) {
          await tx.rallyActionGroundTruth.update({
            where: { id: u.rowId },
            data: {
              resolvedTrackId: u.newTid,
              resolvedSource: u.newSrc,
              resolvedAt: u.newTid !== null ? new Date() : undefined,
            },
          });
        }
      });
    }
  }

  return {
    before,
    after,
    deltas,
    ralliesScanned: rallies.length,
    ralliesWithCanonical,
  };
}

function printCounts(label: string, c: Counts): void {
  const total = c.SNAPSHOT_EXACT + c.IOU_MATCH + c.REID_MATCH + c.NEAREST_CENTER + c.MANUAL + c.UNRESOLVED;
  console.log(`  ${label}: total=${total}`);
  console.log(`    SNAPSHOT_EXACT=${c.SNAPSHOT_EXACT}  IOU_MATCH=${c.IOU_MATCH}  REID_MATCH=${c.REID_MATCH}`);
  console.log(`    NEAREST_CENTER=${c.NEAREST_CENTER}  MANUAL=${c.MANUAL}  UNRESOLVED=${c.UNRESOLVED}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (!args.all && !args.videoId) {
    console.error('Usage: reresolveVideoGt.ts (--all | --video <uuid>) [--apply]');
    process.exit(1);
  }

  let videoIds: string[];
  if (args.all) {
    const rows = await prisma.video.findMany({ select: { id: true } });
    videoIds = rows.map((r) => r.id);
  } else {
    videoIds = [args.videoId!];
  }

  const mode = args.apply ? 'APPLY' : 'DRY-RUN';
  console.log(`[reresolveVideoGt] mode=${mode}, videos=${videoIds.length}`);

  const totalBefore = emptyCounts();
  const totalAfter = emptyCounts();
  let totalDeltas = 0;
  let totalRallies = 0;
  let totalRalliesWithCanonical = 0;

  for (let i = 0; i < videoIds.length; i++) {
    const vid = videoIds[i];
    const { before, after, deltas, ralliesScanned, ralliesWithCanonical } =
      await processVideo(vid, args.apply);
    totalRallies += ralliesScanned;
    totalRalliesWithCanonical += ralliesWithCanonical;
    totalDeltas += deltas.length;
    for (const k of Object.keys(before) as Array<keyof Counts>) {
      totalBefore[k] += before[k];
      totalAfter[k] += after[k];
    }
    console.log(
      `[${i + 1}/${videoIds.length}] video=${vid.slice(0, 8)} ` +
      `rallies=${ralliesScanned} canonical=${ralliesWithCanonical} ` +
      `deltas=${deltas.length}`,
    );
  }

  console.log('\n=== Aggregate counts ===');
  printCounts('BEFORE', totalBefore);
  printCounts('AFTER', totalAfter);
  console.log(`\nTotal rallies scanned: ${totalRallies}`);
  console.log(`  with canonical positionsJson: ${totalRalliesWithCanonical}`);
  console.log(`Total deltas: ${totalDeltas}`);

  if (!args.apply) {
    console.log('\nDRY-RUN: no rows were written. Re-run with --apply to commit.');
  } else {
    console.log('\nAPPLIED: rows updated.');
  }

  await prisma.$disconnect();
}
main().catch((e) => { console.error(e); process.exit(1); });
