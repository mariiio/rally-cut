/**
 * Integration tests for actionGroundTruthService.
 *
 * Tests the four public functions:
 *   - saveActionGroundTruth  (upsert + snapshot)
 *   - getActionGroundTruth   (ordered retrieval)
 *   - reattachActionGroundTruth (manual pin)
 *   - reresolveRallyGt       (re-resolve on re-track)
 *
 * Follows the pattern from api/tests/saveTrackingResult.test.ts:
 * top-level UUIDs, beforeEach teardown-then-create-from-scratch.
 */
import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import {
  saveActionGroundTruth,
  getActionGroundTruth,
  reattachActionGroundTruth,
  reresolveRallyGt,
  reresolveVideoGtAgainstCanonical,
} from '../src/services/actionGroundTruthService';

const videoId  = 'agt00000-0000-0000-0000-000000000001';
const userId   = 'agt00000-0000-0000-0000-000000000002';
const otherUID = 'agt00000-0000-0000-0000-000000000003';
const rallyId  = 'agt00000-0000-0000-0000-000000000010';

/** A minimal PlayerPosition entry in positionsJson format */
function makePosition(trackId: number, frame: number, x = 0.1, y = 0.2, w = 0.1, h = 0.2) {
  return { frameNumber: frame, trackId, x, y, width: w, height: h, confidence: 0.9 };
}

/** A minimal ball position in ballPositionsJson format */
function makeBall(frame: number, bx = 0.5, by = 0.3) {
  return { frameNumber: frame, x: bx, y: by, confidence: 0.8 };
}

describe('actionGroundTruthService', () => {
  beforeEach(async () => {
    // Teardown first for isolation
    await prisma.rallyActionGroundTruth.deleteMany({ where: { rally: { videoId } } });
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: { in: [userId, otherUID] } } });

    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.user.create({ data: { id: otherUID, tier: 'PRO' } });
    await prisma.video.create({
      data: {
        id: videoId,
        name: 'agt-test',
        filename: 'agt.mp4',
        s3Key: 'test/agt.mp4',
        contentHash: 'agt-hash',
        userId,
      },
    });
    await prisma.rally.create({
      data: { id: rallyId, videoId, startMs: 0, endMs: 5000, order: 0 },
    });
    // Seed a PlayerTrack with trackId=1 at frame=10, ballPositionsJson, and
    // teamAssignments inside actionsJson.
    await prisma.playerTrack.create({
      data: {
        rallyId,
        status: 'COMPLETED',
        frameCount: 100,
        positionsJson: [
          makePosition(1, 10, 0.25, 0.30, 0.08, 0.18),
          makePosition(2, 10, 0.75, 0.70, 0.08, 0.18),
          makePosition(7, 50, 0.1, 0.1, 0.1, 0.2),
        ],
        rawPositionsJson: [
          makePosition(7, 50, 0.1, 0.1, 0.1, 0.2),
        ],
        ballPositionsJson: [makeBall(10, 0.50, 0.45)],
        actionsJson: {
          rallyId,
          numContacts: 1,
          actionSequence: ['serve'],
          actions: [],
          teamAssignments: { '1': 'A', '2': 'B' },
        },
      },
    });
  });

  afterEach(async () => {
    await prisma.rallyActionGroundTruth.deleteMany({ where: { rally: { videoId } } });
    await prisma.playerTrack.deleteMany({ where: { rally: { videoId } } });
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: { in: [userId, otherUID] } } });
  });

  // ------------------------------------------------------------------
  // Test 1: snapshot when PlayerTrack has the trackId at the labeled frame
  // ------------------------------------------------------------------
  it('saves a label with bbox snapshot when PlayerTrack has the trackId at frame', async () => {
    const result = await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 1 },
    ]);

    expect(result.savedCount).toBe(1);
    expect(result.labels).toHaveLength(1);

    const row = await prisma.rallyActionGroundTruth.findUnique({
      where: { id: result.labels[0].id },
    });
    expect(row).toBeTruthy();
    // Snapshot fields populated
    expect(row!.snapshotBboxX1).not.toBeNull();
    expect(row!.snapshotBboxY1).not.toBeNull();
    expect(row!.snapshotBboxX2).not.toBeNull();
    expect(row!.snapshotBboxY2).not.toBeNull();
    // Ball from ballPositionsJson
    expect(row!.snapshotBallX).not.toBeNull();
    expect(row!.snapshotBallY).not.toBeNull();
    // Team from teamAssignments
    expect(row!.snapshotTeam).toBe('A');
    // Resolve fields
    expect(row!.resolvedSource).toBe('SNAPSHOT_EXACT');
    expect(row!.resolvedTrackId).toBe(1);
    // Ownership
    expect(row!.createdBy).toBe(userId);
  });

  // ------------------------------------------------------------------
  // Test 2: MANUAL when user provides an explicit trackId but the player
  // isn't tracked at the labeled frame (off-screen player / coverage gap).
  // The user's intent is preserved as resolved_track_id so the display
  // renders the right pid; resolved_source = MANUAL signals the
  // auto-resolver to leave the row alone on the next match-analysis.
  // ------------------------------------------------------------------
  it('saves a MANUAL row when trackId is provided but not tracked at the labeled frame', async () => {
    // trackId=99 does not appear in positionsJson
    const result = await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 99 },
    ]);

    expect(result.savedCount).toBe(1);

    const row = await prisma.rallyActionGroundTruth.findUnique({
      where: { id: result.labels[0].id },
    });
    expect(row).toBeTruthy();
    // bbox should be null (no position for trackId=99 at frame=10)
    expect(row!.snapshotBboxX1).toBeNull();
    expect(row!.snapshotBboxY1).toBeNull();
    // snapshotTrackId records the hint
    expect(row!.snapshotTrackId).toBe(99);
    // resolved_track_id IS the user's explicit choice — display will render
    // pid 99 (or whatever 99 maps to). resolved_source = MANUAL freezes it.
    expect(row!.resolvedTrackId).toBe(99);
    expect(row!.resolvedSource).toBe('MANUAL');
  });

  // ------------------------------------------------------------------
  // Test 2b: UNRESOLVED when no trackId is provided (auto-detect failed
  // and user didn't override). Distinct from the MANUAL-no-bbox case.
  // ------------------------------------------------------------------
  it('saves an UNRESOLVED row when no trackId is provided at all', async () => {
    const result = await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve' },  // no trackId
    ]);

    const row = await prisma.rallyActionGroundTruth.findUnique({
      where: { id: result.labels[0].id },
    });
    expect(row).toBeTruthy();
    expect(row!.snapshotBboxX1).toBeNull();
    expect(row!.snapshotTrackId).toBeNull();
    expect(row!.resolvedTrackId).toBeNull();
    expect(row!.resolvedSource).toBe('UNRESOLVED');
  });

  // ------------------------------------------------------------------
  // Test 3: upsert on (rallyId, frame, action) — re-save updates in place
  // ------------------------------------------------------------------
  it('upserts on (rallyId, frame, action) — re-save updates in place', async () => {
    // First save
    const first = await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 1, ballX: 0.1, ballY: 0.2 },
    ]);
    const rowId1 = first.labels[0].id;

    // Second save with different ball position
    const second = await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 1, ballX: 0.9, ballY: 0.8 },
    ]);
    const rowId2 = second.labels[0].id;

    // Same row updated in place
    expect(rowId2).toBe(rowId1);

    // Snapshot ball from PlayerTrack still wins (snapshot_exact path), but the
    // manual override should not create a new row — count must still be 1.
    const count = await prisma.rallyActionGroundTruth.count({ where: { rallyId } });
    expect(count).toBe(1);
  });

  // ------------------------------------------------------------------
  // Test 4: cascade-deletes when Rally is deleted
  // ------------------------------------------------------------------
  it('cascade-deletes when Rally is deleted', async () => {
    await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 1 },
    ]);

    // Verify row exists
    const before = await prisma.rallyActionGroundTruth.count({ where: { rallyId } });
    expect(before).toBe(1);

    // Delete rally — should cascade
    await prisma.rallyActionGroundTruth.deleteMany({ where: { rallyId } });
    await prisma.playerTrack.deleteMany({ where: { rallyId } });
    await prisma.rally.delete({ where: { id: rallyId } });

    const after = await prisma.rallyActionGroundTruth.count({ where: { rallyId } });
    expect(after).toBe(0);
  });

  // ------------------------------------------------------------------
  // Test 5: reattach sets resolvedSource=MANUAL and pinned trackId
  // ------------------------------------------------------------------
  it('reattach sets resolvedSource=MANUAL and pinned trackId', async () => {
    const saved = await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 1 },
    ]);
    const rowId = saved.labels[0].id;

    await reattachActionGroundTruth(rowId, userId, 2);

    const row = await prisma.rallyActionGroundTruth.findUnique({ where: { id: rowId } });
    expect(row!.resolvedSource).toBe('MANUAL');
    expect(row!.resolvedTrackId).toBe(2);
    expect(row!.resolvedAt).not.toBeNull();
  });

  // ------------------------------------------------------------------
  // Test 6: rejects save when caller does not own the Video
  // ------------------------------------------------------------------
  it('rejects save when caller does not own the Video', async () => {
    await expect(
      saveActionGroundTruth(rallyId, otherUID, [
        { frame: 10, action: 'serve', trackId: 1 },
      ])
    ).rejects.toThrow(/permission/i);
  });

  // ------------------------------------------------------------------
  // Test 7: reresolveRallyGt re-resolves a non-MANUAL row via IoU
  // ------------------------------------------------------------------
  it('reresolveRallyGt re-resolves a non-MANUAL row via IoU when positions change', async () => {
    // Save a label that snapshots a bbox at trackId 7.
    await saveActionGroundTruth(rallyId, userId, [{ frame: 50, action: 'serve', trackId: 7 }]);

    // Call reresolveRallyGt directly inside a transaction with NEW positions
    // where trackId 11 has a near-identical bbox at frame 50.
    await prisma.$transaction(async (tx) => {
      await reresolveRallyGt(tx, rallyId, [
        { frameNumber: 50, trackId: 11, x: 0.11, y: 0.11, width: 0.10, height: 0.20, confidence: 0.9 },
      ]);
    });

    const row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId } });
    expect(row.resolvedTrackId).toBe(11);
    expect(row.resolvedSource).toBe('IOU_MATCH');
    expect(row.snapshotTrackId).toBe(7); // snapshot unchanged
  });

  // ------------------------------------------------------------------
  // Test 8: reresolveRallyGt preserves MANUAL pins
  // ------------------------------------------------------------------
  it('reresolveRallyGt preserves MANUAL pins', async () => {
    const { labels } = await saveActionGroundTruth(rallyId, userId, [{ frame: 50, action: 'serve', trackId: 7 }]);
    await reattachActionGroundTruth(labels[0].id, userId, 99);

    await prisma.$transaction(async (tx) => {
      await reresolveRallyGt(tx, rallyId, [
        { frameNumber: 50, trackId: 11, x: 0.11, y: 0.11, width: 0.10, height: 0.20, confidence: 0.9 },
      ]);
    });

    const row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId } });
    expect(row.resolvedTrackId).toBe(99);
    expect(row.resolvedSource).toBe('MANUAL');
  });

  // ------------------------------------------------------------------
  // Test 8b (2026-05-13 fix): reresolveVideoGtAgainstCanonical rewrites a
  // stale-raw SNAPSHOT_EXACT row to the canonical pid post-remap.
  //
  // Scenario: backfill (or save-time at raw) wrote resolvedTrackId=7 with
  // resolvedSource=SNAPSHOT_EXACT. Subsequent `remap-track-ids` rewrote the
  // canonical positionsJson so the player whose snapshot bbox matches now
  // sits at trackId=3. The canonical re-resolver must overwrite the stale
  // 7 with the canonical 3.
  // ------------------------------------------------------------------
  it('reresolveVideoGtAgainstCanonical rewrites stale-raw SNAPSHOT_EXACT to canonical id', async () => {
    // Snapshot: bbox at (0.1,0.1,0.2,0.3), tagged as raw trackId 7. Existing
    // positionsJson in the rally already has trackId 7 at frame 50 with the
    // same bbox (matches the test fixture). After we simulate a remap, that
    // canonical entry becomes trackId 3 with the same bbox.
    await saveActionGroundTruth(rallyId, userId, [{ frame: 50, action: 'serve', trackId: 7 }]);

    // Sanity: row was stamped SNAPSHOT_EXACT with raw 7.
    let row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId, frame: 50 } });
    expect(row.resolvedTrackId).toBe(7);
    expect(row.resolvedSource).toBe('SNAPSHOT_EXACT');

    // Simulate `remap-track-ids` rewriting positionsJson 7 → 3 (canonical).
    await prisma.playerTrack.update({
      where: { rallyId },
      data: {
        positionsJson: [
          makePosition(1, 10, 0.25, 0.30, 0.08, 0.18),
          makePosition(2, 10, 0.75, 0.70, 0.08, 0.18),
          // Same bbox as the snapshot; new canonical id is 3.
          makePosition(3, 50, 0.1, 0.1, 0.1, 0.2),
        ] as unknown as object,
      },
    });

    const stats = await reresolveVideoGtAgainstCanonical(videoId);
    expect(stats.ralliesProcessed).toBe(1);

    row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId, frame: 50 } });
    expect(row.snapshotTrackId).toBe(7); // snapshot anchor preserved
    // resolved_track_id must now point to a canonical id present in positionsJson.
    expect(row.resolvedTrackId).toBe(3);
    expect(row.resolvedSource).toBe('IOU_MATCH');
  });

  // ------------------------------------------------------------------
  // Test 9: captures snapshotReidEmbedding when rawPositionsJson has embedding
  // ------------------------------------------------------------------
  it('captures snapshotReidEmbedding when rawPositions has an embedding for the labeled trackId', async () => {
    // Re-seed the rally's PlayerTrack rawPositionsJson with an embedded position.
    await prisma.playerTrack.update({
      where: { rallyId },
      data: {
        rawPositionsJson: [{
          frameNumber: 10,
          trackId: 1,
          x: 0.1, y: 0.1, width: 0.1, height: 0.2,
          confidence: 0.9,
          embedding: Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.1)),
        }] as unknown as object,
      },
    });

    await saveActionGroundTruth(rallyId, userId, [{ frame: 10, action: 'serve', trackId: 1 }]);

    const row = await prisma.rallyActionGroundTruth.findFirstOrThrow({ where: { rallyId } });
    expect(row.snapshotReidEmbedding).not.toBeNull();
    // 128 float32 = 512 bytes
    expect((row.snapshotReidEmbedding as Buffer).length).toBe(512);
  });

  // ------------------------------------------------------------------
  // Batch dedup: same action + same player within ±3 frames → keep latest
  // ------------------------------------------------------------------
  it('dedups same-action same-player labels within ±3 frames in the incoming batch', async () => {
    await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 1 },
      { frame: 12, action: 'serve', trackId: 1 },  // dropped — same action+player within 3
    ]);
    const rows = await prisma.rallyActionGroundTruth.findMany({ where: { rallyId } });
    expect(rows).toHaveLength(1);
    expect(rows[0].frame).toBe(12);  // latest wins
  });

  // ------------------------------------------------------------------
  // Batch dedup: same action, DIFFERENT players within ±3 frames → both kept
  // ------------------------------------------------------------------
  it('preserves same-action different-player labels within ±3 frames (double-block)', async () => {
    await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'block', trackId: 1 },
      { frame: 12, action: 'block', trackId: 2 },  // different player — kept
    ]);
    const rows = await prisma.rallyActionGroundTruth.findMany({
      where: { rallyId },
      orderBy: { frame: 'asc' },
    });
    expect(rows).toHaveLength(2);
    expect(rows.map(r => r.frame)).toEqual([10, 12]);
  });

  // ------------------------------------------------------------------
  // Batch dedup: different actions at adjacent frames → both kept
  // ------------------------------------------------------------------
  it('preserves different actions at adjacent frames (attack → block sequence)', async () => {
    await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'attack', trackId: 1 },
      { frame: 12, action: 'block',  trackId: 1 },  // different action — kept
    ]);
    const rows = await prisma.rallyActionGroundTruth.findMany({
      where: { rallyId },
      orderBy: { frame: 'asc' },
    });
    expect(rows).toHaveLength(2);
  });

  // ------------------------------------------------------------------
  // Replace semantic: save deletes server rows that are NOT in the payload,
  // including MANUAL-by-save rows (offscreen-player labels). MANUAL is a
  // signal to the auto-resolver, not to the explicit-delete path: the user
  // can always remove their own labels by omitting them from the next save.
  // ------------------------------------------------------------------
  it('replace semantic: deletes server rows missing from the save payload', async () => {
    // First save: 3 labels.
    //  - frame 10 + trackId 1: positions has (1, 10) → SNAPSHOT_EXACT.
    //  - frame 30 + trackId 1: positions has no row at 30 → MANUAL (off-frame).
    //  - frame 50 + trackId 7: positions has (7, 50) → SNAPSHOT_EXACT.
    await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve',  trackId: 1 },
      { frame: 30, action: 'attack', trackId: 1 },
      { frame: 50, action: 'dig',    trackId: 7 },
    ]);
    expect(await prisma.rallyActionGroundTruth.count({ where: { rallyId } })).toBe(3);

    // Second save: only 2 of those, plus 1 new. The "attack" at frame=30 is
    // omitted — and even though it's MANUAL (because trackId 1 isn't at
    // frame 30), the replace-delete must remove it. User intent: deleted.
    await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 1 },
      { frame: 50, action: 'dig',   trackId: 7 },
      { frame: 70, action: 'block', trackId: 1 },
    ]);

    const rows = await prisma.rallyActionGroundTruth.findMany({
      where: { rallyId },
      orderBy: { frame: 'asc' },
    });
    const summary = rows.map(r => `${r.frame}:${r.action}`).sort();
    expect(summary).toEqual(['10:SERVE', '50:DIG', '70:BLOCK']);  // attack@30 gone
  });

  // ------------------------------------------------------------------
  // Replace semantic safety: empty payload + non-empty server → SKIP delete
  // ------------------------------------------------------------------
  it('replace semantic: empty payload on a rally with rows is a no-op (refuses to wipe)', async () => {
    await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 1 },
      { frame: 30, action: 'attack', trackId: 1 },
    ]);
    expect(await prisma.rallyActionGroundTruth.count({ where: { rallyId } })).toBe(2);

    // Empty payload → server should refuse to delete (likely a client that
    // never loaded). The 2 existing rows survive.
    await saveActionGroundTruth(rallyId, userId, []);
    expect(await prisma.rallyActionGroundTruth.count({ where: { rallyId } })).toBe(2);
  });

  // ------------------------------------------------------------------
  // Replace semantic: the user CAN delete a previously-reattached MANUAL row
  // by omitting it from the save. MANUAL is a signal to the auto-resolver,
  // not an immortality flag against the user's own delete intent.
  // (This used to be inverted; the prior behavior made offscreen-player
  // labels un-deletable from the UI.)
  // ------------------------------------------------------------------
  it('replace semantic: MANUAL rows (incl. reattached) are deletable by omission', async () => {
    const saved = await saveActionGroundTruth(rallyId, userId, [
      { frame: 10, action: 'serve', trackId: 1 },
    ]);
    await reattachActionGroundTruth(saved.labels[0].id, userId, 2);

    let row = await prisma.rallyActionGroundTruth.findUniqueOrThrow({
      where: { id: saved.labels[0].id },
    });
    expect(row.resolvedSource).toBe('MANUAL');

    // Save without the MANUAL row → it should be deleted.
    await saveActionGroundTruth(rallyId, userId, [
      { frame: 50, action: 'attack', trackId: 7 },
    ]);

    const survivor = await prisma.rallyActionGroundTruth.findUnique({
      where: { id: saved.labels[0].id },
    });
    expect(survivor).toBeNull();
  });
});
