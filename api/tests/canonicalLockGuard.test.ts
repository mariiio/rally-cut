import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import { isRallyLocked, assertNotLocked } from '../src/services/canonicalLockGuard';
import { LockedRallyError } from '../src/middleware/errorHandler';

const userId = '11111111-1111-1111-1111-000000000b03';
const videoId = '22222222-2222-2222-2222-000000000b03';
const rallyIdLocked = '33333333-3333-3333-3333-000000000b03';
const rallyIdUnlocked = '33333333-3333-3333-3333-000000000b04';

describe('canonicalLockGuard', () => {
  beforeEach(async () => {
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
    await prisma.user.create({ data: { id: userId, tier: 'PRO' } });
    await prisma.video.create({
      data: {
        id: videoId,
        userId,
        name: 'lock-guard-test-video',
        filename: 'b.mp4',
        s3Key: 'test/b.mp4',
        contentHash: 'lock-guard-test-hash',
        durationMs: 60000,
        matchAnalysisJson: {
          videoId,
          numRallies: 2,
          rallies: [
            { rallyId: rallyIdLocked, canonicalLocked: true, trackToPlayer: {}, assignmentConfidence: 1 },
            { rallyId: rallyIdUnlocked, canonicalLocked: false, trackToPlayer: {}, assignmentConfidence: 1 },
          ],
        },
      },
    });
    await prisma.rally.createMany({
      data: [
        { id: rallyIdLocked, videoId, startMs: 0, endMs: 5000 },
        { id: rallyIdUnlocked, videoId, startMs: 5000, endMs: 10000 },
      ],
    });
  });

  afterEach(async () => {
    await prisma.rally.deleteMany({ where: { videoId } });
    await prisma.video.deleteMany({ where: { id: videoId } });
    await prisma.user.deleteMany({ where: { id: userId } });
  });

  it('isRallyLocked returns true for locked rally', async () => {
    expect(await isRallyLocked(prisma, rallyIdLocked)).toBe(true);
  });

  it('isRallyLocked returns false for unlocked rally', async () => {
    expect(await isRallyLocked(prisma, rallyIdUnlocked)).toBe(false);
  });

  it('isRallyLocked returns false when rally is missing from matchAnalysisJson', async () => {
    expect(await isRallyLocked(prisma, '44444444-4444-4444-4444-000000000b03')).toBe(false);
  });

  it('assertNotLocked throws LockedRallyError for locked rally', async () => {
    await expect(assertNotLocked(prisma, rallyIdLocked, 'SPLIT')).rejects.toBeInstanceOf(LockedRallyError);
  });

  it('assertNotLocked is a no-op for unlocked rally', async () => {
    await expect(assertNotLocked(prisma, rallyIdUnlocked, 'SPLIT')).resolves.toBeUndefined();
  });
});
