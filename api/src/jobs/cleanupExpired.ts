import { prisma } from "../lib/prisma.js";
import { deleteObject } from "../lib/s3.js";

/**
 * Cleanup expired content for FREE tier users.
 *
 * Two-phase retention:
 * 1. Day 7: Soft delete (set deletedAt, hide from UI)
 * 2. Day 30: Hard delete (remove from S3 + DB permanently)
 *
 * Exception: If a PREMIUM user is a member of the session, skip deletion entirely.
 */

const SOFT_DELETE_DAYS = 7;
const HARD_DELETE_DAYS = 30;

export interface CleanupResult {
  softDeletedSessions: number;
  softDeletedVideos: number;
  hardDeletedSessions: number;
  hardDeletedVideos: number;
  errors: string[];
}

export async function cleanupExpiredContent(): Promise<CleanupResult> {
  const now = new Date();
  const errors: string[] = [];

  let softDeletedSessions = 0;
  let softDeletedVideos = 0;
  let hardDeletedSessions = 0;
  let hardDeletedVideos = 0;

  // Calculate dates
  const hardDeleteCutoff = new Date(
    now.getTime() - (HARD_DELETE_DAYS - SOFT_DELETE_DAYS) * 24 * 60 * 60 * 1000
  );

  console.log("[CLEANUP] Starting expired content cleanup...");
  console.log(`[CLEANUP] Current time: ${now.toISOString()}`);
  console.log(`[CLEANUP] Hard delete cutoff: ${hardDeleteCutoff.toISOString()}`);

  // ============================================================================
  // Phase 1: Soft delete sessions at 7 days
  // ============================================================================

  // Find sessions to soft delete:
  // - expiresAt <= now
  // - deletedAt is null (not already soft deleted)
  // - Owner is FREE tier
  // - No PREMIUM members
  const sessionsToSoftDelete = await prisma.session.findMany({
    where: {
      expiresAt: { lte: now },
      deletedAt: null,
      user: { tier: "FREE" },
      // Exclude sessions where any member is PREMIUM
      NOT: {
        share: {
          members: {
            some: {
              user: { tier: "PREMIUM" },
            },
          },
        },
      },
    },
    select: { id: true },
  });

  if (sessionsToSoftDelete.length > 0) {
    const result = await prisma.session.updateMany({
      where: { id: { in: sessionsToSoftDelete.map((s) => s.id) } },
      data: { deletedAt: now },
    });
    softDeletedSessions = result.count;
    console.log(`[CLEANUP] Soft deleted ${softDeletedSessions} sessions`);
  }

  // Find videos to soft delete:
  // - expiresAt <= now
  // - deletedAt is null
  // - Owner is FREE tier
  // - Not in any session with a PREMIUM member
  const videosToSoftDelete = await prisma.video.findMany({
    where: {
      expiresAt: { lte: now },
      deletedAt: null,
      user: { tier: "FREE" },
      // Exclude videos in sessions where any member is PREMIUM
      NOT: {
        sessionVideos: {
          some: {
            session: {
              share: {
                members: {
                  some: {
                    user: { tier: "PREMIUM" },
                  },
                },
              },
            },
          },
        },
      },
    },
    select: { id: true },
  });

  if (videosToSoftDelete.length > 0) {
    const result = await prisma.video.updateMany({
      where: { id: { in: videosToSoftDelete.map((v) => v.id) } },
      data: { deletedAt: now },
    });
    softDeletedVideos = result.count;
    console.log(`[CLEANUP] Soft deleted ${softDeletedVideos} videos`);
  }

  // ============================================================================
  // Phase 2: Hard delete sessions at 30 days
  // ============================================================================

  // Find sessions to hard delete (same criteria + deleted >= 23 days ago)
  const sessionsToHardDelete = await prisma.session.findMany({
    where: {
      deletedAt: { lte: hardDeleteCutoff },
      user: { tier: "FREE" },
      NOT: {
        share: {
          members: {
            some: {
              user: { tier: "PREMIUM" },
            },
          },
        },
      },
    },
    select: { id: true },
  });

  if (sessionsToHardDelete.length > 0) {
    const result = await prisma.session.deleteMany({
      where: { id: { in: sessionsToHardDelete.map((s) => s.id) } },
    });
    hardDeletedSessions = result.count;
    console.log(`[CLEANUP] Hard deleted ${hardDeletedSessions} sessions`);
  }

  // Find videos to hard delete with S3 keys
  const videosToHardDelete = await prisma.video.findMany({
    where: {
      deletedAt: { lte: hardDeleteCutoff },
      user: { tier: "FREE" },
      NOT: {
        sessionVideos: {
          some: {
            session: {
              share: {
                members: {
                  some: {
                    user: { tier: "PREMIUM" },
                  },
                },
              },
            },
          },
        },
      },
    },
    select: { id: true, s3Key: true },
  });

  // Delete from S3 in batches for better performance
  const S3_BATCH_SIZE = 10;
  for (let i = 0; i < videosToHardDelete.length; i += S3_BATCH_SIZE) {
    const batch = videosToHardDelete.slice(i, i + S3_BATCH_SIZE);
    const results = await Promise.allSettled(
      batch.map((video) => deleteObject(video.s3Key))
    );

    results.forEach((result, idx) => {
      const video = batch[idx];
      if (result.status === "fulfilled") {
        console.log(`[CLEANUP] Deleted S3 object: ${video.s3Key}`);
      } else {
        const message = `Failed to delete S3 object ${video.s3Key}: ${result.reason}`;
        console.error(`[CLEANUP] ${message}`);
        errors.push(message);
      }
    });
  }

  // Delete from database
  if (videosToHardDelete.length > 0) {
    const result = await prisma.video.deleteMany({
      where: { id: { in: videosToHardDelete.map((v) => v.id) } },
    });
    hardDeletedVideos = result.count;
    console.log(`[CLEANUP] Hard deleted ${hardDeletedVideos} videos from database`);
  }

  // ============================================================================
  // Summary
  // ============================================================================

  console.log("[CLEANUP] Cleanup completed:");
  console.log(`  - Soft deleted sessions: ${softDeletedSessions}`);
  console.log(`  - Soft deleted videos: ${softDeletedVideos}`);
  console.log(`  - Hard deleted sessions: ${hardDeletedSessions}`);
  console.log(`  - Hard deleted videos: ${hardDeletedVideos}`);
  console.log(`  - Errors: ${errors.length}`);

  return {
    softDeletedSessions,
    softDeletedVideos,
    hardDeletedSessions,
    hardDeletedVideos,
    errors,
  };
}

/**
 * Reset monthly usage quotas for all users.
 * Should be run on the 1st of each month.
 */
export async function resetMonthlyQuotas(): Promise<number> {
  const monthStart = new Date();
  monthStart.setDate(1);
  monthStart.setHours(0, 0, 0, 0);

  const result = await prisma.userUsageQuota.updateMany({
    where: {
      periodStart: { lt: monthStart },
    },
    data: {
      periodStart: monthStart,
      detectionsUsed: 0,
    },
  });

  console.log(`[CLEANUP] Reset ${result.count} usage quotas for new month`);
  return result.count;
}
