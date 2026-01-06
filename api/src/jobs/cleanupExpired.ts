import { prisma } from "../lib/prisma.js";
import { deleteObject } from "../lib/s3.js";
import { TIER_LIMITS } from "../services/tierService.js";

/**
 * Cleanup for FREE tier users:
 *
 * Phase A: Quality Downgrade (Day 7)
 * - Delete original/optimized S3 files, keep proxy (720p)
 * - Video remains accessible at lower quality
 * - Set qualityDowngradedAt to track when this happened
 *
 * Phase B: Inactivity Hard Delete (2 months)
 * - Find inactive FREE users (lastActiveAt > 60 days ago)
 * - Delete all their videos and sessions permanently
 *
 * PREMIUM users and sessions shared with PREMIUM users are never auto-deleted.
 */

const ORIGINAL_QUALITY_DAYS = TIER_LIMITS.FREE.originalQualityDays!; // 7 days
const INACTIVITY_DELETE_DAYS = TIER_LIMITS.FREE.inactivityDeleteDays!; // 60 days

export interface CleanupResult {
  qualityDowngradedVideos: number;
  hardDeletedUsers: number;
  hardDeletedVideos: number;
  hardDeletedSessions: number;
  errors: string[];
}

export async function cleanupExpiredContent(): Promise<CleanupResult> {
  const now = new Date();
  const errors: string[] = [];

  let qualityDowngradedVideos = 0;
  let hardDeletedUsers = 0;
  let hardDeletedVideos = 0;
  let hardDeletedSessions = 0;

  // Calculate cutoff dates
  const qualityDowngradeCutoff = new Date(
    now.getTime() - ORIGINAL_QUALITY_DAYS * 24 * 60 * 60 * 1000
  );
  const inactivityCutoff = new Date(
    now.getTime() - INACTIVITY_DELETE_DAYS * 24 * 60 * 60 * 1000
  );

  console.log("[CLEANUP] Starting cleanup...");
  console.log(`[CLEANUP] Current time: ${now.toISOString()}`);
  console.log(`[CLEANUP] Quality downgrade cutoff (7 days): ${qualityDowngradeCutoff.toISOString()}`);
  console.log(`[CLEANUP] Inactivity cutoff (60 days): ${inactivityCutoff.toISOString()}`);

  // ============================================================================
  // Phase A: Quality Downgrade for FREE tier videos older than 7 days
  // ============================================================================

  // Find videos to downgrade:
  // - Owner is FREE tier
  // - createdAt <= 7 days ago
  // - qualityDowngradedAt is null (not already downgraded)
  // - Has a proxy to fallback to
  // - Not in any session with a PREMIUM member
  const videosToDowngrade = await prisma.video.findMany({
    where: {
      user: { tier: "FREE" },
      createdAt: { lte: qualityDowngradeCutoff },
      qualityDowngradedAt: null,
      proxyS3Key: { not: null }, // Must have proxy to downgrade to
      deletedAt: null,
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
    select: {
      id: true,
      s3Key: true,
      originalS3Key: true,
      processedS3Key: true,
      proxyS3Key: true,
    },
  });

  console.log(`[CLEANUP] Found ${videosToDowngrade.length} videos to quality downgrade`);

  for (const video of videosToDowngrade) {
    // Collect S3 keys to delete (original quality files)
    const keysToDelete: string[] = [];

    // Delete original upload file if it exists and differs from proxy
    if (video.originalS3Key && video.originalS3Key !== video.proxyS3Key) {
      keysToDelete.push(video.originalS3Key);
    }

    // Delete optimized/processed file if it exists and differs from proxy
    if (video.s3Key && video.s3Key !== video.proxyS3Key) {
      keysToDelete.push(video.s3Key);
    }
    if (video.processedS3Key && video.processedS3Key !== video.proxyS3Key && video.processedS3Key !== video.s3Key) {
      keysToDelete.push(video.processedS3Key);
    }

    // Delete S3 files
    for (const key of keysToDelete) {
      try {
        await deleteObject(key);
        console.log(`[CLEANUP] Deleted high-quality S3 object: ${key}`);
      } catch (error) {
        const message = `Failed to delete S3 object ${key}: ${error}`;
        console.error(`[CLEANUP] ${message}`);
        errors.push(message);
      }
    }

    // Update video to use proxy as primary, clear original keys
    await prisma.video.update({
      where: { id: video.id },
      data: {
        s3Key: video.proxyS3Key!, // Proxy becomes the primary video
        originalS3Key: null,
        processedS3Key: null,
        qualityDowngradedAt: now,
      },
    });

    qualityDowngradedVideos++;
  }

  if (qualityDowngradedVideos > 0) {
    console.log(`[CLEANUP] Quality downgraded ${qualityDowngradedVideos} videos`);
  }

  // ============================================================================
  // Phase B: Hard Delete for inactive FREE users (2 months)
  // ============================================================================

  // Find inactive FREE users:
  // - tier = FREE
  // - lastActiveAt <= 60 days ago (or null, meaning never accessed)
  // - Has videos or sessions to delete
  const inactiveUsers = await prisma.user.findMany({
    where: {
      tier: "FREE",
      OR: [
        { lastActiveAt: { lte: inactivityCutoff } },
        // Users who never accessed (lastActiveAt is null) and created > 60 days ago
        {
          lastActiveAt: null,
          createdAt: { lte: inactivityCutoff },
        },
      ],
    },
    select: {
      id: true,
      videos: {
        where: { deletedAt: null },
        select: {
          id: true,
          s3Key: true,
          originalS3Key: true,
          posterS3Key: true,
          proxyS3Key: true,
          processedS3Key: true,
          confirmation: { select: { trimmedS3Key: true } },
        },
      },
      sessions: {
        where: { deletedAt: null },
        select: { id: true },
      },
    },
  });

  console.log(`[CLEANUP] Found ${inactiveUsers.length} inactive FREE users to cleanup`);

  for (const user of inactiveUsers) {
    // Skip users with no content to delete
    if (user.videos.length === 0 && user.sessions.length === 0) {
      continue;
    }

    // Collect all S3 keys for this user's videos
    const s3KeysToDelete: string[] = [];
    for (const video of user.videos) {
      if (video.s3Key) s3KeysToDelete.push(video.s3Key);
      if (video.originalS3Key) s3KeysToDelete.push(video.originalS3Key);
      if (video.posterS3Key) s3KeysToDelete.push(video.posterS3Key);
      if (video.proxyS3Key) s3KeysToDelete.push(video.proxyS3Key);
      if (video.processedS3Key) s3KeysToDelete.push(video.processedS3Key);
      if (video.confirmation?.trimmedS3Key) s3KeysToDelete.push(video.confirmation.trimmedS3Key);
    }

    // Delete S3 files in batches
    const S3_BATCH_SIZE = 10;
    for (let i = 0; i < s3KeysToDelete.length; i += S3_BATCH_SIZE) {
      const batch = s3KeysToDelete.slice(i, i + S3_BATCH_SIZE);
      const results = await Promise.allSettled(
        batch.map((key) => deleteObject(key))
      );

      results.forEach((result, idx) => {
        const key = batch[idx];
        if (result.status === "fulfilled") {
          console.log(`[CLEANUP] Deleted S3 object: ${key}`);
        } else {
          const message = `Failed to delete S3 object ${key}: ${result.reason}`;
          console.error(`[CLEANUP] ${message}`);
          errors.push(message);
        }
      });
    }

    // Delete videos from database
    if (user.videos.length > 0) {
      const videoIds = user.videos.map((v) => v.id);
      await prisma.video.deleteMany({
        where: { id: { in: videoIds } },
      });
      hardDeletedVideos += user.videos.length;
    }

    // Delete sessions from database
    if (user.sessions.length > 0) {
      const sessionIds = user.sessions.map((s) => s.id);
      await prisma.session.deleteMany({
        where: { id: { in: sessionIds } },
      });
      hardDeletedSessions += user.sessions.length;
    }

    hardDeletedUsers++;
    console.log(`[CLEANUP] Hard deleted content for inactive user ${user.id}`);
  }

  // ============================================================================
  // Summary
  // ============================================================================

  console.log("[CLEANUP] Cleanup completed:");
  console.log(`  - Quality downgraded videos: ${qualityDowngradedVideos}`);
  console.log(`  - Hard deleted users (inactive): ${hardDeletedUsers}`);
  console.log(`  - Hard deleted videos: ${hardDeletedVideos}`);
  console.log(`  - Hard deleted sessions: ${hardDeletedSessions}`);
  console.log(`  - Errors: ${errors.length}`);

  return {
    qualityDowngradedVideos,
    hardDeletedUsers,
    hardDeletedVideos,
    hardDeletedSessions,
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
