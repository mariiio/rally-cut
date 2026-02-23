import { prisma } from "../lib/prisma.js";
import {
  type UserTier,
  type TierConfig,
  TIER_CONFIG,
  getTierConfig,
} from "../config/tiers.js";

// Re-export types and config for backward compatibility
export type { UserTier, TierConfig };
export { TIER_CONFIG, getTierConfig };

// Legacy export for backward compatibility with existing code
// Maps the new config structure to the old TIER_LIMITS format
export const TIER_LIMITS = {
  FREE: TIER_CONFIG.FREE,
  PRO: TIER_CONFIG.PRO,
  ELITE: TIER_CONFIG.ELITE,
} as const;

export type TierLimits = TierConfig;

export function getTierLimits(tier: UserTier): TierLimits {
  return getTierConfig(tier);
}

export async function getUserTier(userId: string | undefined): Promise<UserTier> {
  if (!userId) {
    return "FREE";
  }

  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { tier: true, tierExpiresAt: true },
  });

  if (!user) {
    return "FREE";
  }

  // Check if tier has expired (for paid tiers)
  if ((user.tier === "PRO" || user.tier === "ELITE") && user.tierExpiresAt) {
    if (new Date() > user.tierExpiresAt) {
      // Tier expired, downgrade to FREE
      await prisma.user.update({
        where: { id: userId },
        data: { tier: "FREE", tierExpiresAt: null },
      });
      return "FREE";
    }
  }

  return user.tier as UserTier;
}

export async function getUserTierLimits(
  userId: string | undefined
): Promise<TierLimits> {
  const tier = await getUserTier(userId);
  return getTierLimits(tier);
}

function getMonthStart(): Date {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth(), 1);
}

export async function getOrCreateUsageQuota(userId: string) {
  const monthStart = getMonthStart();

  const existing = await prisma.userUsageQuota.findUnique({
    where: { userId },
  });

  if (existing) {
    // Check if we need to reset for a new month
    if (existing.periodStart < monthStart) {
      return prisma.userUsageQuota.update({
        where: { userId },
        data: {
          periodStart: monthStart,
          detectionsUsed: 0,
          uploadsThisMonth: 0,
        },
      });
    }
    return existing;
  }

  return prisma.userUsageQuota.create({
    data: {
      userId,
      periodStart: monthStart,
      detectionsUsed: 0,
      uploadsThisMonth: 0,
    },
  });
}

export interface DetectionQuotaResult {
  allowed: boolean;
  used: number;
  limit: number;
  remaining: number;
}

export async function checkDetectionQuota(
  userId: string | undefined
): Promise<DetectionQuotaResult> {
  if (!userId) {
    // Anonymous users can't detect (need to be tracked)
    return { allowed: false, used: 0, limit: 0, remaining: 0 };
  }

  const tier = await getUserTier(userId);
  const limits = getTierLimits(tier);
  const quota = await getOrCreateUsageQuota(userId);

  const remaining = Math.max(0, limits.detectionsPerMonth - quota.detectionsUsed);
  const allowed = quota.detectionsUsed < limits.detectionsPerMonth;

  return {
    allowed,
    used: quota.detectionsUsed,
    limit: limits.detectionsPerMonth,
    remaining,
  };
}

export async function incrementDetectionUsage(userId: string): Promise<void> {
  // Ensure quota record exists first
  await getOrCreateUsageQuota(userId);

  // Use atomic increment to prevent race conditions
  await prisma.userUsageQuota.update({
    where: { userId },
    data: {
      detectionsUsed: { increment: 1 },
    },
  });
}

/**
 * Atomically check quota and reserve a detection slot.
 * Returns the quota result with the reserved slot already counted.
 * If quota is exceeded, throws LimitExceededError.
 *
 * This prevents race conditions where two concurrent requests both pass
 * the quota check before either increments.
 */
export async function checkAndReserveDetectionQuota(
  userId: string,
  tierLimits: TierLimits
): Promise<DetectionQuotaResult> {
  // Use a transaction to atomically check and reserve
  return prisma.$transaction(async (tx) => {
    const monthStart = getMonthStart();

    // Get or create quota, resetting if new month
    let quota = await tx.userUsageQuota.findUnique({
      where: { userId },
    });

    if (!quota) {
      quota = await tx.userUsageQuota.create({
        data: {
          userId,
          periodStart: monthStart,
          detectionsUsed: 0,
        },
      });
    } else if (quota.periodStart < monthStart) {
      // Reset for new month
      quota = await tx.userUsageQuota.update({
        where: { userId },
        data: {
          periodStart: monthStart,
          detectionsUsed: 0,
        },
      });
    }

    // Check if we can reserve a slot
    if (quota.detectionsUsed >= tierLimits.detectionsPerMonth) {
      return {
        allowed: false,
        used: quota.detectionsUsed,
        limit: tierLimits.detectionsPerMonth,
        remaining: 0,
      };
    }

    // Atomically increment and return the updated quota
    const updated = await tx.userUsageQuota.update({
      where: { userId },
      data: {
        detectionsUsed: { increment: 1 },
      },
    });

    return {
      allowed: true,
      used: updated.detectionsUsed,
      limit: tierLimits.detectionsPerMonth,
      remaining: Math.max(0, tierLimits.detectionsPerMonth - updated.detectionsUsed),
    };
  });
}

/**
 * Release a previously reserved detection quota slot (e.g., on detection failure).
 * Decrements detectionsUsed by 1, clamped to 0.
 */
export async function releaseDetectionQuota(userId: string): Promise<void> {
  const quota = await getOrCreateUsageQuota(userId);
  if (quota.detectionsUsed > 0) {
    await prisma.userUsageQuota.update({
      where: { userId },
      data: {
        detectionsUsed: { decrement: 1 },
      },
    });
  }
}

// ============================================================================
// Upload Quota
// ============================================================================

export interface UploadQuotaResult {
  allowed: boolean;
  used: number;
  limit: number;
  remaining: number;
}

export async function checkUploadQuota(
  userId: string | undefined
): Promise<UploadQuotaResult> {
  if (!userId) {
    return { allowed: false, used: 0, limit: 0, remaining: 0 };
  }

  const tier = await getUserTier(userId);
  const limits = getTierLimits(tier);
  const quota = await getOrCreateUsageQuota(userId);

  const remaining = Math.max(0, limits.monthlyUploadCount - quota.uploadsThisMonth);
  const allowed = quota.uploadsThisMonth < limits.monthlyUploadCount;

  return {
    allowed,
    used: quota.uploadsThisMonth,
    limit: limits.monthlyUploadCount,
    remaining,
  };
}

export async function incrementUploadUsage(userId: string): Promise<void> {
  await getOrCreateUsageQuota(userId);

  await prisma.userUsageQuota.update({
    where: { userId },
    data: {
      uploadsThisMonth: { increment: 1 },
    },
  });
}

/**
 * Atomically check quota and reserve an upload slot.
 * Returns the quota result with the reserved slot already counted.
 *
 * This prevents race conditions where two concurrent uploads both pass
 * the quota check before either increments.
 */
export async function checkAndReserveUploadQuota(
  userId: string,
  tierLimits: TierLimits
): Promise<UploadQuotaResult> {
  // Use a transaction to atomically check and reserve
  return prisma.$transaction(async (tx) => {
    const monthStart = getMonthStart();

    // Get or create quota, resetting if new month
    let quota = await tx.userUsageQuota.findUnique({
      where: { userId },
    });

    if (!quota) {
      quota = await tx.userUsageQuota.create({
        data: {
          userId,
          periodStart: monthStart,
          uploadsThisMonth: 0,
        },
      });
    } else if (quota.periodStart < monthStart) {
      // Reset for new month
      quota = await tx.userUsageQuota.update({
        where: { userId },
        data: {
          periodStart: monthStart,
          uploadsThisMonth: 0,
        },
      });
    }

    // Check if we can reserve a slot
    if (quota.uploadsThisMonth >= tierLimits.monthlyUploadCount) {
      return {
        allowed: false,
        used: quota.uploadsThisMonth,
        limit: tierLimits.monthlyUploadCount,
        remaining: 0,
      };
    }

    // Atomically increment and return the updated quota
    const updated = await tx.userUsageQuota.update({
      where: { userId },
      data: {
        uploadsThisMonth: { increment: 1 },
      },
    });

    return {
      allowed: true,
      used: updated.uploadsThisMonth,
      limit: tierLimits.monthlyUploadCount,
      remaining: Math.max(0, tierLimits.monthlyUploadCount - updated.uploadsThisMonth),
    };
  });
}

// ============================================================================
// Storage Quota
// ============================================================================

export interface StorageQuotaResult {
  allowed: boolean;
  usedBytes: number;
  limitBytes: number;
  remainingBytes: number;
}

/**
 * Calculate total storage used by a user.
 * Sums up file sizes of all non-deleted videos.
 */
export async function calculateUserStorageUsed(userId: string): Promise<number> {
  const result = await prisma.video.aggregate({
    where: {
      userId,
      deletedAt: null,
    },
    _sum: {
      fileSizeBytes: true,
    },
  });

  return Number(result._sum.fileSizeBytes ?? 0);
}

/**
 * Check if user has storage capacity for a new upload.
 */
export async function checkStorageQuota(
  userId: string,
  newFileSizeBytes: number
): Promise<StorageQuotaResult> {
  const tier = await getUserTier(userId);
  const limits = getTierLimits(tier);
  const usedBytes = await calculateUserStorageUsed(userId);

  const wouldUse = usedBytes + newFileSizeBytes;
  const allowed = wouldUse <= limits.storageCapBytes;

  return {
    allowed,
    usedBytes,
    limitBytes: limits.storageCapBytes,
    remainingBytes: Math.max(0, limits.storageCapBytes - usedBytes),
  };
}

/**
 * Get current storage usage for a user.
 */
export async function getStorageUsage(userId: string): Promise<{
  usedBytes: number;
  limitBytes: number;
  usedPercent: number;
}> {
  const tier = await getUserTier(userId);
  const limits = getTierLimits(tier);
  const usedBytes = await calculateUserStorageUsed(userId);

  return {
    usedBytes,
    limitBytes: limits.storageCapBytes,
    usedPercent: Math.round((usedBytes / limits.storageCapBytes) * 100),
  };
}
