import { prisma } from "../lib/prisma.js";

// Define UserTier locally until Prisma client is regenerated
export type UserTier = "FREE" | "PREMIUM";

export const TIER_LIMITS = {
  FREE: {
    detectionsPerMonth: 1,
    maxVideoDurationMs: 15 * 60 * 1000, // 15 minutes
    maxFileSizeBytes: 1 * 1024 * 1024 * 1024, // 1 GB
    monthlyUploadCount: 5,
    exportQuality: "720p" as const,
    exportWatermark: true,
    lambdaExportEnabled: false,
    retentionDays: null, // Videos kept until 2 months inactive
    originalQualityDays: 3, // Original quality kept for 3 days, then downgraded to 720p proxy
    inactivityDeleteDays: 60, // Hard delete after 2 months of inactivity
    serverSyncEnabled: false,
    highlightsEnabled: true,
  },
  PREMIUM: {
    detectionsPerMonth: 6,
    maxVideoDurationMs: 25 * 60 * 1000, // 25 minutes
    maxFileSizeBytes: 2 * 1024 * 1024 * 1024, // 2 GB
    monthlyUploadCount: null as number | null, // unlimited
    exportQuality: "original" as const,
    exportWatermark: false,
    lambdaExportEnabled: true,
    retentionDays: null, // indefinite
    originalQualityDays: null, // Original quality kept forever
    inactivityDeleteDays: null, // Never auto-deleted
    serverSyncEnabled: true,
    highlightsEnabled: true,
  },
} as const;

export type TierLimits = (typeof TIER_LIMITS)["FREE"] | (typeof TIER_LIMITS)["PREMIUM"];

export function getTierLimits(tier: UserTier): TierLimits {
  return tier === "PREMIUM" ? TIER_LIMITS.PREMIUM : TIER_LIMITS.FREE;
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

  // Check if tier has expired
  if (user.tier === "PREMIUM" && user.tierExpiresAt) {
    if (new Date() > user.tierExpiresAt) {
      // Tier expired, downgrade to FREE
      await prisma.user.update({
        where: { id: userId },
        data: { tier: "FREE", tierExpiresAt: null },
      });
      return "FREE";
    }
  }

  return user.tier;
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

// ============================================================================
// Upload Quota
// ============================================================================

export interface UploadQuotaResult {
  allowed: boolean;
  used: number;
  limit: number | null;
  remaining: number | null;
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

  // null limit means unlimited
  if (limits.monthlyUploadCount === null) {
    return {
      allowed: true,
      used: quota.uploadsThisMonth,
      limit: null,
      remaining: null,
    };
  }

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

    // Null limit means unlimited - just increment and return
    if (tierLimits.monthlyUploadCount === null) {
      const updated = await tx.userUsageQuota.update({
        where: { userId },
        data: {
          uploadsThisMonth: { increment: 1 },
        },
      });
      return {
        allowed: true,
        used: updated.uploadsThisMonth,
        limit: null,
        remaining: null,
      };
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
