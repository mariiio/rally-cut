import { prisma } from "../lib/prisma.js";
import { NotFoundError } from "../middleware/errorHandler.js";
import type { UserResponse, IdentityResponse } from "../schemas/user.js";
import {
  getUserTier,
  getTierLimits,
  getOrCreateUsageQuota,
  calculateUserStorageUsed,
} from "./tierService.js";

/**
 * Get or create a user from a visitor ID.
 * If the visitor ID exists, returns the associated user.
 * If not, creates a new anonymous user and links it to the visitor ID.
 *
 * Uses retry logic to handle race conditions where multiple requests
 * with the same visitor ID arrive simultaneously.
 */
export async function getOrCreateUser(
  visitorId: string,
  userAgent?: string
): Promise<IdentityResponse> {
  // First, try to find existing identity
  const existingIdentity = await prisma.anonymousIdentity.findUnique({
    where: { visitorId },
    select: { userId: true },
  });

  if (existingIdentity) {
    return {
      userId: existingIdentity.userId,
      isNew: false,
    };
  }

  // Try to create new user and identity
  // Handle race condition where another request might create it first
  try {
    const result = await prisma.$transaction(async (tx) => {
      const user = await tx.user.create({
        data: {
          // Anonymous user - no email, name, etc.
        },
      });

      await tx.anonymousIdentity.create({
        data: {
          visitorId,
          userId: user.id,
          userAgent,
        },
      });

      return user;
    });

    return {
      userId: result.id,
      isNew: true,
    };
  } catch (error) {
    // Check if this is a unique constraint violation (race condition)
    if (
      error instanceof Error &&
      error.message.includes("Unique constraint failed")
    ) {
      // Another request created the identity first, fetch it
      const identity = await prisma.anonymousIdentity.findUnique({
        where: { visitorId },
        select: { userId: true },
      });

      if (identity) {
        return {
          userId: identity.userId,
          isNew: false,
        };
      }
    }

    // Re-throw other errors
    throw error;
  }
}

/**
 * Get user by ID with counts and tier information.
 */
export async function getUserById(id: string): Promise<UserResponse> {
  const user = await prisma.user.findUnique({
    where: { id },
    include: {
      _count: {
        select: {
          videos: {
            where: { deletedAt: null },
          },
          sessions: {
            where: { deletedAt: null },
          },
        },
      },
    },
  });

  if (!user) {
    throw new NotFoundError("User", id);
  }

  const tier = await getUserTier(id);
  const tierLimits = getTierLimits(tier);
  const quota = await getOrCreateUsageQuota(id);
  const storageUsedBytes = await calculateUserStorageUsed(id);

  return {
    id: user.id,
    email: user.email,
    name: user.name,
    avatarUrl: user.avatarUrl,
    tier,
    tierLimits,
    usage: {
      detectionsUsed: quota.detectionsUsed,
      detectionsLimit: tierLimits.detectionsPerMonth,
      detectionsRemaining: Math.max(
        0,
        tierLimits.detectionsPerMonth - quota.detectionsUsed
      ),
      uploadsThisMonth: quota.uploadsThisMonth,
      uploadsLimit: tierLimits.monthlyUploadCount,
      uploadsRemaining: Math.max(0, tierLimits.monthlyUploadCount - quota.uploadsThisMonth),
      storageUsedBytes,
      storageLimitBytes: tierLimits.storageCapBytes,
      storageRemainingBytes: Math.max(0, tierLimits.storageCapBytes - storageUsedBytes),
      periodStart: quota.periodStart.toISOString(),
    },
    createdAt: user.createdAt.toISOString(),
    convertedAt: user.convertedAt?.toISOString() ?? null,
    videoCount: user._count.videos,
    sessionCount: user._count.sessions,
  };
}

/**
 * Update user profile.
 */
export async function updateUser(
  id: string,
  data: { name?: string; avatarUrl?: string | null }
): Promise<UserResponse> {
  const user = await prisma.user.update({
    where: { id },
    data: {
      name: data.name,
      avatarUrl: data.avatarUrl,
    },
    include: {
      _count: {
        select: {
          videos: {
            where: { deletedAt: null },
          },
          sessions: {
            where: { deletedAt: null },
          },
        },
      },
    },
  });

  const tier = await getUserTier(id);
  const tierLimits = getTierLimits(tier);
  const quota = await getOrCreateUsageQuota(id);
  const storageUsedBytes = await calculateUserStorageUsed(id);

  return {
    id: user.id,
    email: user.email,
    name: user.name,
    avatarUrl: user.avatarUrl,
    tier,
    tierLimits,
    usage: {
      detectionsUsed: quota.detectionsUsed,
      detectionsLimit: tierLimits.detectionsPerMonth,
      detectionsRemaining: Math.max(
        0,
        tierLimits.detectionsPerMonth - quota.detectionsUsed
      ),
      uploadsThisMonth: quota.uploadsThisMonth,
      uploadsLimit: tierLimits.monthlyUploadCount,
      uploadsRemaining: Math.max(0, tierLimits.monthlyUploadCount - quota.uploadsThisMonth),
      storageUsedBytes,
      storageLimitBytes: tierLimits.storageCapBytes,
      storageRemainingBytes: Math.max(0, tierLimits.storageCapBytes - storageUsedBytes),
      periodStart: quota.periodStart.toISOString(),
    },
    createdAt: user.createdAt.toISOString(),
    convertedAt: user.convertedAt?.toISOString() ?? null,
    videoCount: user._count.videos,
    sessionCount: user._count.sessions,
  };
}

