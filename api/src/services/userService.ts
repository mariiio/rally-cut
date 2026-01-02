import { prisma } from "../lib/prisma.js";
import { NotFoundError } from "../middleware/errorHandler.js";
import type { UserResponse, IdentityResponse } from "../schemas/user.js";

/**
 * Get or create a user from a visitor ID.
 * If the visitor ID exists, returns the associated user.
 * If not, creates a new anonymous user and links it to the visitor ID.
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

  // Create new user and identity in a transaction
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
}

/**
 * Get user by ID with counts.
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
          sessions: true,
        },
      },
    },
  });

  if (!user) {
    throw new NotFoundError("User", id);
  }

  return {
    id: user.id,
    email: user.email,
    name: user.name,
    avatarUrl: user.avatarUrl,
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
  data: { name?: string }
): Promise<UserResponse> {
  const user = await prisma.user.update({
    where: { id },
    data: {
      name: data.name,
    },
    include: {
      _count: {
        select: {
          videos: {
            where: { deletedAt: null },
          },
          sessions: true,
        },
      },
    },
  });

  return {
    id: user.id,
    email: user.email,
    name: user.name,
    avatarUrl: user.avatarUrl,
    createdAt: user.createdAt.toISOString(),
    convertedAt: user.convertedAt?.toISOString() ?? null,
    videoCount: user._count.videos,
    sessionCount: user._count.sessions,
  };
}

