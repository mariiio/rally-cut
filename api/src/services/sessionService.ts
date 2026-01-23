import { prisma } from "../lib/prisma.js";
import {
  NotFoundError,
  ValidationError,
  AccessDeniedError,
} from "../middleware/errorHandler.js";
import type { Pagination } from "../schemas/common.js";
import type {
  CreateSessionInput,
  UpdateSessionInput,
} from "../schemas/session.js";

const ALL_VIDEOS_SESSION_NAME = "All Videos";

// Activity tracking debounce: only update if >1 hour since last update
const ACTIVITY_DEBOUNCE_MS = 60 * 60 * 1000; // 1 hour

/**
 * Update user's lastActiveAt timestamp for inactivity-based cleanup.
 * Uses debouncing to avoid excessive database writes.
 * Fire-and-forget: doesn't block the calling function.
 */
async function trackUserActivity(userId: string): Promise<void> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { lastActiveAt: true },
  });

  if (!user) return;

  const now = Date.now();
  const lastActive = user.lastActiveAt?.getTime() ?? 0;

  // Only update if more than 1 hour since last activity
  if (now - lastActive > ACTIVITY_DEBOUNCE_MS) {
    prisma.user
      .update({
        where: { id: userId },
        data: { lastActiveAt: new Date() },
      })
      .catch((err) => {
        console.error("[trackUserActivity] Failed to update lastActiveAt:", err);
      });
  }
}

// Helper to convert BigInt fields to strings for JSON serialization
function serializeBigInts<T>(obj: T): T {
  return JSON.parse(
    JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value
    )
  );
}

export async function createSession(data: CreateSessionInput, userId?: string) {
  return prisma.session.create({
    data: {
      name: data.name,
      userId,
      type: "REGULAR",
      // No time-based expiration - videos kept until 2 months of user inactivity
      expiresAt: null,
    },
  });
}

/**
 * Get or create the "All Videos" session for a user.
 * This special session automatically contains all of the user's videos.
 */
export async function getOrCreateAllVideosSession(userId: string) {
  // Try to find existing ALL_VIDEOS session
  let session = await prisma.session.findFirst({
    where: { userId, type: "ALL_VIDEOS", deletedAt: null },
  });

  if (!session) {
    session = await prisma.session.create({
      data: {
        userId,
        name: ALL_VIDEOS_SESSION_NAME,
        type: "ALL_VIDEOS",
        // No time-based expiration - videos kept until 2 months of user inactivity
        expiresAt: null,
      },
    });
  }

  return session;
}

export async function listSessions(pagination: Pagination, userId?: string) {
  const skip = (pagination.page - 1) * pagination.limit;

  const where = userId ? { userId, deletedAt: null } : { deletedAt: null };

  const [sessions, total] = await Promise.all([
    prisma.session.findMany({
      where,
      orderBy: [
        { type: "asc" }, // ALL_VIDEOS first (alphabetically before REGULAR)
        { updatedAt: "desc" },
      ],
      skip,
      take: pagination.limit,
      include: {
        _count: {
          select: { sessionVideos: true, highlights: true },
        },
      },
    }),
    prisma.session.count({ where }),
  ]);

  return {
    sessions: sessions.map((s) => ({
      ...s,
      type: s.type,
      videoCount: s._count.sessionVideos,
      highlightCount: s._count.highlights,
    })),
    total,
  };
}

export async function getSessionById(id: string, userId?: string) {
  // Combined query: fetch session with all related data + access check in one query
  const session = await prisma.session.findFirst({
    where: { id, deletedAt: null },
    include: {
      sessionVideos: {
        orderBy: { order: "asc" },
        include: {
          video: {
            include: {
              rallies: {
                orderBy: { order: "asc" },
                include: {
                  cameraEdit: {
                    include: {
                      keyframes: {
                        orderBy: { timeOffset: "asc" },
                      },
                    },
                  },
                },
              },
              confirmation: true,
              cameraSettings: true,
            },
          },
        },
      },
      highlights: {
        orderBy: { createdAt: "asc" },
        include: {
          highlightRallies: {
            orderBy: { order: "asc" },
            include: {
              rally: true,
            },
          },
          createdByUser: {
            select: {
              id: true,
              name: true,
            },
          },
        },
      },
      user: {
        select: {
          id: true,
          name: true,
        },
      },
      // Include share membership check for access validation (always include for consistent type)
      share: {
        include: {
          members: userId
            ? {
                where: { userId },
                select: { id: true },
              }
            : false,
        },
      },
      // Include pending access request check for access denied response
      accessRequests: userId
        ? {
            where: { userId, status: "PENDING" },
            select: { id: true },
          }
        : undefined,
    },
  });

  if (session === null) {
    throw new NotFoundError("Session", id);
  }

  // Compute user role from combined query result
  let userRole: "owner" | "member" | null = null;
  if (userId) {
    if (session.userId === userId) {
      userRole = "owner";
    } else if (session.share?.members && session.share.members.length > 0) {
      userRole = "member";
    } else {
      // User has no access - throw AccessDeniedError with session info
      const hasPendingRequest =
        session.accessRequests && session.accessRequests.length > 0;
      throw new AccessDeniedError(
        session.name,
        session.user?.name ?? null,
        hasPendingRequest
      );
    }

    // Track activity for inactivity-based cleanup (fire-and-forget)
    void trackUserActivity(userId);
  }

  // Transform to expected format with videos array
  // For confirmed videos, use confirmation proxy and duration
  const transformed = {
    ...session,
    userRole,
    videos: session.sessionVideos.map((sv) => {
      const video = sv.video;
      const isConfirmed = video.confirmation?.status === "CONFIRMED";
      return {
        ...video,
        order: sv.order,
        // For confirmed videos, use confirmation proxy for editing
        proxyS3Key: isConfirmed
          ? video.confirmation?.proxyS3Key ?? video.proxyS3Key
          : video.proxyS3Key,
        // For confirmed videos, use trimmed duration
        durationMs: isConfirmed
          ? video.confirmation?.trimmedDurationMs ?? video.durationMs
          : video.durationMs,
        // Clear processedS3Key for confirmed (use proxy instead)
        processedS3Key: isConfirmed ? null : video.processedS3Key,
      };
    }),
  };

  // Remove internal fields from response (sessionVideos replaced by videos, share/accessRequests for access check)
  const { sessionVideos: _sessionVideos, share: _share, accessRequests: _accessRequests, ...result } = transformed;

  // Convert BigInt fields to strings for JSON serialization
  return serializeBigInts(result);
}

export async function updateSession(
  id: string,
  data: UpdateSessionInput,
  userId?: string
) {
  const where = userId ? { id, userId } : { id };

  const session = await prisma.session.findFirst({ where });

  if (session === null) {
    throw new NotFoundError("Session", id);
  }

  return prisma.session.update({
    where: { id },
    data,
  });
}

export async function deleteSession(id: string, userId?: string) {
  const where = userId ? { id, userId } : { id };

  const session = await prisma.session.findFirst({
    where,
    include: {
      sessionVideos: {
        include: {
          video: {
            select: { s3Key: true },
          },
        },
      },
    },
  });

  if (session === null) {
    throw new NotFoundError("Session", id);
  }

  // Prevent deletion of ALL_VIDEOS session
  if (session.type === "ALL_VIDEOS") {
    throw new ValidationError("Cannot delete the 'All Videos' session");
  }

  await prisma.session.delete({
    where: { id },
  });

  // Note: Videos are NOT deleted, only the session and SessionVideo junctions
  // Return s3Keys for reference but don't delete from S3
  return session.sessionVideos.map((sv) => sv.video.s3Key);
}
