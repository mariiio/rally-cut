import { prisma } from "../lib/prisma.js";
import { NotFoundError, ValidationError } from "../middleware/errorHandler.js";
import type { Pagination } from "../schemas/common.js";
import type {
  CreateSessionInput,
  UpdateSessionInput,
} from "../schemas/session.js";
import { canAccessSession } from "./shareService.js";
import { getUserTier, calculateExpirationDate } from "./tierService.js";

const ALL_VIDEOS_SESSION_NAME = "All Videos";

// Helper to convert BigInt fields to strings for JSON serialization
function serializeBigInts<T>(obj: T): T {
  return JSON.parse(
    JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value
    )
  );
}

export async function createSession(data: CreateSessionInput, userId?: string) {
  const tier = await getUserTier(userId);
  const expiresAt = calculateExpirationDate(tier);

  return prisma.session.create({
    data: {
      name: data.name,
      userId,
      type: "REGULAR",
      expiresAt,
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
    const tier = await getUserTier(userId);
    const expiresAt = calculateExpirationDate(tier);

    session = await prisma.session.create({
      data: {
        userId,
        name: ALL_VIDEOS_SESSION_NAME,
        type: "ALL_VIDEOS",
        expiresAt,
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
  // Check access if userId is provided
  let userRole: "owner" | "member" | null = null;
  if (userId) {
    const access = await canAccessSession(id, userId);
    if (!access.hasAccess) {
      throw new NotFoundError("Session", id);
    }
    userRole = access.role;
  }

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
              },
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
    },
  });

  if (session === null) {
    throw new NotFoundError("Session", id);
  }

  // Transform to expected format with videos array
  const transformed = {
    ...session,
    userRole,
    videos: session.sessionVideos.map((sv) => ({
      ...sv.video,
      order: sv.order,
    })),
  };

  // Remove sessionVideos from response (replaced by videos)
  const { sessionVideos: _, ...result } = transformed;

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
