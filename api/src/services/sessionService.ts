import { prisma } from "../lib/prisma.js";
import { NotFoundError } from "../middleware/errorHandler.js";
import type { Pagination } from "../schemas/common.js";
import type {
  CreateSessionInput,
  UpdateSessionInput,
} from "../schemas/session.js";

// Helper to convert BigInt fields to strings for JSON serialization
function serializeBigInts<T>(obj: T): T {
  return JSON.parse(
    JSON.stringify(obj, (_key, value) =>
      typeof value === "bigint" ? value.toString() : value
    )
  );
}

export async function createSession(data: CreateSessionInput) {
  return prisma.session.create({
    data: {
      name: data.name,
    },
  });
}

export async function listSessions(pagination: Pagination) {
  const skip = (pagination.page - 1) * pagination.limit;

  const [sessions, total] = await Promise.all([
    prisma.session.findMany({
      orderBy: { updatedAt: "desc" },
      skip,
      take: pagination.limit,
      include: {
        _count: {
          select: { videos: true, highlights: true },
        },
      },
    }),
    prisma.session.count(),
  ]);

  return { sessions, total };
}

export async function getSessionById(id: string) {
  const session = await prisma.session.findUnique({
    where: { id },
    include: {
      videos: {
        orderBy: { order: "asc" },
        include: {
          rallies: {
            orderBy: { order: "asc" },
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
        },
      },
    },
  });

  if (session === null) {
    throw new NotFoundError("Session", id);
  }

  // Convert BigInt fields to strings for JSON serialization
  return serializeBigInts(session);
}

export async function updateSession(id: string, data: UpdateSessionInput) {
  const session = await prisma.session.findUnique({
    where: { id },
  });

  if (session === null) {
    throw new NotFoundError("Session", id);
  }

  return prisma.session.update({
    where: { id },
    data,
  });
}

export async function deleteSession(id: string) {
  const session = await prisma.session.findUnique({
    where: { id },
    include: {
      videos: {
        select: { s3Key: true },
      },
    },
  });

  if (session === null) {
    throw new NotFoundError("Session", id);
  }

  await prisma.session.delete({
    where: { id },
  });

  return session.videos.map((v) => v.s3Key);
}
