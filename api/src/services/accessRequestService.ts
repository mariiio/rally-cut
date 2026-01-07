import { prisma } from "../lib/prisma.js";
import {
  ForbiddenError,
  NotFoundError,
  ConflictError,
} from "../middleware/errorHandler.js";

/**
 * Create an access request for a session
 */
export async function createAccessRequest(
  sessionId: string,
  userId: string,
  message?: string
) {
  // Find session and check if user already has access
  const session = await prisma.session.findUnique({
    where: { id: sessionId, deletedAt: null },
    include: {
      share: {
        include: {
          members: {
            where: { userId },
          },
        },
      },
    },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  // Check if user is already the owner
  if (session.userId === userId) {
    throw new ConflictError("You are the owner of this session");
  }

  // Check if user is already a member
  if (session.share?.members && session.share.members.length > 0) {
    throw new ConflictError("You already have access to this session");
  }

  // Check if there's already a pending request
  const existingRequest = await prisma.accessRequest.findUnique({
    where: {
      sessionId_userId: { sessionId, userId },
    },
  });

  if (existingRequest) {
    if (existingRequest.status === "PENDING") {
      throw new ConflictError("You already have a pending access request");
    }
    if (existingRequest.status === "REJECTED") {
      // Allow re-requesting after rejection by updating the existing request
      return prisma.accessRequest.update({
        where: { id: existingRequest.id },
        data: {
          status: "PENDING",
          message,
          requestedAt: new Date(),
          resolvedAt: null,
          resolvedBy: null,
        },
      });
    }
  }

  // Create new request
  return prisma.accessRequest.create({
    data: {
      sessionId,
      userId,
      message,
    },
  });
}

/**
 * Get pending access requests for a session (owner only)
 */
export async function getPendingRequests(sessionId: string, ownerId: string) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId, deletedAt: null },
    select: { userId: true },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session.userId !== ownerId) {
    throw new ForbiddenError("Only the session owner can view access requests");
  }

  const requests = await prisma.accessRequest.findMany({
    where: {
      sessionId,
      status: "PENDING",
    },
    include: {
      user: {
        select: {
          id: true,
          name: true,
          email: true,
          avatarUrl: true,
        },
      },
    },
    orderBy: { requestedAt: "asc" },
  });

  return requests.map((r) => ({
    id: r.id,
    userId: r.userId,
    userName: r.user.name,
    userEmail: r.user.email,
    userAvatarUrl: r.user.avatarUrl,
    message: r.message,
    requestedAt: r.requestedAt,
  }));
}

/**
 * Accept an access request (owner only)
 * Creates a SessionMember and updates request status
 */
export async function acceptRequest(
  sessionId: string,
  requestId: string,
  ownerId: string
) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId, deletedAt: null },
    include: { share: true },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session.userId !== ownerId) {
    throw new ForbiddenError("Only the session owner can accept requests");
  }

  const request = await prisma.accessRequest.findUnique({
    where: { id: requestId },
  });

  if (!request || request.sessionId !== sessionId) {
    throw new NotFoundError("Access request", requestId);
  }

  if (request.status !== "PENDING") {
    throw new ConflictError("This request has already been resolved");
  }

  // Create share if it doesn't exist (needed for members)
  let share = session.share;
  if (!share) {
    share = await prisma.sessionShare.create({
      data: { sessionId },
    });
  }

  // Create member and update request in a transaction
  await prisma.$transaction([
    prisma.sessionMember.create({
      data: {
        sessionShareId: share.id,
        userId: request.userId,
      },
    }),
    prisma.accessRequest.update({
      where: { id: requestId },
      data: {
        status: "ACCEPTED",
        resolvedAt: new Date(),
        resolvedBy: ownerId,
      },
    }),
  ]);

  return { success: true };
}

/**
 * Reject an access request (owner only)
 */
export async function rejectRequest(
  sessionId: string,
  requestId: string,
  ownerId: string
) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId, deletedAt: null },
    select: { userId: true },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session.userId !== ownerId) {
    throw new ForbiddenError("Only the session owner can reject requests");
  }

  const request = await prisma.accessRequest.findUnique({
    where: { id: requestId },
  });

  if (!request || request.sessionId !== sessionId) {
    throw new NotFoundError("Access request", requestId);
  }

  if (request.status !== "PENDING") {
    throw new ConflictError("This request has already been resolved");
  }

  await prisma.accessRequest.update({
    where: { id: requestId },
    data: {
      status: "REJECTED",
      resolvedAt: new Date(),
      resolvedBy: ownerId,
    },
  });

  return { success: true };
}

/**
 * Get count of pending access requests (owner only)
 */
export async function getPendingCount(sessionId: string, ownerId: string) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId, deletedAt: null },
    select: { userId: true },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session.userId !== ownerId) {
    throw new ForbiddenError("Only the session owner can view access requests");
  }

  const count = await prisma.accessRequest.count({
    where: {
      sessionId,
      status: "PENDING",
    },
  });

  return { pending: count };
}
