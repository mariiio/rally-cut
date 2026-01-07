import { prisma } from "../lib/prisma.js";
import {
  ForbiddenError,
  NotFoundError,
} from "../middleware/errorHandler.js";

/**
 * Check if a user can access a session (either as owner or member)
 */
export async function canAccessSession(
  sessionId: string,
  userId: string
): Promise<{ hasAccess: boolean; role: "owner" | "member" | null }> {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
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
    return { hasAccess: false, role: null };
  }

  // Check if user is owner
  if (session.userId === userId) {
    return { hasAccess: true, role: "owner" };
  }

  // Check if user is a member via share
  if (session.share && session.share.members.length > 0) {
    return { hasAccess: true, role: "member" };
  }

  return { hasAccess: false, role: null };
}

/**
 * Check if user is the session owner
 */
export async function isSessionOwner(
  sessionId: string,
  userId: string
): Promise<boolean> {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    select: { userId: true },
  });

  return session?.userId === userId;
}

/**
 * Create or get existing share link for a session (owner only)
 */
export async function createShare(sessionId: string, ownerId: string) {
  // Verify ownership
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: { share: true },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session.userId !== ownerId) {
    throw new ForbiddenError("Only the session owner can share it");
  }

  // Return existing share if one exists
  if (session.share) {
    return session.share;
  }

  // Create new share
  return prisma.sessionShare.create({
    data: {
      sessionId,
    },
  });
}

/**
 * Get share info including members (owner only)
 */
export async function getShare(sessionId: string, ownerId: string) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    select: { userId: true },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session.userId !== ownerId) {
    throw new ForbiddenError("Only the session owner can view share info");
  }

  return prisma.sessionShare.findUnique({
    where: { sessionId },
    include: {
      members: {
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
        orderBy: { joinedAt: "asc" },
      },
    },
  });
}

/**
 * Delete share and revoke all access (owner only)
 */
export async function deleteShare(sessionId: string, ownerId: string) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    select: { userId: true },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session.userId !== ownerId) {
    throw new ForbiddenError("Only the session owner can delete the share");
  }

  const share = await prisma.sessionShare.findUnique({
    where: { sessionId },
  });

  if (!share) {
    throw new NotFoundError("Share not found for this session");
  }

  // Cascades to delete all members
  await prisma.sessionShare.delete({
    where: { id: share.id },
  });
}

/**
 * Remove a specific member from a shared session (owner only)
 */
export async function removeMember(
  sessionId: string,
  ownerId: string,
  memberUserId: string
) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: { share: true },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (session.userId !== ownerId) {
    throw new ForbiddenError("Only the session owner can remove members");
  }

  if (!session.share) {
    throw new NotFoundError("Share not found for this session");
  }

  const member = await prisma.sessionMember.findFirst({
    where: {
      sessionShareId: session.share.id,
      userId: memberUserId,
    },
  });

  if (!member) {
    throw new NotFoundError("Member not found");
  }

  await prisma.sessionMember.delete({
    where: { id: member.id },
  });
}

/**
 * Get share preview (public - for accept page)
 */
export async function getSharePreview(token: string) {
  const share = await prisma.sessionShare.findUnique({
    where: { token },
    include: {
      session: {
        select: {
          id: true,
          name: true,
          user: {
            select: {
              id: true,
              name: true,
            },
          },
        },
      },
    },
  });

  if (!share) {
    throw new NotFoundError("Share link not found or expired");
  }

  return {
    sessionId: share.session.id,
    sessionName: share.session.name,
    ownerName: share.session.user?.name ?? null,
  };
}

/**
 * Accept a share invite (any user with valid token)
 * Optionally updates user's name if provided
 */
export async function acceptShare(
  token: string,
  userId: string,
  name?: string
) {
  const share = await prisma.sessionShare.findUnique({
    where: { token },
    include: {
      session: {
        select: { id: true, userId: true },
      },
    },
  });

  if (!share) {
    throw new NotFoundError("Share link not found or expired");
  }

  // Update user's name if provided
  if (name) {
    await prisma.user.update({
      where: { id: userId },
      data: { name },
    });
  }

  // Don't allow owner to join as member
  if (share.session.userId === userId) {
    return { sessionId: share.session.id, alreadyOwner: true };
  }

  // Check if already a member
  const existingMember = await prisma.sessionMember.findFirst({
    where: {
      sessionShareId: share.id,
      userId,
    },
  });

  if (existingMember) {
    return { sessionId: share.session.id, alreadyMember: true };
  }

  // Add as member
  await prisma.sessionMember.create({
    data: {
      sessionShareId: share.id,
      userId,
    },
  });

  return { sessionId: share.session.id };
}

/**
 * List sessions shared with a user (not owned by them)
 */
export async function listSharedSessions(userId: string) {
  const memberships = await prisma.sessionMember.findMany({
    where: { userId },
    include: {
      sessionShare: {
        include: {
          session: {
            include: {
              user: {
                select: {
                  id: true,
                  name: true,
                },
              },
              _count: {
                select: { sessionVideos: true, highlights: true },
              },
            },
          },
        },
      },
    },
    orderBy: { joinedAt: "desc" },
  });

  return memberships.map((m) => ({
    ...m.sessionShare.session,
    videoCount: m.sessionShare.session._count.sessionVideos,
    highlightCount: m.sessionShare.session._count.highlights,
    ownerName: m.sessionShare.session.user?.name ?? null,
    joinedAt: m.joinedAt,
  }));
}
