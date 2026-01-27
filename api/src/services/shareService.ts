import { prisma } from "../lib/prisma.js";
import type { MemberRole } from "@prisma/client";
import {
  ForbiddenError,
  NotFoundError,
} from "../middleware/errorHandler.js";

export type SessionRole = "owner" | MemberRole | null;

/**
 * Check if a user can access a session (either as owner or member).
 * Returns the specific role: "owner", "VIEWER", "EDITOR", "ADMIN", or null.
 */
export async function canAccessSession(
  sessionId: string,
  userId: string
): Promise<{ hasAccess: boolean; role: SessionRole }> {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: {
      share: {
        include: {
          members: {
            where: { userId },
            select: { role: true },
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

  // Check if user is a member via share - return their specific role
  if (session.share && session.share.members.length > 0) {
    return { hasAccess: true, role: session.share.members[0].role };
  }

  return { hasAccess: false, role: null };
}

/**
 * Check if user can access a video's rallies through session membership.
 * Used by rally service for permission checks.
 * Throws NotFoundError if the video doesn't exist.
 */
export async function canAccessVideoRallies(
  videoId: string,
  userId: string,
  requireEdit: boolean
): Promise<boolean> {
  // Fast path: check direct ownership
  const video = await prisma.video.findUnique({
    where: { id: videoId },
    select: { userId: true },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }
  if (video.userId === userId) return true;

  // Check via session membership
  const sessionVideo = await prisma.sessionVideo.findFirst({
    where: { videoId },
    include: {
      session: {
        include: {
          share: {
            include: {
              members: {
                where: { userId },
                select: { role: true },
              },
            },
          },
        },
      },
    },
  });

  if (!sessionVideo?.session.share?.members.length) return false;

  const role = sessionVideo.session.share.members[0].role;

  if (requireEdit) {
    return role === "EDITOR" || role === "ADMIN";
  }

  // VIEWER, EDITOR, ADMIN can all read
  return true;
}

/**
 * Create or get existing share link for a session (owner or admin).
 * Uses a single query to check permissions and share existence.
 */
export async function createShare(
  sessionId: string,
  actorId: string,
  defaultRole?: MemberRole
) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: {
      share: {
        include: {
          members: {
            where: { userId: actorId },
            select: { role: true },
          },
        },
      },
    },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  // Check permission: must be owner or admin
  const isOwner = session.userId === actorId;
  const memberRole = session.share?.members[0]?.role;

  if (!isOwner && memberRole !== "ADMIN") {
    throw new ForbiddenError("Only the session owner or admin can share it");
  }

  // Return existing share if one exists
  if (session.share) {
    return session.share;
  }

  // Create new share
  return prisma.sessionShare.create({
    data: {
      sessionId,
      ...(defaultRole && { defaultRole }),
    },
  });
}

/**
 * Get share info including members (owner or admin)
 */
export async function getShare(sessionId: string, actorId: string) {
  const { role } = await canAccessSession(sessionId, actorId);

  if (role !== "owner" && role !== "ADMIN") {
    throw new ForbiddenError(
      "Only the session owner or admin can view share info"
    );
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
 * Remove a specific member from a shared session (owner or admin).
 * Admin cannot remove other admins.
 * Uses a single query to fetch session, share, and both actor/target members.
 */
export async function removeMember(
  sessionId: string,
  actorId: string,
  memberUserId: string
) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: {
      share: {
        include: {
          members: {
            where: { userId: { in: [actorId, memberUserId] } },
          },
        },
      },
    },
  });

  if (!session) {
    throw new NotFoundError("Session", sessionId);
  }

  if (!session.share) {
    throw new NotFoundError("Share not found for this session");
  }

  // Determine actor role
  const isOwner = session.userId === actorId;
  const actorMember = session.share.members.find((m) => m.userId === actorId);
  const actorRole = isOwner ? "owner" : actorMember?.role ?? null;

  if (actorRole !== "owner" && actorRole !== "ADMIN") {
    throw new ForbiddenError("Only the session owner or admin can remove members");
  }

  // Find target member
  const targetMember = session.share.members.find((m) => m.userId === memberUserId);
  if (!targetMember) {
    throw new NotFoundError("Member not found");
  }

  // Admin cannot remove other admins
  if (actorRole === "ADMIN" && targetMember.role === "ADMIN") {
    throw new ForbiddenError("Admins cannot remove other admins");
  }

  await prisma.sessionMember.delete({
    where: { id: targetMember.id },
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
    defaultRole: share.defaultRole,
  };
}

/**
 * Accept a share invite (any user with valid token)
 * Optionally updates user's name if provided.
 * Assigns the share's defaultRole to the new member.
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
    return {
      sessionId: share.session.id,
      alreadyMember: true,
      role: existingMember.role,
    };
  }

  // Add as member with the share's default role
  const member = await prisma.sessionMember.create({
    data: {
      sessionShareId: share.id,
      userId,
      role: share.defaultRole,
    },
  });

  return { sessionId: share.session.id, role: member.role };
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
    role: m.role,
  }));
}

/**
 * Update a member's role.
 * Owner can set any role. Admin can set viewer/editor only (not admin).
 * Nobody can change their own role.
 */
export async function updateMemberRole(
  sessionId: string,
  actorId: string,
  targetUserId: string,
  newRole: MemberRole
) {
  const { role: actorRole } = await canAccessSession(sessionId, actorId);

  if (actorRole !== "owner" && actorRole !== "ADMIN") {
    throw new ForbiddenError("Only the session owner or admin can change roles");
  }

  // Admin cannot promote to admin
  if (actorRole === "ADMIN" && newRole === "ADMIN") {
    throw new ForbiddenError("Only the owner can promote members to admin");
  }

  const share = await prisma.sessionShare.findUnique({
    where: { sessionId },
  });

  if (!share) {
    throw new NotFoundError("Share not found for this session");
  }

  const member = await prisma.sessionMember.findFirst({
    where: {
      sessionShareId: share.id,
      userId: targetUserId,
    },
  });

  if (!member) {
    throw new NotFoundError("Member not found");
  }

  // Admin cannot change other admins' roles
  if (actorRole === "ADMIN" && member.role === "ADMIN") {
    throw new ForbiddenError("Admins cannot change other admins' roles");
  }

  return prisma.sessionMember.update({
    where: { id: member.id },
    data: { role: newRole },
  });
}

/**
 * Update the default role assigned to new members who join via the share link.
 * Owner or admin can change this.
 */
export async function updateDefaultRole(
  sessionId: string,
  actorId: string,
  newDefaultRole: MemberRole
) {
  const { role: actorRole } = await canAccessSession(sessionId, actorId);

  if (actorRole !== "owner" && actorRole !== "ADMIN") {
    throw new ForbiddenError(
      "Only the session owner or admin can change the default role"
    );
  }

  const share = await prisma.sessionShare.findUnique({
    where: { sessionId },
  });

  if (!share) {
    throw new NotFoundError("Share not found for this session");
  }

  return prisma.sessionShare.update({
    where: { id: share.id },
    data: { defaultRole: newDefaultRole },
  });
}
