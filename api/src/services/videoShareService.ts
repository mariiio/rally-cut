import { prisma } from "../lib/prisma.js";
import type { MemberRole } from "@prisma/client";
import {
  ForbiddenError,
  NotFoundError,
} from "../middleware/errorHandler.js";

export type VideoRole = "owner" | MemberRole | null;

/**
 * Check if a user can access a video (either as owner or member).
 * Returns the specific role: "owner", "VIEWER", "EDITOR", "ADMIN", or null.
 */
export async function canAccessVideo(
  videoId: string,
  userId: string
): Promise<{ hasAccess: boolean; role: VideoRole }> {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
    include: {
      shares: {
        include: {
          members: {
            where: { userId },
            select: { role: true },
          },
        },
      },
    },
  });

  if (!video) {
    return { hasAccess: false, role: null };
  }

  // Check if user is owner
  if (video.userId === userId) {
    return { hasAccess: true, role: "owner" };
  }

  // Check if user is a member via any share - return their role
  for (const share of video.shares) {
    if (share.members.length > 0) {
      return { hasAccess: true, role: share.members[0].role };
    }
  }

  return { hasAccess: false, role: null };
}

/**
 * Create share links for a video (one for each role).
 * Idempotent: returns existing shares if they already exist.
 * Only owner or admin can create shares.
 */
export async function createVideoShares(videoId: string, actorId: string) {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
    include: {
      shares: {
        include: {
          members: {
            where: { userId: actorId },
            select: { role: true },
          },
        },
      },
    },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  // Check permission: must be owner or admin
  const isOwner = video.userId === actorId;
  const memberRole = video.shares
    .flatMap((s) => s.members)
    .find((m) => m)?.role;

  if (!isOwner && memberRole !== "ADMIN") {
    throw new ForbiddenError("Only the video owner or admin can share it");
  }

  // Return existing shares if they exist (idempotent)
  if (video.shares.length > 0) {
    return video.shares.map((s) => ({
      token: s.token,
      role: s.role,
      createdAt: s.createdAt,
    }));
  }

  // Create all 3 role links atomically
  const roles: MemberRole[] = ["VIEWER", "EDITOR", "ADMIN"];
  const created = await prisma.$transaction(
    roles.map((role) =>
      prisma.videoShare.create({
        data: { videoId, role },
      })
    )
  );

  return created.map((s) => ({
    token: s.token,
    role: s.role,
    createdAt: s.createdAt,
  }));
}

/**
 * Get share info including all share links and members (owner or admin)
 */
export async function getVideoShares(videoId: string, actorId: string) {
  const { role } = await canAccessVideo(videoId, actorId);

  if (role !== "owner" && role !== "ADMIN") {
    throw new ForbiddenError(
      "Only the video owner or admin can view share info"
    );
  }

  const shares = await prisma.videoShare.findMany({
    where: { videoId },
    orderBy: { role: "asc" }, // ADMIN, EDITOR, VIEWER
  });

  // Get all members across all shares
  const members = await prisma.videoMember.findMany({
    where: {
      videoShareId: { in: shares.map((s) => s.id) },
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
    orderBy: { joinedAt: "asc" },
  });

  return {
    shares: shares.map((s) => ({
      token: s.token,
      role: s.role,
      createdAt: s.createdAt,
    })),
    members: members.map((m) => ({
      userId: m.user.id,
      name: m.user.name,
      email: m.user.email,
      avatarUrl: m.user.avatarUrl,
      role: m.role,
      joinedAt: m.joinedAt,
    })),
  };
}

/**
 * Delete all shares and revoke all access (owner only)
 */
export async function deleteVideoShares(videoId: string, ownerId: string) {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
    select: { userId: true },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  if (video.userId !== ownerId) {
    throw new ForbiddenError("Only the video owner can delete the share");
  }

  // Delete all shares (cascades to delete all members)
  await prisma.videoShare.deleteMany({
    where: { videoId },
  });
}

/**
 * Remove a specific member from a shared video (owner or admin).
 * Admin cannot remove other admins.
 */
export async function removeVideoMember(
  videoId: string,
  actorId: string,
  memberUserId: string
) {
  const video = await prisma.video.findUnique({
    where: { id: videoId },
    include: {
      shares: {
        include: {
          members: {
            where: { userId: { in: [actorId, memberUserId] } },
          },
        },
      },
    },
  });

  if (!video) {
    throw new NotFoundError("Video", videoId);
  }

  if (video.shares.length === 0) {
    throw new NotFoundError("Share not found for this video");
  }

  // Determine actor role
  const isOwner = video.userId === actorId;
  const actorMember = video.shares
    .flatMap((s) => s.members)
    .find((m) => m.userId === actorId);
  const actorRole = isOwner ? "owner" : actorMember?.role ?? null;

  if (actorRole !== "owner" && actorRole !== "ADMIN") {
    throw new ForbiddenError("Only the video owner or admin can remove members");
  }

  // Find target member
  const targetMember = video.shares
    .flatMap((s) => s.members)
    .find((m) => m.userId === memberUserId);

  if (!targetMember) {
    throw new NotFoundError("Member not found");
  }

  // Admin cannot remove other admins
  if (actorRole === "ADMIN" && targetMember.role === "ADMIN") {
    throw new ForbiddenError("Admins cannot remove other admins");
  }

  await prisma.videoMember.delete({
    where: { id: targetMember.id },
  });
}

/**
 * Get video share preview (public - for accept page)
 */
export async function getVideoSharePreview(token: string) {
  const share = await prisma.videoShare.findUnique({
    where: { token },
    include: {
      video: {
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
    return null;
  }

  return {
    videoId: share.video.id,
    videoName: share.video.name,
    ownerName: share.video.user?.name ?? null,
    role: share.role,
  };
}

/**
 * Accept a video share invite (any user with valid token)
 * Optionally updates user's name if provided.
 * Assigns the share's role to the new member.
 */
export async function acceptVideoShare(
  token: string,
  userId: string,
  name?: string
) {
  const share = await prisma.videoShare.findUnique({
    where: { token },
    include: {
      video: {
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
  if (share.video.userId === userId) {
    return { videoId: share.video.id, alreadyOwner: true };
  }

  // Check if already a member (in any share for this video)
  const existingMember = await prisma.videoMember.findFirst({
    where: {
      videoShare: { videoId: share.videoId },
      userId,
    },
  });

  if (existingMember) {
    return {
      videoId: share.video.id,
      alreadyMember: true,
      role: existingMember.role,
    };
  }

  // Add as member with the share's role
  const member = await prisma.videoMember.create({
    data: {
      videoShareId: share.id,
      userId,
      role: share.role,
    },
  });

  return { videoId: share.video.id, role: member.role };
}

/**
 * Update a video member's role.
 * Owner can set any role. Admin can set viewer/editor only (not admin).
 * Nobody can change their own role.
 */
export async function updateVideoMemberRole(
  videoId: string,
  actorId: string,
  targetUserId: string,
  newRole: MemberRole
) {
  const { role: actorRole } = await canAccessVideo(videoId, actorId);

  if (actorRole !== "owner" && actorRole !== "ADMIN") {
    throw new ForbiddenError("Only the video owner or admin can change roles");
  }

  // Admin cannot promote to admin
  if (actorRole === "ADMIN" && newRole === "ADMIN") {
    throw new ForbiddenError("Only the owner can promote members to admin");
  }

  // Find the member across all shares for this video
  const member = await prisma.videoMember.findFirst({
    where: {
      videoShare: { videoId },
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

  return prisma.videoMember.update({
    where: { id: member.id },
    data: { role: newRole },
  });
}

