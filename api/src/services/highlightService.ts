import { prisma } from "../lib/prisma.js";
import {
  ConflictError,
  ForbiddenError,
  NotFoundError,
} from "../middleware/errorHandler.js";
import type {
  AddRallyToHighlightInput,
  CreateHighlightInput,
  UpdateHighlightInput,
} from "../schemas/highlight.js";

export async function createHighlight(
  sessionId: string,
  data: CreateHighlightInput,
  userId?: string
) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
  });

  if (session === null) {
    throw new NotFoundError("Session", sessionId);
  }

  // Get user name for default highlight name
  let defaultName = data.name;
  if (!defaultName && userId) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { name: true },
    });
    if (user?.name) {
      defaultName = `${user.name}'s Highlight`;
    }
  }

  // Fallback name if still not set
  if (!defaultName) {
    const count = await prisma.highlight.count({ where: { sessionId } });
    defaultName = `Highlight ${count + 1}`;
  }

  return prisma.highlight.create({
    data: {
      sessionId,
      createdByUserId: userId,
      name: defaultName,
      color: data.color,
    },
  });
}

export async function updateHighlight(
  id: string,
  data: UpdateHighlightInput,
  userId?: string
) {
  const highlight = await prisma.highlight.findUnique({
    where: { id },
  });

  if (highlight === null) {
    throw new NotFoundError("Highlight", id);
  }

  // Only creator can edit (if userId is provided for check)
  if (userId && highlight.createdByUserId && highlight.createdByUserId !== userId) {
    throw new ForbiddenError("Only the highlight creator can edit it");
  }

  return prisma.highlight.update({
    where: { id },
    data,
  });
}

export async function deleteHighlight(id: string, userId?: string) {
  const highlight = await prisma.highlight.findUnique({
    where: { id },
  });

  if (highlight === null) {
    throw new NotFoundError("Highlight", id);
  }

  // Only creator can delete (if userId is provided for check)
  if (userId && highlight.createdByUserId && highlight.createdByUserId !== userId) {
    throw new ForbiddenError("Only the highlight creator can delete it");
  }

  await prisma.highlight.delete({
    where: { id },
  });
}

export async function addRallyToHighlight(
  highlightId: string,
  data: AddRallyToHighlightInput
) {
  const highlight = await prisma.highlight.findUnique({
    where: { id: highlightId },
  });

  if (highlight === null) {
    throw new NotFoundError("Highlight", highlightId);
  }

  const rally = await prisma.rally.findUnique({
    where: { id: data.rallyId },
    include: {
      video: {
        include: {
          sessionVideos: {
            where: { sessionId: highlight.sessionId },
          },
        },
      },
    },
  });

  if (rally === null) {
    throw new NotFoundError("Rally", data.rallyId);
  }

  // Check if the video is in the same session as the highlight
  if (rally.video.sessionVideos.length === 0) {
    throw new ConflictError("Rally does not belong to a video in this session");
  }

  const existing = await prisma.highlightRally.findFirst({
    where: { highlightId, rallyId: data.rallyId },
  });

  if (existing !== null) {
    throw new ConflictError("Rally already in highlight");
  }

  const maxOrder = await prisma.highlightRally.aggregate({
    where: { highlightId },
    _max: { order: true },
  });

  return prisma.highlightRally.create({
    data: {
      highlightId,
      rallyId: data.rallyId,
      order: data.order ?? (maxOrder._max.order ?? -1) + 1,
    },
    include: { rally: true },
  });
}

export async function removeRallyFromHighlight(
  highlightId: string,
  rallyId: string
) {
  const highlightRally = await prisma.highlightRally.findFirst({
    where: { highlightId, rallyId },
  });

  if (highlightRally === null) {
    throw new NotFoundError("HighlightRally");
  }

  await prisma.highlightRally.delete({
    where: { id: highlightRally.id },
  });
}
