import { prisma } from "../lib/prisma.js";
import { ConflictError, NotFoundError } from "../middleware/errorHandler.js";
import type {
  AddRallyToHighlightInput,
  CreateHighlightInput,
  UpdateHighlightInput,
} from "../schemas/highlight.js";

export async function createHighlight(
  sessionId: string,
  data: CreateHighlightInput
) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
  });

  if (session === null) {
    throw new NotFoundError("Session", sessionId);
  }

  return prisma.highlight.create({
    data: {
      sessionId,
      name: data.name,
      color: data.color,
    },
  });
}

export async function updateHighlight(id: string, data: UpdateHighlightInput) {
  const highlight = await prisma.highlight.findUnique({
    where: { id },
  });

  if (highlight === null) {
    throw new NotFoundError("Highlight", id);
  }

  return prisma.highlight.update({
    where: { id },
    data,
  });
}

export async function deleteHighlight(id: string) {
  const highlight = await prisma.highlight.findUnique({
    where: { id },
  });

  if (highlight === null) {
    throw new NotFoundError("Highlight", id);
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
    include: { video: true },
  });

  if (rally === null) {
    throw new NotFoundError("Rally", data.rallyId);
  }

  if (rally.video.sessionId !== highlight.sessionId) {
    throw new ConflictError("Rally does not belong to the same session");
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
