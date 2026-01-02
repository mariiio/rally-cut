import { prisma } from "../lib/prisma.js";
import { NotFoundError, ValidationError } from "../middleware/errorHandler.js";
import type { BatchOperation, BatchResponse } from "../schemas/batch.js";

export async function processBatch(
  sessionId: string,
  userId: string | undefined,
  operations: BatchOperation[]
): Promise<BatchResponse> {
  const where = userId ? { id: sessionId, userId } : { id: sessionId };

  const session = await prisma.session.findFirst({ where });

  if (session === null) {
    throw new NotFoundError("Session", sessionId);
  }

  const created: Record<string, string> = {};

  await prisma.$transaction(async (tx) => {
    for (const op of operations) {
      if (op.type === "create") {
        if (op.entity === "rally") {
          // Check video is in this session via junction
          const sessionVideo = await tx.sessionVideo.findFirst({
            where: { sessionId, videoId: op.data.videoId },
          });
          if (sessionVideo === null) {
            throw new ValidationError(
              `Video ${op.data.videoId} not found in session`
            );
          }
          const maxOrder = await tx.rally.aggregate({
            where: { videoId: op.data.videoId },
            _max: { order: true },
          });
          const rally = await tx.rally.create({
            data: {
              videoId: op.data.videoId,
              startMs: op.data.startMs,
              endMs: op.data.endMs,
              confidence: op.data.confidence,
              scoreA: op.data.scoreA,
              scoreB: op.data.scoreB,
              servingTeam: op.data.servingTeam,
              notes: op.data.notes,
              order: (maxOrder._max.order ?? -1) + 1,
            },
          });
          created[op.tempId] = rally.id;
        } else if (op.entity === "highlight") {
          const highlight = await tx.highlight.create({
            data: {
              sessionId,
              name: op.data.name,
              color: op.data.color,
            },
          });
          created[op.tempId] = highlight.id;
        }
      } else if (op.type === "update") {
        if (op.entity === "video") {
          // Check video is in this session via junction
          const sessionVideo = await tx.sessionVideo.findFirst({
            where: { sessionId, videoId: op.id },
          });
          if (sessionVideo === null) {
            throw new ValidationError(`Video ${op.id} not found in session`);
          }
          await tx.video.update({
            where: { id: op.id },
            data: op.data,
          });
        } else if (op.entity === "rally") {
          const rally = await tx.rally.findFirst({
            where: {
              id: op.id,
              video: {
                sessionVideos: { some: { sessionId } },
              },
            },
          });
          if (rally === null) {
            throw new ValidationError(`Rally ${op.id} not found in session`);
          }
          await tx.rally.update({
            where: { id: op.id },
            data: op.data,
          });
        } else if (op.entity === "highlight") {
          const highlight = await tx.highlight.findFirst({
            where: { id: op.id, sessionId },
          });
          if (highlight === null) {
            throw new ValidationError(
              `Highlight ${op.id} not found in session`
            );
          }
          await tx.highlight.update({
            where: { id: op.id },
            data: op.data,
          });
        }
      } else if (op.type === "delete") {
        if (op.entity === "video") {
          // In batch context, delete = remove from session (not delete video)
          const sessionVideo = await tx.sessionVideo.findFirst({
            where: { sessionId, videoId: op.id },
          });
          if (sessionVideo === null) {
            throw new ValidationError(`Video ${op.id} not found in session`);
          }
          // Remove from session, video remains in library
          await tx.sessionVideo.delete({ where: { id: sessionVideo.id } });
        } else if (op.entity === "rally") {
          const rally = await tx.rally.findFirst({
            where: {
              id: op.id,
              video: {
                sessionVideos: { some: { sessionId } },
              },
            },
          });
          if (rally === null) {
            throw new ValidationError(`Rally ${op.id} not found in session`);
          }
          await tx.rally.delete({ where: { id: op.id } });
        } else if (op.entity === "highlight") {
          const highlight = await tx.highlight.findFirst({
            where: { id: op.id, sessionId },
          });
          if (highlight === null) {
            throw new ValidationError(
              `Highlight ${op.id} not found in session`
            );
          }
          await tx.highlight.delete({ where: { id: op.id } });
        } else if (op.entity === "highlightRally") {
          const highlightRally = await tx.highlightRally.findFirst({
            where: {
              id: op.id,
              highlight: { sessionId },
            },
          });
          if (highlightRally === null) {
            throw new ValidationError(
              `HighlightRally ${op.id} not found in session`
            );
          }
          await tx.highlightRally.delete({ where: { id: op.id } });
        }
      } else if (op.type === "reorder") {
        if (op.entity === "video") {
          // Reorder updates SessionVideo.order, not Video.order
          for (let i = 0; i < op.order.length; i++) {
            const videoId = op.order[i];
            if (videoId === undefined) continue;
            await tx.sessionVideo.updateMany({
              where: { sessionId, videoId },
              data: { order: i },
            });
          }
        } else if (op.entity === "rally") {
          // Check video is in session
          const sessionVideo = await tx.sessionVideo.findFirst({
            where: { sessionId, videoId: op.parentId },
          });
          if (sessionVideo === null) {
            throw new ValidationError(
              `Video ${op.parentId} not found in session`
            );
          }
          for (let i = 0; i < op.order.length; i++) {
            const rallyId = op.order[i];
            if (rallyId === undefined) continue;
            await tx.rally.updateMany({
              where: { id: rallyId, videoId: op.parentId },
              data: { order: i },
            });
          }
        } else if (op.entity === "highlightRally") {
          const highlight = await tx.highlight.findFirst({
            where: { id: op.parentId, sessionId },
          });
          if (highlight === null) {
            throw new ValidationError(
              `Highlight ${op.parentId} not found in session`
            );
          }
          for (let i = 0; i < op.order.length; i++) {
            const hrId = op.order[i];
            if (hrId === undefined) continue;
            await tx.highlightRally.updateMany({
              where: { id: hrId, highlightId: op.parentId },
              data: { order: i },
            });
          }
        }
      } else if (op.type === "addRallyToHighlight") {
        const highlight = await tx.highlight.findFirst({
          where: { id: op.highlightId, sessionId },
        });
        if (highlight === null) {
          throw new ValidationError(
            `Highlight ${op.highlightId} not found in session`
          );
        }
        const rally = await tx.rally.findFirst({
          where: {
            id: op.rallyId,
            video: {
              sessionVideos: { some: { sessionId } },
            },
          },
        });
        if (rally === null) {
          throw new ValidationError(`Rally ${op.rallyId} not found in session`);
        }
        const maxOrder = await tx.highlightRally.aggregate({
          where: { highlightId: op.highlightId },
          _max: { order: true },
        });
        const hr = await tx.highlightRally.create({
          data: {
            highlightId: op.highlightId,
            rallyId: op.rallyId,
            order: (maxOrder._max.order ?? -1) + 1,
          },
        });
        if (op.tempId !== undefined) {
          created[op.tempId] = hr.id;
        }
      }
    }

    await tx.session.update({
      where: { id: sessionId },
      data: { updatedAt: new Date() },
    });
  });

  return {
    success: true,
    created,
    updatedAt: new Date().toISOString(),
  };
}
