import { prisma } from "../lib/prisma.js";
import { NotFoundError } from "../middleware/errorHandler.js";
import type { SyncStateInput } from "../schemas/sync.js";

/**
 * Sync the full state from the frontend.
 * This replaces the current rallies/highlights with the provided state.
 */
export async function syncState(
  sessionId: string,
  input: SyncStateInput
): Promise<{ success: boolean; syncedAt: string }> {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: {
      videos: {
        select: { id: true },
      },
    },
  });

  if (session === null) {
    throw new NotFoundError("Session", sessionId);
  }

  const videoIds = new Set(session.videos.map((v) => v.id));

  await prisma.$transaction(async (tx) => {
    // Sync rallies for each video
    for (const [videoId, rallies] of Object.entries(input.ralliesPerVideo)) {
      if (!videoIds.has(videoId)) {
        continue; // Skip unknown videos
      }

      // Get existing rallies for this video
      const existingRallies = await tx.rally.findMany({
        where: { videoId },
        select: { id: true },
      });
      const existingIds = new Set(existingRallies.map((r) => r.id));

      // Track which IDs we've seen
      const seenIds = new Set<string>();

      // Update or create rallies
      for (let order = 0; order < rallies.length; order++) {
        const rally = rallies[order];
        if (!rally) continue;

        if (rally.id && existingIds.has(rally.id)) {
          // Update existing rally
          await tx.rally.update({
            where: { id: rally.id },
            data: {
              startMs: rally.startMs,
              endMs: rally.endMs,
              order,
            },
          });
          seenIds.add(rally.id);
        } else {
          // Create new rally
          const created = await tx.rally.create({
            data: {
              videoId,
              startMs: rally.startMs,
              endMs: rally.endMs,
              order,
            },
          });
          seenIds.add(created.id);
        }
      }

      // Delete rallies that are no longer in the list
      const toDelete = [...existingIds].filter((id) => !seenIds.has(id));
      if (toDelete.length > 0) {
        await tx.rally.deleteMany({
          where: { id: { in: toDelete } },
        });
      }
    }

    // Sync highlights
    // Get existing highlights
    const existingHighlights = await tx.highlight.findMany({
      where: { sessionId },
      include: { highlightRallies: true },
    });
    const existingHighlightIds = new Set(existingHighlights.map((h) => h.id));
    const seenHighlightIds = new Set<string>();

    // Get all rally IDs in this session for validation
    const allRallies = await tx.rally.findMany({
      where: { video: { sessionId } },
      select: { id: true, videoId: true, order: true },
    });

    // Build a map from frontend rally ID pattern to backend ID
    // Frontend uses: `${videoId}_rally_${order + 1}`
    const rallyIdMap = new Map<string, string>();
    for (const r of allRallies) {
      const frontendId = `${r.videoId}_rally_${r.order + 1}`;
      rallyIdMap.set(frontendId, r.id);
    }

    for (const highlight of input.highlights) {
      if (highlight.id && existingHighlightIds.has(highlight.id)) {
        // Update existing highlight
        await tx.highlight.update({
          where: { id: highlight.id },
          data: {
            name: highlight.name,
            color: highlight.color,
          },
        });
        seenHighlightIds.add(highlight.id);

        // Sync highlight rallies
        await tx.highlightRally.deleteMany({
          where: { highlightId: highlight.id },
        });

        for (let order = 0; order < highlight.rallyIds.length; order++) {
          const frontendRallyId = highlight.rallyIds[order];
          if (!frontendRallyId) continue;

          // Map frontend ID to backend ID
          const backendRallyId = rallyIdMap.get(frontendRallyId);
          if (backendRallyId) {
            await tx.highlightRally.create({
              data: {
                highlightId: highlight.id,
                rallyId: backendRallyId,
                order,
              },
            });
          }
        }
      } else {
        // Create new highlight
        const created = await tx.highlight.create({
          data: {
            sessionId,
            name: highlight.name,
            color: highlight.color,
          },
        });
        seenHighlightIds.add(created.id);

        // Add highlight rallies
        for (let order = 0; order < highlight.rallyIds.length; order++) {
          const frontendRallyId = highlight.rallyIds[order];
          if (!frontendRallyId) continue;

          const backendRallyId = rallyIdMap.get(frontendRallyId);
          if (backendRallyId) {
            await tx.highlightRally.create({
              data: {
                highlightId: created.id,
                rallyId: backendRallyId,
                order,
              },
            });
          }
        }
      }
    }

    // Delete highlights that are no longer in the list
    const highlightsToDelete = [...existingHighlightIds].filter(
      (id) => !seenHighlightIds.has(id)
    );
    if (highlightsToDelete.length > 0) {
      await tx.highlight.deleteMany({
        where: { id: { in: highlightsToDelete } },
      });
    }

    // Update session timestamp
    await tx.session.update({
      where: { id: sessionId },
      data: { updatedAt: new Date() },
    });
  });

  return { success: true, syncedAt: new Date().toISOString() };
}
