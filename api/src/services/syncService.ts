import { prisma } from "../lib/prisma.js";
import { ForbiddenError, NotFoundError } from "../middleware/errorHandler.js";
import type { SyncStateInput } from "../schemas/sync.js";
import { canAccessSession } from "./shareService.js";
import { getUserTier, getTierLimits } from "./tierService.js";

/**
 * Sync the full state from the frontend.
 * This replaces the current rallies/highlights with the provided state.
 *
 * Permission rules:
 * - Only owners can modify rallies
 * - Members can create new highlights (they become the owner)
 * - Users can only edit/delete highlights they created
 * - FREE tier users cannot sync to server (localStorage only)
 */
export async function syncState(
  sessionId: string,
  userId: string | undefined,
  input: SyncStateInput
): Promise<{ success: boolean; syncedAt: string }> {
  // Check tier restrictions
  const tier = await getUserTier(userId);
  const limits = getTierLimits(tier);

  if (!limits.serverSyncEnabled) {
    throw new ForbiddenError(
      "Cross-device sync requires a paid tier (Pro or Elite). Your changes are saved locally."
    );
  }

  // Check access
  let userRole: "owner" | "member" | null = null;
  if (userId) {
    const access = await canAccessSession(sessionId, userId);
    if (!access.hasAccess) {
      throw new NotFoundError("Session", sessionId);
    }
    userRole = access.role;
  }

  const session = await prisma.session.findFirst({
    where: { id: sessionId },
    include: {
      sessionVideos: {
        select: { videoId: true },
      },
    },
  });

  if (session === null) {
    throw new NotFoundError("Session", sessionId);
  }

  // Members cannot modify rallies
  const isOwner = userRole === "owner" || !userId;
  const hasRallyChanges = Object.keys(input.ralliesPerVideo).length > 0;
  if (!isOwner && hasRallyChanges) {
    throw new ForbiddenError("Members cannot modify rallies");
  }

  const videoIds = new Set(session.sessionVideos.map((sv) => sv.videoId));

  await prisma.$transaction(async (tx) => {
    // Sync rallies for each video (only if user is owner)
    if (isOwner) {
      // Pre-fetch all rallies for all session videos in one query to avoid N+1
      // Include data needed for comparison to skip unchanged rallies
      const allExistingRallies = await tx.rally.findMany({
        where: { videoId: { in: [...videoIds] } },
        select: { id: true, videoId: true, startMs: true, endMs: true, order: true },
      });

      // Group by videoId for O(1) lookups, storing full rally data for comparison
      const existingRalliesByVideo = new Map<
        string,
        Map<string, { startMs: number; endMs: number; order: number }>
      >();
      for (const rally of allExistingRallies) {
        if (!existingRalliesByVideo.has(rally.videoId)) {
          existingRalliesByVideo.set(rally.videoId, new Map());
        }
        existingRalliesByVideo.get(rally.videoId)!.set(rally.id, {
          startMs: rally.startMs,
          endMs: rally.endMs,
          order: rally.order,
        });
      }

      // Pre-fetch all camera edits for this user to avoid N+1 queries
      const allExistingCameraEdits = await tx.rallyCameraEdit.findMany({
        where: {
          rally: { videoId: { in: [...videoIds] } },
          userId: userId ?? null,
        },
        include: { keyframes: { orderBy: { timeOffset: "asc" } } },
      });
      const cameraEditsByRallyId = new Map(
        allExistingCameraEdits.map((ce) => [ce.rallyId, ce])
      );

      for (const [videoId, rallies] of Object.entries(input.ralliesPerVideo)) {
        if (!videoIds.has(videoId)) {
          continue; // Skip unknown videos
        }

        // Get existing rallies from pre-fetched data
        const existingRallies = existingRalliesByVideo.get(videoId) ?? new Map();

        // Track which IDs we've seen
        const seenIds = new Set<string>();

        // Update or create rallies
        for (let order = 0; order < rallies.length; order++) {
          const rally = rallies[order];
          if (!rally) continue;

          let rallyId: string;

          const existing = rally.id ? existingRallies.get(rally.id) : undefined;
          if (rally.id && existing) {
            // Only update if something changed
            const hasChanges =
              existing.startMs !== rally.startMs ||
              existing.endMs !== rally.endMs ||
              existing.order !== order;

            if (hasChanges) {
              await tx.rally.update({
                where: { id: rally.id },
                data: {
                  startMs: rally.startMs,
                  endMs: rally.endMs,
                  order,
                },
              });
            }
            rallyId = rally.id;
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
            rallyId = created.id;
            seenIds.add(created.id);
          }

          // Sync camera edit if provided (per-user: each user has their own camera edits)
          if (rally.cameraEdit) {
            const existingCameraEdit = cameraEditsByRallyId.get(rallyId);

            if (existingCameraEdit) {
              // Check if anything actually changed before doing delete+recreate
              const metaChanged =
                existingCameraEdit.enabled !== rally.cameraEdit.enabled ||
                existingCameraEdit.aspectRatio !== rally.cameraEdit.aspectRatio;
              const existingKfs = existingCameraEdit.keyframes;
              const newKfs = rally.cameraEdit.keyframes;
              const keyframesChanged =
                existingKfs.length !== newKfs.length ||
                existingKfs.some((kf, i) => {
                  const nkf = newKfs[i]!;
                  return (
                    kf.timeOffset !== nkf.timeOffset ||
                    kf.positionX !== nkf.positionX ||
                    kf.positionY !== nkf.positionY ||
                    kf.zoom !== nkf.zoom ||
                    kf.rotation !== (nkf.rotation ?? 0) ||
                    kf.easing !== nkf.easing
                  );
                });

              if (metaChanged || keyframesChanged) {
                // Delete old keyframes and create new ones
                await tx.cameraKeyframe.deleteMany({
                  where: { rallyCameraId: existingCameraEdit.id },
                });

                await tx.rallyCameraEdit.update({
                  where: { id: existingCameraEdit.id },
                  data: {
                    enabled: rally.cameraEdit.enabled,
                    aspectRatio: rally.cameraEdit.aspectRatio,
                    keyframes: {
                      createMany: {
                        data: rally.cameraEdit.keyframes.map((kf) => ({
                          timeOffset: kf.timeOffset,
                          positionX: kf.positionX,
                          positionY: kf.positionY,
                          zoom: kf.zoom,
                          rotation: kf.rotation ?? 0,
                          easing: kf.easing,
                        })),
                      },
                    },
                  },
                });
              }
            } else {
              // Create new camera edit with keyframes (per-user)
              await tx.rallyCameraEdit.create({
                data: {
                  rallyId,
                  userId: userId ?? null,
                  enabled: rally.cameraEdit.enabled,
                  aspectRatio: rally.cameraEdit.aspectRatio,
                  keyframes: {
                    createMany: {
                      data: rally.cameraEdit.keyframes.map((kf) => ({
                        timeOffset: kf.timeOffset,
                        positionX: kf.positionX,
                        positionY: kf.positionY,
                        zoom: kf.zoom,
                        rotation: kf.rotation ?? 0,
                        easing: kf.easing,
                      })),
                    },
                  },
                },
              });
            }
          } else if (rally.cameraEdit === null) {
            // Only delete if a camera edit actually exists for this rally
            if (cameraEditsByRallyId.has(rallyId)) {
              await tx.rallyCameraEdit.deleteMany({
                where: { rallyId, userId: userId ?? null },
              });
            }
          }
        }

        // Delete rallies that are no longer in the list
        const toDelete = [...existingRallies.keys()].filter((id) => !seenIds.has(id));
        if (toDelete.length > 0) {
          await tx.rally.deleteMany({
            where: { id: { in: toDelete } },
          });
        }
      }

      // Sync global camera settings for each video (per-user: each user has their own settings)
      if (input.globalCameraSettings) {
        // Pre-fetch all existing camera settings for this user to avoid N+1 queries
        const allExistingCameraSettings = await tx.videoCameraSettings.findMany({
          where: {
            videoId: { in: [...videoIds] },
            userId: userId ?? null,
          },
        });
        const globalSettingsByVideoId = new Map(
          allExistingCameraSettings.map((s) => [s.videoId, s])
        );

        for (const [videoId, settings] of Object.entries(input.globalCameraSettings)) {
          if (!videoIds.has(videoId)) continue;

          const existingSettings = globalSettingsByVideoId.get(videoId);

          if (settings === null) {
            // Only delete if settings actually exist for this user
            if (existingSettings) {
              await tx.videoCameraSettings.deleteMany({
                where: { videoId, userId: userId ?? null },
              });
            }
          } else if (settings) {
            if (existingSettings) {
              // Only update if values actually changed
              const hasChanges =
                existingSettings.zoom !== settings.zoom ||
                existingSettings.positionX !== settings.positionX ||
                existingSettings.positionY !== settings.positionY ||
                existingSettings.rotation !== settings.rotation;

              if (hasChanges) {
                await tx.videoCameraSettings.update({
                  where: { id: existingSettings.id },
                  data: {
                    zoom: settings.zoom,
                    positionX: settings.positionX,
                    positionY: settings.positionY,
                    rotation: settings.rotation,
                  },
                });
              }
            } else {
              // Create new settings for this user
              await tx.videoCameraSettings.create({
                data: {
                  videoId,
                  userId: userId ?? null,
                  zoom: settings.zoom,
                  positionX: settings.positionX,
                  positionY: settings.positionY,
                  rotation: settings.rotation,
                },
              });
            }
          }
        }
      }
    }

    // Sync highlights
    // Get existing highlights with creator info
    const existingHighlights = await tx.highlight.findMany({
      where: { sessionId },
      include: { highlightRallies: true },
    });
    const existingHighlightMap = new Map(
      existingHighlights.map((h) => [h.id, h])
    );
    const seenHighlightIds = new Set<string>();

    // Get all rally IDs for videos in this session for validation
    const allRallies = await tx.rally.findMany({
      where: {
        video: {
          sessionVideos: {
            some: { sessionId },
          },
        },
      },
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
      const existingHighlight = highlight.id
        ? existingHighlightMap.get(highlight.id)
        : undefined;

      if (existingHighlight) {
        // Check if user can edit this highlight
        const canEdit =
          !userId ||
          !existingHighlight.createdByUserId ||
          existingHighlight.createdByUserId === userId;

        if (canEdit) {
          // Update existing highlight
          await tx.highlight.update({
            where: { id: highlight.id },
            data: {
              name: highlight.name,
              color: highlight.color,
            },
          });

          // Sync highlight rallies - use batch insert for performance
          await tx.highlightRally.deleteMany({
            where: { highlightId: highlight.id },
          });

          // Build batch of highlight rallies to create
          const highlightRalliesToCreate = highlight.rallyIds
            .map((frontendRallyId, order) => {
              if (!frontendRallyId) return null;
              const backendRallyId = rallyIdMap.get(frontendRallyId);
              if (!backendRallyId) return null;
              return {
                highlightId: highlight.id!,
                rallyId: backendRallyId,
                order,
              };
            })
            .filter((r): r is NonNullable<typeof r> => r !== null);

          if (highlightRalliesToCreate.length > 0) {
            await tx.highlightRally.createMany({
              data: highlightRalliesToCreate,
            });
          }
        }
        seenHighlightIds.add(highlight.id!);
      } else {
        // Create new highlight with creator tracking
        const created = await tx.highlight.create({
          data: {
            sessionId,
            createdByUserId: userId,
            name: highlight.name,
            color: highlight.color,
          },
        });
        seenHighlightIds.add(created.id);

        // Add highlight rallies - use batch insert for performance
        const newHighlightRallies = highlight.rallyIds
          .map((frontendRallyId, order) => {
            if (!frontendRallyId) return null;
            const backendRallyId = rallyIdMap.get(frontendRallyId);
            if (!backendRallyId) return null;
            return {
              highlightId: created.id,
              rallyId: backendRallyId,
              order,
            };
          })
          .filter((r): r is NonNullable<typeof r> => r !== null);

        if (newHighlightRallies.length > 0) {
          await tx.highlightRally.createMany({
            data: newHighlightRallies,
          });
        }
      }
    }

    // Delete highlights that are no longer in the list
    // Only delete highlights the user owns
    const highlightsToDelete = [...existingHighlightMap.entries()]
      .filter(([id, h]) => {
        if (seenHighlightIds.has(id)) return false;
        // Can only delete if user is owner or created it
        if (!userId) return true;
        if (!h.createdByUserId) return true;
        return h.createdByUserId === userId;
      })
      .map(([id]) => id);

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
