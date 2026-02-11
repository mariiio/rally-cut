import { Router } from "express";
import { z } from "zod";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import { createRallySchema, updateRallySchema } from "../schemas/rally.js";
import {
  createRally,
  deleteRally,
  listRallies,
  updateRally,
} from "../services/rallyService.js";
import { trackPlayersForRally, getPlayerTrack } from "../services/playerTrackingService.js";
import {
  exportToLabelStudio,
  importFromLabelStudio,
  getLabelStudioStatus,
} from "../services/labelStudioService.js";

const router = Router();

router.get(
  "/v1/videos/:videoId/rallies",
  requireUser,
  validateRequest({ params: z.object({ videoId: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const rallies = await listRallies(req.params.videoId, req.userId!);
      res.json(rallies);
    } catch (error) {
      next(error);
    }
  }
);

router.post(
  "/v1/videos/:videoId/rallies",
  requireUser,
  validateRequest({
    params: z.object({ videoId: uuidSchema }),
    body: createRallySchema,
  }),
  async (req, res, next) => {
    try {
      const rally = await createRally(req.params.videoId, req.userId!, req.body);
      res.status(201).json(rally);
    } catch (error) {
      next(error);
    }
  }
);

router.patch(
  "/v1/rallies/:id",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: updateRallySchema,
  }),
  async (req, res, next) => {
    try {
      const rally = await updateRally(req.params.id, req.userId!, req.body);
      res.json(rally);
    } catch (error) {
      next(error);
    }
  }
);

router.delete(
  "/v1/rallies/:id",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      await deleteRally(req.params.id, req.userId!);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Player Tracking Routes
// ============================================================================

// Calibration corner schema (corners can be outside 0-1 if dragged outside video)
const calibrationCornerSchema = z.object({
  x: z.number(),
  y: z.number(),
});

router.post(
  "/v1/rallies/:id/track-players",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: z.object({
      calibrationCorners: z.array(calibrationCornerSchema).length(4).optional(),
    }).optional(),
  }),
  async (req, res, next) => {
    try {
      const calibrationCorners = req.body?.calibrationCorners;
      const result = await trackPlayersForRally(req.params.id, req.userId!, calibrationCorners);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

router.get(
  "/v1/rallies/:id/player-track",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const result = await getPlayerTrack(req.params.id, req.userId!);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

// Label Studio integration
router.get(
  "/v1/rallies/:id/label-studio",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const status = await getLabelStudioStatus(req.params.id, req.userId!);
      res.json(status);
    } catch (error) {
      next(error);
    }
  }
);

router.post(
  "/v1/rallies/:id/label-studio/export",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: z.object({
      videoUrl: z.string().url(),
      apiKey: z.string().optional(),
      apiUrl: z.string().url().optional(),
      forceRegenerate: z.boolean().optional(),
    }),
  }),
  async (req, res, next) => {
    try {
      const result = await exportToLabelStudio(
        req.params.id,
        req.userId!,
        req.body.videoUrl,
        {
          config: {
            apiKey: req.body.apiKey,
            url: req.body.apiUrl,
          },
          forceRegenerate: req.body.forceRegenerate,
        }
      );
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

router.post(
  "/v1/rallies/:id/label-studio/import",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: z.object({
      taskId: z.number().int().positive(),
      apiKey: z.string().optional(),
      apiUrl: z.string().url().optional(),
    }),
  }),
  async (req, res, next) => {
    try {
      const result = await importFromLabelStudio(
        req.params.id,
        req.userId!,
        req.body.taskId,
        {
          apiKey: req.body.apiKey,
          url: req.body.apiUrl,
        }
      );
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
