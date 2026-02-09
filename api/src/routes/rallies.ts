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
import {
  trackBallForRally,
  getBallTrackStatus,
} from "../services/ballTrackingService.js";
import { trackPlayersForRally, getPlayerTrack } from "../services/playerTrackingService.js";

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
// Ball Tracking Routes
// ============================================================================

const trackBallSchema = z.object({
  aspectRatio: z.enum(["ORIGINAL", "VERTICAL"]).optional(),
  generateKeyframes: z.boolean().optional(),
});

router.post(
  "/v1/rallies/:id/track-ball",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: trackBallSchema,
  }),
  async (req, res, next) => {
    try {
      const result = await trackBallForRally(req.params.id, req.userId!, {
        aspectRatio: req.body.aspectRatio,
        generateKeyframes: req.body.generateKeyframes,
      });
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

router.get(
  "/v1/rallies/:id/ball-track",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    query: z.object({ includePositions: z.enum(["true", "false"]).optional() }),
  }),
  async (req, res, next) => {
    try {
      const includePositions = req.query.includePositions === "true";
      const result = await getBallTrackStatus(req.params.id, req.userId!, { includePositions });
      res.json(result);
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

export default router;
