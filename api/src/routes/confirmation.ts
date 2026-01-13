import { Router } from "express";
import { z } from "zod";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import {
  initiateConfirmation,
  getConfirmationStatus,
  restoreOriginal,
} from "../services/confirmationService.js";

const router = Router();

/**
 * POST /v1/videos/:id/confirm-rallies
 * Initiate rally confirmation for a video.
 * Creates a trimmed video with only rally segments (dead time removed).
 * Paid tiers only (Pro/Elite).
 */
router.post(
  "/v1/videos/:id/confirm-rallies",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const result = await initiateConfirmation(req.params.id, userId);
      res.status(202).json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /v1/videos/:id/confirmation-status
 * Get the current confirmation status for a video.
 */
router.get(
  "/v1/videos/:id/confirmation-status",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const status = await getConfirmationStatus(req.params.id, userId);
      res.json(status);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /v1/videos/:id/restore-original
 * Restore the original video, deleting the trimmed version.
 * Reverts rally timestamps to original values.
 * Paid tiers only (Pro/Elite).
 */
router.post(
  "/v1/videos/:id/restore-original",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const result = await restoreOriginal(req.params.id, userId);
      res.status(200).json(result);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
