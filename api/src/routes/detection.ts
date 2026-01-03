import { Router } from "express";
import { z } from "zod";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import {
  getDetectionStatus,
  triggerRallyDetection,
} from "../services/detectionService.js";

const router = Router();

router.post(
  "/v1/videos/:id/detect-rallies",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const result = await triggerRallyDetection(req.params.id, userId);
      res.status(202).json(result);
    } catch (error) {
      next(error);
    }
  }
);

router.get(
  "/v1/videos/:id/detection-status",
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const status = await getDetectionStatus(req.params.id);
      res.json(status);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
