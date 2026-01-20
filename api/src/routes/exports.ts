import { Router } from "express";
import { z } from "zod";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import {
  createExportJob,
  getExportJob,
  getExportDownloadUrl,
} from "../services/exportService.js";

const router = Router();

// Camera keyframe schema for export
const cameraKeyframeSchema = z.object({
  timeOffset: z.number().min(0).max(1),
  positionX: z.number().min(0).max(1),
  positionY: z.number().min(0).max(1),
  zoom: z.number().min(0.5).max(3),
  rotation: z.number().min(-30).max(30),
  easing: z.enum(["LINEAR", "EASE_IN", "EASE_OUT", "EASE_IN_OUT"]),
});

// Camera edit schema for export (only ORIGINAL aspect ratio supported)
const cameraEditSchema = z.object({
  aspectRatio: z.literal("ORIGINAL"),
  keyframes: z.array(cameraKeyframeSchema),
});

const createExportJobSchema = z.object({
  sessionId: uuidSchema,
  // tier is intentionally NOT accepted from client - determined by backend from user
  config: z.object({
    format: z.enum(["mp4", "webm"]).default("mp4"),
    quality: z.enum(["original", "720p"]).optional(),
    withFade: z.boolean().optional(),
  }),
  rallies: z.array(
    z.object({
      videoId: z.string(),
      videoS3Key: z.string(),
      startMs: z.number().int().min(0),
      endMs: z.number().int().min(0),
      camera: cameraEditSchema.optional(),
    })
  ),
});

/**
 * POST /v1/export-jobs
 * Create a new export job and trigger Lambda processing
 */
router.post(
  "/v1/export-jobs",
  requireUser,
  validateRequest({ body: createExportJobSchema }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const job = await createExportJob(userId, req.body);
      res.status(202).json(job);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /v1/export-jobs/:id
 * Get export job status and progress
 */
router.get(
  "/v1/export-jobs/:id",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const job = await getExportJob(req.params.id, userId);
      if (!job) {
        return res.status(404).json({ error: "Export job not found" });
      }
      return res.json(job);
    } catch (error) {
      return next(error);
    }
  }
);

/**
 * GET /v1/export-jobs/:id/download
 * Get presigned download URL for completed export
 */
router.get(
  "/v1/export-jobs/:id/download",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const result = await getExportDownloadUrl(req.params.id, userId);
      if (!result) {
        return res.status(404).json({ error: "Export job not found" });
      }
      if (!result.downloadUrl) {
        return res.status(400).json({ error: "Export not yet complete" });
      }
      return res.json(result);
    } catch (error) {
      return next(error);
    }
  }
);

export default router;
