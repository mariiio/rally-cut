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

const createExportJobSchema = z.object({
  sessionId: uuidSchema,
  tier: z.enum(["FREE", "PREMIUM"]).default("FREE"),
  config: z.object({
    format: z.enum(["mp4", "webm"]).default("mp4"),
  }),
  rallies: z.array(
    z.object({
      videoId: z.string(),
      videoS3Key: z.string(),
      startMs: z.number().int().min(0),
      endMs: z.number().int().min(0),
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
      const job = await createExportJob(req.userId!, req.body);
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
      const job = await getExportJob(req.params.id, req.userId!);
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
      const result = await getExportDownloadUrl(req.params.id, req.userId!);
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
