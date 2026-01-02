import { Router } from "express";
import { z } from "zod";
import { env } from "../config/env.js";
import { ValidationError } from "../middleware/errorHandler.js";
import { validateRequest } from "../middleware/validateRequest.js";
import {
  handleDetectionComplete,
  updateDetectionProgress,
} from "../services/detectionService.js";
import {
  handleExportComplete,
  updateExportProgress,
} from "../services/exportService.js";

const router = Router();

const detectionCompleteSchema = z.object({
  job_id: z.string().uuid(),
  status: z.enum(["completed", "failed"]),
  error_message: z.string().optional(),
  rallies: z
    .array(
      z.object({
        start_ms: z.number().int().nonnegative(),
        end_ms: z.number().int().positive(),
        confidence: z.number().min(0).max(1).optional(),
      })
    )
    .optional(),
  result_s3_key: z.string().optional(),
});

router.post(
  "/v1/webhooks/detection-complete",
  validateRequest({ body: detectionCompleteSchema }),
  async (req, res, next) => {
    try {
      const secret = req.headers["x-webhook-secret"];

      if (secret !== env.MODAL_WEBHOOK_SECRET) {
        throw new ValidationError("Invalid webhook secret");
      }

      const result = await handleDetectionComplete(req.body);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

// Progress update endpoint - called by local runner during detection
const progressUpdateSchema = z.object({
  job_id: z.string().uuid(),
  progress: z.number().min(0).max(100),
  message: z.string(),
});

router.post(
  "/v1/webhooks/detection-progress",
  validateRequest({ body: progressUpdateSchema }),
  async (req, res, next) => {
    try {
      const secret = req.headers["x-webhook-secret"];

      if (secret !== env.MODAL_WEBHOOK_SECRET) {
        throw new ValidationError("Invalid webhook secret");
      }

      await updateDetectionProgress(
        req.body.job_id,
        req.body.progress,
        req.body.message
      );
      res.json({ success: true });
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Export Webhooks
// ============================================================================

const exportCompleteSchema = z.object({
  job_id: z.string().uuid(),
  status: z.enum(["completed", "failed"]),
  error_message: z.string().optional(),
  output_s3_key: z.string().optional(),
});

router.post(
  "/v1/webhooks/export-complete",
  validateRequest({ body: exportCompleteSchema }),
  async (req, res, next) => {
    try {
      const secret = req.headers["x-webhook-secret"];

      if (secret !== env.MODAL_WEBHOOK_SECRET) {
        throw new ValidationError("Invalid webhook secret");
      }

      const result = await handleExportComplete(req.body);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

const exportProgressSchema = z.object({
  job_id: z.string().uuid(),
  progress: z.number().min(0).max(100),
});

router.post(
  "/v1/webhooks/export-progress",
  validateRequest({ body: exportProgressSchema }),
  async (req, res, next) => {
    try {
      const secret = req.headers["x-webhook-secret"];

      if (secret !== env.MODAL_WEBHOOK_SECRET) {
        throw new ValidationError("Invalid webhook secret");
      }

      await updateExportProgress(req.body.job_id, req.body.progress);
      res.json({ success: true });
    } catch (error) {
      next(error);
    }
  }
);

export default router;
