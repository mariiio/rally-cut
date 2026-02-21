import crypto from "crypto";
import type { RequestHandler } from "express";
import { Router } from "express";
import { z } from "zod";
import { env } from "../config/env.js";
import { validateRequest } from "../middleware/validateRequest.js";
import {
  handleDetectionComplete,
  updateDetectionProgress,
} from "../services/detectionService.js";
import {
  handleExportComplete,
  updateExportProgress,
} from "../services/exportService.js";
import {
  handleConfirmationComplete,
  updateConfirmationProgress,
} from "../services/confirmationService.js";
import { handleProcessingComplete } from "../services/processingService.js";
import {
  handleTrackingRallyComplete,
  handleTrackingBatchComplete,
} from "../services/modalTrackingService.js";

const router = Router();

// Middleware to validate webhook secret before body parsing
// Uses timing-safe comparison to prevent timing attacks
const requireWebhookSecret: RequestHandler = (req, res, next) => {
  const secret = req.headers["x-webhook-secret"];
  if (typeof secret !== "string") {
    res.status(401).json({ error: "Invalid webhook secret" });
    return;
  }

  const expectedSecret = env.MODAL_WEBHOOK_SECRET;
  const secretBuffer = Buffer.from(secret);
  const expectedBuffer = Buffer.from(expectedSecret);

  // Timing-safe comparison - both buffers must be same length
  // If lengths differ, compare expected with itself to maintain constant time
  const isValid =
    secretBuffer.length === expectedBuffer.length &&
    crypto.timingSafeEqual(secretBuffer, expectedBuffer);

  if (!isValid) {
    res.status(401).json({ error: "Invalid webhook secret" });
    return;
  }
  next();
};

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
  suggested_rallies: z
    .array(
      z.object({
        start_ms: z.number().int().nonnegative(),
        end_ms: z.number().int().positive(),
        confidence: z.number().min(0).max(1),
        rejection_reason: z.enum(["insufficient_windows", "too_short", "sparse_density"]),
      })
    )
    .optional(),
  result_s3_key: z.string().optional(),
});

router.post(
  "/v1/webhooks/detection-complete",
  requireWebhookSecret,
  validateRequest({ body: detectionCompleteSchema }),
  async (req, res, next) => {
    try {
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
  requireWebhookSecret,
  validateRequest({ body: progressUpdateSchema }),
  async (req, res, next) => {
    try {
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
  requireWebhookSecret,
  validateRequest({ body: exportCompleteSchema }),
  async (req, res, next) => {
    try {
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
  requireWebhookSecret,
  validateRequest({ body: exportProgressSchema }),
  async (req, res, next) => {
    try {
      await updateExportProgress(req.body.job_id, req.body.progress);
      res.json({ success: true });
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Confirmation Webhooks (Rally Confirmation / Trimmed Video Generation)
// ============================================================================

const confirmationCompleteSchema = z.object({
  confirmation_id: z.string().uuid(),
  status: z.enum(["completed", "failed"]),
  error_message: z.string().optional(),
  output_s3_key: z.string().optional(),
  duration_ms: z.number().int().positive().optional(),
  proxy_s3_key: z.string().optional(),  // Trimmed proxy video (if available)
});

router.post(
  "/v1/webhooks/confirmation-complete",
  requireWebhookSecret,
  validateRequest({ body: confirmationCompleteSchema }),
  async (req, res, next) => {
    try {
      const result = await handleConfirmationComplete(req.body);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

const confirmationProgressSchema = z.object({
  confirmation_id: z.string().uuid(),
  progress: z.number().min(0).max(100),
});

router.post(
  "/v1/webhooks/confirmation-progress",
  requireWebhookSecret,
  validateRequest({ body: confirmationProgressSchema }),
  async (req, res, next) => {
    try {
      await updateConfirmationProgress(
        req.body.confirmation_id,
        req.body.progress
      );
      res.json({ success: true });
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Video Processing Webhooks (Upload Optimization)
// ============================================================================

const processingCompleteSchema = z.object({
  video_id: z.string().uuid(),
  status: z.enum(["completed", "failed", "skipped"]),
  processed_s3_key: z.string().optional(),
  processed_size_bytes: z.number().int().positive().optional(),
  was_optimized: z.boolean().optional(),
  error_message: z.string().optional(),
});

router.post(
  "/v1/webhooks/processing-complete",
  requireWebhookSecret,
  validateRequest({ body: processingCompleteSchema }),
  async (req, res, next) => {
    try {
      const result = await handleProcessingComplete(req.body);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Batch Tracking Webhooks (Modal GPU Player Tracking)
// ============================================================================

const trackingRallyCompleteSchema = z.object({
  batch_job_id: z.string().uuid(),
  video_id: z.string().uuid(),
  rally_id: z.string().uuid(),
  status: z.enum(["completed", "failed"]),
  tracking_data: z.record(z.unknown()).optional(),
  error: z.string().optional(),
});

router.post(
  "/v1/webhooks/tracking-rally-complete",
  requireWebhookSecret,
  validateRequest({ body: trackingRallyCompleteSchema }),
  async (req, res, next) => {
    try {
      const result = await handleTrackingRallyComplete(req.body);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

const trackingBatchCompleteSchema = z.object({
  batch_job_id: z.string().uuid(),
  video_id: z.string().uuid(),
  status: z.enum(["completed", "failed"]),
  completed_rallies: z.number().int().nonnegative(),
  failed_rallies: z.number().int().nonnegative(),
  error: z.string().optional(),
});

router.post(
  "/v1/webhooks/tracking-batch-complete",
  requireWebhookSecret,
  validateRequest({ body: trackingBatchCompleteSchema }),
  async (req, res, next) => {
    try {
      const result = await handleTrackingBatchComplete(req.body);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
