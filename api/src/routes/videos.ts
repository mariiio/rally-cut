import { Router, Request } from "express";
import { Readable } from "stream";
import { z } from "zod";
import { Prisma } from "@prisma/client";
import { env } from "../config/env.js";
import { prisma } from "../lib/prisma.js";

// Get allowed CORS origin from request (supports Label Studio for dev)
function getAllowedOrigin(req: Request): string {
  const origin = req.headers.origin;
  const allowedOrigins = [env.CORS_ORIGIN];
  if (env.LABEL_STUDIO_URL) {
    allowedOrigins.push(env.LABEL_STUDIO_URL);
  }
  if (origin && allowedOrigins.includes(origin)) {
    return origin;
  }
  return env.CORS_ORIGIN;
}
import { getObject, generateDownloadUrl } from "../lib/s3.js";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { paginationSchema, uuidSchema } from "../schemas/common.js";
import {
  abortMultipartSchema,
  completeMultipartSchema,
  confirmUploadSchema,
  initiateMultipartSchema,
  requestUploadUrlSchema,
  updateVideoSchema,
} from "../schemas/video.js";
import {
  abortMultipartUpload,
  completeMultipartUpload,
  confirmUpload,
  confirmVideoUpload,
  createVideoUploadUrl,
  getVideoForEditor,
  hardDeleteVideo,
  initiateMultipartUpload,
  listUserVideos,
  requestUploadUrl,
  restoreVideo,
  updateVideo,
} from "../services/videoService.js";
import { queueVideoProcessing } from "../services/processingService.js";
import { trackAllRallies, getBatchTrackingStatus } from "../services/batchTrackingService.js";
import { getMatchAnalysis, getMatchStats } from "../services/matchAnalysisService.js";
import { assessVideoQuality, getAnalysisPipelineStatus, savePlayerNames } from "../services/qualityService.js";

// Calibration corners can be outside 0-1 if dragged outside video bounds
const calibrationCornerSchema = z.object({
  x: z.number(),
  y: z.number(),
});

const router = Router();

// ============================================================================
// Video Streaming (Local development proxy)
// ============================================================================

/**
 * GET /confirmations/:confirmationId/:filename
 * Stream confirmed/trimmed video content from S3 (for local development without CloudFront)
 * Supports range requests for video seeking
 */
router.get(
  "/confirmations/:confirmationId/:filename",
  async (req, res, next) => {
    try {
      const { confirmationId, filename } = req.params;
      const s3Key = `confirmations/${confirmationId}/${filename}`;
      const rangeHeader = req.headers.range;

      const s3Response = await getObject(s3Key, rangeHeader);

      // Set appropriate headers
      if (s3Response.ContentType) {
        res.setHeader("Content-Type", s3Response.ContentType);
      }

      // CORS headers for fetch - use configured origin
      res.setHeader("Access-Control-Allow-Origin", getAllowedOrigin(req));
      res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
      res.setHeader("Accept-Ranges", "bytes");

      // Cache headers - trimmed videos can be cached for a long time
      res.setHeader("Cache-Control", "public, max-age=31536000");
      res.setHeader("ETag", `"${confirmationId}-${filename}"`);

      // Handle range request response
      if (rangeHeader && s3Response.ContentRange) {
        res.status(206); // Partial Content
        res.setHeader("Content-Range", s3Response.ContentRange);
        if (s3Response.ContentLength) {
          res.setHeader("Content-Length", s3Response.ContentLength);
        }
      } else {
        if (s3Response.ContentLength) {
          res.setHeader("Content-Length", s3Response.ContentLength);
        }
      }

      // Stream the body to the response
      if (s3Response.Body) {
        const stream = s3Response.Body as Readable;

        const cleanup = (): void => {
          if (!stream.destroyed) {
            stream.destroy();
          }
        };

        res.on("close", cleanup);
        res.on("error", cleanup);

        stream.on("error", (err) => {
          if ((err as NodeJS.ErrnoException).code !== "ECONNRESET") {
            console.error("S3 stream error:", err);
          }
          if (!res.headersSent) {
            res.status(500).send("Error streaming video");
          }
        });

        stream.pipe(res);
      } else {
        res.status(404).send("Video not found");
      }
    } catch (error: unknown) {
      if (error !== null && error !== undefined && typeof error === "object" && "name" in error && error.name === "NoSuchKey") {
        res.status(404).send("Video not found");
      } else {
        next(error);
      }
    }
  }
);

/**
 * GET /videos/:userId/:videoId/:filename
 * Stream video content from S3 (for local development without CloudFront)
 * Supports range requests for video seeking
 */
router.get(
  "/videos/:userId/:videoId/:filename",
  async (req, res, next) => {
    try {
      const { userId, videoId, filename } = req.params;
      const s3Key = `videos/${userId}/${videoId}/${filename}`;
      const rangeHeader = req.headers.range;

      const s3Response = await getObject(s3Key, rangeHeader);

      // Set appropriate headers
      if (s3Response.ContentType) {
        res.setHeader("Content-Type", s3Response.ContentType);
      }

      // CORS headers for fetch - use configured origin
      res.setHeader("Access-Control-Allow-Origin", getAllowedOrigin(req));
      res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
      res.setHeader("Accept-Ranges", "bytes");

      // Cache headers - allow browser to cache video
      // Use longer cache for optimized videos (filename contains _optimized)
      const isOptimized = filename.includes("_optimized");
      const maxAge = isOptimized ? 31536000 : 86400; // 1 year vs 1 day
      res.setHeader("Cache-Control", `public, max-age=${maxAge}`);
      res.setHeader("ETag", `"${videoId}-${filename}"`);

      // Handle range request response
      if (rangeHeader && s3Response.ContentRange) {
        res.status(206); // Partial Content
        res.setHeader("Content-Range", s3Response.ContentRange);
        if (s3Response.ContentLength) {
          res.setHeader("Content-Length", s3Response.ContentLength);
        }
      } else {
        if (s3Response.ContentLength) {
          res.setHeader("Content-Length", s3Response.ContentLength);
        }
      }

      // Stream the body to the response
      if (s3Response.Body) {
        const stream = s3Response.Body as Readable;

        // Handle client disconnect (browser abort, navigation, etc.)
        // This is critical for video seeking which triggers many aborted requests
        const cleanup = (): void => {
          if (!stream.destroyed) {
            stream.destroy();
          }
        };

        res.on("close", cleanup);
        res.on("error", cleanup);

        // Handle stream errors
        stream.on("error", (err) => {
          // ECONNRESET is expected when client aborts - don't log as error
          if ((err as NodeJS.ErrnoException).code !== "ECONNRESET") {
            console.error("S3 stream error:", err);
          }
          if (!res.headersSent) {
            res.status(500).send("Error streaming video");
          }
        });

        // Pipe stream to response (end: true is default and correct here)
        stream.pipe(res);
      } else {
        res.status(404).send("Video not found");
      }
    } catch (error: unknown) {
      if (error !== null && error !== undefined && typeof error === "object" && "name" in error && error.name === "NoSuchKey") {
        res.status(404).send("Video not found");
      } else {
        next(error);
      }
    }
  }
);

/**
 * GET /v1/videos/download-url
 * Get a presigned S3 download URL for a video (used by export to bypass CORS)
 */
router.get(
  "/v1/videos/download-url",
  requireUser,
  validateRequest({
    query: z.object({ s3Key: z.string().min(1) }),
  }),
  async (req, res, next) => {
    try {
      const { s3Key } = req.query as { s3Key: string };

      // Verify the user owns a video with this s3Key
      const video = await prisma.video.findFirst({
        where: { s3Key, userId: req.userId!, deletedAt: null },
      });

      if (!video) {
        return res.status(404).json({ error: "Video not found" });
      }

      const downloadUrl = await generateDownloadUrl(s3Key);

      return res.json({ downloadUrl });
    } catch (error) {
      return next(error);
    }
  }
);

// ============================================================================
// Video Library Endpoints (User-scoped)
// ============================================================================

/**
 * GET /v1/videos
 * List all videos for the current user
 */
router.get(
  "/v1/videos",
  requireUser,
  validateRequest({
    query: paginationSchema.extend({
      search: z.string().optional(),
    }),
  }),
  async (req, res, next) => {
    try {
      const { page, limit, search } = req.query as {
        page?: number;
        limit?: number;
        search?: string;
      };
      const result = await listUserVideos(req.userId!, { search, page, limit });
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /v1/videos/:id/editor
 * Get video data for single-video editor mode
 * Returns video with rallies, camera edits, and filtered highlights
 */
router.get(
  "/v1/videos/:id/editor",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const result = await getVideoForEditor(req.params.id, req.userId!);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /v1/videos/upload-url
 * Get presigned URL to upload a new video (not linked to a session)
 */
router.post(
  "/v1/videos/upload-url",
  requireUser,
  validateRequest({ body: requestUploadUrlSchema }),
  async (req, res, next) => {
    try {
      const result = await createVideoUploadUrl({
        userId: req.userId!,
        filename: req.body.filename,
        contentType: req.body.contentType,
        contentHash: req.body.contentHash,
        fileSize: req.body.fileSize,
        durationMs: req.body.durationMs,
      });
      res.status(201).json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /v1/videos/:id/confirm
 * Confirm video upload complete
 */
router.post(
  "/v1/videos/:id/confirm",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: z.object({
      durationMs: z.number().int().positive().optional(),
      width: z.number().int().positive().optional(),
      height: z.number().int().positive().optional(),
    }),
  }),
  async (req, res, next) => {
    try {
      const video = await confirmVideoUpload(req.params.id, req.userId!, req.body);
      res.json(video);
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Multipart Upload Endpoints (for large files)
// ============================================================================

/**
 * POST /v1/videos/multipart/init
 * Initiate a multipart upload and get presigned URLs for all parts
 */
router.post(
  "/v1/videos/multipart/init",
  requireUser,
  validateRequest({ body: initiateMultipartSchema }),
  async (req, res, next) => {
    try {
      const result = await initiateMultipartUpload({
        userId: req.userId!,
        filename: req.body.filename,
        contentType: req.body.contentType,
        contentHash: req.body.contentHash,
        fileSize: req.body.fileSize,
        durationMs: req.body.durationMs,
      });
      res.status(201).json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /v1/videos/:id/multipart/complete
 * Complete a multipart upload by combining all parts
 */
router.post(
  "/v1/videos/:id/multipart/complete",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: completeMultipartSchema,
  }),
  async (req, res, next) => {
    try {
      await completeMultipartUpload(
        req.params.id,
        req.userId!,
        req.body.uploadId,
        req.body.parts
      );
      res.json({ success: true });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /v1/videos/:id/multipart/abort
 * Abort a multipart upload and cleanup parts
 */
router.post(
  "/v1/videos/:id/multipart/abort",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: abortMultipartSchema,
  }),
  async (req, res, next) => {
    try {
      await abortMultipartUpload(req.params.id, req.userId!, req.body.uploadId);
      res.json({ success: true });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * DELETE /v1/videos/:id
 * Permanently delete a video and all associated data
 */
router.delete(
  "/v1/videos/:id",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      await hardDeleteVideo(req.params.id, req.userId!);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /v1/videos/:id/restore
 * Restore a soft-deleted video
 */
router.post(
  "/v1/videos/:id/restore",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      await restoreVideo(req.params.id, req.userId!);
      res.json({ success: true });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * PATCH /v1/videos/:id
 * Update video (name, etc.)
 */
router.patch(
  "/v1/videos/:id",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: updateVideoSchema,
  }),
  async (req, res, next) => {
    try {
      const video = await updateVideo(req.params.id, req.userId!, req.body);
      res.json(video);
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Court Calibration Endpoints
// ============================================================================

/**
 * PUT /v1/videos/:id/court-calibration
 * Save court calibration corners for a video
 */
router.put(
  "/v1/videos/:id/court-calibration",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: z.object({
      corners: z.array(calibrationCornerSchema).length(4),
    }),
  }),
  async (req, res, next) => {
    try {
      const video = await prisma.video.findFirst({
        where: { id: req.params.id, userId: req.userId!, deletedAt: null },
      });

      if (!video) {
        return res.status(404).json({ error: "Video not found" });
      }

      await prisma.video.update({
        where: { id: req.params.id },
        data: { courtCalibrationJson: req.body.corners },
      });

      return res.json({ success: true });
    } catch (error) {
      return next(error);
    }
  }
);

/**
 * DELETE /v1/videos/:id/court-calibration
 * Clear court calibration for a video
 */
router.delete(
  "/v1/videos/:id/court-calibration",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const video = await prisma.video.findFirst({
        where: { id: req.params.id, userId: req.userId!, deletedAt: null },
      });

      if (!video) {
        return res.status(404).json({ error: "Video not found" });
      }

      await prisma.video.update({
        where: { id: req.params.id },
        data: { courtCalibrationJson: Prisma.DbNull },
      });

      return res.json({ success: true });
    } catch (error) {
      return next(error);
    }
  }
);

// ============================================================================
// Session-scoped Video Endpoints (Legacy/Compatibility)
// ============================================================================

/**
 * POST /v1/sessions/:sessionId/videos/upload-url
 * Get presigned URL to upload a video directly to a session
 */
router.post(
  "/v1/sessions/:sessionId/videos/upload-url",
  requireUser,
  validateRequest({
    params: z.object({ sessionId: uuidSchema }),
    body: requestUploadUrlSchema,
  }),
  async (req, res, next) => {
    try {
      const result = await requestUploadUrl(
        req.params.sessionId,
        req.userId!,
        req.body
      );
      res.status(201).json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /v1/sessions/:sessionId/videos
 * Confirm video upload for a session
 */
router.post(
  "/v1/sessions/:sessionId/videos",
  requireUser,
  validateRequest({
    params: z.object({ sessionId: uuidSchema }),
    body: confirmUploadSchema,
  }),
  async (req, res, next) => {
    try {
      const video = await confirmUpload(
        req.params.sessionId,
        req.userId!,
        req.body
      );
      res.json(video);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * POST /v1/videos/:id/reprocess
 * Re-trigger video processing (for development/retry failed processing)
 */
router.post(
  "/v1/videos/:id/reprocess",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      await queueVideoProcessing(req.params.id, req.userId!);
      res.json({ success: true, message: "Processing queued" });
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Batch Tracking Routes
// ============================================================================

/**
 * POST /v1/videos/:id/track-all-rallies
 * Track all rallies in a video. Returns 202 with job ID.
 * Downloads video once, processes rallies sequentially in background.
 */
router.post(
  "/v1/videos/:id/track-all-rallies",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const result = await trackAllRallies(req.params.id, req.userId!);
      res.status(202).json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /v1/videos/:id/batch-tracking-status
 * Poll for batch tracking progress.
 */
router.get(
  "/v1/videos/:id/batch-tracking-status",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const status = await getBatchTrackingStatus(req.params.id, req.userId!);
      res.json(status);
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Match Analysis Routes
// ============================================================================

/**
 * GET /v1/videos/:id/match-analysis
 * Get cross-rally player identity and match statistics.
 */
router.get(
  "/v1/videos/:id/match-analysis",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const analysis = await getMatchAnalysis(req.params.id, req.userId!);
      res.json(analysis ?? { status: 'not_available' });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /v1/videos/:id/match-stats
 * Get match statistics (kills, aces, efficiency, score progression).
 */
router.get(
  "/v1/videos/:id/match-stats",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const stats = await getMatchStats(req.params.id, req.userId!);
      res.json(stats ?? { status: 'not_available' });
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// Analysis Pipeline Routes
// ============================================================================

/**
 * POST /v1/videos/:id/assess-quality
 * Run quality assessment + court auto-detection.
 * Returns warnings + court detection result.
 */
router.post(
  "/v1/videos/:id/assess-quality",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const result = await assessVideoQuality(req.params.id, req.userId!);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /v1/videos/:id/analysis-pipeline-status
 * Unified pipeline status by reading existing DB state.
 */
router.get(
  "/v1/videos/:id/analysis-pipeline-status",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const status = await getAnalysisPipelineStatus(req.params.id, req.userId!);
      res.json(status);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * PUT /v1/videos/:id/player-names
 * Save player name assignments.
 */
router.put(
  "/v1/videos/:id/player-names",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: z.object({
      names: z.record(z.string(), z.string().max(100)),
    }),
  }),
  async (req, res, next) => {
    try {
      const result = await savePlayerNames(req.params.id, req.userId!, req.body.names);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
