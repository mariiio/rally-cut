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
import { getObject, generateDownloadUrl, uploadPlayerCrop, deleteObject } from "../lib/s3.js";
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
import { getMatchAnalysis, getMatchStats, runMatchAnalysis, type ProgressCallback } from "../services/matchAnalysisService.js";
import { runPreflightChecks, getAnalysisPipelineStatus, savePlayerMatchingGt, getPlayerMatchingGt } from "../services/qualityService.js";

const MAX_REFERENCE_CROPS_PER_PLAYER = 6;

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
        data: {
          courtCalibrationJson: req.body.corners,
          courtCalibrationSource: 'manual',
        },
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
        data: {
          courtCalibrationJson: Prisma.DbNull,
          courtCalibrationSource: null,
        },
      });

      return res.json({ success: true });
    } catch (error) {
      return next(error);
    }
  }
);

// ============================================================================
// Player Reference Crops Endpoints
// ============================================================================

/**
 * GET /v1/videos/:id/player-reference-crops
 * List all reference crops for a video (with presigned download URLs)
 */
router.get(
  "/v1/videos/:id/player-reference-crops",
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

      const crops = await prisma.playerReferenceCrop.findMany({
        where: { videoId: req.params.id },
        orderBy: [{ playerId: "asc" }, { createdAt: "asc" }],
      });

      const cropsWithUrls = await Promise.all(
        crops.map(async (crop) => ({
          id: crop.id,
          playerId: crop.playerId,
          frameMs: crop.frameMs,
          bbox: { x: crop.bboxX, y: crop.bboxY, w: crop.bboxW, h: crop.bboxH },
          downloadUrl: await generateDownloadUrl(crop.s3Key),
          createdAt: crop.createdAt,
        }))
      );

      return res.json({ crops: cropsWithUrls });
    } catch (error) {
      return next(error);
    }
  }
);

/**
 * POST /v1/videos/:id/player-reference-crops
 * Upload a player reference crop (base64 JPEG in body)
 */
router.post(
  "/v1/videos/:id/player-reference-crops",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: z.object({
      playerId: z.number().int().min(1).max(4),
      frameMs: z.number().int().nonnegative(),
      bbox: z.object({
        x: z.number(),
        y: z.number(),
        w: z.number(),
        h: z.number(),
      }),
      imageData: z.string().max(200000), // base64 JPEG, ~150KB max
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

      // Decode base64 and upload to S3
      const imageBuffer = Buffer.from(req.body.imageData, "base64");
      if (imageBuffer.length < 100) {
        return res.status(400).json({ error: "Image data too small" });
      }
      const cropId = crypto.randomUUID();
      const s3Key = `player-crops/${req.params.id}/${req.body.playerId}/${cropId}.jpg`;

      // Enforce max crops per player atomically
      const crop = await prisma.$transaction(async (tx) => {
        const existingCount = await tx.playerReferenceCrop.count({
          where: { videoId: req.params.id, playerId: req.body.playerId },
        });
        if (existingCount >= MAX_REFERENCE_CROPS_PER_PLAYER) {
          throw new Error(`Player ${req.body.playerId} already has ${existingCount} reference crops (max ${MAX_REFERENCE_CROPS_PER_PLAYER})`);
        }
        await uploadPlayerCrop(s3Key, imageBuffer);
        return tx.playerReferenceCrop.create({
          data: {
            videoId: req.params.id,
            playerId: req.body.playerId,
            s3Key,
            frameMs: req.body.frameMs,
            bboxX: req.body.bbox.x,
            bboxY: req.body.bbox.y,
            bboxW: req.body.bbox.w,
            bboxH: req.body.bbox.h,
          },
        });
      });

      const downloadUrl = await generateDownloadUrl(s3Key);

      return res.status(201).json({
        id: crop.id,
        playerId: crop.playerId,
        frameMs: crop.frameMs,
        bbox: { x: crop.bboxX, y: crop.bboxY, w: crop.bboxW, h: crop.bboxH },
        downloadUrl,
        createdAt: crop.createdAt,
      });
    } catch (error) {
      return next(error);
    }
  }
);

/**
 * DELETE /v1/videos/:id/player-reference-crops/:cropId
 * Delete a player reference crop
 */
router.delete(
  "/v1/videos/:id/player-reference-crops/:cropId",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema, cropId: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const crop = await prisma.playerReferenceCrop.findFirst({
        where: { id: req.params.cropId, videoId: req.params.id },
        include: { video: { select: { userId: true } } },
      });

      if (!crop || crop.video.userId !== req.userId) {
        return res.status(404).json({ error: "Crop not found" });
      }

      // Delete DB first (recoverable), then S3 (best-effort cleanup)
      await prisma.playerReferenceCrop.delete({ where: { id: crop.id } });
      await deleteObject(crop.s3Key).catch(() => {});

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
 * POST /v1/videos/:id/run-match-analysis
 * Re-run the post-tracking match analysis pipeline without re-tracking.
 * Runs: match-players → repair-identities → remap-track-ids → reattribute-actions → compute-match-stats
 */
router.post(
  "/v1/videos/:id/run-match-analysis",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const video = await prisma.video.findUnique({ where: { id: req.params.id } });
      if (!video) { res.status(404).json({ error: 'Video not found' }); return; }
      if (video.userId !== req.userId) { res.status(403).json({ error: 'Forbidden' }); return; }

      // Stream progress via SSE
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders();

      const onProgress: ProgressCallback = (step, index, total) => {
        res.write(`data: ${JSON.stringify({ step, index, total })}\n\n`);
      };

      const result = await runMatchAnalysis(req.params.id, onProgress);
      res.write(`data: ${JSON.stringify({ step: 'done', index: 6, total: 6, result })}\n\n`);
      res.end();
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
      const result = await runPreflightChecks(req.params.id, req.userId!);
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
 * GET /v1/videos/:id/player-matching-gt
 * Get player matching ground truth labels.
 */
router.get(
  "/v1/videos/:id/player-matching-gt",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const gt = await getPlayerMatchingGt(req.params.id, req.userId!);
      if (!gt) {
        res.json({ status: 'not_available' });
        return;
      }
      res.json(gt);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * PUT /v1/videos/:id/player-matching-gt
 * Save player matching ground truth labels.
 */
router.put(
  "/v1/videos/:id/player-matching-gt",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    // Player matching GT is bbox-keyed: each label anchors a player to a
    // (frame, bbox) in the rally's positions_json coordinate system, so the
    // same label resolves to the current track id via IoU matching no
    // matter how many times tracking has been re-run. See
    // analysis/rallycut/evaluation/gt_loader.py for the format + resolver.
    body: z.object({
      rallies: z.record(
        z.string(),
        z.object({
          // Bbox coords are center-normalized in [0,1] (same convention
          // as PlayerPosition). Clients must send values inside the
          // frame — negative or >1 values would poison the IoU resolver.
          // Partial rallies (< 4 labels) are intentionally allowed to
          // support intermediate save states from the web dialog; the
          // evaluator treats missing players as unlabeled.
          labels: z
            .array(
              z.object({
                playerId: z.number().int().min(1).max(4),
                frame: z.number().int().min(0),
                cx: z.number().gte(0).lte(1),
                cy: z.number().gte(0).lte(1),
                w: z.number().gt(0).lte(1),
                h: z.number().gt(0).lte(1),
              })
            )
            .min(1)
            .max(4),
        })
      ),
      sideSwitches: z.array(z.number()),
      excludedRallies: z.array(z.string()).optional(),
    }),
  }),
  async (req, res, next) => {
    try {
      const result = await savePlayerMatchingGt(req.params.id, req.userId!, req.body);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
