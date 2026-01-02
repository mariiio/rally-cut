import { Router } from "express";
import { Readable } from "stream";
import { z } from "zod";
import { getObject } from "../lib/s3.js";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { paginationSchema, uuidSchema } from "../schemas/common.js";
import {
  confirmUploadSchema,
  requestUploadUrlSchema,
  updateVideoSchema,
} from "../schemas/video.js";
import {
  confirmUpload,
  confirmVideoUpload,
  createVideoUploadUrl,
  listUserVideos,
  requestUploadUrl,
  restoreVideo,
  softDeleteVideo,
  updateVideo,
} from "../services/videoService.js";

const router = Router();

// ============================================================================
// Video Streaming (Local development proxy)
// ============================================================================

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

      // CORS headers for fetch
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
      res.setHeader("Accept-Ranges", "bytes");

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

        // Handle stream errors
        stream.on("error", (err) => {
          console.error("S3 stream error:", err);
          if (!res.headersSent) {
            res.status(500).send("Error streaming video");
          } else {
            res.end();
          }
        });

        // Ensure response ends when stream ends
        stream.on("end", () => {
          res.end();
        });

        stream.pipe(res, { end: false });
      } else {
        res.status(404).send("Video not found");
      }
    } catch (error: unknown) {
      if (error && typeof error === "object" && "name" in error && error.name === "NoSuchKey") {
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
      const video = await import("../lib/prisma.js").then((m) =>
        m.prisma.video.findFirst({
          where: { s3Key, userId: req.userId!, deletedAt: null },
        })
      );

      if (!video) {
        return res.status(404).json({ error: "Video not found" });
      }

      const { generateDownloadUrl } = await import("../lib/s3.js");
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

/**
 * DELETE /v1/videos/:id
 * Soft delete a video
 */
router.delete(
  "/v1/videos/:id",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      await softDeleteVideo(req.params.id, req.userId!);
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

export default router;
