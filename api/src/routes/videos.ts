import { Router } from "express";
import { z } from "zod";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { paginate, paginationSchema, uuidSchema } from "../schemas/common.js";
import {
  confirmUploadSchema,
  requestUploadUrlSchema,
  updateVideoSchema,
} from "../schemas/video.js";
import {
  confirmUpload,
  confirmVideoUpload,
  createVideoUploadUrl,
  deleteVideo,
  listUserVideos,
  requestUploadUrl,
  restoreVideo,
  softDeleteVideo,
  updateVideo,
} from "../services/videoService.js";

const router = Router();

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
