import { Router } from "express";
import { z } from "zod";
import { deleteObject } from "../lib/s3.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import {
  confirmUploadSchema,
  requestUploadUrlSchema,
  updateVideoSchema,
} from "../schemas/video.js";
import {
  confirmUpload,
  deleteVideo,
  requestUploadUrl,
  updateVideo,
} from "../services/videoService.js";

const router = Router();

router.post(
  "/v1/sessions/:sessionId/videos/upload-url",
  validateRequest({
    params: z.object({ sessionId: uuidSchema }),
    body: requestUploadUrlSchema,
  }),
  async (req, res, next) => {
    try {
      const result = await requestUploadUrl(req.params.sessionId, req.body);
      res.status(201).json(result);
    } catch (error) {
      next(error);
    }
  }
);

router.post(
  "/v1/sessions/:sessionId/videos",
  validateRequest({
    params: z.object({ sessionId: uuidSchema }),
    body: confirmUploadSchema,
  }),
  async (req, res, next) => {
    try {
      const video = await confirmUpload(req.params.sessionId, req.body);
      res.json(video);
    } catch (error) {
      next(error);
    }
  }
);

router.patch(
  "/v1/videos/:id",
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: updateVideoSchema,
  }),
  async (req, res, next) => {
    try {
      const video = await updateVideo(req.params.id, req.body);
      res.json(video);
    } catch (error) {
      next(error);
    }
  }
);

router.delete(
  "/v1/videos/:id",
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const s3Key = await deleteVideo(req.params.id);
      await deleteObject(s3Key);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

export default router;
