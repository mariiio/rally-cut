import { Router } from "express";
import { z } from "zod";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import { createRallySchema, updateRallySchema } from "../schemas/rally.js";
import {
  createRally,
  deleteRally,
  listRallies,
  updateRally,
} from "../services/rallyService.js";

const router = Router();

router.get(
  "/v1/videos/:videoId/rallies",
  validateRequest({ params: z.object({ videoId: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const rallies = await listRallies(req.params.videoId);
      res.json(rallies);
    } catch (error) {
      next(error);
    }
  }
);

router.post(
  "/v1/videos/:videoId/rallies",
  validateRequest({
    params: z.object({ videoId: uuidSchema }),
    body: createRallySchema,
  }),
  async (req, res, next) => {
    try {
      const rally = await createRally(req.params.videoId, req.body);
      res.status(201).json(rally);
    } catch (error) {
      next(error);
    }
  }
);

router.patch(
  "/v1/rallies/:id",
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: updateRallySchema,
  }),
  async (req, res, next) => {
    try {
      const rally = await updateRally(req.params.id, req.body);
      res.json(rally);
    } catch (error) {
      next(error);
    }
  }
);

router.delete(
  "/v1/rallies/:id",
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      await deleteRally(req.params.id);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

export default router;
