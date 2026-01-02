import { Router } from "express";
import { z } from "zod";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import {
  addRallyToHighlightSchema,
  createHighlightSchema,
  updateHighlightSchema,
} from "../schemas/highlight.js";
import {
  addRallyToHighlight,
  createHighlight,
  deleteHighlight,
  removeRallyFromHighlight,
  updateHighlight,
} from "../services/highlightService.js";

const router = Router();

router.post(
  "/v1/sessions/:sessionId/highlights",
  validateRequest({
    params: z.object({ sessionId: uuidSchema }),
    body: createHighlightSchema,
  }),
  async (req, res, next) => {
    try {
      const highlight = await createHighlight(
        req.params.sessionId,
        req.body,
        req.userId
      );
      res.status(201).json(highlight);
    } catch (error) {
      next(error);
    }
  }
);

router.patch(
  "/v1/highlights/:id",
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: updateHighlightSchema,
  }),
  async (req, res, next) => {
    try {
      const highlight = await updateHighlight(req.params.id, req.body, req.userId);
      res.json(highlight);
    } catch (error) {
      next(error);
    }
  }
);

router.delete(
  "/v1/highlights/:id",
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      await deleteHighlight(req.params.id, req.userId);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

router.post(
  "/v1/highlights/:id/rallies",
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: addRallyToHighlightSchema,
  }),
  async (req, res, next) => {
    try {
      const result = await addRallyToHighlight(req.params.id, req.body);
      res.status(201).json(result);
    } catch (error) {
      next(error);
    }
  }
);

router.delete(
  "/v1/highlights/:id/rallies/:rallyId",
  validateRequest({
    params: z.object({ id: uuidSchema, rallyId: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      await removeRallyFromHighlight(req.params.id, req.params.rallyId);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

export default router;
