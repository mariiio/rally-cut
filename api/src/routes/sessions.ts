import { Router } from "express";
import { z } from "zod";
import { generateSignedCookies } from "../lib/cloudfront.js";
import { deleteObject } from "../lib/s3.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { batchRequestSchema } from "../schemas/batch.js";
import { paginate, paginationSchema, uuidSchema } from "../schemas/common.js";
import {
  createSessionSchema,
  updateSessionSchema,
} from "../schemas/session.js";
import { syncStateSchema } from "../schemas/sync.js";
import { processBatch } from "../services/batchService.js";
import {
  createSession,
  deleteSession,
  getSessionById,
  listSessions,
  updateSession,
} from "../services/sessionService.js";
import { syncState } from "../services/syncService.js";

const router = Router();

router.post(
  "/v1/sessions",
  validateRequest({ body: createSessionSchema }),
  async (req, res, next) => {
    try {
      const session = await createSession(req.body);
      res.status(201).json(session);
    } catch (error) {
      next(error);
    }
  }
);

router.get(
  "/v1/sessions",
  validateRequest({ query: paginationSchema }),
  async (req, res, next) => {
    try {
      const pagination = paginationSchema.parse(req.query);
      const { sessions, total } = await listSessions(pagination);
      res.json(paginate(sessions, total, pagination));
    } catch (error) {
      next(error);
    }
  }
);

router.get(
  "/v1/sessions/:id",
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const session = await getSessionById(req.params.id);
      const cookies = generateSignedCookies(session.id);

      if (cookies !== null) {
        for (const [name, value] of Object.entries(cookies)) {
          res.cookie(name, value, {
            httpOnly: true,
            secure: process.env["NODE_ENV"] === "production",
            sameSite: "strict",
            maxAge: 24 * 60 * 60 * 1000,
            path: "/",
          });
        }
      }

      res.json(session);
    } catch (error) {
      next(error);
    }
  }
);

router.patch(
  "/v1/sessions/:id",
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: updateSessionSchema,
  }),
  async (req, res, next) => {
    try {
      const session = await updateSession(req.params.id, req.body);
      res.json(session);
    } catch (error) {
      next(error);
    }
  }
);

router.delete(
  "/v1/sessions/:id",
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const s3Keys = await deleteSession(req.params.id);

      await Promise.all(s3Keys.map((key) => deleteObject(key)));

      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

router.post(
  "/v1/sessions/:id/batch",
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: batchRequestSchema,
  }),
  async (req, res, next) => {
    try {
      const result = await processBatch(req.params.id, req.body.operations);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

router.post(
  "/v1/sessions/:id/sync-state",
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: syncStateSchema,
  }),
  async (req, res, next) => {
    try {
      const result = await syncState(req.params.id, req.body);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
