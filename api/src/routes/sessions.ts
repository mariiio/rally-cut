import { Router } from "express";
import { z } from "zod";
import { generateSignedCookies } from "../lib/cloudfront.js";
import { requireUser } from "../middleware/resolveUser.js";
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
import { listSharedSessions } from "../services/shareService.js";
import { syncState } from "../services/syncService.js";
import {
  addVideoToSession,
  removeVideoFromSession,
  reorderSessionVideos,
} from "../services/videoService.js";

const router = Router();

router.post(
  "/v1/sessions",
  requireUser,
  validateRequest({ body: createSessionSchema }),
  async (req, res, next) => {
    try {
      const session = await createSession(req.body, req.userId);
      res.status(201).json(session);
    } catch (error) {
      next(error);
    }
  }
);

router.get(
  "/v1/sessions",
  requireUser,
  validateRequest({ query: paginationSchema }),
  async (req, res, next) => {
    try {
      const pagination = paginationSchema.parse(req.query);
      const { sessions, total } = await listSessions(pagination, req.userId);
      res.json(paginate(sessions, total, pagination));
    } catch (error) {
      next(error);
    }
  }
);

// Must be before /v1/sessions/:id to avoid route conflict
router.get("/v1/sessions/shared", requireUser, async (req, res, next) => {
  try {
    const userId = req.userId as string; // Guaranteed by requireUser
    const sessions = await listSharedSessions(userId);
    res.json({ data: sessions });
  } catch (error) {
    next(error);
  }
});

// Check if user has any content (for returning user banner)
router.get("/v1/sessions/has-content", async (req, res, next) => {
  try {
    if (!req.userId) {
      res.json({ hasContent: false });
      return;
    }
    const { total } = await listSessions({ page: 1, limit: 1 }, req.userId);
    res.json({ hasContent: total > 0, sessionCount: total });
  } catch (error) {
    next(error);
  }
});

router.get(
  "/v1/sessions/:id",
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const session = await getSessionById(req.params.id, req.userId);
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
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: updateSessionSchema,
  }),
  async (req, res, next) => {
    try {
      const session = await updateSession(req.params.id, req.body, req.userId);
      res.json(session);
    } catch (error) {
      next(error);
    }
  }
);

router.delete(
  "/v1/sessions/:id",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      // Note: Videos are NOT deleted when session is deleted
      // They remain in the user's library
      await deleteSession(req.params.id, req.userId);
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
      const result = await processBatch(
        req.params.id,
        req.userId,
        req.body.operations
      );
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
      const result = await syncState(req.params.id, req.userId, req.body);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

// Session-Video junction management
router.post(
  "/v1/sessions/:sessionId/videos/:videoId",
  validateRequest({
    params: z.object({ sessionId: uuidSchema, videoId: uuidSchema }),
    body: z.object({ order: z.number().int().nonnegative().optional() }).optional(),
  }),
  async (req, res, next) => {
    try {
      if (!req.userId) {
        res.status(400).json({ error: "X-Visitor-Id header required" });
        return;
      }
      const sessionVideo = await addVideoToSession(
        req.params.sessionId,
        req.params.videoId,
        req.userId,
        req.body?.order
      );
      res.status(201).json(sessionVideo);
    } catch (error) {
      next(error);
    }
  }
);

router.delete(
  "/v1/sessions/:sessionId/videos/:videoId",
  validateRequest({
    params: z.object({ sessionId: uuidSchema, videoId: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      if (!req.userId) {
        res.status(400).json({ error: "X-Visitor-Id header required" });
        return;
      }
      await removeVideoFromSession(
        req.params.sessionId,
        req.params.videoId,
        req.userId
      );
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

router.put(
  "/v1/sessions/:sessionId/videos/reorder",
  validateRequest({
    params: z.object({ sessionId: uuidSchema }),
    body: z.object({
      videos: z.array(
        z.object({
          videoId: uuidSchema,
          order: z.number().int().nonnegative(),
        })
      ),
    }),
  }),
  async (req, res, next) => {
    try {
      if (!req.userId) {
        res.status(400).json({ error: "X-Visitor-Id header required" });
        return;
      }
      await reorderSessionVideos(
        req.params.sessionId,
        req.userId,
        req.body.videos
      );
      res.json({ success: true });
    } catch (error) {
      next(error);
    }
  }
);

export default router;
