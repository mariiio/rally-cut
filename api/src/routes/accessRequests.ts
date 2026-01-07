import { Router } from "express";
import { z } from "zod";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import {
  createAccessRequest,
  getPendingRequests,
  acceptRequest,
  rejectRequest,
  getPendingCount,
} from "../services/accessRequestService.js";

const router = Router();

// Request body schema
const createAccessRequestSchema = z.object({
  message: z.string().max(500).optional(),
});

/**
 * Create an access request for a session
 * POST /v1/sessions/:id/access-requests
 */
router.post(
  "/v1/sessions/:id/access-requests",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
    body: createAccessRequestSchema,
  }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      const request = await createAccessRequest(
        req.params.id,
        userId,
        req.body.message
      );
      res.status(201).json({
        id: request.id,
        status: request.status,
        requestedAt: request.requestedAt,
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Get pending access requests for a session (owner only)
 * GET /v1/sessions/:id/access-requests
 */
router.get(
  "/v1/sessions/:id/access-requests",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      const requests = await getPendingRequests(req.params.id, userId);
      res.json({ requests });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Get count of pending access requests (owner only)
 * GET /v1/sessions/:id/access-requests/count
 */
router.get(
  "/v1/sessions/:id/access-requests/count",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      const result = await getPendingCount(req.params.id, userId);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Accept an access request (owner only)
 * POST /v1/sessions/:id/access-requests/:requestId/accept
 */
router.post(
  "/v1/sessions/:id/access-requests/:requestId/accept",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema, requestId: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      const result = await acceptRequest(
        req.params.id,
        req.params.requestId,
        userId
      );
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Reject an access request (owner only)
 * POST /v1/sessions/:id/access-requests/:requestId/reject
 */
router.post(
  "/v1/sessions/:id/access-requests/:requestId/reject",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema, requestId: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      const result = await rejectRequest(
        req.params.id,
        req.params.requestId,
        userId
      );
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
