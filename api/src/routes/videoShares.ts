import { Router } from "express";
import { z } from "zod";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import {
  createVideoShares,
  deleteVideoShares,
  getVideoShares,
  removeVideoMember,
  updateVideoMemberRole,
} from "../services/videoShareService.js";

const memberRoleSchema = z.enum(["VIEWER", "EDITOR", "ADMIN"]);

const router = Router();

/**
 * Create share links for a video (creates all 3 role links atomically)
 * POST /v1/videos/:id/share
 */
router.post(
  "/v1/videos/:id/share",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      const shares = await createVideoShares(req.params.id, userId);
      res.status(201).json({ shares });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Get share info including all share links and members (owner or admin)
 * GET /v1/videos/:id/share
 */
router.get(
  "/v1/videos/:id/share",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      const result = await getVideoShares(req.params.id, userId);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Delete all shares (revokes all access)
 * DELETE /v1/videos/:id/share
 */
router.delete(
  "/v1/videos/:id/share",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      await deleteVideoShares(req.params.id, userId);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Remove a specific member from shared video
 * DELETE /v1/videos/:id/share/members/:userId
 */
router.delete(
  "/v1/videos/:id/share/members/:userId",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema, userId: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      await removeVideoMember(req.params.id, userId, req.params.userId);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Update a member's role
 * PATCH /v1/videos/:id/share/members/:userId/role
 */
router.patch(
  "/v1/videos/:id/share/members/:userId/role",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema, userId: uuidSchema }),
    body: z.object({ role: memberRoleSchema }),
  }),
  async (req, res, next) => {
    try {
      const actorId = req.userId as string;
      const updated = await updateVideoMemberRole(
        req.params.id,
        actorId,
        req.params.userId,
        req.body.role
      );
      res.json({ role: updated.role });
    } catch (error) {
      next(error);
    }
  }
);

export default router;
