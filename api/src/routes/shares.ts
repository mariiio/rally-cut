import { Router } from "express";
import { z } from "zod";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import {
  acceptShare,
  createShare,
  deleteShare,
  getShare,
  getSharePreview,
  removeMember,
} from "../services/shareService.js";

const router = Router();

/**
 * Create or get existing share link for a session
 * POST /v1/sessions/:id/share
 */
router.post(
  "/v1/sessions/:id/share",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const share = await createShare(req.params.id, userId);
      res.status(201).json({
        token: share.token,
        createdAt: share.createdAt,
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Get share info including members (owner only)
 * GET /v1/sessions/:id/share
 */
router.get(
  "/v1/sessions/:id/share",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const share = await getShare(req.params.id, userId);
      if (!share) {
        res.json(null);
        return;
      }
      res.json({
        token: share.token,
        createdAt: share.createdAt,
        members: share.members.map((m) => ({
          userId: m.user.id,
          name: m.user.name,
          email: m.user.email,
          avatarUrl: m.user.avatarUrl,
          joinedAt: m.joinedAt,
        })),
      });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Delete share (revokes all access)
 * DELETE /v1/sessions/:id/share
 */
router.delete(
  "/v1/sessions/:id/share",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      await deleteShare(req.params.id, userId);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Remove a specific member from shared session
 * DELETE /v1/sessions/:id/share/members/:userId
 */
router.delete(
  "/v1/sessions/:id/share/members/:userId",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema, userId: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      await removeMember(req.params.id, userId, req.params.userId);
      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Get share preview (public - for accept page)
 * GET /v1/share/:token
 */
router.get(
  "/v1/share/:token",
  validateRequest({ params: z.object({ token: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const preview = await getSharePreview(req.params.token);
      res.json(preview);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Accept a share invite
 * POST /v1/share/:token/accept
 */
router.post(
  "/v1/share/:token/accept",
  requireUser,
  validateRequest({
    params: z.object({ token: uuidSchema }),
    body: z.object({ name: z.string().min(1).max(100).optional() }).optional(),
  }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const name = req.body?.name;
      const result = await acceptShare(req.params.token, userId, name);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
