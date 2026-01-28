import { Router } from "express";
import { z } from "zod";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import { uuidSchema } from "../schemas/common.js";
import {
  acceptShare,
  createShares,
  deleteShares,
  getShares,
  getSharePreview,
  removeMember,
  updateMemberRole,
} from "../services/shareService.js";

const memberRoleSchema = z.enum(["VIEWER", "EDITOR", "ADMIN"]);

const router = Router();

/**
 * Create share links for a session (creates all 3 role links atomically)
 * POST /v1/sessions/:id/share
 */
router.post(
  "/v1/sessions/:id/share",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema }),
  }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const shares = await createShares(req.params.id, userId);
      res.status(201).json({ shares });
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Get share info including all share links and members (owner or admin)
 * GET /v1/sessions/:id/share
 */
router.get(
  "/v1/sessions/:id/share",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      const result = await getShares(req.params.id, userId);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * Delete all shares (revokes all access)
 * DELETE /v1/sessions/:id/share
 */
router.delete(
  "/v1/sessions/:id/share",
  requireUser,
  validateRequest({ params: z.object({ id: uuidSchema }) }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string; // Guaranteed by requireUser
      await deleteShares(req.params.id, userId);
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
 * Update a member's role
 * PATCH /v1/sessions/:id/share/members/:userId/role
 */
router.patch(
  "/v1/sessions/:id/share/members/:userId/role",
  requireUser,
  validateRequest({
    params: z.object({ id: uuidSchema, userId: uuidSchema }),
    body: z.object({ role: memberRoleSchema }),
  }),
  async (req, res, next) => {
    try {
      const actorId = req.userId as string;
      const updated = await updateMemberRole(
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
