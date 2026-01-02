import { Router } from "express";
import { validateRequest } from "../middleware/validateRequest.js";
import { requireUser } from "../middleware/resolveUser.js";
import {
  identityRequestSchema,
  updateUserSchema,
  type IdentityRequest,
} from "../schemas/user.js";
import * as userService from "../services/userService.js";

const router = Router();

/**
 * POST /v1/identity
 * Create or get user from visitor ID.
 */
router.post(
  "/v1/identity",
  validateRequest({ body: identityRequestSchema }),
  async (req, res, next) => {
    try {
      const body = req.body as IdentityRequest;
      const userAgent = req.headers["user-agent"] ?? undefined;
      const result = await userService.getOrCreateUser(body.visitorId, userAgent);
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);

/**
 * GET /v1/me
 * Get current user info from X-Visitor-Id header.
 */
router.get("/v1/me", requireUser, async (req, res, next) => {
  try {
    // requireUser middleware ensures userId exists
    const userId = req.userId as string;
    const user = await userService.getUserById(userId);
    res.json(user);
  } catch (error) {
    next(error);
  }
});

/**
 * PATCH /v1/me
 * Update current user profile.
 */
router.patch(
  "/v1/me",
  requireUser,
  validateRequest({ body: updateUserSchema }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      const user = await userService.updateUser(userId, req.body);
      res.json(user);
    } catch (error) {
      next(error);
    }
  }
);

export default router;
