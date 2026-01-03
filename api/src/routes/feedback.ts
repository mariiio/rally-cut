import { Router } from "express";
import { prisma } from "../lib/prisma.js";
import { validateRequest } from "../middleware/validateRequest.js";
import {
  createFeedbackSchema,
  type CreateFeedbackInput,
} from "../schemas/feedback.js";

const router = Router();

/**
 * POST /v1/feedback
 * Submit user feedback (bug report, feature request, or general feedback).
 * Does not require authentication - anonymous users can submit with optional email.
 */
router.post(
  "/v1/feedback",
  validateRequest({ body: createFeedbackSchema }),
  async (req, res, next) => {
    try {
      const body = req.body as CreateFeedbackInput;
      const userId = req.userId;
      const userAgent = req.headers["user-agent"] ?? null;

      const feedback = await prisma.feedback.create({
        data: {
          userId: userId ?? null,
          type: body.type,
          message: body.message,
          email: body.email ?? null,
          userAgent,
          pageUrl: body.pageUrl ?? null,
        },
      });

      res.status(201).json({
        id: feedback.id,
        type: feedback.type,
        message: feedback.message,
        createdAt: feedback.createdAt.toISOString(),
      });
    } catch (error) {
      next(error);
    }
  }
);

export default router;
