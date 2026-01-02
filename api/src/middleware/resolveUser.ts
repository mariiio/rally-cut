import type { RequestHandler } from "express";
import { getOrCreateUser } from "../services/userService.js";

// Extend Express Request to include user
declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      userId?: string;
    }
  }
}

const VISITOR_ID_HEADER = "x-visitor-id";

/**
 * Middleware to resolve user from X-Visitor-Id header.
 * Creates a new anonymous user if the visitor ID is new.
 * Attaches userId to req.userId.
 */
export const resolveUser: RequestHandler = async (req, _res, next) => {
  try {
    const visitorId = req.headers[VISITOR_ID_HEADER];

    if (typeof visitorId !== "string" || visitorId.trim() === "") {
      // No visitor ID provided - proceed without user context
      // Some endpoints may require it and will return 400
      next();
      return;
    }

    const userAgent = req.headers["user-agent"];
    const result = await getOrCreateUser(visitorId.trim(), userAgent);
    req.userId = result.userId;

    next();
  } catch (error) {
    next(error);
  }
};

/**
 * Middleware to require a user (returns 400 if no visitor ID).
 * Must be used after resolveUser middleware.
 */
export const requireUser: RequestHandler = (req, res, next) => {
  if (req.userId === undefined || req.userId === "") {
    res.status(400).json({
      error: "Bad Request",
      message: "X-Visitor-Id header is required",
    });
    return;
  }
  next();
};
