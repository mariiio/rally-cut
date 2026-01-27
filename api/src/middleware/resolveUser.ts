import type { RequestHandler } from "express";
import jwt from "jsonwebtoken";
import { env } from "../config/env.js";
import { getOrCreateUser } from "../services/userService.js";

// Extend Express Request to include user
declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      userId?: string;
      isAuthenticated?: boolean;
    }
  }
}

const VISITOR_ID_HEADER = "x-visitor-id";

/**
 * Middleware to resolve user from JWT token or X-Visitor-Id header.
 * JWT takes precedence over visitor ID.
 * Creates a new anonymous user if the visitor ID is new.
 * Attaches userId and isAuthenticated to req.
 */
export const resolveUser: RequestHandler = async (req, _res, next) => {
  try {
    // 1. Try JWT authentication first
    const authHeader = req.headers.authorization;
    if (authHeader?.startsWith("Bearer ") && env.AUTH_JWT_SECRET) {
      const token = authHeader.slice(7);
      try {
        const decoded = jwt.verify(token, env.AUTH_JWT_SECRET, {
          algorithms: ["HS256"],
        }) as jwt.JwtPayload;

        if (decoded.sub) {
          req.userId = decoded.sub;
          req.isAuthenticated = true;
          next();
          return;
        }
      } catch {
        // Invalid JWT - fall through to visitor ID
      }
    }

    // 2. Fall back to X-Visitor-Id
    const visitorId = req.headers[VISITOR_ID_HEADER];

    if (typeof visitorId !== "string" || visitorId.trim() === "") {
      // No visitor ID provided - proceed without user context
      req.isAuthenticated = false;
      next();
      return;
    }

    const userAgent = req.headers["user-agent"];
    const result = await getOrCreateUser(visitorId.trim(), userAgent);
    req.userId = result.userId;
    req.isAuthenticated = false;

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

/**
 * Middleware to require an authenticated user (JWT-based).
 * Returns 401 if user is not authenticated.
 * Must be used after resolveUser middleware.
 */
export const requireAuthenticated: RequestHandler = (req, res, next) => {
  if (!req.isAuthenticated || !req.userId) {
    res.status(401).json({
      error: {
        code: "AUTH_REQUIRED",
        message: "Please sign in to use this feature",
      },
    });
    return;
  }
  next();
};
