import { Router } from "express";
import { z } from "zod";
import crypto from "crypto";
import bcrypt from "bcryptjs";
import { Prisma } from "@prisma/client";
import { prisma } from "../lib/prisma.js";
import { requireUser } from "../middleware/resolveUser.js";
import { validateRequest } from "../middleware/validateRequest.js";
import {
  isDisposableEmail,
  sendVerificationEmail,
} from "../services/mailService.js";
import { env } from "../config/env.js";

const router = Router();

// ============================================================================
// POST /v1/auth/register
// ============================================================================

const registerSchema = z.object({
  email: z.string().email().max(255),
  password: z.string().min(8).max(128),
  name: z.string().min(1).max(100).optional(),
});

router.post(
  "/v1/auth/register",
  requireUser,
  validateRequest({ body: registerSchema }),
  async (req, res, next) => {
    try {
      const userId = req.userId as string;
      const { email, password, name } = req.body;
      const normalizedEmail = email.toLowerCase().trim();

      // Check disposable email
      if (await isDisposableEmail(normalizedEmail)) {
        res.status(400).json({
          error: {
            code: "DISPOSABLE_EMAIL",
            message: "Disposable email addresses are not allowed",
          },
        });
        return;
      }

      // Hash password
      const passwordHash = await bcrypt.hash(password, 12);

      // Create verification token
      const token = crypto.randomUUID();
      const expires = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours

      const isDev = env.NODE_ENV === "development";

      // Update existing anonymous user record (in-place conversion)
      // The unique constraint on email handles the race condition
      try {
        await prisma.$transaction(async (tx) => {
          await tx.user.update({
            where: { id: userId },
            data: {
              email: normalizedEmail,
              passwordHash,
              name: name || undefined,
              convertedAt: new Date(),
              // Auto-verify in development to skip email roundtrip
              ...(isDev ? { emailVerified: new Date() } : {}),
            },
          });

          // Still create token (for the verification URL log) but it's optional in dev
          await tx.verificationToken.create({
            data: {
              identifier: normalizedEmail,
              token,
              expires,
            },
          });
        });
      } catch (error) {
        if (
          error instanceof Prisma.PrismaClientKnownRequestError &&
          error.code === "P2002"
        ) {
          res.status(409).json({
            error: {
              code: "EMAIL_EXISTS",
              message: "An account with this email already exists",
            },
          });
          return;
        }
        throw error;
      }

      // Send verification email (fire-and-forget)
      sendVerificationEmail(normalizedEmail, token).catch((err) => {
        console.error("Failed to send verification email:", err);
      });

      res.status(201).json({
        message: isDev
          ? "Registration successful. Email auto-verified in development."
          : "Registration successful. Please verify your email.",
        autoVerified: isDev,
      });
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// POST /v1/auth/verify-email
// ============================================================================

const verifyEmailSchema = z.object({
  token: z.string().min(1),
});

router.post(
  "/v1/auth/verify-email",
  validateRequest({ body: verifyEmailSchema }),
  async (req, res, next) => {
    try {
      const { token } = req.body;

      const verificationToken = await prisma.verificationToken.findUnique({
        where: { token },
      });

      if (!verificationToken) {
        res.status(400).json({
          error: {
            code: "INVALID_TOKEN",
            message: "Invalid or expired verification token",
          },
        });
        return;
      }

      if (verificationToken.expires < new Date()) {
        // Clean up expired token
        await prisma.verificationToken.delete({
          where: { token },
        });
        res.status(400).json({
          error: {
            code: "TOKEN_EXPIRED",
            message:
              "Verification token has expired. Please request a new one.",
          },
        });
        return;
      }

      // Set emailVerified and delete token
      await prisma.$transaction(async (tx) => {
        await tx.user.update({
          where: { email: verificationToken.identifier },
          data: { emailVerified: new Date() },
        });

        await tx.verificationToken.delete({
          where: { token },
        });
      });

      res.json({ message: "Email verified successfully" });
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// POST /v1/auth/resend-verification
// ============================================================================

const RESEND_COOLDOWN_MS = 2 * 60 * 1000; // 2 minutes
const RESEND_MAP_MAX_SIZE = 10_000;

// Rate limit tracking (in-memory, reset on restart)
const resendTimestamps = new Map<string, number>();

function cleanupResendTimestamps() {
  if (resendTimestamps.size <= RESEND_MAP_MAX_SIZE) return;
  const now = Date.now();
  for (const [key, ts] of resendTimestamps) {
    if (now - ts > RESEND_COOLDOWN_MS) {
      resendTimestamps.delete(key);
    }
  }
}

router.post(
  "/v1/auth/resend-verification",
  requireUser,
  async (req, res, next) => {
    try {
      const userId = req.userId as string;

      const user = await prisma.user.findUnique({
        where: { id: userId },
        select: { email: true, emailVerified: true },
      });

      if (!user?.email) {
        res.status(400).json({
          error: {
            code: "NO_EMAIL",
            message: "No email address found for this account",
          },
        });
        return;
      }

      if (user.emailVerified) {
        res.status(400).json({
          error: {
            code: "ALREADY_VERIFIED",
            message: "Email is already verified",
          },
        });
        return;
      }

      // Rate limit: max 1 per 2 minutes
      const lastSent = resendTimestamps.get(userId);
      if (lastSent && Date.now() - lastSent < RESEND_COOLDOWN_MS) {
        res.status(429).json({
          error: {
            code: "RATE_LIMITED",
            message:
              "Please wait before requesting another verification email",
          },
        });
        return;
      }

      // Delete old tokens for this email
      await prisma.verificationToken.deleteMany({
        where: { identifier: user.email },
      });

      // Create new token
      const token = crypto.randomUUID();
      const expires = new Date(Date.now() + 24 * 60 * 60 * 1000);

      await prisma.verificationToken.create({
        data: {
          identifier: user.email,
          token,
          expires,
        },
      });

      resendTimestamps.set(userId, Date.now());
      cleanupResendTimestamps();

      sendVerificationEmail(user.email, token).catch((err) => {
        console.error("Failed to send verification email:", err);
      });

      res.json({ message: "Verification email sent" });
    } catch (error) {
      next(error);
    }
  }
);

// ============================================================================
// POST /v1/auth/link-anonymous
// ============================================================================

const linkAnonymousSchema = z.object({
  visitorId: z.string().min(1),
});

router.post(
  "/v1/auth/link-anonymous",
  validateRequest({ body: linkAnonymousSchema }),
  async (req, res, next) => {
    try {
      const authenticatedUserId = req.userId;

      if (!req.isAuthenticated || !authenticatedUserId) {
        res.status(401).json({
          error: {
            code: "AUTH_REQUIRED",
            message: "Must be authenticated to link anonymous data",
          },
        });
        return;
      }

      const { visitorId } = req.body;

      // Find the anonymous user by visitorId
      const anonIdentity = await prisma.anonymousIdentity.findUnique({
        where: { visitorId },
        select: { userId: true },
      });

      if (!anonIdentity) {
        res.json({ linked: false, message: "No anonymous data found" });
        return;
      }

      // If same user, nothing to merge
      if (anonIdentity.userId === authenticatedUserId) {
        res.json({ linked: true, message: "Already linked" });
        return;
      }

      const anonUserId = anonIdentity.userId;

      // Merge all data from anonymous user to authenticated user
      await prisma.$transaction(async (tx) => {
        // Transfer videos
        await tx.video.updateMany({
          where: { userId: anonUserId },
          data: { userId: authenticatedUserId },
        });

        // Transfer sessions (owned)
        await tx.session.updateMany({
          where: { userId: anonUserId },
          data: { userId: authenticatedUserId },
        });

        // Transfer highlights
        await tx.highlight.updateMany({
          where: { createdByUserId: anonUserId },
          data: { createdByUserId: authenticatedUserId },
        });

        // For tables with unique(entityId, userId) constraints,
        // delete anon records that would conflict before transferring the rest
        const authCameraRallies = await tx.rallyCameraEdit.findMany({
          where: { userId: authenticatedUserId },
          select: { rallyId: true },
        });
        if (authCameraRallies.length > 0) {
          await tx.rallyCameraEdit.deleteMany({
            where: {
              userId: anonUserId,
              rallyId: { in: authCameraRallies.map((r) => r.rallyId) },
            },
          });
        }
        await tx.rallyCameraEdit.updateMany({
          where: { userId: anonUserId },
          data: { userId: authenticatedUserId },
        });

        const authVideoSettings = await tx.videoCameraSettings.findMany({
          where: { userId: authenticatedUserId },
          select: { videoId: true },
        });
        if (authVideoSettings.length > 0) {
          await tx.videoCameraSettings.deleteMany({
            where: {
              userId: anonUserId,
              videoId: { in: authVideoSettings.map((v) => v.videoId) },
            },
          });
        }
        await tx.videoCameraSettings.updateMany({
          where: { userId: anonUserId },
          data: { userId: authenticatedUserId },
        });

        // Session memberships: delete conflicts, transfer rest
        const authMemberships = await tx.sessionMember.findMany({
          where: { userId: authenticatedUserId },
          select: { sessionShareId: true },
        });
        if (authMemberships.length > 0) {
          await tx.sessionMember.deleteMany({
            where: {
              userId: anonUserId,
              sessionShareId: {
                in: authMemberships.map((m) => m.sessionShareId),
              },
            },
          });
        }
        await tx.sessionMember.updateMany({
          where: { userId: anonUserId },
          data: { userId: authenticatedUserId },
        });

        // Transfer feedback (no unique constraints to worry about)
        await tx.feedback.updateMany({
          where: { userId: anonUserId },
          data: { userId: authenticatedUserId },
        });

        // Access requests: delete conflicts, transfer rest
        const authRequests = await tx.accessRequest.findMany({
          where: { userId: authenticatedUserId },
          select: { sessionId: true },
        });
        if (authRequests.length > 0) {
          await tx.accessRequest.deleteMany({
            where: {
              userId: anonUserId,
              sessionId: { in: authRequests.map((r) => r.sessionId) },
            },
          });
        }
        await tx.accessRequest.updateMany({
          where: { userId: anonUserId },
          data: { userId: authenticatedUserId },
        });

        // Merge usage quotas
        const anonQuota = await tx.userUsageQuota.findUnique({
          where: { userId: anonUserId },
        });
        if (anonQuota) {
          const authQuota = await tx.userUsageQuota.findUnique({
            where: { userId: authenticatedUserId },
          });
          if (authQuota) {
            await tx.userUsageQuota.update({
              where: { userId: authenticatedUserId },
              data: {
                detectionsUsed:
                  authQuota.detectionsUsed + anonQuota.detectionsUsed,
                uploadsThisMonth:
                  authQuota.uploadsThisMonth + anonQuota.uploadsThisMonth,
              },
            });
          }
          await tx.userUsageQuota.delete({
            where: { userId: anonUserId },
          });
        }

        // Re-point anonymous identity to authenticated user
        await tx.anonymousIdentity.update({
          where: { visitorId },
          data: { userId: authenticatedUserId },
        });

        // Delete orphaned anonymous user (cascade handles remaining refs)
        await tx.user.delete({
          where: { id: anonUserId },
        });
      });

      res.json({ linked: true });
    } catch (error) {
      next(error);
    }
  }
);

export default router;
