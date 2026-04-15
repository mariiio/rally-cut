import type { ErrorRequestHandler } from "express";
import { ZodError } from "zod";

export type ErrorCode =
  | "VALIDATION_ERROR"
  | "NOT_FOUND"
  | "FORBIDDEN"
  | "ACCESS_DENIED"
  | "LIMIT_EXCEEDED"
  | "CONFLICT"
  | "INTERNAL_ERROR"
  | "LOCKED_RALLY_CANNOT_EXTEND"
  | "LOCKED_RALLY_CANNOT_SPLIT"
  | "LOCKED_RALLY_CANNOT_MERGE"
  | "LOCKED_RALLY_REQUIRES_CONFIRM"
  | "RALLY_TRACKING_IN_PROGRESS"
  | "RALLY_TRACKING_FAILED"
  | "RALLIES_OVERLAP"
  | "SPLIT_BOUNDS_INVALID";

export interface ApiError {
  error: {
    code: ErrorCode;
    message: string;
    details?: Record<string, unknown>;
  };
}

export class AppError extends Error {
  constructor(
    public readonly code: ErrorCode,
    message: string,
    public readonly statusCode: number = 500,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = "AppError";
  }

  toResponse(): ApiError {
    return {
      error: {
        code: this.code,
        message: this.message,
        details: this.details,
      },
    };
  }
}

export class NotFoundError extends AppError {
  constructor(resource: string, id?: string) {
    super(
      "NOT_FOUND",
      id !== undefined
        ? `${resource} with id '${id}' not found`
        : `${resource} not found`,
      404
    );
  }
}

export class ValidationError extends AppError {
  constructor(message: string, details?: Record<string, unknown>) {
    super("VALIDATION_ERROR", message, 400, details);
  }
}

export class LimitExceededError extends AppError {
  constructor(message: string, details?: Record<string, unknown>) {
    super("LIMIT_EXCEEDED", message, 403, details);
  }
}

export class ConflictError extends AppError {
  constructor(message: string, details?: Record<string, unknown>) {
    super("CONFLICT", message, 409, details);
  }
}

export class ForbiddenError extends AppError {
  constructor(message: string, details?: Record<string, unknown>) {
    super("FORBIDDEN", message, 403, details);
  }
}

export class AccessDeniedError extends AppError {
  constructor(
    sessionName: string,
    ownerName: string | null,
    hasPendingRequest: boolean
  ) {
    super("ACCESS_DENIED", "You don't have access to this session", 403, {
      accessRequestable: true,
      sessionName,
      ownerName,
      hasPendingRequest,
    });
  }
}

type LockedOp = "EXTEND" | "SPLIT" | "MERGE";

export class LockedRallyError extends AppError {
  constructor(op: LockedOp, rallyId: string) {
    super(
      `LOCKED_RALLY_CANNOT_${op}` as ErrorCode,
      `Rally '${rallyId}' is canonical-locked; ${op.toLowerCase()} is not allowed. Unlock the rally first.`,
      409,
      { rallyId, op }
    );
    this.name = "LockedRallyError";
  }
}

export class LockedRallyRequiresConfirmError extends AppError {
  constructor(rallyId: string, gtFrameCount: number) {
    super(
      "LOCKED_RALLY_REQUIRES_CONFIRM",
      `Rally '${rallyId}' is canonical-locked with ${gtFrameCount} GT frames. Pass {confirmUnlock: true} to proceed.`,
      409,
      { rallyId, gtFrameCount }
    );
    this.name = "LockedRallyRequiresConfirmError";
  }
}

type RallyTrackingFailReason = "IN_PROGRESS" | "FAILED";

export class RallyTrackingStateError extends AppError {
  constructor(reason: RallyTrackingFailReason, rallyId: string) {
    super(
      `RALLY_TRACKING_${reason}` as ErrorCode,
      reason === "IN_PROGRESS"
        ? `Rally '${rallyId}' is currently being tracked. Retry once tracking completes.`
        : `Rally '${rallyId}' tracking failed. Retrack or delete before retrying this operation.`,
      409,
      { rallyId, reason }
    );
    this.name = "RallyTrackingStateError";
  }
}

export class RalliesOverlapError extends AppError {
  constructor(rallyIds: string[]) {
    super(
      "RALLIES_OVERLAP",
      "Rallies overlap in time and cannot be merged.",
      400,
      { rallyIds }
    );
    this.name = "RalliesOverlapError";
  }
}

export class SplitBoundsError extends AppError {
  constructor(detail: string, bounds: Record<string, number>) {
    super(
      "SPLIT_BOUNDS_INVALID",
      `Split bounds invalid: ${detail}`,
      400,
      bounds
    );
    this.name = "SplitBoundsError";
  }
}

export const errorHandler: ErrorRequestHandler = (err, _req, res, _next) => {
  if (err instanceof AppError) {
    res.status(err.statusCode).json(err.toResponse());
    return;
  }

  if (err instanceof ZodError) {
    const response: ApiError = {
      error: {
        code: "VALIDATION_ERROR",
        message: "Request validation failed",
        details: { issues: err.errors },
      },
    };
    res.status(400).json(response);
    return;
  }

  console.error("Unhandled error:", err);

  const response: ApiError = {
    error: {
      code: "INTERNAL_ERROR",
      message:
        process.env["NODE_ENV"] === "production"
          ? "An unexpected error occurred"
          : err instanceof Error
            ? err.message
            : "Unknown error",
    },
  };

  res.status(500).json(response);
};
