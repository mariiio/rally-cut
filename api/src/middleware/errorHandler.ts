import type { ErrorRequestHandler } from "express";
import { ZodError } from "zod";

export type ErrorCode =
  | "VALIDATION_ERROR"
  | "NOT_FOUND"
  | "LIMIT_EXCEEDED"
  | "CONFLICT"
  | "INTERNAL_ERROR";

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
