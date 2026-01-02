import { z } from "zod";

// No body required for initiation - uses current rallies
export const initiateConfirmationSchema = z.object({}).optional();

// Webhook payload from Lambda
export const confirmationCompleteWebhookSchema = z.object({
  confirmation_id: z.string().uuid(),
  status: z.enum(["completed", "failed"]),
  error_message: z.string().optional(),
  output_s3_key: z.string().optional(),
  duration_ms: z.number().int().positive().optional(),
});

// Response types
export const timestampMappingSchema = z.object({
  rallyId: z.string().uuid(),
  originalStartMs: z.number().int(),
  originalEndMs: z.number().int(),
  trimmedStartMs: z.number().int(),
  trimmedEndMs: z.number().int(),
});

export const confirmationStatusSchema = z.object({
  id: z.string().uuid(),
  status: z.enum(["PENDING", "PROCESSING", "CONFIRMED", "FAILED"]),
  progress: z.number().int().min(0).max(100),
  error: z.string().nullable().optional(),
  confirmedAt: z.date().nullable().optional(),
  originalDurationMs: z.number().int(),
  trimmedDurationMs: z.number().int().nullable().optional(),
  timestampMappings: z.array(timestampMappingSchema).optional(),
});

export const confirmationStatusResponseSchema = z.object({
  videoId: z.string().uuid(),
  confirmation: confirmationStatusSchema.nullable(),
});

export const initiateConfirmationResponseSchema = z.object({
  confirmationId: z.string().uuid(),
  status: z.enum(["PENDING", "PROCESSING", "CONFIRMED", "FAILED"]),
  progress: z.number().int().min(0).max(100),
  createdAt: z.date(),
});

export const restoreOriginalResponseSchema = z.object({
  success: z.boolean(),
});

export type ConfirmationCompleteWebhookInput = z.infer<
  typeof confirmationCompleteWebhookSchema
>;
export type TimestampMapping = z.infer<typeof timestampMappingSchema>;
export type ConfirmationStatus = z.infer<typeof confirmationStatusSchema>;
export type ConfirmationStatusResponse = z.infer<
  typeof confirmationStatusResponseSchema
>;
export type InitiateConfirmationResponse = z.infer<
  typeof initiateConfirmationResponseSchema
>;
export type RestoreOriginalResponse = z.infer<
  typeof restoreOriginalResponseSchema
>;
