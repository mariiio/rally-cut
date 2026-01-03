import { z } from "zod";

export const requestUploadUrlSchema = z.object({
  filename: z.string().min(1).max(255),
  // SHA-256 hex hash (64 chars). For files >50MB, computed from first/last 10MB + metadata.
  contentHash: z.string().length(64),
  fileSize: z.number().int().positive(), // Tier-based limit enforced in service
  durationMs: z.number().int().positive().optional(),
  contentType: z
    .string()
    .regex(/^video\//)
    .default("video/mp4"),
});

export const confirmUploadSchema = z.object({
  videoId: z.string().uuid(),
  durationMs: z.number().int().positive(),
  width: z.number().int().positive().optional(),
  height: z.number().int().positive().optional(),
});

export const updateVideoSchema = z.object({
  name: z.string().min(1).max(255).optional(),
});

// Multipart upload schemas
export const initiateMultipartSchema = z.object({
  filename: z.string().min(1).max(255),
  contentHash: z.string().length(64),
  fileSize: z.number().int().positive(),
  durationMs: z.number().int().positive().optional(),
  contentType: z.string().regex(/^video\//).default("video/mp4"),
});

export const completeMultipartSchema = z.object({
  uploadId: z.string().min(1),
  parts: z
    .array(
      z.object({
        partNumber: z.number().int().positive(),
        etag: z.string().min(1),
      })
    )
    .min(1),
});

export const abortMultipartSchema = z.object({
  uploadId: z.string().min(1),
});

export type RequestUploadUrlInput = z.infer<typeof requestUploadUrlSchema>;
export type ConfirmUploadInput = z.infer<typeof confirmUploadSchema>;
export type UpdateVideoInput = z.infer<typeof updateVideoSchema>;
export type InitiateMultipartInput = z.infer<typeof initiateMultipartSchema>;
export type CompleteMultipartInput = z.infer<typeof completeMultipartSchema>;
export type AbortMultipartInput = z.infer<typeof abortMultipartSchema>;
