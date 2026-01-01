import { z } from "zod";

export const requestUploadUrlSchema = z.object({
  filename: z.string().min(1).max(255),
  contentHash: z.string().length(64),
  fileSize: z.number().int().positive().max(2 * 1024 * 1024 * 1024),
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

export type RequestUploadUrlInput = z.infer<typeof requestUploadUrlSchema>;
export type ConfirmUploadInput = z.infer<typeof confirmUploadSchema>;
export type UpdateVideoInput = z.infer<typeof updateVideoSchema>;
