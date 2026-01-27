import { z } from "zod";

export const visitorIdSchema = z.string().min(1).max(100);

export const identityRequestSchema = z.object({
  visitorId: visitorIdSchema,
});

export const tierLimitsSchema = z.object({
  detectionsPerMonth: z.number().int().nonnegative(),
  maxVideoDurationMs: z.number().int().nonnegative(),
  maxFileSizeBytes: z.number().int().nonnegative(),
  monthlyUploadCount: z.number().int().nonnegative(),
  storageCapBytes: z.number().int().nonnegative(),
  exportQuality: z.enum(["720p", "original"]),
  exportWatermark: z.boolean(),
  lambdaExportEnabled: z.boolean(),
  originalQualityDays: z.number().int().nonnegative().nullable(),
  inactivityDeleteDays: z.number().int().nonnegative().nullable(),
  serverSyncEnabled: z.boolean(),
  highlightsEnabled: z.boolean(),
});

export const usageQuotaSchema = z.object({
  detectionsUsed: z.number().int().nonnegative(),
  detectionsLimit: z.number().int().nonnegative(),
  detectionsRemaining: z.number().int().nonnegative(),
  uploadsThisMonth: z.number().int().nonnegative(),
  uploadsLimit: z.number().int().nonnegative(),
  uploadsRemaining: z.number().int().nonnegative(),
  storageUsedBytes: z.number().int().nonnegative(),
  storageLimitBytes: z.number().int().nonnegative(),
  storageRemainingBytes: z.number().int().nonnegative(),
  periodStart: z.string().datetime(),
});

export const userResponseSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email().nullable(),
  name: z.string().nullable(),
  avatarUrl: z.string().url().nullable(),
  tier: z.enum(["FREE", "PRO", "ELITE"]),
  tierLimits: tierLimitsSchema,
  usage: usageQuotaSchema,
  createdAt: z.string().datetime(),
  convertedAt: z.string().datetime().nullable(),
  videoCount: z.number().int().nonnegative(),
  sessionCount: z.number().int().nonnegative(),
});

export const identityResponseSchema = z.object({
  userId: z.string().uuid(),
  isNew: z.boolean(),
});

export const updateUserSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  avatarUrl: z.string().url().max(2048).nullable().optional(),
});

export type VisitorId = z.infer<typeof visitorIdSchema>;
export type IdentityRequest = z.infer<typeof identityRequestSchema>;
export type TierLimits = z.infer<typeof tierLimitsSchema>;
export type UsageQuota = z.infer<typeof usageQuotaSchema>;
export type UserResponse = z.infer<typeof userResponseSchema>;
export type IdentityResponse = z.infer<typeof identityResponseSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;
