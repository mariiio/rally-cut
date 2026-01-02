import { z } from "zod";

export const visitorIdSchema = z.string().min(1).max(100);

export const identityRequestSchema = z.object({
  visitorId: visitorIdSchema,
});

export const userResponseSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email().nullable(),
  name: z.string().nullable(),
  avatarUrl: z.string().url().nullable(),
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
});

export type VisitorId = z.infer<typeof visitorIdSchema>;
export type IdentityRequest = z.infer<typeof identityRequestSchema>;
export type UserResponse = z.infer<typeof userResponseSchema>;
export type IdentityResponse = z.infer<typeof identityResponseSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;
