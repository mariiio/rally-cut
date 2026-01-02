import { z } from "zod";

export const shareTokenParamSchema = z.object({
  token: z.string().uuid(),
});

export const sessionIdParamSchema = z.object({
  id: z.string().uuid(),
});

export const memberIdParamSchema = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
});

export type ShareTokenParam = z.infer<typeof shareTokenParamSchema>;
export type SessionIdParam = z.infer<typeof sessionIdParamSchema>;
export type MemberIdParam = z.infer<typeof memberIdParamSchema>;
