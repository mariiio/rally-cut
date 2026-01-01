import { z } from "zod";

export const createSessionSchema = z.object({
  name: z.string().min(1).max(255),
});

export const updateSessionSchema = z.object({
  name: z.string().min(1).max(255).optional(),
});

export type CreateSessionInput = z.infer<typeof createSessionSchema>;
export type UpdateSessionInput = z.infer<typeof updateSessionSchema>;
