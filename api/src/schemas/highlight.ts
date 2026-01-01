import { z } from "zod";

export const createHighlightSchema = z.object({
  name: z.string().min(1).max(255),
  color: z
    .string()
    .regex(/^#[0-9A-Fa-f]{6}$/)
    .default("#FF6B6B"),
});

export const updateHighlightSchema = z.object({
  name: z.string().min(1).max(255).optional(),
  color: z
    .string()
    .regex(/^#[0-9A-Fa-f]{6}$/)
    .optional(),
});

export const addRallyToHighlightSchema = z.object({
  rallyId: z.string().uuid(),
  order: z.number().int().nonnegative().optional(),
});

export type CreateHighlightInput = z.infer<typeof createHighlightSchema>;
export type UpdateHighlightInput = z.infer<typeof updateHighlightSchema>;
export type AddRallyToHighlightInput = z.infer<typeof addRallyToHighlightSchema>;
