import { z } from "zod";

export const servingTeamSchema = z.enum(["A", "B"]);

export const createRallySchema = z.object({
  startMs: z.number().int().nonnegative(),
  endMs: z.number().int().positive(),
  confidence: z.number().min(0).max(1).optional(),
  scoreA: z.number().int().nonnegative().optional(),
  scoreB: z.number().int().nonnegative().optional(),
  servingTeam: servingTeamSchema.optional(),
  notes: z.string().max(1000).optional(),
});

export const updateRallySchema = z.object({
  startMs: z.number().int().nonnegative().optional(),
  endMs: z.number().int().positive().optional(),
  scoreA: z.number().int().nonnegative().optional().nullable(),
  scoreB: z.number().int().nonnegative().optional().nullable(),
  servingTeam: servingTeamSchema.optional().nullable(),
  notes: z.string().max(1000).optional().nullable(),
});

export type CreateRallyInput = z.infer<typeof createRallySchema>;
export type UpdateRallyInput = z.infer<typeof updateRallySchema>;
