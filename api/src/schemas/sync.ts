import { z } from "zod";

const rallyInputSchema = z.object({
  id: z.string().uuid().optional(),
  startMs: z.number().int().min(0),
  endMs: z.number().int().min(0),
});

const highlightInputSchema = z.object({
  id: z.string().uuid().optional(),
  name: z.string().min(1).max(255),
  color: z.string().regex(/^#[0-9A-Fa-f]{6}$/),
  rallyIds: z.array(z.string()),
});

export const syncStateSchema = z.object({
  ralliesPerVideo: z.record(z.string().uuid(), z.array(rallyInputSchema)),
  highlights: z.array(highlightInputSchema),
});

export type SyncStateInput = z.infer<typeof syncStateSchema>;
