import { z } from "zod";
import { rallyCameraEditSchema, globalCameraSettingsSchema } from "./camera.js";

const rallyInputSchema = z.object({
  id: z.string().uuid().optional(),
  startMs: z.number().int().min(0),
  endMs: z.number().int().min(0),
  // nullish() allows: undefined (field omitted), null (explicit deletion), or object (create/update)
  cameraEdit: rallyCameraEditSchema.nullish(),
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
  // Global camera settings per video (videoId -> settings)
  // nullish per-video: undefined (omit), null (delete), object (create/update)
  globalCameraSettings: z.record(z.string().uuid(), globalCameraSettingsSchema.nullish()).optional(),
});

export type SyncStateInput = z.infer<typeof syncStateSchema>;
