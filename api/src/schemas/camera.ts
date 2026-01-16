import { z } from "zod";

export const cameraEasingSchema = z.enum([
  "LINEAR",
  "EASE_IN",
  "EASE_OUT",
  "EASE_IN_OUT",
]);

export const aspectRatioSchema = z.enum(["ORIGINAL", "VERTICAL"]);

export const cameraKeyframeSchema = z.object({
  id: z.string().optional(),
  timeOffset: z.number().min(0).max(1),
  positionX: z.number().min(0).max(1),
  positionY: z.number().min(0).max(1),
  zoom: z.number().min(0.5).max(4),
  rotation: z.number().min(-180).max(180).default(0),
  easing: cameraEasingSchema,
});

export const rallyCameraEditSchema = z.object({
  enabled: z.boolean(),
  aspectRatio: aspectRatioSchema,
  keyframes: z.array(cameraKeyframeSchema),
});

// Global camera settings at video level (applies as base for all rallies)
export const globalCameraSettingsSchema = z.object({
  zoom: z.number().min(0.5).max(4).default(1.0),
  positionX: z.number().min(0).max(1).default(0.5),
  positionY: z.number().min(0).max(1).default(0.5),
  rotation: z.number().min(-180).max(180).default(0),
});

export type CameraKeyframeInput = z.infer<typeof cameraKeyframeSchema>;
export type RallyCameraEditInput = z.infer<typeof rallyCameraEditSchema>;
export type GlobalCameraSettingsInput = z.infer<typeof globalCameraSettingsSchema>;
