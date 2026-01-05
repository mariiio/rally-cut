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
  easing: cameraEasingSchema,
});

export const rallyCameraEditSchema = z.object({
  enabled: z.boolean(),
  aspectRatio: aspectRatioSchema,
  keyframes: z.array(cameraKeyframeSchema),
});

export type CameraKeyframeInput = z.infer<typeof cameraKeyframeSchema>;
export type RallyCameraEditInput = z.infer<typeof rallyCameraEditSchema>;
