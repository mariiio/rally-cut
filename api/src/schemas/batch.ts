import { z } from "zod";
import { createHighlightSchema, updateHighlightSchema } from "./highlight.js";
import { createRallySchema, updateRallySchema } from "./rally.js";
import { updateVideoSchema } from "./video.js";

const createOperationSchema = z.discriminatedUnion("entity", [
  z.object({
    type: z.literal("create"),
    entity: z.literal("rally"),
    tempId: z.string(),
    data: createRallySchema.extend({ videoId: z.string().uuid() }),
  }),
  z.object({
    type: z.literal("create"),
    entity: z.literal("highlight"),
    tempId: z.string(),
    data: createHighlightSchema,
  }),
]);

const updateOperationSchema = z.discriminatedUnion("entity", [
  z.object({
    type: z.literal("update"),
    entity: z.literal("video"),
    id: z.string().uuid(),
    data: updateVideoSchema,
  }),
  z.object({
    type: z.literal("update"),
    entity: z.literal("rally"),
    id: z.string().uuid(),
    data: updateRallySchema,
  }),
  z.object({
    type: z.literal("update"),
    entity: z.literal("highlight"),
    id: z.string().uuid(),
    data: updateHighlightSchema,
  }),
]);

const deleteOperationSchema = z.object({
  type: z.literal("delete"),
  entity: z.enum(["video", "rally", "highlight", "highlightRally"]),
  id: z.string().uuid(),
});

const reorderOperationSchema = z.object({
  type: z.literal("reorder"),
  entity: z.enum(["video", "rally", "highlightRally"]),
  parentId: z.string().uuid(),
  order: z.array(z.string().uuid()),
});

const addRallyToHighlightOperationSchema = z.object({
  type: z.literal("addRallyToHighlight"),
  highlightId: z.string().uuid(),
  rallyId: z.string().uuid(),
  tempId: z.string().optional(),
});

const operationSchema = z.union([
  createOperationSchema,
  updateOperationSchema,
  deleteOperationSchema,
  reorderOperationSchema,
  addRallyToHighlightOperationSchema,
]);

export const batchRequestSchema = z.object({
  operations: z.array(operationSchema).max(50),
});

export type BatchOperation = z.infer<typeof operationSchema>;
export type BatchRequest = z.infer<typeof batchRequestSchema>;

export interface BatchResponse {
  success: boolean;
  created: Record<string, string>;
  updatedAt: string;
}
