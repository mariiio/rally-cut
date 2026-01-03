import { z } from "zod";

export const feedbackTypeSchema = z.enum(["BUG", "FEATURE", "FEEDBACK"]);

export const createFeedbackSchema = z.object({
  type: feedbackTypeSchema,
  message: z.string().min(1).max(5000),
  email: z.string().email().optional(),
  pageUrl: z.string().url().max(500).optional(),
});

export const feedbackResponseSchema = z.object({
  id: z.string().uuid(),
  type: feedbackTypeSchema,
  message: z.string(),
  createdAt: z.string().datetime(),
});

export type FeedbackType = z.infer<typeof feedbackTypeSchema>;
export type CreateFeedbackInput = z.infer<typeof createFeedbackSchema>;
export type FeedbackResponse = z.infer<typeof feedbackResponseSchema>;
