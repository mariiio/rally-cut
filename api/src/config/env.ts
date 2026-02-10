import "dotenv/config";
import { z } from "zod";

const envSchema = z.object({
  NODE_ENV: z
    .enum(["development", "production", "test"])
    .default("development"),
  PORT: z.coerce.number().default(3001),

  DATABASE_URL: z.string().url(),

  AWS_REGION: z.string().default("us-east-1"),
  AWS_ACCESS_KEY_ID: z.string().default(""),
  AWS_SECRET_ACCESS_KEY: z.string().default(""),

  S3_BUCKET_NAME: z.string(),
  // Optional S3 endpoint for MinIO/local development
  S3_ENDPOINT: z.string().url().optional(),

  // CloudFront (optional for local development)
  CLOUDFRONT_DOMAIN: z.string().default(""),
  CLOUDFRONT_KEY_PAIR_ID: z.string().default(""),
  CLOUDFRONT_PRIVATE_KEY: z.string().default(""),

  MODAL_WEBHOOK_SECRET: z.string(),
  // Optional for local development (uses local Python subprocess)
  MODAL_FUNCTION_URL: z.string().url().optional(),

  CORS_ORIGIN: z.string().default("http://localhost:3000"),

  // Export Lambda (optional - export won't work without it)
  EXPORT_LAMBDA_FUNCTION_NAME: z.string().optional(),
  API_BASE_URL: z.string().default("http://localhost:3001"),

  // Video Processing Lambda (optional - local FFmpeg used if not set)
  PROCESSING_LAMBDA_FUNCTION_NAME: z.string().optional(),

  // Auth JWT (must match AUTH_SECRET in web)
  AUTH_JWT_SECRET: z.string().min(16).optional(),

  // Email (AWS SES)
  SES_FROM_EMAIL: z.string().email().default("noreply@rallycut.com"),
  SES_REGION: z.string().default("us-east-1"),
  FRONTEND_URL: z.string().default("http://localhost:3000"),

  // Label Studio (optional - for ground truth labeling)
  LABEL_STUDIO_URL: z.string().url().optional(),
  LABEL_STUDIO_API_KEY: z.string().optional(),
});

function loadEnv(): z.infer<typeof envSchema> {
  const result = envSchema.safeParse(process.env);

  if (!result.success) {
    console.error("Environment validation failed:");
    console.error(result.error.format());
    throw new Error("Invalid environment variables");
  }

  return result.data;
}

export const env = loadEnv();
export type Env = z.infer<typeof envSchema>;
