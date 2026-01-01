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
  CLOUDFRONT_DOMAIN: z.string(),
  CLOUDFRONT_KEY_PAIR_ID: z.string(),
  CLOUDFRONT_PRIVATE_KEY: z.string(),

  MODAL_WEBHOOK_SECRET: z.string(),
  MODAL_FUNCTION_URL: z.string().url(),

  CORS_ORIGIN: z.string().default("http://localhost:3000"),
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
