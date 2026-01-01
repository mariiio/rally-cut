import {
  DeleteObjectCommand,
  GetObjectCommand,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "../config/env.js";

const s3Client = new S3Client({
  region: env.AWS_REGION,
});

export interface GenerateUploadUrlParams {
  key: string;
  contentType: string;
  contentLength: number;
}

export async function generateUploadUrl(
  params: GenerateUploadUrlParams
): Promise<string> {
  const command = new PutObjectCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: params.key,
    ContentType: params.contentType,
    ContentLength: params.contentLength,
  });

  return getSignedUrl(s3Client, command, { expiresIn: 300 });
}

export async function generateDownloadUrl(key: string): Promise<string> {
  const command = new GetObjectCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
  });

  return getSignedUrl(s3Client, command, { expiresIn: 3600 });
}

export async function deleteObject(key: string): Promise<void> {
  const command = new DeleteObjectCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
  });

  await s3Client.send(command);
}

export function getVideoS3Key(
  sessionId: string,
  videoId: string,
  filename: string
): string {
  return `videos/${sessionId}/${videoId}/${filename}`;
}

export function getAnalysisS3Key(videoId: string): string {
  return `analysis/${videoId}/results.json`;
}
