import {
  AbortMultipartUploadCommand,
  CompleteMultipartUploadCommand,
  CreateMultipartUploadCommand,
  DeleteObjectCommand,
  GetObjectCommand,
  PutObjectCommand,
  S3Client,
  type S3ClientConfig,
  UploadPartCommand,
} from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "../config/env.js";

// Configure S3 client with optional endpoint for MinIO/local development
const s3Config: S3ClientConfig = {
  region: env.AWS_REGION,
};

// If S3_ENDPOINT is set, use it (for MinIO/local development)
if (env.S3_ENDPOINT) {
  s3Config.endpoint = env.S3_ENDPOINT;
  s3Config.forcePathStyle = true; // Required for MinIO (path-style URLs)
  s3Config.credentials = {
    accessKeyId: env.AWS_ACCESS_KEY_ID,
    secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
  };
}

const s3Client = new S3Client(s3Config);

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
    // Short cache for originals (1 day) - will be replaced by optimized version
    CacheControl: "public, max-age=86400",
  });

  return getSignedUrl(s3Client, command, { expiresIn: 3600 });
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

/**
 * Generate S3 key for a video file.
 * Pattern: videos/{userId}/{videoId}/{filename}
 */
export function getVideoS3Key(
  userId: string,
  videoId: string,
  filename: string
): string {
  return `videos/${userId}/${videoId}/${filename}`;
}

export function getAnalysisS3Key(videoId: string): string {
  return `analysis/${videoId}/results.json`;
}

/**
 * Get object from S3 and return the stream/metadata.
 * Used for proxying video content in local development.
 * Supports range requests for video seeking.
 */
export async function getObject(key: string, range?: string) {
  const command = new GetObjectCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
    Range: range,
  });

  return s3Client.send(command);
}

/**
 * Upload a processed video to S3 with optimized caching headers.
 * Used by video processing service after optimization.
 */
export async function uploadProcessedVideo(
  key: string,
  data: Buffer
): Promise<void> {
  const upload = new Upload({
    client: s3Client,
    params: {
      Bucket: env.S3_BUCKET_NAME,
      Key: key,
      Body: data,
      ContentType: "video/mp4",
      // Long cache for processed videos (1 year)
      CacheControl: "public, max-age=31536000",
    },
  });

  await upload.done();
}

/**
 * Upload a poster image to S3.
 */
export async function uploadPoster(key: string, data: Buffer): Promise<void> {
  const upload = new Upload({
    client: s3Client,
    params: {
      Bucket: env.S3_BUCKET_NAME,
      Key: key,
      Body: data,
      ContentType: "image/jpeg",
      CacheControl: "public, max-age=31536000", // 1 year
    },
  });

  await upload.done();
}

// ============================================================================
// Multipart Upload (for large files)
// ============================================================================

/**
 * Initiate a multipart upload and return the upload ID.
 */
export async function initiateMultipartUpload(
  key: string,
  contentType: string
): Promise<string> {
  const command = new CreateMultipartUploadCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
    ContentType: contentType,
    CacheControl: "public, max-age=86400",
  });

  const response = await s3Client.send(command);
  return response.UploadId!;
}

/**
 * Generate a presigned URL for uploading a single part.
 */
export async function generatePartUploadUrl(
  key: string,
  uploadId: string,
  partNumber: number
): Promise<string> {
  const command = new UploadPartCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
    UploadId: uploadId,
    PartNumber: partNumber,
  });

  return getSignedUrl(s3Client, command, { expiresIn: 3600 });
}

/**
 * Complete a multipart upload by combining all parts.
 */
export async function completeMultipartUpload(
  key: string,
  uploadId: string,
  parts: { PartNumber: number; ETag: string }[]
): Promise<void> {
  const command = new CompleteMultipartUploadCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
    UploadId: uploadId,
    MultipartUpload: { Parts: parts },
  });

  await s3Client.send(command);
}

/**
 * Abort a multipart upload and cleanup uploaded parts.
 */
export async function abortMultipartUpload(
  key: string,
  uploadId: string
): Promise<void> {
  const command = new AbortMultipartUploadCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
    UploadId: uploadId,
  });

  await s3Client.send(command);
}
