/**
 * Undo a bad auto-rotation on a specific video.
 *
 * Re-encodes the ORIGINAL S3 file (preserved on rotation) without the
 * rotate filter, overwrites the rotated optimized version, and rolls back
 * the qualityReportJson auto-rotate state. The restored `tiltDeg` lets
 * the (now tightened) `shouldAutoRotate` predicate inspect the same
 * measurement and reject it via the new ±8° cap.
 *
 * Usage: cd api && npx tsx scripts/undoVideoRotation.ts <videoId>
 *
 * Created 2026-05-17 to undo the -20° false positive on video
 * 952e1bf8-6d11-4960-a01a-d76fa444b179. Keep as a general-purpose tool.
 */

import { spawn } from "child_process";
import { createWriteStream } from "fs";
import fs from "fs/promises";
import os from "os";
import path from "path";
import { Readable } from "stream";
import { pipeline } from "stream/promises";
import { Prisma, PrismaClient } from "@prisma/client";
import {
  GetObjectCommand,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

import type { QualityReport } from "../src/services/qualityReport.js";

const prisma = new PrismaClient();

const s3 = new S3Client({
  region: process.env.AWS_REGION || "us-east-1",
  ...(process.env.S3_ENDPOINT && {
    endpoint: process.env.S3_ENDPOINT,
    forcePathStyle: true,
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID || "minioadmin",
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "minioadmin",
    },
  }),
});

const BUCKET = process.env.S3_BUCKET_NAME || "rallycut-local";

async function downloadFromS3(key: string, destPath: string): Promise<void> {
  const url = await getSignedUrl(
    s3,
    new GetObjectCommand({ Bucket: BUCKET, Key: key }),
    { expiresIn: 3600 },
  );
  const response = await fetch(url);
  if (!response.ok) throw new Error(`S3 download ${key} failed: ${response.status}`);
  if (!response.body) throw new Error(`S3 download ${key}: no body`);
  const ws = createWriteStream(destPath);
  await pipeline(Readable.fromWeb(response.body as never), ws);
}

async function uploadToS3(key: string, data: Buffer, contentType: string): Promise<void> {
  await s3.send(
    new PutObjectCommand({
      Bucket: BUCKET,
      Key: key,
      Body: data,
      ContentType: contentType,
    }),
  );
}

function runFFmpeg(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn("ffmpeg", args, { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    proc.stderr?.on("data", (d: Buffer) => { stderr += d.toString(); });
    proc.on("error", (err) => reject(new Error(`ffmpeg spawn: ${err.message}`)));
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg exit ${code}: ${stderr.slice(-500)}`));
    });
  });
}

async function main(): Promise<void> {
  const videoId = process.argv[2];
  if (!videoId) {
    console.error("Usage: npx tsx scripts/undoVideoRotation.ts <videoId>");
    process.exit(1);
  }

  const video = await prisma.video.findUnique({ where: { id: videoId } });
  if (!video) {
    console.error(`Video ${videoId} not found`);
    process.exit(1);
  }

  const qr = (video.qualityReportJson as QualityReport | null) ?? null;
  // Soft precondition: this script exists to re-derive optimize/proxy/poster
  // from the preserved original. That requires the video to actually HAVE an
  // original distinct from the current s3Key (i.e. optimization ran at some
  // point). It does NOT require autoRotated to currently be true — re-runs
  // are valid (e.g. to regenerate a stale proxy after a partial cleanup).
  if (!video.originalS3Key || video.originalS3Key === video.s3Key) {
    console.error(
      `Video ${videoId}: no separate originalS3Key. Nothing to undo — the ` +
        `current s3Key already points at the original upload.`,
    );
    process.exit(1);
  }
  const straightened = (qr?.autoFixes ?? []).find((f) => f.id === "auto_straightened");
  const originalTilt = straightened?.data?.originalTiltDeg;
  if (!qr?.autoRotated && originalTilt == null) {
    console.warn(
      `Video ${videoId}: autoRotated is already false and no auto_straightened ` +
        `entry remains. Re-running anyway to re-derive optimize/proxy/poster ` +
        `from the original.`,
    );
  }

  // Both non-null and distinct by the precondition above.
  const originalKey = video.originalS3Key;
  const optimizedKey = video.s3Key; // currently points at the (possibly bad) _optimized.mp4
  const proxyKey = video.proxyS3Key; // 720p variant, read first by the web editor
  const posterKey = video.posterS3Key; // 1280px JPEG thumb

  console.log(`[undo-rotation] video=${videoId}`);
  console.log(`[undo-rotation]   original  = ${originalKey}`);
  console.log(`[undo-rotation]   optimized = ${optimizedKey}`);
  console.log(`[undo-rotation]   proxy     = ${proxyKey ?? "(none)"}`);
  console.log(`[undo-rotation]   poster    = ${posterKey ?? "(none)"}`);
  console.log(`[undo-rotation]   was rotated by ${originalTilt ?? "?"}°`);

  const tmpDir = path.join(os.tmpdir(), `undo-rotation-${videoId}`);
  await fs.mkdir(tmpDir, { recursive: true });
  const inputPath = path.join(tmpDir, "input.mp4");
  const outputPath = path.join(tmpDir, "output.mp4");
  const proxyPath = path.join(tmpDir, "proxy.mp4");
  const posterPath = path.join(tmpDir, "poster.jpg");

  try {
    console.log("[undo-rotation] downloading original from S3...");
    await downloadFromS3(originalKey, inputPath);

    console.log("[undo-rotation] re-encoding optimized (no rotate)...");
    await runFFmpeg([
      "-i", inputPath,
      "-c:v", "libx264",
      "-pix_fmt", "yuv420p",
      "-crf", "23",
      "-preset", "fast",
      "-tune", "film",
      "-profile:v", "high",
      "-level", "4.1",
      "-movflags", "+faststart",
      "-c:a", "aac",
      "-b:a", "128k",
      "-ac", "2",
      "-y", outputPath,
    ]);
    console.log(`[undo-rotation] uploading optimized → ${optimizedKey}`);
    const optimizedData = await fs.readFile(outputPath);
    await uploadToS3(optimizedKey, optimizedData, "video/mp4");

    if (proxyKey) {
      console.log("[undo-rotation] re-encoding 720p proxy (no rotate)...");
      await runFFmpeg([
        "-i", outputPath,
        "-vf", "scale=-2:720",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "28",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "96k",
        "-y", proxyPath,
      ]);
      console.log(`[undo-rotation] uploading proxy → ${proxyKey}`);
      const proxyData = await fs.readFile(proxyPath);
      await uploadToS3(proxyKey, proxyData, "video/mp4");
    }

    if (posterKey) {
      console.log("[undo-rotation] regenerating poster (no rotate)...");
      await runFFmpeg([
        "-ss", "1",
        "-i", inputPath,
        "-vframes", "1",
        "-vf", "scale=1280:-1",
        "-q:v", "2",
        "-y", posterPath,
      ]);
      console.log(`[undo-rotation] uploading poster → ${posterKey}`);
      const posterData = await fs.readFile(posterPath);
      await uploadToS3(posterKey, posterData, "image/jpeg");
    }

    console.log("[undo-rotation] updating DB (clearing autoRotated + auto_straightened)...");
    await prisma.$transaction(async (tx) => {
      const row = await tx.video.findUnique({ where: { id: videoId } });
      const prior = (row?.qualityReportJson as QualityReport | null) ?? { version: 2, issues: [] };
      const updated: QualityReport = {
        ...prior,
        autoRotated: false,
        // Restore the original measurement so the (now tightened) predicate
        // can re-evaluate and reject it via the 8° cap.
        tiltDeg: Number.isFinite(originalTilt) ? (originalTilt as number) : null,
        autoFixes: (prior.autoFixes ?? []).filter((f) => f.id !== "auto_straightened"),
      };
      await tx.video.update({
        where: { id: videoId },
        data: {
          qualityReportJson: updated as unknown as Prisma.InputJsonValue,
          fileSizeBytes: BigInt(optimizedData.length),
        },
      });
    });
    console.log("[undo-rotation] done.");
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
}

main()
  .catch((err) => {
    console.error(err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
