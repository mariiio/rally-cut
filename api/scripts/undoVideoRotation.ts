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

async function uploadToS3(key: string, data: Buffer): Promise<void> {
  await s3.send(
    new PutObjectCommand({
      Bucket: BUCKET,
      Key: key,
      Body: data,
      ContentType: "video/mp4",
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
  if (!qr?.autoRotated) {
    console.error(`Video ${videoId} has autoRotated !== true; nothing to undo`);
    process.exit(1);
  }
  const straightened = (qr.autoFixes ?? []).find((f) => f.id === "auto_straightened");
  const originalTilt = straightened?.data?.originalTiltDeg;
  if (!Number.isFinite(originalTilt)) {
    console.warn(
      `Video ${videoId}: no auto_straightened entry with originalTiltDeg; ` +
        `tiltDeg will be restored to null.`,
    );
  }

  const originalKey = video.originalS3Key ?? video.s3Key;
  const optimizedKey = video.s3Key; // currently points at the rotated _optimized.mp4
  if (!originalKey || !optimizedKey) {
    console.error(`Video ${videoId} missing s3Key/originalS3Key`);
    process.exit(1);
  }

  console.log(`[undo-rotation] video=${videoId}`);
  console.log(`[undo-rotation]   original  = ${originalKey}`);
  console.log(`[undo-rotation]   optimized = ${optimizedKey}`);
  console.log(`[undo-rotation]   was rotated by ${originalTilt ?? "?"}°`);

  const tmpDir = path.join(os.tmpdir(), `undo-rotation-${videoId}`);
  await fs.mkdir(tmpDir, { recursive: true });
  const inputPath = path.join(tmpDir, "input.mp4");
  const outputPath = path.join(tmpDir, "output.mp4");

  try {
    console.log("[undo-rotation] downloading original from S3...");
    await downloadFromS3(originalKey, inputPath);

    console.log("[undo-rotation] re-encoding without rotation...");
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

    console.log(`[undo-rotation] uploading clean optimized to ${optimizedKey}...`);
    const data = await fs.readFile(outputPath);
    await uploadToS3(optimizedKey, data);

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
          fileSizeBytes: BigInt(data.length),
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
