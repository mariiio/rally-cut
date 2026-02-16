/**
 * Backfill video characteristics for existing videos.
 *
 * Phase 1: Brightness from video frames (downloads from S3)
 * Phase 2: Camera distance + scene complexity from existing tracking data
 *
 * Usage: cd api && npx tsx scripts/backfill-characteristics.ts
 */

import { spawn } from "child_process";
import { createWriteStream } from "fs";
import fs from "fs/promises";
import os from "os";
import path from "path";
import { Readable } from "stream";
import { pipeline } from "stream/promises";
import { PrismaClient } from "@prisma/client";
import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";

const prisma = new PrismaClient();

// S3 config (matches api env)
const s3 = new S3Client({
  region: process.env.AWS_REGION || "us-east-1",
  ...(process.env.S3_ENDPOINT && {
    endpoint: process.env.S3_ENDPOINT,
    forcePathStyle: true,
  }),
  ...(process.env.S3_ENDPOINT && {
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID || "minioadmin",
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "minioadmin",
    },
  }),
});

const BUCKET = process.env.S3_BUCKET_NAME || "rallycut-local";

function runFFmpegToBuffer(args: string[]): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const proc = spawn("ffmpeg", args, { stdio: ["ignore", "pipe", "pipe"] });
    const chunks: Buffer[] = [];
    proc.stdout?.on("data", (data: Buffer) => chunks.push(data));
    let stderr = "";
    proc.stderr?.on("data", (data: Buffer) => { stderr += data.toString(); });
    proc.on("error", (err) => reject(new Error(`FFmpeg: ${err.message}`)));
    proc.on("close", (code) => {
      if (code === 0) resolve(Buffer.concat(chunks));
      else reject(new Error(`FFmpeg exit ${code}: ${stderr.slice(-300)}`));
    });
  });
}

async function computeBrightness(inputPath: string): Promise<{ mean: number; category: "dark" | "normal" | "bright" }> {
  const buf = await runFFmpegToBuffer([
    "-i", inputPath,
    "-vf", "select='not(mod(n\\,100))',scale=320:240",
    "-frames:v", "5",
    "-f", "rawvideo",
    "-pix_fmt", "gray",
    "pipe:1",
  ]);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i];
  const mean = buf.length > 0 ? sum / buf.length : 128;
  const rounded = Math.round(mean * 10) / 10;
  const category = mean < 90 ? "dark" : mean > 180 ? "bright" : "normal";
  return { mean: rounded, category };
}

async function downloadFromS3(s3Key: string, destPath: string): Promise<void> {
  const cmd = new GetObjectCommand({ Bucket: BUCKET, Key: s3Key });
  const resp = await s3.send(cmd);
  if (!resp.Body) throw new Error("Empty S3 response");
  const readable = resp.Body as unknown as Readable;
  await pipeline(readable, createWriteStream(destPath));
}

async function main() {
  const videos = await prisma.video.findMany({
    where: { deletedAt: null },
    select: {
      id: true,
      name: true,
      s3Key: true,
      characteristicsJson: true,
      rallies: {
        select: { id: true },
        take: 1,
      },
    },
  });

  console.log(`Found ${videos.length} videos to process`);

  for (const video of videos) {
    const existing = (video.characteristicsJson as Record<string, unknown>) ?? {};
    const needsBrightness = !existing.brightness;
    const needsTracking = !existing.cameraDistance;

    if (!needsBrightness && !needsTracking) {
      console.log(`  [SKIP] ${video.name} — already complete`);
      continue;
    }

    const updates: Record<string, unknown> = { ...existing, version: 1 };

    // Phase 1: Brightness
    if (needsBrightness && video.s3Key) {
      const tmpDir = path.join(os.tmpdir(), `backfill-${video.id}`);
      await fs.mkdir(tmpDir, { recursive: true });
      const inputPath = path.join(tmpDir, "input.mp4");
      try {
        await downloadFromS3(video.s3Key, inputPath);
        const brightness = await computeBrightness(inputPath);
        updates.brightness = brightness;
        console.log(`  [BRIGHTNESS] ${video.name}: ${brightness.mean} (${brightness.category})`);
      } catch (err) {
        console.log(`  [BRIGHTNESS FAIL] ${video.name}: ${err}`);
      } finally {
        await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
      }
    }

    // Phase 2: Camera distance + scene complexity from tracking data
    if (needsTracking) {
      const trackData = await prisma.playerTrack.findFirst({
        where: {
          rally: { videoId: video.id },
          status: "COMPLETED",
          positionsJson: { not: undefined },
        },
        select: {
          positionsJson: true,
          rawPositionsJson: true,
          primaryTrackIds: true,
        },
        orderBy: { completedAt: "desc" },
      });

      if (trackData && trackData.positionsJson) {
        // Camera distance
        const primaryIds = new Set((trackData.primaryTrackIds as number[]) ?? []);
        const positions = trackData.positionsJson as Array<{ trackId: number; height: number }>;
        const heights = positions
          .filter(p => primaryIds.has(p.trackId))
          .map(p => p.height);

        if (heights.length > 0) {
          const sorted = [...heights].sort((a, b) => a - b);
          const avgBboxHeight = sorted[Math.floor(sorted.length / 2)];
          const category = avgBboxHeight > 0.35 ? "close" : avgBboxHeight < 0.20 ? "far" : "medium";
          updates.cameraDistance = {
            avgBboxHeight: Math.round(avgBboxHeight * 1000) / 1000,
            category,
          };
          console.log(`  [CAMERA] ${video.name}: ${(avgBboxHeight * 100).toFixed(1)}% (${category})`);
        }

        // Scene complexity from raw (unfiltered) detections
        const rawPositions = (trackData.rawPositionsJson ?? trackData.positionsJson) as Array<{ frameNumber: number }>;
        const rawFrameCounts: Record<number, number> = {};
        for (const p of rawPositions) {
          rawFrameCounts[p.frameNumber] = (rawFrameCounts[p.frameNumber] ?? 0) + 1;
        }
        const rawFrameValues = Object.values(rawFrameCounts);
        const avgPeople = rawFrameValues.length > 0
          ? rawFrameValues.reduce((a, b) => a + b, 0) / rawFrameValues.length
          : 0;
        updates.sceneComplexity = {
          avgPeople: Math.round(avgPeople * 10) / 10,
          category: avgPeople > 6 ? "complex" : "simple",
        };
        console.log(`  [SCENE] ${video.name}: ${avgPeople.toFixed(1)} raw people/frame (${avgPeople > 6 ? "complex" : "simple"})`);
      } else {
        console.log(`  [TRACKING SKIP] ${video.name}: no tracking data`);
      }
    }

    // Update DB
    await prisma.video.update({
      where: { id: video.id },
      data: { characteristicsJson: updates },
    });
  }

  console.log("\nDone!");
  await prisma.$disconnect();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
