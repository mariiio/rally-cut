-- CreateEnum
CREATE TYPE "VideoStatus" AS ENUM ('PENDING', 'UPLOADED', 'DETECTING', 'DETECTED', 'ERROR');

-- CreateEnum
CREATE TYPE "ServingTeam" AS ENUM ('A', 'B');

-- CreateEnum
CREATE TYPE "DetectionJobStatus" AS ENUM ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED');

-- CreateTable
CREATE TABLE "sessions" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "sessions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "videos" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "s3_key" TEXT NOT NULL,
    "content_hash" TEXT NOT NULL,
    "status" "VideoStatus" NOT NULL DEFAULT 'PENDING',
    "duration_ms" INTEGER,
    "width" INTEGER,
    "height" INTEGER,
    "file_size_bytes" BIGINT,
    "order" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "videos_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "rallies" (
    "id" TEXT NOT NULL,
    "video_id" TEXT NOT NULL,
    "start_ms" INTEGER NOT NULL,
    "end_ms" INTEGER NOT NULL,
    "confidence" DOUBLE PRECISION,
    "score_a" INTEGER,
    "score_b" INTEGER,
    "serving_team" "ServingTeam",
    "notes" TEXT,
    "order" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "rallies_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "highlights" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "color" TEXT NOT NULL DEFAULT '#FF6B6B',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "highlights_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "highlight_rallies" (
    "id" TEXT NOT NULL,
    "highlight_id" TEXT NOT NULL,
    "rally_id" TEXT NOT NULL,
    "order" INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT "highlight_rallies_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "rally_detection_jobs" (
    "id" TEXT NOT NULL,
    "video_id" TEXT NOT NULL,
    "content_hash" TEXT NOT NULL,
    "status" "DetectionJobStatus" NOT NULL DEFAULT 'PENDING',
    "result_s3_key" TEXT,
    "error_message" TEXT,
    "started_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "rally_detection_jobs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "videos_content_hash_idx" ON "videos"("content_hash");

-- CreateIndex
CREATE INDEX "videos_session_id_idx" ON "videos"("session_id");

-- CreateIndex
CREATE INDEX "rallies_video_id_idx" ON "rallies"("video_id");

-- CreateIndex
CREATE INDEX "highlights_session_id_idx" ON "highlights"("session_id");

-- CreateIndex
CREATE UNIQUE INDEX "highlight_rallies_highlight_id_rally_id_key" ON "highlight_rallies"("highlight_id", "rally_id");

-- CreateIndex
CREATE INDEX "rally_detection_jobs_content_hash_status_idx" ON "rally_detection_jobs"("content_hash", "status");

-- AddForeignKey
ALTER TABLE "videos" ADD CONSTRAINT "videos_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "rallies" ADD CONSTRAINT "rallies_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "highlights" ADD CONSTRAINT "highlights_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "highlight_rallies" ADD CONSTRAINT "highlight_rallies_highlight_id_fkey" FOREIGN KEY ("highlight_id") REFERENCES "highlights"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "highlight_rallies" ADD CONSTRAINT "highlight_rallies_rally_id_fkey" FOREIGN KEY ("rally_id") REFERENCES "rallies"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "rally_detection_jobs" ADD CONSTRAINT "rally_detection_jobs_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;
