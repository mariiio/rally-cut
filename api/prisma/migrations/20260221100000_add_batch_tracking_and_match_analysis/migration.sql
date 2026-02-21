-- CreateEnum
CREATE TYPE "BatchTrackingStatus" AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');

-- AlterTable
ALTER TABLE "videos" ADD COLUMN "match_analysis_json" JSONB;

-- CreateTable
CREATE TABLE "batch_tracking_jobs" (
    "id" TEXT NOT NULL,
    "video_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "status" "BatchTrackingStatus" NOT NULL DEFAULT 'PENDING',
    "total_rallies" INTEGER NOT NULL,
    "completed_rallies" INTEGER NOT NULL DEFAULT 0,
    "failed_rallies" INTEGER NOT NULL DEFAULT 0,
    "current_rally_id" TEXT,
    "error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completed_at" TIMESTAMP(3),

    CONSTRAINT "batch_tracking_jobs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "batch_tracking_jobs_video_id_idx" ON "batch_tracking_jobs"("video_id");

-- CreateIndex
CREATE INDEX "batch_tracking_jobs_status_idx" ON "batch_tracking_jobs"("status");

-- AddForeignKey
ALTER TABLE "batch_tracking_jobs" ADD CONSTRAINT "batch_tracking_jobs_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;
