-- CreateEnum
CREATE TYPE "UserTier" AS ENUM ('FREE', 'PREMIUM');

-- CreateEnum
CREATE TYPE "SessionType" AS ENUM ('REGULAR', 'ALL_VIDEOS');

-- CreateEnum
CREATE TYPE "VideoStatus" AS ENUM ('PENDING', 'UPLOADED', 'DETECTING', 'DETECTED', 'ERROR');

-- CreateEnum
CREATE TYPE "ProcessingStatus" AS ENUM ('PENDING', 'QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED', 'SKIPPED');

-- CreateEnum
CREATE TYPE "ServingTeam" AS ENUM ('A', 'B');

-- CreateEnum
CREATE TYPE "DetectionJobStatus" AS ENUM ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED');

-- CreateEnum
CREATE TYPE "ExportStatus" AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');

-- CreateEnum
CREATE TYPE "ConfirmationStatus" AS ENUM ('PENDING', 'PROCESSING', 'CONFIRMED', 'FAILED');

-- CreateEnum
CREATE TYPE "FeedbackType" AS ENUM ('BUG', 'FEATURE', 'FEEDBACK');

-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "email" TEXT,
    "name" TEXT,
    "avatar_url" TEXT,
    "tier" "UserTier" NOT NULL DEFAULT 'FREE',
    "tier_expires_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "converted_at" TIMESTAMP(3),

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "user_usage_quotas" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "period_start" TIMESTAMP(3) NOT NULL,
    "detections_used" INTEGER NOT NULL DEFAULT 0,
    "uploads_this_month" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "user_usage_quotas_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "anonymous_identities" (
    "id" TEXT NOT NULL,
    "visitor_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "user_agent" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "anonymous_identities_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "session_shares" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "session_shares_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "session_members" (
    "id" TEXT NOT NULL,
    "session_share_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "joined_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "session_members_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "sessions" (
    "id" TEXT NOT NULL,
    "user_id" TEXT,
    "name" TEXT NOT NULL,
    "type" "SessionType" NOT NULL DEFAULT 'REGULAR',
    "expires_at" TIMESTAMP(3),
    "deleted_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "sessions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "videos" (
    "id" TEXT NOT NULL,
    "user_id" TEXT,
    "name" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "s3_key" TEXT NOT NULL,
    "content_hash" TEXT NOT NULL,
    "status" "VideoStatus" NOT NULL DEFAULT 'PENDING',
    "duration_ms" INTEGER,
    "width" INTEGER,
    "height" INTEGER,
    "file_size_bytes" BIGINT,
    "expires_at" TIMESTAMP(3),
    "deleted_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "processing_status" "ProcessingStatus" NOT NULL DEFAULT 'PENDING',
    "processed_s3_key" TEXT,
    "processed_at" TIMESTAMP(3),
    "processing_error" TEXT,
    "original_file_size_bytes" BIGINT,

    CONSTRAINT "videos_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "session_videos" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "video_id" TEXT NOT NULL,
    "order" INTEGER NOT NULL DEFAULT 0,
    "added_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "session_videos_pkey" PRIMARY KEY ("id")
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
    "created_by_user_id" TEXT,
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
    "progress" DOUBLE PRECISION DEFAULT 0,
    "progress_message" TEXT,
    "result_s3_key" TEXT,
    "error_message" TEXT,
    "started_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "rally_detection_jobs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "export_jobs" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "tier" "UserTier" NOT NULL DEFAULT 'FREE',
    "status" "ExportStatus" NOT NULL DEFAULT 'PENDING',
    "progress" INTEGER NOT NULL DEFAULT 0,
    "output_key" TEXT,
    "error" TEXT,
    "config" JSONB NOT NULL,
    "rallies" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "export_jobs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "rally_confirmations" (
    "id" TEXT NOT NULL,
    "video_id" TEXT NOT NULL,
    "status" "ConfirmationStatus" NOT NULL DEFAULT 'PENDING',
    "original_s3_key" TEXT NOT NULL,
    "original_duration_ms" INTEGER NOT NULL,
    "trimmed_s3_key" TEXT,
    "trimmed_duration_ms" INTEGER,
    "timestamp_mappings" JSONB NOT NULL,
    "progress" INTEGER NOT NULL DEFAULT 0,
    "error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "confirmed_at" TIMESTAMP(3),

    CONSTRAINT "rally_confirmations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "feedback" (
    "id" TEXT NOT NULL,
    "user_id" TEXT,
    "type" "FeedbackType" NOT NULL,
    "message" TEXT NOT NULL,
    "email" TEXT,
    "user_agent" TEXT,
    "page_url" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "feedback_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE INDEX "users_tier_idx" ON "users"("tier");

-- CreateIndex
CREATE UNIQUE INDEX "user_usage_quotas_user_id_key" ON "user_usage_quotas"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "anonymous_identities_visitor_id_key" ON "anonymous_identities"("visitor_id");

-- CreateIndex
CREATE INDEX "anonymous_identities_user_id_idx" ON "anonymous_identities"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "session_shares_session_id_key" ON "session_shares"("session_id");

-- CreateIndex
CREATE UNIQUE INDEX "session_shares_token_key" ON "session_shares"("token");

-- CreateIndex
CREATE INDEX "session_members_user_id_idx" ON "session_members"("user_id");

-- CreateIndex
CREATE INDEX "session_members_session_share_id_idx" ON "session_members"("session_share_id");

-- CreateIndex
CREATE UNIQUE INDEX "session_members_session_share_id_user_id_key" ON "session_members"("session_share_id", "user_id");

-- CreateIndex
CREATE INDEX "sessions_user_id_idx" ON "sessions"("user_id");

-- CreateIndex
CREATE INDEX "sessions_user_id_deleted_at_idx" ON "sessions"("user_id", "deleted_at");

-- CreateIndex
CREATE INDEX "sessions_expires_at_idx" ON "sessions"("expires_at");

-- CreateIndex
CREATE INDEX "sessions_deleted_at_idx" ON "sessions"("deleted_at");

-- CreateIndex
CREATE INDEX "videos_user_id_idx" ON "videos"("user_id");

-- CreateIndex
CREATE INDEX "videos_user_id_deleted_at_idx" ON "videos"("user_id", "deleted_at");

-- CreateIndex
CREATE INDEX "videos_content_hash_idx" ON "videos"("content_hash");

-- CreateIndex
CREATE INDEX "videos_status_idx" ON "videos"("status");

-- CreateIndex
CREATE INDEX "videos_expires_at_idx" ON "videos"("expires_at");

-- CreateIndex
CREATE INDEX "videos_deleted_at_idx" ON "videos"("deleted_at");

-- CreateIndex
CREATE INDEX "videos_processing_status_idx" ON "videos"("processing_status");

-- CreateIndex
CREATE INDEX "session_videos_session_id_idx" ON "session_videos"("session_id");

-- CreateIndex
CREATE INDEX "session_videos_video_id_idx" ON "session_videos"("video_id");

-- CreateIndex
CREATE UNIQUE INDEX "session_videos_session_id_video_id_key" ON "session_videos"("session_id", "video_id");

-- CreateIndex
CREATE INDEX "rallies_video_id_idx" ON "rallies"("video_id");

-- CreateIndex
CREATE INDEX "rallies_video_id_start_ms_idx" ON "rallies"("video_id", "start_ms");

-- CreateIndex
CREATE INDEX "highlights_session_id_idx" ON "highlights"("session_id");

-- CreateIndex
CREATE INDEX "highlights_created_by_user_id_idx" ON "highlights"("created_by_user_id");

-- CreateIndex
CREATE INDEX "highlight_rallies_rally_id_idx" ON "highlight_rallies"("rally_id");

-- CreateIndex
CREATE UNIQUE INDEX "highlight_rallies_highlight_id_rally_id_key" ON "highlight_rallies"("highlight_id", "rally_id");

-- CreateIndex
CREATE INDEX "rally_detection_jobs_video_id_idx" ON "rally_detection_jobs"("video_id");

-- CreateIndex
CREATE INDEX "rally_detection_jobs_status_idx" ON "rally_detection_jobs"("status");

-- CreateIndex
CREATE INDEX "rally_detection_jobs_content_hash_status_idx" ON "rally_detection_jobs"("content_hash", "status");

-- CreateIndex
CREATE INDEX "export_jobs_session_id_idx" ON "export_jobs"("session_id");

-- CreateIndex
CREATE INDEX "export_jobs_user_id_idx" ON "export_jobs"("user_id");

-- CreateIndex
CREATE INDEX "export_jobs_status_idx" ON "export_jobs"("status");

-- CreateIndex
CREATE UNIQUE INDEX "rally_confirmations_video_id_key" ON "rally_confirmations"("video_id");

-- CreateIndex
CREATE INDEX "rally_confirmations_video_id_idx" ON "rally_confirmations"("video_id");

-- CreateIndex
CREATE INDEX "rally_confirmations_status_idx" ON "rally_confirmations"("status");

-- CreateIndex
CREATE INDEX "feedback_user_id_idx" ON "feedback"("user_id");

-- CreateIndex
CREATE INDEX "feedback_type_idx" ON "feedback"("type");

-- AddForeignKey
ALTER TABLE "user_usage_quotas" ADD CONSTRAINT "user_usage_quotas_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "anonymous_identities" ADD CONSTRAINT "anonymous_identities_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "session_shares" ADD CONSTRAINT "session_shares_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "session_members" ADD CONSTRAINT "session_members_session_share_id_fkey" FOREIGN KEY ("session_share_id") REFERENCES "session_shares"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "session_members" ADD CONSTRAINT "session_members_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "videos" ADD CONSTRAINT "videos_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "session_videos" ADD CONSTRAINT "session_videos_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "session_videos" ADD CONSTRAINT "session_videos_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "rallies" ADD CONSTRAINT "rallies_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "highlights" ADD CONSTRAINT "highlights_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "highlights" ADD CONSTRAINT "highlights_created_by_user_id_fkey" FOREIGN KEY ("created_by_user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "highlight_rallies" ADD CONSTRAINT "highlight_rallies_highlight_id_fkey" FOREIGN KEY ("highlight_id") REFERENCES "highlights"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "highlight_rallies" ADD CONSTRAINT "highlight_rallies_rally_id_fkey" FOREIGN KEY ("rally_id") REFERENCES "rallies"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "rally_detection_jobs" ADD CONSTRAINT "rally_detection_jobs_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "export_jobs" ADD CONSTRAINT "export_jobs_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "rally_confirmations" ADD CONSTRAINT "rally_confirmations_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "feedback" ADD CONSTRAINT "feedback_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
