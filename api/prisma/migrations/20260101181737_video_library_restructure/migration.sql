/*
  Warnings:

  - You are about to drop the column `order` on the `videos` table. All the data in the column will be lost.
  - You are about to drop the column `session_id` on the `videos` table. All the data in the column will be lost.

*/
-- DropForeignKey
ALTER TABLE "videos" DROP CONSTRAINT "videos_session_id_fkey";

-- DropIndex
DROP INDEX "videos_session_id_idx";

-- AlterTable
ALTER TABLE "rally_detection_jobs" ADD COLUMN     "progress" DOUBLE PRECISION DEFAULT 0,
ADD COLUMN     "progress_message" TEXT;

-- AlterTable
ALTER TABLE "sessions" ADD COLUMN     "user_id" TEXT;

-- AlterTable
ALTER TABLE "videos" DROP COLUMN "order",
DROP COLUMN "session_id",
ADD COLUMN     "deleted_at" TIMESTAMP(3),
ADD COLUMN     "user_id" TEXT;

-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "email" TEXT,
    "name" TEXT,
    "avatar_url" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "converted_at" TIMESTAMP(3),

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
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
CREATE TABLE "session_videos" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "video_id" TEXT NOT NULL,
    "order" INTEGER NOT NULL DEFAULT 0,
    "added_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "session_videos_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "anonymous_identities_visitor_id_key" ON "anonymous_identities"("visitor_id");

-- CreateIndex
CREATE INDEX "session_videos_session_id_idx" ON "session_videos"("session_id");

-- CreateIndex
CREATE INDEX "session_videos_video_id_idx" ON "session_videos"("video_id");

-- CreateIndex
CREATE UNIQUE INDEX "session_videos_session_id_video_id_key" ON "session_videos"("session_id", "video_id");

-- CreateIndex
CREATE INDEX "sessions_user_id_idx" ON "sessions"("user_id");

-- CreateIndex
CREATE INDEX "videos_user_id_idx" ON "videos"("user_id");

-- AddForeignKey
ALTER TABLE "anonymous_identities" ADD CONSTRAINT "anonymous_identities_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "videos" ADD CONSTRAINT "videos_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "session_videos" ADD CONSTRAINT "session_videos_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "session_videos" ADD CONSTRAINT "session_videos_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;
