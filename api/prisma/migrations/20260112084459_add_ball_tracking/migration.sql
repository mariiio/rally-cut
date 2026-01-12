-- CreateEnum
CREATE TYPE "BallTrackStatus" AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');

-- CreateTable
CREATE TABLE "ball_tracks" (
    "id" TEXT NOT NULL,
    "rally_id" TEXT NOT NULL,
    "status" "BallTrackStatus" NOT NULL DEFAULT 'PENDING',
    "frame_count" INTEGER,
    "detection_rate" DOUBLE PRECISION,
    "positions_json" JSONB,
    "processing_time_ms" INTEGER,
    "model_version" TEXT,
    "error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completed_at" TIMESTAMP(3),

    CONSTRAINT "ball_tracks_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "ball_tracks_rally_id_key" ON "ball_tracks"("rally_id");

-- CreateIndex
CREATE INDEX "ball_tracks_rally_id_idx" ON "ball_tracks"("rally_id");

-- CreateIndex
CREATE INDEX "ball_tracks_status_idx" ON "ball_tracks"("status");

-- AddForeignKey
ALTER TABLE "ball_tracks" ADD CONSTRAINT "ball_tracks_rally_id_fkey" FOREIGN KEY ("rally_id") REFERENCES "rallies"("id") ON DELETE CASCADE ON UPDATE CASCADE;
