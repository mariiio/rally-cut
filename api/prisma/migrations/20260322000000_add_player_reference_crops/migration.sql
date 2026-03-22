-- CreateTable
CREATE TABLE "player_reference_crops" (
    "id" TEXT NOT NULL,
    "video_id" TEXT NOT NULL,
    "player_id" INTEGER NOT NULL,
    "s3_key" TEXT NOT NULL,
    "frame_ms" INTEGER NOT NULL,
    "bbox_x" DOUBLE PRECISION NOT NULL,
    "bbox_y" DOUBLE PRECISION NOT NULL,
    "bbox_w" DOUBLE PRECISION NOT NULL,
    "bbox_h" DOUBLE PRECISION NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "player_reference_crops_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "player_reference_crops_video_id_idx" ON "player_reference_crops"("video_id");

-- CreateIndex
CREATE INDEX "player_reference_crops_video_id_player_id_idx" ON "player_reference_crops"("video_id", "player_id");

-- AddForeignKey
ALTER TABLE "player_reference_crops" ADD CONSTRAINT "player_reference_crops_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;
