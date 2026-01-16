-- AlterTable
ALTER TABLE "camera_keyframes" ADD COLUMN     "rotation" DOUBLE PRECISION NOT NULL DEFAULT 0.0;

-- CreateTable
CREATE TABLE "video_camera_settings" (
    "id" TEXT NOT NULL,
    "video_id" TEXT NOT NULL,
    "zoom" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    "position_x" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "position_y" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "rotation" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "video_camera_settings_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "video_camera_settings_video_id_key" ON "video_camera_settings"("video_id");

-- AddForeignKey
ALTER TABLE "video_camera_settings" ADD CONSTRAINT "video_camera_settings_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;
