-- CreateEnum
CREATE TYPE "AspectRatio" AS ENUM ('ORIGINAL', 'VERTICAL');

-- CreateEnum
CREATE TYPE "CameraEasing" AS ENUM ('LINEAR', 'EASE_IN', 'EASE_OUT', 'EASE_IN_OUT');

-- CreateTable
CREATE TABLE "rally_camera_edits" (
    "id" TEXT NOT NULL,
    "rally_id" TEXT NOT NULL,
    "aspect_ratio" "AspectRatio" NOT NULL DEFAULT 'ORIGINAL',
    "enabled" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "rally_camera_edits_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "camera_keyframes" (
    "id" TEXT NOT NULL,
    "rally_camera_id" TEXT NOT NULL,
    "time_offset" DOUBLE PRECISION NOT NULL,
    "position_x" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "position_y" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "zoom" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    "easing" "CameraEasing" NOT NULL DEFAULT 'EASE_IN_OUT',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "camera_keyframes_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "rally_camera_edits_rally_id_key" ON "rally_camera_edits"("rally_id");

-- CreateIndex
CREATE INDEX "camera_keyframes_rally_camera_id_idx" ON "camera_keyframes"("rally_camera_id");

-- AddForeignKey
ALTER TABLE "rally_camera_edits" ADD CONSTRAINT "rally_camera_edits_rally_id_fkey" FOREIGN KEY ("rally_id") REFERENCES "rallies"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "camera_keyframes" ADD CONSTRAINT "camera_keyframes_rally_camera_id_fkey" FOREIGN KEY ("rally_camera_id") REFERENCES "rally_camera_edits"("id") ON DELETE CASCADE ON UPDATE CASCADE;
