-- CreateEnum
CREATE TYPE "ActionLabel" AS ENUM ('SERVE', 'RECEIVE', 'SET', 'ATTACK', 'BLOCK', 'DIG');

-- CreateEnum
CREATE TYPE "ResolveSource" AS ENUM ('SNAPSHOT_EXACT', 'IOU_MATCH', 'NEAREST_CENTER', 'MANUAL', 'UNRESOLVED');

-- CreateTable
CREATE TABLE "rally_action_ground_truth" (
    "id" TEXT NOT NULL,
    "rally_id" TEXT NOT NULL,
    "frame" INTEGER NOT NULL,
    "action" "ActionLabel" NOT NULL,
    "snapshot_bbox_x1" DOUBLE PRECISION,
    "snapshot_bbox_y1" DOUBLE PRECISION,
    "snapshot_bbox_x2" DOUBLE PRECISION,
    "snapshot_bbox_y2" DOUBLE PRECISION,
    "snapshot_ball_x" DOUBLE PRECISION,
    "snapshot_ball_y" DOUBLE PRECISION,
    "snapshot_team" "ServingTeam",
    "snapshot_track_id" INTEGER,
    "resolved_track_id" INTEGER,
    "resolved_at" TIMESTAMPTZ(6),
    "resolved_source" "ResolveSource",
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "created_by" TEXT,

    CONSTRAINT "rally_action_ground_truth_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "rally_action_ground_truth_rally_id_idx" ON "rally_action_ground_truth"("rally_id");

-- CreateIndex
CREATE INDEX "rally_action_ground_truth_rally_id_frame_idx" ON "rally_action_ground_truth"("rally_id", "frame");

-- CreateIndex
CREATE UNIQUE INDEX "rally_action_ground_truth_rally_id_frame_action_key" ON "rally_action_ground_truth"("rally_id", "frame", "action");

-- AddForeignKey
ALTER TABLE "rally_action_ground_truth" ADD CONSTRAINT "rally_action_ground_truth_rally_id_fkey" FOREIGN KEY ("rally_id") REFERENCES "rallies"("id") ON DELETE CASCADE ON UPDATE CASCADE;
