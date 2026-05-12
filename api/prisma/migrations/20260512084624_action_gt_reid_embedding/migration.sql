-- AlterEnum
ALTER TYPE "ResolveSource" ADD VALUE 'REID_MATCH' BEFORE 'NEAREST_CENTER';

-- AlterTable
ALTER TABLE "rally_action_ground_truth" ADD COLUMN "snapshot_reid_embedding" BYTEA;
