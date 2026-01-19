-- CreateEnum
CREATE TYPE "RallyStatus" AS ENUM ('CONFIRMED', 'SUGGESTED');

-- CreateEnum
CREATE TYPE "RejectionReason" AS ENUM ('INSUFFICIENT_WINDOWS', 'TOO_SHORT', 'SPARSE_DENSITY');

-- AlterTable
ALTER TABLE "rallies" ADD COLUMN     "rejection_reason" "RejectionReason",
ADD COLUMN     "status" "RallyStatus" NOT NULL DEFAULT 'CONFIRMED';

-- CreateIndex
CREATE INDEX "rallies_video_id_status_idx" ON "rallies"("video_id", "status");
