-- AlterTable
ALTER TABLE "users" ADD COLUMN     "last_active_at" TIMESTAMP(3);

-- AlterTable
ALTER TABLE "videos" ADD COLUMN     "quality_downgraded_at" TIMESTAMP(3);
