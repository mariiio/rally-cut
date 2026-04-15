-- AlterTable
ALTER TABLE "videos" ADD COLUMN "pending_analysis_edits_json" JSONB;
ALTER TABLE "videos" ADD COLUMN "match_analysis_ran_at" TIMESTAMP(3);
