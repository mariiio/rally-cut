-- Add match-analysis run-state tracking so the UI can detect true completion
-- instead of inferring from matchStatsJson presence.

ALTER TABLE "videos"
  ADD COLUMN "match_analysis_started_at" TIMESTAMP(3),
  ADD COLUMN "match_analysis_error" TEXT;
