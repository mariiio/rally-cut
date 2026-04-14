-- Add new column (before touching the old one)
ALTER TABLE "videos" ADD COLUMN "quality_report_json" JSONB;

-- Migrate brightness + resolution out of characteristicsJson into the new report.
-- If either field is missing, it's set to NULL in the new report — non-destructive.
UPDATE "videos"
SET "quality_report_json" = jsonb_build_object(
  'version', 2,
  'issues', '[]'::jsonb,
  'brightness', "characteristics_json" -> 'brightness',
  'resolution', "characteristics_json" -> 'resolution'
)
WHERE "characteristics_json" IS NOT NULL;

-- Add REJECTED enum value
-- Note: ALTER TYPE ... ADD VALUE cannot run inside a transaction in PostgreSQL < 12.
-- Prisma runs this migration outside a transaction block for this reason.
ALTER TYPE "VideoStatus" ADD VALUE 'REJECTED';

-- Drop old column
ALTER TABLE "videos" DROP COLUMN "characteristics_json";
