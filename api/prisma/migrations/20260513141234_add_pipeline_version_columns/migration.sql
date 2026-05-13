-- AlterTable
ALTER TABLE "player_tracks" ADD COLUMN     "contacts_pipeline_version" TEXT,
                            ADD COLUMN     "actions_pipeline_version"  TEXT;

-- Backfill existing rows with the 'v0' sentinel where content exists.
UPDATE "player_tracks" SET "contacts_pipeline_version" = 'v0' WHERE "contacts_json" IS NOT NULL;
UPDATE "player_tracks" SET "actions_pipeline_version"  = 'v0' WHERE "actions_json"  IS NOT NULL;
