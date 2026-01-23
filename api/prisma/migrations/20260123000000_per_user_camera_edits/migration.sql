-- Per-user camera edits: Each user has their own camera edits for rallies and videos
-- This allows session members to have their own camera edits without affecting the owner's edits

-- Step 1: Add user_id column to rally_camera_edits
ALTER TABLE "rally_camera_edits" ADD COLUMN "user_id" TEXT;

-- Step 2: Backfill existing camera edits with video owner's userId
UPDATE "rally_camera_edits" rce
SET "user_id" = v."user_id"
FROM "rallies" r
JOIN "videos" v ON r."video_id" = v."id"
WHERE rce."rally_id" = r."id";

-- Step 3: Drop old unique index and add new composite unique constraint
DROP INDEX IF EXISTS "rally_camera_edits_rally_id_key";
ALTER TABLE "rally_camera_edits" ADD CONSTRAINT "rally_camera_edits_rally_id_user_id_key" UNIQUE ("rally_id", "user_id");

-- Step 4: Add foreign key to users table
ALTER TABLE "rally_camera_edits" ADD CONSTRAINT "rally_camera_edits_user_id_fkey"
  FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Step 5: Add index on user_id for performance
CREATE INDEX "rally_camera_edits_user_id_idx" ON "rally_camera_edits"("user_id");

-- Step 6: Add user_id column to video_camera_settings
ALTER TABLE "video_camera_settings" ADD COLUMN "user_id" TEXT;

-- Step 7: Backfill existing camera settings with video owner's userId
UPDATE "video_camera_settings" vcs
SET "user_id" = v."user_id"
FROM "videos" v
WHERE vcs."video_id" = v."id";

-- Step 8: Drop old unique index and add new composite unique constraint
DROP INDEX IF EXISTS "video_camera_settings_video_id_key";
ALTER TABLE "video_camera_settings" ADD CONSTRAINT "video_camera_settings_video_id_user_id_key" UNIQUE ("video_id", "user_id");

-- Step 9: Add foreign key to users table
ALTER TABLE "video_camera_settings" ADD CONSTRAINT "video_camera_settings_user_id_fkey"
  FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Step 10: Add index on user_id for performance
CREATE INDEX "video_camera_settings_user_id_idx" ON "video_camera_settings"("user_id");
