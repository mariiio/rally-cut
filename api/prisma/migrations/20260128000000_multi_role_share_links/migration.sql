-- CreateEnum
CREATE TYPE "MemberRole" AS ENUM ('VIEWER', 'EDITOR', 'ADMIN');

-- Multi-role share links: Each session can have 3 share links (one per role)
-- Delete existing data for clean migration
DELETE FROM "session_members";
DELETE FROM "session_shares";

-- Drop old unique constraint on sessionId (allowing multiple shares per session)
DROP INDEX IF EXISTS "session_shares_session_id_key";

-- Add role column to session_shares
ALTER TABLE "session_shares" ADD COLUMN "role" "MemberRole" NOT NULL DEFAULT 'VIEWER';

-- Add role column to session_members
ALTER TABLE "session_members" ADD COLUMN "role" "MemberRole" NOT NULL DEFAULT 'VIEWER';

-- Add compound unique constraint (one link per role per session)
CREATE UNIQUE INDEX "session_shares_session_id_role_key" ON "session_shares"("session_id", "role");

-- Add index for efficient session lookups
CREATE INDEX "session_shares_session_id_idx" ON "session_shares"("session_id");
