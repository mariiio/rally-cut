-- Rename PREMIUM to PRO and add ELITE tier
-- PostgreSQL enum modification requires creating a new enum type

-- Step 1: Create new enum with updated values
CREATE TYPE "UserTier_new" AS ENUM ('FREE', 'PRO', 'ELITE');

-- Step 2: Update users table - drop default, convert column, re-add default
ALTER TABLE "users" ALTER COLUMN "tier" DROP DEFAULT;
ALTER TABLE "users"
  ALTER COLUMN "tier" TYPE "UserTier_new"
  USING (CASE WHEN "tier"::text = 'PREMIUM' THEN 'PRO' ELSE "tier"::text END::"UserTier_new");
ALTER TABLE "users" ALTER COLUMN "tier" SET DEFAULT 'FREE'::"UserTier_new";

-- Step 3: Update export_jobs table - drop default if exists, convert column
ALTER TABLE "export_jobs" ALTER COLUMN "tier" DROP DEFAULT;
ALTER TABLE "export_jobs"
  ALTER COLUMN "tier" TYPE "UserTier_new"
  USING (CASE WHEN "tier"::text = 'PREMIUM' THEN 'PRO' ELSE "tier"::text END::"UserTier_new");

-- Step 4: Drop old enum and rename new one
DROP TYPE "UserTier";
ALTER TYPE "UserTier_new" RENAME TO "UserTier";
