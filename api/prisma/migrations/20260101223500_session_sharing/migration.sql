-- CreateTable
CREATE TABLE "session_shares" (
    "id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "session_shares_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "session_members" (
    "id" TEXT NOT NULL,
    "session_share_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "joined_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "session_members_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "session_shares_session_id_key" ON "session_shares"("session_id");

-- CreateIndex
CREATE UNIQUE INDEX "session_shares_token_key" ON "session_shares"("token");

-- CreateIndex
CREATE INDEX "session_members_user_id_idx" ON "session_members"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "session_members_session_share_id_user_id_key" ON "session_members"("session_share_id", "user_id");

-- AlterTable
ALTER TABLE "highlights" ADD COLUMN "created_by_user_id" TEXT;

-- CreateIndex
CREATE INDEX "highlights_created_by_user_id_idx" ON "highlights"("created_by_user_id");

-- AddForeignKey
ALTER TABLE "session_shares" ADD CONSTRAINT "session_shares_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "session_members" ADD CONSTRAINT "session_members_session_share_id_fkey" FOREIGN KEY ("session_share_id") REFERENCES "session_shares"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "session_members" ADD CONSTRAINT "session_members_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "highlights" ADD CONSTRAINT "highlights_created_by_user_id_fkey" FOREIGN KEY ("created_by_user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
