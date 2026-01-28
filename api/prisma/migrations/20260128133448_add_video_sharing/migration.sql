-- AlterTable
ALTER TABLE "users" ADD COLUMN     "email_verified" TIMESTAMP(3),
ADD COLUMN     "password_hash" TEXT;

-- CreateTable
CREATE TABLE "accounts" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "provider_account_id" TEXT NOT NULL,
    "refresh_token" TEXT,
    "access_token" TEXT,
    "expires_at" INTEGER,
    "token_type" TEXT,
    "scope" TEXT,
    "id_token" TEXT,
    "session_state" TEXT,

    CONSTRAINT "accounts_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "verification_tokens" (
    "identifier" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "expires" TIMESTAMP(3) NOT NULL
);

-- CreateTable
CREATE TABLE "video_shares" (
    "id" TEXT NOT NULL,
    "video_id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "role" "MemberRole" NOT NULL DEFAULT 'VIEWER',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "video_shares_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "video_members" (
    "id" TEXT NOT NULL,
    "video_share_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "role" "MemberRole" NOT NULL DEFAULT 'VIEWER',
    "joined_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "video_members_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "accounts_user_id_idx" ON "accounts"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "accounts_provider_provider_account_id_key" ON "accounts"("provider", "provider_account_id");

-- CreateIndex
CREATE UNIQUE INDEX "verification_tokens_token_key" ON "verification_tokens"("token");

-- CreateIndex
CREATE UNIQUE INDEX "verification_tokens_identifier_token_key" ON "verification_tokens"("identifier", "token");

-- CreateIndex
CREATE UNIQUE INDEX "video_shares_token_key" ON "video_shares"("token");

-- CreateIndex
CREATE INDEX "video_shares_video_id_idx" ON "video_shares"("video_id");

-- CreateIndex
CREATE UNIQUE INDEX "video_shares_video_id_role_key" ON "video_shares"("video_id", "role");

-- CreateIndex
CREATE INDEX "video_members_user_id_idx" ON "video_members"("user_id");

-- CreateIndex
CREATE INDEX "video_members_video_share_id_idx" ON "video_members"("video_share_id");

-- CreateIndex
CREATE UNIQUE INDEX "video_members_video_share_id_user_id_key" ON "video_members"("video_share_id", "user_id");

-- AddForeignKey
ALTER TABLE "accounts" ADD CONSTRAINT "accounts_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "video_shares" ADD CONSTRAINT "video_shares_video_id_fkey" FOREIGN KEY ("video_id") REFERENCES "videos"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "video_members" ADD CONSTRAINT "video_members_video_share_id_fkey" FOREIGN KEY ("video_share_id") REFERENCES "video_shares"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "video_members" ADD CONSTRAINT "video_members_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
