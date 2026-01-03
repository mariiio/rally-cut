-- AlterTable
ALTER TABLE "videos" ADD COLUMN     "poster_s3_key" TEXT,
ADD COLUMN     "proxy_file_size_bytes" BIGINT,
ADD COLUMN     "proxy_generated_at" TIMESTAMP(3),
ADD COLUMN     "proxy_last_accessed_at" TIMESTAMP(3),
ADD COLUMN     "proxy_s3_key" TEXT;
