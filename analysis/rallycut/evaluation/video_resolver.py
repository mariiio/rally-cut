"""Resolve video files from S3/MinIO for local analysis."""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.config import Config
from platformdirs import user_cache_dir


class VideoResolver:
    """Resolves S3 keys to local file paths by downloading from S3/MinIO.

    Supports both AWS S3 and local MinIO. Configuration is loaded from
    environment variables, falling back to api/.env for local development.
    """

    def __init__(
        self,
        s3_endpoint: str | None = None,
        bucket_name: str | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize video resolver.

        Args:
            s3_endpoint: S3 endpoint URL. Defaults to S3_ENDPOINT env var or MinIO local.
            bucket_name: S3 bucket name. Defaults to S3_BUCKET_NAME env var.
            cache_dir: Directory for caching downloaded videos.
        """
        # Load config from env or api/.env
        self._load_env_from_api()

        self.s3_endpoint = s3_endpoint or os.getenv("S3_ENDPOINT", "http://localhost:9000")
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", "rallycut-dev")
        self.cache_dir = cache_dir or Path(user_cache_dir("rallycut")) / "evaluation"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 client
        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            config=Config(signature_version="s3v4"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

    def _load_env_from_api(self) -> None:
        """Load environment variables from api/.env if not already set."""
        if os.getenv("S3_BUCKET_NAME"):
            return  # Already configured

        api_env = Path(__file__).parents[4] / "api" / ".env"
        if not api_env.exists():
            return

        for line in api_env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # Only set if not already in environment
            if key not in os.environ:
                os.environ[key] = value

    def resolve(self, s3_key: str, content_hash: str) -> Path:
        """Download video from S3 and return local path.

        Uses content_hash for caching to avoid re-downloads. The same video
        content will only be downloaded once regardless of s3_key.

        Args:
            s3_key: The S3 key/path of the video.
            content_hash: SHA-256 hash of video content for cache key.

        Returns:
            Path to the downloaded video file.

        Raises:
            botocore.exceptions.ClientError: If download fails.
        """
        # Use content hash for cache filename to enable deduplication
        ext = Path(s3_key).suffix or ".mp4"
        cache_path = self.cache_dir / f"{content_hash}{ext}"

        if cache_path.exists():
            return cache_path

        # Download from S3/MinIO
        self.s3.download_file(self.bucket_name, s3_key, str(cache_path))
        return cache_path

    def is_cached(self, content_hash: str, ext: str = ".mp4") -> bool:
        """Check if a video is already cached.

        Args:
            content_hash: SHA-256 hash of video content.
            ext: File extension (default .mp4).

        Returns:
            True if video is cached locally.
        """
        cache_path = self.cache_dir / f"{content_hash}{ext}"
        return cache_path.exists()

    def get_cached_path(self, content_hash: str, ext: str = ".mp4") -> Path | None:
        """Get the cached path for a video if it exists.

        Args:
            content_hash: SHA-256 hash of video content.
            ext: File extension (default .mp4).

        Returns:
            Path to cached video or None if not cached.
        """
        cache_path = self.cache_dir / f"{content_hash}{ext}"
        return cache_path if cache_path.exists() else None

    def clear_cache(self) -> int:
        """Clear the video cache.

        Returns:
            Number of files deleted.
        """
        count = 0
        for f in self.cache_dir.glob("*"):
            if f.is_file():
                f.unlink()
                count += 1
        return count

    def cache_size_bytes(self) -> int:
        """Get total size of cached videos in bytes."""
        total = 0
        for f in self.cache_dir.glob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def cache_size_human(self) -> str:
        """Get human-readable cache size."""
        size_f = float(self.cache_size_bytes())
        for unit in ["B", "KB", "MB", "GB"]:
            if size_f < 1024:
                return f"{size_f:.1f} {unit}"
            size_f /= 1024
        return f"{size_f:.1f} TB"
