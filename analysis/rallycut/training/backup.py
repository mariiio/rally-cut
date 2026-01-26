"""S3 backup for training datasets.

Backs up training datasets (videos + ground truth + manifest) to AWS S3
so the complete training pipeline survives local DB resets and MinIO volume clears.

Videos are stored by content_hash for deduplication across dataset versions.

S3 key structure:
    {prefix}/
      datasets/{name}/
        manifest.json
        ground_truth.json
      videos/
        {content_hash}.mp4
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from platformdirs import user_cache_dir
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from rallycut.core.config import get_config

# 50 MB threshold for multipart uploads
MULTIPART_THRESHOLD = 50 * 1024 * 1024


@dataclass
class DatasetInfo:
    """Summary of a remote dataset."""

    name: str
    video_count: int = 0
    rally_count: int = 0
    total_duration_min: float = 0.0
    created: str = ""


@dataclass
class PushResult:
    """Result from pushing a dataset to S3."""

    uploaded_videos: int = 0
    skipped_videos: int = 0
    uploaded_bytes: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class PullResult:
    """Result from pulling a dataset from S3."""

    downloaded_videos: int = 0
    cached_videos: int = 0
    downloaded_bytes: int = 0
    errors: list[str] = field(default_factory=list)


class DatasetBackup:
    """Manages S3 backup of training datasets.

    Uses the default AWS credential chain (~/.aws/credentials, env vars, etc.)
    NOT the MinIO credentials used by the app.
    """

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str | None = None,
        region: str | None = None,
    ):
        cfg = get_config().training_backup

        self.bucket = bucket or os.getenv("TRAINING_S3_BUCKET", "") or cfg.s3_bucket
        self.prefix = prefix or os.getenv("TRAINING_S3_PREFIX", "") or cfg.s3_prefix
        self.region = region or os.getenv("TRAINING_S3_REGION", "") or cfg.s3_region

        if not self.bucket:
            raise ValueError(
                "TRAINING_S3_BUCKET not set. Either set the environment variable, "
                "configure training_backup.s3_bucket in rallycut.yaml, or pass --bucket."
            )

        # Use default credential chain (NOT MinIO creds)
        self.s3 = boto3.client(
            "s3",
            config=BotoConfig(signature_version="s3v4"),
            region_name=self.region,
        )

        self.cache_dir = Path(user_cache_dir("rallycut")) / "evaluation"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_key(self, name: str, filename: str) -> str:
        """Build S3 key for dataset metadata."""
        return f"{self.prefix}/datasets/{name}/{filename}"

    def _video_key(self, content_hash: str) -> str:
        """Build S3 key for a video file."""
        return f"{self.prefix}/videos/{content_hash}.mp4"

    def _video_exists_remote(self, content_hash: str) -> bool:
        """Check if a video already exists in S3 by content hash."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._video_key(content_hash))
            return True
        except ClientError:
            return False

    def push_dataset(
        self,
        dataset_dir: Path,
        name: str,
        progress: Progress | None = None,
    ) -> PushResult:
        """Upload dataset metadata and videos to S3.

        Args:
            dataset_dir: Local dataset directory (with manifest.json, ground_truth.json, videos/).
            name: Dataset name used as S3 key prefix.
            progress: Optional Rich progress instance for UI updates.

        Returns:
            PushResult with upload statistics.
        """
        result = PushResult()

        manifest_path = dataset_dir / "manifest.json"
        ground_truth_path = dataset_dir / "ground_truth.json"

        # Upload metadata files
        for path, filename in [
            (manifest_path, "manifest.json"),
            (ground_truth_path, "ground_truth.json"),
        ]:
            self.s3.upload_file(
                str(path),
                self.bucket,
                self._dataset_key(name, filename),
            )

        # Load manifest for video list
        with open(manifest_path) as f:
            manifest = json.load(f)

        videos = manifest.get("videos", [])
        if not videos:
            return result

        # Set up progress tracking and transfer config
        task_id = None
        if progress:
            task_id = progress.add_task("Uploading videos...", total=len(videos))

        transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD,
        )

        for video in videos:
            content_hash = video["content_hash"]
            filename = video["filename"]

            try:
                # Check if already uploaded (dedup by content hash)
                if self._video_exists_remote(content_hash):
                    result.skipped_videos += 1
                    if progress and task_id is not None:
                        progress.advance(task_id)
                    continue

                # Find the video file locally
                local_path = self._find_video_locally(dataset_dir, content_hash, filename)
                if local_path is None:
                    result.errors.append(f"{filename}: not found locally")
                    if progress and task_id is not None:
                        progress.advance(task_id)
                    continue

                file_size = local_path.stat().st_size

                self.s3.upload_file(
                    str(local_path),
                    self.bucket,
                    self._video_key(content_hash),
                    ExtraArgs={"StorageClass": "STANDARD_IA"},
                    Config=transfer_config,
                )

                result.uploaded_videos += 1
                result.uploaded_bytes += file_size

            except Exception as e:
                result.errors.append(f"{filename}: {e}")

            if progress and task_id is not None:
                progress.advance(task_id)

        return result

    def pull_dataset(
        self,
        name: str,
        output_dir: Path,
        progress: Progress | None = None,
    ) -> PullResult:
        """Download dataset metadata and videos from S3.

        Videos are cached at ~/.cache/rallycut/evaluation/{content_hash}.mp4
        and symlinked into the dataset's videos/ directory.

        Args:
            name: Dataset name to pull.
            output_dir: Local directory to write dataset into.
            progress: Optional Rich progress instance for UI updates.

        Returns:
            PullResult with download statistics.
        """
        result = PullResult()

        dataset_dir = output_dir / name
        videos_dir = dataset_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Download metadata files
        for filename in ["manifest.json", "ground_truth.json"]:
            local_path = dataset_dir / filename
            self.s3.download_file(
                self.bucket,
                self._dataset_key(name, filename),
                str(local_path),
            )

        # Load manifest for video list
        with open(dataset_dir / "manifest.json") as f:
            manifest = json.load(f)

        videos = manifest.get("videos", [])
        if not videos:
            return result

        # Set up progress tracking and transfer config
        task_id = None
        if progress:
            task_id = progress.add_task("Downloading videos...", total=len(videos))

        transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD,
        )

        for video in videos:
            content_hash = video["content_hash"]
            filename = video["filename"]

            try:
                cache_path = self.cache_dir / f"{content_hash}.mp4"

                if cache_path.exists():
                    result.cached_videos += 1
                else:
                    self.s3.download_file(
                        self.bucket,
                        self._video_key(content_hash),
                        str(cache_path),
                        Config=transfer_config,
                    )
                    result.downloaded_videos += 1
                    result.downloaded_bytes += cache_path.stat().st_size

                # Symlink into dataset videos/ directory
                link_path = videos_dir / filename
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(cache_path.resolve())

            except Exception as e:
                result.errors.append(f"{filename}: {e}")

            if progress and task_id is not None:
                progress.advance(task_id)

        return result

    def list_datasets(self) -> list[DatasetInfo]:
        """List all datasets backed up in S3.

        Returns:
            List of DatasetInfo with name, video count, rally count, duration.
        """
        prefix = f"{self.prefix}/datasets/"
        datasets: list[DatasetInfo] = []

        # List "directories" under datasets/
        paginator = self.s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter="/",
        )

        dataset_names: list[str] = []
        for page in pages:
            for cp in page.get("CommonPrefixes", []):
                # cp["Prefix"] = "training/datasets/beach_v1/"
                name = cp["Prefix"][len(prefix) :].rstrip("/")
                if name:
                    dataset_names.append(name)

        # Download each manifest to get details
        for name in sorted(dataset_names):
            try:
                response = self.s3.get_object(
                    Bucket=self.bucket,
                    Key=self._dataset_key(name, "manifest.json"),
                )
                manifest = json.loads(response["Body"].read().decode("utf-8"))

                stats = manifest.get("stats", {})
                info = DatasetInfo(
                    name=name,
                    video_count=stats.get("total_videos", len(manifest.get("videos", []))),
                    rally_count=stats.get("total_rallies", 0),
                    total_duration_min=stats.get("total_duration_min", 0.0),
                    created=manifest.get("created", ""),
                )
                datasets.append(info)
            except ClientError:
                # Manifest missing, still list the dataset
                datasets.append(DatasetInfo(name=name))

        return datasets

    def _find_video_locally(
        self,
        dataset_dir: Path,
        content_hash: str,
        filename: str,
    ) -> Path | None:
        """Find a video file locally, checking cache and dataset directory.

        Search order:
        1. Local cache (~/.cache/rallycut/evaluation/{content_hash}.mp4)
        2. Dataset videos/ directory (may be symlink to cache)
        3. Dataset videos/ directory by filename
        """
        # Check cache first
        cache_path = self.cache_dir / f"{content_hash}.mp4"
        if cache_path.exists():
            return cache_path

        # Check dataset videos dir (resolve symlinks)
        video_path = dataset_dir / "videos" / filename
        if video_path.exists():
            return video_path.resolve()

        return None


def make_transfer_progress() -> Progress:
    """Create a Rich progress bar for file transfers."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
