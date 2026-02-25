"""S3 backup for training datasets and model weights.

Backs up training datasets (videos + ground truth + manifest) to AWS S3
so the complete training pipeline survives local DB resets and MinIO volume clears.

Videos are stored by content_hash for deduplication across dataset versions.
Model weights are stored by SHA-256 content hash for deduplication across snapshots.

S3 key structure:
    {prefix}/
      datasets/{name}/
        manifest.json
        ground_truth.json
        tracking_ground_truth.json  (optional)
        action_ground_truth.json    (optional)
      videos/
        {content_hash}.mp4
      weights/
        manifest.json                       # "latest" manifest
        snapshots/{name}/manifest.json      # Named snapshots
        files/{sha256_hex}                  # Deduplicated weight files
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
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

# Multipart transfer tuning for large video files
MULTIPART_THRESHOLD = 50 * 1024 * 1024  # 50 MB: switch to multipart above this
MULTIPART_CHUNKSIZE = 25 * 1024 * 1024  # 25 MB: fewer parts than default 8 MB
# Concurrent video transfers (HEAD checks + uploads/downloads)
MAX_WORKERS = 4


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


# Weight file groups for backup. Paths are relative to weights/ directory.
# "trained" = fine-tuned/trained locally, "external" = downloaded from external sources.
WEIGHT_GROUPS: dict[str, list[str]] = {
    "trained": [
        "videomae/beach_volleyball/model.safetensors",
        "videomae/beach_volleyball/config.json",
        "videomae/beach_volleyball/training_args.bin",
        "videomae/beach_volleyball/preprocessor_config.json",
        "temporal_maxer/best_temporal_maxer.pt",
        "temporal_maxer/temporal_maxer_result.json",
        "wasb/wasb_finetuned.pth.tar",
        "contact_classifier/contact_classifier.pkl",
        "action_classifier/action_classifier.pkl",
    ],
    "external": [
        "videomae/game_state_classifier/model.safetensors",
        "videomae/game_state_classifier/config.json",
        "videomae/game_state_classifier/preprocessor_config.json",
        "wasb/wasb_volleyball_best.pth.tar",
    ],
}


@dataclass
class WeightFileInfo:
    """Info about a single weight file in a manifest."""

    relative_path: str
    content_hash: str
    size_bytes: int
    group: str


@dataclass
class WeightPushResult:
    """Result from pushing weights to S3."""

    uploaded_files: int = 0
    skipped_files: int = 0
    uploaded_bytes: int = 0
    total_files: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class WeightPullResult:
    """Result from pulling weights from S3."""

    downloaded_files: int = 0
    skipped_files: int = 0
    downloaded_bytes: int = 0
    total_files: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class WeightSnapshotInfo:
    """Summary of a weight snapshot in S3."""

    name: str
    created: str = ""
    file_count: int = 0
    total_size_bytes: int = 0


def _compute_file_hash(path: Path) -> str:
    """Compute SHA-256 content hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):  # 1 MB reads for large weight files
            h.update(chunk)
    return h.hexdigest()


class DatasetBackup:
    """Manages S3 backup of training datasets and model weights.

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

        Videos are uploaded concurrently (4 workers) with per-video
        dedup checks. boto3 upload_file handles multipart internally.

        Args:
            dataset_dir: Local dataset directory (with manifest.json, ground_truth.json, videos/).
            name: Dataset name used as S3 key prefix.
            progress: Optional Rich progress instance for UI updates.

        Returns:
            PushResult with upload statistics.
        """
        result = PushResult()
        lock = threading.Lock()

        manifest_path = dataset_dir / "manifest.json"
        ground_truth_path = dataset_dir / "ground_truth.json"

        # Upload metadata files
        metadata_files = [
            (manifest_path, "manifest.json"),
            (ground_truth_path, "ground_truth.json"),
        ]
        tracking_gt_path = dataset_dir / "tracking_ground_truth.json"
        if tracking_gt_path.exists():
            metadata_files.append((tracking_gt_path, "tracking_ground_truth.json"))
        action_gt_path = dataset_dir / "action_ground_truth.json"
        if action_gt_path.exists():
            metadata_files.append((action_gt_path, "action_ground_truth.json"))

        for path, filename in metadata_files:
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
            multipart_chunksize=MULTIPART_CHUNKSIZE,
        )

        def _upload_one(video: dict[str, str]) -> None:
            content_hash = video["content_hash"]
            filename = video["filename"]

            try:
                # Check if already uploaded (dedup by content hash)
                if self._video_exists_remote(content_hash):
                    with lock:
                        result.skipped_videos += 1
                else:
                    # Find the video file locally
                    local_path = self._find_video_locally(dataset_dir, content_hash, filename)
                    if local_path is None:
                        with lock:
                            result.errors.append(f"{filename}: not found locally")
                    else:
                        file_size = local_path.stat().st_size
                        self.s3.upload_file(
                            str(local_path),
                            self.bucket,
                            self._video_key(content_hash),
                            ExtraArgs={"StorageClass": "STANDARD_IA"},
                            Config=transfer_config,
                        )
                        with lock:
                            result.uploaded_videos += 1
                            result.uploaded_bytes += file_size
            except Exception as e:
                with lock:
                    result.errors.append(f"{filename}: {e}")

            if progress and task_id is not None:
                progress.advance(task_id)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_upload_one, v) for v in videos]
            for future in as_completed(futures):
                future.result()  # Propagate unexpected exceptions

        return result

    def pull_dataset(
        self,
        name: str,
        output_dir: Path,
        progress: Progress | None = None,
    ) -> PullResult:
        """Download dataset metadata and videos from S3.

        Videos are downloaded concurrently (4 workers), cached at
        ~/.cache/rallycut/evaluation/{content_hash}.mp4, and symlinked
        into the dataset's videos/ directory.

        Args:
            name: Dataset name to pull.
            output_dir: Local directory to write dataset into.
            progress: Optional Rich progress instance for UI updates.

        Returns:
            PullResult with download statistics.
        """
        result = PullResult()
        lock = threading.Lock()

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

        # Download tracking ground truth (optional — older datasets may not have it)
        try:
            tracking_gt_path = dataset_dir / "tracking_ground_truth.json"
            self.s3.download_file(
                self.bucket,
                self._dataset_key(name, "tracking_ground_truth.json"),
                str(tracking_gt_path),
            )
        except ClientError:
            pass  # Not present in older datasets

        # Download action ground truth (optional — older datasets may not have it)
        try:
            action_gt_path = dataset_dir / "action_ground_truth.json"
            self.s3.download_file(
                self.bucket,
                self._dataset_key(name, "action_ground_truth.json"),
                str(action_gt_path),
            )
        except ClientError:
            pass  # Not present in older datasets

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
            multipart_chunksize=MULTIPART_CHUNKSIZE,
        )

        def _download_one(video: dict[str, str]) -> None:
            content_hash = video["content_hash"]
            filename = video["filename"]

            try:
                cache_path = self.cache_dir / f"{content_hash}.mp4"

                if cache_path.exists():
                    with lock:
                        result.cached_videos += 1
                else:
                    self.s3.download_file(
                        self.bucket,
                        self._video_key(content_hash),
                        str(cache_path),
                        Config=transfer_config,
                    )
                    with lock:
                        result.downloaded_videos += 1
                        result.downloaded_bytes += cache_path.stat().st_size

                # Symlink into dataset videos/ directory (serialized via lock)
                with lock:
                    link_path = videos_dir / filename
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    link_path.symlink_to(cache_path.resolve())

            except Exception as e:
                with lock:
                    result.errors.append(f"{filename}: {e}")

            if progress and task_id is not None:
                progress.advance(task_id)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_download_one, v) for v in videos]
            for future in as_completed(futures):
                future.result()  # Propagate unexpected exceptions

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

    def _weight_file_key(self, content_hash: str) -> str:
        """Build S3 key for a weight file (stored by content hash)."""
        return f"{self.prefix}/weights/files/{content_hash}"

    def _weight_manifest_key(self, name: str | None = None) -> str:
        """Build S3 key for a weight manifest."""
        if name is None:
            return f"{self.prefix}/weights/manifest.json"
        return f"{self.prefix}/weights/snapshots/{name}/manifest.json"

    def _weight_file_exists_remote(self, content_hash: str) -> bool:
        """Check if a weight file already exists in S3 by content hash."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._weight_file_key(content_hash))
            return True
        except ClientError:
            return False

    def push_weights(
        self,
        weights_dir: Path,
        name: str | None = None,
        include_external: bool = False,
        progress: Progress | None = None,
    ) -> WeightPushResult:
        """Upload model weights to S3 with content-hash deduplication.

        Scans weight groups, computes SHA-256 hashes, uploads new files
        concurrently, and writes a manifest to "latest" (and optionally
        a named snapshot).

        Args:
            weights_dir: Local weights/ directory.
            name: Optional snapshot name. If set, writes both latest and named snapshot.
            include_external: Also backup external (downloaded) weights.
            progress: Optional Rich progress instance for UI updates.

        Returns:
            WeightPushResult with upload statistics.
        """
        result = WeightPushResult()
        lock = threading.Lock()

        # Collect files to push
        groups = ["trained"]
        if include_external:
            groups.append("external")

        file_infos: list[WeightFileInfo] = []
        for group in groups:
            for rel_path in WEIGHT_GROUPS[group]:
                full_path = weights_dir / rel_path
                if not full_path.exists():
                    continue
                content_hash = _compute_file_hash(full_path)
                file_infos.append(
                    WeightFileInfo(
                        relative_path=rel_path,
                        content_hash=content_hash,
                        size_bytes=full_path.stat().st_size,
                        group=group,
                    )
                )

        result.total_files = len(file_infos)

        if not file_infos:
            return result

        # Upload files concurrently with dedup
        task_id = None
        if progress:
            task_id = progress.add_task("Uploading weights...", total=len(file_infos))

        transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD,
            multipart_chunksize=MULTIPART_CHUNKSIZE,
        )

        failed_paths: set[str] = set()

        def _upload_one(info: WeightFileInfo) -> None:
            try:
                if self._weight_file_exists_remote(info.content_hash):
                    with lock:
                        result.skipped_files += 1
                else:
                    local_path = weights_dir / info.relative_path
                    self.s3.upload_file(
                        str(local_path),
                        self.bucket,
                        self._weight_file_key(info.content_hash),
                        ExtraArgs={"StorageClass": "STANDARD_IA"},
                        Config=transfer_config,
                    )
                    with lock:
                        result.uploaded_files += 1
                        result.uploaded_bytes += info.size_bytes
            except Exception as e:
                with lock:
                    result.errors.append(f"{info.relative_path}: {e}")
                    failed_paths.add(info.relative_path)

            if progress and task_id is not None:
                progress.advance(task_id)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_upload_one, info) for info in file_infos]
            for future in as_completed(futures):
                future.result()

        # Build and upload manifest (exclude failed uploads)
        successful = [f for f in file_infos if f.relative_path not in failed_paths]
        manifest = {
            "created": datetime.now(UTC).isoformat(),
            "name": name or "latest",
            "total_size_bytes": sum(f.size_bytes for f in successful),
            "file_count": len(successful),
            "files": [
                {
                    "relative_path": f.relative_path,
                    "content_hash": f"sha256:{f.content_hash}",
                    "size_bytes": f.size_bytes,
                    "group": f.group,
                }
                for f in successful
            ],
        }
        manifest_json = json.dumps(manifest, indent=2)

        # Always write to "latest"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._weight_manifest_key(None),
            Body=manifest_json.encode(),
            ContentType="application/json",
        )

        # Also write named snapshot if requested
        if name:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._weight_manifest_key(name),
                Body=manifest_json.encode(),
                ContentType="application/json",
            )

        return result

    def pull_weights(
        self,
        weights_dir: Path,
        name: str | None = None,
        include_external: bool = False,
        progress: Progress | None = None,
    ) -> WeightPullResult:
        """Download model weights from S3, skipping files that match locally.

        Downloads the manifest (latest or named snapshot), compares local
        file hashes, and downloads only changed/missing files concurrently.

        Args:
            weights_dir: Local weights/ directory to write into.
            name: Snapshot name to pull, or None for "latest".
            include_external: Also pull external (downloaded) weights.
            progress: Optional Rich progress instance for UI updates.

        Returns:
            WeightPullResult with download statistics.
        """
        result = WeightPullResult()
        lock = threading.Lock()

        # Download manifest
        manifest_key = self._weight_manifest_key(name)
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=manifest_key)
            manifest = json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            snapshot_label = name or "latest"
            result.errors.append(f"Manifest not found for '{snapshot_label}': {e}")
            return result

        files = manifest.get("files", [])

        # Filter by group
        if not include_external:
            files = [f for f in files if f.get("group") != "external"]

        result.total_files = len(files)

        if not files:
            return result

        task_id = None
        if progress:
            task_id = progress.add_task("Downloading weights...", total=len(files))

        transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD,
            multipart_chunksize=MULTIPART_CHUNKSIZE,
        )

        def _download_one(file_entry: dict[str, str | int]) -> None:
            rel_path = str(file_entry["relative_path"])
            remote_hash = str(file_entry["content_hash"]).removeprefix("sha256:")
            local_path = weights_dir / rel_path

            try:
                # Skip if local file matches
                if local_path.exists():
                    local_hash = _compute_file_hash(local_path)
                    if local_hash == remote_hash:
                        with lock:
                            result.skipped_files += 1
                        if progress and task_id is not None:
                            progress.advance(task_id)
                        return

                # Download from content-addressed storage
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(
                    self.bucket,
                    self._weight_file_key(remote_hash),
                    str(local_path),
                    Config=transfer_config,
                )
                with lock:
                    result.downloaded_files += 1
                    result.downloaded_bytes += local_path.stat().st_size
            except Exception as e:
                with lock:
                    result.errors.append(f"{rel_path}: {e}")

            if progress and task_id is not None:
                progress.advance(task_id)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_download_one, f) for f in files]
            for future in as_completed(futures):
                future.result()

        return result

    def list_weight_snapshots(self) -> list[WeightSnapshotInfo]:
        """List all weight snapshots backed up in S3.

        Returns:
            List of WeightSnapshotInfo with name, created date, file count, size.
        """
        snapshots: list[WeightSnapshotInfo] = []

        # Check "latest" manifest
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self._weight_manifest_key(None),
            )
            manifest = json.loads(response["Body"].read().decode("utf-8"))
            snapshots.append(
                WeightSnapshotInfo(
                    name="latest",
                    created=manifest.get("created", ""),
                    file_count=manifest.get("file_count", 0),
                    total_size_bytes=manifest.get("total_size_bytes", 0),
                )
            )
        except ClientError:
            pass  # No latest manifest

        # List named snapshots
        prefix = f"{self.prefix}/weights/snapshots/"
        paginator = self.s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
            Delimiter="/",
        )

        snapshot_names: list[str] = []
        for page in pages:
            for cp in page.get("CommonPrefixes", []):
                snap_name = cp["Prefix"][len(prefix) :].rstrip("/")
                if snap_name:
                    snapshot_names.append(snap_name)

        for snap_name in sorted(snapshot_names):
            try:
                response = self.s3.get_object(
                    Bucket=self.bucket,
                    Key=self._weight_manifest_key(snap_name),
                )
                manifest = json.loads(response["Body"].read().decode("utf-8"))
                snapshots.append(
                    WeightSnapshotInfo(
                        name=snap_name,
                        created=manifest.get("created", ""),
                        file_count=manifest.get("file_count", 0),
                        total_size_bytes=manifest.get("total_size_bytes", 0),
                    )
                )
            except ClientError:
                snapshots.append(WeightSnapshotInfo(name=snap_name))

        return snapshots

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
