"""Prune orphaned and superseded objects from the training S3 backup.

Complements backup.py:
- `backup.py` uploads datasets, videos, and weight files with dedup.
- `prune.py` removes snapshots, datasets, GT revisions, and orphan dedup pool
  entries that are no longer referenced.

All operations are batched (up to 1000 keys per delete_objects call) and
report what they would delete without an explicit execute=True.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from botocore.exceptions import ClientError

from rallycut.training.backup import DatasetBackup


@dataclass
class PruneResult:
    """Result of a single prune step."""

    step: str
    deleted_keys: list[tuple[str, int]] = field(default_factory=list)  # (key, size)
    errors: list[str] = field(default_factory=list)
    skipped_reason: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(sz for _, sz in self.deleted_keys)

    @property
    def count(self) -> int:
        return len(self.deleted_keys)


def _list_subprefixes(backup: DatasetBackup, prefix: str) -> list[str]:
    """List top-level subprefix names under an S3 prefix (using Delimiter='/')."""
    paginator = backup.s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=backup.bucket, Prefix=prefix, Delimiter="/")
    names: list[str] = []
    for page in pages:
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"][len(prefix):].rstrip("/")
            if name:
                names.append(name)
    return names


def _iter_all_objects(backup: DatasetBackup, prefix: str) -> Iterator[dict]:
    """Paginate every object under a prefix (flat)."""
    paginator = backup.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=backup.bucket, Prefix=prefix):
        yield from page.get("Contents", [])


def _read_manifest(backup: DatasetBackup, key: str) -> dict:
    """Download and decode a JSON manifest from S3."""
    response = backup.s3.get_object(Bucket=backup.bucket, Key=key)
    manifest: dict = json.loads(response["Body"].read().decode("utf-8"))
    return manifest


def _batch_delete(
    backup: DatasetBackup,
    targets: list[tuple[str, int]],
    execute: bool,
) -> list[str]:
    """Delete up to 1000 keys per API call. Returns error strings."""
    errors: list[str] = []
    if not targets or not execute:
        return errors
    for i in range(0, len(targets), 1000):
        batch = targets[i : i + 1000]
        response = backup.s3.delete_objects(
            Bucket=backup.bucket,
            Delete={"Objects": [{"Key": k} for k, _ in batch], "Quiet": True},
        )
        for err in response.get("Errors", []):
            errors.append(f"{err.get('Key')}: {err.get('Message')}")
    return errors


def prune_snapshots(
    backup: DatasetBackup,
    keep: set[str],
    *,
    execute: bool,
) -> PruneResult:
    """Delete weight snapshot manifests not in the keep set."""
    result = PruneResult(step="snapshots")
    root = f"{backup.prefix}/weights/snapshots/"
    names = _list_subprefixes(backup, root)
    if not names:
        result.skipped_reason = "no snapshots found"
        return result
    surviving = [n for n in names if n in keep]
    if not surviving:
        result.skipped_reason = f"refusing to delete all snapshots (keep={sorted(keep)})"
        return result
    for name in names:
        if name in keep:
            continue
        for obj in _iter_all_objects(backup, f"{root}{name}/"):
            result.deleted_keys.append((obj["Key"], obj["Size"]))
    result.errors.extend(_batch_delete(backup, result.deleted_keys, execute))
    return result


def prune_datasets(
    backup: DatasetBackup,
    keep: set[str],
    *,
    execute: bool,
) -> PruneResult:
    """Delete dataset subprefixes not in the keep set (recursive)."""
    result = PruneResult(step="datasets")
    root = f"{backup.prefix}/datasets/"
    names = _list_subprefixes(backup, root)
    if not names:
        result.skipped_reason = "no datasets found"
        return result
    surviving = [n for n in names if n in keep]
    if not surviving:
        result.skipped_reason = f"refusing to delete all datasets (keep={sorted(keep)})"
        return result
    for name in names:
        if name in keep:
            continue
        for obj in _iter_all_objects(backup, f"{root}{name}/"):
            result.deleted_keys.append((obj["Key"], obj["Size"]))
    result.errors.extend(_batch_delete(backup, result.deleted_keys, execute))
    return result


def prune_player_matching_gt(
    backup: DatasetBackup,
    *,
    keep_latest: int = 1,
    execute: bool,
) -> PruneResult:
    """Keep the newest N player_matching_gt JSON files, delete the rest.

    Filenames are timestamped (YYYY-MM-DD-HHMMSS.json) so lex order = chronological.
    """
    result = PruneResult(step="player_matching_gt")
    root = f"{backup.prefix}/player_matching_gt/"
    objects = [
        obj for obj in _iter_all_objects(backup, root)
        if obj["Key"].endswith(".json")
    ]
    if len(objects) <= keep_latest:
        result.skipped_reason = f"only {len(objects)} object(s), keep_latest={keep_latest}"
        return result
    objects.sort(key=lambda o: o["Key"])
    for obj in objects[:-keep_latest]:
        result.deleted_keys.append((obj["Key"], obj["Size"]))
    result.errors.extend(_batch_delete(backup, result.deleted_keys, execute))
    return result


def prune_wasb_images(backup: DatasetBackup, *, execute: bool) -> PruneResult:
    """Delete every object under wasb_gt_labels/images/ (sibling CSVs untouched)."""
    result = PruneResult(step="wasb_images")
    root = f"{backup.prefix}/wasb_gt_labels/images/"
    for obj in _iter_all_objects(backup, root):
        result.deleted_keys.append((obj["Key"], obj["Size"]))
    if not result.deleted_keys:
        result.skipped_reason = "no wasb image objects found"
        return result
    result.errors.extend(_batch_delete(backup, result.deleted_keys, execute))
    return result


def prune_orphan_weight_files(
    backup: DatasetBackup,
    *,
    execute: bool,
) -> PruneResult:
    """Delete weight dedup-pool entries not referenced by any surviving manifest.

    Must run AFTER prune_snapshots so the surviving snapshot set is correct.

    Note: when chained with prune_snapshots in the same CLI invocation, the
    dry-run orphan set will be smaller than the execute-mode set, because
    dry-run leaves old snapshot manifests in place (their hashes still count
    as referenced). The execute run recomputes orphans after snapshots are
    physically deleted.
    """
    result = PruneResult(step="orphan_weight_files")

    referenced: set[str] = set()

    # Latest manifest
    latest_key = f"{backup.prefix}/weights/manifest.json"
    try:
        manifest = _read_manifest(backup, latest_key)
        for entry in manifest.get("files", []):
            h = str(entry.get("content_hash", "")).removeprefix("sha256:")
            if h:
                referenced.add(h)
    except ClientError as e:
        result.errors.append(f"{latest_key}: {e}")
        result.skipped_reason = "missing latest manifest — refusing to GC"
        return result

    # Named snapshots. A missing/unreadable snapshot manifest must abort — otherwise
    # we could drop weight files that only the failing snapshot references.
    snap_root = f"{backup.prefix}/weights/snapshots/"
    snap_names = _list_subprefixes(backup, snap_root)
    for name in snap_names:
        snap_key = f"{snap_root}{name}/manifest.json"
        try:
            manifest = _read_manifest(backup, snap_key)
        except ClientError as e:
            result.errors.append(f"{snap_key}: {e}")
            result.skipped_reason = f"unreadable snapshot manifest {snap_key} — refusing to GC"
            return result
        for entry in manifest.get("files", []):
            h = str(entry.get("content_hash", "")).removeprefix("sha256:")
            if h:
                referenced.add(h)

    if not referenced:
        result.skipped_reason = "referenced set empty — refusing to GC"
        return result

    # Walk dedup pool
    pool_root = f"{backup.prefix}/weights/files/"
    for obj in _iter_all_objects(backup, pool_root):
        basename = obj["Key"].rsplit("/", 1)[-1]
        if basename not in referenced:
            result.deleted_keys.append((obj["Key"], obj["Size"]))

    result.errors.extend(_batch_delete(backup, result.deleted_keys, execute))
    return result


def prune_orphan_training_videos(
    backup: DatasetBackup,
    *,
    execute: bool,
) -> PruneResult:
    """Delete training videos not referenced by any surviving dataset manifest.

    Must run AFTER prune_datasets so the surviving dataset set is correct.
    Same dry-run vs execute caveat as prune_orphan_weight_files.
    """
    result = PruneResult(step="orphan_training_videos")

    referenced: set[str] = set()
    ds_root = f"{backup.prefix}/datasets/"
    ds_names = _list_subprefixes(backup, ds_root)
    if not ds_names:
        result.skipped_reason = "no surviving datasets — refusing to GC"
        return result

    for name in ds_names:
        manifest_key = f"{ds_root}{name}/manifest.json"
        try:
            manifest = _read_manifest(backup, manifest_key)
            for v in manifest.get("videos", []):
                h = str(v.get("content_hash", "")).removeprefix("sha256:")
                if h:
                    referenced.add(h)
        except ClientError as e:
            result.errors.append(f"{manifest_key}: {e}")
            result.skipped_reason = f"missing manifest {manifest_key} — refusing to GC"
            return result

    if not referenced:
        result.skipped_reason = "no video hashes referenced — refusing to GC"
        return result

    vid_root = f"{backup.prefix}/videos/"
    for obj in _iter_all_objects(backup, vid_root):
        basename = obj["Key"].rsplit("/", 1)[-1]
        content_hash = basename.removesuffix(".mp4")
        if content_hash not in referenced:
            result.deleted_keys.append((obj["Key"], obj["Size"]))

    result.errors.extend(_batch_delete(backup, result.deleted_keys, execute))
    return result
