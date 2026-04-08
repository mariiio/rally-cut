"""Push a committed player_matching_gt backup to S3.

Durable off-site copy of the GT backup in case the repo is lost or the
backup file is deleted. Uses the same S3 bucket/credentials as the
training backup system (TRAINING_S3_BUCKET, default AWS credential chain).

The target key is:

    <prefix>/player_matching_gt/<filename>

where <prefix> defaults to `training` (matching rallycut.core.config
TrainingBackupConfig.s3_prefix).

Usage:
    uv run python analysis/scripts/push_gt_backup_to_s3.py <backup.json>
    uv run python analysis/scripts/push_gt_backup_to_s3.py <backup.json> --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from rallycut.core.config import get_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path, help="Backup JSON file to upload")
    parser.add_argument("--bucket", default=None,
                        help="Override S3 bucket (else TRAINING_S3_BUCKET / config)")
    parser.add_argument("--prefix", default=None,
                        help="Override S3 key prefix (else config s3_prefix)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the upload plan without touching S3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.backup.exists():
        logger.error("Backup file not found: %s", args.backup)
        sys.exit(1)

    cfg = get_config().training_backup
    bucket = args.bucket or os.getenv("TRAINING_S3_BUCKET", "") or cfg.s3_bucket
    prefix = args.prefix or os.getenv("TRAINING_S3_PREFIX", "") or cfg.s3_prefix
    region = os.getenv("TRAINING_S3_REGION", "") or cfg.s3_region

    if not bucket:
        logger.error(
            "No S3 bucket configured. Set TRAINING_S3_BUCKET, pass --bucket, "
            "or configure training_backup.s3_bucket in rallycut.yaml."
        )
        sys.exit(1)

    key = f"{prefix.rstrip('/')}/player_matching_gt/{args.backup.name}"
    size = args.backup.stat().st_size
    logger.info("Backup:  %s  (%d bytes)", args.backup, size)
    logger.info("Target:  s3://%s/%s  (region=%s)", bucket, key, region)

    if args.dry_run:
        logger.info("DRY RUN — pass without --dry-run to actually upload.")
        return

    s3 = boto3.client(
        "s3",
        config=BotoConfig(signature_version="s3v4"),
        region_name=region,
    )

    # Idempotence: if the key already exists with the same size, skip.
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        existing_size = int(head.get("ContentLength", -1))
        if existing_size == size:
            logger.info("Already present with matching size — nothing to do.")
            return
        logger.info(
            "Key exists with different size (%d → %d); overwriting.",
            existing_size, size,
        )
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "404":
            raise

    s3.upload_file(
        str(args.backup), bucket, key,
        ExtraArgs={
            "ContentType": "application/json",
            "ServerSideEncryption": "AES256",
        },
    )
    logger.info("Uploaded.")


if __name__ == "__main__":
    main()
