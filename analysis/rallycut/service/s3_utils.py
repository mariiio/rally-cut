"""Shared S3 utilities for service runners."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import boto3


def create_s3_client() -> Any:
    """Create an S3 client with optional MinIO endpoint support."""
    s3_config: dict = {
        "region_name": os.environ.get("AWS_REGION", "us-east-1"),
    }
    # Only pass explicit credentials if set; otherwise let boto3 use its
    # default credential chain (env vars, instance roles, ~/.aws/).
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        s3_config["aws_access_key_id"] = access_key
        s3_config["aws_secret_access_key"] = secret_key
    if os.environ.get("S3_ENDPOINT"):
        s3_config["endpoint_url"] = os.environ["S3_ENDPOINT"]
    return boto3.client("s3", **s3_config)


def download_from_s3(
    s3_key: str,
    bucket: str,
    dest: Path,
    *,
    max_retries: int = 3,
    label: str = "",
) -> Path:
    """Download a file from S3 with retry logic.

    Args:
        s3_key: S3 object key
        bucket: S3 bucket name
        dest: Destination path (file or directory). If directory, uses filename from s3_key.
        max_retries: Number of retry attempts (exponential backoff 1s/2s/4s)
        label: Optional log prefix (e.g. "[LOCAL]")

    Returns:
        Path to the downloaded file
    """
    s3 = create_s3_client()

    if dest.is_dir():
        local_path = dest / Path(s3_key).name
    else:
        local_path = dest

    prefix = f"{label} " if label else ""
    print(f"{prefix}Downloading s3://{bucket}/{s3_key} to {local_path}")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            s3.download_file(bucket, s3_key, str(local_path))
            return local_path
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
                print(f"{prefix}S3 download failed (attempt {attempt}/{max_retries}), retrying in {delay}s: {e}")
                time.sleep(delay)

    raise RuntimeError(f"S3 download failed after {max_retries} attempts: {last_error}")
