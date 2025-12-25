"""Cloud storage utilities for video download."""

import shutil
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import httpx


def download_video(
    url: str,
    temp_dir: Path,
    progress_callback: Callable[[float, str], None] | None = None,
    timeout: float = 600.0,
) -> Path:
    """
    Download video from cloud storage URL to local temp directory.

    Supports:
    - S3 presigned URLs (https://bucket.s3.amazonaws.com/...)
    - GCS signed URLs (https://storage.googleapis.com/...)
    - Public HTTP/HTTPS URLs

    Args:
        url: Video URL to download
        temp_dir: Local directory to save the video
        progress_callback: Optional callback for progress updates (progress 0-1, message)
        timeout: Download timeout in seconds

    Returns:
        Local path to downloaded video file

    Raises:
        httpx.HTTPError: If download fails
    """
    parsed = urlparse(url)
    # Extract filename from path, removing query params
    path_part = parsed.path.split("/")[-1] if parsed.path else "video.mp4"
    # Remove any query string from filename
    filename = path_part.split("?")[0] or "video.mp4"
    local_path = temp_dir / filename

    if progress_callback:
        progress_callback(0.0, f"Downloading {filename}...")

    # Stream download with progress tracking
    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=httpx.Timeout(timeout, connect=30.0),
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(local_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)

                if progress_callback and total > 0:
                    progress = downloaded / total
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    progress_callback(
                        progress,
                        f"Downloading: {mb_downloaded:.1f}MB / {mb_total:.1f}MB",
                    )

    if progress_callback:
        progress_callback(1.0, "Download complete")

    return local_path


def cleanup_temp(temp_dir: Path) -> None:
    """
    Remove all files and subdirectories in temp directory.

    Args:
        temp_dir: Directory to clean up
    """
    if not temp_dir.exists():
        return

    for item in temp_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
