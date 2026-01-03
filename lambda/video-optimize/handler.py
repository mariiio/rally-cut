"""
Video Optimization Lambda Handler

Optimizes uploaded videos for web playback:
1. Downloads original video from S3
2. Checks if optimization is needed (moov atom location, bitrate)
3. Re-encodes with H.264, CRF 23, faststart
4. Uploads optimized version to S3
5. Sends webhook notification
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import boto3
import requests

s3_client = boto3.client("s3")


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main Lambda entry point."""
    print(f"Received event: {json.dumps(event)}")

    video_id = event["videoId"]
    original_s3_key = event["originalS3Key"]
    s3_bucket = event["s3Bucket"]
    callback_url = event["callbackUrl"]
    webhook_secret = event["webhookSecret"]
    tier = event.get("tier", "FREE")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = process_video(
                video_id=video_id,
                original_s3_key=original_s3_key,
                s3_bucket=s3_bucket,
                tmpdir=Path(tmpdir),
                tier=tier,
            )

        # Notify success
        send_webhook(
            callback_url,
            webhook_secret,
            {
                "video_id": video_id,
                "status": result["status"],
                "processed_s3_key": result.get("processed_s3_key"),
                "processed_size_bytes": result.get("processed_size_bytes"),
                "poster_s3_key": result.get("poster_s3_key"),
                "proxy_s3_key": result.get("proxy_s3_key"),
                "proxy_size_bytes": result.get("proxy_size_bytes"),
                "was_optimized": result.get("was_optimized", False),
            },
        )

        return {"statusCode": 200, "body": json.dumps({"success": True})}

    except Exception as e:
        print(f"Optimization failed: {e}")
        send_webhook(
            callback_url,
            webhook_secret,
            {
                "video_id": video_id,
                "status": "failed",
                "error_message": str(e),
            },
        )
        raise


def process_video(
    video_id: str,
    original_s3_key: str,
    s3_bucket: str,
    tmpdir: Path,
    tier: str,
) -> dict[str, Any]:
    """Process video and return result."""

    # Download original
    input_path = tmpdir / "input.mp4"
    print(f"Downloading {original_s3_key}...")
    s3_client.download_file(s3_bucket, original_s3_key, str(input_path))

    original_size = input_path.stat().st_size

    # Generate S3 key components for all outputs
    path_parts = original_s3_key.rsplit("/", 1)
    if len(path_parts) == 2:
        folder, filename = path_parts
    else:
        folder = ""
        filename = path_parts[0]

    # Split filename and extension
    if "." in filename:
        name, _ = filename.rsplit(".", 1)
    else:
        name = filename

    base_key = f"{folder}/{name}" if folder else name

    # Always generate poster (small cost, big UX win)
    poster_path = tmpdir / "poster.jpg"
    poster_key = f"{base_key}_poster.jpg"
    generate_poster(input_path, poster_path)
    print(f"Uploading poster to {poster_key}...")
    s3_client.upload_file(
        str(poster_path),
        s3_bucket,
        poster_key,
        ExtraArgs={
            "ContentType": "image/jpeg",
            "CacheControl": "public, max-age=31536000",  # 1 year
        },
    )

    # Check if optimization is needed
    if not needs_optimization(input_path):
        print("Video already optimized, skipping processing")
        return {
            "status": "skipped",
            "processed_s3_key": original_s3_key,
            "processed_size_bytes": original_size,
            "poster_s3_key": poster_key,
            "was_optimized": False,
        }

    # Optimize video
    output_path = tmpdir / "output.mp4"
    optimize_video(input_path, output_path, tier)

    processed_size = output_path.stat().st_size
    processed_key = f"{base_key}_optimized.mp4"

    print(f"Uploading optimized video to {processed_key}...")
    s3_client.upload_file(
        str(output_path),
        s3_bucket,
        processed_key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "CacheControl": "public, max-age=31536000",  # 1 year
        },
    )

    reduction = (1 - processed_size / original_size) * 100
    print(
        f"Optimization complete: {original_size} -> {processed_size} bytes "
        f"({reduction:.1f}% reduction)"
    )

    # Generate 720p proxy for fast editing (all tiers)
    proxy_path = tmpdir / "proxy.mp4"
    proxy_key = f"{base_key}_proxy.mp4"
    generate_proxy(output_path, proxy_path)
    proxy_size_bytes = proxy_path.stat().st_size
    print(f"Uploading proxy to {proxy_key}...")
    s3_client.upload_file(
        str(proxy_path),
        s3_bucket,
        proxy_key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "CacheControl": "public, max-age=31536000",
        },
    )

    return {
        "status": "completed",
        "processed_s3_key": processed_key,
        "processed_size_bytes": processed_size,
        "poster_s3_key": poster_key,
        "proxy_s3_key": proxy_key,
        "proxy_size_bytes": proxy_size_bytes,
        "was_optimized": True,
    }


def needs_optimization(video_path: Path) -> bool:
    """Check if video needs optimization (moov atom not at start, or high bitrate)."""
    try:
        # Check bitrate and codec info with ffprobe
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=bit_rate",
                "-show_entries", "stream=codec_name",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            info = json.loads(result.stdout)
            bit_rate = int(info.get("format", {}).get("bit_rate", 0) or 0)

            # If bitrate > 8 Mbps, optimize to reduce size
            if bit_rate > 8_000_000:
                print(f"High bitrate detected ({bit_rate / 1_000_000:.1f} Mbps), will optimize")
                return True

        # Check moov atom position by looking at first bytes
        # A fast-start file will have moov near the beginning
        with open(video_path, "rb") as f:
            # Read first 64KB
            header = f.read(65536)
            # Look for moov atom marker
            moov_pos = header.find(b"moov")
            if moov_pos == -1 or moov_pos > 32768:
                print(f"moov atom not at start (pos: {moov_pos}), will optimize")
                return True

        print("Video appears already optimized")
        return False

    except Exception as e:
        print(f"Error checking video: {e}, will optimize to be safe")
        return True


def optimize_video(input_path: Path, output_path: Path, tier: str) -> None:
    """Re-encode video with optimization settings."""

    # FFmpeg command for optimization
    # - CRF 23: visually lossless quality
    # - preset fast: good balance of speed/compression
    # - movflags +faststart: put moov atom at beginning
    # - tune film: optimized for typical video content
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-tune", "film",
        "-profile:v", "high",
        "-level", "4.1",  # Wide compatibility
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ac", "2",  # Stereo
        str(output_path),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"FFmpeg stderr: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")


def generate_poster(input_path: Path, output_path: Path) -> None:
    """Generate poster image from video (1 second in, 1280px width)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", "1",  # 1 second in
        "-i", str(input_path),
        "-vframes", "1",
        "-vf", "scale=1280:-1",
        "-q:v", "2",  # High quality JPEG
        str(output_path),
    ]

    print(f"Generating poster: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"Poster generation failed: {result.stderr}")
        raise RuntimeError(f"Poster generation failed: {result.stderr[-500:]}")


def generate_proxy(input_path: Path, output_path: Path) -> None:
    """Generate 720p proxy video for editing (lower bitrate, faster loading)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", "scale=-2:720",
        "-c:v", "libx264",
        "-crf", "28",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "96k",
        str(output_path),
    ]

    print(f"Generating proxy: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"Proxy generation failed: {result.stderr}")
        raise RuntimeError(f"Proxy generation failed: {result.stderr[-500:]}")


def send_webhook(url: str, secret: str, payload: dict[str, Any]) -> None:
    """Send webhook notification."""
    print(f"Sending webhook to {url}: {payload}")
    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                "X-Webhook-Secret": secret,
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        print(f"Webhook response: {response.status_code}")
    except Exception as e:
        print(f"Webhook failed: {e}")
