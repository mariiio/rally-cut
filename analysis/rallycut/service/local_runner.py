"""Local detection runner for development.

This script runs detection locally and calls the webhook when complete,
just like Modal would in production. Used for local development where
Modal can't callback to localhost.

Usage:
    python -m rallycut.service.local_runner \
        --job-id <uuid> \
        --video-path <local_path_or_s3_key> \
        --callback-url http://localhost:3001/v1/webhooks/detection-complete \
        --webhook-secret <secret>
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import boto3
import httpx


def download_from_s3(s3_key: str, bucket: str, temp_dir: Path) -> Path:
    """Download video from S3 to temp directory."""
    s3_config = {
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "region_name": os.environ.get("AWS_REGION", "us-east-1"),
    }
    # Support MinIO/local S3 endpoint
    if os.environ.get("S3_ENDPOINT"):
        s3_config["endpoint_url"] = os.environ["S3_ENDPOINT"]
    s3 = boto3.client("s3", **s3_config)

    filename = Path(s3_key).name
    local_path = temp_dir / filename

    print(f"[LOCAL] Downloading s3://{bucket}/{s3_key} to {local_path}")
    s3.download_file(bucket, s3_key, str(local_path))
    return local_path


def send_progress(callback_url: str, webhook_secret: str | None, job_id: str, progress: float, message: str) -> None:
    """Send progress update to the API."""
    print(f"[{progress*100:.0f}%] {message}")

    # Build progress endpoint URL from callback URL
    progress_url = callback_url.replace("detection-complete", "detection-progress")

    headers = {"Content-Type": "application/json"}
    if webhook_secret:
        headers["X-Webhook-Secret"] = webhook_secret

    payload = {
        "job_id": job_id,
        "progress": round(progress * 100, 1),  # Convert to 0-100
        "message": message,
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(progress_url, json=payload, headers=headers)
    except Exception:
        pass  # Don't fail on progress update errors


def send_webhook(
    callback_url: str,
    webhook_secret: str | None,
    payload: dict,
) -> bool:
    """Send webhook with results."""
    headers = {"Content-Type": "application/json"}
    if webhook_secret:
        headers["X-Webhook-Secret"] = webhook_secret

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(callback_url, json=payload, headers=headers)
            response.raise_for_status()
            print(f"[LOCAL] Webhook sent successfully to {callback_url}")
            return True
    except Exception as e:
        print(f"[LOCAL] Webhook failed: {e}")
        return False


def save_results_to_s3(
    results: dict,
    video_s3_key: str,
    bucket: str,
) -> str | None:
    """Save detection results JSON to S3 alongside the video."""
    try:
        s3_config = {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "region_name": os.environ.get("AWS_REGION", "us-east-1"),
        }
        # Support MinIO/local S3 endpoint
        if os.environ.get("S3_ENDPOINT"):
            s3_config["endpoint_url"] = os.environ["S3_ENDPOINT"]
        s3 = boto3.client("s3", **s3_config)

        # Store results in same folder as video
        video_dir = str(Path(video_s3_key).parent)
        results_key = f"{video_dir}/detection-results.json"

        # Upload JSON
        s3.put_object(
            Bucket=bucket,
            Key=results_key,
            Body=json.dumps(results, indent=2),
            ContentType="application/json",
        )
        print(f"[LOCAL] Saved results to s3://{bucket}/{results_key}")
        return results_key
    except Exception as e:
        print(f"[LOCAL] Failed to save results to S3: {e}")
        return None


def run_detection(
    job_id: str,
    video_path: str,
    callback_url: str,
    webhook_secret: str | None,
    s3_bucket: str | None = None,
    model_variant: str = "indoor",
) -> None:
    """Run detection and send results via webhook."""
    start_time = time.time()
    temp_dir = None
    local_video_path = None
    video_s3_key = video_path  # Store original S3 key for results upload

    try:
        # Import detection service
        from rallycut.service.detection import DetectionService
        from rallycut.service.schemas import DetectionConfig, DetectionRequest

        # Determine if we need to download from S3
        if video_path.startswith("s3://") or (s3_bucket and not Path(video_path).exists()):
            # Download from S3
            if video_path.startswith("s3://"):
                # Parse s3://bucket/key format
                parts = video_path[5:].split("/", 1)
                s3_bucket = parts[0]
                video_s3_key = parts[1] if len(parts) > 1 else ""
            else:
                video_s3_key = video_path

            temp_dir = Path(tempfile.mkdtemp(prefix="rallycut_"))
            local_video_path = download_from_s3(video_s3_key, s3_bucket or "", temp_dir)
        else:
            local_video_path = Path(video_path)

        if not local_video_path.exists():
            raise FileNotFoundError(f"Video not found: {local_video_path}")

        print(f"[LOCAL] Starting detection for {local_video_path}")

        # Use CPU for local development (MPS on Mac, CUDA if available)
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"[LOCAL] Using device: {device}")
        print(f"[LOCAL] Using model: {model_variant}")

        # Run detection
        service = DetectionService(device=device, temp_dir=temp_dir or Path("/tmp/rallycut"))

        # Import ModelVariant enum for config
        from rallycut.service.schemas import ModelVariant
        model_enum = ModelVariant(model_variant)

        config = DetectionConfig(
            model_variant=model_enum,
            min_play_duration=5.0,  # Minimum 5 seconds for a rally
            use_proxy=True,  # Use 480p proxy for faster processing
        )

        request = DetectionRequest(
            video_url=f"file://{local_video_path}",
            job_id=job_id,
            config=config,
        )

        def progress_callback(pct: float, msg: str) -> None:
            send_progress(callback_url, webhook_secret, job_id, pct, msg)

        response = service.detect(request, progress_callback)

        # Transform to API webhook format
        rallies = []
        for segment in response.segments:
            if segment.segment_type.value == "rally":
                rallies.append({
                    "start_ms": int(segment.start_time * 1000),
                    "end_ms": int(segment.end_time * 1000),
                })

        # Include suggested rallies (segments that almost passed detection)
        suggested_rallies = []
        for sugg in response.suggested_segments:
            suggested_rallies.append({
                "start_ms": int(sugg.start_time * 1000),
                "end_ms": int(sugg.end_time * 1000),
                "confidence": sugg.avg_confidence,
                "rejection_reason": sugg.rejection_reason.value,
            })

        # Build full results for S3 storage
        full_results = {
            "job_id": job_id,
            "video_key": video_s3_key,
            "processing_time_seconds": time.time() - start_time,
            "device": device,
            "segments": [
                {
                    "segment_id": seg.segment_id,
                    "segment_type": seg.segment_type.value,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "duration": seg.duration,
                }
                for seg in response.segments
            ],
            "statistics": response.statistics.model_dump() if response.statistics else None,
            "video_metadata": response.video.model_dump() if response.video else None,
        }

        # Save full results to S3
        result_s3_key = None
        if s3_bucket:
            result_s3_key = save_results_to_s3(full_results, video_s3_key, s3_bucket)

        # Build webhook payload (simplified format for API)
        webhook_payload = {
            "job_id": job_id,
            "status": "completed",
            "rallies": rallies,
            "suggested_rallies": suggested_rallies,
            "result_s3_key": result_s3_key,
        }

        print(f"[LOCAL] Detection complete: {len(rallies)} rallies found")
        print(f"[LOCAL] Processing time: {time.time() - start_time:.1f}s")

        # Send webhook
        send_webhook(callback_url, webhook_secret, webhook_payload)

    except Exception as e:
        print(f"[LOCAL] Detection failed: {e}")
        import traceback
        traceback.print_exc()

        # Send error webhook
        error_payload = {
            "job_id": job_id,
            "status": "failed",
            "error_message": str(e),
        }
        send_webhook(callback_url, webhook_secret, error_payload)

    finally:
        # Cleanup temp directory
        if temp_dir and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rally detection locally")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--video-path", required=True, help="Local path or S3 key")
    parser.add_argument("--callback-url", required=True, help="Webhook callback URL")
    parser.add_argument("--webhook-secret", help="Webhook secret for auth")
    parser.add_argument("--s3-bucket", help="S3 bucket name (for S3 downloads)")
    parser.add_argument(
        "--model",
        choices=["indoor", "beach"],
        default="beach",
        help="Model variant: 'beach' (default) or 'indoor' (original)",
    )

    args = parser.parse_args()

    run_detection(
        job_id=args.job_id,
        video_path=args.video_path,
        callback_url=args.callback_url,
        webhook_secret=args.webhook_secret,
        s3_bucket=args.s3_bucket,
        model_variant=args.model,
    )


if __name__ == "__main__":
    main()
