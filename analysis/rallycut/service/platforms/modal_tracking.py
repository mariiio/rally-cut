"""Modal deployment for RallyCut batch player tracking service.

Separate app from detection — different deps, different timeout, independent deploys.

Deploy with:
    modal deploy rallycut/service/platforms/modal_tracking.py

Environment setup:
    1. pip install modal
    2. modal token new
    3. modal secret create aws-credentials \
         AWS_ACCESS_KEY_ID=xxx \
         AWS_SECRET_ACCESS_KEY=xxx \
         AWS_REGION=us-east-1
"""

from __future__ import annotations

import modal

app = modal.App("rallycut-tracking")

# Container image extends detection image with tracking deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        # Core ML
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        # Player tracking (YOLO + BoT-SORT)
        "ultralytics>=8.2.0",
        "lap>=0.5.12",
        "scikit-learn>=1.3.0",
        # Video processing
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "ffmpeg-python>=0.2.0",
        # API and config
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "httpx>=0.25.0",
        "pyyaml>=6.0.0",
        "typer>=0.9.0",
        # Utilities
        "tqdm>=4.65.0",
        "scipy>=1.11.0",
        # ONNX inference (GPU-accelerated for WASB ball tracker)
        "onnxruntime-gpu>=1.17.0",
        # AWS S3 access
        "boto3>=1.34.0",
        # Web endpoint
        "fastapi",
    )
    # Pre-download YOLO11s weights during image build
    .run_commands("python -c \"from ultralytics import YOLO; YOLO('yolo11s.pt')\"")
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir("rallycut", "/app/rallycut")
    .add_local_dir("lib", "/app/lib")
    .add_local_dir("weights", "/app/weights")
)


@app.function(image=image)
@modal.fastapi_endpoint(method="GET", docs=True)
def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "rallycut-tracking"}


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,  # 60 min max for large batches
    memory=16384,  # 16GB RAM
    secrets=[modal.Secret.from_name("aws-credentials")],
)
@modal.fastapi_endpoint(method="POST", docs=True)
def track_batch(request: dict) -> dict:
    """
    Batch player tracking endpoint. Downloads video once, tracks all rallies.

    Expected request body:
    {
        "batch_job_id": "uuid",
        "video_id": "uuid",
        "video_key": "s3/path/to/video.mp4",
        "rallies": [{"id": "uuid", "start_ms": 0, "end_ms": 15000}, ...],
        "calibration_corners": [{"x": 0.1, "y": 0.8}, ...] or null,
        "callback_url": "https://api.example.com/v1/webhooks",
        "webhook_secret": "secret",
        "s3_bucket": "rallycut-dev"
    }
    """
    import os
    import tempfile
    import time as _time

    import boto3
    import httpx

    batch_job_id = request.get("batch_job_id")
    video_id = request.get("video_id")
    # API sends original-quality key — tracking needs full resolution for accurate
    # ball detection (720p proxy degrades VballNet/WASB significantly)
    video_key = request.get("video_key")
    rallies = request.get("rallies", [])
    calibration_corners = request.get("calibration_corners")
    callback_url = request.get("callback_url")
    webhook_secret = request.get("webhook_secret")
    s3_bucket = request.get("s3_bucket") or os.environ["S3_BUCKET_NAME"]

    if not batch_job_id or not video_id or not video_key or not callback_url:
        return {"error": "Missing required fields: batch_job_id, video_id, video_key, callback_url"}

    if not rallies:
        return {"error": "No rallies to track"}

    print(f"Starting batch tracking job {batch_job_id}")
    print(f"  Video: {video_id} ({video_key})")
    print(f"  Rallies: {len(rallies)}")
    print(f"  Callback URL: {callback_url}")

    headers = {"Content-Type": "application/json"}
    if webhook_secret:
        headers["X-Webhook-Secret"] = webhook_secret

    completed_count = 0
    failed_count = 0
    video_path = None

    try:
        # Download video from S3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name

        print(f"Downloading s3://{s3_bucket}/{video_key} to {video_path}")
        for attempt in range(1, 4):
            try:
                s3.download_file(s3_bucket, video_key, video_path)
                break
            except Exception as dl_err:
                if attempt < 3:
                    delay = 2 ** (attempt - 1)
                    print(f"S3 download failed (attempt {attempt}/3), retrying in {delay}s: {dl_err}")
                    _time.sleep(delay)
                else:
                    raise
        print("Video downloaded successfully")

        # Process each rally
        for rally in rallies:
            rally_id = rally["id"]
            start_ms = rally["start_ms"]
            end_ms = rally["end_ms"]
            start_s = start_ms / 1000
            duration_s = (end_ms - start_ms) / 1000

            print(f"\n--- Rally {rally_id}: {start_s:.1f}s - {end_ms / 1000:.1f}s ({duration_s:.1f}s) ---")

            rally_result = _track_single_rally(
                video_path=video_path,
                rally_id=rally_id,
                start_s=start_s,
                duration_s=duration_s,
                calibration_corners=calibration_corners,
            )

            # Send per-rally webhook
            rally_payload = {
                "batch_job_id": batch_job_id,
                "video_id": video_id,
                "rally_id": rally_id,
                **rally_result,
            }

            webhook_ok = False
            for wh_attempt in range(1, 4):
                try:
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.post(
                            f"{callback_url}/tracking-rally-complete",
                            json=rally_payload,
                            headers=headers,
                        )
                        resp.raise_for_status()
                        webhook_ok = True
                        print(f"Rally webhook sent: {rally_result['status']}")
                        break
                except Exception as e:
                    if wh_attempt < 3:
                        delay = 3 ** (wh_attempt - 1)  # 1s, 3s
                        print(f"Rally webhook failed (attempt {wh_attempt}/3), retrying in {delay}s: {e}")
                        _time.sleep(delay)
                    else:
                        print(f"Rally webhook failed after 3 attempts: {e}")

            if rally_result["status"] == "completed" and webhook_ok:
                completed_count += 1
            else:
                failed_count += 1

    except Exception as e:
        print(f"Batch tracking failed: {e}")
        import traceback

        traceback.print_exc()

        # Send batch-complete with error
        try:
            with httpx.Client(timeout=30.0) as client:
                client.post(
                    f"{callback_url}/tracking-batch-complete",
                    json={
                        "batch_job_id": batch_job_id,
                        "video_id": video_id,
                        "status": "failed",
                        "completed_rallies": completed_count,
                        "failed_rallies": failed_count,
                        "error": str(e),
                    },
                    headers=headers,
                )
        except Exception:
            pass

        return {
            "status": "failed",
            "batch_job_id": batch_job_id,
            "error": str(e),
        }
    finally:
        # Clean up downloaded video
        if video_path:
            try:
                os.unlink(video_path)
            except OSError as e:
                print(f"Failed to unlink temp file {video_path}: {e}")

    # Send batch-complete webhook
    batch_status = "failed" if failed_count == len(rallies) else "completed"
    batch_payload = {
        "batch_job_id": batch_job_id,
        "video_id": video_id,
        "status": batch_status,
        "completed_rallies": completed_count,
        "failed_rallies": failed_count,
    }
    if failed_count > 0:
        batch_payload["error"] = f"{failed_count}/{len(rallies)} rallies failed"

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{callback_url}/tracking-batch-complete",
                json=batch_payload,
                headers=headers,
            )
            resp.raise_for_status()
            print(f"\nBatch webhook sent: {batch_status}")
    except Exception as e:
        print(f"Batch webhook failed: {e}")

    print(f"\nBatch complete: {completed_count} completed, {failed_count} failed")
    return {
        "status": batch_status,
        "batch_job_id": batch_job_id,
        "completed_rallies": completed_count,
        "failed_rallies": failed_count,
    }


def _track_single_rally(
    video_path: str,
    rally_id: str,
    start_s: float,
    duration_s: float,
    calibration_corners: list[dict] | None,
) -> dict:
    """Extract segment and run track-players CLI. Returns result dict."""
    import json
    import os
    import subprocess
    import tempfile

    segment_path = None
    output_path = None

    try:
        # Create temp files
        segment_fd, segment_path = tempfile.mkstemp(suffix=".mp4")
        os.close(segment_fd)
        output_fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(output_fd)

        # Extract segment with FFmpeg (stream copy — no re-encoding needed for tracking)
        ffmpeg_cmd = [
            "ffmpeg",
            "-ss", str(start_s),
            "-i", video_path,
            "-t", str(duration_s),
            "-c:v", "copy",
            "-an",
            "-y",
            segment_path,
        ]
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")

        # Run track-players CLI
        cli_cmd = [
            "python", "-m", "rallycut.cli.main",
            "track-players",
            segment_path,
            "--output", output_path,
            "--filter",
            "--actions",
            "--quiet",
        ]
        if calibration_corners and len(calibration_corners) == 4:
            cli_cmd.extend(["--calibration", json.dumps(calibration_corners)])

        print(f"Running: {' '.join(cli_cmd)}")
        result = subprocess.run(
            cli_cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per rally max
            cwd="/app",
            env={**os.environ, "PYTHONPATH": "/app"},
        )

        if result.returncode != 0:
            output = result.stderr or result.stdout
            raise RuntimeError(f"track-players failed (exit {result.returncode}): {output[-1000:]}")

        # Parse output JSON
        with open(output_path) as f:
            tracking_data = json.load(f)

        print(
            f"Rally {rally_id}: {tracking_data.get('frameCount', 0)} frames, "
            f"{len(tracking_data.get('positions', []))} positions, "
            f"{len(tracking_data.get('ballPositions', []))} ball positions"
        )

        return {
            "status": "completed",
            "tracking_data": tracking_data,
        }

    except Exception as e:
        print(f"Rally {rally_id} failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }
    finally:
        if segment_path:
            try:
                os.unlink(segment_path)
            except OSError as e:
                print(f"Failed to unlink temp file {segment_path}: {e}")
        if output_path:
            try:
                os.unlink(output_path)
            except OSError as e:
                print(f"Failed to unlink temp file {output_path}: {e}")
