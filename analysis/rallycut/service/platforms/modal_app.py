"""Modal deployment for RallyCut detection service.

Deploy with:
    modal deploy rallycut/service/platforms/modal_app.py

Test locally:
    modal run rallycut/service/platforms/modal_app.py

Environment setup:
    1. pip install modal
    2. modal token new  # Authenticate with Modal
    3. Create Modal secrets for AWS credentials:
       modal secret create aws-credentials \
         AWS_ACCESS_KEY_ID=xxx \
         AWS_SECRET_ACCESS_KEY=xxx \
         AWS_REGION=us-east-1
"""

from __future__ import annotations

import modal

# Define the Modal app
app = modal.App("rallycut-detection")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        # Core ML
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        # Video processing
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "ffmpeg-python>=0.2.0",
        # API and config
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "httpx>=0.25.0",
        "pyyaml>=6.0.0",
        # Utilities
        "tqdm>=4.65.0",
        "scipy>=1.11.0",
        "filterpy>=1.4.5",
        # AWS S3 access
        "boto3>=1.34.0",
        # Web endpoint
        "fastapi",
    )
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir("rallycut", "/app/rallycut")
    .add_local_dir("lib", "/app/lib")
    .add_local_dir("weights", "/app/weights")
)

# Volume for model weights (optional - for larger models)
model_volume = modal.Volume.from_name("rallycut-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # Cost-effective GPU, sufficient for VideoMAE
    timeout=1800,  # 30 min max
    volumes={"/models": model_volume},
    memory=16384,  # 16GB RAM
)
def detect_rallies(
    job_id: str,
    video_url: str,
    callback_url: str,
    webhook_secret: str | None = None,
    config: dict | None = None,
) -> dict:
    """
    Core detection function - runs on GPU, posts result to callback URL.

    This function is called via .spawn() from the Vercel API.
    When complete, it POSTs the result to callback_url as a webhook.

    Args:
        job_id: Unique job identifier
        video_url: S3/GCS presigned URL to download video from
        callback_url: URL to POST results when complete
        webhook_secret: Secret token for webhook authentication
        config: Optional detection config dict

    Returns:
        DetectionResponse as dict (also sent to callback_url)
    """
    import sys

    import httpx

    sys.path.insert(0, "/app")

    from rallycut.service.detection import DetectionService
    from rallycut.service.schemas import DetectionConfig, DetectionRequest

    # Build request from parameters
    detection_config = DetectionConfig(**config) if config else None
    request = DetectionRequest(
        video_url=video_url,
        job_id=job_id,
        config=detection_config,
    )

    service = DetectionService(device="cuda")

    # Run detection with progress tracking
    def progress_callback(pct: float, msg: str) -> None:
        print(f"[{pct*100:.1f}%] {msg}")

    response = service.detect(request, progress_callback)
    result = response.model_dump(mode="json")

    # POST result to callback URL (webhook)
    if callback_url:
        headers = {"Content-Type": "application/json"}
        if webhook_secret:
            headers["X-Webhook-Secret"] = webhook_secret

        try:
            with httpx.Client(timeout=30.0) as client:
                webhook_response = client.post(callback_url, json=result, headers=headers)
                webhook_response.raise_for_status()
                print(f"Webhook sent successfully to {callback_url}")
        except Exception as e:
            print(f"Webhook failed: {e}")
            # Don't fail the job if webhook fails - result is still returned

    return result


@app.function(image=image)
@modal.web_endpoint(method="GET", docs=True)
def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "rallycut-detection"}


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/models": model_volume},
    memory=16384,
    secrets=[modal.Secret.from_name("aws-credentials")],
)
@modal.web_endpoint(method="POST", docs=True)
def detect(request: dict) -> dict:
    """
    Web endpoint for detection requests from the API.

    Expected request body:
    {
        "job_id": "string",
        "video_key": "string",  # S3 key for the video
        "callback_url": "string",
        "webhook_secret": "string" (optional),
        "config": {} (optional)
    }

    Returns immediately with job status, results sent via webhook.
    """
    import os
    import tempfile

    import boto3
    import httpx

    job_id = request.get("job_id")
    video_key = request.get("video_key")
    callback_url = request.get("callback_url")
    webhook_secret = request.get("webhook_secret")
    config = request.get("config")

    if not job_id or not video_key or not callback_url:
        return {"error": "Missing required fields: job_id, video_key, callback_url"}

    print(f"Starting detection job {job_id}")
    print(f"  Video key: {video_key}")
    print(f"  Callback URL: {callback_url}")

    try:
        # Download video from S3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )

        bucket = os.environ.get("S3_BUCKET_NAME", "rallycut-dev")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name
            print(f"Downloading s3://{bucket}/{video_key} to {temp_path}")
            s3.download_file(bucket, video_key, temp_path)

        # Run detection
        import sys

        sys.path.insert(0, "/app")

        from rallycut.service.detection import DetectionService
        from rallycut.service.schemas import DetectionConfig, DetectionRequest

        detection_config = DetectionConfig(**config) if config else None
        detection_request = DetectionRequest(
            video_url=f"file://{temp_path}",
            job_id=job_id,
            config=detection_config,
        )

        service = DetectionService(device="cuda")

        def progress_callback(pct: float, msg: str) -> None:
            print(f"[{pct*100:.1f}%] {msg}")

        response = service.detect(detection_request, progress_callback)

        # Clean up temp file
        os.unlink(temp_path)

        # Transform response to API webhook format
        # API expects: {job_id, status, rallies: [{start_ms, end_ms, confidence}], error_message}
        # DetectionResponse has: {job_id, status, segments: [{start_time, end_time}]}
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

        webhook_payload = {
            "job_id": job_id,
            "status": "completed",
            "rallies": rallies,
            "suggested_rallies": suggested_rallies,
        }

        # POST result to callback URL (webhook)
        headers = {"Content-Type": "application/json"}
        if webhook_secret:
            headers["X-Webhook-Secret"] = webhook_secret

        try:
            with httpx.Client(timeout=30.0) as client:
                webhook_response = client.post(callback_url, json=webhook_payload, headers=headers)
                webhook_response.raise_for_status()
                print(f"Webhook sent successfully to {callback_url}")
        except Exception as e:
            print(f"Webhook failed: {e}")

        return {"status": "completed", "job_id": job_id, "rallies_found": len(rallies)}

    except Exception as e:
        print(f"Detection failed: {e}")
        import traceback

        traceback.print_exc()

        # Try to send error to callback
        try:
            error_result = {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
            }
            headers = {"Content-Type": "application/json"}
            if webhook_secret:
                headers["X-Webhook-Secret"] = webhook_secret

            with httpx.Client(timeout=30.0) as client:
                client.post(callback_url, json=error_result, headers=headers)
        except Exception:
            pass

        return {"status": "failed", "job_id": job_id, "error": str(e)}


# Local testing entrypoint
@app.local_entrypoint()
def main(video_url: str = "", callback_url: str = "") -> None:
    """
    Test the detection service locally.

    Usage:
        modal run rallycut/service/platforms/modal_app.py --video-url "https://..." --callback-url "https://..."

    The callback_url is optional - if not provided, results are just printed.
    """
    import json

    if not video_url:
        print("Usage: modal run modal_app.py --video-url <URL> [--callback-url <URL>]")
        print("\nExample:")
        print('  modal run modal_app.py --video-url "https://bucket.s3.amazonaws.com/video.mp4"')
        return

    job_id = f"test_{__import__('uuid').uuid4().hex[:8]}"

    print("Testing detection service...")
    print(f"  Job ID: {job_id}")
    print(f"  Video URL: {video_url}")
    print(f"  Callback URL: {callback_url or '(none - results printed only)'}")
    print()

    # This will run on Modal's infrastructure
    result = detect_rallies.remote(
        job_id=job_id,
        video_url=video_url,
        callback_url=callback_url,
        webhook_secret=None,
        config={"min_play_duration": 5.0},
    )

    print(f"\nResponse:\n{json.dumps(result, indent=2, default=str)}")
