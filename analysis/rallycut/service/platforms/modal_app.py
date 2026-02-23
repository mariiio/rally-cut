"""Modal deployment for RallyCut detection service.

Deploy with:
    modal deploy rallycut/service/platforms/modal_app.py

Test locally:
    modal serve rallycut/service/platforms/modal_app.py

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

import traceback

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
        # ONNX inference (GPU-accelerated for WASB ball tracker)
        "onnxruntime-gpu>=1.17.0",
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


@app.function(image=image)
@modal.fastapi_endpoint(method="GET", docs=True)
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
@modal.fastapi_endpoint(method="POST", docs=True)
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
    import time as _time

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
            for attempt in range(1, 4):
                try:
                    s3.download_file(bucket, video_key, temp_path)
                    break
                except Exception as dl_err:
                    if attempt < 3:
                        delay = 2 ** (attempt - 1)
                        print(f"S3 download failed (attempt {attempt}/3), retrying in {delay}s: {dl_err}")
                        _time.sleep(delay)
                    else:
                        raise

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

        # Check if detection failed internally
        if response.status == "failed":
            raise RuntimeError(response.error or "Detection failed with no error details")

        # Clean up temp file
        os.unlink(temp_path)

        # Transform response to API webhook format
        from rallycut.service.webhook_utils import build_detection_webhook_payload

        webhook_payload = build_detection_webhook_payload(response, job_id)

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

        return {"status": "completed", "job_id": job_id, "rallies_found": len(webhook_payload["rallies"])}

    except Exception as e:
        print(f"Detection failed: {e}")
        traceback.print_exc()

        # Try to send error to callback
        try:
            error_result = {
                "job_id": job_id,
                "status": "failed",
                "error_message": str(e),
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
def main() -> None:
    """
    Deploy or serve the detection app.

    Usage:
        modal deploy rallycut/service/platforms/modal_app.py
        modal serve rallycut/service/platforms/modal_app.py
    """
    print("Use 'modal serve' or 'modal deploy' to run. The detect() web endpoint accepts POST requests.")
