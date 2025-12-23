"""Modal deployment for RallyCut detection service.

Deploy with:
    modal deploy rallycut/service/platforms/modal_app.py

Test locally:
    modal run rallycut/service/platforms/modal_app.py

Environment setup:
    1. pip install modal
    2. modal token new  # Authenticate with Modal
    3. Create Modal secrets for cloud storage credentials (optional)
"""

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
    )
    .copy_local_dir("rallycut", "/app/rallycut")
    .copy_local_dir("lib", "/app/lib")
    .copy_local_dir("weights", "/app/weights", ignore=[".git", "__pycache__"])
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
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
def detect_rallies(request_json: dict) -> dict:
    """
    Core detection function - runs on GPU.

    Args:
        request_json: DetectionRequest as dict

    Returns:
        DetectionResponse as dict
    """
    import sys

    sys.path.insert(0, "/app")

    from rallycut.service.detection import DetectionService
    from rallycut.service.schemas import DetectionRequest

    request = DetectionRequest(**request_json)
    service = DetectionService(device="cuda")

    # Run detection with progress tracking
    def progress_callback(pct: float, msg: str) -> None:
        # Modal doesn't have built-in progress streaming,
        # but we can log for monitoring
        print(f"[{pct*100:.1f}%] {msg}")

    response = service.detect(request, progress_callback)
    return response.model_dump(mode="json")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/models": model_volume},
    memory=16384,
)
@modal.web_endpoint(method="POST", docs=True)
def detect(request_json: dict) -> dict:
    """
    REST API endpoint for synchronous detection.

    POST https://<your-app>.modal.run/detect
    Content-Type: application/json

    Request body:
    {
        "video_url": "https://...",  # S3/GCS presigned URL
        "config": {                   # Optional
            "min_play_duration": 5.0,
            "padding_seconds": 1.0,
            "use_two_pass": true
        }
    }

    Returns:
        Full DetectionResponse with segments, scores, statistics
    """
    return detect_rallies.remote(request_json)


@app.function(image=image)
@modal.web_endpoint(method="POST", docs=True)
def detect_async(request_json: dict) -> dict:
    """
    Async API endpoint - submits job and returns immediately.

    POST https://<your-app>.modal.run/detect_async
    Content-Type: application/json

    Request body: Same as /detect, with optional callback_url

    Returns:
        {
            "job_id": "det_abc123",
            "status": "processing",
            "message": "Detection job started"
        }
    """
    import uuid

    # Generate job ID if not provided
    job_id = request_json.get("job_id") or f"det_{uuid.uuid4().hex[:12]}"
    request_json["job_id"] = job_id

    # Spawn detection as background task
    detect_rallies.spawn(request_json)

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Detection job started. Use callback_url for results or poll /jobs/{job_id}",
    }


@app.function(image=image)
@modal.web_endpoint(method="GET", docs=True)
def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "rallycut-detection"}


# Local testing entrypoint
@app.local_entrypoint()
def main():
    """Test the detection service locally."""
    import json

    # Example test request
    test_request = {
        "video_url": "https://example.com/test.mp4",  # Replace with real URL
        "config": {
            "min_play_duration": 5.0,
            "use_two_pass": True,
        },
    }

    print("Testing detection service...")
    print(f"Request: {json.dumps(test_request, indent=2)}")

    # This will run on Modal's infrastructure
    result = detect_rallies.remote(test_request)

    print(f"\nResponse: {json.dumps(result, indent=2, default=str)}")
