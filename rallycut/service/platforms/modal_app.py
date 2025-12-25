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
