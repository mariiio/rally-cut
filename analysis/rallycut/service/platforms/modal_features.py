"""Modal app for bulk VideoMAE feature extraction.

Used by Phase 1 of the VideoMAE contact validation (see plan file).

Design notes:
- One Modal function per video so failures/retries are per-video.
- Caller (`scripts/extract_contact_features.py`) checks local FeatureCache
  before invoking this function, so re-runs skip completed videos.
- Modal retries=2 covers T4 preemptions transparently.
- Features are returned via the function's return value (one-shot batch of
  68 videos × ~90MB = ~6 GB total; well within Modal's per-call return size
  limit). This avoids volume coordination and keeps local cache authoritative.
- Atomic behavior: extraction either fully succeeds and returns features, or
  fails cleanly. Callers save to disk atomically via tmp-then-rename.

Deploy:
    modal deploy rallycut/service/platforms/modal_features.py

Ping to verify auth + image:
    modal run rallycut/service/platforms/modal_features.py::ping
"""

from __future__ import annotations

import modal

APP_NAME = "rallycut-features"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pyyaml>=6.0.0",
        "httpx>=0.25.0",
        "platformdirs>=4.0.0",
        "tqdm>=4.65.0",
        "scipy>=1.11.0",
        "boto3>=1.34.0",
        "ffmpeg-python>=0.2.0",
    )
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir("rallycut", "/app/rallycut")
    .add_local_dir("lib", "/app/lib")
    .add_local_dir("weights", "/app/weights")
)

app = modal.App(APP_NAME, image=image)

# Weights volume reused from the detection app so any locally-staged finetunes
# are available on Modal without baking into the image.
model_volume = modal.Volume.from_name("rallycut-models", create_if_missing=True)

# Data volume for source videos. Videos are uploaded here once via
# scripts/upload_videos_to_modal.py (they live in local MinIO and can't be
# reached from Modal directly). Layout: /videos/{content_hash}.{ext}.
data_volume = modal.Volume.from_name("rallycut-features-data", create_if_missing=True)


@app.function(image=image, timeout=60)
def ping() -> dict:
    """Health check: verifies auth, image, and imports."""
    import sys

    sys.path.insert(0, "/app")

    status = {"status": "ok", "app": APP_NAME}
    try:
        from rallycut.temporal.features import FEATURE_DIM  # noqa: F401

        status["imports"] = "ok"
    except Exception as e:
        status["imports"] = f"failed: {e}"
        status["status"] = "degraded"
    return status


@app.function(
    image=image,
    gpu="T4",
    timeout=5400,  # 90 min to accommodate 3+ GB originals (e.g. IMG_2313.MOV)
    memory=16384,
    volumes={
        "/models": model_volume,
        "/data": data_volume,
    },
    retries=modal.Retries(
        max_retries=2,
        initial_delay=10.0,
        backoff_coefficient=2.0,
    ),
)
def extract_features_remote(
    content_hash: str,
    ext: str = ".mp4",
    stride: int = 4,
    backbone: str = "videomae-v1",
    batch_size: int = 16,
    pooling: str = "cls",
) -> dict:
    """Read video from Modal volume, extract VideoMAE features, return to caller.

    The video must have been uploaded to the ``rallycut-features-data`` volume
    at ``/videos/{content_hash}{ext}`` via
    ``scripts/upload_videos_to_modal.py``. Local MinIO is unreachable from
    Modal, so volume upload is the staging step.

    Returns on success::

        {
            "ok": True,
            "features": np.ndarray of shape (num_windows, 768),
            "metadata": {FeatureMetadata fields as dict},
            "elapsed_s": float,
            "locate_s": float,
            "extract_s": float,
        }

    On failure, returns {"ok": False, "error": str, "traceback": str}.
    Modal's retries=2 policy covers transient preemption.

    Args:
        content_hash: SHA-256 content hash (stamped into returned metadata and
            used to locate the video under ``/data/videos/``).
        ext: File extension (``.mp4``, ``.MOV``, etc.). Defaults to ``.mp4``.
        stride: Frame stride between VideoMAE windows.
        backbone: Backbone identifier — passed through to FeatureMetadata.
        batch_size: VideoMAE windows per GPU batch (tuned for T4 16 GB).
        pooling: "cls" (default) or "mean" — pooling mode for encoder output.
    """
    import sys
    import time
    import traceback
    from dataclasses import asdict
    from pathlib import Path

    sys.path.insert(0, "/app")

    t_start = time.time()

    try:
        # 1) Locate video on the mounted data volume. Reload the volume so we
        #    see any recently-uploaded files from other processes.
        data_volume.reload()
        video_path = Path(f"/data/videos/{content_hash}{ext}")
        if not video_path.exists():
            # Fall back: scan for content_hash.* (handles extension mismatch)
            candidates = list(Path("/data/videos").glob(f"{content_hash}.*"))
            if not candidates:
                raise FileNotFoundError(
                    f"Video not found on volume: {video_path} (or any "
                    f"/data/videos/{content_hash}.*). "
                    "Run scripts/upload_videos_to_modal.py first."
                )
            video_path = candidates[0]

        locate_s = time.time() - t_start
        size_mb = video_path.stat().st_size / 1e6
        print(f"Located {video_path} ({size_mb:.1f} MB) in {locate_s:.2f}s")

        # 2) Extract features using the same local code path that the LOO
        #    training scripts read from — apples-to-apples guarantee.
        from lib.volleyball_ml.video_mae import GameStateClassifier
        from rallycut.temporal.features import extract_features_for_video

        print(f"Loading VideoMAE (backbone={backbone})")
        classifier = GameStateClassifier(device="cuda")

        def _progress(pct: float, msg: str) -> None:
            # Log roughly every 10% to avoid log spam
            print(f"  [{pct*100:5.1f}%] {msg}")

        t_ext = time.time()
        features, metadata = extract_features_for_video(
            str(video_path),
            classifier,
            stride=stride,
            batch_size=batch_size,
            pooling=pooling,
            progress_callback=_progress,
        )
        extract_s = time.time() - t_ext
        metadata.content_hash = content_hash
        metadata.backbone = backbone

        elapsed = time.time() - t_start
        print(
            f"Extracted {len(features)} windows ({features.shape}) "
            f"in {extract_s:.1f}s (total {elapsed:.1f}s)"
        )

        return {
            "ok": True,
            "content_hash": content_hash,  # echoed back for unordered matching
            "features": features,
            "metadata": asdict(metadata),
            "elapsed_s": float(elapsed),
            "locate_s": float(locate_s),
            "extract_s": float(extract_s),
            "num_windows": int(len(features)),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Extraction failed: {e}\n{tb}")
        return {
            "ok": False,
            "content_hash": content_hash,  # echoed back for unordered matching
            "error": str(e),
            "traceback": tb,
            "elapsed_s": float(time.time() - t_start),
        }


@app.local_entrypoint()
def main() -> None:
    """Quick sanity check: call ping() remotely."""
    result = ping.remote()
    print(result)
