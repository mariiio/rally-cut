"""Modal deployment for VideoMAE fine-tuning.

Deploy with:
    modal deploy rallycut/training/modal_train.py

Run training:
    modal run rallycut/training/modal_train.py --epochs 25

Training data must be uploaded to Modal volume first:
    modal volume put rallycut-training training_data/ training_data/
"""

from __future__ import annotations

import modal

# Define the Modal app
app = modal.App("rallycut-training")

# Container image with all training dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        # Core ML
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        # Training
        "accelerate>=0.25.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.3.0",
        # Video processing
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        # Config and utilities
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
        "platformdirs>=4.0.0",
    )
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir("rallycut", "/app/rallycut")
    .add_local_dir("lib", "/app/lib")
    .add_local_dir("weights", "/app/weights")
)

# Volume for training data and model outputs
training_volume = modal.Volume.from_name("rallycut-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # T4: 16GB VRAM, cheapest option (~$0.59/hr)
    timeout=14400,  # 4 hours max
    volumes={"/data": training_volume},
    memory=16384,  # 16GB RAM
)
def train_model(
    epochs: int = 25,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    data_dir: str = "/data/training_data",
    output_dir: str = "/data/models/beach_volleyball",
) -> dict:
    """
    Fine-tune VideoMAE on beach volleyball data.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size (can be larger on A10G)
        learning_rate: Learning rate
        data_dir: Path to training data on volume
        output_dir: Path to save model on volume

    Returns:
        Training results dict
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app")

    from rallycut.training.config import TrainingConfig
    from rallycut.training.sampler import TrainingSample
    from rallycut.training.train import train

    print(f"Starting training on Modal A10G GPU")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Check data exists
    if not data_path.exists():
        raise ValueError(
            f"Training data not found at {data_dir}. "
            "Upload with: modal volume put rallycut-training training_data/ training_data/"
        )

    # Load samples
    with open(data_path / "train_samples.json") as f:
        train_data = json.load(f)

    with open(data_path / "val_samples.json") as f:
        val_data = json.load(f)

    with open(data_path / "video_paths.json") as f:
        video_paths_data = json.load(f)

    # Convert to TrainingSample objects
    train_samples = [
        TrainingSample(
            video_id=s["video_id"],
            start_frame=s["start_frame"],
            label=s["label"],
            label_name=["NO_PLAY", "PLAY", "SERVICE"][s["label"]],
        )
        for s in train_data
    ]

    val_samples = [
        TrainingSample(
            video_id=s["video_id"],
            start_frame=s["start_frame"],
            label=s["label"],
            label_name=["NO_PLAY", "PLAY", "SERVICE"][s["label"]],
        )
        for s in val_data
    ]

    # Video paths - need to remap to volume location
    video_paths = {}
    for vid, path in video_paths_data.items():
        # Original paths are local, remap to volume
        # Videos should be uploaded to /data/videos/
        filename = Path(path).name
        volume_path = Path("/data/videos") / filename
        if volume_path.exists():
            video_paths[vid] = volume_path
        else:
            print(f"Warning: Video not found at {volume_path}")

    if not video_paths:
        raise ValueError(
            "No videos found. Upload videos with: "
            "modal volume put rallycut-training <local_video_dir>/ videos/"
        )

    print(f"Loaded {len(train_samples)} train samples, {len(val_samples)} val samples")
    print(f"Found {len(video_paths)} videos")

    # Configure training for A10G GPU
    config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,  # Can use larger batch on A10G (24GB VRAM)
        gradient_accumulation_steps=2,  # Effective batch = batch_size * 2
        learning_rate=learning_rate,
        output_dir=output_path,
        use_mps=False,  # Use CUDA on Modal
        dataloader_num_workers=4,  # Can use multiprocessing on Modal
    )

    # Run training
    model_path = train(
        train_samples=train_samples,
        val_samples=val_samples,
        video_paths=video_paths,
        config=config,
    )

    # Commit volume to persist results
    training_volume.commit()

    return {
        "status": "completed",
        "model_path": str(model_path),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "epochs": epochs,
    }


@app.function(image=image, volumes={"/data": training_volume})
def upload_training_data(local_data_dir: str) -> dict:
    """Upload training data to Modal volume."""
    import shutil
    from pathlib import Path

    local_path = Path(local_data_dir)
    volume_path = Path("/data/training_data")

    if not local_path.exists():
        raise ValueError(f"Local data directory not found: {local_data_dir}")

    # Copy files to volume
    volume_path.mkdir(parents=True, exist_ok=True)

    for file in local_path.glob("*.json"):
        shutil.copy(file, volume_path / file.name)
        print(f"Uploaded {file.name}")

    training_volume.commit()

    return {"status": "uploaded", "path": str(volume_path)}


@app.function(image=image, volumes={"/data": training_volume})
def download_model(output_dir: str = "weights/videomae/beach_volleyball") -> dict:
    """Download trained model from Modal volume."""
    from pathlib import Path

    volume_path = Path("/data/models/beach_volleyball/best")
    local_path = Path(output_dir)

    if not volume_path.exists():
        raise ValueError(f"Model not found at {volume_path}")

    # List files to download
    files = list(volume_path.glob("*"))
    return {
        "status": "ready",
        "files": [str(f.name) for f in files],
        "download_cmd": f"modal volume get rallycut-training models/beach_volleyball/best/ {output_dir}/",
    }


@app.local_entrypoint()
def main(
    epochs: int = 25,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    upload_data: bool = False,
    download: bool = False,
) -> None:
    """
    Train VideoMAE on Modal GPU.

    Usage:
        # First, upload training data
        modal run rallycut/training/modal_train.py --upload-data

        # Then upload videos (run separately)
        modal volume put rallycut-training <path-to-videos>/ videos/

        # Run training
        modal run rallycut/training/modal_train.py --epochs 25

        # Download trained model
        modal run rallycut/training/modal_train.py --download
    """
    import json

    if upload_data:
        print("Uploading training data to Modal volume...")
        result = upload_training_data.remote("training_data")
        print(json.dumps(result, indent=2))
        return

    if download:
        print("Checking trained model...")
        result = download_model.remote()
        print(json.dumps(result, indent=2))
        print(f"\nTo download, run:\n  {result['download_cmd']}")
        return

    print(f"Starting training on Modal...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    result = train_model.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print(f"\nTraining complete!")
    print(json.dumps(result, indent=2))
