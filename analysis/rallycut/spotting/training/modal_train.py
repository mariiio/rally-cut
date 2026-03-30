"""Modal deployment for E2E-Spot training on GPU.

Upload data first, then run training:
    uv run python scripts/spot_train.py --export-modal     # Export frames + labels to Modal
    modal run rallycut/spotting/training/modal_train.py     # Train on A10G GPU
    uv run python scripts/spot_train.py --download-modal    # Download trained model

The Modal volume stores:
    /data/spotting/frames/{rally_id}/000000.jpg ...
    /data/spotting/metadata.json   # Rally metadata + labels + split info
    /data/spotting/checkpoints/    # Training outputs
"""

from __future__ import annotations

import modal

app = modal.App("rallycut-spotting")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=1.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "rich>=13.0.0",
    )
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    # Create minimal package init (no core module deps)
    .run_commands("mkdir -p /app/rallycut && echo '\"\"\"RallyCut spotting (Modal).\"\"\"' > /app/rallycut/__init__.py")
    # Add local code LAST (Modal optimization: mounted at startup, not baked into image)
    .add_local_dir("rallycut/spotting", "/app/rallycut/spotting")
)

volume = modal.Volume.from_name("rallycut-spotting", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=28800,  # 8 hours
    volumes={"/data": volume},
    memory=49152,  # 48GB: ~3GB frame cache (shared mem) + workers + model
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
def train_spot(
    epochs: int = 150,
    batch_size: int = 8,
    clip_length: int = 96,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    num_layers: int = 2,
    patience: int = 20,
    focal_gamma: float = 2.0,
    offset_weight: float = 0.1,
    pretrained_name: str | None = None,
    freeze_backbone_epochs: int = 0,
    backbone_lr_scale: float = 0.1,
    fresh: bool = False,
) -> dict:
    """Train E2E-Spot on GPU with data from Modal volume."""
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import torch

    sys.path.insert(0, "/app")

    import subprocess
    import time

    from rallycut.spotting.config import E2ESpotConfig
    from rallycut.spotting.data.beach import RallyInfo
    from rallycut.spotting.training.trainer import train

    data_dir = Path("/data/spotting")
    # Use local container disk for frames (fast SSD, not network volume)
    local_dir = Path("/tmp/spotting")
    frames_dir = local_dir / "frames"
    # Save checkpoints to local SSD during training (fast), copy to volume at end
    checkpoint_dir = Path("/tmp/spotting/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    volume_checkpoint_dir = data_dir / "checkpoints"
    volume_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=== E2E-Spot Training on Modal GPU ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram: int = getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)
    print(f"  VRAM: {vram / 1e9:.1f} GB")

    # Extract tar to local SSD (much faster than network volume reads)
    tar_path = data_dir / "frames.tar"
    if not frames_dir.exists():
        if tar_path.exists():
            print("  Extracting frames to local SSD...")
            t0 = time.time()
            local_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["tar", "xf", str(tar_path), "-C", str(local_dir)],
                check=True, stderr=subprocess.DEVNULL,
            )
            # tar extracts to spotting_frames/, rename to frames/
            extracted = local_dir / "spotting_frames"
            if extracted.exists():
                extracted.rename(frames_dir)
            print(f"  Extracted in {time.time() - t0:.0f}s")
        else:
            # Fallback: try volume frames dir
            vol_frames = data_dir / "frames"
            if vol_frames.exists():
                frames_dir = vol_frames
                print("  Using frames from volume (slower)")
            else:
                raise FileNotFoundError("No frames found on volume")

    # Load metadata
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "metadata.json not found. Run: uv run python scripts/spot_train.py --export-modal"
        )

    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"  Rallies: {len(metadata['rallies'])}")

    # Reconstruct RallyInfo objects from metadata
    train_rallies: list[RallyInfo] = []
    val_rallies: list[RallyInfo] = []

    for entry in metadata["rallies"]:
        frame_dir = frames_dir / entry["rally_id"]
        if not frame_dir.exists():
            print(f"  [WARN] Missing frames for {entry['rally_id']}")
            continue

        labels = np.array(entry["labels"], dtype=np.int64)
        offsets = np.array(entry["offsets"], dtype=np.float32)

        # Reconstruct GtLabel-like objects for evaluation
        from types import SimpleNamespace

        gt_labels = [
            SimpleNamespace(
                frame=gt["frame"],
                action=gt["action"],
                player_track_id=gt.get("player_track_id", -1),
                ball_x=gt.get("ball_x"),
                ball_y=gt.get("ball_y"),
            )
            for gt in entry.get("gt_labels", [])
        ]

        rally = RallyInfo(
            rally_id=entry["rally_id"],
            video_id=entry["video_id"],
            frame_dir=frame_dir,
            frame_count=entry["frame_count"],
            fps=entry["fps"],
            labels=labels,
            offsets=offsets,
            gt_labels=gt_labels,
            start_ms=entry.get("start_ms", 0),
        )

        if entry["split"] == "train":
            train_rallies.append(rally)
        else:
            val_rallies.append(rally)

    print(f"  Train: {len(train_rallies)} rallies")
    print(f"  Val:   {len(val_rallies)} rallies")

    # Clean previous run if --fresh
    if fresh:
        for fname in ["checkpoint.pt", "best.pt"]:
            p = checkpoint_dir / fname
            if p.exists():
                p.unlink()
                print(f"  Removed {p.name} (--fresh)")

    # Build config
    config = E2ESpotConfig()
    config.temporal.hidden_dim = hidden_dim
    config.temporal.num_layers = num_layers
    config.training.clip_length = clip_length
    config.training.epochs = epochs
    config.training.lr = lr
    config.training.batch_size = batch_size
    config.training.patience = patience
    config.training.focal_gamma = focal_gamma
    config.training.offset_weight = offset_weight
    config.training.freeze_backbone_epochs = freeze_backbone_epochs
    config.training.backbone_lr_scale = backbone_lr_scale

    # Load pretrained weights if specified
    pretrained_path = None
    if pretrained_name:
        pretrained_path = volume_checkpoint_dir / pretrained_name
        if not pretrained_path.exists():
            print(f"  [WARN] Pretrained checkpoint not found: {pretrained_path}")
            pretrained_path = None

    device = torch.device("cuda")
    model, stats = train(
        train_rallies, val_rallies, config, device,
        pretrained_path=pretrained_path,
        output_dir=checkpoint_dir,
        mirror_dir=volume_checkpoint_dir,  # persist to volume on every save
    )

    volume.commit()
    print(f"\n  Saved to Modal volume: {volume_checkpoint_dir}")

    return stats


@app.local_entrypoint()
def main(
    epochs: int = 150,
    batch_size: int = 8,
    clip_length: int = 96,
    lr: float = 1e-3,
    fresh: bool = False,
    pretrained: str | None = None,
    freeze_backbone_epochs: int = 0,
) -> None:
    stats = train_spot.remote(
        epochs=epochs,
        batch_size=batch_size,
        clip_length=clip_length,
        lr=lr,
        fresh=fresh,
        pretrained_name=pretrained,
        freeze_backbone_epochs=freeze_backbone_epochs,
    )
    print(f"\nTraining complete: {stats}")
