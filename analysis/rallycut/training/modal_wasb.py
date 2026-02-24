"""Modal deployment for WASB HRNet ball tracking fine-tuning.

Fine-tunes WASB HRNet on pseudo-labels to improve ball detection on beach volleyball.
Uses A10G GPU on Modal.

Data must be uploaded to Modal volume first:
    rallycut train wasb-modal --upload

Then train:
    modal run rallycut/training/modal_wasb.py --epochs 30

Training is preemption-resilient: checkpoints are saved every 3 epochs
and training auto-resumes from the latest checkpoint on retry.
"""

from __future__ import annotations

import modal

app = modal.App("rallycut-wasb")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "platformdirs>=4.0.0",
    )
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir("rallycut", "/app/rallycut")
)

training_volume = modal.Volume.from_name("rallycut-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours max
    volumes={"/data": training_volume},
    memory=16384,
    retries=modal.Retries(
        max_retries=2,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
)
def train_model(
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    val_ratio: float = 0.2,
    max_rallies: int = 0,
    data_dir: str = "/data/wasb_data",
    output_dir: str = "/data/models/wasb",
    fresh: bool = False,
) -> dict:
    """Train WASB HRNet on Modal A10G GPU.

    Scans data_dir for available rallies (CSV + images), splits by rally
    into train/val, and fine-tunes the model.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size (A10G 24GB VRAM supports 8-16 for WASB).
        val_ratio: Fraction of rallies held out for validation.
        data_dir: Path to training data on volume.
        output_dir: Path to save model on volume.
        fresh: If True, ignore existing checkpoints and start fresh.

    Returns:
        Training results dict.
    """
    import json
    import random
    import re
    import shutil
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app")

    from rallycut.training.wasb import WASBConfig, train_wasb

    print("Starting WASB HRNet training on Modal A10G GPU")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Val ratio: {val_ratio}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Extract tarball if data dir doesn't exist but tarball does
    tarball = Path("/data/wasb_data.tar.gz")
    if not data_path.exists() and tarball.exists():
        import tarfile

        print(f"Extracting {tarball} → /data/ ...")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall("/data/")
        # Tarball contains wasb_pseudo_labels/ — rename to wasb_data/
        extracted = Path("/data/wasb_pseudo_labels")
        if extracted.exists() and not data_path.exists():
            extracted.rename(data_path)
        print(f"  Extracted to {data_path}")

    if not data_path.exists():
        raise ValueError(
            f"Training data not found at {data_dir}. "
            "Upload with: rallycut train wasb-modal --upload"
        )

    # Discover available rallies (must have both CSV and images)
    csv_files = sorted(data_path.glob("*.csv"))
    rally_ids: list[str] = []
    for csv_file in csv_files:
        rally_id = csv_file.stem
        if rally_id.endswith("_gold") or rally_id.startswith("._"):
            continue
        img_dir = data_path / "images" / rally_id
        if img_dir.is_dir() and any(img_dir.iterdir()):
            rally_ids.append(rally_id)

    if not rally_ids:
        raise ValueError(
            f"No valid rally data found in {data_dir}. "
            "Need <rally_id>.csv and images/<rally_id>/ for each rally."
        )

    print(f"  Found {len(rally_ids)} rallies with labels + images")

    # Read manifest to discover GT rallies (written by pseudo_label_export.py).
    # GT rallies are forced into validation to prevent eval contamination.
    manifest_path = data_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"No manifest.json found in {data_dir}. "
            "Re-export pseudo-labels: python -m experiments.pseudo_label_export "
            "--output-dir experiments/wasb_pseudo_labels --all-tracked"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    gt_rally_ids = set(manifest.get("gold_rally_ids", []))
    print(f"  Manifest: {len(gt_rally_ids)} GT rallies (eval holdout)")

    # Split: GT rallies → val, remaining → train/val
    gt_ids = [r for r in rally_ids if r in gt_rally_ids]
    non_gt_ids = [r for r in rally_ids if r not in gt_rally_ids]

    # Quality-based rally selection (if max_rallies > 0)
    if max_rallies > 0 and len(non_gt_ids) > max_rallies:
        import csv as csv_mod
        import math

        import numpy as np

        def _score(rid: str) -> float:
            csv_path = data_path / f"{rid}.csv"
            xs, ys, confs = [], [], []
            total = 0
            with open(csv_path) as f:
                reader = csv_mod.reader(f)
                next(reader)
                for r in reader:
                    total += 1
                    if int(r[1]) >= 1:
                        xs.append(float(r[2]))
                        ys.append(float(r[3]))
                        confs.append(float(r[4]) if len(r) > 4 else 1.0)
            if len(xs) < 10 or total < 10:
                return 0.0
            det_rate = len(xs) / total
            spread = math.sqrt(float(np.std(xs)) ** 2 + float(np.std(ys)) ** 2)
            mean_conf = float(np.mean(confs))
            spread_score = min(1.0, spread / 0.10)
            det_score = 1.0 if 0.4 <= det_rate <= 0.90 else max(0, 1.0 - abs(det_rate - 0.65) * 3)
            return spread_score * 0.5 + det_score * 0.3 + mean_conf * 0.2

        scored = sorted([(rid, _score(rid)) for rid in non_gt_ids], key=lambda x: x[1], reverse=True)
        non_gt_ids = [rid for rid, _ in scored[:max_rallies]]
        print(f"  Selected {len(non_gt_ids)}/{len(scored)} pseudo-label rallies by quality")

    random.seed(42)
    random.shuffle(non_gt_ids)
    extra_val_count = max(0, int(len(non_gt_ids) * val_ratio) - len(gt_ids))
    train_ids = non_gt_ids[: len(non_gt_ids) - extra_val_count]
    val_ids = gt_ids + non_gt_ids[len(non_gt_ids) - extra_val_count :]

    print(f"  Train: {len(train_ids)} rallies (no GT)")
    print(f"  Val: {len(val_ids)} rallies ({len(gt_ids)} with ball GT)")

    # Handle checkpoints
    resume_checkpoint = None
    if fresh:
        print("  Fresh training: ignoring existing checkpoints")
        if output_path.exists():
            ckpt_dir = output_path / "checkpoint"
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
                print("    Deleted existing checkpoints")
    elif output_path.exists():
        ckpt_latest = output_path / "checkpoint" / "ckpt_latest.pt"
        if ckpt_latest.exists():
            resume_checkpoint = ckpt_latest
            checkpoints = list((output_path / "checkpoint").glob("ckpt_[0-9]*.pt"))
            if checkpoints:
                latest = max(
                    checkpoints,
                    key=lambda p: int(re.search(r"ckpt_(\d+)", p.name).group(1))  # type: ignore[union-attr]
                    if re.search(r"ckpt_(\d+)", p.name)
                    else 0,
                )
                print(f"  Resuming from checkpoint: {latest.name}")

    # Find pretrained weights
    pretrained_path = Path("/data/pretrained/wasb_volleyball_best.pth.tar")
    if pretrained_path.exists() and not resume_checkpoint:
        print(f"  Pretrained weights: {pretrained_path}")
    else:
        pretrained_path = None  # type: ignore[assignment]

    config = WASBConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=4,
    )

    result = train_wasb(
        data_dir=data_path,
        output_dir=output_path,
        config=config,
        train_rally_ids=train_ids,
        val_rally_ids=val_ids,
        resume_checkpoint=resume_checkpoint,
        pretrained_weights=pretrained_path,
    )

    training_volume.commit()

    return {
        "status": "completed",
        "model_path": result.model_path,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "accuracy": result.accuracy,
        "train_rallies": len(train_ids),
        "val_rallies": len(val_ids),
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours max
    volumes={"/data": training_volume},
    memory=16384,
    retries=modal.Retries(
        max_retries=2,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
)
def train_mean_teacher(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_ratio: float = 0.2,
    max_rallies: int = 80,
    data_dir: str = "/data/wasb_data",
    output_dir: str = "/data/models/wasb_mt",
    fresh: bool = False,
) -> dict:
    """Train WASB with Mean Teacher on Modal A10G GPU.

    Same data layout as train_model(), but uses Mean Teacher semi-supervised
    training loop for noise-robust learning from pseudo-labels.

    Smart rally selection: all gold GT rallies + best pseudo-label rallies
    ranked by detection quality (spread, detection rate, confidence).
    """
    import csv
    import json
    import math
    import random
    import re
    import shutil
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app")

    from rallycut.training.mean_teacher import MeanTeacherConfig, train_wasb_mean_teacher

    print("Starting WASB Mean Teacher training on Modal A10G GPU")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Val ratio: {val_ratio}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Extract tarball if data dir doesn't exist but tarball does
    tarball = Path("/data/wasb_data.tar.gz")
    if not data_path.exists() and tarball.exists():
        import tarfile

        print(f"Extracting {tarball} → /data/ ...")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall("/data/")
        extracted = Path("/data/wasb_pseudo_labels")
        if extracted.exists() and not data_path.exists():
            extracted.rename(data_path)
        print(f"  Extracted to {data_path}")

    if not data_path.exists():
        raise ValueError(
            f"Training data not found at {data_dir}. "
            "Upload with: rallycut train wasb-mt-modal --upload"
        )

    # Discover available rallies
    csv_files = sorted(data_path.glob("*.csv"))
    rally_ids: list[str] = []
    for csv_file in csv_files:
        rally_id = csv_file.stem
        if rally_id.endswith("_gold") or rally_id.startswith("._"):
            continue
        img_dir = data_path / "images" / rally_id
        if img_dir.is_dir() and any(img_dir.iterdir()):
            rally_ids.append(rally_id)

    if not rally_ids:
        raise ValueError(
            f"No valid rally data found in {data_dir}. "
            "Need <rally_id>.csv and images/<rally_id>/ for each rally."
        )

    print(f"  Found {len(rally_ids)} rallies with labels + images")

    # Read manifest for GT rally IDs
    manifest_path = data_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"No manifest.json found in {data_dir}. "
            "Re-export pseudo-labels with confidence column."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    gt_rally_ids = set(manifest.get("gold_rally_ids", []))
    print(f"  Manifest: {len(gt_rally_ids)} GT rallies (gold labels)")

    # ── Smart rally selection: rank pseudo-label rallies by quality ──
    gt_ids = [r for r in rally_ids if r in gt_rally_ids]
    non_gt_ids = [r for r in rally_ids if r not in gt_rally_ids]

    # Score each pseudo-label rally from its CSV
    def _score_rally(rid: str) -> float:
        """Quick quality score from CSV: spatial spread + detection rate."""
        csv_path = data_path / f"{rid}.csv"
        xs, ys, confs = [], [], []
        total_frames = 0
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for r in reader:
                total_frames += 1
                if int(r[1]) >= 1:  # visible
                    xs.append(float(r[2]))
                    ys.append(float(r[3]))
                    conf = float(r[4]) if len(r) > 4 else 1.0
                    confs.append(conf)
        if len(xs) < 10 or total_frames < 10:
            return 0.0
        import numpy as np
        det_rate = len(xs) / total_frames
        spread = math.sqrt(float(np.std(xs)) ** 2 + float(np.std(ys)) ** 2)
        mean_conf = float(np.mean(confs))
        # Higher spread = real ball trajectory (not player lock-on)
        # Moderate detection rate (40-90%) = realistic
        # Higher confidence = model is sure
        spread_score = min(1.0, spread / 0.10)
        det_score = 1.0 if 0.4 <= det_rate <= 0.90 else max(0, 1.0 - abs(det_rate - 0.65) * 3)
        return spread_score * 0.5 + det_score * 0.3 + mean_conf * 0.2

    scored = [(rid, _score_rally(rid)) for rid in non_gt_ids]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Select top rallies up to max_rallies budget (gold always included)
    non_gt_budget = max(0, max_rallies - len(gt_ids))
    selected_non_gt = [rid for rid, _ in scored[:non_gt_budget]]

    if len(scored) > non_gt_budget:
        top_score = scored[0][1] if scored else 0
        cutoff_score = scored[non_gt_budget - 1][1] if non_gt_budget > 0 else 0
        print(f"  Rally quality: top={top_score:.3f}, cutoff={cutoff_score:.3f} "
              f"(selected {len(selected_non_gt)}/{len(non_gt_ids)} pseudo-label rallies)")
    else:
        print(f"  Using all {len(selected_non_gt)} pseudo-label rallies")

    # Split: gold GT in both train and val, pseudo-label split by quality
    random.seed(42)
    val_non_gt_count = max(1, int(len(selected_non_gt) * val_ratio))
    # Val gets a random sample (not worst — want honest evaluation)
    random.shuffle(selected_non_gt)
    train_non_gt = selected_non_gt[val_non_gt_count:]
    val_non_gt = selected_non_gt[:val_non_gt_count]

    train_ids = gt_ids + train_non_gt
    val_ids = gt_ids + val_non_gt

    print(f"  Train: {len(train_ids)} rallies ({len(gt_ids)} gold GT + {len(train_non_gt)} pseudo)")
    print(f"  Val: {len(val_ids)} rallies ({len(gt_ids)} gold GT + {len(val_non_gt)} pseudo)")

    # Handle checkpoints
    resume_checkpoint = None
    if fresh:
        print("  Fresh training: ignoring existing checkpoints")
        if output_path.exists():
            ckpt_dir = output_path / "checkpoint"
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
                print("    Deleted existing checkpoints")
    elif output_path.exists():
        ckpt_latest = output_path / "checkpoint" / "ckpt_latest.pt"
        if ckpt_latest.exists():
            resume_checkpoint = ckpt_latest
            checkpoints = list((output_path / "checkpoint").glob("ckpt_[0-9]*.pt"))
            if checkpoints:
                latest = max(
                    checkpoints,
                    key=lambda p: int(re.search(r"ckpt_(\d+)", p.name).group(1))  # type: ignore[union-attr]
                    if re.search(r"ckpt_(\d+)", p.name)
                    else 0,
                )
                print(f"  Resuming from checkpoint: {latest.name}")

    # Find pretrained weights
    pretrained_path = Path("/data/pretrained/wasb_volleyball_best.pth.tar")
    if pretrained_path.exists() and not resume_checkpoint:
        print(f"  Pretrained weights: {pretrained_path}")
    else:
        pretrained_path = None  # type: ignore[assignment]

    config = MeanTeacherConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=4,
    )

    result = train_wasb_mean_teacher(
        data_dir=data_path,
        output_dir=output_path,
        config=config,
        train_rally_ids=train_ids,
        val_rally_ids=val_ids,
        gold_rally_ids=gt_rally_ids,
        resume_checkpoint=resume_checkpoint,
        pretrained_weights=pretrained_path,
    )

    training_volume.commit()

    return {
        "status": "completed",
        "model_path": result.model_path,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "accuracy": result.accuracy,
        "train_rallies": len(train_ids),
        "val_rallies": len(val_ids),
        "gold_rallies": len(gt_rally_ids),
    }


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    val_ratio: float = 0.2,
    max_rallies: int = 0,
    download: bool = False,
    fresh: bool = False,
) -> None:
    """Train WASB HRNet on Modal GPU.

    Workflow:
        1. Export pseudo-labels:
           python -m experiments.pseudo_label_export --all-tracked --extract-frames
        2. Upload: rallycut train wasb-modal --upload
        3. Train: modal run rallycut/training/modal_wasb.py --epochs 30
        4. Download: rallycut train wasb-modal --download
    """
    import json

    if download:
        print("Download trained model with:")
        print("  modal volume get rallycut-training models/wasb/best.pt weights/wasb/")
        print("  modal volume get rallycut-training models/wasb/last.pt weights/wasb/")
        return

    print("Starting WASB HRNet training on Modal...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Val ratio: {val_ratio}")
    if max_rallies > 0:
        print(f"  Max rallies: {max_rallies} (gold GT always included)")
    if fresh:
        print("  Mode: Fresh (ignoring existing checkpoints)")

    result = train_model.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        max_rallies=max_rallies,
        fresh=fresh,
    )

    print("\nTraining complete!")
    print(json.dumps(result, indent=2))
