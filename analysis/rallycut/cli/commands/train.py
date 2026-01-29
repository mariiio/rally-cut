"""CLI commands for training VideoMAE on beach volleyball."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rallycut.core.proxy import ProxyGenerator
from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.evaluation.video_resolver import VideoResolver
from rallycut.training.config import TrainingConfig
from rallycut.training.sampler import generate_training_samples, get_sample_statistics

if TYPE_CHECKING:
    from rallycut.training.sampler import TrainingSample

app = typer.Typer(help="Train VideoMAE model on beach volleyball data")
console = Console()


def extract_frames_for_sample(
    cap: cv2.VideoCapture,
    start_frame: int,
    num_frames: int = 16,
) -> np.ndarray:
    """Extract frames for a single sample.

    Args:
        cap: OpenCV video capture (should be at or before start_frame)
        start_frame: Starting frame number
        num_frames: Number of frames to extract (default 16 for VideoMAE)

    Returns:
        Array of shape (num_frames, H, W, 3) in RGB format
    """
    # Seek to start frame
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_pos > start_frame:
        # Need to rewind - seek from beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current_pos = 0

    # Skip to start frame using grab (faster than read)
    while current_pos < start_frame:
        cap.grab()
        current_pos += 1

    # Read frames
    frames: list[np.ndarray] = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # Repeat last frame if video ends early
            if frames:
                frames.append(frames[-1].copy())
            else:
                # Black fallback
                frames.append(np.zeros((480, 854, 3), dtype=np.uint8))
        else:
            # Convert BGR to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return np.array(frames, dtype=np.uint8)


def extract_all_frames(
    samples: list[TrainingSample],
    video_paths: dict[str, Path],
    output_dir: Path,
    num_frames: int = 16,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Extract frames for all samples and save as .npy files.

    Samples are processed in order sorted by (video_id, start_frame) to minimize
    video seeking/rewinding.

    Args:
        samples: List of training samples
        video_paths: Mapping from video_id to video file path
        output_dir: Directory to save extracted frames
        num_frames: Frames per sample (default 16)
        progress_callback: Optional callback(current, total) for progress updates
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort samples by video and frame for sequential reading
    indexed_samples = list(enumerate(samples))
    indexed_samples.sort(key=lambda x: (x[1].video_id, x[1].start_frame))

    # Track current video capture
    current_video_id: str | None = None
    cap: cv2.VideoCapture | None = None

    try:
        for i, (original_idx, sample) in enumerate(indexed_samples):
            # Open new video if needed
            if sample.video_id != current_video_id:
                if cap is not None:
                    cap.release()
                video_path = video_paths[sample.video_id]
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {video_path}")
                current_video_id = sample.video_id

            # Extract frames (cap is guaranteed not None here)
            assert cap is not None
            frames = extract_frames_for_sample(cap, sample.start_frame, num_frames)

            # Save with original index (maintains correspondence with labels)
            np.save(output_dir / f"{original_idx}.npy", frames)

            if progress_callback is not None:
                progress_callback(i + 1, len(samples))
    finally:
        if cap is not None:
            cap.release()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Train VideoMAE model on beach volleyball data."""
    if ctx.invoked_subcommand is None:
        rprint("[yellow]Use 'rallycut train --help' to see available commands[/yellow]")
        raise typer.Exit(0)


@app.command()
def prepare(
    output: Path = typer.Option(
        Path("training_data"),
        "--output",
        "-o",
        help="Output directory for prepared data",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
) -> None:
    """Prepare training data from ground truth rallies.

    Downloads videos and generates balanced training samples.
    """
    config = TrainingConfig(seed=seed)

    rprint("[bold]Preparing Training Data[/bold]")
    rprint()

    # Load ground truth
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading ground truth from database...", total=None)
        videos = load_evaluation_videos(require_ground_truth=True)

    rprint(f"Found [green]{len(videos)}[/green] videos with ground truth")

    total_rallies = sum(len(v.ground_truth_rallies) for v in videos)
    rprint(f"Total ground truth rallies: [green]{total_rallies}[/green]")
    rprint()

    # Download videos and generate proxies
    rprint("[bold]Downloading videos and generating proxies...[/bold]")
    rprint("(Using 480p@30fps proxies for efficient training - VideoMAE only needs 224x224)")

    from rallycut.core.video import Video as VideoReader

    resolver = VideoResolver()
    proxy_gen = ProxyGenerator()
    video_paths: dict[str, Path] = {}
    video_metadata: dict[str, dict[str, float | int]] = {}

    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(videos))
        for video in videos:
            try:
                # Download full video
                local_path = resolver.resolve(video.s3_key, video.content_hash)

                # Get video FPS and frame count
                # Training uses proxy videos which normalize high-FPS to 30fps
                with VideoReader(local_path) as v:
                    original_fps = v.info.fps
                    original_frame_count = v.info.frame_count

                # Proxy FPS matches ProxyGenerator.FPS_NORMALIZE_THRESHOLD
                proxy_fps = 30.0 if original_fps > ProxyGenerator.FPS_NORMALIZE_THRESHOLD else original_fps

                video_metadata[video.id] = {
                    "original_fps": original_fps,
                    "proxy_fps": proxy_fps,
                    "frame_count": original_frame_count,
                }

                # Generate or get cached proxy (480p@30fps - much smaller!)
                proxy_path = proxy_gen.generate_proxy(local_path)

                # If proxy generation was skipped (already small), use original
                video_paths[video.id] = proxy_path if proxy_path != local_path else local_path
                progress.advance(task)
            except Exception as e:
                rprint(f"[red]Failed to process {video.filename}: {e}[/red]")

    rprint(f"Processed [green]{len(video_paths)}[/green] videos")
    rprint()

    # Generate samples
    rprint("[bold]Generating training samples...[/bold]")
    samples = generate_training_samples(
        videos=[v for v in videos if v.id in video_paths],
        config=config,
        seed=seed,
    )

    stats = get_sample_statistics(samples)

    # Display statistics
    table = Table(title="Sample Statistics")
    table.add_column("Class", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right")

    for label in ["NO_PLAY", "PLAY", "SERVICE"]:
        count = stats.get(label, 0)
        pct = count / stats["total"] * 100 if stats["total"] > 0 else 0
        table.add_row(label, str(count), f"{pct:.1f}%")

    table.add_row("Total", str(stats["total"]), "100%", style="bold")
    console.print(table)

    # Split into train/val
    random.seed(seed)
    random.shuffle(samples)

    split_idx = int(len(samples) * config.train_split)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    rprint()
    rprint(f"Train samples: [green]{len(train_samples)}[/green]")
    rprint(f"Val samples: [green]{len(val_samples)}[/green]")

    # Save to output directory
    output.mkdir(parents=True, exist_ok=True)

    # Save samples
    train_data = [
        {"video_id": s.video_id, "start_frame": s.start_frame, "label": s.label}
        for s in train_samples
    ]
    val_data = [
        {"video_id": s.video_id, "start_frame": s.start_frame, "label": s.label}
        for s in val_samples
    ]

    with open(output / "train_samples.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(output / "val_samples.json", "w") as f:
        json.dump(val_data, f, indent=2)

    # Save video paths
    paths_data = {vid: str(path) for vid, path in video_paths.items()}
    with open(output / "video_paths.json", "w") as f:
        json.dump(paths_data, f, indent=2)

    # Save video metadata (fps, frame_count) for accurate training
    with open(output / "video_metadata.json", "w") as f:
        json.dump(video_metadata, f, indent=2)

    # Save metadata
    metadata = {
        "total_videos": len(videos),
        "total_rallies": total_rallies,
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "class_distribution": stats,
        "seed": seed,
    }
    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Extract frames for fast training
    rprint()
    rprint("[bold]Extracting frames for all samples...[/bold]")
    rprint("(Pre-extraction enables multiprocessing and ~10x faster training)")

    train_frames_dir = output / "train_frames"
    val_frames_dir = output / "val_frames"

    with Progress(console=console) as progress:
        # Extract training frames
        task = progress.add_task("Extracting train frames...", total=len(train_samples))
        extract_all_frames(
            train_samples,
            video_paths,
            train_frames_dir,
            num_frames=config.num_frames,
            progress_callback=lambda cur, _: progress.update(task, completed=cur),
        )

        # Extract validation frames
        task = progress.add_task("Extracting val frames...", total=len(val_samples))
        extract_all_frames(
            val_samples,
            video_paths,
            val_frames_dir,
            num_frames=config.num_frames,
            progress_callback=lambda cur, _: progress.update(task, completed=cur),
        )

    # Calculate total size
    train_size = sum(f.stat().st_size for f in train_frames_dir.glob("*.npy"))
    val_size = sum(f.stat().st_size for f in val_frames_dir.glob("*.npy"))
    total_mb = (train_size + val_size) / (1024 * 1024)
    rprint(f"Extracted frames: [green]{total_mb:.0f} MB[/green]")

    rprint()
    rprint(f"[green]Data saved to: {output}[/green]")
    rprint()
    rprint("Next step: [cyan]rallycut train modal --upload --epochs 10[/cyan]")


@app.command()
def run(
    data: Path = typer.Option(
        Path("training_data"),
        "--data",
        "-d",
        help="Path to prepared training data",
    ),
    epochs: int = typer.Option(25, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(5e-5, "--lr", help="Learning rate"),
    output: Path = typer.Option(
        Path("weights/videomae/beach_volleyball"),
        "--output",
        "-o",
        help="Output directory for trained model",
    ),
    resume: str | None = typer.Option(None, "--resume", help="Resume from checkpoint"),
) -> None:
    """Run training on prepared data."""
    from rallycut.training.sampler import TrainingSample
    from rallycut.training.train import train

    rprint("[bold]VideoMAE Fine-Tuning[/bold]")
    rprint()

    # Load prepared data
    if not data.exists():
        rprint(f"[red]Data directory not found: {data}[/red]")
        rprint("Run [cyan]rallycut train prepare[/cyan] first")
        raise typer.Exit(1)

    # Load samples
    with open(data / "train_samples.json") as f:
        train_data = json.load(f)

    with open(data / "val_samples.json") as f:
        val_data = json.load(f)

    with open(data / "video_paths.json") as f:
        video_paths_data = json.load(f)

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

    video_paths = {vid: Path(path) for vid, path in video_paths_data.items()}

    rprint(f"Train samples: [green]{len(train_samples)}[/green]")
    rprint(f"Val samples: [green]{len(val_samples)}[/green]")
    rprint(f"Videos: [green]{len(video_paths)}[/green]")
    rprint()

    # Configure training
    config = TrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output,
    )

    rprint(f"Epochs: {epochs}")
    rprint(f"Batch size: {batch_size} (effective: {config.effective_batch_size})")
    rprint(f"Learning rate: {learning_rate}")
    rprint(f"Output: {output}")
    rprint()

    # Run training
    model_path = train(
        train_samples=train_samples,
        val_samples=val_samples,
        video_paths=video_paths,
        config=config,
        resume_from_checkpoint=resume,
    )

    rprint()
    rprint(f"[green]Model saved to: {model_path}[/green]")
    rprint()
    rprint("To evaluate: [cyan]rallycut evaluate --model " + str(model_path) + "[/cyan]")


@app.command()
def modal(
    epochs: int = typer.Option(25, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size (4-8 for T4 GPU)"),
    learning_rate: float = typer.Option(5e-5, "--lr", help="Learning rate (use 1e-5 for fine-tuning)"),
    upload: bool = typer.Option(False, "--upload", help="Upload training data to Modal volume"),
    upload_videos: bool = typer.Option(
        False, "--upload-videos", help="Upload proxy videos to Modal (parallel)"
    ),
    upload_model: bool = typer.Option(
        False, "--upload-model", help="Upload local model weights for incremental training"
    ),
    download: bool = typer.Option(False, "--download", help="Download trained model from Modal"),
    cleanup: bool = typer.Option(
        False, "--cleanup", help="Delete videos/models from Modal (~$0.75/GB/month saved)"
    ),
    resume_from_model: bool = typer.Option(
        False, "--resume-from-model", help="Continue training from existing beach model weights"
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Start fresh training, ignoring existing checkpoints"
    ),
) -> None:
    """Run training on Modal GPU (T4 - ~$0.59/hr).

    Initial training workflow:
        1. rallycut train prepare                    # Prepare data locally (generates proxies)
        2. rallycut train modal --upload             # Upload training JSON to Modal
        3. rallycut train modal --upload-videos      # Upload proxy videos (parallel)
        4. rallycut train modal --epochs 10          # Run training on T4 GPU
        5. rallycut train modal --download           # Download trained model
        6. rallycut train modal --cleanup            # Delete from Modal (saves storage costs)

    Incremental training (add more labeled videos):
        1. Label new videos in the app
        2. rallycut train export-dataset --name beach_v2
        3. rallycut train prepare
        4. rallycut train modal --upload --upload-videos
        5. rallycut train modal --upload-model                     # Upload existing weights
        6. rallycut train modal --resume-from-model --lr 1e-5      # Fine-tune with lower LR
        7. rallycut train modal --download --cleanup
    """
    import subprocess

    if upload_model:
        rprint("[bold]Uploading local model weights to Modal...[/bold]")
        model_dir = Path("weights/videomae/beach_volleyball")
        if not model_dir.exists():
            rprint(f"[red]Model not found at {model_dir}[/red]")
            rprint("Train a model first or download one.")
            raise typer.Exit(1)

        # Check for required model files
        required_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
        missing = [f for f in required_files if not (model_dir / f).exists()]
        if missing:
            rprint(f"[red]Missing model files: {missing}[/red]")
            raise typer.Exit(1)

        cmd = [
            "python3", "-m", "modal", "volume", "put", "-f",
            "rallycut-training", str(model_dir) + "/", "base_model/"
        ]
        rprint(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            rprint("[green]Model weights uploaded successfully![/green]")
            rprint()
            rprint("Now run training with --resume-from-model flag:")
            rprint("  [cyan]rallycut train modal --resume-from-model --lr 1e-5 --epochs 5[/cyan]")
        else:
            rprint(f"[red]Upload failed: {result.stderr}[/red]")
        return

    if upload:
        rprint("[bold]Uploading training data to Modal...[/bold]")
        # Use modal volume put command with --force to overwrite existing files
        cmd = ["python3", "-m", "modal", "volume", "put", "-f", "rallycut-training", "training_data/", "training_data/"]
        rprint(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            rprint("[green]Training data uploaded successfully![/green]")
        else:
            rprint(f"[red]Upload failed: {result.stderr}[/red]")
        return

    if upload_videos:
        import concurrent.futures

        rprint("[bold]Uploading proxy videos to Modal (parallel)...[/bold]")
        # Get video paths from training data
        video_paths_file = Path("training_data/video_paths.json")
        if not video_paths_file.exists():
            rprint("[red]training_data/video_paths.json not found. Run 'rallycut train prepare' first.[/red]")
            raise typer.Exit(1)

        with open(video_paths_file) as f:
            video_paths = json.load(f)

        # Prepare upload tasks
        upload_tasks: list[tuple[Path, str]] = []
        total_size_mb = 0
        for vid, path in video_paths.items():
            local_path = Path(path)
            if local_path.exists():
                remote_path = f"videos/{local_path.name}"
                upload_tasks.append((local_path, remote_path))
                total_size_mb += local_path.stat().st_size // (1024 * 1024)
            else:
                rprint(f"[yellow]Skipping {path} (not found)[/yellow]")

        rprint(f"Uploading {len(upload_tasks)} proxy videos ({total_size_mb} MB total)...")

        def upload_file(task: tuple[Path, str]) -> tuple[str, bool, str]:
            local_path, remote_path = task
            cmd = ["python3", "-m", "modal", "volume", "put", "-f", "rallycut-training", str(local_path), remote_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return local_path.name, result.returncode == 0, result.stderr

        # Upload in parallel (3 concurrent uploads to avoid overwhelming network)
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(upload_file, task): task for task in upload_tasks}
            for future in concurrent.futures.as_completed(futures):
                name, success, error = future.result()
                if success:
                    success_count += 1
                    rprint(f"[green]✓[/green] {name} ({success_count}/{len(upload_tasks)})")
                else:
                    rprint(f"[red]✗[/red] {name}: {error}")

        if success_count == len(upload_tasks):
            rprint(f"[green]All {success_count} videos uploaded successfully![/green]")
        else:
            rprint(f"[yellow]Uploaded {success_count}/{len(upload_tasks)} videos[/yellow]")
        return

    if download:
        import shutil
        import tempfile

        rprint("[bold]Downloading trained model from Modal...[/bold]")
        output_dir = Path("weights/videomae/beach_volleyball")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download to temp dir first, then move files to output_dir.
        # Modal volume get creates a best/ subdirectory when downloading
        # models/beach_volleyball/best/ — we want files at the root.
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                "python3", "-m", "modal",
                "volume",
                "get",
                "--force",
                "rallycut-training",
                "models/beach_volleyball/best/",
                tmp_dir + "/",
            ]
            rprint(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                rprint(f"[red]Download failed: {result.stderr}[/red]")
                return

            # Modal may create a best/ subdirectory — flatten if needed
            best_subdir = Path(tmp_dir) / "best"
            source_dir = best_subdir if best_subdir.is_dir() else Path(tmp_dir)

            # Move all model files to output directory
            for model_file in source_dir.iterdir():
                if model_file.is_file():
                    shutil.copy2(str(model_file), str(output_dir / model_file.name))

        rprint(f"[green]Model downloaded to {output_dir}[/green]")
        rprint()
        rprint(
            "[yellow]Tip: Run 'rallycut train modal --cleanup' to delete videos from Modal\n"
            "and save storage costs (~$0.75/GB/month).[/yellow]"
        )
        return

    if cleanup:
        rprint("[bold]Cleaning up Modal volume...[/bold]")

        # Delete videos folder
        rprint("Deleting videos from Modal volume...")
        cmd = ["python3", "-m", "modal", "volume", "rm", "rallycut-training", "videos/", "-r"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            rprint("[green]✓ Videos deleted[/green]")
        else:
            # Folder might not exist if already deleted
            rprint("[yellow]Videos folder not found or already deleted[/yellow]")

        # Delete trained model folder
        rprint("Deleting model outputs from Modal volume...")
        cmd = ["python3", "-m", "modal", "volume", "rm", "rallycut-training", "models/", "-r"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            rprint("[green]✓ Model outputs deleted[/green]")
        else:
            rprint("[yellow]Model folder not found or already deleted[/yellow]")

        # Delete base model folder (uploaded for incremental training)
        rprint("Deleting base model from Modal volume...")
        cmd = ["python3", "-m", "modal", "volume", "rm", "rallycut-training", "base_model/", "-r"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            rprint("[green]✓ Base model deleted[/green]")
        else:
            rprint("[yellow]Base model folder not found or already deleted[/yellow]")

        rprint()
        rprint("[green]Cleanup complete![/green]")
        rprint("Training data JSON kept for reference (~4KB).")
        rprint("Proxy videos remain cached locally at: ~/.cache/rallycut/proxies/")
        return

    # Run training on Modal
    rprint("[bold]Starting training on Modal T4 GPU...[/bold]")
    rprint(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    if fresh:
        rprint("[yellow]Fresh training: ignoring existing checkpoints[/yellow]")
    if resume_from_model:
        rprint("[cyan]Resuming from existing beach volleyball model weights[/cyan]")
    rprint()

    cmd = [
        "python3", "-m", "modal",
        "run",
        "--detach",  # Keep running even if local client disconnects
        "rallycut/training/modal_train.py",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
    ]

    if resume_from_model:
        cmd.append("--resume-from-model")

    if fresh:
        cmd.append("--fresh")

    rprint(f"Running: {' '.join(cmd)}")
    rprint()

    # Run interactively so user sees progress
    subprocess.run(cmd)


@app.command()
def info(
    data: Path = typer.Option(
        Path("training_data"),
        "--data",
        "-d",
        help="Path to prepared training data",
    ),
) -> None:
    """Show information about prepared training data."""
    if not data.exists():
        rprint(f"[red]Data directory not found: {data}[/red]")
        raise typer.Exit(1)

    metadata_file = data / "metadata.json"
    if not metadata_file.exists():
        rprint(f"[red]Metadata not found in: {data}[/red]")
        raise typer.Exit(1)

    with open(metadata_file) as f:
        metadata = json.load(f)

    rprint("[bold]Training Data Info[/bold]")
    rprint()

    table = Table()
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Videos", str(metadata["total_videos"]))
    table.add_row("Ground Truth Rallies", str(metadata["total_rallies"]))
    table.add_row("Total Samples", str(metadata["total_samples"]))
    table.add_row("Train Samples", str(metadata["train_samples"]))
    table.add_row("Val Samples", str(metadata["val_samples"]))
    table.add_row("Seed", str(metadata["seed"]))

    console.print(table)

    rprint()
    rprint("[bold]Class Distribution[/bold]")

    dist_table = Table()
    dist_table.add_column("Class", style="cyan")
    dist_table.add_column("Count", justify="right", style="green")

    dist = metadata.get("class_distribution", {})
    for label in ["NO_PLAY", "PLAY", "SERVICE"]:
        dist_table.add_row(label, str(dist.get(label, 0)))

    console.print(dist_table)


@app.command("export-dataset")
def export_dataset(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Dataset name (e.g., 'beach_v1')",
    ),
    output: Path = typer.Option(
        Path("training_datasets"),
        "--output",
        "-o",
        help="Output directory for dataset",
    ),
    copy_videos: bool = typer.Option(
        False,
        "--copy-videos",
        help="Copy video files (default: symlink to cache)",
    ),
) -> None:
    """Export ground truth from database to local dataset directory.

    Creates an organized dataset with:
    - manifest.json: Dataset metadata
    - ground_truth.json: Rally annotations per video
    - videos/: Video files (symlinked or copied)

    Example:
        rallycut train export-dataset --name beach_v1
    """
    import shutil
    from datetime import datetime

    rprint("[bold]Exporting Dataset[/bold]")
    rprint()

    # Load ground truth from database
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading ground truth from database...", total=None)
        videos = load_evaluation_videos(require_ground_truth=True)

    if not videos:
        rprint("[red]No videos with ground truth found![/red]")
        raise typer.Exit(1)

    rprint(f"Found [green]{len(videos)}[/green] videos with ground truth")

    total_rallies = sum(len(v.ground_truth_rallies) for v in videos)
    rprint(f"Total ground truth rallies: [green]{total_rallies}[/green]")
    rprint()

    # Create output directory
    dataset_dir = output / name
    videos_dir = dataset_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Download/link videos
    rprint("[bold]Resolving videos...[/bold]")
    resolver = VideoResolver()
    video_info_list: list[dict[str, str | int]] = []

    with Progress(console=console) as progress:
        task = progress.add_task("Processing videos...", total=len(videos))
        for video in videos:
            try:
                local_path = resolver.resolve(video.s3_key, video.content_hash)

                # Create symlink or copy to dataset directory
                target_path = videos_dir / video.filename
                if target_path.exists():
                    target_path.unlink()

                if copy_videos:
                    shutil.copy2(local_path, target_path)
                else:
                    target_path.symlink_to(local_path.resolve())

                # Get video duration
                from rallycut.core.video import Video as VideoReader
                with VideoReader(local_path) as v:
                    duration_ms = int(v.info.duration * 1000)

                video_info_list.append({
                    "filename": video.filename,
                    "video_id": video.id,
                    "content_hash": video.content_hash,
                    "duration_ms": duration_ms,
                    "rally_count": len(video.ground_truth_rallies),
                })

                progress.advance(task)
            except Exception as e:
                rprint(f"[red]Failed to process {video.filename}: {e}[/red]")

    rprint(f"Processed [green]{len(video_info_list)}[/green] videos")
    rprint()

    # Generate manifest.json
    total_duration_ms = sum(int(v["duration_ms"]) for v in video_info_list)
    total_duration_min = round(total_duration_ms / 60000, 1)
    manifest = {
        "name": name,
        "description": f"Training dataset: {name}",
        "created": datetime.now().isoformat(),
        "videos": video_info_list,
        "stats": {
            "total_videos": len(video_info_list),
            "total_rallies": total_rallies,
            "total_duration_min": total_duration_min,
        },
    }

    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    rprint(f"Created [cyan]{manifest_path}[/cyan]")

    # Generate ground_truth.json
    ground_truth_data = []
    for video in videos:
        rallies = [
            {
                "start_ms": int(r.start_seconds * 1000),
                "end_ms": int(r.end_seconds * 1000),
            }
            for r in video.ground_truth_rallies
        ]
        ground_truth_data.append({
            "video_id": video.id,
            "filename": video.filename,
            "rallies": rallies,
        })

    ground_truth_path = dataset_dir / "ground_truth.json"
    with open(ground_truth_path, "w") as f:
        json.dump(ground_truth_data, f, indent=2)
    rprint(f"Created [cyan]{ground_truth_path}[/cyan]")

    # Summary
    rprint()
    rprint("[green]Dataset exported successfully![/green]")
    rprint()
    rprint(f"  Location: [cyan]{dataset_dir}[/cyan]")
    rprint(f"  Videos: {len(video_info_list)}")
    rprint(f"  Rallies: {total_rallies}")
    rprint(f"  Duration: {total_duration_min} minutes")
    rprint()
    rprint("Next steps:")
    rprint(f"  1. [cyan]cd {dataset_dir}[/cyan]")
    rprint("  2. [cyan]git add manifest.json ground_truth.json[/cyan]")
    rprint("  3. [cyan]rallycut train push --name {name}[/cyan]  (back up to S3)")
    rprint("  4. [cyan]rallycut train prepare --output ../training_data[/cyan]")


@app.command()
def push(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Dataset name (must exist in training_datasets/)",
    ),
    datasets_dir: Path = typer.Option(
        Path("training_datasets"),
        "--datasets-dir",
        help="Base directory containing datasets",
    ),
) -> None:
    """Push a dataset to S3 for backup.

    Uploads manifest.json, ground_truth.json, and video files.
    Videos are deduplicated by content_hash -- only new videos are uploaded.

    Example:
        rallycut train push --name beach_v2
    """
    from rallycut.training.backup import DatasetBackup, make_transfer_progress

    dataset_dir = datasets_dir / name

    # Validate dataset exists
    if not (dataset_dir / "manifest.json").exists():
        rprint(f"[red]Dataset not found: {dataset_dir}/manifest.json[/red]")
        rprint(f"Run [cyan]rallycut train export-dataset --name {name}[/cyan] first")
        raise typer.Exit(1)

    if not (dataset_dir / "ground_truth.json").exists():
        rprint(f"[red]Missing: {dataset_dir}/ground_truth.json[/red]")
        raise typer.Exit(1)

    rprint("[bold]Pushing Dataset to S3[/bold]")
    rprint()

    try:
        backup = DatasetBackup()
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Load manifest for summary
    with open(dataset_dir / "manifest.json") as f:
        manifest = json.load(f)

    video_count = len(manifest.get("videos", []))
    rprint(f"  Dataset: [cyan]{name}[/cyan]")
    rprint(f"  Videos: [cyan]{video_count}[/cyan]")
    rprint(f"  Bucket: [cyan]{backup.bucket}[/cyan]")
    rprint()

    with make_transfer_progress() as progress:
        result = backup.push_dataset(dataset_dir, name, progress=progress)

    rprint()
    rprint("[green]Push complete![/green]")
    rprint(f"  Uploaded: [green]{result.uploaded_videos}[/green] videos ({_human_size(result.uploaded_bytes)})")
    rprint(f"  Skipped: [yellow]{result.skipped_videos}[/yellow] videos (already in S3)")

    if result.errors:
        rprint()
        rprint("[red]Errors:[/red]")
        for err in result.errors:
            rprint(f"  - {err}")


@app.command()
def pull(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Dataset name to download from S3",
    ),
    output: Path = typer.Option(
        Path("training_datasets"),
        "--output",
        "-o",
        help="Output directory for dataset",
    ),
) -> None:
    """Pull a dataset from S3.

    Downloads manifest.json, ground_truth.json, and video files.
    Videos are cached at ~/.cache/rallycut/evaluation/ and symlinked
    into the dataset directory.

    Example:
        rallycut train pull --name beach_v2
    """
    from rallycut.training.backup import DatasetBackup, make_transfer_progress

    rprint("[bold]Pulling Dataset from S3[/bold]")
    rprint()

    try:
        backup = DatasetBackup()
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)

    rprint(f"  Dataset: [cyan]{name}[/cyan]")
    rprint(f"  Bucket: [cyan]{backup.bucket}[/cyan]")
    rprint(f"  Output: [cyan]{output / name}[/cyan]")
    rprint()

    try:
        with make_transfer_progress() as progress:
            result = backup.pull_dataset(name, output, progress=progress)
    except Exception as e:
        rprint(f"[red]Pull failed: {e}[/red]")
        raise typer.Exit(1)

    rprint()
    rprint("[green]Pull complete![/green]")
    rprint(f"  Downloaded: [green]{result.downloaded_videos}[/green] videos ({_human_size(result.downloaded_bytes)})")
    rprint(f"  Cached: [yellow]{result.cached_videos}[/yellow] videos (already local)")

    if result.errors:
        rprint()
        rprint("[red]Errors:[/red]")
        for err in result.errors:
            rprint(f"  - {err}")

    rprint()
    rprint("Next steps:")
    rprint(f"  [cyan]rallycut train restore --name {name}[/cyan]  (import into DB)")


@app.command()
def restore(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Dataset name to restore",
    ),
    datasets_dir: Path = typer.Option(
        Path("training_datasets"),
        "--datasets-dir",
        help="Base directory containing datasets",
    ),
    user_id: str | None = typer.Option(
        None,
        "--user-id",
        help="User ID to assign videos to (auto-detects if omitted)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview what would happen without making changes",
    ),
    upload_to_app_s3: bool = typer.Option(
        False,
        "--upload-to-app-s3",
        help="Also upload videos to the app's MinIO/S3",
    ),
) -> None:
    """Restore a dataset into the application database.

    Inserts videos and rally annotations from a local dataset directory
    into PostgreSQL, creating a session for web editor access.

    Example:
        rallycut train restore --name beach_v2
        rallycut train restore --name beach_v2 --upload-to-app-s3
        rallycut train restore --name beach_v2 --dry-run
    """
    from rallycut.training.restore import restore_dataset_to_db

    dataset_dir = datasets_dir / name

    if not (dataset_dir / "manifest.json").exists():
        rprint(f"[red]Dataset not found: {dataset_dir}/manifest.json[/red]")
        rprint(f"Run [cyan]rallycut train pull --name {name}[/cyan] first")
        raise typer.Exit(1)

    rprint("[bold]Restoring Dataset to Database[/bold]")
    rprint()

    try:
        result = restore_dataset_to_db(
            dataset_dir=dataset_dir,
            name=name,
            user_id=user_id,
            dry_run=dry_run,
            upload_to_app_s3=upload_to_app_s3,
        )
    except Exception as e:
        rprint(f"[red]Restore failed: {e}[/red]")
        raise typer.Exit(1)

    if dry_run:
        return

    rprint()
    rprint("[green]Restore complete![/green]")
    rprint(f"  Videos inserted: [green]{result.videos_inserted}[/green]")
    rprint(f"  Videos skipped: [yellow]{result.videos_skipped}[/yellow] (already in DB)")
    rprint(f"  Rallies inserted: [green]{result.rallies_inserted}[/green]")
    rprint(f"  Session: [cyan]{result.session_created}[/cyan]")

    if result.errors:
        rprint()
        rprint("[red]Errors:[/red]")
        for err in result.errors:
            rprint(f"  - {err}")


@app.command("list-remote")
def list_remote() -> None:
    """List datasets backed up in S3.

    Example:
        rallycut train list-remote
    """
    from rallycut.training.backup import DatasetBackup

    try:
        backup = DatasetBackup()
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)

    rprint("[bold]Remote Datasets[/bold]")
    rprint(f"  Bucket: [cyan]{backup.bucket}[/cyan]")
    rprint()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Listing datasets...", total=None)
        datasets = backup.list_datasets()

    if not datasets:
        rprint("[yellow]No datasets found in S3[/yellow]")
        return

    table = Table(title="Backed-up Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Videos", justify="right", style="green")
    table.add_column("Rallies", justify="right", style="green")
    table.add_column("Duration", justify="right")
    table.add_column("Created")

    for ds in datasets:
        duration = f"{ds.total_duration_min:.1f} min" if ds.total_duration_min else "-"
        created = ds.created[:10] if ds.created else "-"
        table.add_row(
            ds.name,
            str(ds.video_count),
            str(ds.rally_count),
            duration,
            created,
        )

    console.print(table)


def _human_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    size_f = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_f < 1024:
            return f"{size_f:.1f} {unit}"
        size_f /= 1024
    return f"{size_f:.1f} TB"
