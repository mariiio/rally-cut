"""CLI commands for training VideoMAE on beach volleyball."""

from __future__ import annotations

import json
import random
from pathlib import Path

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

app = typer.Typer(help="Train VideoMAE model on beach volleyball data")
console = Console()


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
    resolver = VideoResolver()
    proxy_gen = ProxyGenerator()
    video_paths: dict[str, Path] = {}

    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(videos))
        for video in videos:
            try:
                # Download full video
                local_path = resolver.resolve(video.s3_key, video.content_hash)

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

    rprint()
    rprint(f"[green]Data saved to: {output}[/green]")
    rprint()
    rprint("Next step: [cyan]rallycut train run --data training_data/[/cyan]")


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
            "python3", "-m", "modal", "volume", "put",
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
        # Use modal volume put command
        cmd = ["python3", "-m", "modal", "volume", "put", "rallycut-training", "training_data/", "training_data/"]
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
            cmd = ["python3", "-m", "modal", "volume", "put", "rallycut-training", str(local_path), remote_path]
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
        rprint("[bold]Downloading trained model from Modal...[/bold]")
        output_dir = Path("weights/videomae/beach_volleyball")
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python3", "-m", "modal",
            "volume",
            "get",
            "rallycut-training",
            "models/beach_volleyball/best/",
            str(output_dir) + "/",
        ]
        rprint(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            rprint(f"[green]Model downloaded to {output_dir}[/green]")
            rprint()
            rprint(
                "[yellow]Tip: Run 'rallycut train modal --cleanup' to delete videos from Modal\n"
                "and save storage costs (~$0.75/GB/month).[/yellow]"
            )
        else:
            rprint(f"[red]Download failed: {result.stderr}[/red]")
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
    rprint("  3. [cyan]rallycut train prepare --output ../training_data[/cyan]")
