"""Evaluate a trained E2E-Spot model on held-out beach volleyball data.

Usage:
    cd analysis
    uv run python scripts/spot_evaluate.py
    uv run python scripts/spot_evaluate.py --checkpoint weights/spotting/best.pt
    uv run python scripts/spot_evaluate.py --tolerance-ms 100  # Stricter matching
"""

from __future__ import annotations

import argparse
import time

import torch
from rich.console import Console

from rallycut.spotting.config import E2ESpotConfig
from rallycut.spotting.data.beach import load_beach_rallies
from rallycut.spotting.evaluation.bridge import evaluate_model, print_metrics
from rallycut.spotting.model.e2e_spot import E2ESpot
from rallycut.spotting.training.trainer import WEIGHTS_DIR

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E2E-Spot on beach volleyball")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(WEIGHTS_DIR / "best.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="held_out",
        choices=["train", "held_out", "all"],
        help="Which split to evaluate (default: held_out)",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=int,
        default=167,
        help="Frame matching tolerance in ms (default: 167 = 5 frames @ 30fps)",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames if not already on disk",
    )
    parser.add_argument("--clip-length", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Min confidence for event detection (default: 0.3)",
    )
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    console.print(f"[bold]E2E-Spot Evaluation[/]  device={device}")

    # Build config
    config = E2ESpotConfig()
    config.training.clip_length = args.clip_length
    config.temporal.hidden_dim = args.hidden_dim
    config.temporal.num_layers = args.num_layers
    config.postprocess.confidence_threshold = args.confidence_threshold

    # Load model
    console.print(f"\n  Loading checkpoint: {args.checkpoint}")
    model = E2ESpot(config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"  Model: {param_count:,} params")

    # Load rallies
    console.print(f"\n[bold]Loading {args.split} rallies[/]")
    rallies = load_beach_rallies(
        split=args.split,
        extract_frames=args.extract_frames,
    )

    if not rallies:
        console.print("[red]No rallies found. Run with --extract-frames first.[/]")
        return

    # Evaluate
    console.print(f"\n[bold]Evaluating on {len(rallies)} rallies[/]")
    start_time = time.time()
    metrics = evaluate_model(model, rallies, device, config, args.tolerance_ms)
    elapsed = time.time() - start_time

    print_metrics(metrics)
    console.print(f"\n  Evaluation time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
