#!/usr/bin/env python3
"""Experiment 1/6: Set up TrackNetV2 training environment.

Downloads TrackNetV2 repository, verifies dependencies, and exports
pseudo-labels from VballNet in TrackNet CSV format for fine-tuning.

Usage:
    # Set up training environment
    uv run python scripts/setup_tracknet_training.py --setup

    # Export pseudo-labels for training
    uv run python scripts/setup_tracknet_training.py --export-labels

    # Export gold-only labels for validation
    uv run python scripts/setup_tracknet_training.py --export-gold

    # Full setup + export
    uv run python scripts/setup_tracknet_training.py --setup --export-labels
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Add analysis root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRACKNET_REPO = "https://github.com/ChgygLin/TrackNetV2-pytorch.git"
TRACKNET_DIR = Path(__file__).parent.parent / "vendor" / "TrackNetV2-pytorch"
PSEUDO_LABELS_DIR = Path(__file__).parent.parent / "experiments" / "pseudo_labels"
GOLD_LABELS_DIR = Path(__file__).parent.parent / "experiments" / "gold_labels"


def setup_tracknet() -> bool:
    """Clone TrackNetV2 repo and verify it's usable.

    Returns:
        True if setup succeeded.
    """
    print("Setting up TrackNetV2 training environment...")
    print(f"  Target directory: {TRACKNET_DIR}")

    if TRACKNET_DIR.exists():
        print("  TrackNetV2 repo already exists, checking...")
        # Verify key files exist
        model_file = TRACKNET_DIR / "src" / "model.py"

        if not model_file.exists():
            # Try alternate structure
            model_file = TRACKNET_DIR / "model.py"

        if model_file.exists():
            print(f"  Found model file: {model_file}")
            print("  TrackNetV2 repo is ready.")
            return True
        else:
            print("  Warning: model file not found, repo may be incomplete")
            return True  # Still proceed
    else:
        print(f"  Cloning {TRACKNET_REPO}...")
        TRACKNET_DIR.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", TRACKNET_REPO, str(TRACKNET_DIR)],
                check=True,
                capture_output=True,
                text=True,
            )
            print("  Clone successful!")
        except subprocess.CalledProcessError as e:
            print(f"  Clone failed: {e.stderr}")
            return False

    # List repo structure
    print("\n  Repository structure:")
    for item in sorted(TRACKNET_DIR.iterdir()):
        prefix = "  " if item.is_file() else "  [dir]"
        print(f"    {prefix} {item.name}")

    # Check for key files
    key_files = ["model.py", "train.py", "dataset.py"]
    found_files = []
    for pattern in ["**/" + f for f in key_files]:
        matches = list(TRACKNET_DIR.glob(pattern))
        found_files.extend(matches)

    if found_files:
        print("\n  Key training files found:")
        for f in found_files:
            print(f"    {f.relative_to(TRACKNET_DIR)}")
    else:
        print("\n  Warning: Key training files not found. Check repo structure.")

    return True


def export_pseudo_labels() -> None:
    """Export pseudo-labels from VballNet for TrackNet training."""
    from experiments.pseudo_label_export import (
        ExportStats,
        PseudoLabelConfig,
        export_pseudo_labels,
    )

    print(f"\nExporting pseudo-labels to {PSEUDO_LABELS_DIR}...")

    config = PseudoLabelConfig(
        min_confidence=0.3,
        min_motion_energy=0.02,
    )

    stats: ExportStats = export_pseudo_labels(
        output_dir=PSEUDO_LABELS_DIR,
        config=config,
        include_gold=True,
    )

    print("\n  Export complete:")
    print(f"    Total frames:         {stats.total_frames}")
    print(f"    Visible (labeled):    {stats.visible_frames}")
    print(f"    Gold GT frames:       {stats.gold_label_frames}")
    print(f"    Filtered (low conf):  {stats.filtered_low_confidence}")
    print(f"    Filtered (low motion):{stats.filtered_low_motion}")

    # List exported files
    csv_files = sorted(PSEUDO_LABELS_DIR.glob("*.csv"))
    if csv_files:
        print(f"\n  Exported {len(csv_files)} rally CSV files:")
        for f in csv_files:
            print(f"    {f.name}")


def export_gold_labels() -> None:
    """Export gold-only labels for validation."""
    from experiments.pseudo_label_export import PseudoLabelConfig, export_gold_only

    print(f"\nExporting gold-only labels to {GOLD_LABELS_DIR}...")

    config = PseudoLabelConfig()
    stats = export_gold_only(
        output_dir=GOLD_LABELS_DIR,
        config=config,
    )

    print("\n  Export complete:")
    print(f"    Total frames:     {stats.total_frames}")
    print(f"    Gold GT frames:   {stats.gold_label_frames}")

    csv_files = sorted(GOLD_LABELS_DIR.glob("*.csv"))
    if csv_files:
        print(f"\n  Exported {len(csv_files)} gold CSV files:")
        for f in csv_files:
            print(f"    {f.name}")


def verify_data_split() -> None:
    """Verify pseudo-label and gold label data for training readiness."""
    print("\nVerifying training data...")

    pseudo_csvs = sorted(PSEUDO_LABELS_DIR.glob("*.csv")) if PSEUDO_LABELS_DIR.exists() else []
    gold_csvs = sorted(GOLD_LABELS_DIR.glob("*.csv")) if GOLD_LABELS_DIR.exists() else []

    print(f"  Pseudo-label CSVs: {len(pseudo_csvs)}")
    print(f"  Gold-only CSVs:    {len(gold_csvs)}")

    if not pseudo_csvs:
        print("  No pseudo-labels found. Run --export-labels first.")
        return

    # Count frames in pseudo-labels
    total_frames = 0
    visible_frames = 0

    for csv_path in pseudo_csvs:
        with open(csv_path) as f:
            lines = f.readlines()[1:]  # Skip header
            total_frames += len(lines)
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[1] == "1":
                    visible_frames += 1

    print("\n  Pseudo-label stats:")
    print(f"    Total frames:   {total_frames}")
    print(f"    Visible frames: {visible_frames}")
    print(f"    Visibility:     {visible_frames/total_frames:.1%}" if total_frames > 0 else "")

    # Check minimum data requirements
    min_frames = 2000
    if visible_frames >= min_frames:
        print(f"\n  Data meets minimum requirement ({min_frames} visible frames)")
    else:
        print(
            f"\n  Warning: Only {visible_frames} visible frames. "
            f"Minimum {min_frames} recommended for fine-tuning."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up TrackNetV2 training environment"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Clone TrackNetV2 repo and verify setup",
    )
    parser.add_argument(
        "--export-labels", action="store_true",
        help="Export pseudo-labels from VballNet",
    )
    parser.add_argument(
        "--export-gold", action="store_true",
        help="Export gold-only labels for validation",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify training data readiness",
    )

    args = parser.parse_args()

    if not (args.setup or args.export_labels or args.export_gold or args.verify):
        parser.print_help()
        print("\nQuick start:")
        print("  uv run python scripts/setup_tracknet_training.py --setup --export-labels")
        return

    if args.setup:
        if not setup_tracknet():
            sys.exit(1)

    if args.export_labels:
        export_pseudo_labels()

    if args.export_gold:
        export_gold_labels()

    if args.verify or args.export_labels:
        verify_data_split()

    print("\nDone!")


if __name__ == "__main__":
    main()
