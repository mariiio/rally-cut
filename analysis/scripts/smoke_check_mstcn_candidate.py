"""Smoke-check a fresh MS-TCN++ checkpoint BEFORE promoting to production.

Loads the given checkpoint, runs inference on the first 3 GT-labeled rallies
(deterministic by rally_id sort), and verifies:
  1. Output shape is (NUM_CLASSES, T) with T matching the rally.
  2. Each frame's probabilities sum to ~1.0 (softmax correctness).
  3. At least one GT-action frame has non-bg max in a sane range [0.10, 1.00].
  4. All 7 class indices receive probability mass across the 3 rallies
     (no class collapse).

Exit 0 on all checks passing, 1 otherwise. Prints per-rally diagnostics so
anomalies are visible rather than hidden.

Usage:
    cd analysis
    uv run python scripts/smoke_check_mstcn_candidate.py \\
        --weights weights/sequence_action/ms_tcn_v5_candidate.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from eval_sequence_enriched import prepare_rallies  # noqa: E402

from rallycut.actions.trajectory_features import (  # noqa: E402
    ACTION_TYPES,
    NUM_CLASSES,
)
from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig  # noqa: E402

console = Console()

# ACTION_TYPES is the 6 non-background labels; class 0 is background.
CLASS_NAMES = ["background", *ACTION_TYPES]
assert len(CLASS_NAMES) == NUM_CLASSES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to MS-TCN++ checkpoint (.pt) to smoke-check.",
    )
    parser.add_argument(
        "--n-rallies",
        type=int,
        default=3,
        help="Number of GT-labeled rallies to probe (default 3).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.weights.exists():
        console.print(f"[red]Weights not found: {args.weights}[/red]")
        return 1

    console.print(f"[bold]Smoke-check:[/bold] {args.weights}")
    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    config = MSTCNConfig(**ckpt["config"])
    model = MSTCN(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    console.print(f"  Loaded config: {asdict(config)}")

    # Load rallies deterministically (prepare_rallies reads from DB; we'll take
    # the first n after sorting by rally_id for reproducibility).
    bundles = prepare_rallies(label_spread=2)
    if not bundles:
        console.print("[red]No rallies returned by prepare_rallies[/red]")
        return 1
    bundles = sorted(bundles, key=lambda b: b.rally.rally_id)[: args.n_rallies]
    console.print(f"  Probing {len(bundles)} rallies (sorted by rally_id)")

    failures: list[str] = []
    class_mass_total = np.zeros(NUM_CLASSES, dtype=np.float64)

    diag = Table(title="Per-rally smoke check")
    diag.add_column("Rally")
    diag.add_column("T")
    diag.add_column("Shape OK")
    diag.add_column("Σprobs≈1")
    diag.add_column("GT-frame non-bg max")
    diag.add_column("Argmax dist (non-bg frames)")

    for bundle in bundles:
        feat = (
            torch.from_numpy(bundle.trajectory_features)
            .float()
            .unsqueeze(0)
            .transpose(1, 2)
            .to(device)
        )
        T = feat.shape[2]
        mask = torch.ones(1, 1, T, device=device)

        with torch.no_grad():
            logits = model(feat, mask)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C, T)

        # Check 1: shape
        shape_ok = probs.shape == (NUM_CLASSES, T)
        if not shape_ok:
            failures.append(
                f"{bundle.rally.rally_id}: shape {probs.shape} != ({NUM_CLASSES},{T})"
            )

        # Check 2: softmax sanity — column sums ~1
        col_sums = probs.sum(axis=0)
        sum_ok = bool(np.allclose(col_sums, 1.0, atol=1e-4))
        if not sum_ok:
            failures.append(
                f"{bundle.rally.rally_id}: softmax col sums "
                f"min={col_sums.min():.4f} max={col_sums.max():.4f}"
            )

        # Check 3: per-class max non-bg prob at GT-action frames
        gt_frames = np.where(bundle.trajectory_labels > 0)[0]
        if len(gt_frames) > 0:
            nonbg_at_gt = probs[1:, gt_frames].max(axis=0)
            nonbg_stats = (
                f"min={nonbg_at_gt.min():.3f} med={np.median(nonbg_at_gt):.3f} "
                f"max={nonbg_at_gt.max():.3f} n={len(gt_frames)}"
            )
            if float(nonbg_at_gt.max()) < 0.10:
                failures.append(
                    f"{bundle.rally.rally_id}: non-bg max across GT frames < 0.10"
                )
        else:
            nonbg_stats = "(no GT action frames)"

        # Argmax distribution on non-bg frames for sanity
        argmax = probs.argmax(axis=0)
        nonbg_argmax = argmax[argmax > 0]
        if len(nonbg_argmax) > 0:
            counts = np.bincount(nonbg_argmax, minlength=NUM_CLASSES)
            dist_str = " ".join(
                f"{CLASS_NAMES[i]}:{counts[i]}"
                for i in range(1, NUM_CLASSES)
                if counts[i] > 0
            )
        else:
            dist_str = "(all background)"

        # Accumulate per-class mass for global "no class collapse" check
        class_mass_total += probs.sum(axis=1)

        diag.add_row(
            bundle.rally.rally_id[:12],
            str(T),
            "OK" if shape_ok else "FAIL",
            "OK" if sum_ok else "FAIL",
            nonbg_stats,
            dist_str,
        )

    console.print(diag)

    # Check 4: every class receives non-trivial mass
    console.print("\n[bold]Global per-class cumulative probability mass[/bold]")
    mass_table = Table()
    mass_table.add_column("Class")
    mass_table.add_column("Total mass")
    mass_table.add_column("Status")
    total = class_mass_total.sum()
    for i, name in enumerate(CLASS_NAMES):
        frac = class_mass_total[i] / total if total > 0 else 0.0
        status = "OK" if frac > 1e-4 else "COLLAPSED"
        if frac <= 1e-4:
            failures.append(f"class {name} has near-zero mass (frac={frac:.6f})")
        mass_table.add_row(name, f"{class_mass_total[i]:.1f}", f"{frac:.4f} [{status}]")
    console.print(mass_table)

    if failures:
        console.print("\n[red bold]SMOKE-CHECK FAIL[/red bold]")
        for f in failures:
            console.print(f"  • {f}")
        return 1

    console.print("\n[green bold]Smoke-check passed.[/green bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
