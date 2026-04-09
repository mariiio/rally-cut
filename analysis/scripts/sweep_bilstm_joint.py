"""BiLSTM hyperparameter sweep + feature ablation.

One-time tuning to find optimal config before scaling data.
Sweep on train split, evaluate on held-out (hash-based 75/25).

Usage:
    cd analysis
    uv run python scripts/sweep_bilstm_joint.py --all          # Full sweep + ablation (~45 min)
    uv run python scripts/sweep_bilstm_joint.py --sweep        # Sweep only (~40 min)
    uv run python scripts/sweep_bilstm_joint.py --ablation     # Ablation with current defaults
    uv run python scripts/sweep_bilstm_joint.py --ablation --hidden-dim 128 --dropout 0.3
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.split import video_split
from scripts.eval_bilstm_joint import (
    ContactSample,
    RallySequence,
    _normalize_features,
    evaluate_bilstm,
    load_all_sequences,
    train_bilstm,
)

console = Console()

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

HIDDEN_SIZES = [64, 128, 256]
LAYER_COUNTS = [1, 2]
DROPOUT_VALUES = [0.1, 0.3, 0.5]
LEARNING_RATES = [1e-3, 3e-4, 1e-4]
SEEDS = [42, 123, 7]

# ---------------------------------------------------------------------------
# Feature ablation groups — global dim indices into the 60-dim feature vector
# Action features: dims 0-19, Attribution: dims 20-55, Team: dims 56-59
# ---------------------------------------------------------------------------

ABLATION_GROUPS: dict[str, list[int]] = {
    # contact_index_in_rally + contact_count_on_current_side (action feats)
    "action_seq_context": [8, 9],
    # contact_index + side_count (attribution feats)
    "attr_seq_context": [54, 55],
    # Team assignment per canonical slot
    "team_features": [56, 57, 58, 59],
    # dist_ratio_01, dist_ratio_02, dist_margin_01
    "distance_ratios": [48, 49, 50],
    # ball_speed, ball_dir_change, ball_y_at_contact
    "ball_dynamics": [51, 52, 53],
    # player_speed per slot
    "player_speed": [44, 45, 46, 47],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConfigResult:
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    attr_accs: list[float] = field(default_factory=list)
    action_accs: list[float] = field(default_factory=list)

    @property
    def mean_attr(self) -> float:
        return float(np.mean(self.attr_accs)) if self.attr_accs else 0.0

    @property
    def std_attr(self) -> float:
        return float(np.std(self.attr_accs)) if len(self.attr_accs) > 1 else 0.0

    @property
    def mean_action(self) -> float:
        return float(np.mean(self.action_accs)) if self.action_accs else 0.0

    @property
    def std_action(self) -> float:
        return float(np.std(self.action_accs)) if len(self.action_accs) > 1 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_sequences(
    all_sequences: list[RallySequence],
) -> tuple[list[RallySequence], list[RallySequence]]:
    """Split into train / held-out using deterministic video hash."""
    train = [s for s in all_sequences if video_split(s.video_id) == "train"]
    test = [s for s in all_sequences if video_split(s.video_id) == "held_out"]
    return train, test


def zero_dims(
    sequences: list[RallySequence],
    dims: list[int],
) -> list[RallySequence]:
    """Return a copy of sequences with specified feature dims zeroed out."""
    result = []
    for seq in sequences:
        new_contacts = []
        for c in seq.contacts:
            feats = c.features.copy()
            feats[dims] = 0.0
            new_contacts.append(ContactSample(
                features=feats,
                action_label=c.action_label,
                attribution_label=c.attribution_label,
                rally_id=c.rally_id,
                video_id=c.video_id,
            ))
        result.append(RallySequence(
            contacts=new_contacts,
            rally_id=seq.rally_id,
            video_id=seq.video_id,
        ))
    return result


def evaluate_config(
    train_norm: list[RallySequence],
    test_norm: list[RallySequence],
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    seeds: list[int],
    max_epochs: int = 300,
    patience: int = 30,
) -> ConfigResult:
    """Train + evaluate one config across multiple seeds."""
    result = ConfigResult(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
    )
    for seed in seeds:
        model = train_bilstm(
            train_norm,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
        )
        action_acc, attr_acc, _, _ = evaluate_bilstm(model, test_norm)
        result.action_accs.append(action_acc)
        result.attr_accs.append(attr_acc)
    return result


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    train_seqs: list[RallySequence],
    test_seqs: list[RallySequence],
    output_dir: str,
    max_epochs: int = 300,
    patience: int = 30,
) -> ConfigResult:
    """Run full hyperparameter grid search."""
    train_norm, test_norm = _normalize_features(train_seqs, test_seqs)

    grid = list(itertools.product(HIDDEN_SIZES, LAYER_COUNTS, DROPOUT_VALUES, LEARNING_RATES))
    n_configs = len(grid)
    console.print(f"\n[bold]Hyperparameter Sweep: {n_configs} configs × {len(SEEDS)} seeds "
                  f"= {n_configs * len(SEEDS)} training runs[/bold]")

    results: list[ConfigResult] = []
    t_start = time.time()

    for i, (hidden, layers, drop, lr) in enumerate(grid):
        t0 = time.time()
        cr = evaluate_config(
            train_norm, test_norm,
            hidden_dim=hidden,
            num_layers=layers,
            dropout=drop,
            lr=lr,
            seeds=SEEDS,
            max_epochs=max_epochs,
            patience=patience,
        )
        elapsed = time.time() - t0
        results.append(cr)
        console.print(
            f"  [{i + 1}/{n_configs}] hidden={hidden} layers={layers} "
            f"drop={drop} lr={lr:.0e} → attr={cr.mean_attr:.1%} "
            f"act={cr.mean_action:.1%} ({elapsed:.0f}s)"
        )

    total_time = time.time() - t_start
    console.print(f"\n  Sweep completed in {total_time / 60:.1f} min")

    # Sort by mean attr acc
    results.sort(key=lambda r: r.mean_attr, reverse=True)

    # Print table
    table = Table(title="Sweep Results (ranked by held-out attr acc)")
    table.add_column("Rank", justify="right")
    table.add_column("Hidden", justify="right")
    table.add_column("Layers", justify="right")
    table.add_column("Dropout", justify="right")
    table.add_column("LR", justify="right")
    table.add_column("Attr Acc", justify="right")
    table.add_column("± Std", justify="right")
    table.add_column("Action Acc", justify="right")
    table.add_column("± Std", justify="right")

    for rank, cr in enumerate(results, 1):
        style = "bold green" if rank == 1 else ("" if rank <= 10 else "dim")
        table.add_row(
            str(rank),
            str(cr.hidden_dim),
            str(cr.num_layers),
            f"{cr.dropout:.1f}",
            f"{cr.lr:.0e}",
            f"{cr.mean_attr:.1%}",
            f"{cr.std_attr:.1%}",
            f"{cr.mean_action:.1%}",
            f"{cr.std_action:.1%}",
            style=style,
        )

    console.print(table)

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "rank", "hidden_dim", "num_layers", "dropout", "lr",
            "mean_attr_acc", "std_attr_acc", "mean_action_acc", "std_action_acc",
        ]
        for si, seed in enumerate(SEEDS):
            header.extend([f"attr_s{si + 1}", f"action_s{si + 1}"])
        w.writerow(header)

        for rank, cr in enumerate(results, 1):
            row = [
                rank, cr.hidden_dim, cr.num_layers, cr.dropout, cr.lr,
                f"{cr.mean_attr:.4f}", f"{cr.std_attr:.4f}",
                f"{cr.mean_action:.4f}", f"{cr.std_action:.4f}",
            ]
            for si in range(len(SEEDS)):
                row.extend([
                    f"{cr.attr_accs[si]:.4f}" if si < len(cr.attr_accs) else "",
                    f"{cr.action_accs[si]:.4f}" if si < len(cr.action_accs) else "",
                ])
            w.writerow(row)

    console.print(f"\n  Results saved to {csv_path}")
    return results[0]


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def run_ablation(
    train_seqs: list[RallySequence],
    test_seqs: list[RallySequence],
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    output_dir: str,
    max_epochs: int = 300,
    patience: int = 30,
) -> None:
    """Run feature ablation: zero each group, measure impact."""
    train_norm, test_norm = _normalize_features(train_seqs, test_seqs)

    n_groups = len(ABLATION_GROUPS) + 1  # +1 for baseline
    console.print(f"\n[bold]Feature Ablation: {n_groups} groups × {len(SEEDS)} seeds "
                  f"= {n_groups * len(SEEDS)} training runs[/bold]")
    console.print(f"  Config: hidden={hidden_dim} layers={num_layers} "
                  f"dropout={dropout} lr={lr:.0e}")

    # Baseline (full model)
    console.print("\n  [1/{n}] full_model (baseline)...".format(n=n_groups))
    t0 = time.time()
    baseline = evaluate_config(
        train_norm, test_norm,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        seeds=SEEDS,
        max_epochs=max_epochs,
        patience=patience,
    )
    console.print(
        f"    → attr={baseline.mean_attr:.1%} act={baseline.mean_action:.1%} "
        f"({time.time() - t0:.0f}s)"
    )

    # Ablation groups
    ablation_results: list[tuple[str, list[int], ConfigResult]] = []
    for gi, (group_name, dims) in enumerate(ABLATION_GROUPS.items()):
        console.print(f"  [{gi + 2}/{n_groups}] {group_name} (zeroing dims {dims})...")
        t0 = time.time()

        train_zeroed = zero_dims(train_norm, dims)
        test_zeroed = zero_dims(test_norm, dims)

        cr = evaluate_config(
            train_zeroed, test_zeroed,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            seeds=SEEDS,
            max_epochs=max_epochs,
            patience=patience,
        )
        ablation_results.append((group_name, dims, cr))
        delta_attr = cr.mean_attr - baseline.mean_attr
        delta_act = cr.mean_action - baseline.mean_action
        console.print(
            f"    → attr={cr.mean_attr:.1%} ({delta_attr:+.1%}) "
            f"act={cr.mean_action:.1%} ({delta_act:+.1%}) "
            f"({time.time() - t0:.0f}s)"
        )

    # Print summary table
    table = Table(title="Feature Ablation Results")
    table.add_column("Group", style="bold")
    table.add_column("Dims Zeroed", justify="right")
    table.add_column("Attr Acc", justify="right")
    table.add_column("Δ Attr", justify="right")
    table.add_column("Action Acc", justify="right")
    table.add_column("Δ Action", justify="right")

    table.add_row(
        "full_model",
        "—",
        f"{baseline.mean_attr:.1%} ± {baseline.std_attr:.1%}",
        "—",
        f"{baseline.mean_action:.1%} ± {baseline.std_action:.1%}",
        "—",
        style="bold",
    )

    for group_name, dims, cr in ablation_results:
        delta_attr = cr.mean_attr - baseline.mean_attr
        delta_act = cr.mean_action - baseline.mean_action
        attr_style = "green" if delta_attr > 0.005 else ("red" if delta_attr < -0.005 else "")
        table.add_row(
            group_name,
            str(len(dims)),
            f"{cr.mean_attr:.1%} ± {cr.std_attr:.1%}",
            f"{delta_attr:+.1%}",
            f"{cr.mean_action:.1%} ± {cr.std_action:.1%}",
            f"{delta_act:+.1%}",
            style=attr_style,
        )

    console.print(table)

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "group", "zeroed_dims", "n_dims",
            "mean_attr_acc", "std_attr_acc", "delta_attr",
            "mean_action_acc", "std_action_acc", "delta_action",
        ])
        w.writerow([
            "full_model", "", 0,
            f"{baseline.mean_attr:.4f}", f"{baseline.std_attr:.4f}", "0.0000",
            f"{baseline.mean_action:.4f}", f"{baseline.std_action:.4f}", "0.0000",
        ])
        for group_name, dims, cr in ablation_results:
            w.writerow([
                group_name,
                ";".join(str(d) for d in dims),
                len(dims),
                f"{cr.mean_attr:.4f}",
                f"{cr.std_attr:.4f}",
                f"{cr.mean_attr - baseline.mean_attr:.4f}",
                f"{cr.mean_action:.4f}",
                f"{cr.std_action:.4f}",
                f"{cr.mean_action - baseline.mean_action:.4f}",
            ])

    console.print(f"\n  Results saved to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BiLSTM hyperparameter sweep + feature ablation"
    )
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--ablation", action="store_true", help="Run feature ablation")
    parser.add_argument("--all", action="store_true", help="Run sweep then ablation with best config")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="outputs/bilstm_sweep")
    args = parser.parse_args()

    if not (args.sweep or args.ablation or args.all):
        parser.error("Specify --sweep, --ablation, or --all")

    # Load data once
    all_sequences = load_all_sequences()
    if not all_sequences:
        console.print("[red]No sequences loaded.[/red]")
        return

    train_seqs, test_seqs = split_sequences(all_sequences)
    train_vids = sorted(set(s.video_id for s in train_seqs))
    test_vids = sorted(set(s.video_id for s in test_seqs))
    console.print(
        f"\nSplit: {len(train_vids)} train videos ({len(train_seqs)} rallies), "
        f"{len(test_vids)} held-out videos ({len(test_seqs)} rallies)"
    )

    if not test_seqs or not train_seqs:
        console.print("[red]Empty train or test split.[/red]")
        return

    # Sweep
    best_config = None
    if args.sweep or args.all:
        best_config = run_sweep(
            train_seqs, test_seqs,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

    # Ablation
    if args.ablation or args.all:
        if best_config and args.all:
            h, l, d, lr_ = (
                best_config.hidden_dim,
                best_config.num_layers,
                best_config.dropout,
                best_config.lr,
            )
            console.print(
                f"\n[bold]Using sweep winner for ablation: "
                f"hidden={h} layers={l} dropout={d} lr={lr_:.0e}[/bold]"
            )
        else:
            h, l, d, lr_ = args.hidden_dim, args.num_layers, args.dropout, args.lr

        run_ablation(
            train_seqs, test_seqs,
            hidden_dim=h,
            num_layers=l,
            dropout=d,
            lr=lr_,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
