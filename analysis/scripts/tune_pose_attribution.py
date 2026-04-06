"""Hyperparameter tuning for per-candidate pose attribution.

Sweeps positive_weight, max_depth, and feature sets to find the best
configuration. Then trains the final model on all data and saves it.

Usage:
    cd analysis
    uv run python scripts/tune_pose_attribution.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.tracking.pose_attribution.features import NUM_FEATURES, POSE_FEATURE_COUNT
from rallycut.tracking.pose_attribution.pose_cache import load_pose_cache
from rallycut.tracking.pose_attribution.training import TrainingConfig, train_model
from scripts.eval_action_detection import load_rallies_with_action_gt
from scripts.eval_pose_attribution import (
    compute_proximity_baseline,
    extract_samples,
    run_loocv,
)

console = Console()


def main() -> None:
    rallies = load_rallies_with_action_gt()
    console.print(f"Loaded {len(rallies)} rallies with action GT")

    n_cached = sum(1 for r in rallies if load_pose_cache(r.rally_id) is not None)
    console.print(f"  Pose cache: {n_cached}/{len(rallies)} rallies")

    console.print("\nExtracting features...")
    t0 = time.time()
    samples = extract_samples(rallies, use_pose=True)
    console.print(f"  {len(samples)} evaluable contacts ({time.time() - t0:.1f}s)")

    if not samples:
        console.print("[red]No samples[/red]")
        sys.exit(1)

    proximity = compute_proximity_baseline(samples)
    console.print(f"\nProximity baseline: {proximity.accuracy:.1%} ({proximity.n_correct}/{proximity.n_total})")

    # === Sweep configurations ===
    configs: list[tuple[str, TrainingConfig, np.ndarray | None]] = []

    # Feature masks
    pose_mask = np.zeros(NUM_FEATURES, dtype=bool)
    pose_mask[:POSE_FEATURE_COUNT] = True
    spatial_mask = np.zeros(NUM_FEATURES, dtype=bool)
    spatial_mask[POSE_FEATURE_COUNT:] = True

    # Sweep positive_weight for combined model
    for pw in [1.0, 2.0, 3.0, 4.0, 5.0]:
        cfg = TrainingConfig(positive_weight=pw)
        configs.append((f"combined pw={pw:.0f}", cfg, None))

    # Sweep positive_weight for pose-only
    for pw in [1.0, 2.0, 3.0, 4.0, 5.0]:
        cfg = TrainingConfig(positive_weight=pw)
        configs.append((f"pose-only pw={pw:.0f}", cfg, pose_mask))

    # Sweep max_depth for best pw candidates
    for md in [3, 4, 5, 6]:
        cfg = TrainingConfig(positive_weight=3.0, max_depth=md)
        configs.append((f"combined md={md}", cfg, None))

    # Sweep max_leaf_nodes
    for mln in [10, 15, 20, 31]:
        cfg = TrainingConfig(positive_weight=3.0, max_leaf_nodes=mln)
        configs.append((f"combined mln={mln}", cfg, None))

    # Run all
    results: list[tuple[str, float, float, int]] = []

    console.print(f"\nRunning {len(configs)} configurations...\n")
    for name, cfg, mask in configs:
        t0 = time.time()
        r = run_loocv(samples, feature_mask=mask, config=cfg)
        elapsed = time.time() - t0
        delta = r.accuracy - proximity.accuracy
        results.append((name, r.accuracy, r.auc, r.n_correct))
        console.print(
            f"  {name:30s}  {r.accuracy:.1%} ({r.n_correct}/{r.n_total})  "
            f"AUC={r.auc:.3f}  Δ={delta:+.1%}  [{elapsed:.0f}s]"
        )

    # Summary table sorted by accuracy
    console.print()
    results.sort(key=lambda x: x[1], reverse=True)

    table = Table(title="Tuning Results (sorted by accuracy)")
    table.add_column("Configuration")
    table.add_column("Accuracy", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Delta", justify="right")

    for name, acc, auc, n_correct in results[:15]:
        delta = acc - proximity.accuracy
        table.add_row(name, f"{acc:.1%}", f"{auc:.3f}", f"{delta:+.1%}")

    console.print(table)

    # Train final model with best config
    best_name, best_acc, best_auc, _ = results[0]
    console.print(f"\n[bold]Best: {best_name} ({best_acc:.1%}, AUC={best_auc:.3f})[/bold]")

    # Find the config for the best result
    best_cfg = None
    best_mask = None
    for name, cfg, mask in configs:
        if name == best_name:
            best_cfg = cfg
            best_mask = mask
            break

    if best_cfg is None:
        return

    # Train on all data
    console.print("\nTraining final model on all data...")
    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    for s in samples:
        for feat, tid in zip(s.candidate_features, s.candidate_track_ids):
            f = feat if best_mask is None else feat[best_mask]
            all_x.append(f)
            all_y.append(1 if tid == s.gt_track_id else 0)

    from pathlib import Path

    output_path = Path("weights/pose_attribution/pose_attribution.joblib")
    clf, result = train_model(
        np.stack(all_x),
        np.array(all_y, dtype=np.int32),
        best_cfg,
        output_path=output_path,
    )

    console.print(f"  Train AUC: {result.train_auc:.3f}")
    console.print(f"  Positive: {result.num_positive}, Negative: {result.num_negative}")
    console.print(f"  Saved to: {output_path}")

    # Also save metadata
    import json
    meta = {
        "config": best_name,
        "loocv_accuracy": best_acc,
        "loocv_auc": best_auc,
        "train_auc": result.train_auc,
        "n_positive": result.num_positive,
        "n_negative": result.num_negative,
        "feature_set": "pose-only" if best_mask is not None and best_mask[:POSE_FEATURE_COUNT].all() and not best_mask[POSE_FEATURE_COUNT:].any() else "combined",
        "n_features": int(best_mask.sum()) if best_mask is not None else NUM_FEATURES,
    }
    meta_path = output_path.parent / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    console.print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
