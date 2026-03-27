"""Train temporal contact attribution model.

Loads GT contacts from the database, extracts trajectory windows,
runs leave-one-video-out CV, and trains a final model on all data.

Usage:
    cd analysis
    uv run python scripts/train_temporal_attribution.py
    uv run python scripts/train_temporal_attribution.py --epochs 150
    uv run python scripts/train_temporal_attribution.py --skip-cv  # Skip LOO-CV, train only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.temporal_attribution.features import extract_attribution_window
from rallycut.tracking.temporal_attribution.model import TemporalAttributionConfig
from rallycut.tracking.temporal_attribution.training import TrainingConfig, train_model

console = Console()

WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "temporal_attribution"
CHECKPOINT_PATH = WEIGHTS_DIR / "best_temporal_attribution.pt"


def load_rallies_with_action_gt(
    rally_id: str | None = None,
) -> list[dict]:
    """Load rallies that have action ground truth labels."""
    where_clauses = ["pt.action_ground_truth_json IS NOT NULL"]
    params: list[str] = []

    if rally_id:
        where_clauses.append("r.id = %s")
        params.append(rally_id)

    where_sql = " AND ".join(where_clauses)

    query = f"""
        SELECT
            r.id as rally_id,
            r.video_id,
            pt.action_ground_truth_json,
            pt.ball_positions_json,
            pt.positions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE {where_sql}
        ORDER BY r.video_id, r.start_ms
    """

    results: list[dict] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            for row in rows:
                (
                    rally_id_val,
                    video_id_val,
                    action_gt_json,
                    ball_positions_json,
                    positions_json,
                ) = row

                results.append({
                    "rally_id": rally_id_val,
                    "video_id": video_id_val,
                    "action_gt_json": action_gt_json or [],
                    "ball_positions_json": ball_positions_json or [],
                    "positions_json": positions_json or [],
                })

    return results


def parse_ball_positions(raw: list[dict]) -> list[BallPosition]:
    """Parse ball_positions_json into BallPosition objects."""
    return [
        BallPosition(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in raw
    ]


def parse_player_positions(raw: list[dict]) -> list[PlayerPosition]:
    """Parse positions_json into PlayerPosition objects."""
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp.get("width", 0.05),
            height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
        )
        for pp in raw
    ]


def extract_training_data(
    rallies: list[dict],
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Extract trajectory windows and labels from GT contacts.

    Returns:
        (windows, labels, video_ids) — parallel lists.
        windows[i] is (21, 14), labels[i] is 0-3 (canonical slot of GT player).
    """
    windows: list[np.ndarray] = []
    labels: list[int] = []
    video_ids: list[str] = []

    skipped_no_ball = 0
    skipped_no_player = 0
    skipped_gt_not_in_slots = 0

    for ri, rally in enumerate(rallies):
        ball_positions = parse_ball_positions(rally["ball_positions_json"])
        player_positions = parse_player_positions(rally["positions_json"])

        if not ball_positions or not player_positions:
            continue

        gt_labels = rally["action_gt_json"]
        for gt in gt_labels:
            gt_frame = gt["frame"]
            gt_track_id = gt.get("playerTrackId", -1)

            if gt_track_id < 0:
                skipped_no_player += 1
                continue

            result = extract_attribution_window(
                contact_frame=gt_frame,
                ball_positions=ball_positions,
                player_positions=player_positions,
            )

            if result is None:
                skipped_no_ball += 1
                continue

            window, canonical_tids = result

            # Find which slot the GT player occupies
            if gt_track_id in canonical_tids:
                slot_label = canonical_tids.index(gt_track_id)
            else:
                skipped_gt_not_in_slots += 1
                continue

            windows.append(window)
            labels.append(slot_label)
            video_ids.append(rally["video_id"])

        if (ri + 1) % 20 == 0 or ri == len(rallies) - 1:
            console.print(
                f"  [{ri + 1}/{len(rallies)}] {len(windows)} windows extracted"
            )

    console.print("\nExtraction summary:")
    console.print(f"  Total windows: {len(windows)}")
    console.print(f"  Skipped (no GT player): {skipped_no_player}")
    console.print(f"  Skipped (sparse ball): {skipped_no_ball}")
    console.print(f"  Skipped (GT not in 4 nearest): {skipped_gt_not_in_slots}")

    # Label distribution
    if labels:
        dist = np.bincount(labels, minlength=4)
        console.print(f"  Slot distribution: {dict(enumerate(dist.tolist()))}")
        baseline = dist[0] / len(labels)
        console.print(f"  Baseline (always slot 0): {baseline:.1%}")

    return windows, labels, video_ids


def run_loocv(
    windows: list[np.ndarray],
    labels: list[int],
    video_ids: list[str],
    config: TrainingConfig,
) -> float:
    """Run leave-one-video-out cross-validation.

    Returns aggregate accuracy across all folds.
    """
    unique_videos = sorted(set(video_ids))
    console.print(f"\nLeave-one-video-out CV: {len(unique_videos)} folds")

    all_windows = np.array(windows)
    all_labels = np.array(labels)
    video_arr = np.array(video_ids)

    fold_results: list[tuple[str, int, int, float]] = []
    total_correct = 0
    total_count = 0

    for fold_idx, held_out_video in enumerate(unique_videos):
        val_mask = video_arr == held_out_video
        train_mask = ~val_mask

        train_w = all_windows[train_mask]
        train_l = all_labels[train_mask]
        val_w = all_windows[val_mask]
        val_l = all_labels[val_mask]

        if len(val_l) == 0 or len(train_l) == 0:
            continue

        result = train_model(
            train_windows=train_w,
            train_labels=train_l,
            val_windows=val_w,
            val_labels=val_l,
            config=config,
        )

        n_correct = int(round(result.best_val_accuracy * len(val_l)))
        total_correct += n_correct
        total_count += len(val_l)

        fold_results.append((
            held_out_video[:8],
            len(val_l),
            n_correct,
            result.best_val_accuracy,
        ))

        console.print(
            f"  [{fold_idx + 1}/{len(unique_videos)}] "
            f"{held_out_video[:8]}: "
            f"{n_correct}/{len(val_l)} = {result.best_val_accuracy:.1%} "
            f"(train={len(train_l)}, epoch={result.best_epoch})"
        )

    # Summary table
    table = Table(title="LOO-CV Results")
    table.add_column("Video", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")

    for vid, n, correct, acc in fold_results:
        style = "green" if acc > 0.73 else "red" if acc < 0.60 else ""
        table.add_row(vid, str(n), str(correct), f"{acc:.1%}", style=style)

    agg_acc = total_correct / max(1, total_count)
    table.add_row("TOTAL", str(total_count), str(total_correct), f"{agg_acc:.1%}", style="bold")
    console.print(table)

    return agg_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal contact attribution model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--skip-cv", action="store_true", help="Skip LOO-CV, train only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    model_config = TemporalAttributionConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    train_config = TrainingConfig(
        model_config=model_config,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )

    # --- Load data ---
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"Loaded {len(rallies)} rallies")

    # --- Extract training data ---
    console.print("\n[bold]Extracting trajectory windows...[/bold]")
    windows, labels, video_ids = extract_training_data(rallies)

    if len(windows) < 10:
        console.print("[red]Too few training samples. Exiting.[/red]")
        sys.exit(1)

    # --- LOO-CV ---
    if not args.skip_cv:
        loocv_acc = run_loocv(windows, labels, video_ids, train_config)
        console.print(f"\n[bold]LOO-CV accuracy: {loocv_acc:.1%}[/bold]")
    else:
        console.print("\n[dim]Skipping LOO-CV[/dim]")

    # --- Train final model on all data ---
    console.print("\n[bold]Training final model on all data...[/bold]")
    all_windows = np.array(windows)
    all_labels = np.array(labels)

    # Use 90/10 split for early stopping
    np.random.seed(args.seed)
    n = len(all_windows)
    indices = np.random.permutation(n)
    split = int(0.9 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    result = train_model(
        train_windows=all_windows[train_idx],
        train_labels=all_labels[train_idx],
        val_windows=all_windows[val_idx],
        val_labels=all_labels[val_idx],
        config=train_config,
        output_path=CHECKPOINT_PATH,
    )

    console.print(
        f"\nFinal model: val_acc={result.best_val_accuracy:.1%}, "
        f"epoch={result.best_epoch}, "
        f"train={result.num_train}, val={result.num_val}"
    )
    console.print(f"Saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
