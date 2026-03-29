"""Train temporal contact attribution model.

Loads GT contacts from the database, extracts trajectory features,
runs leave-one-video-out CV with gradient-boosted trees, and trains
a final model on all data.

Usage:
    cd analysis
    uv run python scripts/train_temporal_attribution.py
    uv run python scripts/train_temporal_attribution.py --skip-cv
    uv run python scripts/train_temporal_attribution.py --use-predicted  # train on predicted contacts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.split import add_split_argument, apply_split
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.temporal_attribution.features import (
    extract_attribution_features,
)
from rallycut.tracking.temporal_attribution.training import TrainingConfig, train_model

console = Console()

WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "temporal_attribution"
CHECKPOINT_PATH = WEIGHTS_DIR / "best_temporal_attribution.joblib"


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
            pt.positions_json,
            pt.contacts_json
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
                    contacts_json,
                ) = row

                results.append({
                    "rally_id": rally_id_val,
                    "video_id": video_id_val,
                    "action_gt_json": action_gt_json or [],
                    "ball_positions_json": ball_positions_json or [],
                    "positions_json": positions_json or [],
                    "contacts_json": contacts_json,
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


def _compute_contact_context(
    contacts_json: dict | None,
    gt_frame: int,
    tolerance: int = 5,
) -> tuple[int, int]:
    """Compute contact_index and side_count from stored contacts.

    Matches the GT frame to the nearest stored contact to get sequence context.
    Returns (contact_index, side_count) or (0, 1) as default.
    """
    if not contacts_json:
        return 0, 1

    contacts = contacts_json.get("contacts", [])
    if not contacts:
        return 0, 1

    # Find nearest stored contact
    best_idx = 0
    best_dist = abs(gt_frame - contacts[0].get("frame", 0))
    for i, c in enumerate(contacts[1:], 1):
        dist = abs(gt_frame - c.get("frame", 0))
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_dist > tolerance:
        return 0, 1

    # Count contacts on same court side up to this point
    contact_index = best_idx
    court_side = contacts[best_idx].get("courtSide", "unknown")
    side_count = 1
    for i in range(best_idx - 1, -1, -1):
        if contacts[i].get("courtSide") == court_side:
            side_count += 1
        else:
            break

    return contact_index, min(side_count, 3)


def extract_training_data(
    rallies: list[dict],
    use_predicted: bool = False,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Extract features and labels from GT contacts.

    Args:
        rallies: Loaded rally data from DB.
        use_predicted: If True, use predicted contact frames (from contacts_json)
            matched to GT labels. Matches inference distribution better.

    Returns:
        (features_list, labels, video_ids) — parallel lists.
    """
    features_list: list[np.ndarray] = []
    labels: list[int] = []
    video_ids: list[str] = []

    skipped_no_player = 0
    skipped_sparse = 0
    skipped_not_in_slots = 0

    for ri, rally in enumerate(rallies):
        ball_positions = parse_ball_positions(rally["ball_positions_json"])
        player_positions = parse_player_positions(rally["positions_json"])

        if not ball_positions or not player_positions:
            continue

        gt_labels = rally["action_gt_json"]

        # If using predicted contacts, build frame mapping
        pred_frame_map: dict[int, tuple[int, int]] = {}
        if use_predicted and rally["contacts_json"]:
            contacts = rally["contacts_json"].get("contacts", [])
            for ci, c in enumerate(contacts):
                cf = c.get("frame", -1)
                cs = c.get("courtSide", "unknown")
                # Count side contacts
                sc = 1
                for j in range(ci - 1, -1, -1):
                    if contacts[j].get("courtSide") == cs:
                        sc += 1
                    else:
                        break
                pred_frame_map[cf] = (ci, min(sc, 3))

        for gt in gt_labels:
            gt_frame = gt["frame"]
            gt_track_id = gt.get("playerTrackId", -1)

            if gt_track_id < 0:
                skipped_no_player += 1
                continue

            # Determine contact frame and context
            if use_predicted and pred_frame_map:
                # Find nearest predicted contact to GT frame
                best_pf = min(pred_frame_map.keys(), key=lambda f: abs(f - gt_frame))
                if abs(best_pf - gt_frame) <= 5:
                    contact_frame = best_pf
                    contact_index, side_count = pred_frame_map[best_pf]
                else:
                    contact_frame = gt_frame
                    contact_index, side_count = _compute_contact_context(
                        rally["contacts_json"], gt_frame
                    )
            else:
                contact_frame = gt_frame
                contact_index, side_count = _compute_contact_context(
                    rally["contacts_json"], gt_frame
                )

            result = extract_attribution_features(
                contact_frame=contact_frame,
                ball_positions=ball_positions,
                player_positions=player_positions,
                contact_index=contact_index,
                side_count=side_count,
            )

            if result is None:
                skipped_sparse += 1
                continue

            feats, canonical_tids = result

            if gt_track_id in canonical_tids:
                slot_label = canonical_tids.index(gt_track_id)
            else:
                skipped_not_in_slots += 1
                continue

            features_list.append(feats)
            labels.append(slot_label)
            video_ids.append(rally["video_id"])

        if (ri + 1) % 20 == 0 or ri == len(rallies) - 1:
            console.print(
                f"  [{ri + 1}/{len(rallies)}] {len(features_list)} samples extracted"
            )

    console.print("\nExtraction summary:")
    console.print(f"  Total samples: {len(features_list)}")
    console.print(f"  Skipped (no GT player): {skipped_no_player}")
    console.print(f"  Skipped (sparse ball): {skipped_sparse}")
    console.print(f"  Skipped (GT not in 4 nearest): {skipped_not_in_slots}")

    if labels:
        dist = np.bincount(labels, minlength=4)
        console.print(f"  Slot distribution: {dict(enumerate(dist.tolist()))}")
        baseline = dist[0] / len(labels)
        console.print(f"  Baseline (always slot 0): {baseline:.1%}")

    return features_list, labels, video_ids


def run_loocv(
    features_list: list[np.ndarray],
    labels: list[int],
    video_ids: list[str],
    config: TrainingConfig,
) -> float:
    """Run leave-one-video-out cross-validation."""
    unique_videos = sorted(set(video_ids))
    console.print(f"\nLeave-one-video-out CV: {len(unique_videos)} folds")

    all_features = np.array(features_list)
    all_labels = np.array(labels)
    video_arr = np.array(video_ids)

    fold_results: list[tuple[str, int, int, float]] = []
    total_correct = 0
    total_count = 0

    for fold_idx, held_out_video in enumerate(unique_videos):
        val_mask = video_arr == held_out_video
        train_mask = ~val_mask

        train_f = all_features[train_mask]
        train_l = all_labels[train_mask]
        val_f = all_features[val_mask]
        val_l = all_labels[val_mask]

        if len(val_l) == 0 or len(train_l) == 0:
            continue

        result = train_model(
            train_features=train_f,
            train_labels=train_l,
            val_features=val_f,
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
            f"(train={len(train_l)})"
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
    table.add_row(
        "TOTAL", str(total_count), str(total_correct), f"{agg_acc:.1%}",
        style="bold",
    )
    console.print(table)

    return agg_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train temporal contact attribution model"
    )
    parser.add_argument("--max-iter", type=int, default=200, help="Max boosting iterations")
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--skip-cv", action="store_true", help="Skip LOO-CV, train only")
    parser.add_argument(
        "--use-predicted", action="store_true",
        help="Train on predicted contact frames (matches inference distribution)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--exclude-videos",
        type=str,
        help="Comma-separated video ID prefixes to exclude from training (held-out test set)",
    )
    add_split_argument(parser)
    args = parser.parse_args()

    train_config = TrainingConfig(
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        seed=args.seed,
    )

    # --- Load data ---
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    rallies = apply_split(rallies, args)
    if args.exclude_videos:
        prefixes = [p.strip() for p in args.exclude_videos.split(",")]
        before = len(rallies)
        rallies = [r for r in rallies if not any(r["video_id"].startswith(p) for p in prefixes)]
        console.print(f"  Excluded {before - len(rallies)} rallies from {len(prefixes)} video(s)")
    console.print(f"Loaded {len(rallies)} rallies")

    # --- Extract training data ---
    console.print("\n[bold]Extracting features...[/bold]")
    features_list, labels, video_ids = extract_training_data(
        rallies, use_predicted=args.use_predicted
    )

    if len(features_list) < 10:
        console.print("[red]Too few training samples. Exiting.[/red]")
        sys.exit(1)

    # --- LOO-CV ---
    if not args.skip_cv:
        loocv_acc = run_loocv(features_list, labels, video_ids, train_config)
        console.print(f"\n[bold]LOO-CV accuracy: {loocv_acc:.1%}[/bold]")

    # --- Train final model on all data ---
    console.print("\n[bold]Training final model on all data...[/bold]")
    all_features = np.array(features_list)
    all_labels = np.array(labels)

    # Use 90/10 split for validation reporting
    np.random.seed(args.seed)
    n = len(all_features)
    indices = np.random.permutation(n)
    split = int(0.9 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    result = train_model(
        train_features=all_features[train_idx],
        train_labels=all_labels[train_idx],
        val_features=all_features[val_idx],
        val_labels=all_labels[val_idx],
        config=train_config,
        output_path=CHECKPOINT_PATH,
    )

    console.print(
        f"\nFinal model: val_acc={result.best_val_accuracy:.1%}, "
        f"train={result.num_train}, val={result.num_val}"
    )

    # Feature importance
    if result.feature_importances:
        sorted_imp = sorted(
            result.feature_importances.items(), key=lambda kv: kv[1], reverse=True
        )
        table = Table(title="Top 15 Feature Importances")
        table.add_column("Feature", style="cyan")
        table.add_column("Importance", justify="right")
        for name, imp in sorted_imp[:15]:
            table.add_row(name, f"{imp:.4f}")
        console.print(table)

    console.print(f"Saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
