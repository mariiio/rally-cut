"""Train an action type classifier on labeled rally data.

Loads rallies with action ground truth, re-runs contact detection to get
validated contacts, matches them against GT, extracts ActionFeatures,
and trains a multiclass GradientBoostingClassifier.

Reports both train-on-all and leave-one-rally-out CV metrics.

Usage:
    cd analysis
    uv run python scripts/train_action_classifier.py
    uv run python scripts/train_action_classifier.py --output weights/action_classifier/action_classifier.pkl
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.tracking.action_type_classifier import (
    ActionTypeClassifier,
    extract_action_features,
)
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def extract_features_for_rally(
    rally: RallyData,
    config: ContactDetectionConfig | None = None,
    tolerance: int = 5,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Extract action features for matched contacts in a rally.

    Returns:
        Tuple of (feature_arrays, action_labels, rally_ids).
    """
    if not rally.ball_positions_json:
        return [], [], []

    ball_positions = [
        BallPos(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in rally.ball_positions_json
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]

    if not ball_positions:
        return [], [], []

    player_positions: list[PlayerPos] = []
    if rally.positions_json:
        player_positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"],
                y=pp["y"],
                width=pp["width"],
                height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]

    # Re-run contact detection
    contact_seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=config,
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
    )

    if not contact_seq.contacts:
        return [], [], []

    # Match detected contacts to GT
    pred_actions = [
        {"frame": c.frame, "action": "unknown", "playerTrackId": c.player_track_id}
        for c in contact_seq.contacts
    ]
    matches, _ = match_contacts(rally.gt_labels, pred_actions, tolerance=tolerance)

    # Build contact index lookup: frame -> index in contacts list
    frame_to_idx: dict[int, int] = {}
    for idx, c in enumerate(contact_seq.contacts):
        frame_to_idx[c.frame] = idx

    features_list: list[np.ndarray] = []
    labels: list[str] = []
    rally_ids: list[str] = []

    for m in matches:
        if m.pred_frame is None:
            continue  # Unmatched GT (missed contact)
        if m.gt_action == "block":
            continue  # Skip block (too few samples, stays rule-based)
        if m.gt_action not in ("serve", "receive", "set", "attack", "dig"):
            continue

        contact_idx = frame_to_idx.get(m.pred_frame)
        if contact_idx is None:
            continue

        contact = contact_seq.contacts[contact_idx]
        feat = extract_action_features(
            contact=contact,
            index=contact_idx,
            all_contacts=contact_seq.contacts,
            ball_positions=contact_seq.ball_positions or None,
            net_y=contact_seq.net_y,
            rally_start_frame=contact_seq.rally_start_frame,
        )

        features_list.append(feat.to_array())
        labels.append(m.gt_action)
        rally_ids.append(rally.rally_id)

    return features_list, labels, rally_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action type classifier")
    parser.add_argument(
        "--output",
        type=str,
        default="weights/action_classifier/action_classifier.pkl",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="Frame tolerance for GT matching",
    )
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    console.print(
        f"\n[bold]Extracting action features from {len(rallies)} rallies[/bold]\n"
    )

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []
    all_rally_ids: list[str] = []

    per_rally_table = Table(title="Per-Rally Feature Extraction")
    per_rally_table.add_column("Rally", max_width=10)
    per_rally_table.add_column("Matched", justify="right")
    per_rally_table.add_column("GT Labels", justify="right")
    per_rally_table.add_column("Actions", justify="right")

    for rally in rallies:
        features, labels, rids = extract_features_for_rally(
            rally, tolerance=args.tolerance
        )

        all_features.extend(features)
        all_labels.extend(labels)
        all_rally_ids.extend(rids)

        # Count actions for display
        action_counts: dict[str, int] = defaultdict(int)
        for lbl in labels:
            action_counts[lbl] += 1
        action_str = ", ".join(
            f"{k}:{v}" for k, v in sorted(action_counts.items())
        )

        per_rally_table.add_row(
            rally.rally_id[:8],
            str(len(features)),
            str(len(rally.gt_labels)),
            action_str or "-",
        )

    console.print(per_rally_table)

    if not all_features:
        console.print("[red]No matched contacts extracted.[/red]")
        return

    x_mat = np.array(all_features)
    y = np.array(all_labels)
    rally_ids = np.array(all_rally_ids)

    # Class distribution
    console.print(f"\n[bold]Dataset: {len(y)} samples[/bold]")
    for cls in sorted(set(y)):
        console.print(f"  {cls}: {int(np.sum(y == cls))}")

    # Train on all data
    classifier = ActionTypeClassifier()
    train_metrics = classifier.train(x_mat, y)

    console.print(f"\n[bold]Train-on-all accuracy: {train_metrics['train_accuracy']:.1%}[/bold]")
    for cls, info in sorted(train_metrics["per_class"].items()):
        console.print(f"  {cls}: {info['accuracy']:.1%} ({info['count']} samples)")

    # LOO CV
    loo_metrics = classifier.loo_cv(x_mat, y, rally_ids)
    console.print(
        f"\n[bold]Leave-One-Rally-Out CV "
        f"({loo_metrics['n_rallies']} folds): "
        f"{loo_metrics['loo_accuracy']:.1%}[/bold]"
    )

    # Per-class LOO table
    loo_table = Table(title="LOO CV Per-Class Metrics")
    loo_table.add_column("Action", style="bold")
    loo_table.add_column("TP", justify="right")
    loo_table.add_column("FP", justify="right")
    loo_table.add_column("FN", justify="right")
    loo_table.add_column("P", justify="right")
    loo_table.add_column("R", justify="right")
    loo_table.add_column("F1", justify="right")
    loo_table.add_column("N", justify="right")

    for cls, info in sorted(loo_metrics["per_class"].items()):
        if info["count"] > 0:
            loo_table.add_row(
                cls,
                str(info["tp"]),
                str(info["fp"]),
                str(info["fn"]),
                f"{info['precision']:.1%}",
                f"{info['recall']:.1%}",
                f"{info['f1']:.1%}",
                str(info["count"]),
            )

    console.print(loo_table)

    # Confusion matrix
    confusion = loo_metrics.get("confusion", {})
    if confusion:
        classes = sorted(
            set(list(confusion.keys()) + [
                p for row in confusion.values() for p in row.keys()
            ])
        )
        cm_table = Table(title="\nLOO CV Confusion Matrix (rows=GT, cols=Predicted)")
        cm_table.add_column("GT \\ Pred", style="bold")
        for cls in classes:
            cm_table.add_column(cls[:5], justify="right")

        for gt_cls in classes:
            if gt_cls in confusion:
                cells = [str(confusion[gt_cls].get(p, 0)) for p in classes]
                cm_table.add_row(gt_cls, *cells)

        console.print(cm_table)

    # Feature importance
    importance = classifier.feature_importance()
    if importance:
        imp_table = Table(title="\nFeature Importance")
        imp_table.add_column("Feature")
        imp_table.add_column("Importance", justify="right")

        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            imp_table.add_row(name, f"{imp:.3f}")

        console.print(imp_table)

    # Save model
    classifier.save(args.output)
    console.print(f"\n[green]Saved action type classifier to {args.output}[/green]")


if __name__ == "__main__":
    main()
