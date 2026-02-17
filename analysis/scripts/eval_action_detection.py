"""Evaluate action detection against ground truth labels.

Loads action ground truth from the database (saved via the web editor),
runs contact detection + action classification on each rally, and computes:
- Contact recall: % of GT contacts detected (±tolerance frame window)
- Contact precision: % of detected contacts matching a GT contact
- Action accuracy: % of matched contacts correctly classified
- Per-class F1: serve, receive, set, spike, block, dig
- Confusion matrix
- Per-rally and aggregate tables

Usage:
    cd analysis
    uv run python scripts/eval_action_detection.py
    uv run python scripts/eval_action_detection.py --tolerance 5   # ±5 frame window
    uv run python scripts/eval_action_detection.py --rally <id>    # Specific rally
    uv run python scripts/eval_action_detection.py --redetect --config '{"min_peak_velocity": 0.008}'
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts

console = Console()

ACTION_TYPES = ["serve", "receive", "set", "spike", "block", "dig"]


@dataclass
class GtLabel:
    frame: int
    action: str
    player_track_id: int
    ball_x: float | None = None
    ball_y: float | None = None


@dataclass
class RallyData:
    rally_id: str
    video_id: str
    gt_labels: list[GtLabel]
    ball_positions_json: list[dict] | None
    positions_json: list[dict] | None
    contacts_json: dict | None
    actions_json: dict | None
    frame_count: int
    fps: float
    court_split_y: float | None


def load_rallies_with_action_gt(
    rally_id: str | None = None,
) -> list[RallyData]:
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
            pt.contacts_json,
            pt.actions_json,
            pt.frame_count,
            pt.fps,
            pt.court_split_y
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE {where_sql}
        ORDER BY r.video_id, r.start_ms
    """

    results: list[RallyData] = []

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
                    actions_json,
                    frame_count,
                    fps,
                    court_split_y,
                ) = row

                gt_labels = []
                if action_gt_json:
                    for label in action_gt_json:
                        gt_labels.append(GtLabel(
                            frame=label["frame"],
                            action=label["action"],
                            player_track_id=label.get("playerTrackId", -1),
                            ball_x=label.get("ballX"),
                            ball_y=label.get("ballY"),
                        ))

                results.append(RallyData(
                    rally_id=rally_id_val,
                    video_id=video_id_val,
                    gt_labels=gt_labels,
                    ball_positions_json=ball_positions_json,
                    positions_json=positions_json,
                    contacts_json=contacts_json,
                    actions_json=actions_json,
                    frame_count=frame_count or 0,
                    fps=fps or 30.0,
                    court_split_y=court_split_y,
                ))

    return results


@dataclass
class MatchResult:
    """Result of matching GT to predicted contacts."""
    gt_frame: int
    gt_action: str
    pred_frame: int | None  # None if unmatched
    pred_action: str | None  # None if unmatched
    player_correct: bool = False


def match_contacts(
    gt_labels: list[GtLabel],
    pred_actions: list[dict],
    tolerance: int = 3,
) -> tuple[list[MatchResult], list[dict]]:
    """Match GT labels to predicted actions using frame tolerance.

    Returns:
        Tuple of (matched GT results, unmatched predictions).
    """
    results: list[MatchResult] = []
    used_preds: set[int] = set()

    # Sort GT and predictions by frame
    gt_sorted = sorted(gt_labels, key=lambda gt: gt.frame)
    pred_sorted = sorted(pred_actions, key=lambda a: a.get("frame", 0))

    for gt in gt_sorted:
        best_idx: int | None = None
        best_dist = tolerance + 1

        for i, pred in enumerate(pred_sorted):
            if i in used_preds:
                continue
            dist = abs(gt.frame - pred.get("frame", 0))
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            used_preds.add(best_idx)
            pred = pred_sorted[best_idx]
            results.append(MatchResult(
                gt_frame=gt.frame,
                gt_action=gt.action,
                pred_frame=pred.get("frame"),
                pred_action=pred.get("action"),
                player_correct=(gt.player_track_id == pred.get("playerTrackId", -1)),
            ))
        else:
            results.append(MatchResult(
                gt_frame=gt.frame,
                gt_action=gt.action,
                pred_frame=None,
                pred_action=None,
            ))

    # Collect unmatched predictions
    unmatched = [pred_sorted[i] for i in range(len(pred_sorted)) if i not in used_preds]

    return results, unmatched


def compute_metrics(
    matches: list[MatchResult],
    unmatched_preds: list[dict],
) -> dict:
    """Compute contact detection and action classification metrics."""
    total_gt = len(matches)
    matched = [m for m in matches if m.pred_frame is not None]
    unmatched_gt = [m for m in matches if m.pred_frame is None]

    tp = len(matched)
    fn = len(unmatched_gt)
    fp = len(unmatched_preds)

    recall = tp / max(1, total_gt)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # Action accuracy (among matched contacts)
    action_correct = sum(1 for m in matched if m.gt_action == m.pred_action)
    action_accuracy = action_correct / max(1, tp)

    # Player attribution accuracy
    player_correct = sum(1 for m in matched if m.player_correct)
    player_accuracy = player_correct / max(1, tp)

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    for action in ACTION_TYPES:
        class_gt = [m for m in matches if m.gt_action == action]
        class_tp = [m for m in class_gt if m.pred_action == action]
        class_fn = [m for m in class_gt if m.pred_action != action]
        class_fp_matched = [m for m in matched if m.pred_action == action and m.gt_action != action]
        class_fp_unmatched = [p for p in unmatched_preds if p.get("action") == action]

        c_tp = len(class_tp)
        c_fn = len(class_fn)
        c_fp = len(class_fp_matched) + len(class_fp_unmatched)

        c_precision = c_tp / max(1, c_tp + c_fp)
        c_recall = c_tp / max(1, c_tp + c_fn)
        c_f1 = 2 * c_precision * c_recall / max(1e-9, c_precision + c_recall)

        per_class[action] = {
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": c_precision,
            "recall": c_recall,
            "f1": c_f1,
        }

    return {
        "total_gt": total_gt,
        "total_pred": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "action_accuracy": action_accuracy,
        "player_accuracy": player_accuracy,
        "per_class": per_class,
    }


def build_confusion_matrix(
    matches: list[MatchResult],
) -> dict[str, dict[str, int]]:
    """Build confusion matrix from matched contacts."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for m in matches:
        if m.pred_action is not None:
            matrix[m.gt_action][m.pred_action] += 1
        else:
            matrix[m.gt_action]["MISS"] += 1

    return {k: dict(v) for k, v in matrix.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate action detection vs ground truth")
    parser.add_argument("--rally", type=str, help="Specific rally ID to evaluate")
    parser.add_argument("--tolerance", type=int, default=3, help="Frame tolerance for matching (default: 3)")
    parser.add_argument("--redetect", action="store_true", help="Re-run contact detection instead of using stored results")
    parser.add_argument("--config", type=str, help="JSON config overrides for ContactDetectionConfig (implies --redetect)")
    args = parser.parse_args()

    # Build ContactDetectionConfig from overrides
    contact_config: ContactDetectionConfig | None = None
    if args.config:
        try:
            overrides = json.loads(args.config)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in --config: {e}[/red]")
            return
        try:
            contact_config = ContactDetectionConfig(**overrides)
        except TypeError as e:
            console.print(f"[red]Invalid config field: {e}[/red]")
            return

    rallies = load_rallies_with_action_gt(rally_id=args.rally)

    if not rallies:
        console.print("[red]No rallies found with action ground truth labels.[/red]")
        console.print("Label actions in the web editor first (Label Actions button).")
        return

    console.print(f"\n[bold]Evaluating {len(rallies)} rallies with action ground truth[/bold]\n")

    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []

    # Per-rally results table
    rally_table = Table(title="Per-Rally Contact Detection")
    rally_table.add_column("Rally ID", style="dim", max_width=12)
    rally_table.add_column("GT", justify="right")
    rally_table.add_column("Pred", justify="right")
    rally_table.add_column("TP", justify="right")
    rally_table.add_column("Recall", justify="right")
    rally_table.add_column("Precision", justify="right")
    rally_table.add_column("F1", justify="right")
    rally_table.add_column("Action Acc", justify="right")

    for rally in rallies:
        # Get predicted actions — either from stored data or re-detect
        pred_actions: list[dict] = []

        if (args.redetect or contact_config) and rally.ball_positions_json:
            # Re-run contact detection from ball/player positions
            from rallycut.tracking.ball_tracker import BallPosition as BallPos
            from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

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

            player_positions = []
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

            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                config=contact_config,
                net_y=rally.court_split_y,
            )

            rally_actions = classify_rally_actions(contacts, rally.rally_id)
            pred_actions = [a.to_dict() for a in rally_actions.actions]
        elif rally.actions_json:
            # Use stored actions
            pred_actions = rally.actions_json.get("actions", [])

        matches, unmatched = match_contacts(
            rally.gt_labels,
            pred_actions,
            tolerance=args.tolerance,
        )

        metrics = compute_metrics(matches, unmatched)

        rally_table.add_row(
            rally.rally_id[:8],
            str(metrics["total_gt"]),
            str(metrics["total_pred"]),
            str(metrics["tp"]),
            f"{metrics['recall']:.1%}",
            f"{metrics['precision']:.1%}",
            f"{metrics['f1']:.1%}",
            f"{metrics['action_accuracy']:.1%}",
        )

        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

    console.print(rally_table)

    # Aggregate metrics
    if all_matches:
        agg_metrics = compute_metrics(all_matches, all_unmatched)

        console.print(f"\n[bold]Aggregate Results ({len(rallies)} rallies)[/bold]")
        console.print(f"  GT contacts:        {agg_metrics['total_gt']}")
        console.print(f"  Predicted contacts: {agg_metrics['total_pred']}")
        console.print(f"  True positives:     {agg_metrics['tp']}")
        console.print(f"  False positives:    {agg_metrics['fp']}")
        console.print(f"  False negatives:    {agg_metrics['fn']}")
        console.print(f"  [bold]Contact Recall:    {agg_metrics['recall']:.1%}[/bold]")
        console.print(f"  [bold]Contact Precision: {agg_metrics['precision']:.1%}[/bold]")
        console.print(f"  [bold]Contact F1:        {agg_metrics['f1']:.1%}[/bold]")
        console.print(f"  [bold]Action Accuracy:   {agg_metrics['action_accuracy']:.1%}[/bold]")
        console.print(f"  Player Attribution: {agg_metrics['player_accuracy']:.1%}")

        # Per-class table
        class_table = Table(title="\nPer-Action Metrics")
        class_table.add_column("Action", style="bold")
        class_table.add_column("TP", justify="right")
        class_table.add_column("FP", justify="right")
        class_table.add_column("FN", justify="right")
        class_table.add_column("Precision", justify="right")
        class_table.add_column("Recall", justify="right")
        class_table.add_column("F1", justify="right")

        for action in ACTION_TYPES:
            c = agg_metrics["per_class"].get(action, {})
            if c.get("tp", 0) + c.get("fp", 0) + c.get("fn", 0) > 0:
                class_table.add_row(
                    action.capitalize(),
                    str(int(c.get("tp", 0))),
                    str(int(c.get("fp", 0))),
                    str(int(c.get("fn", 0))),
                    f"{c.get('precision', 0):.1%}",
                    f"{c.get('recall', 0):.1%}",
                    f"{c.get('f1', 0):.1%}",
                )

        console.print(class_table)

        # Confusion matrix
        conf_matrix = build_confusion_matrix(all_matches)
        if conf_matrix:
            all_labels = sorted(set(
                list(conf_matrix.keys()) +
                [pred for row in conf_matrix.values() for pred in row.keys()]
            ))

            cm_table = Table(title="\nConfusion Matrix (rows=GT, cols=Predicted)")
            cm_table.add_column("GT \\ Pred", style="bold")
            for label in all_labels:
                cm_table.add_column(label[:5], justify="right")

            for gt_label in all_labels:
                if gt_label in conf_matrix:
                    row_data = conf_matrix[gt_label]
                    cells = [str(row_data.get(pred, 0)) for pred in all_labels]
                    cm_table.add_row(gt_label[:8], *cells)

            console.print(cm_table)


if __name__ == "__main__":
    main()
