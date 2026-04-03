"""Calibrate Y-displacement threshold for ball_crossed_net().

Sweeps thresholds on the action GT dataset to find the optimal
Y-displacement magnitude for detecting net crossings. Compares
against the current production ball_crossed_net() accuracy.

Usage:
    cd analysis
    uv run python scripts/calibrate_net_crossing.py
"""

from __future__ import annotations

import statistics

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import ball_crossed_net
from scripts.eval_action_detection import load_rallies_with_action_gt

console = Console()

# From diagnose_net_crossing.py
SIDE_CHANGES = {
    ("serve", "receive"),
    ("attack", "receive"),
    ("attack", "dig"),
    ("attack", "block"),
    ("block", "dig"),
    ("block", "receive"),
    ("block", "set"),
    ("block", "attack"),
}

SAME_SIDE = {
    ("receive", "set"),
    ("set", "attack"),
    ("dig", "set"),
    ("dig", "attack"),
    ("receive", "attack"),
}


def main() -> None:
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    console.print(f"\n[bold]Net Crossing Threshold Calibration ({len(rallies)} rallies)[/bold]\n")

    transitions: list[dict] = []

    for rally in rallies:
        if not rally.ball_positions_json:
            continue

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
            continue

        net_y = rally.court_split_y or 0.5
        gt_labels = sorted(rally.gt_labels, key=lambda g: g.frame)

        for i in range(len(gt_labels) - 1):
            prev_gt = gt_labels[i]
            curr_gt = gt_labels[i + 1]
            pair = (prev_gt.action, curr_gt.action)

            if pair in SIDE_CHANGES:
                gt_crossed = True
            elif pair in SAME_SIDE:
                gt_crossed = False
            else:
                continue

            in_range = [
                bp for bp in ball_positions
                if prev_gt.frame < bp.frame_number < curr_gt.frame
            ]

            # Production ball_crossed_net result
            prod_result = ball_crossed_net(
                ball_positions,
                from_frame=prev_gt.frame,
                to_frame=curr_gt.frame,
                net_y=net_y,
            )

            transitions.append({
                "gt_crossed": gt_crossed,
                "prod_result": prod_result,
                "n_positions": len(in_range),
                "pair": pair,
                "rally_id": rally.rally_id,
                "net_y": net_y,
                "positions": in_range,
            })

    console.print(f"Total transitions: {len(transitions)}")

    # Production accuracy (excluding None results)
    prod_evaluable = [t for t in transitions if t["prod_result"] is not None]
    prod_correct = sum(
        1 for t in prod_evaluable if t["prod_result"] == t["gt_crossed"]
    )
    prod_acc = prod_correct / len(prod_evaluable) if prod_evaluable else 0
    console.print(
        f"Production ball_crossed_net: {prod_acc:.1%} "
        f"({prod_correct}/{len(prod_evaluable)} evaluable, "
        f"{len(transitions) - len(prod_evaluable)} None)"
    )

    # Sweep for n_endpoint = 2 and 3
    results_table = Table(title="Threshold Sweep Results")
    results_table.add_column("n_endpoint")
    results_table.add_column("Evaluable")
    results_table.add_column("Best Threshold")
    results_table.add_column("Accuracy")
    results_table.add_column("Crossing Recall")
    results_table.add_column("Same-Side Recall")

    for n_endpoint in [2, 3]:
        min_required = n_endpoint * 2
        evaluated: list[dict] = []

        for t in transitions:
            positions = t["positions"]
            if len(positions) < min_required:
                continue

            start_ys = [bp.y for bp in positions[:n_endpoint]]
            end_ys = [bp.y for bp in positions[-n_endpoint:]]
            start_median = statistics.median(start_ys)
            end_median = statistics.median(end_ys)
            y_delta = abs(end_median - start_median)

            evaluated.append({
                "y_delta": y_delta,
                "gt_crossed": t["gt_crossed"],
                "pair": t["pair"],
            })

        if not evaluated:
            console.print(f"[red]No evaluable transitions for n_endpoint={n_endpoint}[/red]")
            continue

        abs_deltas = [e["y_delta"] for e in evaluated]
        thresholds = np.linspace(0, max(abs_deltas) * 1.2, 200)

        best_acc = 0.0
        best_thresh = 0.0
        best_cross_recall = 0.0
        best_same_recall = 0.0

        for thresh in thresholds:
            correct = 0
            cross_correct = 0
            cross_total = 0
            same_correct = 0
            same_total = 0
            for e in evaluated:
                pred_crossed = e["y_delta"] > thresh
                if pred_crossed == e["gt_crossed"]:
                    correct += 1
                if e["gt_crossed"]:
                    cross_total += 1
                    if pred_crossed:
                        cross_correct += 1
                else:
                    same_total += 1
                    if not pred_crossed:
                        same_correct += 1

            acc = correct / len(evaluated)
            if acc > best_acc:
                best_acc = acc
                best_thresh = float(thresh)
                best_cross_recall = cross_correct / cross_total if cross_total else 0
                best_same_recall = same_correct / same_total if same_total else 0

        results_table.add_row(
            str(n_endpoint),
            str(len(evaluated)),
            f"{best_thresh:.4f}",
            f"{best_acc:.1%}",
            f"{best_cross_recall:.1%}",
            f"{best_same_recall:.1%}",
        )

        # Sensitivity analysis: accuracy at ±10%, ±20% of optimal threshold
        console.print(f"\n[bold]Sensitivity (n_endpoint={n_endpoint}, optimal={best_thresh:.4f}):[/bold]")
        for pct in [-20, -10, 0, 10, 20]:
            t_adj = best_thresh * (1 + pct / 100)
            correct = sum(
                1 for e in evaluated
                if (e["y_delta"] > t_adj) == e["gt_crossed"]
            )
            acc = correct / len(evaluated)
            marker = " ← optimal" if pct == 0 else ""
            console.print(f"  {pct:+d}% ({t_adj:.4f}): {acc:.1%}{marker}")

        # Distribution of y_delta by class
        crossed_deltas = [e["y_delta"] for e in evaluated if e["gt_crossed"]]
        same_deltas = [e["y_delta"] for e in evaluated if not e["gt_crossed"]]
        console.print(f"\n  Crossed y_delta: median={np.median(crossed_deltas):.4f}, "
                       f"mean={np.mean(crossed_deltas):.4f}, "
                       f"min={min(crossed_deltas):.4f}, max={max(crossed_deltas):.4f}")
        console.print(f"  Same    y_delta: median={np.median(same_deltas):.4f}, "
                       f"mean={np.mean(same_deltas):.4f}, "
                       f"min={min(same_deltas):.4f}, max={max(same_deltas):.4f}")

        # Per action-pair breakdown at optimal threshold
        if n_endpoint in (2, 3):
            pair_table = Table(title=f"Per-Pair Breakdown (n={n_endpoint}, thresh={best_thresh:.4f})")
            pair_table.add_column("Pair")
            pair_table.add_column("GT")
            pair_table.add_column("Count")
            pair_table.add_column("Correct")
            pair_table.add_column("Accuracy")

            from collections import Counter
            pair_counts: Counter[tuple] = Counter()
            pair_correct: Counter[tuple] = Counter()
            for e in evaluated:
                pair_counts[e["pair"]] += 1
                pred = e["y_delta"] > best_thresh
                if pred == e["gt_crossed"]:
                    pair_correct[e["pair"]] += 1

            for pair, count in pair_counts.most_common():
                corr = pair_correct[pair]
                gt = "cross" if pair in SIDE_CHANGES else "same"
                pair_table.add_row(
                    f"{pair[0]}→{pair[1]}",
                    gt,
                    str(count),
                    str(corr),
                    f"{corr/count:.0%}",
                )
            console.print(pair_table)

    console.print()
    console.print(results_table)
    console.print(
        "\n[bold]Recommendation:[/bold] Use the threshold + n_endpoint combo "
        "with highest accuracy. Set _NET_CROSSING_Y_THRESHOLD in contact_detector.py."
    )


if __name__ == "__main__":
    main()
