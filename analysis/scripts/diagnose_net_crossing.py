"""Diagnose net crossing detection reliability.

For each pair of consecutive GT contacts, checks:
1. Whether ball_crossed_net returns True/False/None
2. Whether the GT says a crossing occurred (different sides = crossing)
3. What the ball trajectory looks like between contacts

This reveals the root cause of contact_count_on_current_side being unreliable
(0.004 feature importance in action classifier).

Usage:
    cd analysis
    uv run python scripts/diagnose_net_crossing.py
"""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.table import Table

from rallycut.tracking.contact_detector import ball_crossed_net
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from scripts.eval_action_detection import load_rallies_with_action_gt

console = Console()

# Net crossing GT inference: consecutive contacts with different expected sides
# Beach volleyball sequence: serve(A) → receive(B) → set(B) → attack(B) → dig(A) → ...
# Side changes happen between: serve→receive, attack→dig (or receive), block→dig
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
    ("set", "attack"),
    ("dig", "attack"),  # skip set
    ("receive", "attack"),  # skip set
}


def main() -> None:
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    console.print(f"\n[bold]Net Crossing Detection Diagnosis ({len(rallies)} rallies)[/bold]\n")

    results: list[dict] = []

    for rally in rallies:
        if not rally.ball_positions_json:
            continue

        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
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

            # Determine GT expectation
            if pair in SIDE_CHANGES:
                gt_crossed = True
            elif pair in SAME_SIDE:
                gt_crossed = False
            else:
                gt_crossed = None  # Can't determine from action types alone

            # Run ball_crossed_net
            pred = ball_crossed_net(
                ball_positions,
                from_frame=prev_gt.frame,
                to_frame=curr_gt.frame,
                net_y=net_y,
            )

            # Count ball positions in range
            in_range = [
                bp for bp in ball_positions
                if prev_gt.frame < bp.frame_number < curr_gt.frame
            ]
            frame_gap = curr_gt.frame - prev_gt.frame

            # Ball Y trajectory summary
            if in_range:
                ys = [bp.y for bp in in_range]
                min_y = min(ys)
                max_y = max(ys)
                crosses_net_y = min_y < net_y < max_y
            else:
                min_y = max_y = 0.0
                crosses_net_y = False

            results.append({
                "rally_id": rally.rally_id,
                "prev_action": prev_gt.action,
                "curr_action": curr_gt.action,
                "prev_frame": prev_gt.frame,
                "curr_frame": curr_gt.frame,
                "frame_gap": frame_gap,
                "gt_crossed": gt_crossed,
                "pred": pred,
                "n_ball_positions": len(in_range),
                "min_y": min_y,
                "max_y": max_y,
                "net_y": net_y,
                "crosses_net_y": crosses_net_y,
            })

    # === Summary ===
    total = len(results)
    evaluable = [r for r in results if r["gt_crossed"] is not None]
    n_eval = len(evaluable)

    console.print(f"Total consecutive GT pairs: {total}")
    console.print(f"Evaluable (known crossing/no-crossing): {n_eval}")

    # Prediction distribution
    pred_counts = Counter(r["pred"] for r in results)
    console.print(f"\nPrediction distribution (all {total}):")
    console.print(f"  True (crossed):  {pred_counts.get(True, 0)}")
    console.print(f"  False (no cross): {pred_counts.get(False, 0)}")
    console.print(f"  None (unknown):  {pred_counts.get(None, 0)}")

    # Accuracy on evaluable pairs
    if evaluable:
        correct = sum(1 for r in evaluable if r["pred"] == r["gt_crossed"])
        wrong = sum(1 for r in evaluable if r["pred"] is not None and r["pred"] != r["gt_crossed"])
        none_count = sum(1 for r in evaluable if r["pred"] is None)

        console.print(f"\n[bold]Evaluable accuracy:[/bold]")
        console.print(f"  Correct:  {correct} ({correct / n_eval:.0%})")
        console.print(f"  Wrong:    {wrong} ({wrong / n_eval:.0%})")
        console.print(f"  Unknown:  {none_count} ({none_count / n_eval:.0%})")

        # Break down by GT expectation
        for gt_val, label in [(True, "Should cross"), (False, "Should NOT cross")]:
            subset = [r for r in evaluable if r["gt_crossed"] == gt_val]
            if not subset:
                continue
            n = len(subset)
            t = sum(1 for r in subset if r["pred"] is True)
            f = sum(1 for r in subset if r["pred"] is False)
            u = sum(1 for r in subset if r["pred"] is None)
            console.print(f"\n  {label} ({n}):")
            console.print(f"    pred=True:  {t} ({t / n:.0%})")
            console.print(f"    pred=False: {f} ({f / n:.0%})")
            console.print(f"    pred=None:  {u} ({u / n:.0%})")

    # === Failure analysis ===
    failures = [r for r in evaluable if r["pred"] is not None and r["pred"] != r["gt_crossed"]]
    nones = [r for r in evaluable if r["pred"] is None]

    if failures:
        fail_table = Table(title=f"\nWrong Predictions ({len(failures)})")
        fail_table.add_column("Rally", style="dim", max_width=8)
        fail_table.add_column("Prev→Curr")
        fail_table.add_column("Frames", justify="right")
        fail_table.add_column("Gap", justify="right")
        fail_table.add_column("BallPts", justify="right")
        fail_table.add_column("GT", justify="center")
        fail_table.add_column("Pred", justify="center")
        fail_table.add_column("Y range")
        fail_table.add_column("NetY", justify="right")

        for r in sorted(failures, key=lambda x: x["rally_id"]):
            fail_table.add_row(
                r["rally_id"][:8],
                f"{r['prev_action'][:3]}→{r['curr_action'][:3]}",
                f"{r['prev_frame']}-{r['curr_frame']}",
                str(r["frame_gap"]),
                str(r["n_ball_positions"]),
                str(r["gt_crossed"]),
                str(r["pred"]),
                f"{r['min_y']:.3f}-{r['max_y']:.3f}",
                f"{r['net_y']:.3f}",
            )
        console.print(fail_table)

    if nones:
        # Analyze why None is returned
        none_reasons: Counter[str] = Counter()
        for r in nones:
            if r["n_ball_positions"] < 4:  # min_frames_per_side * 2
                none_reasons["too_few_ball_positions"] += 1
            else:
                none_reasons["cant_determine_starting_side"] += 1

        console.print(f"\n[bold]None (unknown) reasons ({len(nones)}):[/bold]")
        for reason, count in none_reasons.most_common():
            console.print(f"  {reason}: {count}")

        # Frame gap distribution for None cases
        none_gaps = [r["frame_gap"] for r in nones]
        if none_gaps:
            import numpy as np
            console.print(f"  Frame gap: median={int(np.median(none_gaps))}, "
                          f"mean={np.mean(none_gaps):.0f}, "
                          f"<10: {sum(1 for g in none_gaps if g < 10)}/{len(none_gaps)}")

    # === Per-action-pair breakdown ===
    pair_table = Table(title="\nPer Action Pair")
    pair_table.add_column("Pair")
    pair_table.add_column("GT", justify="center")
    pair_table.add_column("N", justify="right")
    pair_table.add_column("Correct", justify="right")
    pair_table.add_column("Wrong", justify="right")
    pair_table.add_column("None", justify="right")
    pair_table.add_column("Accuracy", justify="right")

    pairs_seen: dict[tuple[str, str, bool | None], list[dict]] = {}
    for r in evaluable:
        key = (r["prev_action"], r["curr_action"], r["gt_crossed"])
        pairs_seen.setdefault(key, []).append(r)

    for (prev, curr, gt), items in sorted(pairs_seen.items(), key=lambda x: -len(x[1])):
        n = len(items)
        c = sum(1 for r in items if r["pred"] == gt)
        w = sum(1 for r in items if r["pred"] is not None and r["pred"] != gt)
        u = sum(1 for r in items if r["pred"] is None)
        acc = c / max(1, c + w)
        pair_table.add_row(
            f"{prev[:3]}→{curr[:3]}",
            "cross" if gt else "same",
            str(n),
            str(c),
            str(w),
            str(u),
            f"{acc:.0%}" if (c + w) > 0 else "-",
        )

    console.print(pair_table)


if __name__ == "__main__":
    main()
