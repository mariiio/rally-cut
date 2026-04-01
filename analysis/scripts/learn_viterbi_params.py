"""Learn Viterbi transition and emission parameters from GT data.

Counts actual action transitions in labeled rallies and computes
confusion-based emission probabilities from LOO-CV predictions.

Usage:
    cd analysis
    uv run python scripts/learn_viterbi_params.py
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

ACTION_TYPES = ["serve", "receive", "set", "attack", "block", "dig"]
# Viterbi only relabels these; serve/receive/block are fixed
VITERBI_ACTIONS = ["dig", "set", "attack"]


def count_gt_transitions(rallies: list[RallyData]) -> dict[tuple[str, str], int]:
    """Count action→action transitions in GT labels."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for rally in rallies:
        labels = sorted(rally.gt_labels, key=lambda g: g.frame)
        for i in range(1, len(labels)):
            prev_action = labels[i - 1].action
            curr_action = labels[i].action
            if prev_action in ACTION_TYPES and curr_action in ACTION_TYPES:
                counts[(prev_action, curr_action)] += 1
    return dict(counts)


def transitions_to_probs(
    counts: dict[tuple[str, str], int],
    smoothing: float = 0.5,
) -> dict[tuple[str, str], float]:
    """Convert transition counts to conditional probabilities with smoothing.

    P(next | prev) = (count + smoothing) / (total_from_prev + smoothing * n_actions)
    """
    # Group by prev action
    from_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (prev, curr), count in counts.items():
        from_counts[prev][curr] = count

    probs: dict[tuple[str, str], float] = {}
    n_actions = len(ACTION_TYPES)

    for prev in ACTION_TYPES:
        total = sum(from_counts[prev].values()) + smoothing * n_actions
        for curr in ACTION_TYPES:
            raw = from_counts[prev].get(curr, 0)
            p = (raw + smoothing) / total
            if p > 0.001:  # Only include non-negligible transitions
                probs[(prev, curr)] = round(p, 4)

    return probs


def compute_confusion_matrix(
    rallies: list[RallyData],
) -> dict[str, dict[str, int]]:
    """Compute confusion matrix from stored predictions vs GT.

    Uses the actions_json stored in DB (current pipeline predictions)
    matched against GT labels.
    """
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_matched = 0

    for rally in rallies:
        if not rally.actions_json:
            continue

        # Extract predicted actions from stored pipeline output
        actions_data = rally.actions_json
        pred_list = actions_data.get("actions", []) if isinstance(actions_data, dict) else actions_data
        if not pred_list:
            continue

        real_pred = [a for a in pred_list if not a.get("isSynthetic")]
        tolerance = max(1, round(rally.fps * 167 / 1000))  # ±167ms

        matches, _ = match_contacts(rally.gt_labels, real_pred, tolerance=tolerance)

        for m in matches:
            if m.pred_action is not None and m.gt_action in ACTION_TYPES:
                confusion[m.gt_action][m.pred_action] += 1
                n_matched += 1

    console.print(f"  Confusion matrix from {n_matched} matched contacts")
    return {k: dict(v) for k, v in confusion.items()}


def confusion_to_emission_probs(
    confusion: dict[str, dict[str, int]],
    smoothing: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Convert confusion matrix to emission probabilities.

    P(observed=X | true=Y) for Viterbi-eligible actions.
    """
    emissions: dict[str, dict[str, float]] = {}

    for true_action in VITERBI_ACTIONS:
        row = confusion.get(true_action, {})
        total = sum(row.get(obs, 0) for obs in VITERBI_ACTIONS)
        total += smoothing * len(VITERBI_ACTIONS)

        probs: dict[str, float] = {}
        for obs_action in VITERBI_ACTIONS:
            count = row.get(obs_action, 0)
            probs[obs_action] = round((count + smoothing) / total, 4)
        emissions[true_action] = probs

    return emissions


def format_transition_dict(probs: dict[tuple[str, str], float]) -> str:
    """Format transitions as Python dict literal for action_classifier.py."""
    name_map = {
        "serve": "ActionType.SERVE",
        "receive": "ActionType.RECEIVE",
        "set": "ActionType.SET",
        "attack": "ActionType.ATTACK",
        "block": "ActionType.BLOCK",
        "dig": "ActionType.DIG",
    }

    lines = []
    # Group by prev action for readability
    for prev in ACTION_TYPES:
        group = [(curr, p) for (pr, curr), p in sorted(probs.items()) if pr == prev]
        if not group:
            continue
        lines.append(f"    # After {prev}")
        for curr, p in group:
            lines.append(f"    ({name_map[prev]}, {name_map[curr]}): {p},")

    return "{\n" + "\n".join(lines) + "\n}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn Viterbi parameters from GT data"
    )
    parser.parse_args()

    t_start = time.monotonic()

    # --- Load GT data ---
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies")

    # --- Count GT transitions ---
    console.print("\n[bold]Counting GT transitions...[/bold]")
    counts = count_gt_transitions(rallies)
    total_transitions = sum(counts.values())
    console.print(f"  {total_transitions} total transitions")

    # --- Display raw counts ---
    count_table = Table(title="GT Transition Counts")
    count_table.add_column("From \\ To", style="cyan")
    for a in ACTION_TYPES:
        count_table.add_column(a[:3], justify="right")
    count_table.add_column("Total", justify="right", style="bold")

    for prev in ACTION_TYPES:
        row_total = sum(counts.get((prev, curr), 0) for curr in ACTION_TYPES)
        cells = [str(counts.get((prev, curr), 0)) for curr in ACTION_TYPES]
        count_table.add_row(prev, *cells, str(row_total))
    console.print(count_table)

    # --- Compute transition probabilities ---
    probs = transitions_to_probs(counts, smoothing=0.5)

    prob_table = Table(title="Learned Transition Probabilities P(next | prev)")
    prob_table.add_column("From \\ To", style="cyan")
    for a in ACTION_TYPES:
        prob_table.add_column(a[:3], justify="right")

    for prev in ACTION_TYPES:
        cells = []
        for curr in ACTION_TYPES:
            p = probs.get((prev, curr), 0.0)
            if p >= 0.10:
                cells.append(f"[bold]{p:.2f}[/bold]")
            elif p >= 0.01:
                cells.append(f"{p:.2f}")
            else:
                cells.append(f"[dim]{p:.3f}[/dim]")
        prob_table.add_row(prev, *cells)
    console.print(prob_table)

    # --- Compare with hand-tuned ---
    from rallycut.tracking.action_classifier import _VITERBI_TRANSITIONS, ActionType

    action_type_map = {
        "serve": ActionType.SERVE,
        "receive": ActionType.RECEIVE,
        "set": ActionType.SET,
        "attack": ActionType.ATTACK,
        "block": ActionType.BLOCK,
        "dig": ActionType.DIG,
    }

    compare_table = Table(title="Hand-Tuned vs Learned (key transitions)")
    compare_table.add_column("Transition", style="cyan")
    compare_table.add_column("Hand-tuned", justify="right")
    compare_table.add_column("Learned", justify="right")
    compare_table.add_column("Δ", justify="right")

    for (prev_str, curr_str), learned_p in sorted(probs.items()):
        prev_at = action_type_map[prev_str]
        curr_at = action_type_map[curr_str]
        hand_p = _VITERBI_TRANSITIONS.get((prev_at, curr_at), 0.001)

        if learned_p < 0.02 and hand_p < 0.02:
            continue  # Skip negligible transitions

        delta = learned_p - hand_p
        delta_str = f"{delta:+.2f}"
        style = "green" if abs(delta) < 0.05 else "yellow" if abs(delta) < 0.15 else "red"

        compare_table.add_row(
            f"{prev_str}→{curr_str}",
            f"{hand_p:.2f}",
            f"{learned_p:.2f}",
            delta_str,
            style=style,
        )
    console.print(compare_table)

    # --- Confusion-based emissions ---
    console.print("\n[bold]Computing confusion-based emission probabilities...[/bold]")
    confusion = compute_confusion_matrix(rallies)

    if confusion:
        conf_table = Table(title="Stored Prediction Confusion Matrix")
        conf_table.add_column("True \\ Pred", style="cyan")
        for a in VITERBI_ACTIONS:
            conf_table.add_column(a[:3], justify="right")
        conf_table.add_column("Acc", justify="right", style="bold")

        for true_a in VITERBI_ACTIONS:
            row = confusion.get(true_a, {})
            total = sum(row.get(obs, 0) for obs in VITERBI_ACTIONS)
            correct = row.get(true_a, 0)
            acc = correct / max(1, total)
            cells = [str(row.get(obs, 0)) for obs in VITERBI_ACTIONS]
            conf_table.add_row(true_a, *cells, f"{acc:.0%}")
        console.print(conf_table)

        emissions = confusion_to_emission_probs(confusion)

        em_table = Table(title="Learned Emission Probabilities P(observed | true)")
        em_table.add_column("True \\ Obs", style="cyan")
        for a in VITERBI_ACTIONS:
            em_table.add_column(a[:3], justify="right")

        for true_a in VITERBI_ACTIONS:
            cells = [f"{emissions[true_a][obs]:.2f}" for obs in VITERBI_ACTIONS]
            em_table.add_row(true_a, *cells)
        console.print(em_table)
    else:
        console.print("  [yellow]No stored predictions available for confusion matrix[/yellow]")

    # --- Print Python code for copy-paste ---
    console.print("\n[bold]Generated Python code for _VITERBI_TRANSITIONS:[/bold]")
    print(f"_VITERBI_TRANSITIONS = {format_transition_dict(probs)}")

    elapsed = time.monotonic() - t_start
    console.print(f"\n[bold]Total time: {elapsed:.1f}s[/bold]")


if __name__ == "__main__":
    main()
