"""Compare GBM vs BiLSTM vs Hybrid on a train/held-out split.

Trains each model on the train partition (~75% of videos by hash) and
evaluates on the held-out partition (~25%).  Reports action accuracy,
attribution accuracy, and sanity violation counts side-by-side.

Usage:
    cd analysis
    uv run python scripts/eval_baseline_comparison.py
    uv run python scripts/eval_baseline_comparison.py --skip-lstm   # GBM only (fast)
    uv run python scripts/eval_baseline_comparison.py --max-epochs 100
"""

from __future__ import annotations

import argparse

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.sanity_checks import check_illegal_sequences
from rallycut.evaluation.split import video_split
from scripts.eval_bilstm_joint import (
    ACTION_CLASSES,
    ACTION_FEAT_DIM,
    RallySequence,
    _normalize_features,
    evaluate_bilstm,
    load_all_sequences,
    predict_bilstm_actions,
    run_gbm_fold,
    train_bilstm,
)

console = Console()


# ---------------------------------------------------------------------------
# GBM per-rally action predictions (for sanity checks)
# ---------------------------------------------------------------------------

def predict_gbm_actions(
    train_seqs: list[RallySequence],
    test_seqs: list[RallySequence],
) -> dict[str, list[str]]:
    """Train GBM action classifier and return per-rally predictions."""
    from sklearn.ensemble import GradientBoostingClassifier

    train_x, train_y = [], []
    for seq in train_seqs:
        for c in seq.contacts:
            train_x.append(c.features[:ACTION_FEAT_DIM])
            train_y.append(c.action_label)

    if not train_x:
        return {}

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        min_samples_leaf=3, subsample=0.8, random_state=42,
    )
    clf.fit(np.array(train_x), np.array(train_y))

    predictions: dict[str, list[str]] = {}
    for seq in test_seqs:
        feats = np.array([c.features[:ACTION_FEAT_DIM] for c in seq.contacts])
        preds = clf.predict(feats)
        predictions[seq.rally_id] = [ACTION_CLASSES[int(p)] for p in preds]

    return predictions


# ---------------------------------------------------------------------------
# Sanity check runner
# ---------------------------------------------------------------------------

def count_illegal_sequences(
    predictions: dict[str, list[str]],
) -> int:
    """Count illegal same-side action repeats across all rally predictions."""
    total = 0
    for rally_id, preds in predictions.items():
        total += len(check_illegal_sequences(preds, rally_id=rally_id))
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GBM vs BiLSTM vs Hybrid on train/held-out split"
    )
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM (GBM only)")
    args = parser.parse_args()

    # --- Load data ---
    all_sequences = load_all_sequences()
    if not all_sequences:
        console.print("[red]No sequences extracted. Check DB connection.[/red]")
        return

    # --- Split ---
    train_seqs = [s for s in all_sequences if video_split(s.video_id) == "train"]
    test_seqs = [s for s in all_sequences if video_split(s.video_id) == "held_out"]

    train_vids = sorted(set(s.video_id for s in train_seqs))
    test_vids = sorted(set(s.video_id for s in test_seqs))
    console.print(
        f"\n[bold]Split:[/bold] {len(train_vids)} train videos ({len(train_seqs)} rallies), "
        f"{len(test_vids)} held-out videos ({len(test_seqs)} rallies)"
    )
    console.print(f"  Train videos: {', '.join(v[:8] for v in train_vids)}")
    console.print(f"  Held-out videos: {', '.join(v[:8] for v in test_vids)}")

    if not test_seqs or not train_seqs:
        console.print("[red]Empty split partition. Cannot evaluate.[/red]")
        return

    # --- Config A: GBM ---
    console.print("\n[bold]Config A: GBM + Viterbi[/bold]")
    gbm_action, gbm_attr, n_contacts, n_attr = run_gbm_fold(train_seqs, test_seqs)
    gbm_preds = predict_gbm_actions(train_seqs, test_seqs)
    gbm_illegal = count_illegal_sequences(gbm_preds)
    console.print(
        f"  Action: {gbm_action:.1%}, Attr: {gbm_attr:.1%}, "
        f"Illegal seq: {gbm_illegal}"
    )

    # --- Config B: BiLSTM ---
    lstm_action, lstm_attr = 0.0, 0.0
    lstm_illegal = 0
    if not args.skip_lstm:
        console.print("\n[bold]Config B: BiLSTM[/bold]")
        train_norm, test_norm = _normalize_features(train_seqs, test_seqs)
        model = train_bilstm(
            train_norm,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        )
        lstm_action, lstm_attr, _, _ = evaluate_bilstm(model, test_norm)
        lstm_preds = predict_bilstm_actions(model, test_norm)
        lstm_illegal = count_illegal_sequences(lstm_preds)
        console.print(
            f"  Action: {lstm_action:.1%}, Attr: {lstm_attr:.1%}, "
            f"Illegal seq: {lstm_illegal}"
        )

    # --- Config C: Hybrid (best of each task) ---
    hybrid_action = max(gbm_action, lstm_action)
    hybrid_attr = max(gbm_attr, lstm_attr)
    hybrid_action_src = "GBM" if gbm_action >= lstm_action else "LSTM"
    hybrid_attr_src = "GBM" if gbm_attr >= lstm_attr else "LSTM"
    hybrid_illegal = gbm_illegal if hybrid_action_src == "GBM" else lstm_illegal

    # --- Comparison table ---
    console.print()
    table = Table(title="Baseline Comparison (held-out)")
    table.add_column("Config", style="bold")
    table.add_column("Action Acc", justify="right")
    table.add_column("Attr Acc", justify="right")
    table.add_column("Illegal Seq", justify="right")

    table.add_row(
        "GBM + Viterbi",
        f"{gbm_action:.1%}",
        f"{gbm_attr:.1%}",
        str(gbm_illegal),
    )
    if not args.skip_lstm:
        table.add_row(
            "BiLSTM",
            f"{lstm_action:.1%}",
            f"{lstm_attr:.1%}",
            str(lstm_illegal),
        )
        table.add_row(
            f"Hybrid ({hybrid_action_src} act / {hybrid_attr_src} attr)",
            f"{hybrid_action:.1%}",
            f"{hybrid_attr:.1%}",
            str(hybrid_illegal),
        )

    console.print(table)
    console.print(f"\n  Contacts: {n_contacts}, Attribution-evaluable: {n_attr}")


if __name__ == "__main__":
    main()
