"""Tune contact detection parameters via grid search.

Loads rallies with action ground truth from the database, sweeps
ContactDetectionConfig parameters, evaluates each config using
match_contacts() + compute_metrics(), and reports the Pareto frontier
of precision vs recall.

Three grid presets:
  --grid detection   : detection params only (320 configs, ~2min)
  --grid candidate   : candidate generation params only (729 configs, ~5min)
  --grid full        : candidate + detection extras (6561 configs, ~45min)

Uses trained GBM classifier by default (matching production).

Usage:
    cd analysis
    uv run python scripts/tune_contact_detection.py
    uv run python scripts/tune_contact_detection.py --grid candidate
    uv run python scripts/tune_contact_detection.py --grid full -o results.json
    uv run python scripts/tune_contact_detection.py --no-classifier  # hand-tuned gates
    uv run python scripts/tune_contact_detection.py --rally <id>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import fields

from eval_action_detection import (
    RallyData,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)

console = Console()

# Grid presets
DETECTION_GRID: dict[str, list] = {
    "min_peak_velocity": [0.005, 0.008, 0.010, 0.012, 0.015],
    "min_peak_prominence": [0.002, 0.003, 0.005, 0.006],
    "min_direction_change_deg": [15.0, 20.0, 25.0, 30.0],
    "player_contact_radius": [0.08, 0.10, 0.12, 0.15],
}

CANDIDATE_GRID: dict[str, list] = {
    "warmup_skip_frames": [3, 5, 8],
    "min_candidate_velocity": [0.002, 0.003, 0.005],
    "deceleration_min_speed_before": [0.006, 0.008, 0.012],
    "deceleration_min_drop_ratio": [0.2, 0.3, 0.4],
    "min_direction_change_deg": [15.0, 20.0, 30.0],
    "min_peak_velocity": [0.006, 0.008, 0.012],
}

# Full grid extends CANDIDATE_GRID with detection-specific params.
# Shared keys (min_direction_change_deg, min_peak_velocity) use CANDIDATE_GRID ranges.
GRID_PRESETS: dict[str, dict[str, list]] = {
    "detection": DETECTION_GRID,
    "candidate": CANDIDATE_GRID,
    "full": {**CANDIDATE_GRID, **{
        "min_peak_prominence": [0.002, 0.003, 0.005],
        "player_contact_radius": [0.10, 0.12, 0.15],
    }},
}

DEFAULT_CLASSIFIER_PATH = "weights/contact_classifier/contact_classifier.pkl"


def evaluate_config(
    config: ContactDetectionConfig,
    rallies: list[RallyData],
    tolerance_ms: int = 167,
    classifier: ContactClassifier | None = None,
    use_classifier: bool = True,
) -> dict:
    """Run contact detection + classification with config and compute metrics."""
    from rallycut.tracking.ball_tracker import BallPosition as BallPos
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

    all_matches = []
    all_unmatched = []

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
            config=config,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            classifier=classifier,
            use_classifier=use_classifier,
        )

        rally_actions = classify_rally_actions(contacts, rally.rally_id)
        pred_actions = [a.to_dict() for a in rally_actions.actions]

        # FPS-adaptive tolerance
        tolerance_frames = max(1, round(rally.fps * tolerance_ms / 1000))

        matches, unmatched = match_contacts(
            rally.gt_labels, pred_actions, tolerance=tolerance_frames
        )
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

    return compute_metrics(all_matches, all_unmatched)


def describe_config_diff(config: ContactDetectionConfig) -> str:
    """Describe how config differs from defaults."""
    default = ContactDetectionConfig()
    diffs = []
    for f in fields(ContactDetectionConfig):
        val = getattr(config, f.name)
        default_val = getattr(default, f.name)
        if val != default_val:
            diffs.append(f"{f.name}={val}")
    return ", ".join(diffs) if diffs else "(defaults)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune contact detection parameters via grid search"
    )
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument("--grid", choices=list(GRID_PRESETS.keys()), default="candidate",
                        help="Grid preset: detection (320), candidate (729), full (6561)")
    parser.add_argument("--no-classifier", action="store_true",
                        help="Disable trained classifier (use hand-tuned gates)")
    parser.add_argument("--tolerance-ms", type=int, default=167, help="Time tolerance in ms (default: 167, ~5 frames at 30fps)")
    parser.add_argument("-o", "--output", help="Save results to JSON file")
    args = parser.parse_args()

    # Load classifier (matching production behavior)
    classifier: ContactClassifier | None = None
    use_classifier = not args.no_classifier
    if use_classifier and os.path.exists(DEFAULT_CLASSIFIER_PATH):
        classifier = ContactClassifier.load(DEFAULT_CLASSIFIER_PATH)
        console.print(f"[bold]Using trained classifier: {DEFAULT_CLASSIFIER_PATH}[/bold]")
    elif use_classifier:
        console.print("[yellow]No classifier found, using hand-tuned gates[/yellow]")
        use_classifier = False

    param_grid = GRID_PRESETS[args.grid]
    rallies = load_rallies_with_action_gt(rally_id=args.rally)

    if not rallies:
        console.print("[red]No rallies with action ground truth found.[/red]")
        sys.exit(1)

    # Filter to rallies that have ball positions for redetection
    rallies_with_data = [r for r in rallies if r.ball_positions_json]
    console.print(f"Found {len(rallies_with_data)} rallies with ball position data\n")

    if not rallies_with_data:
        console.print("[red]No rallies have ball position data for redetection.[/red]")
        sys.exit(1)

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    total = len(combinations)

    console.print(f"[bold]Grid Search ({args.grid}): {total} configs[/bold]\n")

    results: list[dict] = []

    with Progress(console=console) as progress:
        task = progress.add_task("Searching...", total=total)

        for combo in combinations:
            overrides = dict(zip(param_names, combo))
            config = ContactDetectionConfig(**overrides)

            metrics = evaluate_config(
                config, rallies_with_data, args.tolerance_ms,
                classifier=classifier, use_classifier=use_classifier,
            )
            result = {
                "config": overrides,
                "config_desc": describe_config_diff(config),
                **metrics,
            }
            results.append(result)
            progress.advance(task)

    # Sort by F1
    results.sort(key=lambda r: r["f1"], reverse=True)

    # Print top 20
    console.print("\n[bold]Top 20 Configs (by F1):[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", max_width=3)
    table.add_column("Config", min_width=40)
    table.add_column("Recall", justify="right")
    table.add_column("Prec", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ActAcc", justify="right")
    table.add_column("GT", justify="right")
    table.add_column("Pred", justify="right")

    for i, r in enumerate(results[:20]):
        f1_str = f"[green]{r['f1']:.1%}[/green]" if r["f1"] > 0.25 else f"{r['f1']:.1%}"
        table.add_row(
            str(i + 1),
            r["config_desc"][:55],
            f"{r['recall']:.1%}",
            f"{r['precision']:.1%}",
            f1_str,
            f"{r['action_accuracy']:.1%}",
            str(r["total_gt"]),
            str(r["total_pred"]),
        )

    console.print(table)

    # Print Pareto frontier (non-dominated configs in recall-precision space)
    pareto: list[dict] = []
    for r in results:
        dominated = False
        for other in results:
            if (other["recall"] >= r["recall"] and other["precision"] > r["precision"]) or \
               (other["recall"] > r["recall"] and other["precision"] >= r["precision"]):
                dominated = True
                break
        if not dominated:
            pareto.append(r)

    pareto.sort(key=lambda r: r["recall"])

    console.print(f"\n[bold]Pareto Frontier ({len(pareto)} non-dominated configs):[/bold]")
    pareto_table = Table(show_header=True, header_style="bold")
    pareto_table.add_column("Config", min_width=40)
    pareto_table.add_column("Recall", justify="right")
    pareto_table.add_column("Prec", justify="right")
    pareto_table.add_column("F1", justify="right")

    for r in pareto:
        pareto_table.add_row(
            r["config_desc"][:55],
            f"{r['recall']:.1%}",
            f"{r['precision']:.1%}",
            f"{r['f1']:.1%}",
        )

    console.print(pareto_table)

    # Best by F1 summary
    best = results[0]
    console.print("\n[bold]Best Config (by F1):[/bold]")
    console.print(f"  {best['config_desc']}")
    console.print(f"  Recall: {best['recall']:.1%}, Precision: {best['precision']:.1%}, "
                  f"F1: {best['f1']:.1%}, Action Accuracy: {best['action_accuracy']:.1%}")
    console.print("\n  Apply with:")
    config_json = json.dumps(best["config"])
    console.print(f"  uv run python scripts/eval_action_detection.py --redetect --config '{config_json}'")

    # Save results
    if args.output:
        output = {
            "grid_name": args.grid,
            "grid": param_grid,
            "total_configs": total,
            "tolerance": args.tolerance_ms,
            "num_rallies": len(rallies_with_data),
            "use_classifier": use_classifier,
            "results": results[:50],  # Top 50
            "pareto": pareto,
            "best": best,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        console.print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
