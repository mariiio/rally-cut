#!/usr/bin/env python3
"""Compare tracking experiment results side-by-side.

Loads evaluation JSON outputs from each experiment and outputs a comparison
table in markdown format. Highlights per-rally regressions.

Usage:
    # Compare experiment results (expects files in current dir or specified paths)
    uv run python scripts/compare_tracking_experiments.py \
        --baseline baseline.json \
        --experiments exp_yolo11n.json exp_reid.json

    # Auto-discover experiment files
    uv run python scripts/compare_tracking_experiments.py --auto

    # Output markdown table
    uv run python scripts/compare_tracking_experiments.py --auto --format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Key metrics to compare (higher is better unless noted)
METRICS: dict[str, dict[str, Any]] = {
    "hota": {"label": "HOTA", "higher_better": True, "format": ".1f", "unit": "%"},
    "mota": {"label": "MOTA", "higher_better": True, "format": ".1f", "unit": "%"},
    "idf1": {"label": "IDF1", "higher_better": True, "format": ".1f", "unit": "%"},
    "deta": {"label": "DetA", "higher_better": True, "format": ".1f", "unit": "%"},
    "assa": {"label": "AssA", "higher_better": True, "format": ".1f", "unit": "%"},
    "id_switches": {"label": "IDsw", "higher_better": False, "format": "d", "unit": ""},
    "fragmentation": {"label": "Frag", "higher_better": False, "format": "d", "unit": ""},
    "mostly_tracked": {"label": "MT", "higher_better": True, "format": "d", "unit": ""},
    "precision": {"label": "Prec", "higher_better": True, "format": ".1f", "unit": "%"},
    "recall": {"label": "Recall", "higher_better": True, "format": ".1f", "unit": "%"},
    "f1": {"label": "F1", "higher_better": True, "format": ".1f", "unit": "%"},
}

# Regression threshold (5% relative decline flags a warning)
REGRESSION_THRESHOLD = 0.05


def load_results(path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(path) as f:
        result: dict[str, Any] = json.load(f)
    return result


def extract_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Extract key metrics from evaluation results.

    Handles both aggregate results and per-rally results.
    """
    metrics = {}

    # Try different JSON structures
    if "aggregate" in results:
        agg = results["aggregate"]
    elif "metrics" in results:
        agg = results["metrics"]
    else:
        agg = results

    for key in METRICS:
        # Try exact key
        if key in agg:
            metrics[key] = float(agg[key])
        # Try camelCase
        elif _to_camel(key) in agg:
            metrics[key] = float(agg[_to_camel(key)])
        # Try uppercase
        elif key.upper() in agg:
            metrics[key] = float(agg[key.upper()])

    return metrics


def extract_per_rally(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract per-rally metrics for regression detection."""
    per_rally = {}

    rallies = results.get("per_rally", results.get("rallies", {}))
    if isinstance(rallies, list):
        for r in rallies:
            rally_id = r.get("rally_id", r.get("rallyId", "unknown"))
            per_rally[rally_id] = extract_metrics(r)
    elif isinstance(rallies, dict):
        for rally_id, r in rallies.items():
            per_rally[rally_id] = extract_metrics(r)

    return per_rally


def _to_camel(snake: str) -> str:
    """Convert snake_case to camelCase."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def find_regressions(
    baseline_per_rally: dict[str, dict[str, float]],
    experiment_per_rally: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Find per-rally regressions compared to baseline."""
    regressions = []

    for rally_id in baseline_per_rally:
        if rally_id not in experiment_per_rally:
            continue

        base = baseline_per_rally[rally_id]
        exp = experiment_per_rally[rally_id]

        for metric_key, metric_info in METRICS.items():
            if metric_key not in base or metric_key not in exp:
                continue

            base_val = base[metric_key]
            exp_val = exp[metric_key]

            if base_val == 0:
                continue

            if metric_info["higher_better"]:
                # Regression = value decreased
                if base_val > 0 and (base_val - exp_val) / base_val > REGRESSION_THRESHOLD:
                    regressions.append({
                        "rally_id": rally_id,
                        "metric": metric_info["label"],
                        "baseline": base_val,
                        "experiment": exp_val,
                        "change": (exp_val - base_val) / base_val * 100,
                    })
            else:
                # Regression = value increased
                if base_val > 0 and (exp_val - base_val) / base_val > REGRESSION_THRESHOLD:
                    regressions.append({
                        "rally_id": rally_id,
                        "metric": metric_info["label"],
                        "baseline": base_val,
                        "experiment": exp_val,
                        "change": (exp_val - base_val) / base_val * 100,
                    })

    return regressions


def format_value(val: float, fmt: str, unit: str) -> str:
    """Format a metric value for display."""
    if fmt == "d":
        return f"{int(val)}{unit}"
    return f"{val:{fmt}}{unit}"


def format_delta(base_val: float, exp_val: float, higher_better: bool) -> str:
    """Format delta with color indicator."""
    if base_val == 0:
        return "—"
    delta = exp_val - base_val
    pct = delta / base_val * 100 if base_val != 0 else 0

    sign = "+" if delta > 0 else ""
    is_improvement = (delta > 0) == higher_better

    if abs(pct) < 0.5:
        indicator = "="
    elif is_improvement:
        indicator = "^"  # Improvement
    else:
        indicator = "v"  # Regression

    return f"{sign}{pct:.1f}% {indicator}"


def print_markdown_table(
    baseline_name: str,
    baseline_metrics: dict[str, float],
    experiments: dict[str, dict[str, float]],
) -> None:
    """Print side-by-side comparison table in markdown."""
    # Header
    exp_names = list(experiments.keys())
    header = f"| Metric | {baseline_name} |"
    separator = "|--------|" + "-" * (len(baseline_name) + 2) + "|"

    for name in exp_names:
        header += f" {name} | delta |"
        separator += "-" * (len(name) + 2) + "|-------|"

    print(header)
    print(separator)

    # Rows
    for metric_key, metric_info in METRICS.items():
        if metric_key not in baseline_metrics:
            continue

        base_val = baseline_metrics[metric_key]
        row = f"| **{metric_info['label']}** | {format_value(base_val, metric_info['format'], metric_info['unit'])} |"

        for name in exp_names:
            exp_metrics = experiments[name]
            if metric_key in exp_metrics:
                exp_val = exp_metrics[metric_key]
                row += f" {format_value(exp_val, metric_info['format'], metric_info['unit'])} |"
                row += f" {format_delta(base_val, exp_val, metric_info['higher_better'])} |"
            else:
                row += " — | — |"

        print(row)


def print_regressions(
    exp_name: str,
    regressions: list[dict[str, Any]],
) -> None:
    """Print regression warnings."""
    if not regressions:
        return

    print(f"\n### Regressions in {exp_name}")
    print()
    print("| Rally | Metric | Baseline | Experiment | Change |")
    print("|-------|--------|----------|------------|--------|")

    for r in sorted(regressions, key=lambda x: abs(x["change"]), reverse=True):
        short_id = r["rally_id"][:8] if len(r["rally_id"]) > 8 else r["rally_id"]
        print(
            f"| {short_id} | {r['metric']} "
            f"| {r['baseline']:.1f} | {r['experiment']:.1f} "
            f"| {r['change']:+.1f}% |"
        )


def auto_discover_results() -> dict[str, Path]:
    """Auto-discover experiment result files."""
    discovered = {}

    # Check common locations
    search_dirs = [
        Path("."),
        Path("results"),
        Path("../"),
    ]

    patterns = {
        "baseline": ["baseline.json", "baseline_tracking.json"],
    }

    # Auto-discover any exp_*.json files
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for f in sorted(search_dir.glob("exp_*.json")):
            name = f.stem.replace("exp_", "").replace("_", " ").title()
            if name not in discovered:
                discovered[name] = f

    for exp_name, filenames in patterns.items():
        for search_dir in search_dirs:
            for filename in filenames:
                path = search_dir / filename
                if path.exists():
                    discovered[exp_name] = path
                    break
            if exp_name in discovered:
                break

    return discovered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare tracking experiment results"
    )
    parser.add_argument(
        "--baseline", "-b",
        type=Path,
        help="Baseline evaluation JSON",
    )
    parser.add_argument(
        "--experiments", "-e",
        type=Path,
        nargs="+",
        help="Experiment evaluation JSONs",
    )
    parser.add_argument(
        "--names", "-n",
        nargs="+",
        help="Names for each experiment (parallel with --experiments)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-discover result files",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    if args.auto:
        discovered = auto_discover_results()
        if not discovered:
            print("No experiment results found. Run experiments first.")
            sys.exit(1)

        print(f"Found {len(discovered)} result files:")
        for name, path in discovered.items():
            print(f"  {name}: {path}")
        print()

        if "baseline" not in discovered:
            print("Warning: No baseline found. Using first experiment as baseline.")
            first_name = next(iter(discovered))
            baseline_path = discovered.pop(first_name)
            baseline_name = first_name
        else:
            baseline_path = discovered.pop("baseline")
            baseline_name = "Baseline"

        baseline_results = load_results(baseline_path)
        baseline_metrics = extract_metrics(baseline_results)
        baseline_per_rally = extract_per_rally(baseline_results)

        experiments = {}
        for name, path in discovered.items():
            results = load_results(path)
            experiments[name] = extract_metrics(results)

        print("## Tracking Experiment Comparison\n")
        print_markdown_table(baseline_name, baseline_metrics, experiments)

        # Check for regressions
        for name, path in discovered.items():
            results = load_results(path)
            exp_per_rally = extract_per_rally(results)
            regressions = find_regressions(baseline_per_rally, exp_per_rally)
            print_regressions(name, regressions)

    elif args.baseline and args.experiments:
        baseline_results = load_results(args.baseline)
        baseline_metrics = extract_metrics(baseline_results)
        baseline_per_rally = extract_per_rally(baseline_results)

        names = args.names or [p.stem for p in args.experiments]
        experiments = {}
        for name, path in zip(names, args.experiments):
            results = load_results(path)
            experiments[name] = extract_metrics(results)

        print("## Tracking Experiment Comparison\n")
        print_markdown_table("Baseline", baseline_metrics, experiments)

        # Check for regressions
        for name, path in zip(names, args.experiments):
            results = load_results(path)
            exp_per_rally = extract_per_rally(results)
            regressions = find_regressions(baseline_per_rally, exp_per_rally)
            print_regressions(name, regressions)

    else:
        parser.print_help()
        print("\nExample:")
        print("  python scripts/compare_tracking_experiments.py --auto")
        print("  python scripts/compare_tracking_experiments.py -b baseline.json -e exp_yolo11n.json exp_reid.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
