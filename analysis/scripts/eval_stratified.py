"""Stratified evaluation: group tracking metrics by video characteristics.

Loads labeled rallies, groups by characteristic categories (brightness,
camera distance, scene complexity), and prints per-group metrics.

Usage:
    cd analysis
    uv run python scripts/eval_stratified.py
    uv run python scripts/eval_stratified.py --ball-only
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.tracking.ball_metrics import (
    BallTrackingMetrics,
    evaluate_ball_tracking,
    find_optimal_frame_offset,
)
from rallycut.evaluation.tracking.db import (
    TrackingEvaluationRally,
    load_labeled_rallies,
)
from rallycut.evaluation.tracking.metrics import evaluate_rally

console = Console()


@dataclass
class GroupMetrics:
    """Aggregated metrics for a group of rallies."""

    rally_count: int = 0
    ball_match_rates: list[float] = field(default_factory=list)
    ball_errors: list[float] = field(default_factory=list)
    hota_scores: list[float] = field(default_factory=list)


def get_category(
    rally: TrackingEvaluationRally, characteristic: str
) -> str:
    """Get category for a rally, or 'unknown' if not available."""
    chars = rally.video_characteristics
    if not chars:
        return "unknown"
    section = chars.get(characteristic)
    if not section or not isinstance(section, dict):
        return "unknown"
    return str(section.get("category", "unknown"))


def evaluate_rallies(
    rallies: list[TrackingEvaluationRally],
    ball_only: bool = False,
) -> dict[str, dict[str, GroupMetrics]]:
    """Evaluate rallies and group by characteristics."""
    characteristics = ["brightness", "cameraDistance", "sceneComplexity"]
    groups: dict[str, dict[str, GroupMetrics]] = {
        c: defaultdict(GroupMetrics) for c in characteristics
    }

    for rally in rallies:
        # Ball metrics
        ball_gt = [
            p
            for p in rally.ground_truth.positions
            if p.label == "ball"
        ]

        ball_metrics: BallTrackingMetrics | None = None
        if ball_gt and rally.predictions and rally.predictions.ball_positions:
            offset, _ = find_optimal_frame_offset(
                ball_gt,
                rally.predictions.ball_positions,
                rally.video_width,
                rally.video_height,
            )
            shifted = [
                type(bp)(
                    frame_number=bp.frame_number + offset,
                    x=bp.x,
                    y=bp.y,
                    confidence=bp.confidence,
                )
                for bp in rally.predictions.ball_positions
            ]
            ball_metrics = evaluate_ball_tracking(
                ball_gt,
                shifted,
                rally.video_width,
                rally.video_height,
                rally.video_fps,
            )

        # Player metrics
        hota: float | None = None
        if not ball_only and rally.predictions:
            try:
                result = evaluate_rally(
                    rally.rally_id,
                    rally.ground_truth,
                    rally.predictions,
                    video_width=rally.video_width,
                    video_height=rally.video_height,
                )
                if result.hota_metrics:
                    hota = result.hota_metrics.hota
            except Exception as e:
                console.print(f"[dim]Skip player eval {rally.rally_id[:8]}: {e}[/dim]")

        # Group by each characteristic
        for char_name in characteristics:
            cat = get_category(rally, char_name)
            g = groups[char_name][cat]
            g.rally_count += 1

            if ball_metrics and ball_metrics.num_gt_frames > 0:
                match_rate = ball_metrics.num_matched / ball_metrics.num_gt_frames
                g.ball_match_rates.append(match_rate)
                if ball_metrics.mean_error_px is not None:
                    g.ball_errors.append(ball_metrics.mean_error_px)

            if hota is not None:
                g.hota_scores.append(hota)

    return groups


def print_results(
    groups: dict[str, dict[str, GroupMetrics]],
    ball_only: bool = False,
) -> None:
    """Print stratified results as tables."""
    for char_name, categories in groups.items():
        if not categories:
            continue

        table = Table(title=char_name, show_header=True)
        table.add_column("Category", style="bold")
        table.add_column("N", justify="right")
        table.add_column("Ball Match%", justify="right")
        table.add_column("Ball Error(px)", justify="right")
        if not ball_only:
            table.add_column("HOTA%", justify="right")

        # Sort categories for consistent display
        for cat in sorted(categories.keys()):
            g = categories[cat]
            match_str = (
                f"{sum(g.ball_match_rates) / len(g.ball_match_rates) * 100:.1f}%"
                if g.ball_match_rates
                else "-"
            )
            error_str = (
                f"{sum(g.ball_errors) / len(g.ball_errors):.1f}"
                if g.ball_errors
                else "-"
            )

            row = [cat, str(g.rally_count), match_str, error_str]

            if not ball_only:
                hota_str = (
                    f"{sum(g.hota_scores) / len(g.hota_scores) * 100:.1f}%"
                    if g.hota_scores
                    else "-"
                )
                row.append(hota_str)

            table.add_row(*row)

        console.print(table)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified tracking evaluation")
    parser.add_argument("--ball-only", "-b", action="store_true", help="Ball metrics only")
    args = parser.parse_args()

    console.print("[bold]Loading labeled rallies...[/bold]")
    rallies = load_labeled_rallies()
    console.print(f"Loaded {len(rallies)} rallies")

    # Show characteristic coverage
    has_chars = sum(1 for r in rallies if r.video_characteristics)
    console.print(f"With characteristics: {has_chars}/{len(rallies)}")
    console.print()

    groups = evaluate_rallies(rallies, ball_only=args.ball_only)
    print_results(groups, ball_only=args.ball_only)


if __name__ == "__main__":
    main()
