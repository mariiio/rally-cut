"""Teacher-student bootstrapping for TrackNet ball tracking.

Round N:
  1. Run TrackNet (teacher) inference on all rally videos
  2. Filter predictions by confidence threshold
  3. Override with gold GT labels
  4. Write new pseudo-label CSVs
  5. Upload to Modal and retrain (student)

Usage:
    cd analysis

    # Generate improved pseudo-labels from TrackNet teacher
    uv run python scripts/tracknet_teacher_student.py --model last --min-conf 0.5

    # Upload new labels and retrain
    uv run rallycut train tracknet-modal --upload --data-dir experiments/pseudo_labels_r1
    uv run rallycut train tracknet-modal --epochs 30 --fresh
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def generate_teacher_labels(
    model_name: str = "last",
    output_dir: Path = Path("experiments/pseudo_labels_r1"),
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    """Run TrackNet teacher on all GT rally videos and export pseudo-labels.

    Returns dict with per-rally stats.
    """
    from rallycut.evaluation.tracking.ball_grid_search import BallRawCache
    from rallycut.evaluation.tracking.db import get_video_path, load_labeled_rallies
    from rallycut.tracking.ball_tracker import BallPosition
    from scripts.eval_tracknet import load_tracknet, run_tracknet_inference

    model = load_tracknet(model_name)
    rallies = load_labeled_rallies()
    ball_cache = BallRawCache()

    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, Any] = {}

    for rally in rallies:
        rally_id = rally.rally_id
        rally_label = rally_id[:8]
        console.print(f"\n[bold]Rally {rally_label}[/bold] (video {rally.video_id[:8]})")

        video_path = get_video_path(rally.video_id)
        if video_path is None:
            console.print("  [yellow]Video not available, skipping[/yellow]")
            continue

        # --- Run TrackNet teacher inference ---
        t0 = time.time()
        tracknet_preds = run_tracknet_inference(
            model, video_path, rally.start_ms, rally.end_ms
        )
        inference_time = time.time() - t0

        # --- Also get VballNet predictions for comparison/fallback ---
        cached = ball_cache.get(rally_id)
        vballnet_preds: list[BallPosition] = []
        if cached is not None:
            vballnet_preds = [
                bp for bp in cached.raw_ball_positions
                if bp.confidence >= 0.3 and bp.motion_energy >= 0.02
            ]

        # --- Build frame -> position mapping ---
        # Start with VballNet as base (fills in frames TrackNet misses)
        frame_data: dict[int, dict[str, Any]] = {}
        max_frame = 0
        vballnet_count = 0

        for bp in vballnet_preds:
            if bp.confidence >= 0.3 and bp.motion_energy >= 0.02:
                frame_data[bp.frame_number] = {
                    "visibility": 1,
                    "x": max(0.0, min(bp.x, 1.0)),
                    "y": max(0.0, min(bp.y, 1.0)),
                    "source": "vballnet",
                }
                max_frame = max(max_frame, bp.frame_number)
                vballnet_count += 1

        # Override with TrackNet predictions (higher quality where available)
        tracknet_count = 0
        tracknet_filtered = 0
        for bp in tracknet_preds:
            max_frame = max(max_frame, bp.frame_number)
            if bp.confidence < min_confidence:
                tracknet_filtered += 1
                continue
            frame_data[bp.frame_number] = {
                "visibility": 1,
                "x": max(0.0, min(bp.x, 1.0)),
                "y": max(0.0, min(bp.y, 1.0)),
                "source": "tracknet",
            }
            tracknet_count += 1

        # Override with gold GT labels (highest priority)
        gold_count = 0
        if rally.ground_truth:
            gt_ball = [p for p in rally.ground_truth.positions if p.label == "ball"]
            for gt_pos in gt_ball:
                frame_data[gt_pos.frame_number] = {
                    "visibility": 1,
                    "x": max(0.0, min(gt_pos.x, 1.0)),
                    "y": max(0.0, min(gt_pos.y, 1.0)),
                    "source": "gold",
                }
                max_frame = max(max_frame, gt_pos.frame_number)
                gold_count += 1

        # --- Write CSV ---
        csv_path = output_dir / f"{rally_id}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_num", "visible", "x", "y"])
            for frame in range(max_frame + 1):
                data = frame_data.get(frame, {"visibility": 0, "x": 0, "y": 0})
                writer.writerow([frame, data["visibility"], data["x"], data["y"]])

        # Count by source
        source_counts = {"tracknet": 0, "vballnet": 0, "gold": 0}
        for data in frame_data.values():
            src = data.get("source", "unknown")
            if src in source_counts:
                source_counts[src] += 1

        visible = sum(1 for d in frame_data.values() if d.get("visibility") == 1)
        total = max_frame + 1

        console.print(
            f"  TrackNet: {len(tracknet_preds)} raw → {tracknet_count} kept "
            f"(filtered {tracknet_filtered} below conf={min_confidence})"
        )
        console.print(
            f"  Labels: {visible}/{total} frames visible "
            f"(tracknet={source_counts['tracknet']}, "
            f"vballnet={source_counts['vballnet']}, "
            f"gold={source_counts['gold']})"
        )
        console.print(f"  Inference: {inference_time:.1f}s → {csv_path.name}")

        all_stats[rally_id] = {
            "total_frames": total,
            "visible_frames": visible,
            "tracknet_labels": source_counts["tracknet"],
            "vballnet_labels": source_counts["vballnet"],
            "gold_labels": source_counts["gold"],
        }

    # Summary
    console.print("\n[bold]== Teacher-Student Label Summary ==[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Rally")
    table.add_column("Total", justify="right")
    table.add_column("Visible", justify="right")
    table.add_column("TrackNet", justify="right")
    table.add_column("VballNet", justify="right")
    table.add_column("Gold", justify="right")

    totals = {"total": 0, "visible": 0, "tracknet": 0, "vballnet": 0, "gold": 0}
    for rid, s in all_stats.items():
        table.add_row(
            rid[:8],
            str(s["total_frames"]),
            str(s["visible_frames"]),
            str(s["tracknet_labels"]),
            str(s["vballnet_labels"]),
            str(s["gold_labels"]),
        )
        totals["total"] += s["total_frames"]
        totals["visible"] += s["visible_frames"]
        totals["tracknet"] += s["tracknet_labels"]
        totals["vballnet"] += s["vballnet_labels"]
        totals["gold"] += s["gold_labels"]

    table.add_row(
        "[bold]Total",
        f"[bold]{totals['total']}",
        f"[bold]{totals['visible']}",
        f"[bold]{totals['tracknet']}",
        f"[bold]{totals['vballnet']}",
        f"[bold]{totals['gold']}",
    )
    console.print(table)

    console.print(f"\nLabels written to: {output_dir}")
    console.print("Next steps:")
    console.print(f"  1. uv run rallycut train tracknet-modal --upload --data-dir {output_dir}")
    console.print("  2. uv run rallycut train tracknet-modal --epochs 30 --fresh")

    return all_stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="TrackNet teacher-student bootstrapping")
    parser.add_argument("--model", default="last", help="Teacher model: best or last")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/pseudo_labels_r1"))
    parser.add_argument("--min-conf", type=float, default=0.5,
                        help="Min heatmap confidence for TrackNet labels (higher = more precise)")
    args = parser.parse_args()

    generate_teacher_labels(
        model_name=args.model,
        output_dir=args.output_dir,
        min_confidence=args.min_conf,
    )
