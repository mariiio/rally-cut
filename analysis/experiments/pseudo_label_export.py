"""Export VballNet detections as pseudo-labels for TrackNet training.

Converts cached VballNet ball positions into TrackNet CSV format,
applying confidence and motion energy filters to keep only high-quality
pseudo-labels. Also exports gold GT labels in the same format.

Output format (TrackNet CSV):
    Frame,Visibility,X,Y
    0,1,423,287       # visible ball at (423, 287)
    1,0,0,0           # ball not visible / out of frame
    2,2,0,0           # ball occluded

Usage:
    python -m experiments.pseudo_label_export \
        --output-dir pseudo_labels/ \
        --min-confidence 0.3 \
        --min-motion-energy 0.02
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PseudoLabelConfig:
    """Configuration for pseudo-label export."""

    min_confidence: float = 0.3
    min_motion_energy: float = 0.02
    require_post_filter_pass: bool = True
    video_width: int = 1920
    video_height: int = 1080
    # TrackNet uses pixel coordinates
    tracknet_width: int = 512
    tracknet_height: int = 288


@dataclass
class ExportStats:
    """Statistics from pseudo-label export."""

    total_frames: int = 0
    visible_frames: int = 0
    filtered_low_confidence: int = 0
    filtered_low_motion: int = 0
    filtered_post_filter: int = 0
    gold_label_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "totalFrames": self.total_frames,
            "visibleFrames": self.visible_frames,
            "filteredLowConfidence": self.filtered_low_confidence,
            "filteredLowMotion": self.filtered_low_motion,
            "filteredPostFilter": self.filtered_post_filter,
            "goldLabelFrames": self.gold_label_frames,
        }


def export_pseudo_labels(
    output_dir: Path,
    config: PseudoLabelConfig | None = None,
    include_gold: bool = True,
    video_ids: list[str] | None = None,
) -> ExportStats:
    """Export VballNet detections as TrackNet training data.

    Reads from BallRawCache (raw VballNet positions with motion_energy),
    applies quality filters, and writes TrackNet CSV files (one per rally).
    Gold GT labels are included where available and overwrite pseudo-labels.

    Args:
        output_dir: Directory to write CSV files.
        config: Export configuration.
        include_gold: Whether to include gold GT labels.
        video_ids: Specific video IDs to export (default: all with ball GT).

    Returns:
        ExportStats with counts of exported frames.
    """
    from rallycut.evaluation.tracking.ball_grid_search import BallRawCache
    from rallycut.evaluation.tracking.db import load_labeled_rallies

    cfg = config or PseudoLabelConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ExportStats()
    ball_cache = BallRawCache()

    # Load rallies for GT labels and rally metadata
    rallies = load_labeled_rallies(ball_gt_only=True)
    if video_ids:
        rallies = [r for r in rallies if r.video_id in video_ids]

    for rally in rallies:
        # Read raw VballNet positions from cache (before filtering)
        cached = ball_cache.get(rally.rally_id)
        if cached is None:
            logger.warning(
                f"No raw ball cache for rally {rally.rally_id[:8]}. "
                "Run `evaluate-tracking tune-ball-filter --all --cache-only` first."
            )
            continue

        ball_positions = cached.raw_ball_positions
        if not ball_positions:
            logger.warning(f"Empty raw ball cache for rally {rally.rally_id[:8]}")
            continue

        # Build frame -> position mapping from raw VballNet output
        frame_data: dict[int, dict[str, Any]] = {}
        max_frame = 0

        for bp in ball_positions:
            frame = bp.frame_number
            max_frame = max(max_frame, frame)

            # Skip VballNet zero-confidence placeholders (0.5, 0.5 at conf=0.0)
            if bp.confidence < cfg.min_confidence:
                stats.filtered_low_confidence += 1
                continue

            # Filter stationary false positives (low motion energy = player position)
            if bp.motion_energy < cfg.min_motion_energy:
                stats.filtered_low_motion += 1
                continue

            # Convert normalized coords to TrackNet pixel coords
            px = int(bp.x * cfg.tracknet_width)
            py = int(bp.y * cfg.tracknet_height)
            px = max(0, min(px, cfg.tracknet_width - 1))
            py = max(0, min(py, cfg.tracknet_height - 1))

            frame_data[frame] = {"visibility": 1, "x": px, "y": py}
            stats.visible_frames += 1

        # Include gold GT labels (overwrite pseudo-labels where available)
        if include_gold and rally.ground_truth:
            gt_ball = [
                p for p in rally.ground_truth.positions if p.label == "ball"
            ]
            for gt_pos in gt_ball:
                px = int(gt_pos.x * cfg.tracknet_width)
                py = int(gt_pos.y * cfg.tracknet_height)
                px = max(0, min(px, cfg.tracknet_width - 1))
                py = max(0, min(py, cfg.tracknet_height - 1))

                frame_data[gt_pos.frame_number] = {
                    "visibility": 1,
                    "x": px,
                    "y": py,
                    "is_gold": True,
                }
                stats.gold_label_frames += 1

        # Write TrackNet CSV
        csv_path = output_dir / f"{rally.rally_id}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Visibility", "X", "Y"])

            for frame in range(max_frame + 1):
                data = frame_data.get(frame, {"visibility": 0, "x": 0, "y": 0})
                writer.writerow([
                    frame,
                    data["visibility"],
                    data["x"],
                    data["y"],
                ])
                stats.total_frames += 1

        visible_count = sum(
            1 for d in frame_data.values() if d.get("visibility") == 1
        )
        logger.info(
            f"Exported {rally.rally_id[:8]}: "
            f"{visible_count} visible / {max_frame + 1} total frames → {csv_path}"
        )

    logger.info(
        f"Export complete: {stats.visible_frames} pseudo-labels, "
        f"{stats.gold_label_frames} gold labels, "
        f"{stats.total_frames} total frames"
    )

    return stats


def export_gold_only(
    output_dir: Path,
    config: PseudoLabelConfig | None = None,
    video_ids: list[str] | None = None,
) -> ExportStats:
    """Export only gold GT labels in TrackNet CSV format.

    Useful for fine-tuning evaluation or as a validation set.

    Args:
        output_dir: Directory to write CSV files.
        config: Export configuration.
        video_ids: Specific video IDs to export.

    Returns:
        ExportStats.
    """
    from rallycut.evaluation.tracking.db import load_labeled_rallies

    cfg = config or PseudoLabelConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ExportStats()

    rallies = load_labeled_rallies(ball_gt_only=True)
    if video_ids:
        rallies = [r for r in rallies if r.video_id in video_ids]

    for rally in rallies:
        if not rally.ground_truth:
            continue

        gt_ball = [p for p in rally.ground_truth.positions if p.label == "ball"]
        if not gt_ball:
            continue

        max_frame = max(p.frame_number for p in gt_ball)

        # Build frame -> position mapping
        frame_data: dict[int, dict[str, Any]] = {}
        for gt_pos in gt_ball:
            px = int(gt_pos.x * cfg.tracknet_width)
            py = int(gt_pos.y * cfg.tracknet_height)
            px = max(0, min(px, cfg.tracknet_width - 1))
            py = max(0, min(py, cfg.tracknet_height - 1))
            frame_data[gt_pos.frame_number] = {"visibility": 1, "x": px, "y": py}
            stats.gold_label_frames += 1
            stats.visible_frames += 1

        # Write CSV
        csv_path = output_dir / f"{rally.rally_id}_gold.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Visibility", "X", "Y"])

            for frame in range(max_frame + 1):
                data = frame_data.get(frame, {"visibility": 0, "x": 0, "y": 0})
                writer.writerow([frame, data["visibility"], data["x"], data["y"]])
                stats.total_frames += 1

        logger.info(f"Exported gold GT: {rally.rally_id} ({len(gt_ball)} frames)")

    return stats


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export pseudo-labels for TrackNet training")
    parser.add_argument("--output-dir", type=Path, default=Path("pseudo_labels"))
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-motion-energy", type=float, default=0.02)
    parser.add_argument("--gold-only", action="store_true", help="Export only gold GT labels")
    args = parser.parse_args()

    config = PseudoLabelConfig(
        min_confidence=args.min_confidence,
        min_motion_energy=args.min_motion_energy,
    )

    if args.gold_only:
        stats = export_gold_only(args.output_dir, config)
    else:
        stats = export_pseudo_labels(args.output_dir, config)

    print(f"\nExport stats: {stats.to_dict()}")
