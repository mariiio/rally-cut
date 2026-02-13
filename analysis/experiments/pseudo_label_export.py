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
    # TrackNetV2 uses normalized coordinates (0-1) in its CSV labels
    # and frame images resized to 512x288 for training
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

            # Store normalized coordinates (0-1) — TrackNetV2 expects this
            x_norm = max(0.0, min(bp.x, 1.0))
            y_norm = max(0.0, min(bp.y, 1.0))

            frame_data[frame] = {"visibility": 1, "x": x_norm, "y": y_norm}
            stats.visible_frames += 1

        # Include gold GT labels (overwrite pseudo-labels where available)
        if include_gold and rally.ground_truth:
            gt_ball = [
                p for p in rally.ground_truth.positions if p.label == "ball"
            ]
            for gt_pos in gt_ball:
                x_norm = max(0.0, min(gt_pos.x, 1.0))
                y_norm = max(0.0, min(gt_pos.y, 1.0))

                frame_data[gt_pos.frame_number] = {
                    "visibility": 1,
                    "x": x_norm,
                    "y": y_norm,
                    "is_gold": True,
                }
                stats.gold_label_frames += 1

        # Write TrackNetV2 CSV (frame_num, visible, x, y — normalized coords)
        csv_path = output_dir / f"{rally.rally_id}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_num", "visible", "x", "y"])

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
            x_norm = max(0.0, min(gt_pos.x, 1.0))
            y_norm = max(0.0, min(gt_pos.y, 1.0))
            frame_data[gt_pos.frame_number] = {"visibility": 1, "x": x_norm, "y": y_norm}
            stats.gold_label_frames += 1
            stats.visible_frames += 1

        # Write CSV
        csv_path = output_dir / f"{rally.rally_id}_gold.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_num", "visible", "x", "y"])

            for frame in range(max_frame + 1):
                data = frame_data.get(frame, {"visibility": 0, "x": 0, "y": 0})
                writer.writerow([frame, data["visibility"], data["x"], data["y"]])
                stats.total_frames += 1

        logger.info(f"Exported gold GT: {rally.rally_id} ({len(gt_ball)} frames)")

    return stats


def extract_frames(
    output_dir: Path,
    config: PseudoLabelConfig | None = None,
    video_ids: list[str] | None = None,
) -> int:
    """Extract video frames as JPEGs for TrackNetV2 training.

    Creates directory structure:
        output_dir/images/{rally_id}/0.jpg, 1.jpg, ...

    Args:
        output_dir: Root output directory.
        config: Export configuration (uses tracknet_width/height for resize).
        video_ids: Specific video IDs to process.

    Returns:
        Number of frames extracted.
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python required for frame extraction: pip install opencv-python")
        return 0

    from rallycut.evaluation.tracking.db import get_video_path, load_labeled_rallies

    cfg = config or PseudoLabelConfig()
    images_dir = output_dir / "images"

    rallies = load_labeled_rallies(ball_gt_only=True)
    if video_ids:
        rallies = [r for r in rallies if r.video_id in video_ids]

    total_frames = 0

    for rally in rallies:
        video_path = get_video_path(rally.video_id)
        if video_path is None:
            logger.warning(f"Video not found for rally {rally.rally_id[:8]}")
            continue

        rally_dir = images_dir / rally.rally_id
        rally_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(rally.start_ms / 1000.0 * fps)
        end_frame = int(rally.end_ms / 1000.0 * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = 0
        while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to TrackNet dimensions
            resized = cv2.resize(
                frame, (cfg.tracknet_width, cfg.tracknet_height)
            )

            frame_path = rally_dir / f"{frame_idx}.jpg"
            cv2.imwrite(str(frame_path), resized)
            frame_idx += 1

        cap.release()
        total_frames += frame_idx
        logger.info(
            f"Extracted {frame_idx} frames for rally {rally.rally_id[:8]}"
        )

    return total_frames


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export pseudo-labels for TrackNet training")
    parser.add_argument("--output-dir", type=Path, default=Path("pseudo_labels"))
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-motion-energy", type=float, default=0.02)
    parser.add_argument("--gold-only", action="store_true", help="Export only gold GT labels")
    parser.add_argument(
        "--extract-frames", action="store_true",
        help="Extract video frames as JPEGs for training",
    )
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

    if args.extract_frames:
        num_frames = extract_frames(args.output_dir, config)
        print(f"\nExtracted {num_frames} frames")
