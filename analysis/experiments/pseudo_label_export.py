"""Export ball detections as pseudo-labels for WASB fine-tuning.

Converts cached ball positions (VballNet or ensemble) into CSV format,
applying confidence and motion energy filters to keep only high-quality
pseudo-labels. Gold GT labels override where available.

Supports two cache sources:
  --cache-type vballnet  (default) VballNet-only raw positions
  --cache-type ensemble  WASB+VballNet ensemble positions (higher quality)

Use --all-tracked to export all rallies with player tracks (not just
the 9 with ball GT). Ensemble pseudo-labels are the key input for
WASB fine-tuning on beach volleyball.

Output format (CSV):
    Frame,Visibility,X,Y
    0,1,0.423,0.287   # visible ball at normalized (0.423, 0.287)
    1,0,0,0           # ball not visible / out of frame

Usage:
    python -m experiments.pseudo_label_export \
        --output-dir experiments/wasb_pseudo_labels/ \
        --cache-type ensemble \
        --all-tracked \
        --extract-frames
"""

from __future__ import annotations

import csv
import json
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
    # Output frame dimensions for extracted training images
    frame_width: int = 512
    frame_height: int = 288


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


ENSEMBLE_CACHE_DIR = Path.home() / ".cache" / "rallycut" / "ensemble_grid_search"


def export_pseudo_labels(
    output_dir: Path,
    config: PseudoLabelConfig | None = None,
    include_gold: bool = True,
    video_ids: list[str] | None = None,
    cache_type: str = "vballnet",
    all_tracked: bool = False,
    all_rallies: bool = False,
) -> ExportStats:
    """Export ball detections as pseudo-label training data.

    Reads from BallRawCache (raw positions with motion_energy),
    applies quality filters, and writes CSV files (one per rally).
    Gold GT labels are included where available and overwrite pseudo-labels.

    Args:
        output_dir: Directory to write CSV files.
        config: Export configuration.
        include_gold: Whether to include gold GT labels.
        video_ids: Specific video IDs to export (default: all with ball GT).
        cache_type: "vballnet" (default) or "ensemble" for ensemble cache.
        all_tracked: Load all rallies with player tracks, not just ball GT.

    Returns:
        ExportStats with counts of exported frames.
    """
    from rallycut.evaluation.tracking.ball_grid_search import BallRawCache
    from rallycut.evaluation.tracking.db import (
        load_all_rallies as _load_all_rallies,
        load_all_tracked_rallies,
        load_labeled_rallies,
    )

    cfg = config or PseudoLabelConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ExportStats()
    gold_rally_ids: list[str] = []

    # Select cache based on cache_type
    if cache_type == "ensemble":
        ball_cache = BallRawCache(cache_dir=ENSEMBLE_CACHE_DIR)
        logger.info(f"Using ensemble cache: {ENSEMBLE_CACHE_DIR}")
    else:
        ball_cache = BallRawCache()

    # Load rallies — all rallies, all tracked, or just ball GT
    if all_rallies:
        loaded = _load_all_rallies()
        if video_ids:
            loaded = [r for r in loaded if r.video_id in video_ids]

        gt_rallies = load_labeled_rallies()
        gt_map = {r.rally_id: r for r in gt_rallies}

        rally_list: list[tuple[str, str, int, int, Any]] = [
            (r.rally_id, r.video_id, r.start_ms, r.end_ms, gt_map.get(r.rally_id))
            for r in loaded
        ]
        logger.info(
            f"Loading {len(rally_list)} rallies (all) "
            f"({len(gt_map)} have ball GT)"
        )
    elif all_tracked:
        tracked = load_all_tracked_rallies()
        if video_ids:
            tracked = [r for r in tracked if r.video_id in video_ids]

        gt_rallies = load_labeled_rallies()
        gt_map = {r.rally_id: r for r in gt_rallies}

        rally_list: list[tuple[str, str, int, int, Any]] = [
            (r.rally_id, r.video_id, r.start_ms, r.end_ms, gt_map.get(r.rally_id))
            for r in tracked
        ]
        logger.info(
            f"Loading {len(rally_list)} tracked rallies "
            f"({len(gt_map)} have ball GT)"
        )
    else:
        gt_rallies = load_labeled_rallies()
        if video_ids:
            gt_rallies = [r for r in gt_rallies if r.video_id in video_ids]
        rally_list = [
            (r.rally_id, r.video_id, r.start_ms, r.end_ms, r)
            for r in gt_rallies
        ]

    for rally_id, video_id, start_ms, end_ms, gt_rally in rally_list:
        # Read raw ball positions from cache
        cached = ball_cache.get(rally_id)
        if cached is None:
            logger.warning(
                f"No {cache_type} cache for rally {rally_id[:8]}. "
                f"Run cache script first."
            )
            continue

        ball_positions = cached.raw_ball_positions
        if not ball_positions:
            logger.warning(f"Empty {cache_type} cache for rally {rally_id[:8]}")
            continue

        # Build frame -> position mapping from raw output
        frame_data: dict[int, dict[str, Any]] = {}
        max_frame = 0

        for bp in ball_positions:
            frame = bp.frame_number
            max_frame = max(max_frame, frame)

            # Skip zero-confidence placeholders
            if bp.confidence < cfg.min_confidence:
                stats.filtered_low_confidence += 1
                continue

            # Filter stationary false positives (low motion energy = player position)
            # For ensemble: WASB positions have motion_energy=1.0, always pass
            if bp.motion_energy < cfg.min_motion_energy:
                stats.filtered_low_motion += 1
                continue

            # Store normalized coordinates (0-1)
            x_norm = max(0.0, min(bp.x, 1.0))
            y_norm = max(0.0, min(bp.y, 1.0))

            frame_data[frame] = {"visibility": 1, "x": x_norm, "y": y_norm}
            stats.visible_frames += 1

        # Check if this rally has real ball GT (for val-split marker,
        # independent of whether we bake it into pseudo-labels)
        has_gold = False
        if gt_rally is not None and gt_rally.ground_truth:
            gt_ball = [
                p for p in gt_rally.ground_truth.positions if p.label == "ball"
            ]
            has_gold = len(gt_ball) > 0

        # Include gold GT labels (overwrite pseudo-labels where available)
        if has_gold and include_gold:
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

        # Write CSV (frame_num, visible, x, y — normalized coords)
        csv_path = output_dir / f"{rally_id}.csv"
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

        if has_gold:
            gold_rally_ids.append(rally_id)

        visible_count = sum(
            1 for d in frame_data.values() if d.get("visibility") == 1
        )
        gt_tag = " (has GT)" if gt_rally is not None else ""
        logger.info(
            f"Exported {rally_id[:8]}: "
            f"{visible_count} visible / {max_frame + 1} total frames{gt_tag} → {csv_path}"
        )

    # Write manifest so training can discover GT rallies for val split
    manifest = {
        "gold_rally_ids": gold_rally_ids,
        "cache_type": cache_type,
        "include_gold": include_gold,
        "total_rallies": len(rally_list),
        "gold_label_frames": stats.gold_label_frames,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest: {len(gold_rally_ids)} GT rallies → {manifest_path}")

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
    """Export only gold GT labels in CSV format.

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

    rallies = load_labeled_rallies()
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
    all_tracked: bool = False,
    all_rallies: bool = False,
) -> int:
    """Extract video frames as JPEGs for training.

    Creates directory structure:
        output_dir/images/{rally_id}/0.jpg, 1.jpg, ...

    Args:
        output_dir: Root output directory.
        config: Export configuration (uses frame_width/height for resize).
        video_ids: Specific video IDs to process.
        all_tracked: Extract frames for all tracked rallies, not just ball GT.

    Returns:
        Number of frames extracted.
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python required for frame extraction: pip install opencv-python")
        return 0

    from rallycut.evaluation.tracking.db import (
        get_video_path,
        load_all_rallies as _load_all_rallies,
        load_all_tracked_rallies,
        load_labeled_rallies,
    )

    cfg = config or PseudoLabelConfig()
    images_dir = output_dir / "images"

    # Load rallies
    if all_rallies:
        loaded = _load_all_rallies()
        if video_ids:
            loaded = [r for r in loaded if r.video_id in video_ids]
        rally_list = [
            (r.rally_id, r.video_id, r.start_ms, r.end_ms) for r in loaded
        ]
    elif all_tracked:
        tracked = load_all_tracked_rallies()
        if video_ids:
            tracked = [r for r in tracked if r.video_id in video_ids]
        rally_list = [
            (r.rally_id, r.video_id, r.start_ms, r.end_ms) for r in tracked
        ]
    else:
        gt_rallies = load_labeled_rallies()
        if video_ids:
            gt_rallies = [r for r in gt_rallies if r.video_id in video_ids]
        rally_list = [
            (r.rally_id, r.video_id, r.start_ms, r.end_ms) for r in gt_rallies
        ]

    total_frames = 0

    # Group by video to avoid re-downloading
    by_video: dict[str, list[tuple[str, int, int]]] = {}
    for rally_id, video_id, start_ms, end_ms in rally_list:
        by_video.setdefault(video_id, []).append((rally_id, start_ms, end_ms))

    for video_id, video_rallies in by_video.items():
        video_path = get_video_path(video_id)
        if video_path is None:
            logger.warning(f"Video not found: {video_id[:8]}")
            continue

        for rally_id, start_ms, end_ms in video_rallies:
            rally_dir = images_dir / rally_id
            if rally_dir.exists() and any(rally_dir.iterdir()):
                logger.info(f"Skipping {rally_id[:8]}: frames already extracted")
                # Count existing frames
                total_frames += len(list(rally_dir.glob("*.jpg")))
                continue

            rally_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Cannot open video: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(start_ms / 1000.0 * fps)
            end_frame = int(end_ms / 1000.0 * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_idx = 0
            while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize to training dimensions
                resized = cv2.resize(
                    frame, (cfg.frame_width, cfg.frame_height)
                )

                frame_path = rally_dir / f"{frame_idx}.jpg"
                cv2.imwrite(str(frame_path), resized)
                frame_idx += 1

            cap.release()
            total_frames += frame_idx
            logger.info(
                f"Extracted {frame_idx} frames for rally {rally_id[:8]}"
            )

    return total_frames


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export pseudo-labels for ball model training")
    parser.add_argument("--output-dir", type=Path, default=Path("pseudo_labels"))
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-motion-energy", type=float, default=0.02)
    parser.add_argument("--gold-only", action="store_true", help="Export only gold GT labels")
    parser.add_argument(
        "--cache-type",
        choices=["vballnet", "ensemble"],
        default="vballnet",
        help="Ball position cache to use (default: vballnet)",
    )
    parser.add_argument(
        "--all-tracked",
        action="store_true",
        help="Export all tracked rallies, not just ball GT rallies",
    )
    parser.add_argument(
        "--all-rallies",
        action="store_true",
        help="Export ALL rallies in DB (no player tracks required)",
    )
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
        stats = export_pseudo_labels(
            args.output_dir,
            config,
            cache_type=args.cache_type,
            all_tracked=args.all_tracked,
            all_rallies=args.all_rallies,
        )

    # Print detailed export summary
    csv_count = len(list(args.output_dir.glob("*.csv")))
    vis_pct = 100 * stats.visible_frames / max(stats.total_frames, 1)
    gold_pct = 100 * stats.gold_label_frames / max(stats.visible_frames, 1)
    print(f"\n{'=' * 50}")
    print("Export Summary")
    print(f"{'=' * 50}")
    print(f"  Rallies exported:  {csv_count}")
    print(f"  Total frames:      {stats.total_frames:,}")
    print(f"  Visible (ball):    {stats.visible_frames:,} ({vis_pct:.1f}%)")
    print(f"  Gold GT labels:    {stats.gold_label_frames:,} ({gold_pct:.1f}% of visible)")
    print(f"  Filtered (conf):   {stats.filtered_low_confidence:,}")
    print(f"  Filtered (motion): {stats.filtered_low_motion:,}")
    print(f"  Filtered (post):   {stats.filtered_post_filter:,}")

    if vis_pct < 20:
        print(f"\n  WARNING: Only {vis_pct:.0f}% visible — check cache quality")
    elif vis_pct > 80:
        print(f"\n  WARNING: {vis_pct:.0f}% visible is unusually high — check for false positives")

    if args.extract_frames:
        num_frames = extract_frames(
            args.output_dir, config,
            all_tracked=args.all_tracked,
            all_rallies=args.all_rallies,
        )
        # Estimate disk usage (~15KB per 512x288 JPEG)
        est_mb = num_frames * 15 / 1024
        print(f"\n  Extracted frames:  {num_frames:,} (~{est_mb:.0f} MB)")

    print()
