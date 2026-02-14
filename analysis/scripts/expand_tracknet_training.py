"""Expand TrackNet training data to all rallies in the database.

Orchestrates the full expansion pipeline:
1. Load all rallies from DB (not just GT-labeled ones)
2. For each rally: use cached raw positions, DB positions, or run VballNet inference
3. Apply full BallFilter pipeline (segment pruning, oscillation/blip/ghost removal)
4. Override with gold GT labels where available
5. Write TrackNet CSVs with source labels + extract 512x288 frames

CSV format: frame_num, visible, x, y, source
  source: 0=not visible, 1=filtered (BallFilter output), 2=gold (GT label)
  Source weights used in training: gold=1.0, filtered=0.7, none=0.3

Usage:
    cd analysis

    # Quick: 5 rallies per video (default)
    uv run python scripts/expand_tracknet_training.py --extract-frames

    # All rallies
    uv run python scripts/expand_tracknet_training.py --all --extract-frames

    # Specific video
    uv run python scripts/expand_tracknet_training.py --video a5866029-... --extract-frames

    # Custom output dir
    uv run python scripts/expand_tracknet_training.py --output-dir training_data_r2/
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.tracking.ball_grid_search import (
    BallRawCache,
    CachedBallData,
    apply_ball_filter_config,
)
from rallycut.evaluation.tracking.db import (
    RallyMetadata,
    get_ball_positions_from_db,
    get_video_path,
    load_all_rallies,
    load_labeled_rallies,
)
from rallycut.tracking.ball_filter import BallFilterConfig
from rallycut.tracking.ball_tracker import BallPosition, BallTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
console = Console()

# TrackNet input dimensions
TRACKNET_WIDTH = 512
TRACKNET_HEIGHT = 288

# Label source codes (written to CSV, used for loss weighting in training)
SOURCE_NONE = 0  # Not visible / no detection
SOURCE_FILTERED = 1  # BallFilter pipeline output (high-quality pseudo-label)
SOURCE_GOLD = 2  # Human-annotated ground truth


def _get_raw_positions(
    rally: RallyMetadata,
    cache: BallRawCache,
    tracker: BallTracker | None,
    video_path: Path | None,
) -> list[BallPosition] | None:
    """Get raw ball positions for a rally, trying cache -> DB -> inference.

    Returns None if no positions could be obtained.
    """
    # 1. Try BallRawCache (fastest)
    cached = cache.get(rally.rally_id)
    if cached is not None:
        logger.info(
            f"  {rally.rally_id[:8]}: cached raw positions "
            f"({len(cached.raw_ball_positions)} frames)"
        )
        return cached.raw_ball_positions

    # 2. Try DB ball_positions_json
    db_positions = get_ball_positions_from_db(rally.rally_id)
    if db_positions:
        logger.info(
            f"  {rally.rally_id[:8]}: DB ball positions ({len(db_positions)} frames)"
        )
        # Convert to rally-relative frame numbers and cache
        first_frame = min(p.frame_number for p in db_positions)
        relative = [
            BallPosition(
                frame_number=p.frame_number - first_frame,
                x=p.x,
                y=p.y,
                confidence=p.confidence,
                motion_energy=p.motion_energy,
            )
            for p in db_positions
        ]
        cache.put(CachedBallData(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            raw_ball_positions=relative,
            video_fps=rally.video_fps,
            frame_count=len(relative),
            video_width=rally.video_width,
            video_height=rally.video_height,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
        ))
        return relative

    # 3. Run VballNet inference (slowest)
    if tracker is None or video_path is None:
        return None

    logger.info(
        f"  {rally.rally_id[:8]}: VballNet inference "
        f"({rally.start_ms}ms-{rally.end_ms}ms)..."
    )
    result = tracker.track_video(
        video_path,
        start_ms=rally.start_ms,
        end_ms=rally.end_ms,
        enable_filtering=False,
    )

    first_frame = min((p.frame_number for p in result.positions), default=0)
    raw = [
        BallPosition(
            frame_number=p.frame_number - first_frame,
            x=p.x,
            y=p.y,
            confidence=p.confidence,
            motion_energy=p.motion_energy,
        )
        for p in result.positions
    ]

    cache.put(CachedBallData(
        rally_id=rally.rally_id,
        video_id=rally.video_id,
        raw_ball_positions=raw,
        video_fps=rally.video_fps,
        frame_count=result.frame_count,
        video_width=rally.video_width,
        video_height=rally.video_height,
        start_ms=rally.start_ms,
        end_ms=rally.end_ms,
    ))
    logger.info(
        f"    {len(raw)} raw detections "
        f"(detection rate: {result.detection_rate * 100:.1f}%)"
    )
    return raw


def _write_rally_csv(
    rally_id: str,
    raw_positions: list[BallPosition],
    filter_config: BallFilterConfig,
    gold_gt: dict[int, tuple[float, float]] | None,
    output_dir: Path,
) -> dict[str, int]:
    """Write a TrackNet CSV with full BallFilter pipeline + gold GT override.

    Applies the full filter pipeline (segment pruning, oscillation removal,
    blip removal, exit ghost removal) to produce high-quality pseudo-labels.
    Gold GT labels override filter output at highest priority.

    CSV columns: frame_num, visible, x, y, source
      source: 0=none, 1=filtered, 2=gold

    Returns stats dict with frame counts.
    """
    stats = {"total": 0, "filtered": 0, "gold": 0}

    # Apply full BallFilter pipeline to raw positions
    filtered_positions = apply_ball_filter_config(raw_positions, filter_config)

    # Build frame -> (x, y, source) from filtered output
    frame_data: dict[int, tuple[float, float, int]] = {}
    max_frame = max((p.frame_number for p in raw_positions), default=0)

    for bp in filtered_positions:
        x = max(0.0, min(bp.x, 1.0))
        y = max(0.0, min(bp.y, 1.0))
        frame_data[bp.frame_number] = (x, y, SOURCE_FILTERED)
        stats["filtered"] += 1

    # Override with gold GT labels (highest priority)
    if gold_gt:
        for frame, (gx, gy) in gold_gt.items():
            frame_data[frame] = (gx, gy, SOURCE_GOLD)
            max_frame = max(max_frame, frame)
            stats["gold"] += 1

    stats["total"] = max_frame + 1

    csv_path = output_dir / f"{rally_id}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_num", "visible", "x", "y", "source"])
        for frame in range(max_frame + 1):
            if frame in frame_data:
                x, y, source = frame_data[frame]
                writer.writerow([frame, 1, x, y, source])
            else:
                writer.writerow([frame, 0, 0, 0, SOURCE_NONE])

    return stats


def _extract_rally_frames(
    rally: RallyMetadata,
    video_path: Path,
    output_dir: Path,
) -> int:
    """Extract video frames as JPEGs for a single rally.

    Returns number of frames extracted.
    """
    import cv2

    images_dir = output_dir / "images" / rally.rally_id
    images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(rally.start_ms / 1000.0 * fps)
    end_frame = int(rally.end_ms / 1000.0 * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = 0
    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (TRACKNET_WIDTH, TRACKNET_HEIGHT))
        cv2.imwrite(str(images_dir / f"{frame_idx}.jpg"), resized)
        frame_idx += 1

    cap.release()
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand TrackNet training data to all DB rallies"
    )
    parser.add_argument(
        "--max-per-video", type=int, default=5,
        help="Max rallies per video (default: 5, evenly spaced)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Use ALL rallies (no per-video limit)",
    )
    parser.add_argument("--video", "-v", help="Filter to specific video ID")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/tracknet_expanded"),
        help="Output directory for CSVs and images",
    )
    parser.add_argument(
        "--extract-frames", action="store_true",
        help="Also extract 512x288 video frames as JPEGs",
    )
    parser.add_argument(
        "--model", default="v2",
        help="Ball model for inference (default: v2)",
    )
    args = parser.parse_args()

    max_per_video = None if args.all else args.max_per_video

    # Load all rallies from DB
    rallies = load_all_rallies(video_id=args.video, max_per_video=max_per_video)
    if not rallies:
        console.print("[red]No rallies found in database[/red]")
        sys.exit(1)

    by_video: dict[str, list[RallyMetadata]] = {}
    for r in rallies:
        by_video.setdefault(r.video_id, []).append(r)

    console.print(
        f"Found [bold]{len(rallies)}[/bold] rallies across "
        f"[bold]{len(by_video)}[/bold] videos"
    )

    # Load GT labels for gold override
    gt_rallies = load_labeled_rallies()
    gold_gt_by_rally: dict[str, dict[int, tuple[float, float]]] = {}
    for gt_rally in gt_rallies:
        if gt_rally.ground_truth:
            ball_gt = [p for p in gt_rally.ground_truth.positions if p.label == "ball"]
            if ball_gt:
                gold_gt_by_rally[gt_rally.rally_id] = {
                    p.frame_number: (
                        max(0.0, min(p.x, 1.0)),
                        max(0.0, min(p.y, 1.0)),
                    )
                    for p in ball_gt
                }

    console.print(f"Gold GT available for {len(gold_gt_by_rally)} rallies")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = BallRawCache()
    filter_config = BallFilterConfig()  # Full pipeline: segment pruning + all filters
    tracker: BallTracker | None = None  # Lazy init

    total_stats = {
        "rallies": 0, "total_frames": 0, "filtered_frames": 0,
        "gold_frames": 0, "extracted_frames": 0, "skipped": 0,
    }
    per_rally_stats: list[tuple[str, str, dict[str, int]]] = []

    for video_id, video_rallies in by_video.items():
        console.print(
            f"\n[bold]Video {video_id[:8]}...[/bold] "
            f"({len(video_rallies)} rallies)"
        )

        # Check which rallies need inference (not in cache or DB)
        need_inference = [
            r for r in video_rallies
            if not cache.has(r.rally_id)
            and get_ball_positions_from_db(r.rally_id) is None
        ]

        video_path: Path | None = None
        if need_inference or args.extract_frames:
            video_path = get_video_path(video_id)
            if video_path is None:
                console.print(
                    f"  [yellow]Video not available, "
                    f"skipping {len(video_rallies)} rallies[/yellow]"
                )
                total_stats["skipped"] += len(video_rallies)
                continue

        if need_inference and tracker is None:
            tracker = BallTracker(model=args.model)

        for rally in video_rallies:
            raw_positions = _get_raw_positions(rally, cache, tracker, video_path)
            if raw_positions is None:
                logger.warning(
                    f"  {rally.rally_id[:8]}: no positions available, skipping"
                )
                total_stats["skipped"] += 1
                continue

            gold_gt = gold_gt_by_rally.get(rally.rally_id)
            stats = _write_rally_csv(
                rally.rally_id, raw_positions, filter_config, gold_gt, args.output_dir
            )
            per_rally_stats.append((rally.rally_id, rally.video_id, stats))
            total_stats["rallies"] += 1
            total_stats["total_frames"] += stats["total"]
            total_stats["filtered_frames"] += stats["filtered"]
            total_stats["gold_frames"] += stats["gold"]

            if args.extract_frames and video_path is not None:
                n = _extract_rally_frames(rally, video_path, args.output_dir)
                total_stats["extracted_frames"] += n

    # Summary table
    console.print("\n[bold]== Expansion Summary ==[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Rally")
    table.add_column("Video")
    table.add_column("Total", justify="right")
    table.add_column("Filtered", justify="right")
    table.add_column("Gold", justify="right")
    table.add_column("Label%", justify="right")

    for rally_id, video_id, stats in per_rally_stats:
        labeled = stats["filtered"] + stats["gold"]
        pct = labeled / max(stats["total"], 1) * 100
        table.add_row(
            rally_id[:8],
            video_id[:8],
            str(stats["total"]),
            str(stats["filtered"]),
            str(stats["gold"]),
            f"{pct:.0f}%",
        )

    if per_rally_stats:
        total_labeled = total_stats["filtered_frames"] + total_stats["gold_frames"]
        avg_pct = total_labeled / max(total_stats["total_frames"], 1) * 100
        table.add_row(
            f"[bold]Total ({total_stats['rallies']})",
            "",
            f"[bold]{total_stats['total_frames']}",
            f"[bold]{total_stats['filtered_frames']}",
            f"[bold]{total_stats['gold_frames']}",
            f"[bold]{avg_pct:.0f}%",
        )

    console.print(table)

    console.print(f"\nCSVs written to: {args.output_dir}")
    if total_stats["extracted_frames"] > 0:
        console.print(f"Frames extracted: {total_stats['extracted_frames']}")
    if total_stats["skipped"] > 0:
        console.print(
            f"[yellow]Skipped: {total_stats['skipped']} rallies "
            f"(no video/positions)[/yellow]"
        )

    console.print("\nNext steps:")
    console.print(
        f"  1. uv run rallycut train tracknet-modal --upload "
        f"--data-dir {args.output_dir}"
    )
    console.print("  2. uv run rallycut train tracknet-modal --epochs 10 --fresh")


if __name__ == "__main__":
    main()
