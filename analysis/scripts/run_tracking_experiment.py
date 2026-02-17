#!/usr/bin/env python3
"""Run player tracking with custom settings on all labeled rallies and evaluate.

Loads ground truth from the database, runs tracking with the current codebase's
settings (config files, model choices), and evaluates against GT — all without
modifying the DB.

Usage:
    # Run with current settings (baseline)
    uv run python scripts/run_tracking_experiment.py -o baseline.json

    # Run with specific options
    uv run python scripts/run_tracking_experiment.py --yolo-model yolo11n -o exp_b.json

    # Quick test on single rally
    uv run python scripts/run_tracking_experiment.py --rally-id 1bfcbc4f -o quick.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add analysis root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Run tracking experiment on all labeled rallies."""
    from rallycut.evaluation.tracking.db import get_video_path, load_labeled_rallies
    from rallycut.evaluation.tracking.metrics import aggregate_results, evaluate_rally
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import PlayerTracker

    # Load labeled rallies from DB
    print("Loading labeled rallies from database...")
    rallies = load_labeled_rallies()

    # Filter by rally_id prefix if specified
    if args.rally_id:
        prefix = args.rally_id
        rallies = [r for r in rallies if r.rally_id.startswith(prefix)]

    print(f"  Found {len(rallies)} rally(s) with ground truth\n")

    if not rallies:
        print("No rallies found!")
        return {}

    # Resolve court ROI mode
    court_roi_mode = getattr(args, "court_roi", "none")

    # Create tracker with experiment settings
    tracker_kwargs: dict[str, Any] = {
        "yolo_model": args.yolo_model,
        "confidence": args.confidence,
        "tracker": args.tracker,
        "preprocessing": args.preprocessing,
    }

    if court_roi_mode == "default":
        from rallycut.tracking.player_tracker import DEFAULT_COURT_ROI

        tracker_kwargs["court_roi"] = DEFAULT_COURT_ROI

    # ReID overrides
    if args.appearance_thresh is not None:
        tracker_kwargs["appearance_thresh"] = args.appearance_thresh
    if args.with_reid:
        tracker_kwargs["with_reid"] = True

    print(f"Tracker config: {tracker_kwargs}")
    if court_roi_mode != "none":
        print(f"Court ROI: {court_roi_mode}")
    tracker = PlayerTracker(**tracker_kwargs)

    # Track each rally and evaluate
    all_results: list[Any] = []
    per_rally_metrics: dict[str, dict[str, Any]] = {}
    total_start = time.time()

    for i, rally in enumerate(rallies):
        rally_id_short = rally.rally_id[:8]
        print(f"\n[{i+1}/{len(rallies)}] Rally {rally_id_short}... ", end="", flush=True)

        # Get video path
        video_path = get_video_path(rally.video_id)
        if video_path is None:
            print("SKIP (video not found)")
            continue

        # Run tracking
        track_start = time.time()
        try:
            # For adaptive ROI, run ball tracking first
            ball_positions = None
            if court_roi_mode == "adaptive":
                from rallycut.tracking import create_ball_tracker
                from rallycut.tracking.player_tracker import (
                    DEFAULT_COURT_ROI,
                    compute_court_roi_from_ball,
                )

                ball_tracker = create_ball_tracker()
                ball_result = ball_tracker.track_video(
                    video_path, start_ms=rally.start_ms, end_ms=rally.end_ms,
                )
                ball_positions = ball_result.positions
                adaptive_roi, quality_msg = compute_court_roi_from_ball(ball_positions)
                roi = adaptive_roi if adaptive_roi is not None else DEFAULT_COURT_ROI
                tracker.court_roi = roi
                xs = [p[0] for p in roi]
                ys = [p[1] for p in roi]
                print(
                    f"ROI x:{min(xs):.2f}-{max(xs):.2f} y:{min(ys):.2f}-{max(ys):.2f} ",
                    end="", flush=True,
                )

            result = tracker.track_video(
                video_path=video_path,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
                stride=1,
                filter_enabled=True,
                filter_config=PlayerFilterConfig(),
                ball_positions=ball_positions,
            )

            # Offset frame numbers to rally-relative (0-based) to match GT
            if result.positions:
                first_frame = min(p.frame_number for p in result.positions)
                for p in result.positions:
                    p.frame_number -= first_frame

        except Exception as e:
            print(f"ERROR ({e})")
            continue

        track_time = time.time() - track_start
        print(f"{len(result.positions)} pos, {track_time:.1f}s ", end="", flush=True)

        # Evaluate against ground truth
        try:
            eval_result = evaluate_rally(
                rally_id=rally.rally_id,
                ground_truth=rally.ground_truth,
                predictions=result,
                iou_threshold=args.iou_threshold,
                video_width=rally.video_width,
                video_height=rally.video_height,
            )

            mot = eval_result.aggregate
            hota = eval_result.hota_metrics
            tq = eval_result.track_quality

            hota_val = hota.hota if hota else 0.0
            deta_val = hota.deta if hota else 0.0
            assa_val = hota.assa if hota else 0.0
            idsw = mot.num_id_switches
            frag = tq.fragmentation if tq else 0
            mt = tq.mostly_tracked if tq else 0
            mt_total = tq.gt_track_count if tq else 0

            print(
                f"| MOTA={mot.mota:.1%} HOTA={hota_val:.1%} "
                f"F1={mot.f1:.1%} IDsw={idsw}"
            )

            per_rally_metrics[rally.rally_id] = {
                "rally_id": rally.rally_id,
                "video_id": rally.video_id,
                "mota": mot.mota,
                "hota": hota_val,
                "deta": deta_val,
                "assa": assa_val,
                "f1": mot.f1,
                "precision": mot.precision,
                "recall": mot.recall,
                "id_switches": idsw,
                "fragmentation": frag,
                "mostly_tracked": mt,
                "mostly_tracked_total": mt_total,
                "tracking_time_s": track_time,
                "num_predictions": len(result.positions),
                "unique_tracks": result.unique_track_count,
            }

            all_results.append(eval_result)

        except Exception as e:
            print(f"EVAL ERROR ({e})")
            continue

    total_time = time.time() - total_start

    # Aggregate
    if all_results:
        agg = aggregate_results(all_results)

        # Aggregate HOTA and track quality from per-rally metrics
        agg_hota = sum(m["hota"] for m in per_rally_metrics.values()) / len(per_rally_metrics)
        agg_deta = sum(m["deta"] for m in per_rally_metrics.values()) / len(per_rally_metrics)
        agg_assa = sum(m["assa"] for m in per_rally_metrics.values()) / len(per_rally_metrics)
        agg_frag = sum(m["fragmentation"] for m in per_rally_metrics.values())
        agg_mt = sum(m["mostly_tracked"] for m in per_rally_metrics.values())
        agg_mt_total = sum(m["mostly_tracked_total"] for m in per_rally_metrics.values())

        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(f"  HOTA:     {agg_hota:.1%}")
        print(f"  MOTA:     {agg.mota:.1%}")
        print(f"  F1:       {agg.f1:.1%}")
        print(f"  DetA:     {agg_deta:.1%}")
        print(f"  AssA:     {agg_assa:.1%}")
        print(f"  ID Sw:    {agg.num_id_switches}")
        print(f"  Frag:     {agg_frag}")
        print(f"  MT:       {agg_mt}/{agg_mt_total}")
        print(f"  Prec:     {agg.precision:.1%}")
        print(f"  Recall:   {agg.recall:.1%}")
        print(f"  Time:     {total_time:.1f}s total")

        output = {
            "experiment": {
                "tracker_config": tracker_kwargs,
                "iou_threshold": args.iou_threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_time_s": total_time,
            },
            "aggregate": {
                "hota": agg_hota,
                "mota": agg.mota,
                "deta": agg_deta,
                "assa": agg_assa,
                "f1": agg.f1,
                "precision": agg.precision,
                "recall": agg.recall,
                "id_switches": agg.num_id_switches,
                "fragmentation": agg_frag,
                "mostly_tracked": agg_mt,
                "mostly_tracked_total": agg_mt_total,
            },
            "per_rally": per_rally_metrics,
        }

        return output

    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tracking experiment on labeled rallies")

    # Output
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file")
    parser.add_argument("--rally-id", type=str, help="Test single rally")

    # Tracker settings
    parser.add_argument("--yolo-model", default="yolov8n", help="YOLO model (default: yolov8n)")
    parser.add_argument("--tracker", default="botsort", help="Tracker (botsort/bytetrack)")
    parser.add_argument("--confidence", type=float, default=0.15, help="Detection confidence")
    parser.add_argument("--preprocessing", default="none", help="Preprocessing (none/clahe)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Eval IoU threshold")

    # Court ROI
    parser.add_argument(
        "--court-roi", default="none", choices=["none", "default", "adaptive"],
        help="Court ROI mode: none (no masking), default (fixed rectangle), adaptive (from ball)",
    )

    # Optional overrides
    parser.add_argument("--with-reid", action="store_true", help="Enable BoT-SORT ReID")
    parser.add_argument("--appearance-thresh", type=float, help="ReID appearance threshold")

    args = parser.parse_args()

    output = run_experiment(args)

    if args.output and output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
