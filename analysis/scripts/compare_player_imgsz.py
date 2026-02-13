#!/usr/bin/env python3
"""Experiment 2: Compare player detection at different imgsz values.

Runs PlayerTracker at multiple inference resolutions on all labeled rallies
and reports distance-bucketed recall (near/mid/far).

Usage:
    # Compare baseline (640) vs high-res (1280)
    uv run python scripts/compare_player_imgsz.py

    # Test specific variants
    uv run python scripts/compare_player_imgsz.py --variants baseline yolov8n_1280

    # Quick test on single rally
    uv run python scripts/compare_player_imgsz.py --rally-id 1bfcbc4f

    # Save results
    uv run python scripts/compare_player_imgsz.py -o exp_player_hires.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add analysis root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Experiment variants from exp_player_hires.yaml
VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "yolo_model": "yolov8n",
        "imgsz": 640,
        "description": "Current default",
    },
    "yolov8n_1280": {
        "yolo_model": "yolov8n",
        "imgsz": 1280,
        "description": "Nano model at 2x resolution",
    },
    "yolo11m_1280": {
        "yolo_model": "yolo11m",
        "imgsz": 1280,
        "description": "Medium model at 2x resolution",
    },
}

# Distance buckets based on actual beach volleyball camera angles.
# Camera behind baseline: near-side players at bottom (high y), far-side at top (low y).
# Player GT Y-range is 0.34-0.80 — standard 33% buckets miss all far-court positions.
DISTANCE_BUCKETS = {
    "near": (0.58, 1.0),   # Near-side team (larger bbox, easy to detect)
    "mid": (0.48, 0.58),   # Net area / transition zone
    "far": (0.0, 0.48),    # Far-side team (smaller bbox, harder to detect)
}


@dataclass
class BucketedMetrics:
    """Recall metrics bucketed by distance from camera."""

    near_recall: float = 0.0
    near_total: int = 0
    mid_recall: float = 0.0
    mid_total: int = 0
    far_recall: float = 0.0
    far_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "near": {"recall": self.near_recall, "gt_count": self.near_total},
            "mid": {"recall": self.mid_recall, "gt_count": self.mid_total},
            "far": {"recall": self.far_recall, "gt_count": self.far_total},
        }


def compute_bucketed_recall(
    gt_positions: list[Any],
    pred_positions: list[Any],
    iou_threshold: float = 0.5,
) -> BucketedMetrics:
    """Compute recall bucketed by Y-position distance from camera.

    Args:
        gt_positions: Ground truth positions (GroundTruthPosition).
        pred_positions: Predicted positions (PlayerPosition).
        iou_threshold: IoU threshold for matching.

    Returns:
        BucketedMetrics with per-bucket recall.
    """
    bucket_hits: dict[str, int] = {"near": 0, "mid": 0, "far": 0}
    bucket_total: dict[str, int] = {"near": 0, "mid": 0, "far": 0}

    # Filter to player labels only (player, player_1, player_2, etc.)
    gt_persons = [g for g in gt_positions if g.is_player]

    # Group GT by frame
    gt_by_frame: dict[int, list[Any]] = {}
    for g in gt_persons:
        gt_by_frame.setdefault(g.frame_number, []).append(g)

    # Group predictions by frame
    pred_by_frame: dict[int, list[Any]] = {}
    for p in pred_positions:
        pred_by_frame.setdefault(p.frame_number, []).append(p)

    for frame, gt_list in gt_by_frame.items():
        preds = pred_by_frame.get(frame, [])

        for gt in gt_list:
            # Determine bucket from GT center Y (y is already center-based)
            gt_center_y = gt.y
            bucket = "mid"  # default
            for bucket_name, (y_min, y_max) in DISTANCE_BUCKETS.items():
                if y_min <= gt_center_y <= y_max:
                    bucket = bucket_name
                    break

            bucket_total[bucket] += 1

            # Check if any prediction matches this GT (IoU-based)
            matched = False
            for pred in preds:
                iou = _compute_iou(gt, pred)
                if iou >= iou_threshold:
                    matched = True
                    break

            if matched:
                bucket_hits[bucket] += 1

    return BucketedMetrics(
        near_recall=bucket_hits["near"] / bucket_total["near"] if bucket_total["near"] > 0 else 0.0,
        near_total=bucket_total["near"],
        mid_recall=bucket_hits["mid"] / bucket_total["mid"] if bucket_total["mid"] > 0 else 0.0,
        mid_total=bucket_total["mid"],
        far_recall=bucket_hits["far"] / bucket_total["far"] if bucket_total["far"] > 0 else 0.0,
        far_total=bucket_total["far"],
    )


def _compute_iou(gt: Any, pred: Any) -> float:
    """Compute IoU between a ground truth and prediction bbox."""
    # Convert normalized center-based (x, y, width, height) to corners
    gt_x1 = gt.x - gt.width / 2
    gt_y1 = gt.y - gt.height / 2
    gt_x2 = gt.x + gt.width / 2
    gt_y2 = gt.y + gt.height / 2

    pred_x1 = pred.x - pred.width / 2
    pred_y1 = pred.y - pred.height / 2
    pred_x2 = pred.x + pred.width / 2
    pred_y2 = pred.y + pred.height / 2

    # Intersection
    inter_x1 = max(gt_x1, pred_x1)
    inter_y1 = max(gt_y1, pred_y1)
    inter_x2 = min(gt_x2, pred_x2)
    inter_y2 = min(gt_y2, pred_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    union_area = gt_area + pred_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def run_variant(
    variant_name: str,
    variant_config: dict[str, Any],
    rallies: list[Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run a single variant on all rallies.

    Returns:
        Dict with aggregate and per-rally metrics.
    """
    from rallycut.evaluation.tracking.db import get_video_path
    from rallycut.evaluation.tracking.metrics import aggregate_results, evaluate_rally
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import PlayerTracker

    yolo_model = variant_config["yolo_model"]
    imgsz = variant_config["imgsz"]

    print(f"\n{'='*60}")
    print(f"  Variant: {variant_name}")
    print(f"  Model: {yolo_model}, imgsz: {imgsz}")
    print(f"{'='*60}")

    tracker = PlayerTracker(
        yolo_model=yolo_model,
        imgsz=imgsz,
    )

    all_results = []
    per_rally: dict[str, dict[str, Any]] = {}
    total_bucketed = BucketedMetrics()
    total_inference_time = 0.0
    total_frames = 0

    for i, rally in enumerate(rallies):
        rally_id_short = rally.rally_id[:8]
        print(f"  [{i+1}/{len(rallies)}] Rally {rally_id_short}... ", end="", flush=True)

        video_path = get_video_path(rally.video_id)
        if video_path is None:
            print("SKIP (video not found)")
            continue

        t0 = time.time()
        try:
            result = tracker.track_video(
                video_path=video_path,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
                stride=1,
                filter_enabled=True,
                filter_config=PlayerFilterConfig(),
            )

            # Offset frames to rally-relative
            if result.positions:
                first_frame = min(p.frame_number for p in result.positions)
                for p in result.positions:
                    p.frame_number -= first_frame

        except Exception as e:
            print(f"ERROR ({e})")
            continue

        elapsed = time.time() - t0
        total_inference_time += elapsed
        total_frames += result.frame_count

        # Distance-bucketed recall
        bucketed = compute_bucketed_recall(
            rally.ground_truth.positions,
            result.positions,
            iou_threshold=iou_threshold,
        )

        # Standard MOT evaluation
        try:
            eval_result = evaluate_rally(
                rally_id=rally.rally_id,
                ground_truth=rally.ground_truth,
                predictions=result,
                iou_threshold=iou_threshold,
                video_width=rally.video_width,
                video_height=rally.video_height,
            )
            mot = eval_result.aggregate
            hota = eval_result.hota_metrics
            hota_val = hota.hota if hota else 0.0

            print(
                f"HOTA={hota_val:.1%} F1={mot.f1:.1%} "
                f"near={bucketed.near_recall:.1%} "
                f"mid={bucketed.mid_recall:.1%} "
                f"far={bucketed.far_recall:.1%} "
                f"({elapsed:.1f}s)"
            )

            all_results.append(eval_result)
            per_rally[rally.rally_id] = {
                "hota": hota_val,
                "mota": mot.mota,
                "f1": mot.f1,
                "id_switches": mot.num_id_switches,
                "bucketed_recall": bucketed.to_dict(),
                "inference_time_s": elapsed,
            }

            # Accumulate bucket totals
            total_bucketed.near_total += bucketed.near_total
            total_bucketed.mid_total += bucketed.mid_total
            total_bucketed.far_total += bucketed.far_total
            total_bucketed.near_recall += bucketed.near_recall * bucketed.near_total
            total_bucketed.mid_recall += bucketed.mid_recall * bucketed.mid_total
            total_bucketed.far_recall += bucketed.far_recall * bucketed.far_total

        except Exception as e:
            print(f"EVAL ERROR ({e})")
            continue

    # Compute weighted average recall per bucket
    if total_bucketed.near_total > 0:
        total_bucketed.near_recall /= total_bucketed.near_total
    if total_bucketed.mid_total > 0:
        total_bucketed.mid_recall /= total_bucketed.mid_total
    if total_bucketed.far_total > 0:
        total_bucketed.far_recall /= total_bucketed.far_total

    # Aggregate MOT metrics
    agg_metrics: dict[str, Any] = {}
    if all_results:
        agg = aggregate_results(all_results)
        agg_hota = sum(m["hota"] for m in per_rally.values()) / len(per_rally)
        fps = total_frames / total_inference_time if total_inference_time > 0 else 0.0

        agg_metrics = {
            "hota": agg_hota,
            "mota": agg.mota,
            "f1": agg.f1,
            "precision": agg.precision,
            "recall": agg.recall,
            "id_switches": agg.num_id_switches,
            "fps": fps,
            "bucketed_recall": total_bucketed.to_dict(),
        }

        print(f"\n  Aggregate: HOTA={agg_hota:.1%} F1={agg.f1:.1%} FPS={fps:.1f}")
        print(
            f"  Distance recall: near={total_bucketed.near_recall:.1%} "
            f"mid={total_bucketed.mid_recall:.1%} "
            f"far={total_bucketed.far_recall:.1%}"
        )

    return {
        "variant": variant_name,
        "config": variant_config,
        "aggregate": agg_metrics,
        "per_rally": per_rally,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 2: Compare player detection at different resolutions"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file")
    parser.add_argument("--rally-id", type=str, help="Test single rally (prefix match)")
    parser.add_argument(
        "--variants", nargs="+",
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
        help="Variants to test (default: all)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Eval IoU threshold")

    args = parser.parse_args()

    from rallycut.evaluation.tracking.db import load_labeled_rallies

    print("Loading labeled rallies from database...")
    rallies = load_labeled_rallies()

    if args.rally_id:
        rallies = [r for r in rallies if r.rally_id.startswith(args.rally_id)]

    print(f"  Found {len(rallies)} rally(s) with ground truth")

    if not rallies:
        print("No rallies found!")
        return

    # Run each variant
    results: dict[str, Any] = {
        "experiment": "exp-player-hires",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_rallies": len(rallies),
        "variants": {},
    }

    for variant_name in args.variants:
        variant_config = VARIANTS[variant_name]
        variant_result = run_variant(
            variant_name, variant_config, rallies,
            iou_threshold=args.iou_threshold,
        )
        results["variants"][variant_name] = variant_result

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"{'Variant':<20} {'HOTA':>6} {'F1':>6} {'FPS':>6} "
        f"{'Near':>6} {'Mid':>6} {'Far':>6}"
    )
    print("-" * 80)

    for name in args.variants:
        v = results["variants"][name]
        agg = v.get("aggregate", {})
        br = agg.get("bucketed_recall", {})
        print(
            f"{name:<20} "
            f"{agg.get('hota', 0):.1%}  "
            f"{agg.get('f1', 0):.1%}  "
            f"{agg.get('fps', 0):5.1f}  "
            f"{br.get('near', {}).get('recall', 0):.1%}  "
            f"{br.get('mid', {}).get('recall', 0):.1%}  "
            f"{br.get('far', {}).get('recall', 0):.1%}"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
