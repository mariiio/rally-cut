#!/usr/bin/env python3
"""Experiment: Compare ReID models for BoT-SORT player tracking.

Tests different ReID model configurations to reduce ID switches
in beach volleyball player tracking. All variants use the same
YOLO detection model (yolov8n@1280) and post-processing pipeline.

Variants:
  - baseline: model=auto (YOLO backbone features, current default)
  - no_reid: ReID disabled (pure motion-based tracking)
  - yolo11n_cls: yolo11n-cls.pt (ImageNet classification features)
  - yolo11s_cls: yolo11s-cls.pt (larger classification model)

Usage:
    # Run all variants on all labeled rallies
    uv run python scripts/exp_reid_models.py

    # Quick test on single rally
    uv run python scripts/exp_reid_models.py --rally-id 87ce7bff

    # Test specific variants
    uv run python scripts/exp_reid_models.py --variants baseline yolo11n_cls

    # Save results
    uv run python scripts/exp_reid_models.py -o exp_reid_models.json
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


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "reid_model": None,  # Uses YAML default (model: auto)
        "with_reid": True,
        "description": "Current default (YOLO backbone features)",
    },
    "no_reid": {
        "reid_model": None,
        "with_reid": False,
        "description": "ReID disabled (motion-only tracking)",
    },
    "yolo11n_cls": {
        "reid_model": "yolo11n-cls.pt",
        "with_reid": True,
        "description": "YOLO11 nano classification (ImageNet features)",
    },
    "yolo11s_cls": {
        "reid_model": "yolo11s-cls.pt",
        "with_reid": True,
        "description": "YOLO11 small classification (larger model)",
    },
}


def run_variant(
    variant_name: str,
    variant_config: dict[str, Any],
    rallies: list[Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run a single ReID variant on all rallies."""
    from rallycut.evaluation.tracking.db import get_video_path
    from rallycut.evaluation.tracking.metrics import aggregate_results, evaluate_rally
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import PlayerTracker

    reid_model = variant_config.get("reid_model")
    with_reid = variant_config.get("with_reid", True)

    print(f"\n{'='*60}")
    print(f"  Variant: {variant_name}")
    print(f"  ReID model: {reid_model or 'auto'}, with_reid: {with_reid}")
    print(f"  {variant_config['description']}")
    print(f"{'='*60}")

    tracker = PlayerTracker(
        yolo_model="yolov8n",
        imgsz=1280,
        with_reid=with_reid,
        reid_model=reid_model,
    )

    all_results = []
    per_rally: dict[str, dict[str, Any]] = {}
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
            assa_val = hota.assa if hota else 0.0

            print(
                f"HOTA={hota_val:.1%} AssA={assa_val:.1%} "
                f"IDsw={mot.num_id_switches} "
                f"F1={mot.f1:.1%} "
                f"({elapsed:.1f}s)"
            )

            all_results.append(eval_result)
            per_rally[rally.rally_id] = {
                "hota": hota_val,
                "assa": assa_val,
                "mota": mot.mota,
                "f1": mot.f1,
                "precision": mot.precision,
                "recall": mot.recall,
                "id_switches": mot.num_id_switches,
                "inference_time_s": elapsed,
            }

        except Exception as e:
            print(f"EVAL ERROR ({e})")
            continue

    # Aggregate metrics
    agg_metrics: dict[str, Any] = {}
    if all_results:
        agg = aggregate_results(all_results)
        agg_hota = sum(m["hota"] for m in per_rally.values()) / len(per_rally)
        agg_assa = sum(m["assa"] for m in per_rally.values()) / len(per_rally)
        fps = total_frames / total_inference_time if total_inference_time > 0 else 0.0

        agg_metrics = {
            "hota": agg_hota,
            "assa": agg_assa,
            "mota": agg.mota,
            "f1": agg.f1,
            "precision": agg.precision,
            "recall": agg.recall,
            "id_switches": agg.num_id_switches,
            "fps": fps,
        }

        print(f"\n  Aggregate: HOTA={agg_hota:.1%} AssA={agg_assa:.1%} "
              f"IDsw={agg.num_id_switches} F1={agg.f1:.1%} FPS={fps:.1f}")

    return {
        "variant": variant_name,
        "config": {k: v for k, v in variant_config.items() if k != "description"},
        "description": variant_config["description"],
        "aggregate": agg_metrics,
        "per_rally": per_rally,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ReID models for BoT-SORT player tracking"
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
        "experiment": "exp-reid-models",
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
        f"{'Variant':<16} {'HOTA':>6} {'AssA':>6} {'IDsw':>5} "
        f"{'F1':>6} {'MOTA':>6} {'FPS':>6}  Description"
    )
    print("-" * 80)

    for name in args.variants:
        v = results["variants"][name]
        agg = v.get("aggregate", {})
        print(
            f"{name:<16} "
            f"{agg.get('hota', 0):.1%}  "
            f"{agg.get('assa', 0):.1%}  "
            f"{agg.get('id_switches', 0):4d}  "
            f"{agg.get('f1', 0):.1%}  "
            f"{agg.get('mota', 0):.1%}  "
            f"{agg.get('fps', 0):5.1f}  "
            f"{v.get('description', '')}"
        )

    # Per-rally ID switch comparison
    print("\n" + "=" * 80)
    print("PER-RALLY ID SWITCHES")
    print("=" * 80)
    header = f"{'Rally':<12}"
    for name in args.variants:
        header += f" {name:>14}"
    print(header)
    print("-" * 80)

    # Get all rally IDs
    all_rally_ids = set()
    for name in args.variants:
        v = results["variants"][name]
        all_rally_ids.update(v.get("per_rally", {}).keys())

    for rally_id in sorted(all_rally_ids):
        row = f"{rally_id[:8]}... "
        for name in args.variants:
            v = results["variants"][name]
            rally_data = v.get("per_rally", {}).get(rally_id, {})
            idsw = rally_data.get("id_switches", "-")
            hota = rally_data.get("hota", 0)
            if isinstance(idsw, int):
                row += f" {idsw:>4} ({hota:.0%})"
            else:
                row += f" {'?':>4}      "
        print(row)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
