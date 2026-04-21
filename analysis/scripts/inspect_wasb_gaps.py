"""Raw WASB confidence distribution around Cat 6 FN frames.

For each `6-ball_gap_exceeds_interp` FN in the Phase 4 output, load the stored
ball_positions_json for its rally and inspect what confidences (if any) exist
in the GT frame ± 10f window.

Tells us whether Fix D's premise holds:
  - if many positions with conf 0.1-0.3 exist in the gap window → lowering
    _CONFIDENCE_THRESHOLD from 0.3 to 0.15 would admit them (Fix D viable)
  - if confidences are all ≥ 0.5 but SPECIFIC frames have no detection at all
    (WASB genuine miss) → Fix D doesn't help

Usage:
    cd analysis && uv run python scripts/inspect_wasb_gaps.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_action_detection import load_rallies_with_action_gt

REPO = Path(__file__).resolve().parent.parent
CAT_PATH = REPO / "outputs" / "phase4_category_assignments.jsonl"
OUT_PATH = REPO / "outputs" / "wasb_gap_inspection_2026_04_21.jsonl"

WINDOW = 10  # ± frames around GT


def main() -> None:
    # Load Cat 6 FNs
    cat6 = []
    for line in CAT_PATH.open():
        a = json.loads(line)
        if a["primary_category"] == "6-ball_gap_exceeds_interp":
            cat6.append(a)
    print(f"Loaded {len(cat6)} Cat 6 FNs.")

    from collections import defaultdict
    rallies_of_interest = defaultdict(list)
    for fn in cat6:
        rallies_of_interest[fn["rally_id"]].append(fn)

    print(f"Spanning {len(rallies_of_interest)} rallies. Loading rally data...")
    all_rallies = load_rallies_with_action_gt()
    target_ids = set(rallies_of_interest.keys())

    results = []
    for rally in all_rallies:
        if rally.rally_id not in target_ids:
            continue
        ball_positions = rally.ball_positions_json or []
        # Index by frame
        by_frame = {}
        for bp in ball_positions:
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0:
                by_frame[bp["frameNumber"]] = bp.get("confidence", 1.0)
        for fn in rallies_of_interest[rally.rally_id]:
            gt_frame = fn["gt_frame"]
            window_confs = {}
            for f in range(gt_frame - WINDOW, gt_frame + WINDOW + 1):
                if f in by_frame:
                    window_confs[f] = by_frame[f]
            # Summarize
            confs_in_gap = [c for f, c in window_confs.items() if abs(f - gt_frame) <= 5]
            n_in_gap = len(confs_in_gap)
            below_03 = sum(1 for c in confs_in_gap if c < 0.3)
            below_015 = sum(1 for c in confs_in_gap if c < 0.15)
            p50 = (sorted(confs_in_gap)[len(confs_in_gap)//2]
                   if confs_in_gap else None)
            results.append({
                "rally_id": fn["rally_id"],
                "gt_frame": gt_frame,
                "gt_action": fn["gt_action"],
                "ball_gap_frames_corpus": fn.get("ball_gap_frames"),
                "n_detections_in_gt_pm5": n_in_gap,
                "n_detections_below_0.30": below_03,
                "n_detections_below_0.15": below_015,
                "median_conf_in_window": p50,
                "confs_by_frame_offset": {
                    str(f - gt_frame): round(c, 3)
                    for f, c in sorted(window_confs.items())
                },
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary
    total = len(results)
    no_det_in_gap = sum(1 for r in results if r["n_detections_in_gt_pm5"] == 0)
    has_low_conf = sum(1 for r in results if r["n_detections_below_0.30"] > 0)
    has_very_low_conf = sum(1 for r in results if r["n_detections_below_0.15"] > 0)
    avg_n = sum(r["n_detections_in_gt_pm5"] for r in results) / max(1, total)

    print(f"\n=== Cat 6 WASB gap inspection (n={total}) ===")
    print(f"Cases with ZERO detections in GT ± 5f:                 {no_det_in_gap}/{total}")
    print(f"Cases with ≥1 detection at conf < 0.30 (Fix D viable): {has_low_conf}/{total}")
    print(f"Cases with ≥1 detection at conf < 0.15:                {has_very_low_conf}/{total}")
    print(f"Avg detections in GT ± 5f window:                      {avg_n:.2f}")

    # Histogram of minimum distance to nearest detection in window
    min_dist_hist = Counter()
    for r in results:
        if not r["confs_by_frame_offset"]:
            min_dist_hist["no_det_in_±10f"] += 1
        else:
            min_d = min(abs(int(k)) for k in r["confs_by_frame_offset"].keys())
            if min_d <= 2:
                min_dist_hist["nearest_det_within_2f"] += 1
            elif min_d <= 5:
                min_dist_hist["nearest_det_3-5f"] += 1
            elif min_d <= 10:
                min_dist_hist["nearest_det_6-10f"] += 1
            else:
                min_dist_hist["nearest_det_>10f"] += 1
    print("\nDistance from GT to nearest WASB detection:")
    for k, n in sorted(min_dist_hist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {n}")

    print(f"\nWrote {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
