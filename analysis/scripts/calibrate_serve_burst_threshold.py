"""Calibrate the trajectory direction-change threshold for serve detection.

For each rally with action GT and a correctly-placed real (non-synthetic) serve:
  - Load ball_positions
  - At the GT serve frame, compute compute_direction_change(...) over a
    range of check_frames windows (4, 6, 8).
  - Print the value at the GT serve frame and the max value at any non-
    serve frame in [rally_start, gt_serve_f - 5].

Use the distribution to pick BURST_THRESHOLD such that >=90% of GT serve
frames exceed it AND <10% of non-serve frames in the search windows do.
"""

from __future__ import annotations

import json
from pathlib import Path

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import compute_direction_change

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
# Use every video that has action GT — frame and action-type labels are
# clean; only player attribution is known-noisy and we don't compare it
# here. The 5-video "panel" was just a quick subset used during the prior
# diagnostic session.
WINDOW_VALUES = [4, 6, 8]


def _ball_by_frame(bp_json):
    out = {}
    for b in bp_json or []:
        if not isinstance(b, dict):
            continue
        f = int(b.get("frameNumber", b.get("frame", 0)))
        out[f] = BallPosition(
            frame_number=f,
            x=float(b.get("x", 0.0)),
            y=float(b.get("y", 0.0)),
            confidence=float(b.get("confidence", 0.0)),
        )
    return out


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)

    # Derive video IDs from the GT file: any content_hash referenced is fair game.
    gt_hashes = {r["video_content_hash"] for r in gt["rallies"]}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos "
                "WHERE content_hash = ANY(%s)",
                [list(gt_hashes)],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    hash_to_id = {h: vid for vid, (h, _) in meta.items()}

    print(f"{'video':<8} {'rally':<10} {'gt_serve_f':>10} "
          f"{'dc@4':>7} {'dc@6':>7} {'dc@8':>7} "
          f"{'max_nonserve@6':>16}")
    print("-" * 80)

    serve_dc_values: list[float] = []
    nonserve_dc_values: list[float] = []

    with get_connection() as conn:
        for r in gt["rallies"]:
            chash = r["video_content_hash"]
            if chash not in hash_to_id:
                continue
            vid = hash_to_id[chash]
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.ball_positions_json, pt.actions_json
                       FROM rallies r LEFT JOIN player_tracks pt
                       ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, bp_json, aj = row
            bbf = _ball_by_frame(bp_json)
            actions = (aj or {}).get("actions") or []
            pred_serve = next(
                (a for a in actions if a.get("action") == "serve"),
                None,
            )
            if pred_serve is None or pred_serve.get("isSynthetic", False):
                continue  # Only calibrate on REAL detected serves
            gt_serve_f = next(
                (a["frame"] for a in r["action_ground_truth_json"]
                 if a.get("action") == "serve"),
                None,
            )
            if gt_serve_f is None:
                continue
            # Only count as "correctly placed" if pred and gt agree within 15.
            if abs(pred_serve.get("frame", -999) - gt_serve_f) > 15:
                continue

            dcs = {w: compute_direction_change(bbf, gt_serve_f, w) for w in WINDOW_VALUES}
            # Max direction-change at non-serve frames in [0, gt_serve_f - 5].
            max_nonserve = 0.0
            for f in range(0, max(0, gt_serve_f - 5)):
                v = compute_direction_change(bbf, f, 6)
                if v > max_nonserve:
                    max_nonserve = v

            serve_dc_values.append(dcs[6])
            nonserve_dc_values.append(max_nonserve)

            print(
                f"{name:<8} {rid[:8]:<10} {gt_serve_f:>10} "
                f"{dcs[4]:>7.1f} {dcs[6]:>7.1f} {dcs[8]:>7.1f} "
                f"{max_nonserve:>16.1f}"
            )

    print("-" * 80)
    if serve_dc_values:
        print(f"Serve direction-change @ window=6: "
              f"min={min(serve_dc_values):.1f}  median={sorted(serve_dc_values)[len(serve_dc_values)//2]:.1f}  "
              f"max={max(serve_dc_values):.1f}  n={len(serve_dc_values)}")
    if nonserve_dc_values:
        print(f"Max non-serve dc in window @ window=6: "
              f"min={min(nonserve_dc_values):.1f}  median={sorted(nonserve_dc_values)[len(nonserve_dc_values)//2]:.1f}  "
              f"max={max(nonserve_dc_values):.1f}  n={len(nonserve_dc_values)}")
    # Pick threshold: max non-serve + 5deg margin, but cap at 30deg as a
    # sanity floor (typical contact threshold).
    if serve_dc_values and nonserve_dc_values:
        proposed = max(min(nonserve_dc_values) - 1, 30)
        # If serve values straddle this, dial it to the 25th percentile of serves.
        if any(s < proposed for s in serve_dc_values):
            sorted_serves = sorted(serve_dc_values)
            p25 = sorted_serves[len(sorted_serves) // 4]
            proposed = p25
        print(f"Proposed BURST_THRESHOLD: {proposed:.1f} deg")


if __name__ == "__main__":
    main()
