"""Detect GT-rally-bounds mismatches in the panel.

For each panel rally:
  - Load GT actions (from beach_v11/action_ground_truth.json)
  - Load pred actions (from DB player_tracks.actions_json)
  - For each candidate frame offset O in a wide range:
      compute matches between (pred_frame + O) and gt_frame within +-15
  - The O that maximizes matches is the rally's best alignment.
  - If |O| > 30, flag as GT-bounds mismatch.
  - Report aligned-FN / aligned-FP per rally.

Outputs:
  - Per-rally line: (video, rally, best_offset, base_matches, aligned_matches,
                    base_FN, aligned_FN, base_FP, aligned_FP, classification)
  - Aggregate: panel-wide base FN/FP vs aligned FN/FP.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from rallycut.evaluation.tracking.db import get_connection
from rallycut.training.action_gt_query import load_for_rallies

PANEL_IDS = [
    "b5fb0594-d64f-4a0d-bad9-de8fc36414d0",
    "7d77980f-3006-40e0-adc0-db491a5bb659",
    "854bb250-3e91-47d2-944d-f62413e3cf45",
    "5c756c41-1cc1-4486-a95c-97398912cfbe",
    "073cb11b-c7ba-4fac-8cc9-b032b3152ad6",
]
GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
FRAME_TOLERANCE = 15
OFFSET_RANGE = range(-500, 501, 1)  # +-500 frames at 60fps = +-8.3s
BOUNDS_MISMATCH_THRESHOLD = 30  # |offset| > this -> flag as mismatch


def _match_at_offset(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]], offset: int
) -> int:
    """Count matches between (pred + offset) and gt within +-15."""
    used: set[int] = set()
    n_match = 0
    for g in gt:
        gf = int(g.get("frame", 0))
        best_i = -1
        best_d = FRAME_TOLERANCE + 1
        for i, p in enumerate(pred):
            if i in used:
                continue
            pf = int(p.get("frame", 0)) + offset
            d = abs(pf - gf)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d <= FRAME_TOLERANCE:
            used.add(best_i)
            n_match += 1
    return n_match


def _best_offset(gt: list[dict[str, Any]], pred: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (best_offset, max_matches). If no pred or gt, returns (0, 0)."""
    if not gt or not pred:
        return 0, 0
    # Coarse search at step 5, then refine at step 1.
    best_o, best_m = 0, _match_at_offset(gt, pred, 0)
    for o in range(-500, 501, 5):
        m = _match_at_offset(gt, pred, o)
        if m > best_m or (m == best_m and abs(o) < abs(best_o)):
            best_o, best_m = o, m
    # Refine around the coarse winner.
    for o in range(best_o - 5, best_o + 6):
        m = _match_at_offset(gt, pred, o)
        if m > best_m or (m == best_m and abs(o) < abs(best_o)):
            best_o, best_m = o, m
    return best_o, best_m


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos WHERE id = ANY(%s)",
                [PANEL_IDS],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    gt_data = json.loads(GT_PATH.read_text())
    hash_to_id = {h: vid for vid, (h, _) in meta.items()}
    panel_gt = [
        (hash_to_id[r["video_content_hash"]], r)
        for r in gt_data["rallies"]
        if r["video_content_hash"] in hash_to_id
    ]

    print(
        f"{'video':<6} {'rally':<10} {'#GT':>4} {'#Pred':>6} "
        f"{'best_off':>9} {'base_match':>11} {'aligned_match':>14} "
        f"{'base_FN':>8} {'align_FN':>9} {'base_FP':>8} {'align_FP':>9} "
        f"{'class':<14}"
    )
    print("-" * 130)

    totals = {
        "gt": 0, "pred": 0,
        "base_match": 0, "align_match": 0,
        "base_fn": 0, "align_fn": 0,
        "base_fp": 0, "align_fp": 0,
        "n_aligned_rallies": 0, "n_mismatched_rallies": 0,
    }

    with get_connection() as conn:
        for vid, gt_rally in panel_gt:
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.actions_json FROM rallies r
                       LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, gt_rally["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid = str(row[0])
            pred = list((row[1] or {}).get("actions") or [])
            gt = load_for_rallies(conn, [rid]).get(rid, [])

            base_match = _match_at_offset(gt, pred, 0)
            best_off, align_match = _best_offset(gt, pred)

            base_fn = len(gt) - base_match
            base_fp = len(pred) - base_match
            align_fn = len(gt) - align_match
            align_fp = len(pred) - align_match

            classification = (
                "ALIGNED"
                if abs(best_off) <= BOUNDS_MISMATCH_THRESHOLD
                else f"MISMATCH({best_off:+d})"
            )

            print(
                f"{name:<6} {rid[:8]:<10} {len(gt):>4} {len(pred):>6} "
                f"{best_off:>+9d} {base_match:>11d} {align_match:>14d} "
                f"{base_fn:>8d} {align_fn:>9d} {base_fp:>8d} {align_fp:>9d} "
                f"{classification:<14}"
            )

            totals["gt"] += len(gt)
            totals["pred"] += len(pred)
            totals["base_match"] += base_match
            totals["align_match"] += align_match
            totals["base_fn"] += base_fn
            totals["align_fn"] += align_fn
            totals["base_fp"] += base_fp
            totals["align_fp"] += align_fp
            if abs(best_off) <= BOUNDS_MISMATCH_THRESHOLD:
                totals["n_aligned_rallies"] += 1
            else:
                totals["n_mismatched_rallies"] += 1

    def _f1(matched: int, pred: int, gt: int) -> float:
        if not pred or not gt:
            return 0.0
        p = matched / pred
        r = matched / gt
        return 2 * p * r / (p + r) if (p + r) else 0.0

    print("-" * 130)
    print(
        f"BASE  : GT={totals['gt']}  Pred={totals['pred']}  "
        f"matched={totals['base_match']}  FN={totals['base_fn']}  FP={totals['base_fp']}  "
        f"F1={_f1(totals['base_match'], totals['pred'], totals['gt']):.3f}"
    )
    print(
        f"ALIGN : GT={totals['gt']}  Pred={totals['pred']}  "
        f"matched={totals['align_match']}  FN={totals['align_fn']}  FP={totals['align_fp']}  "
        f"F1={_f1(totals['align_match'], totals['pred'], totals['gt']):.3f}"
    )
    print(
        f"DELTA : aligned_rallies={totals['n_aligned_rallies']}  "
        f"mismatched_rallies={totals['n_mismatched_rallies']}  "
        f"FN: {totals['base_fn']} -> {totals['align_fn']} "
        f"({totals['align_fn'] - totals['base_fn']:+d})  "
        f"FP: {totals['base_fp']} -> {totals['align_fp']} "
        f"({totals['align_fp'] - totals['base_fp']:+d})"
    )


if __name__ == "__main__":
    main()
