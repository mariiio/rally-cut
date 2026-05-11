"""Deduplication-pattern diagnostic.

For every pipeline action across the fleet, classify it as TP/FP vs
beach_v11 GT (HIT_TOLERANCE=15), then measure the distance (in frames)
to the nearest OTHER pipeline action in the same rally.

If FPs cluster at small distances (e.g., <40 frames) and TPs don't, a
min-distance gate in the main contact-detection loop would clean them
up without harming TPs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rallycut.evaluation.tracking.db import get_connection

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15


def _match(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> tuple[set[int], set[int]]:
    """Returns (matched_pred_indices, matched_gt_indices) — greedy nearest,
    same-type preferred among ±15."""
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for gi, g in enumerate(gt):
        gf = int(g.get("frame", 0))
        gt_type = g.get("action")
        candidates = [
            (i, p) for i, p in enumerate(pred)
            if i not in used_pred and abs(int(p.get("frame", 0)) - gf) <= HIT_TOLERANCE
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda c: (
            0 if c[1].get("action") == gt_type else 1,
            abs(int(c[1].get("frame", 0)) - gf),
        ))
        chosen_idx, _ = candidates[0]
        used_pred.add(chosen_idx)
        used_gt.add(gi)
    return used_pred, used_gt


def main() -> None:
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    hashes = {r["video_content_hash"] for r in gt_data["rallies"]}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos WHERE content_hash = ANY(%s)",
                [list(hashes)],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    h2v = {h: v for v, (h, _) in meta.items()}

    tp_min_gaps: list[int] = []
    fp_min_gaps: list[int] = []
    fp_records: list[dict[str, Any]] = []
    tp_records: list[dict[str, Any]] = []

    with get_connection() as conn:
        for r in gt_data["rallies"]:
            ch = r["video_content_hash"]
            if ch not in h2v:
                continue
            vid = h2v[ch]
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT rr.id, pt.actions_json
                       FROM rallies rr JOIN player_tracks pt ON pt.rally_id = rr.id
                       WHERE rr.video_id = %s AND rr.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if not row or not row[1]:
                continue
            rid, aj = row
            pred = sorted(aj.get("actions") or [], key=lambda a: a.get("frame", 0))
            gt_actions = r.get("action_ground_truth_json", []) or []
            used_pred, _ = _match(gt_actions, pred)
            pred_frames = [int(p.get("frame", 0)) for p in pred]

            for i, p in enumerate(pred):
                pf = int(p.get("frame", 0))
                # Distance to nearest other pred (left or right)
                other_frames = [pf2 for j, pf2 in enumerate(pred_frames) if j != i]
                if not other_frames:
                    continue
                min_gap = min(abs(pf - pf2) for pf2 in other_frames)
                if i in used_pred:
                    tp_min_gaps.append(min_gap)
                    if min_gap < 40 and len(tp_records) < 20:
                        tp_records.append({
                            "video": name, "rally": rid[:8],
                            "frame": pf, "action": p.get("action"),
                            "min_gap": min_gap,
                        })
                else:
                    fp_min_gaps.append(min_gap)
                    if min_gap < 40 and len(fp_records) < 30:
                        fp_records.append({
                            "video": name, "rally": rid[:8],
                            "frame": pf, "action": p.get("action"),
                            "min_gap": min_gap,
                        })

    def _qs(a: list[int]) -> tuple[int, int, int, int, int, int]:
        if not a:
            return (0, 0, 0, 0, 0, 0)
        s = sorted(a)
        return (
            s[0],
            s[max(0, len(s) // 20)],  # p5
            s[len(s) // 4],            # p25
            s[len(s) // 2],            # p50
            s[3 * len(s) // 4],        # p75
            s[-1],
        )

    print(f"TP pipeline actions: {len(tp_min_gaps)}")
    print(f"FP pipeline actions: {len(fp_min_gaps)}")
    print()
    print(f"{'group':<5} {'min':>5} {'p5':>5} {'p25':>5} {'p50':>5} {'p75':>5} {'max':>5}")
    tp_q = _qs(tp_min_gaps)
    fp_q = _qs(fp_min_gaps)
    print(f"{'TP':<5} {tp_q[0]:>5} {tp_q[1]:>5} {tp_q[2]:>5} {tp_q[3]:>5} {tp_q[4]:>5} {tp_q[5]:>5}")
    print(f"{'FP':<5} {fp_q[0]:>5} {fp_q[1]:>5} {fp_q[2]:>5} {fp_q[3]:>5} {fp_q[4]:>5} {fp_q[5]:>5}")

    print("\n=== Threshold sweep: min-distance gate (would drop pred contacts whose nearest other is < threshold) ===")
    print(f"{'threshold':>10} {'TP drops':>10} {'FP drops':>10} "
          f"{'precision_gain':>15} {'recall_loss':>15}")
    total_tp = len(tp_min_gaps)
    total_fp = len(fp_min_gaps)
    for thr in [5, 10, 15, 20, 25, 30, 35, 40, 50]:
        tp_drops = sum(1 for g in tp_min_gaps if g < thr)
        fp_drops = sum(1 for g in fp_min_gaps if g < thr)
        # Assume the dropped contact is removed; what's the new precision?
        # Old: TP/(TP+FP). New: (TP - tp_drops) / ((TP - tp_drops) + (FP - fp_drops))
        new_tp = total_tp - tp_drops
        new_fp = total_fp - fp_drops
        new_prec = new_tp / max(1, new_tp + new_fp)
        old_prec = total_tp / max(1, total_tp + total_fp)
        print(f"{thr:>10} {tp_drops:>10} {fp_drops:>10} "
              f"{(new_prec - old_prec) * 100:>+14.2f}pp "
              f"{tp_drops:>14}")

    print("\n=== Sample FPs with min_gap < 40 (by action type) ===")
    by_action = defaultdict(list)
    for r in fp_records:
        by_action[r["action"]].append(r)
    for action, recs in by_action.items():
        print(f"\n  {action} FPs with small min_gap:")
        for rec in recs[:5]:
            print(f"    {rec['video']:<8} {rec['rally']} f={rec['frame']:>4} gap={rec['min_gap']:>3}")

    print("\n=== Sample TPs with min_gap < 40 (would we LOSE these?) ===")
    by_action_tp = defaultdict(list)
    for r in tp_records:
        by_action_tp[r["action"]].append(r)
    for action, recs in by_action_tp.items():
        print(f"\n  {action} TPs with small min_gap:")
        for rec in recs[:5]:
            print(f"    {rec['video']:<8} {rec['rally']} f={rec['frame']:>4} gap={rec['min_gap']:>3}")


if __name__ == "__main__":
    main()
