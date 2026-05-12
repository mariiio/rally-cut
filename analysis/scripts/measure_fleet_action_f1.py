"""Fleet-wide action F1 — every video with beach_v11 action GT.

Same greedy frame-tolerance matching as `post_retrack_measurements.py`
measurement 1, but across ALL videos (not just the 5-video panel).
Reads stored DB actions_json post-deploy; doesn't re-run the pipeline.

Reports per-video subtotals + per-action-type breakdown + final totals.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection
from rallycut.training.action_gt_query import load_for_rallies

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15


def _match_actions(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> tuple[int, int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (matched, fn, fp, fn_records, fp_records) by greedy match."""
    used: set[int] = set()
    matched = 0
    fn_records: list[dict[str, Any]] = []
    for g in gt:
        gf = int(g.get("frame", 0))
        best_i = -1
        best_d = HIT_TOLERANCE + 1
        for i, p in enumerate(pred):
            if i in used:
                continue
            d = abs(int(p.get("frame", 0)) - gf)
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= HIT_TOLERANCE:
            used.add(best_i)
            matched += 1
        else:
            fn_records.append({"action": g.get("action"), "frame": gf})
    fp_records = [p for i, p in enumerate(pred) if i not in used]
    return matched, len(gt) - matched, len(pred) - matched, fn_records, fp_records


def _f1(matched: int, pred: int, gt: int) -> float:
    if not pred or not gt:
        return 0.0
    p = matched / pred
    r = matched / gt
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)

    gt_hashes = {r["video_content_hash"] for r in gt["rallies"]}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos WHERE content_hash = ANY(%s)",
                [list(gt_hashes)],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    hash_to_id: dict[str, str] = {
        cast(str, h): cast(str, vid) for vid, (h, _) in meta.items()
    }

    print(f"Fleet action F1 — beach_v11 GT, HIT_TOLERANCE={HIT_TOLERANCE} frames")
    print("=" * 80)
    print(f"{'video':<8} {'rallies':>8} {'#GT':>5} {'#Pred':>6} "
          f"{'matched':>8} {'FN':>4} {'FP':>4} {'F1':>6}")
    print("-" * 80)

    per_video: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rallies": 0, "gt": 0, "pred": 0, "matched": 0, "fn": 0, "fp": 0}
    )
    per_action_gt: dict[str, int] = defaultdict(int)
    per_action_fn: dict[str, int] = defaultdict(int)
    per_action_fp: dict[str, int] = defaultdict(int)

    with get_connection() as conn:
        for gt_rally in gt["rallies"]:
            chash = gt_rally["video_content_hash"]
            if chash not in hash_to_id:
                continue
            vid = hash_to_id[chash]
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
            aj = cast(dict[str, Any] | None, row[1]) or {}
            pred_actions = list(aj.get("actions") or [])
            gt_actions = load_for_rallies(conn, [rid]).get(rid, [])
            matched, fn, fp, fn_records, fp_records = _match_actions(
                gt_actions, pred_actions,
            )
            v = per_video[name]
            v["rallies"] += 1
            v["gt"] += len(gt_actions)
            v["pred"] += len(pred_actions)
            v["matched"] += matched
            v["fn"] += fn
            v["fp"] += fp
            for g in gt_actions:
                per_action_gt[str(g.get("action") or "unknown")] += 1
            for r in fn_records:
                per_action_fn[str(r.get("action") or "unknown")] += 1
            for r in fp_records:
                per_action_fp[str(r.get("action") or "unknown")] += 1

    totals = {"gt": 0, "pred": 0, "matched": 0, "fn": 0, "fp": 0}
    for name in sorted(per_video.keys()):
        v = per_video[name]
        f1 = _f1(v["matched"], v["pred"], v["gt"])
        print(
            f"{name[:8]:<8} {v['rallies']:>8} {v['gt']:>5} {v['pred']:>6} "
            f"{v['matched']:>8} {v['fn']:>4} {v['fp']:>4} {f1:>6.3f}"
        )
        for k in totals:
            totals[k] += v[k]

    print("-" * 80)
    f1 = _f1(totals["matched"], totals["pred"], totals["gt"])
    print(
        f"{'TOTAL':<8} {sum(v['rallies'] for v in per_video.values()):>8} "
        f"{totals['gt']:>5} {totals['pred']:>6} "
        f"{totals['matched']:>8} {totals['fn']:>4} {totals['fp']:>4} {f1:>6.3f}"
    )
    print()
    print(f"Fleet F1 (HIT_TOL=±{HIT_TOLERANCE}): {f1:.3f}")
    print(f"  Matched: {totals['matched']}/{totals['gt']} GT  "
          f"({100*totals['matched']/max(1, totals['gt']):.1f}%)")
    print(f"  Precision: {totals['matched']}/{totals['pred']} pred  "
          f"({100*totals['matched']/max(1, totals['pred']):.1f}%)")
    print(f"  FN: {totals['fn']}    FP: {totals['fp']}")
    print()
    print("Per-action-type breakdown:")
    print(f"  {'action':<10} {'#GT':>5} {'FN':>4} {'FP':>4} {'recall':>7}")
    for action in sorted(per_action_gt.keys()):
        n = per_action_gt[action]
        fn = per_action_fn.get(action, 0)
        fp = per_action_fp.get(action, 0)
        recall = (n - fn) / n if n else 0.0
        print(f"  {action:<10} {n:>5} {fn:>4} {fp:>4} {recall:>7.3f}")


if __name__ == "__main__":
    main()
