"""FP-sweep for the proposed block-relabel rule.

Across ALL stored pipeline actions (post-v1.3), count how many would be
relabeled as block by the candidate rule:

  - current label is ATTACK / DIG / RECEIVE (i.e., not already block)
  - prev pipeline action's label is ATTACK
  - gap (current.frame − prev.frame) ≤ BLOCK_MAX_GAP_FRAMES
  - opposite team
  - ball at current.frame is above net_y by ≥ BLOCK_BALL_NET_MARGIN

For each fire: classify as TP (GT block within ±15 frames) or FP (no GT
block nearby).

Sweep over (max_gap, net_margin) grid to find the Pareto frontier.
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

MAX_GAPS = [7, 10, 15]
NET_MARGINS = [0.05, 0.08, 0.10]


def _opposite_team(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a != b


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
    hash_to_id = {h: vid for vid, (h, _) in meta.items()}

    print(f"{'max_gap':>8} {'net_margin':>10} {'fires':>6} "
          f"{'TPs':>4} {'FPs':>4} {'FP examples (vid/rally/frame/labels)':<60}")
    print("-" * 110)

    all_fires: list[dict[str, Any]] = []

    for max_gap in MAX_GAPS:
        for margin in NET_MARGINS:
            tps = 0
            fps = 0
            fp_examples: list[str] = []
            with get_connection() as conn:
                for r in gt["rallies"]:
                    chash = r["video_content_hash"]
                    if chash not in hash_to_id:
                        continue
                    vid = hash_to_id[chash]
                    name = meta[vid][1]
                    with conn.cursor() as cur:
                        cur.execute(
                            """SELECT rr.id, pt.actions_json, pt.court_split_y
                               FROM rallies rr JOIN player_tracks pt
                               ON pt.rally_id = rr.id
                               WHERE rr.video_id = %s AND rr.start_ms = %s""",
                            [vid, r["rally_start_ms"]],
                        )
                        row = cur.fetchone()
                    if not row or not row[1]:
                        continue
                    rid_str = str(row[0])
                    aj = cast(dict[str, Any], row[1])
                    net_y = row[2] or 0.5
                    actions = sorted(aj.get("actions") or [],
                                     key=lambda a: a.get("frame", 0))
                    gt_actions = load_for_rallies(conn, [rid_str]).get(rid_str, [])
                    gt_blocks = {int(a.get("frame", 0))
                                 for a in gt_actions
                                 if a.get("action") == "block"}

                    for i in range(1, len(actions)):
                        a = actions[i]
                        prev = actions[i-1]
                        if a.get("action") == "block":
                            continue
                        if prev.get("action") != "attack":
                            continue
                        gap = int(a.get("frame", 0)) - int(prev.get("frame", 0))
                        if gap <= 0 or gap > max_gap:
                            continue
                        if not _opposite_team(a.get("team"), prev.get("team")):
                            continue
                        ball_y = float(a.get("ballY", 0.5))
                        if ball_y >= net_y - margin:
                            continue
                        # Fires
                        af = int(a.get("frame", 0))
                        is_tp = any(abs(af - bf) <= HIT_TOLERANCE
                                    for bf in gt_blocks)
                        if is_tp:
                            tps += 1
                        else:
                            fps += 1
                            if len(fp_examples) < 5:
                                fp_examples.append(
                                    f"{name}/{r.get('rally_start_ms', 0)//1000}s "
                                    f"f{af} ({prev.get('action')}→{a.get('action')})"
                                )
                        if max_gap == 10 and margin == 0.05:
                            all_fires.append({
                                "video": name,
                                "rally_start_ms": r.get("rally_start_ms"),
                                "current_frame": af,
                                "current_action": a.get("action"),
                                "prev_frame": int(prev.get("frame", 0)),
                                "gap": gap,
                                "ball_y_minus_net": ball_y - net_y,
                                "is_tp": is_tp,
                            })

            fp_str = "; ".join(fp_examples[:3]) if fp_examples else "-"
            print(f"{max_gap:>8} {margin:>10.2f} {tps + fps:>6} "
                  f"{tps:>4} {fps:>4}  {fp_str}")

    out_path = Path("reports/block_rule_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"fires_at_max_gap_10_margin_005": all_fires}, indent=2))
    print(f"\nWrote {out_path}")
    print("\nNote: TP = the relabel would correctly produce a block at a GT block frame.")
    print("FP = relabel would produce a block where no GT block exists nearby.")


if __name__ == "__main__":
    main()
