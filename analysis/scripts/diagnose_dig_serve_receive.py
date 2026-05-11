"""Three diagnostics in one pass.

1. DIG FN+FP analysis: for every GT dig that's missed (51) and every pred
   dig that doesn't match a GT (40), identify what each is being
   misclassified as (or what the pipeline put there instead).

2. SERVE near-miss analysis: for every pred serve that's >15 frames off
   GT, characterise the gap direction + check MS-TCN++ peak position.

3. RECEIVE FP analysis: for every pred receive that doesn't match a GT
   receive, identify what the GT action AT THAT frame actually was.

All measurements vs beach_v11 action_ground_truth.json with HIT_TOLERANCE=15.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15


def _match_actions(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> tuple[list[tuple[dict, dict | None]], list[dict]]:
    """Greedy nearest-frame matching, prefer same-type when tied.

    Returns:
      matches: list of (gt_action, matched_pred_or_None) per GT action
      unmatched_pred: pred actions that weren't claimed by any GT
    """
    used: set[int] = set()
    matches: list[tuple[dict, dict | None]] = []
    for g in gt:
        gf = int(g.get("frame", 0))
        gt_type = g.get("action")
        candidates = [
            (i, p) for i, p in enumerate(pred)
            if i not in used and abs(int(p.get("frame", 0)) - gf) <= HIT_TOLERANCE
        ]
        if not candidates:
            matches.append((g, None))
            continue
        candidates.sort(key=lambda c: (
            0 if c[1].get("action") == gt_type else 1,
            abs(int(c[1].get("frame", 0)) - gf),
        ))
        chosen_idx, chosen_pl = candidates[0]
        used.add(chosen_idx)
        matches.append((g, chosen_pl))
    unmatched_pred = [p for i, p in enumerate(pred) if i not in used]
    return matches, unmatched_pred


def _nearest_action(actions: list[dict], frame: int) -> dict | None:
    if not actions:
        return None
    return min(actions, key=lambda a: abs(int(a.get("frame", 0)) - frame))


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

    # Accumulators
    dig_fn_misclass: dict[str, int] = defaultdict(int)  # what pipeline put within ±15 of missed GT dig
    dig_fn_examples: list[dict[str, Any]] = []
    dig_fp_gt_actual: dict[str, int] = defaultdict(int)  # what GT action was at the false-dig frame
    dig_fp_examples: list[dict[str, Any]] = []

    serve_near_miss: list[dict[str, Any]] = []  # pred serves with |Δ| > 15 vs GT serve
    serve_total = 0
    serve_hit = 0
    serve_miss_no_gt = 0  # pred serve without GT serve in rally

    receive_fp_gt_actual: dict[str, int] = defaultdict(int)
    receive_fp_examples: list[dict[str, Any]] = []

    with get_connection() as conn:
        for gt_rally in gt_data["rallies"]:
            ch = gt_rally["video_content_hash"]
            if ch not in h2v:
                continue
            vid = h2v[ch]
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.fps, pt.actions_json
                       FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, gt_rally["rally_start_ms"]],
                )
                row = cur.fetchone()
            if not row or not row[2]:
                continue
            rid, fps, aj = row
            pred_actions = list((aj or {}).get("actions") or [])
            gt_actions = gt_rally.get("action_ground_truth_json", []) or []
            matches, unmatched_pred = _match_actions(gt_actions, pred_actions)

            # 1. DIG FN: GT digs not matched
            for g, p in matches:
                if g.get("action") != "dig":
                    continue
                if p is None or p.get("action") == "dig":
                    continue  # matched and correctly labeled (or no match)
                # Pipeline has an action within ±15 but labeled as something else
                pred_type = p.get("action", "none")
                dig_fn_misclass[pred_type] += 1
                if len(dig_fn_examples) < 20:
                    dig_fn_examples.append({
                        "video": name, "rally_id": rid[:8],
                        "gt_frame": int(g.get("frame", 0)),
                        "pred_frame": int(p.get("frame", 0)),
                        "pred_as": pred_type,
                    })
                # Also count "no_pipeline_within_15" cases
            for g, p in matches:
                if g.get("action") != "dig" or p is not None:
                    continue
                dig_fn_misclass["NO_PIPELINE_NEAR"] += 1

            # 2. DIG FP: pred digs that don't match a GT dig
            #    Find each pipeline "dig" — was there a GT dig within ±15 that
            #    matched it? If not, what was the GT action at that frame?
            gt_digs = [g for g in gt_actions if g.get("action") == "dig"]
            for p in pred_actions:
                if p.get("action") != "dig":
                    continue
                pf = int(p.get("frame", 0))
                # Was this pred dig matched to a GT dig?
                matched_dig = any(
                    g.get("action") == "dig" and matched is not None
                    and int(matched.get("frame", 0)) == pf
                    for g, matched in matches
                )
                if matched_dig:
                    continue
                # Find nearest GT action to this frame
                nearest_gt = _nearest_action(gt_actions, pf)
                gt_type = "NONE" if not nearest_gt else (
                    nearest_gt.get("action", "?")
                    if abs(int(nearest_gt.get("frame", 0)) - pf) <= HIT_TOLERANCE
                    else "GT_GAP"
                )
                dig_fp_gt_actual[gt_type] += 1
                if len(dig_fp_examples) < 20:
                    dig_fp_examples.append({
                        "video": name, "rally_id": rid[:8],
                        "pred_frame": pf,
                        "gt_actual": gt_type,
                        "nearest_gt_frame": (
                            int(nearest_gt.get("frame", 0)) if nearest_gt else -1
                        ),
                    })

            # 3. SERVE near-miss
            gt_serves = [g for g in gt_actions if g.get("action") == "serve"]
            pred_serves = [p for p in pred_actions if p.get("action") == "serve"]
            for p in pred_serves:
                serve_total += 1
                pf = int(p.get("frame", 0))
                if not gt_serves:
                    serve_miss_no_gt += 1
                    continue
                nearest = min(gt_serves, key=lambda g:
                              abs(int(g.get("frame", 0)) - pf))
                delta = pf - int(nearest.get("frame", 0))
                if abs(delta) <= HIT_TOLERANCE:
                    serve_hit += 1
                else:
                    serve_near_miss.append({
                        "video": name, "rally_id": rid[:8],
                        "pred_frame": pf,
                        "gt_frame": int(nearest.get("frame", 0)),
                        "delta": delta,
                        "is_synthetic": p.get("isSynthetic", False),
                    })

            # 4. RECEIVE FP
            for p in pred_actions:
                if p.get("action") != "receive":
                    continue
                pf = int(p.get("frame", 0))
                matched_rec = any(
                    g.get("action") == "receive" and matched is not None
                    and int(matched.get("frame", 0)) == pf
                    for g, matched in matches
                )
                if matched_rec:
                    continue
                nearest_gt = _nearest_action(gt_actions, pf)
                gt_type = "NONE" if not nearest_gt else (
                    nearest_gt.get("action", "?")
                    if abs(int(nearest_gt.get("frame", 0)) - pf) <= HIT_TOLERANCE
                    else "GT_GAP"
                )
                receive_fp_gt_actual[gt_type] += 1
                if len(receive_fp_examples) < 20:
                    receive_fp_examples.append({
                        "video": name, "rally_id": rid[:8],
                        "pred_frame": pf,
                        "gt_actual": gt_type,
                    })

    print("=" * 80)
    print("1. DIG FN analysis (51 missed digs)")
    print("=" * 80)
    print("Where the pipeline put something else (within ±15 of GT dig frame):")
    for k, n in sorted(dig_fn_misclass.items(), key=lambda kv: -kv[1]):
        print(f"  pipeline labeled as {k:<22} {n:>4}")
    print("\nFirst 10 examples:")
    for e in dig_fn_examples[:10]:
        print(f"  {e['video']:<8} {e['rally_id']} gt_f={e['gt_frame']:>4} "
              f"pred_f={e['pred_frame']:>4} as={e['pred_as']}")

    print()
    print("=" * 80)
    print("2. DIG FP analysis (40 false digs)")
    print("=" * 80)
    print("What the GT action ACTUALLY was at false-dig pred frames:")
    for k, n in sorted(dig_fp_gt_actual.items(), key=lambda kv: -kv[1]):
        print(f"  GT was            {k:<22} {n:>4}")
    print("\nFirst 10 examples:")
    for e in dig_fp_examples[:10]:
        print(f"  {e['video']:<8} {e['rally_id']} pred_f={e['pred_frame']:>4} "
              f"gt_was={e['gt_actual']} nearest_gt_f={e['nearest_gt_frame']}")

    print()
    print("=" * 80)
    print(f"3. SERVE near-miss analysis ({serve_total} total preds, "
          f"{serve_hit} hits, {len(serve_near_miss)} near-misses, "
          f"{serve_miss_no_gt} no-GT-serve)")
    print("=" * 80)
    if serve_near_miss:
        deltas = sorted(s["delta"] for s in serve_near_miss)
        print(f"Delta range: [{deltas[0]}, {deltas[-1]}]")
        print(f"Median delta: {deltas[len(deltas)//2]}")
        # Bucket by direction
        early = sum(1 for d in deltas if d < -HIT_TOLERANCE)
        late = sum(1 for d in deltas if d > HIT_TOLERANCE)
        print(f"  pred too EARLY (Δ < −15): {early}")
        print(f"  pred too LATE  (Δ >  15): {late}")
        synth_count = sum(1 for s in serve_near_miss if s["is_synthetic"])
        print(f"  of which synthetic:       {synth_count}")
        print("\nFirst 10 near-misses (sorted by |delta|):")
        for s in sorted(serve_near_miss, key=lambda x: abs(x["delta"]))[:10]:
            print(f"  {s['video']:<8} {s['rally_id']} pred_f={s['pred_frame']:>4} "
                  f"gt_f={s['gt_frame']:>4} Δ={s['delta']:>+4} synth={s['is_synthetic']}")

    print()
    print("=" * 80)
    print("4. RECEIVE FP analysis (26 false receives)")
    print("=" * 80)
    print("What the GT action ACTUALLY was at false-receive pred frames:")
    for k, n in sorted(receive_fp_gt_actual.items(), key=lambda kv: -kv[1]):
        print(f"  GT was            {k:<22} {n:>4}")
    print("\nFirst 10 examples:")
    for e in receive_fp_examples[:10]:
        print(f"  {e['video']:<8} {e['rally_id']} pred_f={e['pred_frame']:>4} "
              f"gt_was={e['gt_actual']}")


if __name__ == "__main__":
    main()
