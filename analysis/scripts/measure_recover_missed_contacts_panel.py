"""Panel measurement for coherence-driven contact recovery (Sub-2.B).

Dry-run only — does NOT modify DB. For each panel rally + 073cb11b:
  - Load action GT from training_datasets/beach_v11.
  - Compute baseline FN/FP against current persisted actions.
  - Run recover_rally; merge proposed recoveries.
  - Recompute FN/FP/F1 with proposed recoveries inserted.
  - Print per-rally diff + aggregate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.contact_recovery import load_rally_inputs, recover_rally

PANEL_IDS = [
    "b5fb0594-d64f-4a0d-bad9-de8fc36414d0",
    "7d77980f-3006-40e0-adc0-db491a5bb659",
    "854bb250-3e91-47d2-944d-f62413e3cf45",
    "5c756c41-1cc1-4486-a95c-97398912cfbe",
    "073cb11b-c7ba-4fac-8cc9-b032b3152ad6",
]
GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
FRAME_TOLERANCE = 15


def _match_actions(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> tuple[int, int, int]:
    """Return (matched, fn, fp) by greedy frame-tolerance matching."""
    used: set[int] = set()
    matched = 0
    for g in gt:
        gf = int(g.get("frame", 0))
        best_i = -1
        best_d = FRAME_TOLERANCE + 1
        for i, p in enumerate(pred):
            if i in used:
                continue
            d = abs(int(p.get("frame", 0)) - gf)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d <= FRAME_TOLERANCE:
            used.add(best_i)
            matched += 1
    return matched, len(gt) - matched, len(pred) - matched


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
        f"{'video':<6} {'rally':<10} {'#GT':>4} "
        f"{'BASE_pred':>10} {'BASE_FN':>8} {'BASE_FP':>8} "
        f"{'POST_pred':>10} {'POST_FN':>8} {'POST_FP':>8} "
        f"{'recov':>5} {'gate-rej':>9} {'audit-rej':>10}"
    )
    print("-" * 110)

    totals = {
        "gt": 0, "base_pred": 0, "base_matched": 0, "base_fn": 0, "base_fp": 0,
        "post_pred": 0, "post_matched": 0, "post_fn": 0, "post_fp": 0,
        "recovered": 0, "gate_rej": 0, "audit_rej": 0,
    }

    with get_connection() as conn:
        for vid, gt_rally in panel_gt:
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id FROM rallies r
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, gt_rally["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None:
                continue
            rid = str(row[0])
            try:
                inputs = load_rally_inputs(rid)
            except ValueError:
                continue
            base_actions = list(inputs.actions_json.get("actions") or [])

            # Baseline match.
            gt_actions = gt_rally["action_ground_truth_json"]
            b_match, b_fn, b_fp = _match_actions(gt_actions, base_actions)

            # Recovery proposal (no DB write).
            res = recover_rally(inputs)
            post_actions = sorted(
                base_actions + res.recovered_actions,
                key=lambda a: int(a.get("frame", 0)),
            )
            p_match, p_fn, p_fp = _match_actions(gt_actions, post_actions)

            print(
                f"{name:<6} {rid[:8]:<10} {len(gt_actions):>4} "
                f"{len(base_actions):>10} {b_fn:>8} {b_fp:>8} "
                f"{len(post_actions):>10} {p_fn:>8} {p_fp:>8} "
                f"{len(res.recovered_actions):>5} {res.rejected_by_gate:>9} "
                f"{res.rejected_by_audit:>10}"
            )
            totals["gt"] += len(gt_actions)
            totals["base_pred"] += len(base_actions)
            totals["base_matched"] += b_match
            totals["base_fn"] += b_fn
            totals["base_fp"] += b_fp
            totals["post_pred"] += len(post_actions)
            totals["post_matched"] += p_match
            totals["post_fn"] += p_fn
            totals["post_fp"] += p_fp
            totals["recovered"] += len(res.recovered_actions)
            totals["gate_rej"] += res.rejected_by_gate
            totals["audit_rej"] += res.rejected_by_audit

    def _f1(matched: int, pred: int, gt: int) -> float:
        if not pred or not gt:
            return 0.0
        p = matched / pred
        r = matched / gt
        if not (p + r):
            return 0.0
        return 2 * p * r / (p + r)

    print("-" * 110)
    print(
        f"BASE  : GT={totals['gt']}  Pred={totals['base_pred']}  "
        f"FN={totals['base_fn']}  FP={totals['base_fp']}  "
        f"F1={_f1(totals['base_matched'], totals['base_pred'], totals['gt']):.3f}"
    )
    print(
        f"POST  : GT={totals['gt']}  Pred={totals['post_pred']}  "
        f"FN={totals['post_fn']}  FP={totals['post_fp']}  "
        f"F1={_f1(totals['post_matched'], totals['post_pred'], totals['gt']):.3f}"
    )
    print(
        f"DELTA : recovered={totals['recovered']}  "
        f"gate-rej={totals['gate_rej']}  audit-rej={totals['audit_rej']}  "
        f"FN: {totals['base_fn']} -> {totals['post_fn']} "
        f"({totals['post_fn'] - totals['base_fn']:+d})  "
        f"FP: {totals['base_fp']} -> {totals['post_fp']} "
        f"({totals['post_fp'] - totals['base_fp']:+d})"
    )


if __name__ == "__main__":
    main()
