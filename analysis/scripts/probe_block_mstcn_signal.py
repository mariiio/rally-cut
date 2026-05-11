"""Check MS-TCN++ block-class probability at each GT block frame.

If MS-TCN++ has a clear block-class peak at these frames, we can use it
as the insertion signal for synthetic-block contacts (parallel to v1.1
synthetic-serve placement).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")


def _bp_from_json(bp_json: Any) -> list[BallPosition]:
    return [
        BallPosition(
            frame_number=int(b.get("frameNumber", 0)),
            x=float(b.get("x", 0.0)),
            y=float(b.get("y", 0.0)),
            confidence=float(b.get("confidence", 0.0)),
        )
        for b in (bp_json or [])
        if isinstance(b, dict)
    ]


def _pp_from_json(pp_json: Any) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=int(p.get("frameNumber", 0)),
            track_id=int(p.get("trackId", -1)),
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            width=float(p.get("width", 0.0)),
            height=float(p.get("height", 0.0)),
            confidence=float(p.get("confidence", 0.0)),
        )
        for p in (pp_json or [])
        if isinstance(p, dict)
    ]


def main() -> None:
    print(f"MS-TCN++ ACTION_TYPES: {ACTION_TYPES}")
    if "block" not in ACTION_TYPES:
        print("WARNING: 'block' is NOT in MS-TCN++ ACTION_TYPES — no block-class probability!")
        block_idx = -1
    else:
        block_idx = ACTION_TYPES.index("block") + 1
        print(f"block class index: {block_idx}")
    print()

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

    print(f"{'video':<8} {'rally':<10} {'gt_block_f':>10} "
          f"{'block_p@gt':>11} {'block_p_peak':>13} {'peak_f':>7} "
          f"{'attack_p@gt':>12}")
    print("-" * 90)

    block_at_gt: list[float] = []
    peak_offsets: list[int] = []

    with get_connection() as conn:
        for gt_rally in gt["rallies"]:
            chash = gt_rally["video_content_hash"]
            if chash not in hash_to_id:
                continue
            actions = gt_rally.get("action_ground_truth_json", [])
            blocks = [a for a in actions if a.get("action") == "block"]
            if not blocks:
                continue
            vid = hash_to_id[chash]
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.frame_count, pt.court_split_y,
                              pt.ball_positions_json, pt.positions_json,
                              pt.actions_json
                       FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, gt_rally["rally_start_ms"]],
                )
                row = cur.fetchone()
            if not row:
                continue
            rid, fcount, csy, bp_json, pp_json, aj = row
            bp = _bp_from_json(bp_json)
            pp = _pp_from_json(pp_json)
            ta_str = (aj or {}).get("teamAssignments", {}) or {}
            ta_int = {int(k): (0 if v == "A" else 1) for k, v in ta_str.items()
                      if v in ("A", "B")}
            seq = get_sequence_probs(
                bp, pp, csy, fcount or 0, ta_int, calibrator=None,
            )
            if seq is None:
                continue
            attack_idx = ACTION_TYPES.index("attack") + 1
            for block in blocks:
                gf = int(block.get("frame", 0))
                if not (0 <= gf < seq.shape[1]):
                    continue
                block_p_at_gt = (
                    float(seq[block_idx, gf]) if block_idx > 0 else 0.0
                )
                attack_p_at_gt = float(seq[attack_idx, gf])
                # Find local block-class peak in [gf-5, gf+5]
                if block_idx > 0:
                    lo = max(0, gf - 5)
                    hi = min(seq.shape[1], gf + 6)
                    window = seq[block_idx, lo:hi]
                    if window.size:
                        peak_offset = int(window.argmax())
                        peak_prob = float(window[peak_offset])
                        peak_frame = lo + peak_offset
                    else:
                        peak_prob, peak_frame = 0.0, -1
                else:
                    peak_prob, peak_frame = 0.0, -1
                block_at_gt.append(block_p_at_gt)
                peak_offsets.append(peak_frame - gf)
                print(
                    f"{name[:8]:<8} {rid[:8]:<10} {gf:>10} "
                    f"{block_p_at_gt:>11.3f} {peak_prob:>13.3f} {peak_frame:>7} "
                    f"{attack_p_at_gt:>12.3f}"
                )

    if block_at_gt:
        print()
        print(f"Median block-class prob AT GT frame:  {sorted(block_at_gt)[len(block_at_gt)//2]:.3f}")
        print(f"Mean block-class prob AT GT frame:    {sum(block_at_gt)/len(block_at_gt):.3f}")
        n_strong = sum(1 for p in block_at_gt if p >= 0.5)
        n_med = sum(1 for p in block_at_gt if p >= 0.3)
        print(f"GT block frames with block-prob >= 0.50: {n_strong}/{len(block_at_gt)}")
        print(f"GT block frames with block-prob >= 0.30: {n_med}/{len(block_at_gt)}")


if __name__ == "__main__":
    main()
