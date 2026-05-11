"""FP-sweep for the proposed synthetic-block insertion rule.

For every pipeline ATTACK action in the fleet, check whether MS-TCN++
has a block-class peak in window [attack.frame + 1, attack.frame + max_gap]
with prob >= peak_floor. Each fire is classified TP/FP vs GT blocks
within ±15 frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15

BLOCK_IDX = ACTION_TYPES.index("block") + 1

PEAK_FLOORS = [0.30, 0.50, 0.70, 0.90]
MAX_GAPS = [10, 12, 15]


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

    # Preload per-rally data so we don't re-compute MS-TCN++ for each sweep cell.
    print("Caching per-rally MS-TCN++ block-class peaks...")
    rally_cache: list[dict[str, Any]] = []
    with get_connection() as conn:
        for r in gt["rallies"]:
            chash = r["video_content_hash"]
            if chash not in hash_to_id:
                continue
            vid = hash_to_id[chash]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT pt.frame_count, pt.court_split_y,
                              pt.ball_positions_json, pt.positions_json,
                              pt.actions_json
                       FROM rallies rr JOIN player_tracks pt
                       ON pt.rally_id = rr.id
                       WHERE rr.video_id = %s AND rr.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if not row:
                continue
            fcount, csy, bp_json, pp_json, aj = row
            if not aj:
                continue
            actions = sorted(aj.get("actions") or [],
                             key=lambda a: a.get("frame", 0))
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
            gt_blocks = sorted([int(a.get("frame", 0))
                                for a in r.get("action_ground_truth_json", []) or []
                                if a.get("action") == "block"])
            rally_cache.append({
                "video": meta[vid][1],
                "rally_start_ms": r["rally_start_ms"],
                "actions": actions,
                "seq_block": seq[BLOCK_IDX, :],
                "gt_blocks": gt_blocks,
            })

    print(f"Cached {len(rally_cache)} rallies.\n")

    print(f"{'peak_p':>7} {'max_gap':>8} {'fires':>6} {'TPs':>4} {'FPs':>4} "
          f"{'rallies_w_fire':>14}  FP examples")
    print("-" * 110)

    for peak_floor in PEAK_FLOORS:
        for max_gap in MAX_GAPS:
            fires = tps = fps = 0
            rallies_with_fire: set[str] = set()
            fp_examples: list[str] = []
            for r in rally_cache:
                actions = r["actions"]
                seq_block = r["seq_block"]
                gt_blocks = r["gt_blocks"]
                gt_blocks_unused = set(gt_blocks)
                for a in actions:
                    if a.get("action") != "attack":
                        continue
                    af = int(a.get("frame", 0))
                    lo = af + 1
                    hi = min(seq_block.shape[0], af + max_gap + 1)
                    if hi <= lo:
                        continue
                    window = seq_block[lo:hi]
                    if window.size == 0:
                        continue
                    peak_off = int(window.argmax())
                    peak_prob = float(window[peak_off])
                    if peak_prob < peak_floor:
                        continue
                    peak_frame = lo + peak_off
                    fires += 1
                    rallies_with_fire.add(f"{r['video']}/{r['rally_start_ms']}")
                    # TP if peak_frame is within ±HIT_TOL of an unused GT block
                    matched_gb = None
                    for gb in list(gt_blocks_unused):
                        if abs(peak_frame - gb) <= HIT_TOLERANCE:
                            matched_gb = gb
                            break
                    if matched_gb is not None:
                        gt_blocks_unused.discard(matched_gb)
                        tps += 1
                    else:
                        fps += 1
                        if len(fp_examples) < 4:
                            fp_examples.append(
                                f"{r['video']}/f{peak_frame} ({peak_prob:.2f})"
                            )
            fp_str = "; ".join(fp_examples[:3])
            print(f"{peak_floor:>7.2f} {max_gap:>8} {fires:>6} {tps:>4} {fps:>4} "
                  f"{len(rallies_with_fire):>14}  {fp_str}")


if __name__ == "__main__":
    main()
