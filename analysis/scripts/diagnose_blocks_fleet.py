"""Block-detection fleet diagnostic.

For every GT block in the fleet (28 total), determine WHY it's missed by
the current pipeline. The user hypothesises three block-specific signals:

  1. Sequence context: a block happens right after an attack (the
     opposing team's attack). So the action preceding the block in GT
     should usually be "attack".
  2. Spatial: blocks happen at the net. Ball Y near `net_y` AND a player
     is at the net with bbox bottom high (player jumping up).
  3. Pose: distinctive block pose (arms up, body extended near net).

This diagnostic checks each block against these signals to see whether
the pipeline COULD detect blocks if it had a block-specific rule.

Per-block report:
  - rally_id, frame, action_before / action_after (from GT context)
  - candidate_status: NO_CAND / VALIDATED / REJECTED (from detect_contacts)
  - if pipeline_action exists: what it was classified as
  - ball_y vs net_y (distance to net plane)
  - nearest-player distance + player height proxy (bbox h normalised)
  - frame gap to nearest preceding attack (GT context, 0 = no prior attack)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.training.action_gt_query import load_for_rallies
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15
CONTEXT_WINDOW = 60  # search window for prior-action context (frames)


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
            keypoints=p.get("keypoints"),
        )
        for p in (pp_json or [])
        if isinstance(p, dict)
    ]


def _nearest_player(
    pp: list[PlayerPosition], ball_x: float, ball_y: float, frame: int
) -> tuple[float, float, float]:
    """Returns (min_distance, nearest_player_bbox_top, nearest_player_height).

    Distance is normalised image distance. bbox_top = p.y. height = p.height.
    """
    best_dist = float("inf")
    best_top = 0.0
    best_height = 0.0
    for p in pp:
        if p.frame_number != frame:
            continue
        px = p.x + p.width / 2
        py = p.y + p.height / 2
        d = ((ball_x - px) ** 2 + (ball_y - py) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_top = p.y
            best_height = p.height
    return best_dist, best_top, best_height


def _ball_at_frame(
    bp: list[BallPosition], frame: int
) -> tuple[float, float] | None:
    """Returns (ball_x, ball_y) at exact frame, or None if not visible."""
    for b in bp:
        if b.frame_number == frame:
            if b.x > 0.01 or b.y > 0.01:
                return (b.x, b.y)
            return None
    return None


def _interpolate_ball(
    bp: list[BallPosition], frame: int
) -> tuple[float, float] | None:
    """Returns nearest visible ball position within ±5 frames."""
    by_frame = {b.frame_number: b for b in bp}
    if frame in by_frame:
        b = by_frame[frame]
        if b.x > 0.01 or b.y > 0.01:
            return (b.x, b.y)
    for d in range(1, 6):
        for f in (frame - d, frame + d):
            b = by_frame.get(f)
            if b is not None and (b.x > 0.01 or b.y > 0.01):
                return (b.x, b.y)
    return None


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

    # Build per-rally GT lookup (block frame only; GT actions loaded from DB below)
    block_cases: list[dict[str, Any]] = []
    for gt_rally in gt["rallies"]:
        chash = gt_rally["video_content_hash"]
        if chash not in hash_to_id:
            continue
        # We record only rally_start_ms for each block frame; gt_actions come from DB.
        block_cases.append({
            "rally_hash": chash,
            "video_id": hash_to_id[chash],
            "video_name": meta[hash_to_id[chash]][1],
            "rally_start_ms": gt_rally["rally_start_ms"],
        })

    print(f"Total GT blocks: {len(block_cases)}")
    print()
    header = (
        f"{'video':<8} {'rally':<10} {'gt_f':>5} {'cand?':>6} "
        f"{'pred_as':>10} {'pred_f':>6} {'prev_act':>10} {'gap_to_attack':>14} "
        f"{'by-ny':>7} {'pdist':>6} {'p_top':>6} {'p_h':>6}"
    )
    print(header)
    print("-" * len(header))

    bucket_counts: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []

    with get_connection() as conn:
        for case in block_cases:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.fps, pt.frame_count, pt.court_split_y,
                              pt.ball_positions_json, pt.positions_json,
                              pt.actions_json, pt.primary_track_ids
                       FROM rallies r LEFT JOIN player_tracks pt
                       ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [case["video_id"], case["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, _fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row
            rid_str = str(rid)
            gt_actions = load_for_rallies(conn, [rid_str]).get(rid_str, [])
            blocks_in_rally = [a for a in gt_actions if a.get("action") == "block"]
            if not blocks_in_rally:
                continue

            # Pre-compute ball + player positions once per rally
            bp = _bp_from_json(bp_json)
            pp = _pp_from_json(pp_json)
            ta_str = (aj or {}).get("teamAssignments", {}) or {}
            ta_int = {
                int(k): (0 if v == "A" else 1)
                for k, v in ta_str.items()
                if v in ("A", "B")
            }
            seq = get_sequence_probs(
                bp, pp, csy, fcount or 0, ta_int, calibrator=None,
            )
            contact_seq = detect_contacts(
                ball_positions=bp,
                player_positions=pp,
                config=ContactDetectionConfig(),
                frame_count=fcount or None,
                team_assignments=ta_int,
                sequence_probs=seq,
                primary_track_ids=list(primary_raw or []) or None,
            )

            # Process each block in this rally
            for block in blocks_in_rally:
                block_f = int(block.get("frame", 0))

                # Find prior action in GT (the action preceding this block)
                gt_sorted = sorted(gt_actions, key=lambda a: a.get("frame", 0))
                prev_action_type = "none"
                prev_attack_gap = -1
                for a in gt_sorted:
                    af = int(a.get("frame", 0))
                    if af >= block_f:
                        break
                    prev_action_type = a.get("action", "none")
                    if a.get("action") == "attack":
                        prev_attack_gap = block_f - af

                # Compute ball at block frame
                ball_xy = _interpolate_ball(bp, block_f)
                ball_y = ball_xy[1] if ball_xy else -1.0
                by_minus_ny = ball_y - (csy or 0.5) if ball_y > 0 else -99.0

                pdist = -1.0
                p_top = -1.0
                p_h = -1.0
                if ball_xy:
                    pdist, p_top, p_h = _nearest_player(pp, ball_xy[0], ball_xy[1], block_f)

                # Pick nearest pipeline-detected contact to GT block frame
                cand_status = "NO_CAND"
                cand_frame = -1
                pred_action = "-"
                if contact_seq.contacts:
                    nearest = min(
                        contact_seq.contacts,
                        key=lambda c: abs(c.frame - block_f),
                    )
                    if abs(nearest.frame - block_f) <= HIT_TOLERANCE:
                        cand_status = "VALIDATED" if nearest.is_validated else "REJECTED"
                        cand_frame = nearest.frame

                # If candidate exists, what does the pipeline classify it as?
                if cand_status != "NO_CAND":
                    stored_actions = (aj or {}).get("actions", []) or []
                    pred = next(
                        (a for a in stored_actions
                         if abs(int(a.get("frame", 0)) - block_f) <= HIT_TOLERANCE),
                        None,
                    )
                    if pred:
                        pred_action = pred.get("action", "?")
                        cand_frame = int(pred.get("frame", 0))

                # Bucket
                if cand_status == "NO_CAND":
                    bucket = "no_candidate"
                elif pred_action == "block":
                    bucket = "correctly_classified"
                elif pred_action != "-":
                    bucket = f"misclassified_as_{pred_action}"
                elif cand_status == "REJECTED":
                    bucket = "candidate_rejected"
                else:
                    bucket = "other"
                bucket_counts[bucket] += 1

                print(
                    f"{case['video_name'][:8]:<8} "
                    f"{rid_str[:8]:<10} {block_f:>5} {cand_status:>6} "
                    f"{pred_action:>10} {cand_frame:>6} "
                    f"{prev_action_type[:10]:>10} {prev_attack_gap:>14} "
                    f"{by_minus_ny:>7.3f} {pdist:>6.3f} {p_top:>6.3f} {p_h:>6.3f}"
                )
                rows.append({
                    "video": case["video_name"], "rally_id": rid_str,
                    "block_frame": block_f, "cand_status": cand_status,
                    "pred_action": pred_action, "pred_frame": cand_frame,
                    "prev_action": prev_action_type, "prev_attack_gap": prev_attack_gap,
                    "ball_y_minus_net": by_minus_ny, "player_distance": pdist,
                    "player_bbox_top": p_top, "player_bbox_height": p_h,
                    "bucket": bucket,
                })

    print()
    print("=== Bucket summary ===")
    total = sum(bucket_counts.values())
    for bucket, n in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {bucket:<32} {n:>3} ({100*n/total:.1f}%)")
    print(f"  {'TOTAL':<32} {total:>3}")

    print()
    print("=== Signal stats (across all GT blocks) ===")
    prev_attack_n = sum(1 for r in rows if r["prev_action"] == "attack")
    print(f"  Previous action is 'attack': {prev_attack_n}/{len(rows)} "
          f"({100*prev_attack_n/max(1,len(rows)):.1f}%)")
    pdist_ok = sum(1 for r in rows if 0 < r["player_distance"] <= 0.10)
    print(f"  Nearest player within 0.10 of ball: {pdist_ok}/{len(rows)}")
    near_net = sum(1 for r in rows if abs(r["ball_y_minus_net"]) <= 0.08)
    print(f"  Ball within 0.08 of net_y: {near_net}/{len(rows)}")
    high_player = sum(1 for r in rows if r["player_bbox_top"] > 0 and r["player_bbox_top"] < 0.35)
    print(f"  Nearest player top above 0.35 (player jumping high): {high_player}/{len(rows)}")

    out_path = Path("reports/block_diagnostic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows, "buckets": dict(bucket_counts)}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
