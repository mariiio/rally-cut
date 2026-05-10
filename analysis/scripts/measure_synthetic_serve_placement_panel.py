"""Fleet measurement for synthetic-serve placement (v1.1).

For each rally with both a GT serve and a pred serve in the action GT pool:
  - Compare current pred serve frame vs GT (BASE).
  - Re-run classify_rally_actions with v1.1 placement (POST).
  - Print per-rally diff + aggregate.

Counts:
  - Synthetic serves: hit (within +-15 of GT) vs miss.
  - Real (detected) serves: hit vs miss (sanity check — must not regress).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15


def _ball_positions_from_json(bp_json: Any) -> list[BallPosition]:
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


def _player_positions_from_json(pp_json: Any) -> list[PlayerPosition]:
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


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)

    # Use every video referenced in the GT file (66 videos / ~340 rallies)
    # — frame and action-type labels are clean across the whole pool;
    # only player attribution is known-noisy and we don't compare it here.
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

    print(f"{'video':<8} {'rally':<10} {'kind':<10} "
          f"{'base_f':>7} {'post_f':>7} {'gt_f':>5} "
          f"{'base_diff':>10} {'post_diff':>10}  {'verdict':<25}")
    print("-" * 100)

    counts = {"synth_hit_base": 0, "synth_hit_post": 0,
              "synth_total": 0,
              "real_hit_base": 0, "real_hit_post": 0,
              "real_total": 0}

    with get_connection() as conn:
        for r in gt["rallies"]:
            chash = r["video_content_hash"]
            if chash not in hash_to_id:
                continue
            vid = hash_to_id[chash]
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.fps, pt.frame_count, pt.court_split_y,
                              pt.ball_positions_json, pt.positions_json,
                              pt.actions_json, pt.primary_track_ids
                       FROM rallies r LEFT JOIN player_tracks pt
                       ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row

            actions = (aj or {}).get("actions") or []
            pred_serve = next(
                (a for a in actions if a.get("action") == "serve"), None,
            )
            gt_serve_f = next(
                (a["frame"] for a in r["action_ground_truth_json"]
                 if a.get("action") == "serve"),
                None,
            )
            if pred_serve is None or gt_serve_f is None:
                continue
            base_f = pred_serve.get("frame", -1)
            base_diff = base_f - gt_serve_f
            kind = "synth" if pred_serve.get("isSynthetic", False) else "real"

            # Re-run with v1.1 placement.
            ta_str = (aj or {}).get("teamAssignments", {}) or {}
            ta_int = {int(k): (0 if v == "A" else 1) for k, v in ta_str.items()
                      if v in ("A", "B")}
            bp = _ball_positions_from_json(bp_json)
            pp = _player_positions_from_json(pp_json)
            seq_probs = get_sequence_probs(
                bp, pp, csy, fcount or 0, ta_int, calibrator=None,
            )
            if seq_probs is None:
                post_f = base_f
            else:
                contact_seq = detect_contacts(
                    ball_positions=bp, player_positions=pp,
                    config=ContactDetectionConfig(),
                    net_y=csy, frame_count=fcount or None,
                    team_assignments=ta_int,
                    sequence_probs=seq_probs,
                    primary_track_ids=list(primary_raw or []) or None,
                )
                ra = classify_rally_actions(
                    contact_seq,
                    team_assignments=ta_int,
                    sequence_probs=seq_probs,
                )
                post_serve = next(
                    (a for a in ra.actions if a.action_type.value == "serve"),
                    None,
                )
                post_f = post_serve.frame if post_serve else -1
            post_diff = post_f - gt_serve_f

            base_hit = abs(base_diff) <= HIT_TOLERANCE
            post_hit = abs(post_diff) <= HIT_TOLERANCE
            if kind == "synth":
                counts["synth_total"] += 1
                if base_hit:
                    counts["synth_hit_base"] += 1
                if post_hit:
                    counts["synth_hit_post"] += 1
            else:
                counts["real_total"] += 1
                if base_hit:
                    counts["real_hit_base"] += 1
                if post_hit:
                    counts["real_hit_post"] += 1

            verdict = (
                "no change"
                if base_hit == post_hit
                else ("FIXED" if not base_hit and post_hit else "REGRESSION")
            )
            print(
                f"{name:<8} {rid[:8]:<10} {kind:<10} "
                f"{base_f:>7} {post_f:>7} {gt_serve_f:>5} "
                f"{base_diff:>+10d} {post_diff:>+10d}  {verdict:<25}"
            )

    print("-" * 100)
    print(
        f"Synthetic: {counts['synth_hit_base']}/{counts['synth_total']} "
        f"-> {counts['synth_hit_post']}/{counts['synth_total']} hits"
    )
    print(
        f"Real:      {counts['real_hit_base']}/{counts['real_total']} "
        f"-> {counts['real_hit_post']}/{counts['real_total']} hits"
    )
    print(
        f"Total fixed: {counts['synth_hit_post'] - counts['synth_hit_base']:+d} "
        f"synth, {counts['real_hit_post'] - counts['real_hit_base']:+d} real"
    )


if __name__ == "__main__":
    main()
