"""Clean A/B fleet measurement for v1.3 serve-peak prepend.

For each rally with action GT:
  - BASE: re-run pipeline with v1.3 prepend DISABLED.
  - POST: re-run pipeline with v1.3 prepend ENABLED.

Compare both first-serve frames to GT. Report:
  - Synthetic serves hit-rate change
  - Real serves hit-rate change (must NOT regress)
  - Per-rally fixes / regressions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rallycut.tracking.serve_prepend as sp_mod
from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.training.action_gt_query import load_for_rallies
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15


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


def _run_pipeline(
    *,
    disable_v13: bool,
    bp: list[BallPosition],
    pp: list[PlayerPosition],
    csy: float | None,
    fcount: int,
    ta_int: dict[int, int],
    primary_raw: list[Any],
    seq_probs: Any,
) -> tuple[int, bool]:
    sp_mod._DISABLE_V13_PREPEND = disable_v13
    try:
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
        serves = [a for a in ra.actions if a.action_type.value == "serve"]
        if not serves:
            return (-1, False)
        first = min(serves, key=lambda a: a.frame)
        return (first.frame, bool(first.is_synthetic))
    finally:
        sp_mod._DISABLE_V13_PREPEND = False


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

    print(f"{'video':<8} {'rally':<10} {'kind':<10} "
          f"{'base_f':>7} {'post_f':>7} {'gt_f':>5} "
          f"{'base_diff':>10} {'post_diff':>10}  {'verdict':<25}")
    print("-" * 100)
    counts = {
        "synth_hit_base": 0, "synth_hit_post": 0, "synth_total": 0,
        "real_hit_base": 0, "real_hit_post": 0, "real_total": 0,
    }

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
                       FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, _fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row
            rid_str = str(rid)
            gt_labels = load_for_rallies(conn, [rid_str]).get(rid_str, [])
            gt_serve_f = next(
                (a["frame"] for a in gt_labels
                 if a.get("action") == "serve"), None,
            )
            if gt_serve_f is None:
                continue
            ta_str = (aj or {}).get("teamAssignments", {}) or {}
            ta_int = {int(k): (0 if v == "A" else 1) for k, v in ta_str.items()
                      if v in ("A", "B")}
            bp = _bp_from_json(bp_json)
            pp = _pp_from_json(pp_json)
            seq_probs = get_sequence_probs(
                bp, pp, csy, fcount or 0, ta_int, calibrator=None,
            )
            if seq_probs is None:
                continue
            base_f, base_synth = _run_pipeline(
                disable_v13=True, bp=bp, pp=pp, csy=csy,
                fcount=fcount or 0, ta_int=ta_int,
                primary_raw=primary_raw or [], seq_probs=seq_probs,
            )
            post_f, post_synth = _run_pipeline(
                disable_v13=False, bp=bp, pp=pp, csy=csy,
                fcount=fcount or 0, ta_int=ta_int,
                primary_raw=primary_raw or [], seq_probs=seq_probs,
            )
            if base_f == -1 and post_f == -1:
                continue
            base_diff = base_f - gt_serve_f
            post_diff = post_f - gt_serve_f
            kind = "synth" if (base_synth or post_synth) else "real"
            base_hit = base_f != -1 and abs(base_diff) <= HIT_TOLERANCE
            post_hit = post_f != -1 and abs(post_diff) <= HIT_TOLERANCE
            if kind == "synth":
                counts["synth_total"] += 1
                counts["synth_hit_base"] += int(base_hit)
                counts["synth_hit_post"] += int(post_hit)
            else:
                counts["real_total"] += 1
                counts["real_hit_base"] += int(base_hit)
                counts["real_hit_post"] += int(post_hit)
            verdict = (
                "no change" if base_hit == post_hit
                else ("FIXED" if not base_hit and post_hit else "REGRESSION")
            )
            print(
                f"{name[:8]:<8} {rid[:8]:<10} {kind:<10} "
                f"{base_f:>7} {post_f:>7} {gt_serve_f:>5} "
                f"{base_diff:>+10d} {post_diff:>+10d}  {verdict:<25}"
            )

    print("-" * 100)
    print(f"Synthetic: {counts['synth_hit_base']}/{counts['synth_total']} -> "
          f"{counts['synth_hit_post']}/{counts['synth_total']} hits")
    print(f"Real:      {counts['real_hit_base']}/{counts['real_total']} -> "
          f"{counts['real_hit_post']}/{counts['real_total']} hits")
    print(f"Total fixed: "
          f"{counts['synth_hit_post'] - counts['synth_hit_base']:+d} synth, "
          f"{counts['real_hit_post'] - counts['real_hit_base']:+d} real")


if __name__ == "__main__":
    main()
