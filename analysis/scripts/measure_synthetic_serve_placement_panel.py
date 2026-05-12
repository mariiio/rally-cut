"""Fleet measurement for synthetic-serve placement (v1.1) — CLEAN A/B.

For each rally with both a GT serve and a pred serve in the action GT pool:
  - BASE: re-run pipeline with v1.1 placement DISABLED.
  - POST: re-run pipeline with v1.1 placement ENABLED.
  - Compare both serve frames to GT.

This isolates v1.1's effect from detector-version drift artifacts. The
prior approach (comparing stored DB action to a fresh re-run) conflated
v1.1's effect with the cumulative drift of the entire pipeline since
the DB was last populated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rallycut.tracking.synthetic_serve_placement as ssp_mod
from rallycut.evaluation.tracking.db import get_connection
from rallycut.training.action_gt_query import load_for_rallies
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


def _run_pipeline(
    *,
    disable_v11: bool,
    bp: list[BallPosition],
    pp: list[PlayerPosition],
    csy: float | None,
    fcount: int,
    ta_int: dict[int, int],
    primary_raw: list[Any],
    seq_probs: Any,
) -> tuple[int, bool]:
    """Run detect_contacts + classify_rally_actions; return (serve_frame, is_synthetic).

    serve_frame=-1 if no serve action emitted.
    """
    ssp_mod._DISABLE_V11_PLACEMENT = disable_v11
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
        serve = next(
            (a for a in ra.actions if a.action_type.value == "serve"),
            None,
        )
        if serve is None:
            return (-1, False)
        return (serve.frame, serve.is_synthetic)
    finally:
        ssp_mod._DISABLE_V11_PLACEMENT = False


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)

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
                       FROM rallies r LEFT JOIN player_tracks pt
                       ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row
            rid_str = str(rid)
            gt_labels = load_for_rallies(conn, [rid_str]).get(rid_str, [])

            gt_serve_f = next(
                (a["frame"] for a in gt_labels
                 if a.get("action") == "serve"),
                None,
            )
            if gt_serve_f is None:
                continue

            ta_str = (aj or {}).get("teamAssignments", {}) or {}
            ta_int = {int(k): (0 if v == "A" else 1)
                      for k, v in ta_str.items()
                      if v in ("A", "B")}
            bp = _ball_positions_from_json(bp_json)
            pp = _player_positions_from_json(pp_json)
            seq_probs = get_sequence_probs(
                bp, pp, csy, fcount or 0, ta_int, calibrator=None,
            )
            if seq_probs is None:
                continue

            base_f, base_synth = _run_pipeline(
                disable_v11=True, bp=bp, pp=pp, csy=csy,
                fcount=fcount or 0, ta_int=ta_int,
                primary_raw=primary_raw or [], seq_probs=seq_probs,
            )
            post_f, post_synth = _run_pipeline(
                disable_v11=False, bp=bp, pp=pp, csy=csy,
                fcount=fcount or 0, ta_int=ta_int,
                primary_raw=primary_raw or [], seq_probs=seq_probs,
            )
            if base_f == -1 and post_f == -1:
                continue  # no serve at all in either run
            base_diff = base_f - gt_serve_f
            post_diff = post_f - gt_serve_f
            # Use whichever run shows a synthetic to bucket the rally.
            kind = "synth" if (base_synth or post_synth) else "real"

            base_hit = base_f != -1 and abs(base_diff) <= HIT_TOLERANCE
            post_hit = post_f != -1 and abs(post_diff) <= HIT_TOLERANCE
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
    print(f"Synthetic: {counts['synth_hit_base']}/{counts['synth_total']} "
          f"-> {counts['synth_hit_post']}/{counts['synth_total']} hits")
    print(f"Real:      {counts['real_hit_base']}/{counts['real_total']} "
          f"-> {counts['real_hit_post']}/{counts['real_total']} hits")
    print(f"Total fixed: "
          f"{counts['synth_hit_post'] - counts['synth_hit_base']:+d} synth, "
          f"{counts['real_hit_post'] - counts['real_hit_base']:+d} real")


if __name__ == "__main__":
    main()
