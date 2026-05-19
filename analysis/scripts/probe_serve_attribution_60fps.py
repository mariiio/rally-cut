"""Probe: classify v4 Serve attribution failures on 60fps videos.

For each GT SERVE action across all 60fps videos, compare:
  - GT player (rally_action_ground_truth.resolved_track_id)
  - Pipeline predicted player (matching action in player_tracks.actions_json)
  - Candidate list at the matching contact (player_tracks.contacts_json playerCandidates)

Classify each failure as:
  EMPTY_CANDIDATES   — playerCandidates is [] or playerTrackId is -1
  GT_NOT_IN_CANDS    — GT player not in candidate list
  WRONG_RANK         — GT in candidates but a different one was picked (ranking error)
  CORRECT            — matched

Compare to 30fps cohort. Read-only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from typing import Any

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
MATCH_WINDOW = 10


def main() -> int:
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, COALESCE(v.fps, 30.0) AS fps,
                   r.id AS rally_id,
                   gt.frame AS gt_frame, gt.resolved_track_id AS gt_player,
                   pt.actions_json, pt.contacts_json
            FROM rally_action_ground_truth gt
            JOIN rallies r ON gt.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE gt.action = 'SERVE'
              AND gt.resolved_track_id IS NOT NULL
              AND pt.actions_json IS NOT NULL
              AND pt.contacts_json IS NOT NULL
            ORDER BY v.fps DESC, v.name, r.id
            """,
        )
        rows = cur.fetchall()

    print(f"Loaded {len(rows)} GT Serve actions with stored predictions", flush=True)

    by_cohort: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, list[dict]] = defaultdict(list)

    for name, fps, rally_id, gt_frame, gt_player, actions_json, contacts_json in rows:
        cohort = "60fps" if fps > 40 else "30fps"
        # Parse
        if isinstance(actions_json, str):
            actions = json.loads(actions_json).get("actions", [])
        else:
            actions = actions_json.get("actions", [])
        if isinstance(contacts_json, str):
            contacts = json.loads(contacts_json).get("contacts", [])
        else:
            contacts = contacts_json.get("contacts", [])

        # Find the SERVE action in pipeline output matching this GT frame.
        # Field is "action" (lowercase) per actions_json schema.
        matched_action = None
        for a in actions:
            if a.get("action") == "serve" and abs(a.get("frame", -10000) - gt_frame) <= MATCH_WINDOW:
                matched_action = a
                break

        if matched_action is None:
            by_cohort[cohort]["NO_SERVE_DETECTED"] += 1
            continue

        picked_pid = matched_action.get("playerTrackId", -1)

        # Find the contact whose frame matches the picked action's frame
        picked_frame = matched_action.get("frame")
        matched_contact = None
        for c in contacts:
            if abs(c.get("frame", -10000) - picked_frame) <= 2:
                matched_contact = c
                break
        candidates = matched_contact.get("playerCandidates", []) if matched_contact else []
        cand_tids = [int(c[0]) for c in candidates]

        gt_player = int(gt_player)
        picked_pid = int(picked_pid)

        if picked_pid == gt_player:
            by_cohort[cohort]["CORRECT"] += 1
            continue

        # Failure — classify
        if picked_pid == -1 or not candidates:
            cls = "EMPTY_CANDIDATES"
        elif gt_player not in cand_tids:
            cls = "GT_NOT_IN_CANDS"
        else:
            cls = "WRONG_RANK"

        by_cohort[cohort][cls] += 1

        # Save up to 5 samples per class per cohort
        if len(samples[f"{cohort}_{cls}"]) < 5:
            samples[f"{cohort}_{cls}"].append({
                "name": name, "rally_id": str(rally_id)[:8], "gt_frame": gt_frame,
                "gt_player": gt_player, "picked": picked_pid,
                "candidates": [(int(t), round(float(d) if d else 0.0, 3)) for t, d in candidates],
                "gt_in_cands": gt_player in cand_tids,
                "gt_rank": (cand_tids.index(gt_player) + 1) if gt_player in cand_tids else None,
            })

    print()
    print(f"{'cohort':<10} {'CORRECT':>8} {'EMPTY':>6} {'GT_OUT':>7} {'WRONG_RANK':>10} {'NO_SERVE':>9} {'TOTAL':>6}")
    for cohort in ("60fps", "30fps"):
        c = by_cohort[cohort]
        total = sum(c.values())
        correct = c.get("CORRECT", 0)
        empty = c.get("EMPTY_CANDIDATES", 0)
        gt_out = c.get("GT_NOT_IN_CANDS", 0)
        wrong = c.get("WRONG_RANK", 0)
        no_serve = c.get("NO_SERVE_DETECTED", 0)
        acc = correct / total * 100 if total else 0
        print(
            f"{cohort:<10} {correct:>8d} {empty:>6d} {gt_out:>7d} {wrong:>10d} {no_serve:>9d} {total:>6d}"
            f"  attrib_acc={acc:.1f}%"
        )

    print()
    print("Sample failures per cohort per class:")
    for key in sorted(samples.keys()):
        print(f"\n--- {key} ({len(samples[key])} samples) ---")
        for s in samples[key]:
            print(
                f"  {s['name']:8s} {s['rally_id']} frame={s['gt_frame']} "
                f"GT={s['gt_player']} picked={s['picked']} "
                f"gt_rank={s['gt_rank']} cands={s['candidates']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
