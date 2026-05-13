"""Per-error inspection of wawa's same_team residual errors with A1 ON.

Runs the same in-process A1 pipeline as ``measure_a1_volleyball_rule_ab.py``,
but instead of aggregating, dumps every wrong_same_team match for wawa with:
- rally_id (short), frame, predicted vs GT player + team
- the contact's full ``player_candidates`` list (so we can see if the GT
  candidate exists in the rank-K and whether the A1 abstention bound bit)
- the immediately-previous pipeline action (to characterise the
  same-team back-to-back pair shape)

The goal is to answer one question: are wawa's 8 residual same_team errors
within-team mis-attributions A1 could still catch with a tweak, or are
they upstream (cross-team team-assignment errors masquerading as
same-team in the GT comparison)?
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Re-use the harness's loaders.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_a1_volleyball_rule_ab import (  # type: ignore[import-not-found]
    FRESH_GT_VIDEOS,
    _build_rally_remap,
    _reconstruct_actions,
    _reconstruct_contacts,
)

from rallycut.evaluation.attribution_bench import score_rally
from rallycut.evaluation.db import get_connection
from rallycut.tracking.action_classifier import reattribute_players
from rallycut.training.action_gt_query import load_for_videos


def main() -> None:
    wawa_id = FRESH_GT_VIDEOS["wawa"]

    # Load wawa rallies + GT
    with get_connection() as conn:
        gt_by_rally = load_for_videos(conn, [wawa_id])
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id, pt.actions_json, pt.contacts_json, v.match_analysis_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                JOIN videos v ON v.id = r.video_id
                WHERE r.video_id = %s
                ORDER BY r.start_ms
                """,
                [wawa_id],
            )
            rows = cur.fetchall()

    # Toggle A1 ON.
    os.environ["USE_VOLLEYBALL_RULE_ATTRIBUTION"] = "1"

    n_inspected = 0
    same_team_errors: list[dict[str, Any]] = []

    for rid, actions_json, contacts_json, match_json in rows:
        gt_actions = gt_by_rally.get(rid, [])
        if not gt_actions:
            continue
        contacts = _reconstruct_contacts(contacts_json or {})
        actions = _reconstruct_actions(actions_json or {})
        team_assignments_str = (actions_json or {}).get("teamAssignments") or {}
        # Convert "1":"A" → {1:0, 2:0, ...} mapping the same way the pass expects.
        team_assignments_int: dict[int, int] = {}
        for k, v in team_assignments_str.items():
            try:
                tid = int(k)
            except ValueError:
                continue
            if v == "A":
                team_assignments_int[tid] = 0
            elif v == "B":
                team_assignments_int[tid] = 1

        # Re-attribute with A1 ON
        reattribute_players(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments_int,
        )

        # Serialize back to dicts for score_rally
        pipeline_actions = [
            {
                "frame": a.frame,
                "action": a.action_type.value,
                "playerTrackId": a.player_track_id,
                "confidence": a.confidence,
                "courtSide": a.court_side,
                "isSynthetic": a.is_synthetic,
                "attributionUncertain": a.attribution_uncertain,
            }
            for a in actions
        ]

        rally = {
            "fixture": "wawa",
            "gt_actions": gt_actions,
            "pipeline_actions": pipeline_actions,
            "team_assignments": team_assignments_str,
        }
        scored = score_rally(rally)
        n_inspected += 1

        # Sort actions by frame for prev-action lookup
        sorted_pipeline = sorted(pipeline_actions, key=lambda a: a["frame"])
        contact_by_frame = {c.frame: c for c in contacts}

        for m in scored["matches"]:
            if m["category"] != "wrong_same_team":
                continue
            # Find the prev pipeline action (same-team back-to-back diagnosis)
            pl_frame = m["pl_frame"]
            prev_action = None
            for pa in sorted_pipeline:
                if pa["frame"] < pl_frame:
                    prev_action = pa
                elif pa["frame"] >= pl_frame:
                    break
            contact = contact_by_frame.get(pl_frame)
            cands = contact.player_candidates if contact else []
            gt_pid = m["gt_pid"]
            pl_pid = m["pl_pid"]
            # GT candidate rank + distance
            gt_in_cands = next(
                (i for i, (tid, _d) in enumerate(cands) if tid == gt_pid), -1
            )
            gt_dist = (
                cands[gt_in_cands][1] if gt_in_cands >= 0 else None
            )
            pl_in_cands = next(
                (i for i, (tid, _d) in enumerate(cands) if tid == pl_pid), -1
            )
            pl_dist = (
                cands[pl_in_cands][1] if pl_in_cands >= 0 else None
            )

            entry = {
                "rally_short": rid[:8],
                "rally_id": rid,
                "gt_frame": m["gt_frame"],
                "gt_action": m["gt_action"],
                "gt_pid": gt_pid,
                "gt_team": team_assignments_str.get(str(gt_pid)),
                "pl_frame": pl_frame,
                "pl_action": m["pl_action"],
                "pl_pid": pl_pid,
                "pl_team": team_assignments_str.get(str(pl_pid)),
                "pl_confidence": round(m["pl_confidence"] or 0.0, 3),
                "prev_action": prev_action.get("action") if prev_action else None,
                "prev_frame": prev_action.get("frame") if prev_action else None,
                "prev_pid": prev_action.get("playerTrackId") if prev_action else None,
                "prev_team": (
                    team_assignments_str.get(str(prev_action["playerTrackId"]))
                    if prev_action else None
                ),
                "candidates_topk": [
                    {
                        "tid": tid,
                        "dist": round(d, 4),
                        "team": team_assignments_str.get(str(tid)),
                    }
                    for tid, d in cands[:5]
                ],
                "gt_in_cands_rank": gt_in_cands,
                "gt_dist_to_ball": (
                    round(gt_dist, 4) if gt_dist is not None else None
                ),
                "pl_dist_to_ball": (
                    round(pl_dist, 4) if pl_dist is not None else None
                ),
                "alt_ratio_gt_over_pl": (
                    round(gt_dist / pl_dist, 2)
                    if gt_dist and pl_dist
                    else None
                ),
                "attribution_uncertain": next(
                    (pa.get("attributionUncertain", False)
                     for pa in sorted_pipeline if pa["frame"] == pl_frame),
                    False,
                ),
            }
            same_team_errors.append(entry)

    print(f"Inspected {n_inspected} wawa rallies. Found {len(same_team_errors)} wrong_same_team errors.")
    print()
    for i, e in enumerate(same_team_errors, 1):
        print(f"--- Error {i} ---")
        print(f"  rally    : {e['rally_short']}")
        print(f"  gt       : frame {e['gt_frame']} {e['gt_action']:>8} pid={e['gt_pid']} team={e['gt_team']}")
        print(f"  pl       : frame {e['pl_frame']} {e['pl_action']:>8} pid={e['pl_pid']} team={e['pl_team']} conf={e['pl_confidence']}")
        print(f"  prev     : frame {e['prev_frame']} {e['prev_action']:>8} pid={e['prev_pid']} team={e['prev_team']}")
        print(f"  same-team back-to-back? prev_team == pl_team: {e['prev_team'] == e['pl_team']}")
        print(f"  candidates (top-5): {e['candidates_topk']}")
        print(f"  GT in candidates rank: {e['gt_in_cands_rank']} (dist={e['gt_dist_to_ball']})")
        print(f"  PL in candidates rank: dist={e['pl_dist_to_ball']}")
        if e['alt_ratio_gt_over_pl']:
            print(f"  alt_ratio (gt/pl): {e['alt_ratio_gt_over_pl']}x")
        print(f"  attribution_uncertain (A1 flag): {e['attribution_uncertain']}")
        print()

    # Summary: how many fit each pattern?
    same_team_b2b = sum(1 for e in same_team_errors if e['prev_team'] == e['pl_team'])
    gt_within_abstain_bound = sum(
        1 for e in same_team_errors
        if e['gt_dist_to_ball'] is not None and e['gt_dist_to_ball'] <= 0.3
    )
    gt_rank1 = sum(1 for e in same_team_errors if e['gt_in_cands_rank'] == 0)
    print("=" * 60)
    print(f"Total errors:                            {len(same_team_errors)}")
    print(f"  same-team back-to-back (A1's domain): {same_team_b2b}")
    print(f"  GT within 0.3 abstain bound:           {gt_within_abstain_bound}")
    print(f"  GT is rank-1 in candidates:            {gt_rank1}")
    print(f"  A1 flagged attribution_uncertain:      {sum(1 for e in same_team_errors if e['attribution_uncertain'])}")


if __name__ == "__main__":
    main()
