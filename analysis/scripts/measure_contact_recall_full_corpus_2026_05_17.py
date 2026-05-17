#!/usr/bin/env python3
"""Measure contact-detection recall on the full action-GT corpus (74 videos,
2775 GT rows). Action-type matching only — doesn't require player attribution
GT, so we can use the videos beyond trusted-29.

This is the right metric for contact-detection improvements (like the
direction-change-max snap): does the pipeline place a contact close enough
to GT that the action_type can be matched?
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import psycopg

DB = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
MATCH_WINDOW = 5


def main() -> int:
    by_action: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "matched": 0, "matched_type": 0}
    )
    n_total = 0
    n_matched_frame = 0
    n_matched_type = 0
    with psycopg.connect(DB) as conn:
        cur = conn.execute(
            """
            SELECT rg.action::text, rg.frame, pt.actions_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            """,
        )
        for gt_action, gt_f, aj in cur.fetchall():
            aj = aj if isinstance(aj, dict) else (json.loads(aj) if aj else {})
            actions = (aj or {}).get("actions") or []
            n_total += 1
            by_action[gt_action.upper()]["total"] += 1
            # Match: within ±MATCH_WINDOW frames
            matched_any = False
            matched_type = False
            for a in actions:
                d = abs(int(a.get("frame", -10**9)) - gt_f)
                if d <= MATCH_WINDOW:
                    matched_any = True
                    if (a.get("action") or "").upper() == gt_action.upper():
                        matched_type = True
                        break
            if matched_any:
                n_matched_frame += 1
                by_action[gt_action.upper()]["matched"] += 1
            if matched_type:
                n_matched_type += 1
                by_action[gt_action.upper()]["matched_type"] += 1

    print(f"Full action-GT corpus contact recall:")
    print(f"  Total GT rows: {n_total}")
    print(f"  Matched within ±{MATCH_WINDOW}f (any action_type): "
          f"{n_matched_frame}/{n_total} = {n_matched_frame/max(1,n_total)*100:.1f}%")
    print(f"  Matched + same action_type: "
          f"{n_matched_type}/{n_total} = {n_matched_type/max(1,n_total)*100:.1f}%")
    print()
    print(f"{'Action':<10s}{'GT':>6s}{'matched':>10s}{'recall':>10s}{'+ type':>10s}{'type%':>10s}")
    for action in ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"):
        d = by_action.get(action, {"total": 0, "matched": 0, "matched_type": 0})
        if d["total"] == 0:
            continue
        recall = d["matched"] / d["total"] * 100
        type_pct = d["matched_type"] / d["total"] * 100
        print(f"{action:<10s}{d['total']:>6d}{d['matched']:>10d}{recall:>9.1f}%{d['matched_type']:>10d}{type_pct:>9.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
