#!/usr/bin/env python3
"""Visually render 3 cross-team attribution errors for inspection.

Picks 1 ATTACK, 1 BLOCK, 1 DIG cross-team case. Renders the frame with:
  - Ball position (yellow circle)
  - GT player bbox (green) labeled "GT tid=X team=Y"
  - Picked player bbox (red) labeled "PICKED tid=X team=Y"
  - Other player bboxes (gray)
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
OUT = Path("/tmp/cross_team_inspect")
OUT.mkdir(exist_ok=True)

# Sample cases from diagnose_wrong_attribution
CASES = [
    ("wawa", 308, "ATTACK", 4, 1),  # gt team B, picked team A
    ("gugu", 277, "DIG", 1, 3),     # gt team A, picked team B
    ("wawa", 209, "SET", 3, 2),     # gt team B, picked team A
]


def main() -> int:
    with psycopg.connect(DB_DSN) as conn:
        for vname, gt_frame, gt_action, gt_tid, picked in CASES:
            cur = conn.execute(
                """SELECT r.id, r.start_ms, v.fps, pt.actions_json,
                          pt.ball_positions_json, pt.positions_json
                FROM rallies r JOIN videos v ON r.video_id=v.id
                JOIN player_tracks pt ON pt.rally_id=r.id
                JOIN rally_action_ground_truth rg ON rg.rally_id=r.id
                WHERE v.name=%s AND rg.frame=%s AND rg.action::text=%s
                  AND rg.resolved_track_id=%s
                LIMIT 1""",
                [vname, gt_frame, gt_action, gt_tid],
            )
            row = cur.fetchone()
            if not row:
                print(f"{vname} f{gt_frame} not found")
                continue
            rid, start_ms, fps, aj, bj, pj = row
            aj = aj if isinstance(aj, dict) else json.loads(aj)
            bj = bj if isinstance(bj, list) else json.loads(bj)
            pj = pj if isinstance(pj, list) else json.loads(pj)

            teams = aj.get("teamAssignments") or {}
            gt_team = teams.get(str(gt_tid), "?")
            picked_team = teams.get(str(picked), "?")

            # Find ball at gt_frame
            ball = next((b for b in bj if b["frameNumber"] == gt_frame), None)
            if not ball or (ball.get("x", 0) == 0 and ball.get("y", 0) == 0):
                # Try nearest frame
                ball = min(
                    (b for b in bj if b["frameNumber"] is not None
                     and abs(b["frameNumber"] - gt_frame) <= 3
                     and (b.get("x", 0) > 0 or b.get("y", 0) > 0)),
                    key=lambda b: abs(b["frameNumber"] - gt_frame),
                    default=None,
                )

            # Find player bboxes at gt_frame
            players_at = [p for p in pj if p.get("frameNumber") == gt_frame]
            if not players_at:
                players_at = [p for p in pj
                              if abs(p.get("frameNumber", -9999) - gt_frame) <= 3]

            # Extract frame
            abs_t = start_ms / 1000.0 + gt_frame / fps
            tmp_jpg = OUT / f"{vname}_f{gt_frame}_raw.jpg"
            video_path = f"/tmp/rca_videos/{vname}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{abs_t:.3f}", "-i", video_path,
                 "-vframes", "1", "-q:v", "2", str(tmp_jpg)],
                check=False,
            )
            if not tmp_jpg.exists():
                print(f"  Could not extract {vname} frame {gt_frame}")
                continue
            img = cv2.imread(str(tmp_jpg))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Draw players
            for p in players_at:
                tid = p.get("trackId", -1)
                cx, cy = p.get("x", 0), p.get("y", 0)
                pw, ph = p.get("width", 0), p.get("height", 0)
                x1 = int((cx - pw / 2) * w)
                y1 = int((cy - ph / 2) * h)
                x2 = int((cx + pw / 2) * w)
                y2 = int((cy + ph / 2) * h)
                team = teams.get(str(tid), "?")
                if tid == gt_tid:
                    color, label = (0, 255, 0), f"GT tid={tid} team={team}"
                    thick = 4
                elif tid == picked:
                    color, label = (0, 0, 255), f"PICKED tid={tid} team={team}"
                    thick = 4
                else:
                    color, label = (180, 180, 180), f"tid={tid} team={team}"
                    thick = 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
                cv2.putText(img, label, (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Draw ball
            if ball:
                bx, by = int(ball["x"] * w), int(ball["y"] * h)
                cv2.circle(img, (bx, by), 14, (0, 255, 255), 3)
                cv2.putText(img, "BALL", (bx + 18, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Title
            title = f"{vname} f{gt_frame} {gt_action}  GT={gt_tid}({gt_team})  PICKED={picked}({picked_team})"
            cv2.putText(img, title, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(img, (10, 15), (10 + len(title) * 13, 50), (0, 0, 0), 1)

            out_path = OUT / f"{vname}_f{gt_frame}_{gt_action}.jpg"
            cv2.imwrite(str(out_path), img)
            tmp_jpg.unlink()
            print(f"  → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
