"""Compare structural features at TP block candidates vs FP block candidates.

For each pipeline ATTACK action with MS-TCN++ block-class peak >= 0.50
in [+1, +10], extract:

  - distance from BALL to nearest player at peak_frame
  - team of nearest player vs attacker's team (opposite/same/unknown)
  - distance from peak_frame's NEAREST PLAYER's center to net_y (closer = more
    likely at net)
  - peak player's bbox top y (smaller = higher in frame, jumping)
  - peak player's bbox height
  - ball direction-change angle at peak_frame (deflection)

Then bucket TPs (within ±15 of GT block) vs FPs and print signal statistics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOL = 15
BLOCK_IDX = ACTION_TYPES.index("block") + 1
PEAK_FLOOR = 0.50
MAX_GAP = 10


def _bp(j: Any) -> list[BallPosition]:
    return [BallPosition(int(b["frameNumber"]), float(b["x"]), float(b["y"]),
                         float(b.get("confidence", 0)))
            for b in (j or []) if isinstance(b, dict)]


def _pp(j: Any) -> list[PlayerPosition]:
    return [PlayerPosition(int(p["frameNumber"]), int(p["trackId"]),
                            float(p["x"]), float(p["y"]),
                            float(p["width"]), float(p["height"]),
                            float(p.get("confidence", 0)))
            for p in (j or []) if isinstance(p, dict)]


def _nearest_player_at_frame(
    pp_list: list[PlayerPosition], bx: float, by: float, frame: int,
) -> PlayerPosition | None:
    candidates = [p for p in pp_list if p.frame_number == frame]
    if not candidates:
        return None
    best = min(candidates, key=lambda p:
               ((p.x + p.width/2 - bx)**2 + (p.y + p.height/2 - by)**2)**0.5)
    return best


def _ball_at(bp_list: list[BallPosition], frame: int) -> tuple[float, float] | None:
    by_frame = {b.frame_number: b for b in bp_list}
    if frame in by_frame:
        b = by_frame[frame]
        if b.x > 0.01 or b.y > 0.01:
            return (b.x, b.y)
    for d in range(1, 6):
        for f in (frame - d, frame + d):
            b = by_frame.get(f)
            if b and (b.x > 0.01 or b.y > 0.01):
                return (b.x, b.y)
    return None


def _ball_direction_change(
    bp_list: list[BallPosition], frame: int,
) -> float:
    """Return absolute direction change (degrees) in trajectory at `frame`."""
    by_frame = {b.frame_number: b for b in bp_list}
    pts = []
    for f in [frame - 4, frame - 2, frame, frame + 2, frame + 4]:
        b = by_frame.get(f)
        if b and (b.x > 0.01 or b.y > 0.01):
            pts.append((b.x, b.y))
    if len(pts) < 3:
        return 0.0
    v1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    v2 = (pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])
    n1 = (v1[0]**2 + v1[1]**2)**0.5
    n2 = (v2[0]**2 + v2[1]**2)**0.5
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)
    hashes = {r["video_content_hash"] for r in gt["rallies"]}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos WHERE content_hash = ANY(%s)",
                [list(hashes)],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    h2v = {h: v for v, (h, _) in meta.items()}

    tps: list[dict[str, float]] = []
    fps: list[dict[str, float]] = []

    with get_connection() as conn:
        for r in gt["rallies"]:
            ch = r["video_content_hash"]
            if ch not in h2v:
                continue
            vid = h2v[ch]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT pt.frame_count, pt.court_split_y,
                              pt.ball_positions_json, pt.positions_json, pt.actions_json
                       FROM rallies rr JOIN player_tracks pt ON pt.rally_id=rr.id
                       WHERE rr.video_id=%s AND rr.start_ms=%s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if not row:
                continue
            fc, csy, bp_j, pp_j, aj = row
            if not aj or csy is None:
                continue
            actions = sorted(aj.get("actions") or [], key=lambda a: a.get("frame", 0))
            ta = (aj or {}).get("teamAssignments") or {}
            if not ta:
                continue
            bp_list = _bp(bp_j)
            pp_list = _pp(pp_j)
            gt_blocks = {int(a.get("frame", 0))
                         for a in r.get("action_ground_truth_json", []) or []
                         if a.get("action") == "block"}
            ta_int = {int(k): (0 if v == "A" else 1) for k, v in ta.items() if v in ("A","B")}
            seq = get_sequence_probs(bp_list, pp_list, csy, fc or 0, ta_int, calibrator=None)
            if seq is None:
                continue

            for a in actions:
                if a.get("action") != "attack":
                    continue
                af = int(a.get("frame", 0))
                attacker_team = a.get("team")
                lo, hi = af + 1, min(seq.shape[1], af + MAX_GAP + 1)
                if hi <= lo:
                    continue
                win = seq[BLOCK_IDX, lo:hi]
                if win.size == 0 or float(win.max()) < PEAK_FLOOR:
                    continue
                peak_f = lo + int(win.argmax())
                peak_p = float(win[peak_f - lo])

                ball_at_peak = _ball_at(bp_list, peak_f)
                if not ball_at_peak:
                    continue
                bx, by = ball_at_peak

                near_player = _nearest_player_at_frame(pp_list, bx, by, peak_f)
                if not near_player:
                    continue
                px = near_player.x + near_player.width/2
                py = near_player.y + near_player.height/2
                ball_player_dist = ((bx - px)**2 + (by - py)**2)**0.5
                player_team = "A" if ta_int.get(near_player.track_id) == 0 else (
                    "B" if ta_int.get(near_player.track_id) == 1 else None
                )
                opposite_team = (
                    1 if (attacker_team and player_team and attacker_team != player_team)
                    else 0
                )
                player_y_to_net = abs(py - csy)
                player_bbox_top = near_player.y
                ball_y_above_net = csy - by  # positive = ball is above net (smaller y)
                dc_deg = _ball_direction_change(bp_list, peak_f)

                is_tp = any(abs(peak_f - gb) <= HIT_TOL for gb in gt_blocks)
                rec = {
                    "peak_p": peak_p,
                    "ball_player_dist": ball_player_dist,
                    "opposite_team": opposite_team,
                    "player_y_to_net": player_y_to_net,
                    "player_bbox_top": player_bbox_top,
                    "ball_y_above_net": ball_y_above_net,
                    "direction_change_deg": dc_deg,
                }
                (tps if is_tp else fps).append(rec)

    def _stat(rows: list[dict[str, float]], key: str) -> tuple[float, float, float, float]:
        vals = [r[key] for r in rows]
        if not vals:
            return (0.0, 0.0, 0.0, 0.0)
        a = sorted(vals)
        return (min(a), a[len(a)//4], a[len(a)//2], a[3*len(a)//4]), max(a)

    print(f"TPs: {len(tps)}    FPs: {len(fps)}\n")
    print(f"{'signal':<26} {'TP min/Q1/med/Q3/max':<40} {'FP min/Q1/med/Q3/max':<40}")
    for key in ["peak_p", "ball_player_dist", "opposite_team",
                "player_y_to_net", "player_bbox_top",
                "ball_y_above_net", "direction_change_deg"]:
        tp_stat = sorted([r[key] for r in tps]) if tps else []
        fp_stat = sorted([r[key] for r in fps]) if fps else []
        def _q(a: list[float]) -> str:
            if not a:
                return "-"
            return (
                f"{a[0]:.2f}/{a[len(a)//4]:.2f}/"
                f"{a[len(a)//2]:.2f}/"
                f"{a[3*len(a)//4]:.2f}/{a[-1]:.2f}"
            )
        print(f"{key:<26} {_q(tp_stat):<40} {_q(fp_stat):<40}")

    # Apply candidate combined filters
    print("\n\n=== Candidate combined-gate sweep ===")
    print(f"{'gate description':<60} {'TPs':>5} {'FPs':>5}")
    gates = [
        ("opposite_team only",
         lambda r: r["opposite_team"] == 1),
        ("opposite_team + ball_above_net>=0.05",
         lambda r: r["opposite_team"] == 1 and r["ball_y_above_net"] >= 0.05),
        ("opposite_team + ball_above_net>=0.10",
         lambda r: r["opposite_team"] == 1 and r["ball_y_above_net"] >= 0.10),
        ("opposite_team + ball_player_dist<=0.10",
         lambda r: r["opposite_team"] == 1 and r["ball_player_dist"] <= 0.10),
        ("opposite_team + ball_above_net>=0.10 + ball_player_dist<=0.15",
         lambda r: r["opposite_team"] == 1 and r["ball_y_above_net"] >= 0.10
                   and r["ball_player_dist"] <= 0.15),
        ("opposite_team + ball_above_net>=0.10 + dc>=30deg",
         lambda r: r["opposite_team"] == 1 and r["ball_y_above_net"] >= 0.10
                   and r["direction_change_deg"] >= 30),
        ("opposite_team + ball_above_net>=0.10 + dc>=60deg",
         lambda r: r["opposite_team"] == 1 and r["ball_y_above_net"] >= 0.10
                   and r["direction_change_deg"] >= 60),
        ("opposite_team + ball_above_net>=0.10 + player_to_net<=0.20",
         lambda r: r["opposite_team"] == 1 and r["ball_y_above_net"] >= 0.10
                   and r["player_y_to_net"] <= 0.20),
        ("opposite_team + ball_above_net>=0.10 + ball_player_dist<=0.15 + dc>=30",
         lambda r: r["opposite_team"] == 1 and r["ball_y_above_net"] >= 0.10
                   and r["ball_player_dist"] <= 0.15
                   and r["direction_change_deg"] >= 30),
    ]
    for label, gate in gates:
        tps_pass = sum(1 for r in tps if gate(r))
        fps_pass = sum(1 for r in fps if gate(r))
        print(f"{label:<60} {tps_pass:>5} {fps_pass:>5}")


if __name__ == "__main__":
    main()
