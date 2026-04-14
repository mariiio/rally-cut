"""W1c: Player-closest-to-ball at first ball frame → team.

Tests whether player positions (positions_json) + ball position (ball_positions_json)
at rally start, combined with GT side switches, give a cleaner signal for
`gt_serving_team` than anything we've tested so far.

Architecture:
  1. For each rally, find the first ball frame (frameNumber where ball is
     validly detected).
  2. Classify each of the 4 tracks into 'near' or 'far' court side via median
     Y over the full rally (same convention as player_filter.classify_teams:
     y > court_split_y → near, y < → far).
  3. At the first ball frame, find the player closest to the ball by (x, y)
     Euclidean distance. That player is the most likely server.
  4. Apply GT sideSwitches: if this rally is on the flipped side, swap
     near ↔ far, which swaps A ↔ B.
  5. Emit team: near → 'A', far → 'B'.

Also tests variants:
  - Player on the ball's side (not closest player)
  - Player farthest from net on the ball's side (likely serving from baseline)
  - Ball-direction signal (dy over first 5 frames) → server is on the side
    the ball is moving away from.

Read-only. No DB writes.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402


@dataclass
class RallyData:
    rally_id: str
    video_id: str
    start_ms: int
    gt_serving_team: str
    positions: list[dict]
    ball_positions: list[dict]
    court_split_y: float | None
    fps: float
    rally_index: int = 0
    side_flipped: bool = False


def _load_all() -> dict[str, list[RallyData]]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, player_matching_gt_json FROM videos
            WHERE id IN (SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL)
        """)
        video_switches: dict[str, set[int]] = {}
        for vid, gt in cur.fetchall():
            sw = list(gt.get("sideSwitches", [])) if isinstance(gt, dict) else []
            video_switches[vid] = set(sw)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.video_id, r.start_ms, r.gt_serving_team,
                   pt.positions_json, pt.ball_positions_json, pt.court_split_y, pt.fps
            FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id IN (SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL)
              AND r.gt_serving_team IS NOT NULL
            ORDER BY r.video_id, r.start_ms
        """)
        raw = defaultdict(list)
        for row in cur.fetchall():
            raw[row[1]].append(row)

    out: dict[str, list[RallyData]] = {}
    for vid, rows in raw.items():
        rows.sort(key=lambda r: r[2])
        switches = video_switches.get(vid, set())
        flipped = False
        vid_out = []
        for idx, (rid, _, sms, gt, pj, bpj, split_y, fps) in enumerate(rows):
            if idx in switches:
                flipped = not flipped
            vid_out.append(RallyData(
                rally_id=rid,
                video_id=vid,
                start_ms=sms or 0,
                gt_serving_team=gt,
                positions=pj or [],
                ball_positions=bpj or [],
                court_split_y=split_y,
                fps=fps or 30.0,
                rally_index=idx,
                side_flipped=flipped,
            ))
        out[vid] = vid_out
    return out


# ---------- Helpers ----------------------------------------------------


def _first_valid_ball(bpj: list[dict]) -> dict | None:
    for b in sorted(bpj, key=lambda x: x.get("frameNumber", 0)):
        x = b.get("x", 0) or 0
        y = b.get("y", 0) or 0
        conf = b.get("confidence", 1.0)
        if conf is not None and conf <= 0:
            continue
        if x <= 0 and y <= 0:
            continue
        return b
    return None


def _early_ball(bpj: list[dict], n: int = 10) -> list[dict]:
    valid = [
        b for b in sorted(bpj, key=lambda x: x.get("frameNumber", 0))
        if (b.get("x", 0) or 0) > 0 and (b.get("y", 0) or 0) > 0
        and (b.get("confidence", 1.0) or 1.0) > 0
    ]
    return valid[:n]


def _track_sides(rally: RallyData) -> dict[int, str]:
    """Median-y per track → 'near' or 'far' via court_split_y."""
    if rally.court_split_y is None:
        return {}
    by_track: dict[int, list[float]] = defaultdict(list)
    for p in rally.positions:
        tid = p.get("trackId")
        if tid is None or tid < 0:
            continue
        by_track[tid].append(float(p.get("y", 0)))
    sides: dict[int, str] = {}
    for tid, ys in by_track.items():
        if not ys:
            continue
        ys.sort()
        median = ys[len(ys) // 2]
        sides[tid] = "near" if median > rally.court_split_y else "far"
    return sides


def _side_to_team(side: str, flipped: bool) -> str:
    """near → A, far → B, unless flipped."""
    if side == "near":
        base = "A"
    else:
        base = "B"
    if flipped:
        return "B" if base == "A" else "A"
    return base


# ---------- Predictors ------------------------------------------------


def p_closest_player(rally: RallyData) -> str | None:
    ball = _first_valid_ball(rally.ball_positions)
    if ball is None or rally.court_split_y is None:
        return None
    f = ball.get("frameNumber", 0)
    bx, by = float(ball["x"]), float(ball["y"])
    # Find positions at that frame (or nearest)
    at_f = [p for p in rally.positions if p.get("frameNumber") == f]
    if not at_f:
        # Use the closest-in-time positions instead
        f_avail = sorted(set(p.get("frameNumber", 0) for p in rally.positions))
        if not f_avail:
            return None
        nearest_f = min(f_avail, key=lambda x: abs(x - f))
        at_f = [p for p in rally.positions if p.get("frameNumber") == nearest_f]
    if not at_f:
        return None
    sides = _track_sides(rally)
    best_tid = None
    best_d = float("inf")
    for p in at_f:
        tid = p.get("trackId")
        if tid not in sides:
            continue
        px = float(p.get("x", 0)) + float(p.get("width", 0)) / 2
        py = float(p.get("y", 0)) + float(p.get("height", 0)) / 2
        d = math.hypot(px - bx, py - by)
        if d < best_d:
            best_d = d
            best_tid = tid
    if best_tid is None:
        return None
    return _side_to_team(sides[best_tid], rally.side_flipped)


def p_ball_side_early(rally: RallyData) -> str | None:
    """Ball's court side at first detection → team (with side switch)."""
    ball = _first_valid_ball(rally.ball_positions)
    if ball is None or rally.court_split_y is None:
        return None
    side = "near" if float(ball["y"]) > rally.court_split_y else "far"
    return _side_to_team(side, rally.side_flipped)


def p_ball_direction(rally: RallyData) -> str | None:
    """Ball dy over first ~10 valid frames. Server is on the side the ball
    is moving AWAY from. If dy > 0 (moving toward near), server was far.
    """
    pts = _early_ball(rally.ball_positions, n=10)
    if len(pts) < 3 or rally.court_split_y is None:
        return None
    y_first = float(pts[0]["y"])
    y_last = float(pts[-1]["y"])
    dy = y_last - y_first
    # If dy > 0 (ball moving down-image, toward near court), server is far
    # If dy < 0 (ball moving up-image, toward far court), server is near
    side = "far" if dy > 0 else "near"
    return _side_to_team(side, rally.side_flipped)


def p_baseline_player_on_ball_side(rally: RallyData) -> str | None:
    """Player with the most extreme Y on the ball's side (likely the server
    standing at the baseline).
    """
    ball = _first_valid_ball(rally.ball_positions)
    if ball is None or rally.court_split_y is None:
        return None
    ball_side = "near" if float(ball["y"]) > rally.court_split_y else "far"
    sides = _track_sides(rally)
    candidates = [tid for tid, s in sides.items() if s == ball_side]
    if not candidates:
        return None
    # For "near" side, the baseline is MAX y; for "far" side, it's MIN y.
    # Use median y per candidate over first 30 frames.
    early_frames = sorted({p.get("frameNumber", 0) for p in rally.positions})[:30]
    early_set = set(early_frames)
    medians: dict[int, float] = {}
    for tid in candidates:
        ys = [float(p.get("y", 0)) for p in rally.positions
              if p.get("trackId") == tid and p.get("frameNumber") in early_set]
        if ys:
            ys.sort()
            medians[tid] = ys[len(ys) // 2]
    if not medians:
        return None
    if ball_side == "near":
        # baseline player = max y
        best = max(medians, key=lambda t: medians[t])
    else:
        best = min(medians, key=lambda t: medians[t])
    return _side_to_team(sides[best], rally.side_flipped)


PREDICTORS = [
    ("P_closest_player", p_closest_player),
    ("P_ball_side_early", p_ball_side_early),
    ("P_ball_direction", p_ball_direction),
    ("P_baseline_on_ball_side", p_baseline_player_on_ball_side),
]


# ---------- Eval ------------------------------------------------------


def main() -> int:
    video_rallies = _load_all()
    total = sum(len(v) for v in video_rallies.values())
    print(f"videos={len(video_rallies)} rallies={total}")
    print(f"Baseline (production score_accuracy): 46.2%  Gate: >= 85%\n")

    class_counts = defaultdict(int)
    for rs in video_rallies.values():
        for r in rs:
            class_counts[r.gt_serving_team] += 1
    print(f"GT class balance: {dict(class_counts)}  "
          f"floor={max(class_counts.values())/total*100:.1f}%\n")

    # Evaluate each predictor
    print(f"{'predictor':28s}  {'acc':>10s}  {'abstain':>8s}")
    print("-" * 55)
    results = []
    for name, fn in PREDICTORS:
        correct = 0
        scored = 0
        abstain = 0
        per_video: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for vid, rs in video_rallies.items():
            for r in rs:
                scored += 1
                pred = fn(r)
                if pred is None:
                    abstain += 1
                    continue
                per_video[vid][1] += 1
                if pred == r.gt_serving_team:
                    correct += 1
                    per_video[vid][0] += 1
        acc = correct / max(1, scored)
        results.append((name, acc, correct, scored, abstain, per_video))
        print(f"{name:28s}  {correct:3d}/{scored:3d}={acc*100:5.1f}%  {abstain:4d}")

    # Per-video breakdown for the best predictor
    best = max(results, key=lambda r: r[1])
    print(f"\nBest: {best[0]} = {best[1]*100:.1f}%")
    print(f"Per-video breakdown:")
    for vid, (c, t) in sorted(best[5].items(), key=lambda kv: kv[1][0]/max(1,kv[1][1])):
        pct = c / max(1, t) * 100
        print(f"  {vid[:8]}  {c:3d}/{t:3d}  {pct:5.1f}%")

    print()
    if best[1] >= 0.85:
        print(f"GO: {best[0]} at {best[1]*100:.1f}% clears 85% gate.")
    elif best[1] >= 0.70:
        print(f"MAYBE: {best[0]} at {best[1]*100:.1f}% — promising, deserves a fallback-combo try.")
    else:
        print(f"NO-GO: best {best[0]} = {best[1]*100:.1f}% still at/near majority floor. "
              "Upstream positions+ball signal confirms no per-rally serving signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
