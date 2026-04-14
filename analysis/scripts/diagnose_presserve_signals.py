"""W1d: Pre-serve signals — frame-0 baseline player, toss apex, backward ball extrapolation.

Three new predictors, all read-only, all using existing positions_json /
ball_positions_json data with GT side-switch correction.

1. P_baseline_player_frame0: at rally frame 0, pick the player with the
   largest |y - court_split_y| (furthest from the net). Their court side
   (via median y over first 30 frames), side-switch-corrected, → team.
2. P_parabola_apex: fit y(frame) = a*frame^2 + b*frame + c over the first
   15 valid ball detections; the apex is argmin(y). The ball's side BEFORE
   the apex = server's side. (Ball rises from server's hand then falls.)
3. P_backward_extrapolate: linearly extrapolate the first 3 ball positions
   back to frame 0. The extrapolated y → side → team.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.diagnose_player_at_serve import (  # noqa: E402
    RallyData,
    _early_ball,
    _load_all,
    _side_to_team,
    _track_sides,
)


def p_baseline_player_frame0(rally: RallyData) -> str | None:
    """Player with max |y - court_split_y| at frame 0 is the server."""
    if rally.court_split_y is None:
        return None
    # Frame 0 positions (or closest available frame)
    frames = sorted({p.get("frameNumber", 0) for p in rally.positions})
    if not frames:
        return None
    f0 = frames[0]
    at_f0 = [p for p in rally.positions if p.get("frameNumber") == f0]
    if not at_f0:
        return None
    sides = _track_sides(rally)
    best_tid = None
    best_dist = -1.0
    for p in at_f0:
        tid = p.get("trackId")
        if tid not in sides:
            continue
        py = float(p.get("y", 0))
        dist = abs(py - rally.court_split_y)
        if dist > best_dist:
            best_dist = dist
            best_tid = tid
    if best_tid is None:
        return None
    return _side_to_team(sides[best_tid], rally.side_flipped)


def p_parabola_apex(rally: RallyData) -> str | None:
    """Fit parabola to ball y(frame) over first 15 valid points; ball side
    BEFORE the apex (highest point, min y) is the server's side.
    """
    if rally.court_split_y is None:
        return None
    pts = _early_ball(rally.ball_positions, n=15)
    if len(pts) < 5:
        return None

    frames = [p.get("frameNumber", 0) for p in pts]
    ys = [float(p["y"]) for p in pts]

    # Fit quadratic: y = a*f^2 + b*f + c
    n = len(frames)
    sum_f = sum(frames)
    sum_f2 = sum(f * f for f in frames)
    sum_f3 = sum(f ** 3 for f in frames)
    sum_f4 = sum(f ** 4 for f in frames)
    sum_y = sum(ys)
    sum_fy = sum(f * y for f, y in zip(frames, ys))
    sum_f2y = sum(f * f * y for f, y in zip(frames, ys))

    # Normal equations (3x3)
    # | sum_f4  sum_f3  sum_f2 | |a|   |sum_f2y|
    # | sum_f3  sum_f2  sum_f  | |b| = |sum_fy |
    # | sum_f2  sum_f   n      | |c|   |sum_y  |
    m = [
        [sum_f4, sum_f3, sum_f2, sum_f2y],
        [sum_f3, sum_f2, sum_f, sum_fy],
        [sum_f2, sum_f, n, sum_y],
    ]
    # Gaussian elimination
    for i in range(3):
        # pivot
        if abs(m[i][i]) < 1e-12:
            return None
        for j in range(i + 1, 3):
            factor = m[j][i] / m[i][i]
            for k in range(i, 4):
                m[j][k] -= factor * m[i][k]
    # Back-substitute
    try:
        a = b = c = 0.0
        c = m[2][3] / m[2][2]
        b = (m[1][3] - m[1][2] * c) / m[1][1]
        a = (m[0][3] - m[0][1] * b - m[0][2] * c) / m[0][0]
    except ZeroDivisionError:
        return None

    if a <= 0:
        # Parabola opening downward (a<0 in y=a*f^2+b*f+c means max, not min)
        # Image y INCREASES downward, so ball toss (going UP image-wise) means
        # y decreases then increases → parabola opens UP (a > 0). If a<=0,
        # this isn't a toss signature; fall back to first point's side.
        y_first = ys[0]
        side = "near" if y_first > rally.court_split_y else "far"
        return _side_to_team(side, rally.side_flipped)

    # Apex frame = -b / (2a)
    apex_f = -b / (2 * a)
    # Pre-apex points
    pre = [(f, y) for f, y in zip(frames, ys) if f < apex_f]
    if not pre:
        # Ball first detected AFTER the apex. Use post-apex points inverted:
        # server was on the OPPOSITE side of where the ball is descending.
        post_median_y = sorted([y for _, y in zip(frames, ys)])[len(ys) // 2]
        side = "far" if post_median_y > rally.court_split_y else "near"
        return _side_to_team(side, rally.side_flipped)
    pre_ys = sorted(y for _, y in pre)
    median_y = pre_ys[len(pre_ys) // 2]
    side = "near" if median_y > rally.court_split_y else "far"
    return _side_to_team(side, rally.side_flipped)


def p_backward_extrapolate(rally: RallyData) -> str | None:
    """Linearly extrapolate the first 3 ball points back to frame 0."""
    if rally.court_split_y is None:
        return None
    pts = _early_ball(rally.ball_positions, n=3)
    if len(pts) < 2:
        return None
    f0 = pts[0].get("frameNumber", 0)
    f1 = pts[-1].get("frameNumber", 0)
    if f1 == f0:
        return None
    y0 = float(pts[0]["y"])
    y1 = float(pts[-1]["y"])
    dy_df = (y1 - y0) / (f1 - f0)
    # Extrapolate backward to frame 0 of the rally (which is frame 0 in this
    # coordinate system)
    y_at_0 = y0 - dy_df * f0
    side = "near" if y_at_0 > rally.court_split_y else "far"
    return _side_to_team(side, rally.side_flipped)


def p_least_moving_player(rally: RallyData) -> str | None:
    """Server is the player who stands stillest during the pre-ball-detection
    window [0, first_ball_frame-5]. Receivers move into formation; partner
    moves toward the net; server holds the ball stationary.
    """
    if rally.court_split_y is None:
        return None
    from scripts.diagnose_player_at_serve import _first_valid_ball
    ball = _first_valid_ball(rally.ball_positions)
    if ball is None:
        return None
    first_f = ball.get("frameNumber", 0) or 0
    window_end = max(5, first_f - 3)
    # Gather positions per track in the window
    by_track: dict[int, list[tuple[float, float]]] = {}
    for p in rally.positions:
        f = p.get("frameNumber", 0)
        if f > window_end:
            continue
        tid = p.get("trackId")
        if tid is None or tid < 0:
            continue
        by_track.setdefault(tid, []).append((float(p.get("x", 0)), float(p.get("y", 0))))
    if len(by_track) < 2:
        return None
    # Variance per track
    sides = _track_sides(rally)
    best_tid = None
    best_var = float("inf")
    for tid, pts in by_track.items():
        if tid not in sides or len(pts) < 3:
            continue
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        var = sum((p[0]-mx)**2 + (p[1]-my)**2 for p in pts) / len(pts)
        if var < best_var:
            best_var = var
            best_tid = tid
    if best_tid is None:
        return None
    return _side_to_team(sides[best_tid], rally.side_flipped)


def p_most_moving_player(rally: RallyData) -> str | None:
    """Opposite hypothesis: server is the MOST moving player during the window
    (serve toss + strike motion), while receivers are braced and still.
    """
    if rally.court_split_y is None:
        return None
    from scripts.diagnose_player_at_serve import _first_valid_ball
    ball = _first_valid_ball(rally.ball_positions)
    if ball is None:
        return None
    first_f = ball.get("frameNumber", 0) or 0
    window_end = max(5, first_f + 10)  # include serve strike motion
    by_track: dict[int, list[tuple[float, float]]] = {}
    for p in rally.positions:
        f = p.get("frameNumber", 0)
        if f > window_end:
            continue
        tid = p.get("trackId")
        if tid is None or tid < 0:
            continue
        by_track.setdefault(tid, []).append((float(p.get("x", 0)), float(p.get("y", 0))))
    if len(by_track) < 2:
        return None
    sides = _track_sides(rally)
    best_tid = None
    best_var = -1.0
    for tid, pts in by_track.items():
        if tid not in sides or len(pts) < 3:
            continue
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        var = sum((p[0]-mx)**2 + (p[1]-my)**2 for p in pts) / len(pts)
        if var > best_var:
            best_var = var
            best_tid = tid
    if best_tid is None:
        return None
    return _side_to_team(sides[best_tid], rally.side_flipped)


PREDICTORS = [
    ("P_baseline_player_frame0", p_baseline_player_frame0),
    ("P_parabola_apex", p_parabola_apex),
    ("P_backward_extrapolate", p_backward_extrapolate),
    ("P_least_moving_player", p_least_moving_player),
    ("P_most_moving_player", p_most_moving_player),
]


def main() -> int:
    video_rallies = _load_all()
    total = sum(len(v) for v in video_rallies.values())
    print(f"rallies={total}")

    class_counts = defaultdict(int)
    for rs in video_rallies.values():
        for r in rs:
            class_counts[r.gt_serving_team] += 1
    floor = max(class_counts.values()) / total
    print(f"class balance: {dict(class_counts)}  floor={floor*100:.1f}%\n")

    print(f"{'predictor':30s}  {'acc':>10s}  {'abstain':>8s}")
    print("-" * 58)
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
        print(f"{name:30s}  {correct:3d}/{scored:3d}={acc*100:5.1f}%  {abstain:4d}")

    # Per-video on best
    best = max(results, key=lambda r: r[1])
    print(f"\nBest: {best[0]} = {best[1]*100:.1f}%")
    print("Per-video:")
    for vid, (c, t) in sorted(best[5].items(), key=lambda kv: -kv[1][0]/max(1, kv[1][1])):
        print(f"  {vid[:8]}  {c:3d}/{t:3d}  {c/max(1,t)*100:5.1f}%")

    # Per-video oracle convention check (same trick as before)
    print("\nPer-video oracle convention (max of identity vs flipped) for best:")
    fn_best = dict(PREDICTORS)[best[0]]
    total_oracle = 0
    for vid, rs in video_rallies.items():
        n_identity = 0
        n_flipped = 0
        scored = 0
        for r in rs:
            pred = fn_best(r)
            if pred is None:
                continue
            scored += 1
            if pred == r.gt_serving_team:
                n_identity += 1
            inverted = "B" if pred == "A" else "A"
            if inverted == r.gt_serving_team:
                n_flipped += 1
        best_v = max(n_identity, n_flipped)
        total_oracle += best_v
        print(f"  {vid[:8]}  identity={n_identity}/{scored}  flipped={n_flipped}/{scored}  best={best_v}")
    print(f"Aggregate oracle: {total_oracle}/{total} = {total_oracle/total*100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
