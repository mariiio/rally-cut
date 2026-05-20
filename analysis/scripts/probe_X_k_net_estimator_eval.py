# ruff: noqa: F841
"""Probe X-K: bench candidate net_top_y estimators against user GT.

For every video with `court_calibration_net_top_y` set (user-labeled GT),
runs four candidate estimators and reports per-video error and fleet
aggregates side-by-side. Goal: pick the estimator that becomes the new
production default in contact_detector.estimate_net_position.

Estimators (all run at analysis time, no manual input):

  A0  per-rally median  -- baseline; what shipped today
                          (median across rallies of contacts_json.netY)
  A1  trimmed mean      -- drop top+bottom 20% of per-rally values, mean
                          the rest. Reduces outlier-driven bias.
  B   calibration proj  -- geometrically derive net top from the 4
                          court corners (perspective-net midline minus
                          the projected net height). Independent of
                          ball trajectory.
  C   ball-cross stat   -- aggregate ALL ball positions across all
                          rallies in the video; find Y values where
                          the ball reverses direction (net-crossing
                          moments); take the low-y mode of those
                          reversal points (= net top in image-y-down).

Output: per-video table (sorted by A0 error) + fleet aggregates per
estimator + winner-per-video tally.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"


@dataclass
class VideoData:
    video_id: str
    name: str
    fps: float
    gt: float
    corners: list[dict]                    # [{x,y}, ...] length 4
    rally_net_ys: list[float]              # per-rally contacts_json.netY
    ball_positions_per_rally: list[list[dict]]  # [{frameNumber,x,y,confidence}]


# ---------------------------------------------------------------------------
# Estimator A0: per-rally median (baseline — what's stored today)
# ---------------------------------------------------------------------------
def estimate_a0(d: VideoData) -> float | None:
    if not d.rally_net_ys:
        return None
    return statistics.median(d.rally_net_ys)


# ---------------------------------------------------------------------------
# Estimator A1: trimmed mean (drop top/bottom 20%)
# ---------------------------------------------------------------------------
def estimate_a1(d: VideoData) -> float | None:
    if not d.rally_net_ys:
        return None
    xs = sorted(d.rally_net_ys)
    n = len(xs)
    if n < 5:
        return statistics.mean(xs)
    drop = max(1, n // 5)  # 20% each side
    trimmed = xs[drop:n - drop]
    return statistics.mean(trimmed) if trimmed else statistics.mean(xs)


# ---------------------------------------------------------------------------
# Estimator B: calibration projection (perspective net top from 4 corners)
# ---------------------------------------------------------------------------
def _line_intersect(
    p1: dict, p2: dict, p3: dict, p4: dict,
) -> tuple[float, float] | None:
    """Intersection of line p1-p2 with line p3-p4. Returns (x, y) or None
    if (near-)parallel."""
    denom = (p1["x"] - p2["x"]) * (p3["y"] - p4["y"]) - \
            (p1["y"] - p2["y"]) * (p3["x"] - p4["x"])
    if abs(denom) < 1e-9:
        return None
    t = ((p1["x"] - p3["x"]) * (p3["y"] - p4["y"]) -
         (p1["y"] - p3["y"]) * (p3["x"] - p4["x"])) / denom
    return (
        p1["x"] + t * (p2["x"] - p1["x"]),
        p1["y"] + t * (p2["y"] - p1["y"]),
    )


# Beach VB net height / half-court depth ≈ 2.43m / 8m = 0.30.
# In image space at the midline this becomes the fraction of the
# image-y distance from the midline to the apparent baseline.
NET_HEIGHT_TO_HALF_COURT_RATIO = 0.30


def estimate_b(d: VideoData) -> float | None:
    if len(d.corners) != 4:
        return None
    # Corners: 0=BL, 1=BR, 2=TR, 3=TL (per CourtCalibrationPanel)
    near_baseline_mid_y = (d.corners[0]["y"] + d.corners[1]["y"]) / 2
    # Find perspective net midline by intersecting diagonals + baseline VP
    diag_isect = _line_intersect(
        d.corners[0], d.corners[2], d.corners[1], d.corners[3],
    )
    baseline_vp = _line_intersect(
        d.corners[0], d.corners[1], d.corners[3], d.corners[2],
    )
    if not diag_isect:
        return None
    center_x, center_y = diag_isect
    # Midline passes through (center_x, center_y) toward baseline VP.
    # We just need its y at the court center — that's center_y itself
    # (the perspective net midline crosses the perspective center).
    midline_y = center_y

    # The "near" baseline is at corners[0,1] (BL, BR) — the closer baseline
    # to the camera. Its y is larger (further down in image) than midline_y.
    # The distance midline_y → near_baseline_y represents the image-y span
    # of the near HALF of the court (8m of real court). The net top is
    # 2.43m above the ground at the midline → projects to roughly
    # 0.30 × (near_baseline_y − midline_y) above midline_y.
    half_court_image_y_span = abs(near_baseline_mid_y - midline_y)
    net_top_y = midline_y - NET_HEIGHT_TO_HALF_COURT_RATIO * half_court_image_y_span
    # Clamp to image bounds
    return max(0.0, min(1.0, net_top_y))


# ---------------------------------------------------------------------------
# Estimator C: ball-crossing statistic
# ---------------------------------------------------------------------------
def estimate_c(d: VideoData) -> float | None:
    """Aggregate ball positions across ALL rallies; collect Y values at
    direction reversals (local minima in image y — ball at apex closest
    to net during crossings); return the median of those.

    Image y is down → minima correspond to ball at maximum height.
    The ball reaches its lowest image-y when crossing the net (passing
    over the top of the net is the minimum-y on cross-net arcs), or at
    serve-toss apex (no crossing). Filter to keep only minima that
    sit in the upper-half range of all observed ball ys (drops
    near-ground digs/sets that also produce local minima).
    """
    minima_ys: list[float] = []
    for rally_balls in d.ball_positions_per_rally:
        if len(rally_balls) < 3:
            continue
        # Sort by frame
        bs = sorted(rally_balls, key=lambda b: b.get("frameNumber", 0))
        ys = [b.get("y", 0.0) for b in bs]
        # Detect local minima in y with a small window
        for i in range(1, len(ys) - 1):
            if ys[i] < ys[i - 1] and ys[i] < ys[i + 1] and ys[i] > 0:
                minima_ys.append(ys[i])
    if not minima_ys:
        return None

    # Filter: minima sitting in the upper half of all observed ys.
    # 'Upper' in image-y-down means lower numeric values.
    all_ys = sorted(minima_ys)
    median_min = statistics.median(all_ys)
    upper = [y for y in all_ys if y <= median_min]
    if len(upper) < 3:
        upper = all_ys
    return statistics.median(upper)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_data() -> list[VideoData]:
    out: list[VideoData] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.id::text, v.name, COALESCE(v.fps, 30) AS fps,
                   v.court_calibration_net_top_y AS gt,
                   v.court_calibration_json
            FROM videos v
            WHERE v.court_calibration_net_top_y IS NOT NULL
              AND v.court_calibration_json IS NOT NULL
            ORDER BY v.name
            """,
        )
        rows = cur.fetchall()
        for vid, vname, fps, gt, corners_json in rows:
            corners = corners_json if isinstance(corners_json, list) else json.loads(corners_json)
            cur2 = conn.execute(
                """
                SELECT pt.contacts_json, pt.ball_positions_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.video_id = %s
                """,
                (vid,),
            )
            rally_net_ys: list[float] = []
            ball_pos_per_rally: list[list[dict]] = []
            for cj, bj in cur2:
                if cj is not None:
                    cj_obj = cj if isinstance(cj, dict) else json.loads(cj or '{}')
                    ny = cj_obj.get("netY")
                    if ny is not None and ny > 0:
                        rally_net_ys.append(float(ny))
                if bj is not None:
                    bj_list = bj if isinstance(bj, list) else json.loads(bj or '[]')
                    if bj_list:
                        ball_pos_per_rally.append(bj_list)
            out.append(VideoData(
                video_id=vid, name=vname, fps=fps, gt=gt,
                corners=corners,
                rally_net_ys=rally_net_ys,
                ball_positions_per_rally=ball_pos_per_rally,
            ))
    return out


def _summarize(name: str, errors: dict[str, float]) -> str:
    if not errors:
        return f"  {name}: no successful estimates"
    abs_errors = [abs(e) for e in errors.values()]
    signed = list(errors.values())
    over_005 = sum(1 for a in abs_errors if a > 0.05)
    over_010 = sum(1 for a in abs_errors if a > 0.10)
    over_015 = sum(1 for a in abs_errors if a > 0.15)
    return (
        f"  {name:<22} n={len(errors):>3}  "
        f"med |Δ|={statistics.median(abs_errors):.3f}  "
        f"mean |Δ|={statistics.mean(abs_errors):.3f}  "
        f"worst |Δ|={max(abs_errors):.3f}  "
        f"mean signed Δ={statistics.mean(signed):+.3f}  "
        f">0.05: {over_005:<2}  >0.10: {over_010:<2}  >0.15: {over_015}"
    )


def main() -> int:
    print("Loading videos with GT + calibration corners + per-rally data...", flush=True)
    data = load_data()
    print(f"Loaded {len(data)} videos.\n", flush=True)

    estimators = [
        ("A0 per-rally median", estimate_a0),
        ("A1 trimmed mean",     estimate_a1),
        ("B  calibration proj", estimate_b),
        ("C  ball-cross stat",  estimate_c),
    ]
    results: dict[str, dict[str, float]] = {name: {} for name, _ in estimators}
    estimates_per_video: dict[str, dict[str, float | None]] = {}

    for d in data:
        per_video: dict[str, float | None] = {}
        for name, fn in estimators:
            try:
                est = fn(d)
            except Exception as e:
                print(f"  [WARN] {name} failed on {d.name}: {e}", flush=True)
                est = None
            per_video[name] = est
            if est is not None and not math.isnan(est):
                results[name][d.video_id] = est - d.gt
        estimates_per_video[d.video_id] = per_video

    # Per-video side-by-side, sorted by A0 |error| descending (worst cases lead)
    sorted_videos = sorted(
        data,
        key=lambda v: -abs(estimates_per_video[v.video_id]["A0 per-rally median"] - v.gt)
        if estimates_per_video[v.video_id]["A0 per-rally median"] is not None
        else 0,
    )

    print("Per-video estimate vs GT (sorted by A0 absolute error desc):", flush=True)
    print(
        f"{'video':<14} {'fps':>5} {'gt':>6} | "
        f"{'A0':>6} {'ΔA0':>7} | {'A1':>6} {'ΔA1':>7} | "
        f"{'B':>6} {'ΔB':>7} | {'C':>6} {'ΔC':>7}",
        flush=True,
    )
    for v in sorted_videos:
        per = estimates_per_video[v.video_id]
        def cell(est: float | None) -> tuple[str, str]:
            if est is None:
                return ("   -- ", "    -- ")
            return (f"{est:>6.3f}", f"{est - v.gt:+7.3f}")
        a0v, a0d = cell(per["A0 per-rally median"])
        a1v, a1d = cell(per["A1 trimmed mean"])
        bv, bd = cell(per["B  calibration proj"])
        cv, cd = cell(per["C  ball-cross stat"])
        print(
            f"{v.name:<14} {v.fps:>5.1f} {v.gt:>6.3f} | "
            f"{a0v} {a0d} | {a1v} {a1d} | {bv} {bd} | {cv} {cd}",
            flush=True,
        )

    # Aggregate per estimator
    print(f"\n=== Fleet aggregates (n={len(data)} videos) ===", flush=True)
    for name, _ in estimators:
        print(_summarize(name, results[name]), flush=True)

    # Per-video winner
    winners: Counter = Counter()
    for v in data:
        per = estimates_per_video[v.video_id]
        best = None
        best_err = float("inf")
        for name, _ in estimators:
            est = per[name]
            if est is None:
                continue
            err = abs(est - v.gt)
            if err < best_err:
                best_err = err
                best = name
        if best:
            winners[best] += 1
    print("\n=== Per-video winners ===", flush=True)
    for name, n in winners.most_common():
        print(f"  {name:<22} {n}", flush=True)

    # Worst-case rescues: how does the best non-A0 do on the videos where A0 is worst?
    a0_top10_worst = sorted(
        data,
        key=lambda v: -abs(results["A0 per-rally median"].get(v.video_id, 0)),
    )[:10]
    print("\n=== Top-10 A0-worst videos: best alternative ===", flush=True)
    for v in a0_top10_worst:
        per = estimates_per_video[v.video_id]
        a0_err = abs(per["A0 per-rally median"] - v.gt) if per["A0 per-rally median"] else None
        alts = {n: abs(per[n] - v.gt) for n in ["A1 trimmed mean", "B  calibration proj", "C  ball-cross stat"]
                if per[n] is not None}
        best = min(alts.items(), key=lambda kv: kv[1]) if alts else None
        print(
            f"  {v.name:<14} A0 |Δ|={a0_err:.3f}  -> best non-A0: "
            f"{best[0]} |Δ|={best[1]:.3f}  (improvement={a0_err - best[1]:+.3f})"
            if best else f"  {v.name}  no alternative",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
