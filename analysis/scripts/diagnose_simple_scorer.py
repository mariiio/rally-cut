"""W1 diagnostic: can score_accuracy be computed from ball + court only?

Baseline score_accuracy is 46.2% (run_2026-04-10-001609.json), produced end-to-end
via contact-detection -> action classifier -> RallyActions.serving_team.

This script tests whether a minimal-dependency predictor using ONLY
(a) ball_positions_json, (b) court_split_y, (c) rally-start geometry
can match or beat that number on the same 91/11 rally set.

Gate: >= +10pp over 46.2% (i.e. >= 56.2%) -> the whole roadmap pivots.
Tie / regression -> current chain validated.

Read-only. No DB writes, no production module edits.

Convention (from player_filter.classify_teams:711 and action_classifier.to_dict:174):
    team 0 = "A" = near court = y > court_split_y
    team 1 = "B" = far court  = y < court_split_y
ball_positions_json and court_split_y are in the same normalized [0, 1] image space.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_action_detection import RallyData, load_rallies_with_action_gt  # noqa: E402


@dataclass
class _PredStub:
    """Minimal RallyActions-shaped object for compute_score_metrics()."""

    rally_id: str
    serving_team: str | None


def _side_from_y(y: float, split: float) -> str:
    """Near (y > split) => 'A', far => 'B'."""
    return "A" if y > split else "B"


def _valid_ball_points(rally: RallyData) -> list[tuple[int, float, float]]:
    """Return (frame, x, y) sorted by frame, only entries with real coords."""
    if not rally.ball_positions_json:
        return []
    out: list[tuple[int, float, float]] = []
    for bp in rally.ball_positions_json:
        x = bp.get("x", 0.0) or 0.0
        y = bp.get("y", 0.0) or 0.0
        conf = bp.get("confidence", 1.0)
        if conf is not None and conf <= 0:
            continue
        if x <= 0 and y <= 0:
            continue
        frame = bp.get("frameNumber", bp.get("frame", 0)) or 0
        out.append((int(frame), float(x), float(y)))
    out.sort(key=lambda t: t[0])
    return out


# ---------- Predictors ---------------------------------------------------


def p0_const_a(rally: RallyData) -> str | None:
    return "A"


def p1_first_ball_side(rally: RallyData) -> str | None:
    pts = _valid_ball_points(rally)
    if not pts or rally.court_split_y is None:
        return None
    _, _, y = pts[0]
    return _side_from_y(y, rally.court_split_y)


def p2_first_n_centroid(rally: RallyData, n: int = 5) -> str | None:
    pts = _valid_ball_points(rally)
    if not pts or rally.court_split_y is None:
        return None
    ys = [y for _, _, y in pts[:n]]
    ys.sort()
    median_y = ys[len(ys) // 2]
    return _side_from_y(median_y, rally.court_split_y)


def p3_pre_peak_side(rally: RallyData) -> str | None:
    """Ball side BEFORE its highest (min y) point in the first ~2s."""
    pts = _valid_ball_points(rally)
    if not pts or rally.court_split_y is None:
        return None
    fps = rally.fps or 30.0
    window_end_frame = pts[0][0] + int(2.5 * fps)
    early = [p for p in pts if p[0] <= window_end_frame]
    if len(early) < 3:
        early = pts[:10]
    # Apex = min y (image y: smaller = higher in frame)
    apex_idx = min(range(len(early)), key=lambda i: early[i][2])
    pre = early[: max(1, apex_idx)]
    ys = [y for _, _, y in pre]
    ys.sort()
    median_y = ys[len(ys) // 2]
    return _side_from_y(median_y, rally.court_split_y)


def p4_p3_with_p2_fallback(rally: RallyData) -> str | None:
    r = p3_pre_peak_side(rally)
    if r is not None:
        return r
    return p2_first_n_centroid(rally)


PREDICTORS: list[tuple[str, Callable[[RallyData], str | None]]] = [
    ("P0_const_A", p0_const_a),
    ("P1_first_ball", p1_first_ball_side),
    ("P2_first5_centroid", p2_first_n_centroid),
    ("P3_pre_peak", p3_pre_peak_side),
    ("P4_p3_or_p2", p4_p3_with_p2_fallback),
]


# ---------- Metric computation -------------------------------------------


@dataclass
class PredictorReport:
    name: str
    score_accuracy: float
    n_scored: int
    n_videos: int
    n_abstain: int
    confusion: dict[tuple[str, str], int]  # (pred, gt) -> count
    per_video: dict[str, tuple[int, int]]  # video -> (correct, total)


def _eval_predictor(
    rallies: list[RallyData],
    name: str,
    fn: Callable[[RallyData], str | None],
) -> PredictorReport:
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    per_video: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    n_scored = 0
    n_correct = 0
    n_abstain = 0
    videos_with_any: set[str] = set()

    for r in rallies:
        if r.gt_serving_team is None:
            continue
        n_scored += 1
        videos_with_any.add(r.video_id)
        pred = fn(r)
        if pred is None:
            n_abstain += 1
            confusion[("NONE", r.gt_serving_team)] += 1
            per_video[r.video_id][1] += 1
            continue
        confusion[(pred, r.gt_serving_team)] += 1
        per_video[r.video_id][1] += 1
        if pred == r.gt_serving_team:
            n_correct += 1
            per_video[r.video_id][0] += 1

    acc = (n_correct / n_scored) if n_scored > 0 else 0.0
    return PredictorReport(
        name=name,
        score_accuracy=acc,
        n_scored=n_scored,
        n_videos=len(videos_with_any),
        n_abstain=n_abstain,
        confusion=dict(confusion),
        per_video={v: (c[0], c[1]) for v, c in per_video.items()},
    )


# ---------- Reporting ----------------------------------------------------


BASELINE = 0.462
GATE = BASELINE + 0.10


def _fmt_conf(conf: dict[tuple[str, str], int]) -> str:
    parts = []
    for pred in ("A", "B", "NONE"):
        for gt in ("A", "B"):
            c = conf.get((pred, gt), 0)
            if c:
                parts.append(f"pred={pred} gt={gt}: {c}")
    return " | ".join(parts)


def _print_report(rep: PredictorReport) -> None:
    delta = rep.score_accuracy - BASELINE
    gate_str = "PASS" if rep.score_accuracy >= GATE else "fail"
    print(
        f"  {rep.name:22s}  acc={rep.score_accuracy*100:5.1f}%  "
        f"Δ={delta*+100:+5.1f}pp  gate={gate_str}  "
        f"n={rep.n_scored} vids={rep.n_videos} abstain={rep.n_abstain}"
    )
    print(f"    confusion: {_fmt_conf(rep.confusion)}")


def main() -> int:
    print("Loading rallies with action GT...")
    rallies = load_rallies_with_action_gt()
    print(f"  loaded: {len(rallies)} rallies")

    scored = [r for r in rallies if r.gt_serving_team is not None]
    videos = {r.video_id for r in scored}
    print(f"  with gt_serving_team: {len(scored)} rallies across {len(videos)} videos")

    if not scored:
        print("  ERROR: no rallies with gt_serving_team — abort")
        return 1

    # Sanity: first-rally dump
    sample = scored[0]
    pts = _valid_ball_points(sample)
    print(
        f"  sample rally {sample.rally_id[:8]}: "
        f"ball_pts={len(pts)} split_y={sample.court_split_y} "
        f"gt_serving_team={sample.gt_serving_team}"
    )
    if pts:
        print(f"    first ball pt: frame={pts[0][0]} x={pts[0][1]:.3f} y={pts[0][2]:.3f}")

    # Class-balance floor
    gt_a = sum(1 for r in scored if r.gt_serving_team == "A")
    gt_b = sum(1 for r in scored if r.gt_serving_team == "B")
    print(f"  class balance: A={gt_a} B={gt_b}  majority-class floor={max(gt_a, gt_b)/len(scored)*100:.1f}%")
    print()

    print(f"Baseline (production_eval): {BASELINE*100:.1f}%  Gate: >= {GATE*100:.1f}%")
    print()

    reports: list[PredictorReport] = []
    for name, fn in PREDICTORS:
        rep = _eval_predictor(scored, name, fn)
        reports.append(rep)
        _print_report(rep)

    # Per-video breakdown for best predictor
    best = max(reports, key=lambda r: r.score_accuracy)
    print()
    print(f"Best predictor: {best.name}  acc={best.score_accuracy*100:.1f}%")
    print("Per-video breakdown:")
    for v, (c, t) in sorted(best.per_video.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1])):
        print(f"  {v[:8]}  {c:3d}/{t:3d}  {c/max(1,t)*100:5.1f}%")

    print()
    if best.score_accuracy >= GATE:
        print(f"VERDICT: GO — {best.name} beats gate by {(best.score_accuracy-GATE)*100:+.1f}pp")
    elif best.score_accuracy >= BASELINE:
        print(f"VERDICT: TIE — {best.name} matches baseline (+{(best.score_accuracy-BASELINE)*100:.1f}pp, below +10pp gate)")
    else:
        print(f"VERDICT: NO-GO — best {best.name} regresses by {(best.score_accuracy-BASELINE)*100:+.1f}pp vs baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
