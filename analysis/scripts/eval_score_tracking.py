"""Score-tracking evaluation harness.

Reusable LOO-video CV framework for evaluating serve-side predictors.
All predictors receive a RallyData and return "A", "B", or None (abstain).

GT rules:
  - gt_serving_team is canonical on all labeled rallies.
  - Point winner is derived: winner[i] = server[i+1] for consecutive rallies
    in the same video. Last rally uses gt_point_winner if available.
  - GT sideSwitches are applied when mapping near/far → A/B.

Usage:
  uv run python scripts/eval_score_tracking.py
  uv run python scripts/eval_score_tracking.py --predictor const_A
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402


# ── Data ─────────────────────────────────────────────────────────────────


@dataclass
class RallyData:
    rally_id: str
    video_id: str
    start_ms: int
    gt_serving_team: str  # "A" or "B"
    gt_point_winner: str | None  # only set on last rally per video
    positions: list[dict]
    ball_positions: list[dict]
    court_split_y: float | None
    fps: float
    rally_index: int = 0
    side_flipped: bool = False


class TrainablePredictor(Protocol):
    """Interface for predictors that need training data."""

    def train(self, rallies: list[RallyData]) -> None: ...
    def predict(self, rally: RallyData) -> str | None: ...


# Simple predictor: Callable[[RallyData], str | None]
SimplePredictor = Callable[[RallyData], str | None]


# ── GT Loader ────────────────────────────────────────────────────────────


def load_score_gt() -> dict[str, list[RallyData]]:
    """Load all rallies with gt_serving_team, grouped by video_id.

    Applies sideSwitches from player_matching_gt_json to compute side_flipped
    per rally. Also loads gt_point_winner for last-rally derivation.
    """
    # Step 1: Load sideSwitches per video
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, player_matching_gt_json FROM videos
            WHERE id IN (
                SELECT DISTINCT video_id FROM rallies
                WHERE gt_serving_team IS NOT NULL
            )
        """)
        video_switches: dict[str, set[int]] = {}
        for vid, gt in cur.fetchall():
            sw = list(gt.get("sideSwitches", [])) if isinstance(gt, dict) else []
            video_switches[vid] = set(sw)

    # Step 2: Load rally data
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.video_id, r.start_ms, r.gt_serving_team,
                   r.gt_point_winner,
                   pt.positions_json, pt.ball_positions_json,
                   pt.court_split_y, pt.fps
            FROM rallies r
            LEFT JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id IN (
                SELECT DISTINCT video_id FROM rallies
                WHERE gt_serving_team IS NOT NULL
            )
              AND r.gt_serving_team IS NOT NULL
            ORDER BY r.video_id, r.start_ms
        """)
        raw: dict[str, list[tuple]] = defaultdict(list)
        for row in cur.fetchall():
            raw[row[1]].append(row)

    # Step 3: Build RallyData with side_flipped
    out: dict[str, list[RallyData]] = {}
    for vid, rows in raw.items():
        rows.sort(key=lambda r: r[2])
        switches = video_switches.get(vid, set())
        flipped = False
        vid_out: list[RallyData] = []
        for idx, (rid, _, sms, gt_st, gt_pw, pj, bpj, split_y, fps) in enumerate(rows):
            if idx in switches:
                flipped = not flipped
            vid_out.append(RallyData(
                rally_id=rid,
                video_id=vid,
                start_ms=sms or 0,
                gt_serving_team=gt_st,
                gt_point_winner=gt_pw,
                positions=pj or [],
                ball_positions=bpj or [],
                court_split_y=split_y,
                fps=fps or 30.0,
                rally_index=idx,
                side_flipped=flipped,
            ))
        out[vid] = vid_out
    return out


# ── Eval Engine ──────────────────────────────────────────────────────────


@dataclass
class VideoResult:
    video_id: str
    n_rallies: int = 0
    n_correct: int = 0
    n_abstain: int = 0
    cm_a_as_a: int = 0
    cm_a_as_b: int = 0
    cm_b_as_a: int = 0
    cm_b_as_b: int = 0

    @property
    def accuracy(self) -> float:
        scored = self.n_rallies - self.n_abstain
        return self.n_correct / scored if scored > 0 else 0.0

    @property
    def coverage(self) -> float:
        return (self.n_rallies - self.n_abstain) / self.n_rallies if self.n_rallies > 0 else 0.0


@dataclass
class EvalResult:
    predictor_name: str
    videos: list[VideoResult] = field(default_factory=list)

    @property
    def total_rallies(self) -> int:
        return sum(v.n_rallies for v in self.videos)

    @property
    def total_correct(self) -> int:
        return sum(v.n_correct for v in self.videos)

    @property
    def total_abstain(self) -> int:
        return sum(v.n_abstain for v in self.videos)

    @property
    def accuracy(self) -> float:
        scored = self.total_rallies - self.total_abstain
        return self.total_correct / scored if scored > 0 else 0.0

    @property
    def coverage(self) -> float:
        return (self.total_rallies - self.total_abstain) / self.total_rallies if self.total_rallies > 0 else 0.0

    @property
    def cm_a_as_a(self) -> int:
        return sum(v.cm_a_as_a for v in self.videos)

    @property
    def cm_a_as_b(self) -> int:
        return sum(v.cm_a_as_b for v in self.videos)

    @property
    def cm_b_as_a(self) -> int:
        return sum(v.cm_b_as_a for v in self.videos)

    @property
    def cm_b_as_b(self) -> int:
        return sum(v.cm_b_as_b for v in self.videos)


def _evaluate_predictor_on_rallies(
    rallies: list[RallyData],
    predict_fn: Callable[[RallyData], str | None],
    video_id: str,
) -> VideoResult:
    """Evaluate a predictor on a list of rallies from one video."""
    vr = VideoResult(video_id=video_id, n_rallies=len(rallies))
    for r in rallies:
        pred = predict_fn(r)
        if pred is None:
            vr.n_abstain += 1
            continue
        gt = r.gt_serving_team
        if pred == gt:
            vr.n_correct += 1
        # Confusion matrix
        if gt == "A" and pred == "A":
            vr.cm_a_as_a += 1
        elif gt == "A" and pred == "B":
            vr.cm_a_as_b += 1
        elif gt == "B" and pred == "A":
            vr.cm_b_as_a += 1
        elif gt == "B" and pred == "B":
            vr.cm_b_as_b += 1
    return vr


def evaluate_simple(
    name: str,
    predict_fn: SimplePredictor,
    video_rallies: dict[str, list[RallyData]],
) -> EvalResult:
    """Evaluate a non-trainable predictor on all videos (no LOO needed)."""
    result = EvalResult(predictor_name=name)
    for vid, rallies in sorted(video_rallies.items()):
        vr = _evaluate_predictor_on_rallies(rallies, predict_fn, vid)
        result.videos.append(vr)
    return result


def evaluate_loo(
    name: str,
    predictor: TrainablePredictor,
    video_rallies: dict[str, list[RallyData]],
) -> EvalResult:
    """Evaluate a trainable predictor with leave-one-video-out CV."""
    result = EvalResult(predictor_name=name)
    video_ids = sorted(video_rallies.keys())
    for held_out_vid in video_ids:
        # Train on all other videos
        train_rallies = []
        for vid in video_ids:
            if vid != held_out_vid:
                train_rallies.extend(video_rallies[vid])
        predictor.train(train_rallies)
        # Predict on held-out
        vr = _evaluate_predictor_on_rallies(
            video_rallies[held_out_vid], predictor.predict, held_out_vid
        )
        result.videos.append(vr)
    return result


def print_result(result: EvalResult) -> None:
    """Print evaluation result with per-video breakdown."""
    print(f"\n{'=' * 70}")
    print(f"Predictor: {result.predictor_name}")
    print(f"{'=' * 70}")
    print(f"Aggregate: {result.total_correct}/{result.total_rallies} = "
          f"{result.accuracy * 100:.1f}%  "
          f"(coverage: {result.coverage * 100:.1f}%, "
          f"abstain: {result.total_abstain})")
    print()

    # Confusion matrix
    print("Confusion matrix (rows=GT, cols=Pred):")
    print(f"          Pred_A  Pred_B")
    print(f"  GT_A  {result.cm_a_as_a:6d}  {result.cm_a_as_b:6d}")
    print(f"  GT_B  {result.cm_b_as_a:6d}  {result.cm_b_as_b:6d}")
    print()

    # Per-video breakdown
    print(f"{'video_id':10s}  {'rallies':>7s}  {'correct':>7s}  {'acc':>7s}  {'abstain':>7s}")
    print("-" * 50)
    for vr in sorted(result.videos, key=lambda v: v.accuracy):
        print(f"{vr.video_id[:10]:10s}  {vr.n_rallies:7d}  {vr.n_correct:7d}  "
              f"{vr.accuracy * 100:6.1f}%  {vr.n_abstain:7d}")

    # Gate check
    print()
    if result.accuracy >= 0.85 and result.coverage >= 0.95:
        worst_video = min(result.videos, key=lambda v: v.accuracy)
        if worst_video.accuracy >= 0.60:
            print(f"PASS: {result.accuracy * 100:.1f}% >= 85% gate, "
                  f"coverage {result.coverage * 100:.1f}% >= 95%, "
                  f"worst video {worst_video.accuracy * 100:.1f}% >= 60%")
        else:
            print(f"PARTIAL: aggregate {result.accuracy * 100:.1f}% >= 85% "
                  f"but worst video {worst_video.video_id[:8]} = "
                  f"{worst_video.accuracy * 100:.1f}% < 60%")
    elif result.accuracy >= 0.75:
        print(f"PROMISING: {result.accuracy * 100:.1f}% — worth combining "
              f"with other signals or expanding data.")
    else:
        print(f"NO-GO: {result.accuracy * 100:.1f}% below 75% threshold.")


# ── Point-winner derivation ──────────────────────────────────────────────


def derive_point_winners(
    rallies: list[RallyData],
    pred_serving: dict[str, str | None],
) -> dict[str, str | None]:
    """Derive point winners from predicted serving teams.

    winner[i] = server[i+1] for consecutive rallies in the same video.
    Last rally uses gt_point_winner if available.
    """
    winners: dict[str, str | None] = {}
    for i, r in enumerate(rallies):
        if i + 1 < len(rallies):
            # Winner of rally i = server of rally i+1
            winners[r.rally_id] = pred_serving.get(rallies[i + 1].rally_id)
        else:
            # Last rally — use gt_point_winner
            winners[r.rally_id] = r.gt_point_winner
    return winners


# ── Baseline Predictors ──────────────────────────────────────────────────


def pred_const_a(_rally: RallyData) -> str | None:
    return "A"


def pred_const_b(_rally: RallyData) -> str | None:
    return "B"


BASELINES: list[tuple[str, SimplePredictor]] = [
    ("const_A", pred_const_a),
    ("const_B", pred_const_b),
]


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Score tracking eval harness")
    parser.add_argument("--predictor", type=str, default=None,
                        help="Run a specific predictor (default: run all baselines)")
    args = parser.parse_args()

    print("Loading score GT...")
    video_rallies = load_score_gt()
    total = sum(len(v) for v in video_rallies.values())
    print(f"Loaded {total} rallies across {len(video_rallies)} videos\n")

    # Class balance
    counts: dict[str, int] = defaultdict(int)
    for rs in video_rallies.values():
        for r in rs:
            counts[r.gt_serving_team] += 1
    floor = max(counts.values()) / total * 100
    print(f"GT class balance: {dict(counts)}  majority-class floor = {floor:.1f}%")

    # Per-video summary
    print(f"\n{'video_id':10s}  {'rallies':>7s}  {'A':>4s}  {'B':>4s}")
    print("-" * 35)
    for vid in sorted(video_rallies.keys()):
        rs = video_rallies[vid]
        a = sum(1 for r in rs if r.gt_serving_team == "A")
        b = len(rs) - a
        print(f"{vid[:10]:10s}  {len(rs):7d}  {a:4d}  {b:4d}")

    # Run baselines
    predictors: list[tuple[str, SimplePredictor]] = []
    if args.predictor:
        for name, fn in BASELINES:
            if name == args.predictor:
                predictors.append((name, fn))
                break
        if not predictors:
            print(f"Unknown predictor: {args.predictor}")
            return 1
    else:
        predictors = BASELINES

    for name, fn in predictors:
        result = evaluate_simple(name, fn, video_rallies)
        print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
