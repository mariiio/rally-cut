"""Diagnostic: how often are both players per side visible at rally start?

Core hypothesis: on many rallies, the server is off-screen (near side) or
occluded (far side), so the pre-toss "formation" is actually 1v2 or 2v1
visible players, not 2v2. This changes the feature design for the
geometric serving_team predictor.

For each of the 304 score-GT rallies:
  1. Group tracked players by side at each frame (using auto-split or
     stored court_split_y).
  2. Count visible players per side at frame 0, 30, 60, and "mode" over
     frames 0-60.
  3. Cross-tabulate the per-side count against gt_serving_team to see
     whether the asymmetric-count pattern encodes the serving team.

Read-only. No DB writes.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_score_tracking import load_score_gt, RallyData  # noqa: E402
from rallycut.tracking.action_classifier import _compute_auto_split_y  # noqa: E402
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402


def _to_player_positions(positions: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=p.get("frameNumber", 0),
            track_id=p.get("trackId", -1),
            x=p.get("x", 0),
            y=p.get("y", 0),
            width=p.get("width", 0.05),
            height=p.get("height", 0.10),
            confidence=p.get("confidence", 1.0),
            keypoints=None,
        )
        for p in positions
    ]


def _effective_split(
    rally: RallyData, positions: list[PlayerPosition]
) -> float | None:
    """Use stored court_split_y if it separates players, else auto."""
    split_y = rally.court_split_y
    if split_y is not None:
        track_medians: dict[int, list[float]] = defaultdict(list)
        for p in positions:
            if p.track_id < 0:
                continue
            track_medians[p.track_id].append(p.y + p.height / 2.0)
        if track_medians:
            sides = [
                1 if float(np.median(ys)) > split_y else 0
                for ys in track_medians.values()
            ]
            if any(s == 1 for s in sides) and any(s == 0 for s in sides):
                return split_y
    return _compute_auto_split_y(positions)


def _side_counts_at_frame(
    positions: list[PlayerPosition], split_y: float, frame: int
) -> tuple[int, int]:
    """Return (n_near, n_far) at a given frame."""
    near = far = 0
    for p in positions:
        if p.track_id < 0:
            continue
        if p.frame_number != frame:
            continue
        foot_y = p.y + p.height / 2.0
        if foot_y > split_y:
            near += 1
        else:
            far += 1
    return near, far


def _mode_side_counts(
    positions: list[PlayerPosition],
    split_y: float,
    window_end: int = 60,
) -> tuple[int, int]:
    """Most common (n_near, n_far) across frames 0..window_end.

    Uses frame-by-frame counts and returns the mode.
    """
    per_frame: dict[int, tuple[int, int]] = {}
    for p in positions:
        if p.track_id < 0 or p.frame_number > window_end:
            continue
        foot_y = p.y + p.height / 2.0
        n, f = per_frame.get(p.frame_number, (0, 0))
        if foot_y > split_y:
            n += 1
        else:
            f += 1
        per_frame[p.frame_number] = (n, f)
    if not per_frame:
        return 0, 0
    counts = Counter(per_frame.values())
    return counts.most_common(1)[0][0]


def _unique_track_sides(
    positions: list[PlayerPosition],
    split_y: float,
    window_end: int = 60,
) -> tuple[int, int]:
    """Count UNIQUE tracks per side within the window.

    A track is a near-track if its median foot-Y across all its visible
    frames in the window is > split_y. Counts unique tracks, not per-frame.
    """
    by_track: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        if p.track_id < 0 or p.frame_number > window_end:
            continue
        by_track[p.track_id].append(p.y + p.height / 2.0)
    near = far = 0
    for ys in by_track.values():
        med = float(np.median(ys))
        if med > split_y:
            near += 1
        else:
            far += 1
    return near, far


def main() -> int:
    video_rallies = load_score_gt()
    total = sum(len(v) for v in video_rallies.values())
    print(f"Loaded {total} score-GT rallies across {len(video_rallies)} videos\n")

    # Data structures for the diagnostic
    t0_counts: Counter = Counter()  # (n_near, n_far) at frame 0
    t30_counts: Counter = Counter()
    t60_counts: Counter = Counter()
    mode_counts: Counter = Counter()  # most frequent per-frame count
    unique_counts: Counter = Counter()  # unique tracks per side
    no_split: int = 0

    # For unique_counts, also cross-tab against gt
    unique_by_gt: dict[tuple[int, int], Counter] = defaultdict(Counter)

    # Diagnostic: which side is "short" when 1v2?
    # If near=1, far=2 → near is serving (partner alone at net)
    # If near=2, far=1 → far is serving
    short_side_predictor_correct = 0
    short_side_predictor_total = 0

    for rs in video_rallies.values():
        for r in rs:
            positions = _to_player_positions(r.positions)
            split_y = _effective_split(r, positions)
            if split_y is None:
                no_split += 1
                continue

            t0 = _side_counts_at_frame(positions, split_y, 0)
            t30 = _side_counts_at_frame(positions, split_y, 30)
            t60 = _side_counts_at_frame(positions, split_y, 60)
            mode = _mode_side_counts(positions, split_y, window_end=60)
            unique = _unique_track_sides(positions, split_y, window_end=60)

            t0_counts[t0] += 1
            t30_counts[t30] += 1
            t60_counts[t60] += 1
            mode_counts[mode] += 1
            unique_counts[unique] += 1
            unique_by_gt[unique][r.gt_serving_team] += 1

            # Short-side predictor: the side with fewer players is serving.
            # Use unique counts. Apply side_flipped correction.
            near_u, far_u = unique
            pred: str | None = None
            if near_u < far_u:
                pred = "near"
            elif far_u < near_u:
                pred = "far"
            if pred is not None:
                base = "A" if pred == "near" else "B"
                if r.side_flipped:
                    base = "B" if base == "A" else "A"
                short_side_predictor_total += 1
                if base == r.gt_serving_team:
                    short_side_predictor_correct += 1

    print(f"No split could be determined: {no_split}/{total}\n")

    def _print_dist(name: str, counts: Counter) -> None:
        print(f"=== {name} ===")
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        total_c = sum(counts.values())
        for (n, f), c in ordered:
            pct = c / total_c * 100
            label = ""
            if n + f < 4:
                label = f" (missing {4 - n - f})"
            print(f"  near={n} far={f}: {c:4d} ({pct:5.1f}%){label}")
        print()

    _print_dist("Side counts at frame 0 (single frame)", t0_counts)
    _print_dist("Side counts at frame 30", t30_counts)
    _print_dist("Side counts at frame 60", t60_counts)
    _print_dist("MODE side counts across frames 0-60", mode_counts)
    _print_dist("UNIQUE tracks per side across frames 0-60", unique_counts)

    # Short-side predictor accuracy
    print(f"=== Short-side predictor (side with fewer tracks serves) ===")
    scored = short_side_predictor_total
    corr = short_side_predictor_correct
    print(f"  Applied to {scored}/{total} rallies (others were symmetric)")
    if scored > 0:
        print(f"  Correct: {corr}/{scored} = {corr/scored*100:.1f}%")
    print()

    # Cross-tab: for each (near, far) count, what's the gt_serving_team split?
    print(f"=== Cross-tab: unique count → gt_serving_team ===")
    print(f"  {'near/far':>10s}  {'total':>6s}  {'gt=A':>6s}  {'gt=B':>6s}  {'A%':>6s}")
    for counts_pair in sorted(unique_by_gt.keys()):
        gt_dist = unique_by_gt[counts_pair]
        a = gt_dist.get("A", 0)
        b = gt_dist.get("B", 0)
        t = a + b
        a_pct = a / t * 100 if t > 0 else 0
        print(f"  {str(counts_pair):>10s}  {t:6d}  {a:6d}  {b:6d}  {a_pct:5.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
