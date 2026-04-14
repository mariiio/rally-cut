"""Measure court-space temporal features for serve-side detection.

Projects player positions to court space (meters) and measures:
1. Baseline proximity: which side has a player closest to the baseline?
2. Forward motion: does the baseline player move toward the net?
3. Server isolation in court space: is one player separated from the group?
4. Combined signals

Usage:
    cd analysis
    uv run python scripts/measure_court_space_serve.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from rallycut.evaluation.tracking.db import load_court_calibration  # noqa: E402
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from scripts.eval_score_tracking import RallyData, load_score_gt  # noqa: E402

# Reuse from diagnostic script
sys.path.insert(0, str(Path(__file__).parent))
from diagnose_formation_serve import (  # noqa: E402
    _calibrate_initial_near_is_a,
    _gt_physical_side,
    _parse_positions,
)

console = Console()

# Court constants
COURT_LENGTH = 16.0  # meters
COURT_NET_Y = 8.0
NEAR_BASELINE = 0.0
FAR_BASELINE = 16.0

# Non-target videos to exclude
EXCLUDE_VIDEOS = {"0a383519", "627c1add"}  # yoyo (low cam), caca (tilted)


@dataclass
class CourtSpaceRally:
    rally_id: str
    video_id: str
    gt_physical: str  # "near" / "far"
    # Per-track court-space trajectories: {track_id: [(frame, court_x, court_y), ...]}
    trajectories: dict[int, list[tuple[int, float, float]]]
    n_tracks: int
    projection_ok: bool


def project_rally_to_court(
    positions: list[PlayerPosition],
    calibrator: CourtCalibrator,
    max_frame: int = 90,
) -> dict[int, list[tuple[int, float, float]]]:
    """Project player positions to court space for first max_frame frames."""
    trajectories: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for p in positions:
        if p.track_id < 0 or p.frame_number >= max_frame:
            continue
        foot_x = p.x
        foot_y = p.y + p.height / 2.0
        try:
            cx, cy = calibrator.image_to_court((foot_x, foot_y), 1, 1)
            # Sanity: court coords should be within reasonable bounds
            if -5 < cx < 13 and -3 < cy < 19:
                trajectories[p.track_id].append((p.frame_number, cx, cy))
        except Exception:
            continue
    return dict(trajectories)


def feature_baseline_proximity(traj: dict[int, list[tuple[int, float, float]]]) -> float | None:
    """Which side has a player closest to their baseline?

    Near baseline at cy=0, far baseline at cy=16.
    Returns positive if near side has closer-to-baseline player (near serves).
    """
    if not traj:
        return None

    # Mean court_y per track (first 60 frames for formation window)
    track_cy: dict[int, float] = {}
    for tid, points in traj.items():
        early = [cy for f, _, cy in points if f < 60]
        if early:
            track_cy[tid] = sum(early) / len(early)

    if len(track_cy) < 2:
        return None

    # Split into near (cy < 8) and far (cy >= 8)
    near_tids = [t for t, cy in track_cy.items() if cy < COURT_NET_Y]
    far_tids = [t for t, cy in track_cy.items() if cy >= COURT_NET_Y]

    if not near_tids or not far_tids:
        return None

    # Distance from baseline: near baseline at 0, far baseline at 16
    near_min_bl = min(track_cy[t] for t in near_tids)  # closest to cy=0
    far_min_bl = min(COURT_LENGTH - track_cy[t] for t in far_tids)  # closest to cy=16

    # Positive = near player is closer to baseline = near likely serving
    return far_min_bl - near_min_bl


def feature_forward_motion(traj: dict[int, list[tuple[int, float, float]]]) -> float | None:
    """Does the baseline player move toward the net?

    Server moves forward after serving. Receivers stay in position or move laterally.
    Returns positive if near-side baseline player moves forward (near serves).
    """
    if not traj:
        return None

    # Find baseline candidates (first 30 frames)
    track_early_cy: dict[int, float] = {}
    for tid, points in traj.items():
        early = [cy for f, _, cy in points if f < 30]
        if early:
            track_early_cy[tid] = sum(early) / len(early)

    if len(track_early_cy) < 2:
        return None

    near_tids = [t for t, cy in track_early_cy.items() if cy < COURT_NET_Y]
    far_tids = [t for t, cy in track_early_cy.items() if cy >= COURT_NET_Y]

    if not near_tids or not far_tids:
        return None

    def _forward_motion(tid: int, baseline_y: float) -> float:
        """Compute forward motion (toward net) for a track."""
        points = traj[tid]
        early = [cy for f, _, cy in points if f < 30]
        late = [cy for f, _, cy in points if 45 <= f < 90]
        if not early or not late:
            return 0.0
        early_cy = sum(early) / len(early)
        late_cy = sum(late) / len(late)
        if baseline_y < COURT_NET_Y:
            # Near side: forward = increasing cy (toward net at 8)
            return late_cy - early_cy
        else:
            # Far side: forward = decreasing cy (toward net at 8)
            return early_cy - late_cy

    # Near baseline candidate: player with lowest cy
    near_server = min(near_tids, key=lambda t: track_early_cy[t])
    far_server = max(far_tids, key=lambda t: track_early_cy[t])

    near_motion = _forward_motion(near_server, 0)
    far_motion = _forward_motion(far_server, COURT_LENGTH)

    # Positive = near server has more forward motion = near serves
    return near_motion - far_motion


def feature_court_separation(traj: dict[int, list[tuple[int, float, float]]]) -> float | None:
    """Vertical separation in court space (meters, camera-invariant).

    Same concept as image-space separation but in meters.
    Returns positive if near side has larger separation.
    """
    if not traj:
        return None

    track_cy: dict[int, float] = {}
    for tid, points in traj.items():
        early = [cy for f, _, cy in points if f < 60]
        if early:
            track_cy[tid] = sum(early) / len(early)

    if len(track_cy) < 2:
        return None

    near_tids = [t for t, cy in track_cy.items() if cy < COURT_NET_Y]
    far_tids = [t for t, cy in track_cy.items() if cy >= COURT_NET_Y]

    if not near_tids or not far_tids:
        return None

    def _sep(tids: list[int]) -> float:
        if len(tids) >= 2:
            ys = [track_cy[t] for t in tids]
            return max(ys) - min(ys)
        return abs(track_cy[tids[0]] - COURT_NET_Y) * 0.5

    return _sep(near_tids) - _sep(far_tids)


def feature_server_candidate_score(traj: dict[int, list[tuple[int, float, float]]]) -> float | None:
    """Combined score: baseline proximity + forward motion + isolation.

    The server is the player who:
    1. Starts closest to a baseline
    2. Moves forward after serving
    3. Is most isolated from the other 3 players

    Returns positive = near side has better server candidate.
    """
    bl = feature_baseline_proximity(traj)
    fm = feature_forward_motion(traj)
    cs = feature_court_separation(traj)

    if bl is None:
        return None

    score = bl * 1.0  # baseline proximity (meters)
    if fm is not None:
        score += fm * 2.0  # forward motion (meters, weighted 2x)
    if cs is not None:
        score += cs * 0.5  # court separation (meters)
    return score


def main() -> int:
    console.print("[bold]Loading data...[/bold]")
    video_rallies = load_score_gt()
    total = sum(len(v) for v in video_rallies.values())
    console.print(f"Loaded {total} rallies across {len(video_rallies)} videos")

    # Build calibrators
    calibrators: dict[str, CourtCalibrator] = {}
    for vid in video_rallies:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated:
                calibrators[vid] = cal
    console.print(f"Court calibration: {len(calibrators)}/{len(video_rallies)} videos")

    # Parse positions + convention calibration
    positions_by_rally: dict[str, list[PlayerPosition]] = {}
    net_ys: dict[str, float] = {}
    for vid, rallies in video_rallies.items():
        for rally in rallies:
            positions_by_rally[rally.rally_id] = _parse_positions(rally.positions)
            net_ys[rally.rally_id] = rally.court_split_y if rally.court_split_y else 0.5

    initial_near_is_a: dict[str, bool] = {}
    for vid, rallies in sorted(video_rallies.items()):
        initial_near_is_a[vid] = _calibrate_initial_near_is_a(
            rallies, positions_by_rally, net_ys,
        )

    # Project all rallies to court space
    console.print("\n[bold]Projecting to court space...[/bold]")
    court_rallies: list[CourtSpaceRally] = []
    projection_failures = 0

    for vid, rallies in sorted(video_rallies.items()):
        if any(vid.startswith(e) for e in EXCLUDE_VIDEOS):
            continue
        cal = calibrators.get(vid)
        if cal is None:
            continue
        near_a = initial_near_is_a[vid]

        for rally in rallies:
            positions = positions_by_rally[rally.rally_id]
            traj = project_rally_to_court(positions, cal, max_frame=90)
            gt_phys = _gt_physical_side(rally.gt_serving_team, rally.side_flipped, near_a)

            ok = len(traj) >= 2
            if not ok:
                projection_failures += 1

            court_rallies.append(CourtSpaceRally(
                rally_id=rally.rally_id,
                video_id=vid,
                gt_physical=gt_phys,
                trajectories=traj,
                n_tracks=len(traj),
                projection_ok=ok,
            ))

    console.print(f"Projected {len(court_rallies)} rallies, "
                  f"{projection_failures} projection failures")

    # Compute features
    features_list = [
        ("baseline_proximity", feature_baseline_proximity),
        ("forward_motion", feature_forward_motion),
        ("court_separation", feature_court_separation),
        ("server_candidate", feature_server_candidate_score),
    ]

    console.print("\n[bold]═══ Court-Space Feature Results ═══[/bold]\n")

    feat_table = Table(title="Court-Space Features (target footage)")
    feat_table.add_column("Feature")
    feat_table.add_column("Correct", justify="right")
    feat_table.add_column("Total", justify="right")
    feat_table.add_column("Accuracy", justify="right")
    feat_table.add_column("Abstain", justify="right")
    feat_table.add_column("Coverage", justify="right")

    # Also compare with image-space formation baseline
    from rallycut.tracking.action_classifier import _find_serving_side_by_formation
    formation_c, formation_p = 0, 0
    for cr in court_rallies:
        positions = positions_by_rally[cr.rally_id]
        net_y = net_ys[cr.rally_id]
        side, _ = _find_serving_side_by_formation(positions, net_y=net_y, start_frame=0)
        if side is not None:
            formation_p += 1
            if side == cr.gt_physical:
                formation_c += 1

    feat_table.add_row(
        "image-space formation (baseline)",
        str(formation_c), str(formation_p),
        f"{formation_c / formation_p:.1%}" if formation_p else "-",
        str(len(court_rallies) - formation_p),
        f"{formation_p / len(court_rallies):.0%}",
    )

    for fname, func in features_list:
        correct, total_pred, abstain = 0, 0, 0
        for cr in court_rallies:
            score = func(cr.trajectories)
            if score is None:
                abstain += 1
                continue
            total_pred += 1
            pred = "near" if score > 0 else "far"
            if pred == cr.gt_physical:
                correct += 1

        feat_table.add_row(
            fname,
            str(correct), str(total_pred),
            f"{correct / total_pred:.1%}" if total_pred else "-",
            str(abstain),
            f"{total_pred / len(court_rallies):.0%}",
        )

    console.print(feat_table)

    # Per-video breakdown for the most promising features
    console.print("\n[bold]Per-Video Breakdown[/bold]\n")
    for fname, func in features_list:
        vid_stats: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for cr in court_rallies:
            score = func(cr.trajectories)
            if score is None:
                continue
            pred = "near" if score > 0 else "far"
            c, t = vid_stats[cr.video_id]
            vid_stats[cr.video_id] = (c + (1 if pred == cr.gt_physical else 0), t + 1)

        # Only print for interesting features
        if fname not in ("baseline_proximity", "forward_motion", "server_candidate"):
            continue

        console.print(f"[bold]{fname}:[/bold]")
        from rallycut.evaluation.tracking.db import get_connection
        with get_connection() as conn, conn.cursor() as cur:
            vid_names = {}
            for vid in vid_stats:
                cur.execute("SELECT name FROM videos WHERE id = %s", (vid,))
                r = cur.fetchone()
                vid_names[vid] = r[0] if r else "?"

        for vid, (c, t) in sorted(vid_stats.items(), key=lambda x: x[1][0] / max(x[1][1], 1)):
            acc = c / t if t > 0 else 0
            print(f"  {vid_names.get(vid, '?'):8s}: {c}/{t} = {acc:.0%}")
        print()

    # AUC for continuous scores
    console.print("[bold]Feature AUC-ROC:[/bold]")
    from sklearn.metrics import roc_auc_score
    for fname, func in features_list:
        scores, labels = [], []
        for cr in court_rallies:
            score = func(cr.trajectories)
            if score is None:
                continue
            scores.append(score)
            labels.append(1 if cr.gt_physical == "near" else 0)
        if len(set(labels)) < 2:
            continue
        auc = roc_auc_score(labels, scores)
        auc = max(auc, 1 - auc)
        console.print(f"  {fname:25s}: AUC={auc:.3f} (n={len(scores)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
