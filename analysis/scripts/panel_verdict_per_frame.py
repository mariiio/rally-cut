"""Per-frame verdict tool for the cross-rally identity panel.

Reads `positions_json` directly from the database — the same artifact the
editor renders frame-by-frame — and reports per-rally signals that can be
compared against the user's visual verdicts in `panel_2026_05_01.py`.

Why this exists: the prior auto-verdict tool (`verdict_reid_vs_panel.py`)
read `match_analysis_json.appliedFullMapping` (one mapping per rally) and
forensic-sidecar Hungarian output. It missed the 7307c1d regression
because positions_json was being written with up-to-37% UNLABELED (-1)
per-frame trackIds while appliedFullMapping looked correct. This tool
reads the per-frame stream instead, so any divergence between "what
Hungarian decided" and "what the editor actually displays" is visible.

The tool is intentionally honest about what it can and cannot detect from
positions_json alone:

  - Hungarian-drop (< 4 distinct PIDs):     YES, trivially
  - UNLABELED frames (trackId == -1):       YES, trivially
  - Within-rally identity swap:             YES, via per-frame bbox-IoU
                                             continuity check
  - Static cross-rally identity error:      NO. positions_json is
    internally consistent; the user's verdict relies on knowing which
    physical player should be PID 1/2/3/4, which is cross-rally state.
    Such cases will read as "GOOD" here even when the user marks BAD,
    and the disagreement is the honest signal.

Usage:
    cd analysis
    uv run python scripts/panel_verdict_per_frame.py

PASS condition: final summary line reads `AGREES: 13/13`.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from panel_2026_05_01 import PANEL_2026_05_01, PanelVerdict  # noqa: E402

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402

UNLABELED_TRACK_ID = -1

# Trajectory continuity threshold: if a PID's bbox center jumps more than
# this distance (in normalized coords, where 1.0 == video width/height)
# between consecutive observations of the same PID, flag it as a
# discontinuity / probable identity swap. 0.3 is conservative — a player
# crossing the entire court takes many frames; a 30%-of-frame teleport
# between adjacent observations is unphysical.
TRAJECTORY_JUMP_THRESHOLD = 0.30

# IoU threshold below which we consider two bboxes to be "different
# objects" for the per-frame swap detector.
IOU_SAME_TRACK_THRESHOLD = 0.10

# Required canonical player IDs. After remap-track-ids, the four primary
# tracks must carry exactly these labels. Missing any of them means
# Hungarian dropped a PID — the dominant pattern in the panel BAD cases.
EXPECTED_PRIMARY_PIDS = {1, 2, 3, 4}

# Unlabeled (-1) frames are noise unless they're a substantial fraction of
# the rally — a few -1s near rally edges are common and not user-visible.
# These thresholds: > 50 absolute frames OR > 1% of positions, whichever
# is smaller, mark a rally as BAD on the unlabeled signal.
UNLABELED_ABS_THRESHOLD = 50
UNLABELED_FRAC_THRESHOLD = 0.01

# Per-frame swap event tolerance. A single Hungarian-detected swap event
# in a multi-thousand-frame rally is significant: it means at SOME pair of
# adjacent frames, the optimal IoU assignment of {1,2,3,4} bboxes was a
# swap rather than the identity. Requires >= 1 event after we've already
# filtered out false-positive ratios via min_iou + min_margin.
SWAP_EVENTS_THRESHOLD = 1

# Minimum IoU advantage of the swap target over self to count as a real
# swap. Eliminates noise where two adjacent players have near-identical
# IoU values regardless of assignment.
SWAP_IOU_MARGIN = 0.10

# "Slow drift" within-rally swap detector: per-PID half-and-half centroid
# shift combined with spatial-range overlap with another PID. Catches the
# case where a PID's identity gradually drifts to another player without
# any single frame transition large enough to fire the IoU detector
# (b5fb0594/r10: "starts GOOD then NEW p1<->p2 within-team swap after
# occlusion"). Both thresholds must fire simultaneously — the half-shift
# alone is sometimes elevated by legitimate movement, and the overlap
# alone is sometimes elevated by closely-positioned same-team players.
HALF_SHIFT_THRESHOLD = 0.25  # max same-PID half-and-half center shift
# Bumped 2026-05-01 from 0.20 → 0.25 after user visual report on b5fb0594/r10:
# rally has shift=0.21 but is visually clean (PID3 shirtless / PID4 green shirt
# stay correctly assigned). Real drift cases (e.g. 5c756c41/r07 PID4 shift=0.58)
# are well above the new gate. Avoids false-positive `slow_drift` flags on
# rallies with legitimate large positional movement that doesn't represent
# identity drift.
XRANGE_OVERLAP_THRESHOLD = 0.50  # max overlap fraction between PID x-ranges


@dataclass
class RallySignals:
    rally_tag: str
    rally_id: str
    n_positions: int
    n_distinct_pids: int  # excluding UNLABELED
    distinct_pids: list[int]
    n_unlabeled: int
    pid_max_jump: float  # max consecutive-frame center distance, any PID
    pid_swap_events: int  # count of frame pairs where PID-IoU continuity broke
    half_shift_max: float  # max same-PID half-and-half centroid shift
    xrange_overlap_max: float  # max overlap fraction between PID x-ranges
    derived_verdict: str  # "GOOD" or "BAD"
    derived_shape: str  # human-readable explanation of the verdict


def _resolve_rally_id(
    cur: Any, video_id: str, rally_idx: int,
) -> tuple[str, int, int] | None:
    """Same ordering as forensic_panel_ground_truth._resolve_rally_id:
    rallies with non-null positions_json, ordered by start_ms."""
    cur.execute(
        """
        SELECT r.id::text, r.start_ms, r.end_ms
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND pt.positions_json IS NOT NULL
        ORDER BY r.start_ms
        """,
        [video_id],
    )
    rows = cur.fetchall()
    if rally_idx >= len(rows):
        return None
    rid, start_ms, end_ms = rows[rally_idx]
    return cast(str, rid), int(start_ms), int(end_ms)


def _load_positions(cur: Any, rally_id: str) -> list[dict[str, Any]]:
    cur.execute(
        "SELECT positions_json FROM player_tracks WHERE rally_id = %s",
        [rally_id],
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return []
    payload = row[0] if isinstance(row[0], list) else json.loads(row[0])
    return cast(list[dict[str, Any]], payload)


def _bbox_center(p: dict[str, Any]) -> tuple[float, float]:
    """Center of the bbox in the same (normalized?) coords positions_json
    stores. We use raw coordinates without normalization since the
    threshold tolerates either: a 0.30 jump in normalized coords or a
    30%-of-frame jump in raw pixel coords both indicate a teleport."""
    return (
        float(p.get("x", 0.0)) + float(p.get("width", 0.0)) / 2.0,
        float(p.get("y", 0.0)) + float(p.get("height", 0.0)) / 2.0,
    )


def _bbox_xyxy(p: dict[str, Any]) -> tuple[float, float, float, float]:
    x = float(p.get("x", 0.0))
    y = float(p.get("y", 0.0))
    w = float(p.get("width", 0.0))
    h = float(p.get("height", 0.0))
    return (x, y, x + w, y + h)


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _slow_drift_signal(
    by_pid: dict[int, list[dict[str, Any]]],
) -> tuple[float, float, int, int]:
    """Compute the slow-drift swap-detector inputs.

    Returns (max_half_shift, max_xrange_overlap, drift_pid, overlap_pids_count).

    For each PID in EXPECTED_PRIMARY_PIDS, compute the Euclidean shift
    between the centroid of its first-half observations and second-half
    observations. The max across PIDs is reported. Sample threshold for
    "elevated drift": ~0.20 normalized units. This stays below the GOOD
    rally ceiling on the 2026-05-01 panel.

    For each pair of PIDs in EXPECTED_PRIMARY_PIDS, compute the fractional
    overlap of their bbox-x ranges relative to the larger PID's range.
    The max across pairs is reported. Threshold for "elevated overlap":
    ~0.50.

    drift_pid is the PID that contributed the maximum shift; useful for
    the human-readable verdict shape label.

    overlap_pids_count is the number of pid pairs with overlap above the
    threshold; informational.
    """
    import statistics
    max_shift = 0.0
    drift_pid = -1
    for pid in sorted(by_pid):
        if pid not in EXPECTED_PRIMARY_PIDS:
            continue
        pts = sorted(by_pid[pid], key=lambda q: int(q.get("frameNumber", 0)))
        if len(pts) < 10:
            continue
        half = len(pts) // 2
        cx_first = [_bbox_center(p)[0] for p in pts[:half]]
        cy_first = [_bbox_center(p)[1] for p in pts[:half]]
        cx_second = [_bbox_center(p)[0] for p in pts[half:]]
        cy_second = [_bbox_center(p)[1] for p in pts[half:]]
        dx = statistics.mean(cx_second) - statistics.mean(cx_first)
        dy = statistics.mean(cy_second) - statistics.mean(cy_first)
        shift = (dx ** 2 + dy ** 2) ** 0.5
        if shift > max_shift:
            max_shift = shift
            drift_pid = pid

    # Pairwise x-range overlap fraction.
    primary_pids = [p for p in by_pid if p in EXPECTED_PRIMARY_PIDS]
    max_overlap = 0.0
    overlap_pairs = 0
    for i in range(len(primary_pids)):
        for j in range(i + 1, len(primary_pids)):
            cx_i = [_bbox_center(p)[0] for p in by_pid[primary_pids[i]]]
            cx_j = [_bbox_center(p)[0] for p in by_pid[primary_pids[j]]]
            if not cx_i or not cx_j:
                continue
            min_i, max_i = min(cx_i), max(cx_i)
            min_j, max_j = min(cx_j), max(cx_j)
            ov_lo, ov_hi = max(min_i, min_j), min(max_i, max_j)
            ov = max(0.0, ov_hi - ov_lo)
            denom = max(max_i - min_i, max_j - min_j, 1e-3)
            frac = ov / denom
            if frac > max_overlap:
                max_overlap = frac
            if frac > XRANGE_OVERLAP_THRESHOLD:
                overlap_pairs += 1
    return max_shift, max_overlap, drift_pid, overlap_pairs


def _compute_signals(positions: list[dict[str, Any]], rally_tag: str, rally_id: str) -> RallySignals:
    """Derive per-rally health signals from positions_json.

    The verdict heuristic — pid count first, then unlabeled, then per-PID
    trajectory jumps, then per-frame-pair IoU swap events — is ordered
    cheapest-to-most-expensive and short-circuits on the first failure
    so the derived_shape labels the dominant anomaly mode.
    """
    n_positions = len(positions)
    if n_positions == 0:
        return RallySignals(
            rally_tag=rally_tag, rally_id=rally_id,
            n_positions=0, n_distinct_pids=0, distinct_pids=[],
            n_unlabeled=0, pid_max_jump=0.0, pid_swap_events=0,
            half_shift_max=0.0, xrange_overlap_max=0.0,
            derived_verdict="BAD", derived_shape="empty positions_json",
        )

    pids: set[int] = set()
    n_unlabeled = 0
    for p in positions:
        tid = p.get("trackId")
        if tid is None:
            continue
        tid_i = int(tid)
        if tid_i == UNLABELED_TRACK_ID:
            n_unlabeled += 1
        else:
            pids.add(tid_i)

    # Per-PID trajectory jump: walk each PID's observations frame-by-frame
    # and compute the max consecutive-frame center distance. A discontinuity
    # exceeding TRAJECTORY_JUMP_THRESHOLD strongly suggests an identity swap
    # (the same PID label was attached to a physically distant box).
    by_pid: dict[int, list[dict[str, Any]]] = {}
    for p in positions:
        tid = p.get("trackId")
        if tid is None:
            continue
        tid_i = int(tid)
        if tid_i == UNLABELED_TRACK_ID:
            continue
        by_pid.setdefault(tid_i, []).append(p)

    pid_max_jump = 0.0
    for pid, pts in by_pid.items():
        pts_sorted = sorted(pts, key=lambda q: int(q.get("frameNumber", 0)))
        for i in range(1, len(pts_sorted)):
            cx0, cy0 = _bbox_center(pts_sorted[i - 1])
            cx1, cy1 = _bbox_center(pts_sorted[i])
            d = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
            if d > pid_max_jump:
                pid_max_jump = d

    # Per-frame-pair Hungarian-IoU continuity: at each consecutive frame F,
    # F+1 with the same set of primary PIDs, find the optimal IoU
    # assignment from F's bboxes to F+1's bboxes via linear_sum_assignment.
    # If the optimal mapping differs from PID-identity (i.e. PID P at F is
    # best matched by a different PID at F+1), the frame pair witnesses a
    # swap event.
    #
    # We DELIBERATELY do not skip frame gaps here. Within-rally swaps
    # frequently happen across occlusion (the user's b5fb0594/r10 case:
    # "starts GOOD, then NEW p1<->p2 within-team swap after occlusion"),
    # and the post-occlusion frame's bbox IoU against pre-occlusion bboxes
    # is exactly the signal we need — players that re-emerge with swapped
    # labels still occupy nearly-the-same physical positions.
    #
    # We restrict the swap detector to PIDs in EXPECTED_PRIMARY_PIDS (1-4)
    # so non-primary tracks that happen to appear and disappear don't
    # generate noise. The signal we care about is which canonical PID is
    # attached to which physical player.
    import numpy as np
    from scipy.optimize import linear_sum_assignment  # local import — lazy

    by_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for p in positions:
        tid = p.get("trackId")
        if tid is None:
            continue
        tid_i = int(tid)
        if tid_i not in EXPECTED_PRIMARY_PIDS:
            continue
        f = int(p.get("frameNumber", 0))
        by_frame.setdefault(f, {})[tid_i] = p

    frames_sorted = sorted(by_frame.keys())
    pid_swap_events = 0
    for i in range(1, len(frames_sorted)):
        f0 = frames_sorted[i - 1]
        f1 = frames_sorted[i]
        common = sorted(set(by_frame[f0].keys()) & set(by_frame[f1].keys()))
        if len(common) < 2:
            continue
        n = len(common)
        cost = np.zeros((n, n), dtype=float)
        for r, pid_r in enumerate(common):
            box_r = _bbox_xyxy(by_frame[f0][pid_r])
            for c, pid_c in enumerate(common):
                # Cost = -IoU so linear_sum_assignment (which minimizes)
                # finds the maximum-IoU bipartite matching.
                cost[r, c] = -_iou(box_r, _bbox_xyxy(by_frame[f1][pid_c]))
        row_ind, col_ind = linear_sum_assignment(cost)
        # Optimal mapping: row r → col col_ind[r]. Compare to identity (r → r).
        for r, c in zip(row_ind, col_ind):
            if r == c:
                continue
            best_iou = -cost[r, c]
            self_iou = -cost[r, r]
            # Only count as a swap when the optimal partner has materially
            # higher IoU than self AND the IoU advantage is at least
            # SWAP_IOU_MARGIN. Tiny differences are within noise; but a
            # 0.1 IoU margin separates "Hungarian is confident this is a
            # swap" from "either assignment fits about equally well".
            if (
                best_iou > IOU_SAME_TRACK_THRESHOLD
                and best_iou - self_iou >= SWAP_IOU_MARGIN
            ):
                pid_swap_events += 1

    # Verdict heuristic — first matching condition wins. Short-circuit
    # ordering ensures derived_shape labels the dominant mode.
    #
    # Rule 1 (missing primary PIDs): Hungarian must have assigned all four
    # canonical PIDs {1,2,3,4}. A missing PID means the cross-rally solver
    # declined to label one of the primary tracks and the editor will
    # render that track as either an unmapped raw ID or as collision-shifted
    # 101+. This is the dominant pattern in three panel BAD cases.
    #
    # Rule 2 (substantial unlabeled): a few -1 frames near rally edges are
    # not user-visible. We require either a high absolute count or a high
    # fraction before flagging.
    #
    # Rule 3 (within-rally swap): per-frame IoU continuity check, with
    # tolerance for one isolated false positive at rally edges.
    #
    # Rule 4 (trajectory teleport): a same-PID center jump > 30% of the
    # frame is unphysical and almost always indicates an identity switch.
    missing_primary = EXPECTED_PRIMARY_PIDS - pids
    extra_pids = pids - EXPECTED_PRIMARY_PIDS

    unlabeled_frac = n_unlabeled / max(1, n_positions)
    is_substantial_unlabeled = (
        n_unlabeled >= UNLABELED_ABS_THRESHOLD
        or unlabeled_frac >= UNLABELED_FRAC_THRESHOLD
    )

    half_shift_max, xrange_overlap_max, drift_pid, overlap_pairs = (
        _slow_drift_signal(by_pid)
    )
    has_slow_drift = (
        half_shift_max > HALF_SHIFT_THRESHOLD
        and xrange_overlap_max > XRANGE_OVERLAP_THRESHOLD
    )

    if missing_primary:
        verdict = "BAD"
        shape = (
            f"hungarian_drop (missing PIDs {sorted(missing_primary)}; "
            f"have {sorted(pids)})"
        )
    elif is_substantial_unlabeled:
        verdict = "BAD"
        shape = (
            f"unlabeled ({n_unlabeled} frames with trackId=-1, "
            f"{unlabeled_frac:.1%} of positions)"
        )
    elif pid_swap_events >= SWAP_EVENTS_THRESHOLD:
        verdict = "BAD"
        shape = f"within_rally_swap ({pid_swap_events} per-frame swap events)"
    elif pid_max_jump > TRAJECTORY_JUMP_THRESHOLD:
        verdict = "BAD"
        shape = f"trajectory_jump (max={pid_max_jump:.2f})"
    elif has_slow_drift:
        verdict = "BAD"
        shape = (
            f"slow_drift (PID{drift_pid} half-shift={half_shift_max:.2f}, "
            f"xrange-overlap={xrange_overlap_max:.2f})"
        )
    else:
        verdict = "GOOD"
        shape = (
            f"clean (extras: {sorted(extra_pids)})" if extra_pids else "clean"
        )

    return RallySignals(
        rally_tag=rally_tag, rally_id=rally_id,
        n_positions=n_positions,
        n_distinct_pids=len(pids), distinct_pids=sorted(pids),
        n_unlabeled=n_unlabeled,
        pid_max_jump=pid_max_jump,
        pid_swap_events=pid_swap_events,
        half_shift_max=half_shift_max,
        xrange_overlap_max=xrange_overlap_max,
        derived_verdict=verdict, derived_shape=shape,
    )


def main() -> None:
    print(
        f"{'rally':<14} {'kind':<5} {'expect':<6} {'actual':<6} "
        f"{'agree':<5} {'pids':<8} {'unlbl':<6} {'jump':<6} "
        f"{'swaps':<5}  {'derived_shape'}"
    )
    print("-" * 120)

    n_agree = 0
    n_disagree = 0
    disagreements: list[tuple[PanelVerdict, RallySignals]] = []

    with get_connection() as conn, conn.cursor() as cur:
        for v in PANEL_2026_05_01:
            resolved = _resolve_rally_id(cur, v.video_id, v.rally_idx)
            if resolved is None:
                print(
                    f"{v.rally_tag:<14} {('CTRL' if v.is_control else 'PNL'):<5} "
                    f"{v.expected_verdict:<6} {'?':<6} {'NO':<5} "
                    f"<rally not found in DB>"
                )
                n_disagree += 1
                continue
            rally_id, _start_ms, _end_ms = resolved
            positions = _load_positions(cur, rally_id)
            signals = _compute_signals(positions, v.rally_tag, rally_id)

            agrees = signals.derived_verdict == v.expected_verdict
            if agrees:
                n_agree += 1
            else:
                n_disagree += 1
                disagreements.append((v, signals))

            kind = "CTRL" if v.is_control else "PNL"
            pids_str = ",".join(str(p) for p in signals.distinct_pids)
            print(
                f"{v.rally_tag:<14} {kind:<5} "
                f"{v.expected_verdict:<6} {signals.derived_verdict:<6} "
                f"{('YES' if agrees else 'NO'):<5} "
                f"{pids_str:<8} "
                f"{signals.n_unlabeled:<6} "
                f"{signals.pid_max_jump:<6.2f} "
                f"{signals.pid_swap_events:<5}  "
                f"{signals.derived_shape}"
            )

    total = len(PANEL_2026_05_01)
    print()
    print(f"AGREES: {n_agree}/{total}")
    if disagreements:
        print()
        print("Disagreements (expected_shape vs derived_shape):")
        for pv, sig in disagreements:
            kind = "CTRL" if pv.is_control else "PANEL"
            print(
                f"  {pv.rally_tag:<14} {kind:<5} "
                f"expected={pv.expected_verdict} ({pv.expected_shape})"
            )
            print(
                f"  {'':<14} {'':<5} "
                f"derived ={sig.derived_verdict} ({sig.derived_shape})"
            )


if __name__ == "__main__":
    main()
