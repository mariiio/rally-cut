"""Ball-tracking failure-mode detectors.

Pure functions that classify ball-tracking gaps against ground truth into
root-cause buckets. Feeds the overlay renderer and cross-rally report in
analysis/scripts/diagnose_ball_gaps.py.

All detectors operate on normalized coordinates (0-1) and convert to pixels
only for the returned event payloads, so callers can render on either the
full-resolution source video or a rescaled view without re-thinking units.

Failure buckets (the taxonomy that rolls up into the per-rally budget):

- matched: GT frame had a prediction within the match threshold.
- missed_no_raw: GT frame had no raw WASB detection within the match
  threshold — the detector never saw it. Fix path: retrain / threshold.
- missed_filter_killed: GT frame had a raw detection, but the filter
  dropped it. Fix path: filter config or a new stage.
- wrong_object: prediction exists at the GT frame but is beyond the wrong
  object threshold from GT. Enriched with nearest_player_distance when
  player tracks are available (player-lock drift).
- teleport: two consecutive filtered predictions separated by more pixels
  per frame than any plausible ball motion.
- stationary_cluster: run of raw detections so spatially tight that the
  likely source is a background distractor (flag, court-side ball).
- two_ball: multiple simultaneous high-confidence raw detections spatially
  far apart over a sustained window — both a game ball and a warmup ball
  visible.

See analysis/.claude/skills/tracking-diagnosis/SKILL.md for the pipeline
stage order these detectors reason about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from rallycut.labeling.ground_truth import GroundTruthPosition
from rallycut.tracking.ball_tracker import BallPosition

# ---------------------------------------------------------------------------
# Event payloads
# ---------------------------------------------------------------------------


@dataclass
class MissedStreakEvent:
    start_frame: int
    end_frame: int
    length: int
    had_raw_detection_frames: int  # subset of the streak where raw WASB did fire
    mean_gt_xy: tuple[float, float]  # normalized

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "missed_streak",
            "start": self.start_frame,
            "end": self.end_frame,
            "length": self.length,
            "had_raw_detection_frames": self.had_raw_detection_frames,
            "mean_gt_xy": [self.mean_gt_xy[0], self.mean_gt_xy[1]],
        }


@dataclass
class TeleportEvent:
    frame: int  # the frame the jump lands on
    prev_xy: tuple[float, float]  # normalized
    curr_xy: tuple[float, float]
    dv_px: float  # per-frame displacement in pixels
    frames_skipped: int  # number of frames between prev and curr (usually 1)
    cross_segment: bool  # True when jumping across a filter-pruned gap

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "teleport",
            "frame": self.frame,
            "prev_xy": [self.prev_xy[0], self.prev_xy[1]],
            "curr_xy": [self.curr_xy[0], self.curr_xy[1]],
            "dv_px": self.dv_px,
            "frames_skipped": self.frames_skipped,
            "cross_segment": self.cross_segment,
        }


@dataclass
class WrongObjectEvent:
    frame: int
    gt_xy: tuple[float, float]
    pred_xy: tuple[float, float]
    error_px: float
    pred_confidence: float
    nearest_player_distance_px: float | None  # None when player positions unavailable
    interpolated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "wrong_object",
            "frame": self.frame,
            "gt_xy": [self.gt_xy[0], self.gt_xy[1]],
            "pred_xy": [self.pred_xy[0], self.pred_xy[1]],
            "error_px": self.error_px,
            "pred_confidence": self.pred_confidence,
            "nearest_player_distance_px": self.nearest_player_distance_px,
            "interpolated": self.interpolated,
        }


@dataclass
class StationaryClusterEvent:
    start_frame: int
    end_frame: int
    length: int
    median_xy: tuple[float, float]
    spread_norm: float  # max(std(x), std(y)) in normalized coords
    mean_confidence: float
    passed_filter: bool  # True when this cluster survived the current filter

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "stationary_cluster",
            "start": self.start_frame,
            "end": self.end_frame,
            "length": self.length,
            "median_xy": [self.median_xy[0], self.median_xy[1]],
            "spread_norm": self.spread_norm,
            "mean_confidence": self.mean_confidence,
            "passed_filter": self.passed_filter,
        }


@dataclass
class RevisitClusterEvent:
    """Spatial hot-spot that the tracker returns to across the rally.

    Unlike StationaryClusterEvent (temporally contiguous frames at one
    position), this catches short-lived revisits interspersed with real ball
    detections — the A-B-A-B pattern of a distractor pulling the WASB peak
    for 1–3 frames at a time while the ball is elsewhere. The signature is
    high spatial density of raw detections in a small bin whose occupancy
    frames are NOT contiguous.
    """

    center_xy: tuple[float, float]   # normalized bin center
    visit_frames: list[int]          # sorted frame numbers that land in the bin
    visit_count: int
    mean_confidence: float
    passed_filter: bool              # True if any visit survived the current filter

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "revisit_cluster",
            "center_xy": [self.center_xy[0], self.center_xy[1]],
            "visit_frames": self.visit_frames,
            "visit_count": self.visit_count,
            "mean_confidence": self.mean_confidence,
            "passed_filter": self.passed_filter,
        }


@dataclass
class TwoBallEvent:
    start_frame: int
    end_frame: int
    length: int
    cluster_a_xy: tuple[float, float]
    cluster_b_xy: tuple[float, float]
    separation_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "two_ball",
            "start": self.start_frame,
            "end": self.end_frame,
            "length": self.length,
            "cluster_a_xy": [self.cluster_a_xy[0], self.cluster_a_xy[1]],
            "cluster_b_xy": [self.cluster_b_xy[0], self.cluster_b_xy[1]],
            "separation_px": self.separation_px,
        }


@dataclass
class FilterKillEvent:
    frame: int
    xy: tuple[float, float]
    raw_confidence: float
    stage: str  # "segment_pruning" | "exit_ghost" | "outlier_removal" | "blip_removal" | "oscillation_pruning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "filter_kill",
            "frame": self.frame,
            "xy": [self.xy[0], self.xy[1]],
            "raw_confidence": self.raw_confidence,
            "stage": self.stage,
        }


@dataclass
class FailureBudget:
    """Per-rally breakdown of GT frames into root-cause buckets.

    matched + missed_no_raw + missed_filter_killed + wrong_object +
    interpolated_correct + interpolated_wrong = total_gt_frames
    (each GT frame lands in exactly one of these).

    teleport_count, stationary_cluster_count, two_ball_count are *event*
    counts, orthogonal to the frame buckets above — they're shape evidence,
    not frame accounting.

    smoothness_px_p90 is p90 of per-frame prediction acceleration magnitude
    in pixels — smooth parabolic motion sits near gravity (~10 px/frame²
    at 30 fps on a 1920-wide frame), jittery tracking spikes much higher.
    """

    rally_id: str
    total_gt_frames: int
    matched: int
    missed_no_raw: int
    missed_filter_killed: int
    wrong_object: int
    interpolated_correct: int
    interpolated_wrong: int
    teleport_count: int
    stationary_cluster_count: int
    stationary_cluster_frames: int  # total frames spanned by stationary clusters
    revisit_cluster_count: int      # spatial hot-spots revisited across the rally
    revisit_cluster_visits: int     # total visits summed across revisit clusters
    two_ball_count: int
    frame_offset: int
    match_threshold_px: float
    wrong_object_threshold_px: float
    teleport_v_max_px_per_frame: float
    smoothness_px_p90: float
    smoothness_px_median: float
    gt_mode: str  # "keyframes" | "interpolated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rally_id": self.rally_id,
            "total_gt_frames": self.total_gt_frames,
            "matched": self.matched,
            "missed_no_raw": self.missed_no_raw,
            "missed_filter_killed": self.missed_filter_killed,
            "wrong_object": self.wrong_object,
            "interpolated_correct": self.interpolated_correct,
            "interpolated_wrong": self.interpolated_wrong,
            "teleport_count": self.teleport_count,
            "stationary_cluster_count": self.stationary_cluster_count,
            "stationary_cluster_frames": self.stationary_cluster_frames,
            "revisit_cluster_count": self.revisit_cluster_count,
            "revisit_cluster_visits": self.revisit_cluster_visits,
            "two_ball_count": self.two_ball_count,
            "frame_offset": self.frame_offset,
            "smoothness_px_p90": self.smoothness_px_p90,
            "smoothness_px_median": self.smoothness_px_median,
            "gt_mode": self.gt_mode,
            "thresholds": {
                "match_px": self.match_threshold_px,
                "wrong_object_px": self.wrong_object_threshold_px,
                "teleport_v_max_px_per_frame": self.teleport_v_max_px_per_frame,
            },
        }


@dataclass
class FailureEvents:
    """All events detected for a rally. Populated by classify_rally()."""

    rally_id: str
    budget: FailureBudget
    missed_streaks: list[MissedStreakEvent] = field(default_factory=list)
    teleports: list[TeleportEvent] = field(default_factory=list)
    wrong_objects: list[WrongObjectEvent] = field(default_factory=list)
    stationary_clusters: list[StationaryClusterEvent] = field(default_factory=list)
    revisit_clusters: list[RevisitClusterEvent] = field(default_factory=list)
    two_ball_events: list[TwoBallEvent] = field(default_factory=list)
    filter_kills: list[FilterKillEvent] = field(default_factory=list)
    per_frame_status: dict[int, str] = field(default_factory=dict)
    # "matched" | "missed_no_raw" | "missed_filter_killed" | "wrong_object"
    # | "interpolated_correct" | "interpolated_wrong"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rally_id": self.rally_id,
            "budget": self.budget.to_dict(),
            "missed_streaks": [e.to_dict() for e in self.missed_streaks],
            "teleports": [e.to_dict() for e in self.teleports],
            "wrong_objects": [e.to_dict() for e in self.wrong_objects],
            "stationary_clusters": [e.to_dict() for e in self.stationary_clusters],
            "revisit_clusters": [e.to_dict() for e in self.revisit_clusters],
            "two_ball_events": [e.to_dict() for e in self.two_ball_events],
            "filter_kills": [e.to_dict() for e in self.filter_kills],
            "per_frame_status": {
                str(k): v for k, v in sorted(self.per_frame_status.items())
            },
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _px_distance(
    a: tuple[float, float],
    b: tuple[float, float],
    width: int,
    height: int,
) -> float:
    dx = (a[0] - b[0]) * width
    dy = (a[1] - b[1]) * height
    return math.sqrt(dx * dx + dy * dy)


def _group_consecutive(frames: list[int]) -> list[tuple[int, int]]:
    """Group sorted frame numbers into [start, end] inclusive runs."""
    if not frames:
        return []
    sorted_frames = sorted(frames)
    runs: list[tuple[int, int]] = []
    start = prev = sorted_frames[0]
    for f in sorted_frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            runs.append((start, prev))
            start = prev = f
    runs.append((start, prev))
    return runs


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def detect_missed_streaks(
    *,
    gt_ball: list[GroundTruthPosition],
    pred_by_frame: dict[int, BallPosition],
    raw_by_frame: dict[int, BallPosition],
    width: int,
    height: int,
    match_threshold_px: float = 50.0,
    min_streak_length: int = 5,
) -> list[MissedStreakEvent]:
    """Runs of GT frames without a near-enough filtered prediction.

    `had_raw_detection_frames` counts frames within the streak where the raw
    WASB output had *any* position within match_threshold_px of GT — i.e. the
    filter had something to keep but didn't. That number drives the
    missed_no_raw vs missed_filter_killed split elsewhere.
    """
    gt_by_frame = {p.frame_number: p for p in gt_ball}
    missed_frames: list[int] = []
    for frame, gt_pos in gt_by_frame.items():
        pred = pred_by_frame.get(frame)
        if pred is None:
            missed_frames.append(frame)
            continue
        err = _px_distance((pred.x, pred.y), (gt_pos.x, gt_pos.y), width, height)
        if err > match_threshold_px:
            # Not a miss for this detector — it's a wrong-object/over-threshold case.
            # detect_wrong_object picks it up; we only report pure absences here.
            continue
    events: list[MissedStreakEvent] = []
    for start, end in _group_consecutive(missed_frames):
        length = end - start + 1
        if length < min_streak_length:
            continue
        had_raw = 0
        xs: list[float] = []
        ys: list[float] = []
        for f in range(start, end + 1):
            gt_pos = gt_by_frame[f]
            xs.append(gt_pos.x)
            ys.append(gt_pos.y)
            raw = raw_by_frame.get(f)
            if raw is not None and raw.confidence > 0:
                err = _px_distance(
                    (raw.x, raw.y), (gt_pos.x, gt_pos.y), width, height
                )
                if err <= match_threshold_px:
                    had_raw += 1
        events.append(
            MissedStreakEvent(
                start_frame=start,
                end_frame=end,
                length=length,
                had_raw_detection_frames=had_raw,
                mean_gt_xy=(sum(xs) / length, sum(ys) / length),
            )
        )
    return events


def detect_teleports(
    *,
    predictions: list[BallPosition],
    width: int,
    height: int,
    v_max_px_per_frame: float = 60.0,
    pruned_segment_boundaries: set[int] | None = None,
) -> list[TeleportEvent]:
    """Flag consecutive filtered predictions with implausible per-frame motion.

    pruned_segment_boundaries (if provided) is the set of frame numbers at
    which the filter's segment pruning introduced a split — jumps across
    those boundaries are labeled cross_segment=True, meaning the chain-based
    anchor connector fused two unrelated segments. Intra-segment teleports
    are rarer but real (fast spikes); cross-segment teleports are almost
    always bad.
    """
    if len(predictions) < 2:
        return []
    sorted_pred = sorted(predictions, key=lambda p: p.frame_number)
    events: list[TeleportEvent] = []
    for i in range(1, len(sorted_pred)):
        prev = sorted_pred[i - 1]
        curr = sorted_pred[i]
        frames_skipped = curr.frame_number - prev.frame_number
        if frames_skipped <= 0:
            continue
        dv = _px_distance(
            (prev.x, prev.y), (curr.x, curr.y), width, height
        ) / max(frames_skipped, 1)
        if dv <= v_max_px_per_frame:
            continue
        cross_segment = False
        if pruned_segment_boundaries:
            cross_segment = any(
                prev.frame_number < b <= curr.frame_number
                for b in pruned_segment_boundaries
            )
        events.append(
            TeleportEvent(
                frame=curr.frame_number,
                prev_xy=(prev.x, prev.y),
                curr_xy=(curr.x, curr.y),
                dv_px=dv,
                frames_skipped=frames_skipped,
                cross_segment=cross_segment,
            )
        )
    return events


def detect_wrong_object(
    *,
    gt_ball: list[GroundTruthPosition],
    pred_by_frame: dict[int, BallPosition],
    interpolated_frames: set[int],
    width: int,
    height: int,
    player_positions_by_frame: dict[int, list[tuple[float, float]]] | None = None,
    threshold_px: float = 80.0,
) -> list[WrongObjectEvent]:
    """Prediction exists at a GT frame but lands far from GT (> threshold).

    Enriches each event with nearest_player_distance_px when player
    positions are provided — a pred glued to a player is almost always a
    body-lock drift, not a real fast shot.
    """
    events: list[WrongObjectEvent] = []
    for gt_pos in gt_ball:
        pred = pred_by_frame.get(gt_pos.frame_number)
        if pred is None:
            continue
        err = _px_distance(
            (pred.x, pred.y), (gt_pos.x, gt_pos.y), width, height
        )
        if err <= threshold_px:
            continue
        nearest = None
        if player_positions_by_frame is not None:
            players_here = player_positions_by_frame.get(gt_pos.frame_number, [])
            if players_here:
                nearest = min(
                    _px_distance((pred.x, pred.y), p, width, height)
                    for p in players_here
                )
        events.append(
            WrongObjectEvent(
                frame=gt_pos.frame_number,
                gt_xy=(gt_pos.x, gt_pos.y),
                pred_xy=(pred.x, pred.y),
                error_px=err,
                pred_confidence=pred.confidence,
                nearest_player_distance_px=nearest,
                interpolated=gt_pos.frame_number in interpolated_frames,
            )
        )
    return events


def detect_stationary_clusters(
    *,
    raw_positions: list[BallPosition],
    width: int,
    height: int,
    min_frames: int = 20,
    max_spread_norm: float = 0.005,
    min_confidence: float = 0.3,
    filtered_frames: set[int] | None = None,
) -> list[StationaryClusterEvent]:
    """Sliding-window detector for background-distractor clusters.

    Walks contiguous runs of high-confidence raw detections and flags any
    run of >= min_frames whose x/y spread stays within max_spread_norm of
    screen. That's the disabled stationarity filter's exact target — we
    re-measure it non-destructively so the report can attribute a bucket
    of gaps to background distractors.

    passed_filter = True means at least one frame in the cluster survived
    the current filter pipeline (i.e. the distractor still pollutes the
    output). When filtered_frames is None we skip the flag.
    """
    if not raw_positions:
        return []
    high_conf = [
        p for p in raw_positions if p.confidence >= min_confidence
    ]
    if not high_conf:
        return []
    sorted_pos = sorted(high_conf, key=lambda p: p.frame_number)

    events: list[StationaryClusterEvent] = []
    n = len(sorted_pos)
    i = 0
    while i < n:
        # Greedily extend a window so long as its spread stays tight.
        start = i
        xs = [sorted_pos[i].x]
        ys = [sorted_pos[i].y]
        j = i + 1
        while j < n:
            xs_test = xs + [sorted_pos[j].x]
            ys_test = ys + [sorted_pos[j].y]
            spread_x = max(xs_test) - min(xs_test)
            spread_y = max(ys_test) - min(ys_test)
            if max(spread_x, spread_y) > max_spread_norm:
                break
            # Also require contiguity in frame numbers (allow gaps up to 3).
            if sorted_pos[j].frame_number - sorted_pos[j - 1].frame_number > 3:
                break
            xs = xs_test
            ys = ys_test
            j += 1
        length_frames = sorted_pos[j - 1].frame_number - sorted_pos[start].frame_number + 1
        if j - start >= min_frames and length_frames >= min_frames:
            xs_sorted = sorted(xs)
            ys_sorted = sorted(ys)
            mid = len(xs_sorted) // 2
            median_xy = (xs_sorted[mid], ys_sorted[mid])
            spread = max(max(xs) - min(xs), max(ys) - min(ys))
            mean_conf = sum(p.confidence for p in sorted_pos[start:j]) / (j - start)
            passed = False
            if filtered_frames is not None:
                passed = any(
                    sorted_pos[k].frame_number in filtered_frames
                    for k in range(start, j)
                )
            events.append(
                StationaryClusterEvent(
                    start_frame=sorted_pos[start].frame_number,
                    end_frame=sorted_pos[j - 1].frame_number,
                    length=j - start,
                    median_xy=median_xy,
                    spread_norm=spread,
                    mean_confidence=mean_conf,
                    passed_filter=passed,
                )
            )
            i = j
        else:
            i += 1
    return events


def detect_revisit_clusters(
    *,
    raw_positions: list[BallPosition],
    width: int,
    height: int,
    bin_size_norm: float = 0.01,
    min_visits: int = 3,
    min_confidence: float = 0.3,
    max_contiguous_fraction: float = 0.6,
    min_teleport_confirmations: int = 2,
    teleport_distance_px: float = 200.0,
    teleport_window_frames: int = 3,
    filtered_frames: set[int] | None = None,
) -> list[RevisitClusterEvent]:
    """Catch stationary distractors the tracker revisits non-contiguously.

    Discretizes high-confidence raw positions into bin_size_norm-sized
    spatial bins. A bin is a *revisit cluster* (distractor) when:
      1. It has ≥ min_visits positions.
      2. The frame set isn't mostly one contiguous run (distinguishes flag
         revisits from a ball that lingers at one point).
      3. At least min_teleport_confirmations of the visits have a raw
         detection ≥ teleport_distance_px away within ±teleport_window_frames
         — the A-B-A-B signature the user observed on 9db9cb6b, where the
         ball is elsewhere at the moment the tracker dips into the flag.

    Condition 3 is what separates "tracker keeps returning to a distractor"
    from "ball legitimately passes through the same spot twice". Serve tosses
    and ball re-entries are stationary for a few frames at each visit and
    don't have a far-away raw nearby — they fail the confirmation test.
    """
    if not raw_positions:
        return []
    hi = [p for p in raw_positions if p.confidence >= min_confidence]
    if not hi:
        return []

    by_frame: dict[int, BallPosition] = {p.frame_number: p for p in hi}
    bins: dict[tuple[int, int], list[BallPosition]] = {}
    for p in hi:
        bx = int(p.x / bin_size_norm)
        by = int(p.y / bin_size_norm)
        bins.setdefault((bx, by), []).append(p)

    events: list[RevisitClusterEvent] = []
    for (bx, by), members in bins.items():
        if len(members) < min_visits:
            continue
        frames = sorted({m.frame_number for m in members})
        # Largest contiguous run inside the cluster's frame set.
        longest_run = 1
        run = 1
        for i in range(1, len(frames)):
            if frames[i] == frames[i - 1] + 1:
                run += 1
                if run > longest_run:
                    longest_run = run
            else:
                run = 1
        if longest_run / len(frames) > max_contiguous_fraction:
            # Sustained cluster — stationary filter territory, skip.
            continue

        xs = [m.x for m in members]
        ys = [m.y for m in members]
        center = (sum(xs) / len(xs), sum(ys) / len(ys))

        # Teleport-LANDING confirmation: for each visit f, check whether
        # the cluster center is far from the raw at f-1 AND/OR f+1. That
        # means the tracker LANDED in the cluster via a jump (not smooth
        # motion through it). Ball-path intersections (ball legitimately
        # passes through the same bin at two different moments) do NOT
        # satisfy this because the ball's approach is smooth — the neighbors
        # are near the cluster. A true distractor is reached by discrete
        # jump, with the ball's actual position elsewhere a frame earlier
        # or later. Require at least min_teleport_confirmations landings.
        confirmations = 0
        for f in frames:
            is_landing = False
            for df in (-1, 1):
                neighbor = by_frame.get(f + df)
                if neighbor is None:
                    continue
                if _px_distance(
                    (neighbor.x, neighbor.y), center, width, height
                ) >= teleport_distance_px:
                    is_landing = True
                    break
            if is_landing:
                confirmations += 1
        if confirmations < min_teleport_confirmations:
            continue

        mean_conf = sum(m.confidence for m in members) / len(members)
        passed = False
        if filtered_frames is not None:
            passed = any(f in filtered_frames for f in frames)
        events.append(
            RevisitClusterEvent(
                center_xy=center,
                visit_frames=frames,
                visit_count=len(frames),
                mean_confidence=mean_conf,
                passed_filter=passed,
            )
        )
    events.sort(key=lambda e: -e.visit_count)
    return events


def detect_two_ball_divergence(
    *,
    raw_multi_by_frame: dict[int, list[BallPosition]],
    width: int,
    height: int,
    min_separation_px: float = 200.0,
    min_confidence: float = 0.5,
    min_cluster_frames: int = 30,
) -> list[TwoBallEvent]:
    """Sustained frames with 2+ high-confidence raw detections far apart.

    WASB normally emits one position per frame (top of heatmap), so this
    detector only fires when callers pre-pass a multi-detection view
    (e.g. all local maxima above threshold). When raw_multi_by_frame is
    empty or univariate, this detector returns [].
    """
    if not raw_multi_by_frame:
        return []
    divergent_frames: list[tuple[int, BallPosition, BallPosition]] = []
    for frame, dets in raw_multi_by_frame.items():
        strong = [d for d in dets if d.confidence >= min_confidence]
        if len(strong) < 2:
            continue
        # Find the pair with maximum separation above threshold.
        best: tuple[float, BallPosition, BallPosition] | None = None
        for a in range(len(strong)):
            for b in range(a + 1, len(strong)):
                sep = _px_distance(
                    (strong[a].x, strong[a].y),
                    (strong[b].x, strong[b].y),
                    width,
                    height,
                )
                if sep < min_separation_px:
                    continue
                if best is None or sep > best[0]:
                    best = (sep, strong[a], strong[b])
        if best is not None:
            divergent_frames.append((frame, best[1], best[2]))
    if not divergent_frames:
        return []
    divergent_frames.sort(key=lambda t: t[0])
    events: list[TwoBallEvent] = []
    run_start = 0
    for i in range(1, len(divergent_frames) + 1):
        at_break = (
            i == len(divergent_frames)
            or divergent_frames[i][0] - divergent_frames[i - 1][0] > 3
        )
        if at_break:
            run = divergent_frames[run_start:i]
            length = run[-1][0] - run[0][0] + 1
            if length >= min_cluster_frames:
                # Use the frame-of-max-separation to represent the clusters.
                apex = max(
                    run,
                    key=lambda t: _px_distance(
                        (t[1].x, t[1].y), (t[2].x, t[2].y), width, height
                    ),
                )
                sep_px = _px_distance(
                    (apex[1].x, apex[1].y),
                    (apex[2].x, apex[2].y),
                    width,
                    height,
                )
                events.append(
                    TwoBallEvent(
                        start_frame=run[0][0],
                        end_frame=run[-1][0],
                        length=length,
                        cluster_a_xy=(apex[1].x, apex[1].y),
                        cluster_b_xy=(apex[2].x, apex[2].y),
                        separation_px=sep_px,
                    )
                )
            run_start = i
    return events


def attribute_filter_kills(
    *,
    stages: dict[str, list[BallPosition]],
    stage_order: list[str],
) -> list[FilterKillEvent]:
    """Label every raw→filtered drop with the earliest stage that killed it.

    Expects the same dict shape produced by
    diagnose_ball_tracking.run_pipeline_stages: keys like "0_raw",
    "1_segment_pruned", "2_ghost_removed", etc. Walks pairs in stage_order
    and records the first stage where a given (frame, xy) disappears.
    """
    if len(stage_order) < 2:
        return []
    events: list[FilterKillEvent] = []
    # Index positions by frame for every stage for quick lookup.
    stage_maps: dict[str, dict[int, BallPosition]] = {
        name: {p.frame_number: p for p in stages.get(name, [])}
        for name in stage_order
    }
    for idx in range(len(stage_order) - 1):
        prev_name = stage_order[idx]
        next_name = stage_order[idx + 1]
        prev_map = stage_maps[prev_name]
        next_map = stage_maps[next_name]
        for frame, pos in prev_map.items():
            if frame not in next_map:
                stage_label = _stage_kill_label(prev_name, next_name)
                events.append(
                    FilterKillEvent(
                        frame=frame,
                        xy=(pos.x, pos.y),
                        raw_confidence=pos.confidence,
                        stage=stage_label,
                    )
                )
    return events


def _stage_kill_label(prev_name: str, next_name: str) -> str:
    """Map stage-pair transitions to a human-friendly stage label."""
    key = next_name.split("_", 1)[1] if "_" in next_name else next_name
    mapping = {
        "segment_pruned": "segment_pruning",
        "ghost_removed": "exit_ghost",
        "oscillation_pruned": "oscillation_pruning",
        "outlier_removed": "outlier_removal",
        "blip_removed": "blip_removal",
        "repruned": "re_prune",
        "interpolated": "interpolation_gap",  # should not drop frames, defensive
    }
    return mapping.get(key, key)


# ---------------------------------------------------------------------------
# Rally-level classifier
# ---------------------------------------------------------------------------


def compute_smoothness(
    *,
    pred_by_frame: dict[int, BallPosition],
    width: int,
    height: int,
) -> tuple[float, float]:
    """p90 and median of per-frame prediction acceleration magnitude.

    Uses second finite difference over three consecutive frames:
    a_n = |(p_{n+1} - 2·p_n + p_{n-1})|_px

    Skipped at segment boundaries (non-consecutive frames).
    """
    if len(pred_by_frame) < 3:
        return 0.0, 0.0
    frames = sorted(pred_by_frame.keys())
    accels: list[float] = []
    for i in range(1, len(frames) - 1):
        fp, fc, fn = frames[i - 1], frames[i], frames[i + 1]
        if fc - fp != 1 or fn - fc != 1:
            continue
        p_prev = pred_by_frame[fp]
        p_curr = pred_by_frame[fc]
        p_next = pred_by_frame[fn]
        ax = (p_next.x - 2 * p_curr.x + p_prev.x) * width
        ay = (p_next.y - 2 * p_curr.y + p_prev.y) * height
        accels.append(math.sqrt(ax * ax + ay * ay))
    if not accels:
        return 0.0, 0.0
    accels.sort()
    p90 = accels[min(int(len(accels) * 0.9), len(accels) - 1)]
    median = accels[len(accels) // 2]
    return p90, median


def classify_rally(
    *,
    rally_id: str,
    gt_ball: list[GroundTruthPosition],
    predictions: list[BallPosition],
    raw_positions: list[BallPosition],
    stages: dict[str, list[BallPosition]],
    stage_order: list[str],
    width: int,
    height: int,
    frame_offset: int = 0,
    match_threshold_px: float = 50.0,
    wrong_object_threshold_px: float = 80.0,
    teleport_v_max_px_per_frame: float = 60.0,
    player_positions_by_frame: dict[int, list[tuple[float, float]]] | None = None,
    raw_multi_by_frame: dict[int, list[BallPosition]] | None = None,
    gt_mode: str = "interpolated",
) -> FailureEvents:
    """Run every detector and roll results into a per-rally FailureEvents.

    `frame_offset` is the constant lag between GT frames and prediction
    frames — pass the value returned by find_optimal_frame_offset so
    predictions align with GT here (pred frame N maps to GT frame N - offset).
    """
    # Shift predictions to GT frame space.
    pred_by_frame: dict[int, BallPosition] = {}
    for p in predictions:
        shifted = p.frame_number - frame_offset
        existing = pred_by_frame.get(shifted)
        if existing is None or p.confidence > existing.confidence:
            pred_by_frame[shifted] = p

    raw_by_frame: dict[int, BallPosition] = {}
    for p in raw_positions:
        shifted = p.frame_number - frame_offset
        existing = raw_by_frame.get(shifted)
        if existing is None or p.confidence > existing.confidence:
            raw_by_frame[shifted] = p

    interpolated_frames = {
        p.frame_number - frame_offset
        for p in predictions
        if abs(p.confidence - 0.5) < 1e-6
        # interpolated_confidence default is 0.5; non-interpolated positions
        # rarely land exactly at 0.5 but this is a best-effort flag.
    }

    filter_kill_events = attribute_filter_kills(
        stages=stages, stage_order=stage_order
    )
    # Shift filter-kill frames to GT space too.
    filter_kill_events = [
        FilterKillEvent(
            frame=e.frame - frame_offset,
            xy=e.xy,
            raw_confidence=e.raw_confidence,
            stage=e.stage,
        )
        for e in filter_kill_events
    ]
    kill_frames_by_stage: dict[int, str] = {}
    for e in filter_kill_events:
        kill_frames_by_stage.setdefault(e.frame, e.stage)

    missed = detect_missed_streaks(
        gt_ball=gt_ball,
        pred_by_frame=pred_by_frame,
        raw_by_frame=raw_by_frame,
        width=width,
        height=height,
        match_threshold_px=match_threshold_px,
    )
    teleports = detect_teleports(
        predictions=[
            BallPosition(
                frame_number=f, x=p.x, y=p.y, confidence=p.confidence
            )
            for f, p in pred_by_frame.items()
        ],
        width=width,
        height=height,
        v_max_px_per_frame=teleport_v_max_px_per_frame,
    )
    wrong = detect_wrong_object(
        gt_ball=gt_ball,
        pred_by_frame=pred_by_frame,
        interpolated_frames=interpolated_frames,
        width=width,
        height=height,
        player_positions_by_frame=player_positions_by_frame,
        threshold_px=wrong_object_threshold_px,
    )
    filtered_frames = {p.frame_number - frame_offset for p in predictions}
    stationary = detect_stationary_clusters(
        raw_positions=raw_positions,
        width=width,
        height=height,
        filtered_frames=filtered_frames,
    )
    # Shift stationary clusters to GT space for consistent reporting.
    stationary = [
        StationaryClusterEvent(
            start_frame=e.start_frame - frame_offset,
            end_frame=e.end_frame - frame_offset,
            length=e.length,
            median_xy=e.median_xy,
            spread_norm=e.spread_norm,
            mean_confidence=e.mean_confidence,
            passed_filter=e.passed_filter,
        )
        for e in stationary
    ]
    revisits = detect_revisit_clusters(
        raw_positions=raw_positions,
        width=width,
        height=height,
        filtered_frames=filtered_frames,
    )
    # Shift revisit cluster frames to GT space for consistent reporting.
    revisits = [
        RevisitClusterEvent(
            center_xy=e.center_xy,
            visit_frames=[f - frame_offset for f in e.visit_frames],
            visit_count=e.visit_count,
            mean_confidence=e.mean_confidence,
            passed_filter=e.passed_filter,
        )
        for e in revisits
    ]
    two_ball = detect_two_ball_divergence(
        raw_multi_by_frame=raw_multi_by_frame or {},
        width=width,
        height=height,
    )

    # Roll GT frames into buckets (exactly one bucket per GT frame).
    per_frame_status: dict[int, str] = {}
    wrong_frames = {e.frame for e in wrong}
    matched = 0
    missed_no_raw = 0
    missed_filter_killed = 0
    wrong_object = 0
    interpolated_correct = 0
    interpolated_wrong = 0
    total_gt_frames = 0
    for gt_pos in gt_ball:
        total_gt_frames += 1
        frame = gt_pos.frame_number
        pred = pred_by_frame.get(frame)
        was_interpolated = frame in interpolated_frames

        if pred is None:
            if raw_by_frame.get(frame) is not None:
                raw = raw_by_frame[frame]
                err = _px_distance(
                    (raw.x, raw.y), (gt_pos.x, gt_pos.y), width, height
                )
                if err <= match_threshold_px and raw.confidence > 0:
                    missed_filter_killed += 1
                    per_frame_status[frame] = "missed_filter_killed"
                    continue
            missed_no_raw += 1
            per_frame_status[frame] = "missed_no_raw"
            continue

        err = _px_distance(
            (pred.x, pred.y), (gt_pos.x, gt_pos.y), width, height
        )
        if err <= match_threshold_px:
            if was_interpolated:
                interpolated_correct += 1
                per_frame_status[frame] = "interpolated_correct"
            else:
                matched += 1
                per_frame_status[frame] = "matched"
        elif frame in wrong_frames:
            if was_interpolated:
                interpolated_wrong += 1
                per_frame_status[frame] = "interpolated_wrong"
            else:
                wrong_object += 1
                per_frame_status[frame] = "wrong_object"
        else:
            # Beyond match threshold but below wrong-object threshold —
            # treat as wrong_object for the budget (still a gap).
            if was_interpolated:
                interpolated_wrong += 1
                per_frame_status[frame] = "interpolated_wrong"
            else:
                wrong_object += 1
                per_frame_status[frame] = "wrong_object"

    smoothness_p90, smoothness_median = compute_smoothness(
        pred_by_frame=pred_by_frame, width=width, height=height,
    )

    budget = FailureBudget(
        rally_id=rally_id,
        total_gt_frames=total_gt_frames,
        matched=matched,
        missed_no_raw=missed_no_raw,
        missed_filter_killed=missed_filter_killed,
        wrong_object=wrong_object,
        interpolated_correct=interpolated_correct,
        interpolated_wrong=interpolated_wrong,
        teleport_count=len(teleports),
        stationary_cluster_count=len(stationary),
        stationary_cluster_frames=sum(e.length for e in stationary),
        revisit_cluster_count=len(revisits),
        revisit_cluster_visits=sum(e.visit_count for e in revisits),
        two_ball_count=len(two_ball),
        frame_offset=frame_offset,
        match_threshold_px=match_threshold_px,
        wrong_object_threshold_px=wrong_object_threshold_px,
        teleport_v_max_px_per_frame=teleport_v_max_px_per_frame,
        smoothness_px_p90=smoothness_p90,
        smoothness_px_median=smoothness_median,
        gt_mode=gt_mode,
    )

    return FailureEvents(
        rally_id=rally_id,
        budget=budget,
        missed_streaks=missed,
        teleports=teleports,
        wrong_objects=wrong,
        stationary_clusters=stationary,
        revisit_clusters=revisits,
        two_ball_events=two_ball,
        filter_kills=filter_kill_events,
        per_frame_status=per_frame_status,
    )
