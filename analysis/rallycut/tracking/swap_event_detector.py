"""Within-rally swap event detector (Workstream 4, 2026-04-29).

Detects within-rally identity swaps between BoT-SORT tracks via the
bbox-overlap-then-mirror-cost mechanism. Departs from the closed per-frame
within-rally rescue (NO-GO 2026-04-28) which used HSV bimodality SEARCH —
W4 uses bbox-overlap candidates + per-event verification.

Mechanism:
    1. For each pair (a, b) of primary tracks, compute mutual bbox IoU at
       every frame both are present.
    2. Find frames with IoU ≥ τ_iou. Cluster nearby peaks (within
       MIN_EVENT_GAP_FRAMES) into one event per cluster.
    3. For each candidate frame s, partition each track's per-frame features
       into pre = [0, s) and post = [s, end). Build pre/post
       TrackAppearanceStats for both tracks.
    4. Compute four cross-rally costs:
            same_a   = c(a_pre, a_post)
            same_b   = c(b_pre, b_post)
            cross_ab = c(a_pre, b_post)
            cross_ba = c(b_pre, a_post)
    5. SWAP detected if cross_ab < same_a − margin AND cross_ba < same_b − margin.
       Two-track mirror-swap corroboration filters single-track false positives.

Probe-stage validation 2026-04-29: detected 4/4 user-confirmed real swaps
(jojo r07 t2↔t37@311, jojo r10 t11↔t38@457, jojo r08 t4↔t25@186, 2d105b7b
r06 t4↔t6@511) with cross-cost ≈ HALF of same-cost on every detection.
Two of those swaps (jojo r10, 2d105b7b r06) were undocumented in production
data — silently within-team mis-attributions.

Activation: env ENABLE_BBOX_SWAP_DETECTION=1 (default off).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rallycut.tracking._subtrack import SubTrackCandidate
from rallycut.tracking.player_features import (
    PlayerAppearanceFeatures,
    TrackAppearanceStats,
    compute_track_similarity,
)

if TYPE_CHECKING:
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


# Detection thresholds — fixed from the 4-rally probe validation 2026-04-29.
IOU_THRESHOLD = 0.30  # bbox-overlap fraction at a candidate event frame
IOU_THRESHOLD_LOW = 0.15  # second-pass for narrow occlusions (jojo r08-style)
MIN_EVENT_GAP_FRAMES = 30  # cluster nearby IoU peaks
MIN_HALF_FRAMES = 4  # need this many pose-available features per half
# cross-cost must beat same-cost by this margin. Matches the cost-surface
# noise floor used elsewhere in the matcher. Don't fit this to FPs — fix
# the underlying cause (e.g., non-player tracks → MIN_TRACK_COVERAGE_FRAMES).
SWAP_MARGIN = 0.03

# Minimum number of positions a track must have in a rally to be considered
# for swap candidacy. Filters non-player tracks (camera artifacts, brief
# spectator occlusions) at the root cause rather than via fitted SWAP_MARGIN.
# Calibrated 2026-04-29: 90266c1d r01's spurious track (caused FP at
# SWAP_MARGIN=0.03) had ~22 positions; real player tracks have hundreds.
MIN_TRACK_COVERAGE_FRAMES = 30


def is_enabled() -> bool:
    """Default: ON (shipped 2026-04-29). Production behavior unchanged on
    rallies without detected swap events; safe to flip OFF via
    `ENABLE_BBOX_SWAP_DETECTION=0` for rollback. 28-rally validation panel:
    5/5 real-event detections, 0 fabricated false positives. Sub-tracks
    flow through existing per-frame writer (`_build_per_frame_pid_map`).
    """
    return os.environ.get("ENABLE_BBOX_SWAP_DETECTION", "1") == "1"


def is_position_jump_enabled() -> bool:
    """Default: ON (shipped 2026-05-01). Position-jump swap detector covers
    W4's blind spot — track-ID swaps that occur without sufficient bbox
    overlap to trip the IoU>=0.30 criterion (e.g., players passing each
    other quickly with momentary occlusion). Detected as paired correlated
    single-frame jumps in opposite directions. Validated on 1/9 panel
    error (5c756c41/r07: t1+t4 paired jump at f310->311) with 0/4 control
    fires. Falsifiable, no fitted thresholds. Safe to flip OFF via
    `ENABLE_POSITION_JUMP_SWAP=0` for rollback.
    """
    return os.environ.get("ENABLE_POSITION_JUMP_SWAP", "1") == "1"


# Position-jump detection thresholds (locked, physical-plausibility-derived).
# A volleyball player at 30fps cannot translate >10% of the frame width in a
# single frame: 10% of frame width ≈ 1.6m on a 16m court @ 30fps = 48 m/s.
# World-record sprinters peak at ~12 m/s, so 48 m/s is unphysical.
POSITION_JUMP_THRESHOLD = 0.10
# Max distance between a's post-jump position and b's pre-jump position (and
# vice versa) for the jump to be classified as a SWAP rather than independent
# YOLO detection noise. 0.05 normalized ≈ 80cm — within "they crossed at the
# same physical spot" tolerance.
POSITION_JUMP_PAIR_DISTANCE = 0.05
# BoxMOT may register the swap on the two tracks at slightly different frames
# (1-2 frame lag depending on which track's detection updates first).
POSITION_JUMP_PAIR_FRAME_TOLERANCE = 2
# Skip jumps separated by >5 frames of absence: long gaps allow legitimate
# motion across the gap (player ran while off-screen), not a true 1-frame jump.
POSITION_JUMP_MAX_GAP_FRAMES = 5


@dataclass
class SwapEvent:
    """A detected within-rally swap of two BoT-SORT tracks at a frame."""

    track_a: int
    track_b: int
    split_frame: int  # rally-relative frame; pre = [0, split), post = [split, end]
    peak_iou: float
    same_a: float
    same_b: float
    cross_ab: float
    cross_ba: float


def _bbox_iou_xyxy_norm(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_xyxy(p: PlayerPosition) -> tuple[float, float, float, float]:
    return (
        p.x - p.width / 2, p.y - p.height / 2,
        p.x + p.width / 2, p.y + p.height / 2,
    )


def _find_iou_events(
    a_positions: list[PlayerPosition],
    b_positions: list[PlayerPosition],
    iou_threshold: float,
    min_event_gap: int,
) -> list[tuple[int, float]]:
    """List of (representative_frame, peak_iou) for clustered high-IoU events."""
    a_by = {p.frame_number: p for p in a_positions}
    b_by = {p.frame_number: p for p in b_positions}
    common = sorted(set(a_by) & set(b_by))
    high: list[tuple[int, float]] = []
    for fn in common:
        iou = _bbox_iou_xyxy_norm(_bbox_xyxy(a_by[fn]), _bbox_xyxy(b_by[fn]))
        if iou >= iou_threshold:
            high.append((fn, iou))
    if not high:
        return []
    high.sort(key=lambda kv: kv[0])
    events: list[tuple[int, float]] = []
    cluster_anchor = high[0][0]
    best = high[0]
    for fn, iou in high[1:]:
        if fn - cluster_anchor <= min_event_gap:
            if iou > best[1]:
                best = (fn, iou)
            cluster_anchor = fn
        else:
            events.append(best)
            cluster_anchor = fn
            best = (fn, iou)
    events.append(best)
    return events


def _stats_from_subset(features: list[PlayerAppearanceFeatures], track_id: int) -> TrackAppearanceStats | None:
    valid = [f for f in features if f is not None]
    if len(valid) < MIN_HALF_FRAMES:
        return None
    stats = TrackAppearanceStats(track_id=track_id)
    stats.features = valid
    stats.compute_averages()
    return stats


def _verify_swap_at_frame(
    a_features: list[PlayerAppearanceFeatures],
    a_frame_numbers: list[int],
    b_features: list[PlayerAppearanceFeatures],
    b_frame_numbers: list[int],
    split_frame: int,
    margin: float,
) -> tuple[bool, dict] | None:
    a_pre = [a_features[i] for i, fn in enumerate(a_frame_numbers) if fn < split_frame]
    a_post = [a_features[i] for i, fn in enumerate(a_frame_numbers) if fn >= split_frame]
    b_pre = [b_features[i] for i, fn in enumerate(b_frame_numbers) if fn < split_frame]
    b_post = [b_features[i] for i, fn in enumerate(b_frame_numbers) if fn >= split_frame]

    sa_pre = _stats_from_subset(a_pre, 0)
    sa_post = _stats_from_subset(a_post, 0)
    sb_pre = _stats_from_subset(b_pre, 0)
    sb_post = _stats_from_subset(b_post, 0)
    if sa_pre is None or sa_post is None or sb_pre is None or sb_post is None:
        return None

    same_a = float(compute_track_similarity(sa_pre, sa_post))
    same_b = float(compute_track_similarity(sb_pre, sb_post))
    cross_ab = float(compute_track_similarity(sa_pre, sb_post))
    cross_ba = float(compute_track_similarity(sb_pre, sa_post))
    swap = (cross_ab < same_a - margin) and (cross_ba < same_b - margin)
    return swap, {
        "same_a": same_a, "same_b": same_b,
        "cross_ab": cross_ab, "cross_ba": cross_ba,
        "n_a_pre": len(a_pre), "n_a_post": len(a_post),
        "n_b_pre": len(b_pre), "n_b_post": len(b_post),
    }


def detect_swap_events(
    primary_track_ids: list[int],
    positions: list[PlayerPosition],
    track_stats: dict[int, TrackAppearanceStats],
    iou_threshold: float = IOU_THRESHOLD,
    iou_threshold_low: float = IOU_THRESHOLD_LOW,
    swap_margin: float = SWAP_MARGIN,
) -> list[SwapEvent]:
    """Detect mirror-swap events between primary track pairs in one rally.

    Two-pass IoU: high-threshold first; if no swaps detected, retry at low
    threshold (catches narrow-occlusion swaps like jojo r08).

    Args:
        primary_track_ids: rally's 4 primary tracks (post-remap track ids).
        positions: rally's PlayerPositions (must include all primary tracks' frames).
        track_stats: per-track aggregated stats; uses .features (per-frame list).

    Returns:
        List of SwapEvent. Empty if no swaps detected.
    """
    if len(primary_track_ids) < 2:
        return []

    by_track: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        if p.track_id in primary_track_ids:
            by_track.setdefault(p.track_id, []).append(p)
    for tid in by_track:
        by_track[tid].sort(key=lambda p: p.frame_number)

    # Index per-track per-frame features by frame_number
    feature_by_frame: dict[int, dict[int, PlayerAppearanceFeatures]] = {}
    for tid, stats in track_stats.items():
        feature_by_frame[tid] = {
            f.frame_number: f for f in stats.features if f is not None
        }

    def _per_pair(iou_thresh: float) -> list[SwapEvent]:
        events: list[SwapEvent] = []
        for i, a in enumerate(primary_track_ids):
            for b in primary_track_ids[i + 1:]:
                a_pos = by_track.get(a, [])
                b_pos = by_track.get(b, [])
                if not a_pos or not b_pos:
                    continue
                # Coverage-floor filter: skip pairs where either track has
                # too few positions to plausibly be a real player.
                if (len(a_pos) < MIN_TRACK_COVERAGE_FRAMES
                        or len(b_pos) < MIN_TRACK_COVERAGE_FRAMES):
                    continue
                iou_events = _find_iou_events(
                    a_pos, b_pos, iou_thresh, MIN_EVENT_GAP_FRAMES,
                )
                if not iou_events:
                    continue
                a_feat_dict = feature_by_frame.get(a, {})
                b_feat_dict = feature_by_frame.get(b, {})
                a_fnums = sorted(a_feat_dict)
                b_fnums = sorted(b_feat_dict)
                a_feats = [a_feat_dict[fn] for fn in a_fnums]
                b_feats = [b_feat_dict[fn] for fn in b_fnums]
                for ev_frame, ev_iou in iou_events:
                    res = _verify_swap_at_frame(
                        a_feats, a_fnums, b_feats, b_fnums,
                        ev_frame, swap_margin,
                    )
                    if res is None:
                        continue
                    is_swap, costs = res
                    if is_swap:
                        events.append(SwapEvent(
                            track_a=a, track_b=b, split_frame=ev_frame,
                            peak_iou=ev_iou,
                            same_a=costs["same_a"], same_b=costs["same_b"],
                            cross_ab=costs["cross_ab"], cross_ba=costs["cross_ba"],
                        ))
        return events

    events = _per_pair(iou_threshold)
    if not events:
        events = _per_pair(iou_threshold_low)
        if events:
            logger.info(
                "swap_event_detector: %d events found at low IoU threshold %.2f",
                len(events), iou_threshold_low,
            )
    return events


def _track_jumps(
    track_positions: list[PlayerPosition],
    threshold: float,
    max_gap_frames: int,
) -> list[tuple[int, float, float, float, float]]:
    """List of (jump_frame, prev_x, prev_y, new_x, new_y) for single-frame
    position jumps in this track. Skips gaps > max_gap_frames where
    legitimate motion across the gap would be confused with a jump.
    """
    out: list[tuple[int, float, float, float, float]] = []
    for i in range(1, len(track_positions)):
        prev = track_positions[i - 1]
        curr = track_positions[i]
        df = curr.frame_number - prev.frame_number
        if df <= 0 or df > max_gap_frames:
            continue
        dx = abs(curr.x - prev.x) / df
        dy = abs(curr.y - prev.y) / df
        if dx > threshold or dy > threshold:
            out.append((
                int(curr.frame_number),
                float(prev.x), float(prev.y),
                float(curr.x), float(curr.y),
            ))
    return out


def detect_position_jump_swap_events(
    primary_track_ids: list[int],
    positions: list[PlayerPosition],
    jump_threshold: float = POSITION_JUMP_THRESHOLD,
    pair_distance: float = POSITION_JUMP_PAIR_DISTANCE,
    pair_frame_tolerance: int = POSITION_JUMP_PAIR_FRAME_TOLERANCE,
    max_gap_frames: int = POSITION_JUMP_MAX_GAP_FRAMES,
) -> list[SwapEvent]:
    """Detect track-ID swap events via paired correlated single-frame jumps.

    Covers W4's blind spot. W4 (`detect_swap_events`) requires bbox IoU
    >= 0.30 at the swap frame; this misses swaps where two players pass
    each other quickly without bbox overlap (e.g., 5c756c41/r07 t1+t4 at
    f310->311). The position-jump signature catches these via:

      1. Per-track: detect any single-frame |Δposition| > jump_threshold.
         Jumps separated by long gaps (> max_gap_frames) are excluded —
         they reflect legitimate motion across an off-screen interval.
      2. Per-pair: a pair (a, b) is a swap iff BOTH tracks have a jump at
         the same frame (within pair_frame_tolerance) AND the post-jump
         position of one matches the pre-jump position of the other,
         within pair_distance, in BOTH directions. This rules out
         independent YOLO instability or coincidental jumps.

    Returns SwapEvents with cost fields set to -1.0 (sentinel) — position-
    jump detector doesn't compute mirror-cost. `build_subtracks_from_events`
    only uses (track_a, track_b, split_frame); cost fields are diagnostic.

    One event per pair maximum: even if multiple correlated-jump candidates
    exist, only the earliest is emitted (rare in practice).

    Args:
        primary_track_ids: rally's primary tracks.
        positions: rally's PlayerPositions.
        jump_threshold: per-frame normalized position jump threshold.
        pair_distance: max swap-correlation distance.
        pair_frame_tolerance: ± frame tolerance for paired jumps.
        max_gap_frames: skip jumps over gaps longer than this.

    Returns:
        List of SwapEvent. Empty if no paired jumps detected.
    """
    if len(primary_track_ids) < 2:
        return []

    by_track: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        if p.track_id in primary_track_ids:
            by_track.setdefault(p.track_id, []).append(p)
    for tid in by_track:
        by_track[tid].sort(key=lambda p: p.frame_number)

    # Detect per-track jumps; skip undersized tracks (likely noise, not players).
    jumps_per_track: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for tid, ps in by_track.items():
        if len(ps) < MIN_TRACK_COVERAGE_FRAMES:
            continue
        track_jumps = _track_jumps(ps, jump_threshold, max_gap_frames)
        if track_jumps:
            jumps_per_track[tid] = track_jumps

    if len(jumps_per_track) < 2:
        return []

    events: list[SwapEvent] = []
    paired: set[tuple[int, int]] = set()

    for i, a in enumerate(primary_track_ids):
        if a not in jumps_per_track:
            continue
        for b in primary_track_ids[i + 1:]:
            if b not in jumps_per_track:
                continue
            pair_key = (min(a, b), max(a, b))
            if pair_key in paired:
                continue
            best_event: SwapEvent | None = None
            for ja in jumps_per_track[a]:
                a_frame, a_prev_x, a_prev_y, a_new_x, a_new_y = ja
                for jb in jumps_per_track[b]:
                    b_frame, b_prev_x, b_prev_y, b_new_x, b_new_y = jb
                    if abs(a_frame - b_frame) > pair_frame_tolerance:
                        continue
                    # Verify bidirectional swap: a's post matches b's pre,
                    # b's post matches a's pre (within pair_distance).
                    d_a_to_b_pre = (
                        (a_new_x - b_prev_x) ** 2
                        + (a_new_y - b_prev_y) ** 2
                    ) ** 0.5
                    d_b_to_a_pre = (
                        (b_new_x - a_prev_x) ** 2
                        + (b_new_y - a_prev_y) ** 2
                    ) ** 0.5
                    if d_a_to_b_pre > pair_distance:
                        continue
                    if d_b_to_a_pre > pair_distance:
                        continue
                    split = min(a_frame, b_frame)
                    candidate = SwapEvent(
                        track_a=a, track_b=b, split_frame=split,
                        peak_iou=-1.0,
                        same_a=-1.0, same_b=-1.0,
                        cross_ab=-1.0, cross_ba=-1.0,
                    )
                    if best_event is None or candidate.split_frame < best_event.split_frame:
                        best_event = candidate
            if best_event is not None:
                events.append(best_event)
                paired.add(pair_key)
                logger.info(
                    "position_jump_swap: pair (t%d, t%d) split @ f%d",
                    best_event.track_a, best_event.track_b,
                    best_event.split_frame,
                )

    return events


def build_subtracks_from_events(
    events: list[SwapEvent],
    primary_track_ids: list[int],
    positions: list[PlayerPosition],
    track_stats: dict[int, TrackAppearanceStats],
) -> list[SubTrackCandidate]:
    """For each detected swap, emit FOUR SubTrackCandidates (pre/post for each track).

    Each candidate's appearance_stats is built from the parent's per-frame
    features restricted to the half [f_start, f_end]. aggregated_argmax_pid
    is left None — Hungarian assigns pids from clean per-half features.

    If multiple events split the same parent track (rare; would mean 2+ swaps
    in one rally), only the FIRST event's split is honored for that parent.
    """
    parent_frame_range: dict[int, tuple[int, int]] = {}
    for tid in primary_track_ids:
        track_pos = [p for p in positions if p.track_id == tid]
        if not track_pos:
            continue
        f_start = min(p.frame_number for p in track_pos)
        f_end = max(p.frame_number for p in track_pos)
        parent_frame_range[tid] = (f_start, f_end)

    used_parents: set[int] = set()
    sub_tracks: list[SubTrackCandidate] = []

    for ev in events:
        for parent_tid in (ev.track_a, ev.track_b):
            if parent_tid in used_parents:
                continue
            if parent_tid not in parent_frame_range:
                continue
            f_start, f_end = parent_frame_range[parent_tid]
            split = ev.split_frame
            if split <= f_start or split > f_end:
                continue

            stats_parent = track_stats.get(parent_tid)
            if stats_parent is None:
                continue
            pre_features = [
                f for f in stats_parent.features
                if f is not None and f.frame_number < split
            ]
            post_features = [
                f for f in stats_parent.features
                if f is not None and f.frame_number >= split
            ]
            if len(pre_features) < MIN_HALF_FRAMES or len(post_features) < MIN_HALF_FRAMES:
                continue

            pre_stats = TrackAppearanceStats(track_id=parent_tid)
            pre_stats.features = pre_features
            pre_stats.compute_averages()
            post_stats = TrackAppearanceStats(track_id=parent_tid)
            post_stats.features = post_features
            post_stats.compute_averages()

            sub_tracks.append(SubTrackCandidate(
                parent_track_id=parent_tid,
                segment_index=0,
                f_start=f_start,
                f_end=split - 1,
                appearance_stats=pre_stats,
            ))
            sub_tracks.append(SubTrackCandidate(
                parent_track_id=parent_tid,
                segment_index=1,
                f_start=split,
                f_end=f_end,
                appearance_stats=post_stats,
            ))
            used_parents.add(parent_tid)

    return sub_tracks
