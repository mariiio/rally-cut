"""Appearance-based within-rally ID-switch detector (Phase 1, split-only).

Targets the same failure pattern as `_slow_drift_split` but uses appearance
consistency over time as the trigger signal instead of position drift.
Robust on cases where position shift is borderline (sub-threshold for the
drift detector) but appearance changes dramatically — which happens when
BoT-SORT loses a player during occlusion and continues the track on a
different physical player whose position happens to be similar.

Algorithm:
  1. Split each post-MatchSolver primary track into N=3 contiguous time
     windows.
  2. Compute per-window appearance features via the same aggregation
     the matcher uses (`extract_rally_appearances`).
  3. For each track, compute pairwise appearance distance between its
     windows. The track is a SPLIT CANDIDATE when the max inter-window
     cost exceeds a RELATIVE gate: `k * median(inter-track-cost)`.
     Relative (not absolute) so the gate adapts to videos with similar
     vs distinct uniforms.
  4. For each candidate, find the changepoint window boundary that
     maximizes the inter-group cost (W0|W1+W2 or W0+W1|W2). Bisect at
     the boundary frame.
  5. Re-Hungarian: each half's appearance features are scored against
     the OTHER tracks' whole-track features; each half is assigned the
     lowest-cost PID consistent with the constraint that distinct
     halves of one parent can't both keep the parent's original PID
     (the parent is being split because its appearance is inconsistent).
  6. Emit `SubTrackCandidate` for the half that gets re-assigned;
     parent keeps the unchanged half via the existing per-frame override
     resolver.

Default OFF behind `ENABLE_WITHIN_RALLY_REPAIR=1`. Phase 1 is
split-only — it doesn't merge sub-tracks back across track boundaries.
That's Phase 2 work; without it, a duplicate-track BoT-SORT failure
(cyan-shirt re-acquired as a new track ID after occlusion) won't fully
collapse to one PID — but the matcher's per-track decisions on the
clean halves should still be more accurate than on the contaminated
whole-track features.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

from rallycut.tracking._subtrack import SubTrackCandidate
from rallycut.tracking.match_tracker import extract_rally_appearances
from rallycut.tracking.player_features import (
    TrackAppearanceStats,
    compute_track_similarity,
)

if TYPE_CHECKING:
    from pathlib import Path

    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

ENV_FLAG = "ENABLE_WITHIN_RALLY_REPAIR"

# Number of contiguous windows per track. Three is the minimum that
# permits both "switch in W2" and "switch in W0" detection without
# blowing up the per-track feature-extraction cost.
NUM_WINDOWS = 3

# Minimum samples per window for the gate to fire. Below this the
# appearance features are too noisy to trust.
MIN_SAMPLES_PER_WINDOW = 8

# Relative gate: a track is a split candidate when its max inter-window
# cost exceeds `RELATIVE_GATE_K * median(inter-track-cost)`. Picked at
# 0.95 from the 09553ef1 probe (`probe_within_rally_appearance_split.py`):
# the suspect track scored 0.491 / 0.420 = 1.17 (above gate); borderline
# tracks scored 0.50–0.54 (below gate). Conservative — favors no-action
# on ambiguous cases.
RELATIVE_GATE_K = 0.95

# ReID blend used for cost computation. Mirrors MatchSolver's default.
REID_BLEND = 0.5


def is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


@dataclass
class _WindowStats:
    track_id: int
    window_idx: int
    f_start: int
    f_end: int
    n_frames: int
    stats: TrackAppearanceStats


def _build_windows(
    positions: list[PlayerPosition],
    track_id: int,
    n_windows: int = NUM_WINDOWS,
) -> list[list[PlayerPosition]] | None:
    """Split a track's positions into N contiguous time windows.

    Returns None when the track has too few samples to support N
    windows of MIN_SAMPLES_PER_WINDOW each.
    """
    pts = sorted(
        (p for p in positions if int(p.track_id) == track_id),
        key=lambda q: q.frame_number,
    )
    if len(pts) < n_windows * MIN_SAMPLES_PER_WINDOW:
        return None
    window_size = len(pts) // n_windows
    windows: list[list[PlayerPosition]] = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(pts)
        windows.append(pts[start:end])
    return windows


def _extract_window_stats(
    track_id: int,
    windows: list[list[PlayerPosition]],
    *,
    video_path: Path,
    rally_start_ms: int,
    rally_end_ms: int,
    reid_model: Any,
) -> list[_WindowStats]:
    """Extract per-window appearance stats via the canonical aggregator.

    Calls `extract_rally_appearances` once per window, restricted to
    this track's frames in that window. Reuses the matcher's medoid
    aggregation so the resulting cost numbers are directly comparable
    to the cross-rally Hungarian's costs.
    """
    out: list[_WindowStats] = []
    for w_idx, w_positions in enumerate(windows):
        if len(w_positions) < MIN_SAMPLES_PER_WINDOW:
            continue
        ts = extract_rally_appearances(
            video_path=video_path,
            positions=w_positions,
            primary_track_ids=[track_id],
            start_ms=rally_start_ms,
            end_ms=rally_end_ms,
            num_samples=min(12, len(w_positions)),
            extract_reid=reid_model is not None,
            reid_model=reid_model,
        )
        if track_id not in ts:
            continue
        out.append(_WindowStats(
            track_id=track_id,
            window_idx=w_idx,
            f_start=int(w_positions[0].frame_number),
            f_end=int(w_positions[-1].frame_number),
            n_frames=len(w_positions),
            stats=ts[track_id],
        ))
    return out


def _pairwise_cost(stats_a: TrackAppearanceStats, stats_b: TrackAppearanceStats) -> float:
    return float(compute_track_similarity(stats_a, stats_b, reid_blend=REID_BLEND))


def _detect_split_candidates(
    by_track_windows: dict[int, list[_WindowStats]],
) -> dict[int, tuple[float, int]]:
    """Return {track_id: (intra_max_cost, changepoint_window_idx)} for
    tracks whose intra-window cost exceeds the relative gate.

    The gate is `RELATIVE_GATE_K * median(inter-track cost)` — relative
    so it adapts to per-video discriminability. The changepoint window
    is the boundary index (1 means "split between W0 and W1+W2"; 2
    means "split between W0+W1 and W2") that MAXIMIZES inter-group
    distance — picks the most informative split.
    """
    # Inter-track baseline: median cost across all distinct (track,track)
    # window pairs (one window per track to keep weights uniform).
    track_ids = sorted(by_track_windows.keys())
    if len(track_ids) < 2:
        return {}
    inter_costs: list[float] = []
    for tid_a, tid_b in combinations(track_ids, 2):
        for ws_a in by_track_windows[tid_a]:
            for ws_b in by_track_windows[tid_b]:
                inter_costs.append(_pairwise_cost(ws_a.stats, ws_b.stats))
    if not inter_costs:
        return {}
    inter_median = float(np.median(inter_costs))
    gate = RELATIVE_GATE_K * inter_median

    candidates: dict[int, tuple[float, int]] = {}
    for tid, windows in by_track_windows.items():
        if len(windows) < NUM_WINDOWS:
            continue
        # Intra-window pairwise costs.
        intra: dict[tuple[int, int], float] = {}
        for ws_a, ws_b in combinations(windows, 2):
            c = _pairwise_cost(ws_a.stats, ws_b.stats)
            intra[(ws_a.window_idx, ws_b.window_idx)] = c
        intra_max = max(intra.values()) if intra else 0.0
        if intra_max < gate:
            logger.debug(
                "within_rally_repair: T%d intra_max=%.3f < gate=%.3f "
                "(median inter=%.3f) — skip",
                tid, intra_max, gate, inter_median,
            )
            continue
        # Find changepoint that maximizes inter-group distance.
        # For NUM_WINDOWS=3 there are 2 binary splits: {W0}|{W1,W2}
        # and {W0,W1}|{W2}. Pick the one with the larger inter-group
        # cost (mean of all cross-group window-pair costs).
        best_split = -1
        best_inter_group = -1.0
        for boundary in range(1, NUM_WINDOWS):
            left = [w for w in windows if w.window_idx < boundary]
            right = [w for w in windows if w.window_idx >= boundary]
            if not left or not right:
                continue
            cross = [
                _pairwise_cost(l.stats, r.stats) for l in left for r in right
            ]
            if not cross:
                continue
            mean_cross = float(np.mean(cross))
            if mean_cross > best_inter_group:
                best_inter_group = mean_cross
                best_split = boundary
        if best_split < 0:
            continue
        logger.info(
            "within_rally_repair: T%d split candidate — intra_max=%.3f "
            "(gate=%.3f, median inter=%.3f), changepoint at window %d "
            "(inter-group=%.3f)",
            tid, intra_max, gate, inter_median, best_split, best_inter_group,
        )
        candidates[tid] = (intra_max, best_split)
    return candidates


def _reassign_split_halves(
    parent_tid: int,
    parent_pid: int,
    first_half_stats: TrackAppearanceStats,
    second_half_stats: TrackAppearanceStats,
    other_track_pids: dict[int, int],
    other_track_stats: dict[int, TrackAppearanceStats],
) -> tuple[int, int] | None:
    """Decide which PID each half of the parent track should take.

    Conservative strategy: ONE half re-assigns to its best-matching
    other-track's PID; the other half KEEPS parent_pid.

    Choice of which half re-assigns: the half with the LOWER cost to
    its best other-track is the more confident match → it re-assigns.
    The other half is "more distinctive" relative to other tracks and
    therefore better trusted to carry the parent's identity.

    Why this conservative shape:
      - The "always re-assign both halves to their best others" variant
        regressed PERMUTED by 11 frames on 7d77980f (cross-fixture
        validation 2026-05-03). When the best other-track for a half
        is borderline, re-assigning blindly causes more errors than
        it fixes.
      - A "tie-break to second-best" variant when both halves point
        to the same other track also regressed (8 frames on 7d77980f).
        Same root cause: the second-best is often unreliable.
      - The original logic (what we revert to here) had clean PERMUTED
        on all 4 fixtures.

    Trade-off: this won't fully resolve cases like 7d77980f / 09553ef1
    where both halves point to the same other track — the second half
    keeps the parent's (contaminated) PID, leading to a residual
    visual inconsistency. Phase 2's clip/drop handles the duplicate-
    PID-per-frame artifact that would otherwise result. The proper
    full fix for the 09553ef1 case is cross-track merge (Phase 3),
    not aggressive within-rally re-assignment.

    Returns (first_half_pid, second_half_pid) or None when no safe
    re-assignment exists.
    """
    if not other_track_stats:
        return None
    inter_half = _pairwise_cost(first_half_stats, second_half_stats)
    if inter_half < 1e-3:
        return None
    first_dists = {
        other_tid: _pairwise_cost(first_half_stats, ts)
        for other_tid, ts in other_track_stats.items()
    }
    second_dists = {
        other_tid: _pairwise_cost(second_half_stats, ts)
        for other_tid, ts in other_track_stats.items()
    }
    first_best_other = min(first_dists, key=first_dists.__getitem__)
    second_best_other = min(second_dists, key=second_dists.__getitem__)
    first_min = first_dists[first_best_other]
    second_min = second_dists[second_best_other]

    first_half_pid: int | None
    second_half_pid: int | None
    if first_min <= second_min:
        # First half is the more confident match → it re-assigns.
        first_half_pid = other_track_pids.get(first_best_other)
        second_half_pid = parent_pid
    else:
        first_half_pid = parent_pid
        second_half_pid = other_track_pids.get(second_best_other)

    if first_half_pid is None or second_half_pid is None:
        return None
    if first_half_pid == second_half_pid:
        # Re-assigned half's other-track shares parent's PID — split
        # would be a no-op (or worse). Reject.
        return None
    return first_half_pid, second_half_pid


def maybe_emit_within_rally_split(
    *,
    rally_id: str,
    video_path: Path,
    rally_start_ms: int,
    rally_end_ms: int,
    positions: list[PlayerPosition],
    track_to_player: dict[int, int],
    reid_model: Any,
) -> list[SubTrackCandidate] | None:
    """Detect within-rally ID switches by appearance and emit split
    sub-tracks. Returns a list of TWO sub-tracks per affected parent
    (first half + second half). Empty when no track triggers the gate.

    No-op when `ENABLE_WITHIN_RALLY_REPAIR=1` is unset.
    """
    if not is_enabled():
        return None
    if not track_to_player:
        return None

    # Build per-track windows.
    by_track_windows: dict[int, list[_WindowStats]] = {}
    track_pids = {int(k): int(v) for k, v in track_to_player.items() if int(k) > 0}
    for tid in track_pids:
        windows = _build_windows(positions, tid)
        if windows is None:
            continue
        ws = _extract_window_stats(
            tid, windows,
            video_path=video_path,
            rally_start_ms=rally_start_ms,
            rally_end_ms=rally_end_ms,
            reid_model=reid_model,
        )
        if len(ws) == NUM_WINDOWS:
            by_track_windows[tid] = ws

    if len(by_track_windows) < 2:
        return None

    candidates = _detect_split_candidates(by_track_windows)
    if not candidates:
        return None

    # Build whole-track stats for the OTHER tracks (used as PID profiles
    # for the re-assignment Hungarian).
    other_track_whole_stats: dict[int, TrackAppearanceStats] = {}
    by_track_positions: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if int(p.track_id) > 0:
            by_track_positions[int(p.track_id)].append(p)
    for tid in track_pids:
        if tid in candidates:
            continue
        pts = by_track_positions.get(tid, [])
        if len(pts) < MIN_SAMPLES_PER_WINDOW:
            continue
        ts = extract_rally_appearances(
            video_path=video_path,
            positions=pts,
            primary_track_ids=[tid],
            start_ms=rally_start_ms,
            end_ms=rally_end_ms,
            num_samples=12,
            extract_reid=reid_model is not None,
            reid_model=reid_model,
        )
        if tid in ts:
            other_track_whole_stats[tid] = ts[tid]

    emitted: list[SubTrackCandidate] = []
    for tid, (_intra_max, boundary) in candidates.items():
        ws_list = by_track_windows[tid]
        first_half_windows = [w for w in ws_list if w.window_idx < boundary]
        second_half_windows = [w for w in ws_list if w.window_idx >= boundary]
        if not first_half_windows or not second_half_windows:
            continue
        # Aggregate halves: re-extract with the union of frames in each
        # half so the half-stats reflect the canonical aggregation over
        # the full half (not just one window).
        first_pos = [
            p for p in by_track_positions[tid]
            if first_half_windows[0].f_start <= p.frame_number
            <= first_half_windows[-1].f_end
        ]
        second_pos = [
            p for p in by_track_positions[tid]
            if second_half_windows[0].f_start <= p.frame_number
            <= second_half_windows[-1].f_end
        ]
        first_ts = extract_rally_appearances(
            video_path=video_path, positions=first_pos,
            primary_track_ids=[tid],
            start_ms=rally_start_ms, end_ms=rally_end_ms,
            num_samples=12,
            extract_reid=reid_model is not None, reid_model=reid_model,
        )
        second_ts = extract_rally_appearances(
            video_path=video_path, positions=second_pos,
            primary_track_ids=[tid],
            start_ms=rally_start_ms, end_ms=rally_end_ms,
            num_samples=12,
            extract_reid=reid_model is not None, reid_model=reid_model,
        )
        if tid not in first_ts or tid not in second_ts:
            continue

        reassign = _reassign_split_halves(
            parent_tid=tid,
            parent_pid=track_pids[tid],
            first_half_stats=first_ts[tid],
            second_half_stats=second_ts[tid],
            other_track_pids={
                ot: track_pids[ot] for ot in other_track_whole_stats
            },
            other_track_stats=other_track_whole_stats,
        )
        if reassign is None:
            logger.info(
                "within_rally_repair: T%d split rejected — no safe "
                "re-assignment found",
                tid,
            )
            continue
        first_pid, second_pid = reassign

        f_start = int(by_track_positions[tid][0].frame_number)
        boundary_frame = int(first_half_windows[-1].f_end)
        f_end = int(by_track_positions[tid][-1].frame_number)

        emitted.append(SubTrackCandidate(
            parent_track_id=tid,
            segment_index=0,
            f_start=f_start,
            f_end=boundary_frame,
            appearance_stats=first_ts[tid],
            aggregated_argmax_pid=first_pid,
        ))
        emitted.append(SubTrackCandidate(
            parent_track_id=tid,
            segment_index=1,
            f_start=boundary_frame + 1,
            f_end=f_end,
            appearance_stats=second_ts[tid],
            aggregated_argmax_pid=second_pid,
        ))
        logger.info(
            "within_rally_repair %s: T%d split at frame %d → "
            "first_half→PID%d, second_half→PID%d (was PID%d)",
            rally_id[:8] if rally_id else "?", tid, boundary_frame,
            first_pid, second_pid, track_pids[tid],
        )

    if not emitted:
        return None

    # Phase 2: deduplicate cross-track same-PID frame overlap.
    # When a sub-track's PID matches another non-parent track's PID and
    # their frame ranges overlap, two bboxes would be labeled the same
    # PID in the overlap window — visible as duplicate identity in the
    # editor. Resolve by CLIPPING the sub-track's range to yield to the
    # other track. The resolver returns UNLABELED for parent frames
    # outside any segment, so the parent's bbox in the clipped frames
    # is suppressed; the other track's bbox is the sole carrier of
    # that PID in those frames.
    #
    # Direction of clipping: yield to the other track's frame range
    # entirely. We never EXTEND the sub-track; we only shrink it.
    # Drop sub-tracks that get clipped to empty.
    emitted = _clip_overlapping_sub_tracks(
        emitted=emitted,
        track_pids=track_pids,
        by_track_positions=by_track_positions,
        rally_id=rally_id,
    )
    return emitted or None


def _track_frame_range(positions: list[PlayerPosition]) -> tuple[int, int]:
    """Min/max frame number for a track. Caller must ensure non-empty."""
    return positions[0].frame_number, positions[-1].frame_number


def _clip_overlapping_sub_tracks(
    *,
    emitted: list[SubTrackCandidate],
    track_pids: dict[int, int],
    by_track_positions: dict[int, list[PlayerPosition]],
    rally_id: str,
) -> list[SubTrackCandidate]:
    """Phase 2: clip each sub-track's range against other tracks that
    share its assigned PID with overlapping frames.

    Resolution rule: when sub-track S (parent=P, pid=K, range [a, b])
    and another track O (not P, mapped to pid=K via track_pids, range
    [c, d]) have overlapping frame ranges, the sub-track YIELDS the
    overlap to O.

    Why yield, not the inverse: the sub-track is a "rescued" segment of
    a contaminated parent track. The other track O is BoT-SORT's primary
    identity for that PID — it's the more authoritative carrier. Yielding
    means S's parent's bbox is suppressed (UNLABELED via the resolver);
    O's bbox is the sole label for those frames.

    A sub-track may be clipped at either end (or both) depending on how
    O's range overlaps. If O's range fully contains S's, S is dropped.
    If O sits strictly inside S, we'd need to split S into two segments
    flanking O — emitted as two clipped sub-tracks with the same
    parent + pid. Rare but handled.

    Sub-tracks that get clipped to empty are dropped from the result.
    Sub-tracks unaffected by any conflict pass through unchanged.
    """
    # Index other tracks (non-parent) by pid → list of (track_id, range).
    pid_to_other_tracks: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    parent_tids = {st.parent_track_id for st in emitted}
    for other_tid, other_pid in track_pids.items():
        if other_tid in parent_tids:
            continue
        pts = by_track_positions.get(other_tid, [])
        if not pts:
            continue
        f_lo, f_hi = _track_frame_range(pts)
        pid_to_other_tracks[other_pid].append((other_tid, f_lo, f_hi))

    out: list[SubTrackCandidate] = []
    for st in emitted:
        st_pid = st.aggregated_argmax_pid
        if st_pid is None:
            out.append(st)
            continue
        pid: int = st_pid
        conflicts = [
            (other_tid, c_lo, c_hi)
            for other_tid, c_lo, c_hi in pid_to_other_tracks.get(pid, [])
            if not (c_hi < st.f_start or c_lo > st.f_end)  # overlap
        ]
        if not conflicts:
            out.append(st)
            continue
        # Build the set of frame ranges OWNED by other tracks for this PID,
        # within st's range. Clip st to the complement.
        # We treat conflicts as inclusive ranges and compute the
        # complement of their union within [st.f_start, st.f_end].
        conflict_ranges = sorted(
            (max(st.f_start, c_lo), min(st.f_end, c_hi))
            for _, c_lo, c_hi in conflicts
        )
        # Merge overlapping conflicts.
        merged: list[tuple[int, int]] = []
        for lo, hi in conflict_ranges:
            if merged and lo <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        # Now produce st-fragments outside the merged conflict ranges.
        kept_fragments: list[tuple[int, int]] = []
        cursor = st.f_start
        for lo, hi in merged:
            if cursor < lo:
                kept_fragments.append((cursor, lo - 1))
            cursor = max(cursor, hi + 1)
        if cursor <= st.f_end:
            kept_fragments.append((cursor, st.f_end))

        if not kept_fragments:
            logger.info(
                "within_rally_repair %s: sub-track parent=%d seg=%d "
                "pid=%d range [%d, %d] DROPPED — fully overlapped by "
                "other tracks %s with same pid",
                rally_id[:8] if rally_id else "?",
                st.parent_track_id, st.segment_index, pid,
                st.f_start, st.f_end,
                [t for t, _, _ in conflicts],
            )
            continue

        if (
            len(kept_fragments) == 1
            and kept_fragments[0] == (st.f_start, st.f_end)
        ):
            # No actual change (shouldn't happen given conflicts != []).
            out.append(st)
            continue

        # Emit one sub-track per kept fragment.
        for frag_idx, (frag_lo, frag_hi) in enumerate(kept_fragments):
            out.append(SubTrackCandidate(
                parent_track_id=st.parent_track_id,
                # Use a derived segment_index that stays unique across
                # the rally: hash original segment + fragment offset
                # into a small range. Synthetic_track_id depends on
                # segment_index, so it must be unique per emitted
                # sub-track for the same parent.
                segment_index=st.segment_index * 10 + frag_idx,
                f_start=frag_lo,
                f_end=frag_hi,
                appearance_stats=st.appearance_stats,
                aggregated_argmax_pid=pid,
            ))
            if (frag_lo, frag_hi) != (st.f_start, st.f_end):
                logger.info(
                    "within_rally_repair %s: sub-track parent=%d seg=%d "
                    "pid=%d clipped from [%d, %d] to [%d, %d] (yielded "
                    "to overlapping tracks with same pid)",
                    rally_id[:8] if rally_id else "?",
                    st.parent_track_id, st.segment_index, pid,
                    st.f_start, st.f_end, frag_lo, frag_hi,
                )

    return out
