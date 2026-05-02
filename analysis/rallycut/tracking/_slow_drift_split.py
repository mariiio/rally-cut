"""Slow-drift bisect splitter for blind-regime within-rally identity drift.

Targets the within-rally drift pattern where two players' tracks merge
during occlusion: BoT-SORT loses one player, picks them back up, and
assigns the wrong ID, so a single "track" actually contains two
different players' frames. This appears as `slow_drift` in
`scripts/panel_verdict_per_frame.py`: a PID's centroid shifts noticeably
between rally halves AND multiple PIDs' x-ranges overlap heavily
(meaning identities have crossed mid-rally).

This module:

1. Detects slow_drift on a finalized per-rally Hungarian assignment
   (`solved` from MatchSolver), using the SAME signal the verdict tool
   uses (`HALF_SHIFT_THRESHOLD = 0.20`, `XRANGE_OVERLAP_THRESHOLD = 0.50`).
2. For the worst-shifting PID, finds its assigned track and bisects
   its frames at the median frame number.
3. Computes lower-body HSV histograms for each half (independent from
   the cross-rally MatchSolver decision).
4. **Safety gate**: halves must be HSV-distinguishable from each other
   AND at least one half must best-match a different PID's profile —
   otherwise the drift is appearance-noise, not identity. See
   `scripts/probe_slow_drift_bisect.py` for the diagnostic that
   produced these gates.
5. Emits a `SubTrackCandidate` with `parent_track_id` = the original
   tracker id, `f_start` = bisect frame, `f_end` = rally end, `pid` = the
   re-assigned PID. The PARENT is left in `top_tracks`; the sub-track
   acts as a per-frame override on the parent's range only — avoids the
   W4 UNLABELED trap (`regression_2026_05_01_7307c1d_revert.md`).

Default OFF behind `ENABLE_SLOW_DRIFT_SPLIT=1`.
"""
from __future__ import annotations

import logging
import os
import statistics
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from rallycut.tracking._subtrack import SubTrackCandidate
from rallycut.tracking.player_features import (
    TrackAppearanceStats,
    extract_appearance_features,
)

if TYPE_CHECKING:
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

ENV_FLAG = "ENABLE_SLOW_DRIFT_SPLIT"

# Trigger thresholds — modeled on `panel_verdict_per_frame.py`. Slightly
# more permissive on x-range overlap (0.45 vs verdict's 0.50) because
# this detector reads pre-remap positions while the verdict tool reads
# post-remap positions_json; small numeric differences in position
# coverage can put borderline rallies just under the verdict-tool
# threshold while still being legitimate slow_drift cases. The HSV
# safety gate (`MIN_INTER_HALF_CHI2`) is the real filter against false
# positives.
HALF_SHIFT_THRESHOLD = 0.20
XRANGE_OVERLAP_THRESHOLD = 0.45

# Safety gate — halves must be HSV-distinguishable. Below this chi2,
# the drift is appearance-noise (per `probe_slow_drift_bisect.py`:
# b5fb0594/r10 had chi2=0.10 across halves, neither half closer to
# another PID).
MIN_INTER_HALF_CHI2 = 0.20

# Maximum number of frames sampled per half for HSV computation.
# Mirrors the per-track default in `extract_rally_appearances`.
MAX_FRAMES_PER_HALF = 12


def is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


def _hist_chi2(a: np.ndarray, b: np.ndarray) -> float:
    """Chi-square distance between normalized histograms."""
    eps = 1e-9
    return float(np.sum((a - b) ** 2 / (a + b + eps)))


def _detect_slow_drift_pid(
    positions: list[PlayerPosition],
    track_to_player: dict[int, int],
) -> tuple[int, int, float] | None:
    """Mirror `_slow_drift_signal` from panel_verdict_per_frame.py.

    Returns (pid, parent_track_id, half_shift) when both gate conditions
    fire, else None. parent_track_id is the original tracker id whose
    positions drove the shift (i.e., this rally's track for `pid`).
    """
    if not positions or not track_to_player:
        return None
    # Group per-frame positions by PID via track_to_player mapping.
    by_pid: dict[int, list[PlayerPosition]] = {}
    track_id_to_pid = {int(k): int(v) for k, v in track_to_player.items() if int(k) > 0}
    for p in positions:
        pid = track_id_to_pid.get(int(p.track_id))
        if pid in (1, 2, 3, 4):
            by_pid.setdefault(pid, []).append(p)

    # Per-PID half-shift.
    max_shift = 0.0
    drift_pid = -1
    for pid, pts in by_pid.items():
        if len(pts) < 10:
            continue
        pts_sorted = sorted(pts, key=lambda q: q.frame_number)
        half = len(pts_sorted) // 2
        first, second = pts_sorted[:half], pts_sorted[half:]
        dx = (
            statistics.mean(p.x for p in second) - statistics.mean(p.x for p in first)
        )
        dy = (
            statistics.mean(p.y for p in second) - statistics.mean(p.y for p in first)
        )
        shift = (dx ** 2 + dy ** 2) ** 0.5
        if shift > max_shift:
            max_shift = shift
            drift_pid = pid

    logger.debug(
        "slow_drift_split scan: shift_max=%.3f drift_pid=%d (n_pids=%d)",
        max_shift, drift_pid, len(by_pid),
    )
    if max_shift < HALF_SHIFT_THRESHOLD or drift_pid < 0:
        return None

    # Pairwise x-range overlap.
    pids = list(by_pid.keys())
    max_overlap = 0.0
    for i, pi in enumerate(pids):
        cx_i = [p.x for p in by_pid[pi]]
        for pj in pids[i + 1:]:
            cx_j = [p.x for p in by_pid[pj]]
            if not cx_i or not cx_j:
                continue
            min_i, max_i = min(cx_i), max(cx_i)
            min_j, max_j = min(cx_j), max(cx_j)
            ov = max(0.0, min(max_i, max_j) - max(min_i, min_j))
            denom = max(max_i - min_i, max_j - min_j, 1e-3)
            frac = ov / denom
            if frac > max_overlap:
                max_overlap = frac

    if max_overlap < XRANGE_OVERLAP_THRESHOLD:
        logger.debug(
            "slow_drift_split: skip — shift=%.3f overlap=%.3f below "
            "XRANGE_OVERLAP_THRESHOLD=%.2f",
            max_shift, max_overlap, XRANGE_OVERLAP_THRESHOLD,
        )
        return None

    # Find parent_track_id for drift_pid.
    parent_track_id = -1
    for tid, pid in track_id_to_pid.items():
        if pid == drift_pid:
            parent_track_id = tid
            break

    if parent_track_id < 0:
        return None

    return drift_pid, parent_track_id, max_shift


def _extract_half_region_hists(
    video_path: Any,
    rally_start_ms: int,
    samples: list[PlayerPosition],
    max_frames: int = MAX_FRAMES_PER_HALF,
) -> dict[str, np.ndarray] | None:
    """Average HS histograms by region (lower/upper/head) across frames.

    Returns a dict {"lower": np.ndarray, "upper": np.ndarray | None,
    "head": np.ndarray | None}. Caller takes the MAX chi² across
    available regions as the inter-half distinguishability signal —
    catches both shorts-distinct (lower) and shirts-distinct (upper)
    identity differences without diluting the signal by averaging.
    """
    if not samples:
        return None
    sorted_samples = sorted(samples, key=lambda q: q.frame_number)
    step = max(1, len(sorted_samples) // max_frames)
    chosen = sorted_samples[::step][:max_frames]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    accum_lower: np.ndarray | None = None
    accum_upper: np.ndarray | None = None
    accum_head: np.ndarray | None = None
    count = 0
    for p in chosen:
        ms = rally_start_ms + (int(p.frame_number) * 1000 / fps)
        cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        try:
            features = extract_appearance_features(
                frame=np.asarray(frame, dtype=np.uint8),
                track_id=int(p.track_id),
                frame_number=int(p.frame_number),
                bbox=(float(p.x), float(p.y), float(p.width or 0), float(p.height or 0)),
                frame_width=fw,
                frame_height=fh,
            )
        except Exception:  # noqa: BLE001
            continue
        if features.lower_body_hist is None:
            continue
        accum_lower = (
            features.lower_body_hist.astype(np.float32) if accum_lower is None
            else accum_lower + features.lower_body_hist
        )
        if features.upper_body_hist is not None:
            accum_upper = (
                features.upper_body_hist.astype(np.float32) if accum_upper is None
                else accum_upper + features.upper_body_hist
            )
        if features.head_hist is not None:
            accum_head = (
                features.head_hist.astype(np.float32) if accum_head is None
                else accum_head + features.head_hist
            )
        count += 1
    cap.release()
    if accum_lower is None or count == 0:
        return None

    out: dict[str, np.ndarray] = {"lower": accum_lower / count}
    if accum_upper is not None:
        out["upper"] = accum_upper / count
    if accum_head is not None:
        out["head"] = accum_head / count
    return out


def _best_region_chi2(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray],
) -> tuple[float, str]:
    """MAX chi² across shared regions, with the region name."""
    best_d = 0.0
    best_region = "lower"
    for region in ("lower", "upper", "head"):
        if region in a and region in b:
            d = _hist_chi2(a[region], b[region])
            if d > best_d:
                best_d = d
                best_region = region
    return best_d, best_region


def maybe_emit_slow_drift_split(
    *,
    rally_id: str,
    video_path: Any,
    rally_start_ms: int,
    positions: list[PlayerPosition],
    track_to_player: dict[int, int],
) -> list[SubTrackCandidate] | None:
    """Detect slow-drift on a per-rally assignment and emit sub-tracks.

    Returns a list of TWO sub-tracks covering the full rally range:
    segment 0 (first half, original pid) + segment 1 (second half,
    re-assigned pid). Both must be emitted to avoid the
    `_resolve_with_overrides` UNLABELED trap on parent-track frames
    outside an override range.

    No-op when `ENABLE_SLOW_DRIFT_SPLIT=1` is unset. Caller should
    extend `RallyTrackingResult.sub_tracks` with the returned list.
    """
    if not is_enabled():
        return None

    detected = _detect_slow_drift_pid(positions, track_to_player)
    if detected is None:
        return None
    drift_pid, parent_tid, shift = detected

    # Bisect at median frame number.
    sorted_pos = sorted(
        (p for p in positions if int(p.track_id) == parent_tid),
        key=lambda q: q.frame_number,
    )
    if len(sorted_pos) < 24:
        logger.debug(
            "slow_drift_split %s: PID%d on track %d has only %d positions; skip",
            rally_id[:8] if rally_id else "?", drift_pid, parent_tid, len(sorted_pos),
        )
        return None
    half = len(sorted_pos) // 2
    bisect_frame = int(sorted_pos[half].frame_number)
    rally_end_frame = int(sorted_pos[-1].frame_number)
    first_half = sorted_pos[:half]
    second_half = sorted_pos[half:]

    h_first = _extract_half_region_hists(video_path, rally_start_ms, first_half)
    h_second = _extract_half_region_hists(video_path, rally_start_ms, second_half)
    if h_first is None or h_second is None:
        logger.info(
            "slow_drift_split %s: PID%d feature extraction failed; skip",
            rally_id[:8] if rally_id else "?", drift_pid,
        )
        return None

    inter_half, best_region = _best_region_chi2(h_first, h_second)
    if inter_half < MIN_INTER_HALF_CHI2:
        logger.info(
            "slow_drift_split %s: PID%d halves indistinguishable "
            "(chi2=%.4f best_region=%s < %.2f); skip — sensing limit",
            rally_id[:8] if rally_id else "?", drift_pid, inter_half,
            best_region, MIN_INTER_HALF_CHI2,
        )
        return None

    # Compare each half against OTHER PIDs' first-half profiles, using
    # the SAME region that drove inter-half discriminability.
    track_id_to_pid = {int(k): int(v) for k, v in track_to_player.items() if int(k) > 0}
    other_profiles: dict[int, dict[str, np.ndarray]] = {}
    for tid, other_pid in track_id_to_pid.items():
        if other_pid == drift_pid:
            continue
        other_pos = sorted(
            (p for p in positions if int(p.track_id) == tid),
            key=lambda q: q.frame_number,
        )
        if len(other_pos) < 12:
            continue
        # Use this PID's first half as its representative (its own
        # second half may be drift-contaminated symmetrically).
        oh = _extract_half_region_hists(
            video_path, rally_start_ms, other_pos[: len(other_pos) // 2],
        )
        if oh is not None:
            other_profiles[other_pid] = oh

    if not other_profiles:
        logger.debug(
            "slow_drift_split %s: PID%d no other-PID profiles; skip",
            rally_id[:8] if rally_id else "?", drift_pid,
        )
        return None

    # Distance on the discriminating region only — using the same
    # region that proved inter-half distinguishability.
    distances = {
        other_pid: _hist_chi2(h_second[best_region], prof[best_region])
        for other_pid, prof in other_profiles.items()
        if best_region in prof
    }
    if not distances:
        return None
    best_other_pid = min(distances, key=distances.__getitem__)
    best_other_dist = distances[best_other_pid]
    if best_other_dist >= inter_half:
        logger.info(
            "slow_drift_split %s: PID%d second-half best other-PID dist (%.4f, "
            "PID%d, region=%s) >= inter-half (%.4f); skip — no clear "
            "re-assignment target",
            rally_id[:8] if rally_id else "?", drift_pid, best_other_dist,
            best_other_pid, best_region, inter_half,
        )
        return None

    logger.info(
        "slow_drift_split %s: PID%d (track %d) bisect at frame %d → "
        "second-half re-assigned to PID%d "
        "(region=%s inter-half=%.3f target=%.3f shift=%.3f)",
        rally_id[:8] if rally_id else "?", drift_pid, parent_tid, bisect_frame,
        best_other_pid, best_region, inter_half, best_other_dist, shift,
    )

    # Find the OTHER PID's track — we need to symmetrically swap its
    # late-half assignment to avoid duplicate-PID-per-frame in
    # positions_json. Without this swap, the override on the drift
    # track's late half would create two bboxes labeled PID2 in the
    # same frame (the override's, plus the original PID2 track's).
    other_track_id = -1
    for tid, pid in track_id_to_pid.items():
        if pid == best_other_pid:
            other_track_id = tid
            break
    if other_track_id < 0:
        logger.info(
            "slow_drift_split %s: target PID%d has no track in mapping; skip",
            rally_id[:8] if rally_id else "?", best_other_pid,
        )
        return None

    # Stub appearance stats — required by the SubTrackCandidate
    # dataclass; not serialized into match_analysis_json.
    stub_a = TrackAppearanceStats(track_id=parent_tid)
    stub_b = TrackAppearanceStats(track_id=parent_tid)
    stub_c = TrackAppearanceStats(track_id=other_track_id)
    stub_d = TrackAppearanceStats(track_id=other_track_id)

    # Emit FOUR sub-tracks for a paired late-half swap:
    #   drift_track:    [start, mid-1] → drift_pid (preserves first half)
    #                   [mid,    end]  → best_other_pid (re-assigned)
    #   other_track:    [start, mid-1] → best_other_pid (preserves first half)
    #                   [mid,    end]  → drift_pid (symmetric swap)
    # Both tracks get full-rally coverage so `_resolve_with_overrides`
    # never returns UNLABELED for them. The swap makes the per-frame
    # PID labeling consistent (no duplicates).
    rally_start_frame = int(sorted_pos[0].frame_number)
    first_end = max(rally_start_frame, bisect_frame - 1)
    margin = float(inter_half - best_other_dist)
    return [
        # drift_track first-half: keep drift_pid
        SubTrackCandidate(
            parent_track_id=parent_tid, segment_index=0,
            f_start=rally_start_frame, f_end=first_end,
            appearance_stats=stub_a, aggregated_argmax_pid=drift_pid,
            aggregated_margin=margin,
        ),
        # drift_track second-half: re-assigned to best_other_pid
        SubTrackCandidate(
            parent_track_id=parent_tid, segment_index=1,
            f_start=bisect_frame, f_end=rally_end_frame,
            appearance_stats=stub_b, aggregated_argmax_pid=best_other_pid,
            aggregated_margin=margin,
        ),
        # other_track first-half: keep best_other_pid
        SubTrackCandidate(
            parent_track_id=other_track_id, segment_index=0,
            f_start=rally_start_frame, f_end=first_end,
            appearance_stats=stub_c, aggregated_argmax_pid=best_other_pid,
            aggregated_margin=margin,
        ),
        # other_track second-half: symmetric swap → drift_pid
        SubTrackCandidate(
            parent_track_id=other_track_id, segment_index=1,
            f_start=bisect_frame, f_end=rally_end_frame,
            appearance_stats=stub_d, aggregated_argmax_pid=drift_pid,
            aggregated_margin=margin,
        ),
    ]
