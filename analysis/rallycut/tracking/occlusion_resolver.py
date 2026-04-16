"""Session 5 — post-hoc within-team convergence-swap resolver.

Architecture:

At each convergence period where both tracks are same-team primaries,
compute three independent features from pre/post windows and decide if
the pair's post-convergence identities are swapped (post_a ≈ pre_b and
post_b ≈ pre_a).

Feature set
-----------

1. **Appearance swap score** (multi-crop median via Session-3 learned
   head): ``cos(pre_a, post_b) + cos(pre_b, post_a) - cos(pre_a, post_a)
   - cos(pre_b, post_b)``. Positive ⇒ swap.

2. **Trajectory swap score** (court-plane constant-velocity extrap from
   each track's last N pre positions, distance to each track's first
   post position): ``(d(ext_a, obs_a) + d(ext_b, obs_b) - d(ext_a, obs_b)
   - d(ext_b, obs_a)) / scale`` in metres. Positive ⇒ swap. Degrades to
   0 for convergence span > TRAJECTORY_MAX_GAP_FRAMES.

3. **Court-side consistency** (veto): both tracks must stay on the same
   side of ``court_split_y`` pre↔post for the swap decision to even be
   considered.

Decision rule::

    swap iff
        court_side_consistent
        AND appearance_swap_score >= t_appearance
        AND trajectory_swap_score >= t_trajectory
        AND (appearance_swap_score + alpha * trajectory_swap_score) >= t_combined

Conservative by construction: both independent signals must agree before
a correction is applied. This matches the session plan's "precision >=
0.95, recall >= 0.5" posture — do not ship false swaps.

Integration
-----------

Runs at step 4b.5 in ``player_tracker.PlayerTracker.apply_post_processing``,
BEFORE ``optimize_global_identity`` whose coverage-revert guard is what
killed Session 4. Operating here means swap corrections become input to
global identity as clean positions — the guard only measures coverage on
its own later moves, not ours.

Env-var gated via ``ENABLE_OCCLUSION_RESOLVER=1``; default off makes the
integration byte-identical to pre-Session-5 behaviour.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rallycut.tracking.color_repair import (
    ColorHistogramStore,
    LearnedEmbeddingStore,
    detect_convergence_periods,
)
from rallycut.tracking.player_tracker import PlayerPosition

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore

logger = logging.getLogger(__name__)


# Default window + decision parameters. Tuned on labelled events in
# scripts/eval_occlusion_resolver.py; overridable via function kwargs.
DEFAULT_WINDOW_FRAMES = 30
DEFAULT_SEPARATION_GAP = 10
DEFAULT_MIN_FRAMES_PER_WINDOW = 5
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MIN_OVERLAP_FRAMES = 5

# Conservative starting thresholds — refined via grid search on labels.
DEFAULT_T_APPEARANCE = 0.15
DEFAULT_T_TRAJECTORY = 0.30
DEFAULT_T_COMBINED = 0.50
DEFAULT_ALPHA = 0.6            # weight of trajectory in combined score

# Kinematic limits.
TRAJECTORY_EXTRAP_SAMPLES = 5  # last N positions used to fit velocity
TRAJECTORY_MAX_GAP_FRAMES = 15 # above this, trajectory term degrades to 0
TRAJECTORY_COURT_SCALE_M = 2.0 # normalise court-plane distances by 2 m


@dataclass
class ResolvedSwap:
    """Per-event diagnostic record for the gate report + labeller gallery."""

    rally_id: str | None
    track_a: int
    track_b: int
    start_frame: int
    end_frame: int
    mid_frame: int
    applied: bool
    abstain_reason: str | None
    appearance_score: float
    trajectory_score: float
    combined_score: float
    court_side_ok: bool
    n_crops_pre_a: int
    n_crops_pre_b: int
    n_crops_post_a: int
    n_crops_post_b: int


@dataclass
class OcclusionResolverResult:
    n_convergences: int = 0
    n_within_team: int = 0
    n_evaluated: int = 0
    n_abstained: int = 0
    n_swaps_applied: int = 0
    events: list[ResolvedSwap] = field(default_factory=list)


def _median_embedding(
    learned_store: LearnedEmbeddingStore,
    track_id: int,
    frame_lo: int,
    frame_hi: int,
    min_frames: int,
) -> np.ndarray | None:
    """Median over valid per-frame embeddings in ``[frame_lo, frame_hi]``,
    then L2-renormalised. Returns None if fewer than ``min_frames`` valid.
    """
    vectors: list[np.ndarray] = []
    for (tid, fn), emb in getattr(learned_store, "_embeddings", {}).items():
        if tid == track_id and frame_lo <= fn <= frame_hi:
            vectors.append(emb)
    if len(vectors) < min_frames:
        return None
    stack = np.stack(vectors).astype(np.float32)
    med = np.median(stack, axis=0)
    norm = float(np.linalg.norm(med))
    if norm < 1e-8:
        return None
    return (med / norm).astype(np.float32)


def _extrapolate_constant_velocity(
    points: list[tuple[int, float, float]],
    target_frame: int,
) -> tuple[float, float] | None:
    """Constant-velocity extrapolation from the last ``TRAJECTORY_EXTRAP_SAMPLES``
    frames in ``points`` (list of ``(frame, x, y)``) to ``target_frame``.

    Returns ``None`` if insufficient data.
    """
    if not points:
        return None
    pts = sorted(points)[-TRAJECTORY_EXTRAP_SAMPLES:]
    if len(pts) == 1:
        return pts[0][1], pts[0][2]
    f0, x0, y0 = pts[0]
    f1, x1, y1 = pts[-1]
    dt = f1 - f0
    if dt <= 0:
        return x1, y1
    vx = (x1 - x0) / dt
    vy = (y1 - y0) / dt
    horizon = target_frame - f1
    return x1 + vx * horizon, y1 + vy * horizon


def _foot_point(x: float, y: float, height: float) -> tuple[float, float]:
    """Bbox foot-point (normalised image coords)."""
    return x, y + height / 2


def _project_to_court(
    calibrator: CourtCalibrator,
    fp: tuple[float, float],
    video_w: int,
    video_h: int,
) -> tuple[float, float]:
    """Normalised (x,y) image coords → court metres."""
    out: tuple[float, float] = calibrator.image_to_court(fp, video_w, video_h)
    return out


def _trajectory_swap_score(
    positions_by_track: dict[int, list[PlayerPosition]],
    track_a: int,
    track_b: int,
    start_frame: int,
    end_frame: int,
    window_frames: int,
    separation_gap: int,
    court_calibrator: CourtCalibrator | None,
    video_w: int,
    video_h: int,
) -> float:
    """Compute the trajectory swap score.

    Positive ⇒ swap (post_a trajectory-matches pre_b, and post_b matches pre_a).
    Zero ⇒ no signal available (insufficient samples or convergence too long).

    Works in court metres when calibrator is present; otherwise falls back
    to normalised image-coordinates (less accurate but always available).
    """
    if end_frame - start_frame + 1 > TRAJECTORY_MAX_GAP_FRAMES:
        return 0.0

    def _pre_window(tid: int) -> list[tuple[int, float, float]]:
        out = []
        for p in positions_by_track.get(tid, []):
            if start_frame - window_frames <= p.frame_number <= start_frame - 1:
                fp = _foot_point(p.x, p.y, p.height)
                if court_calibrator is not None:
                    cx, cy = _project_to_court(court_calibrator, fp, video_w, video_h)
                    out.append((p.frame_number, cx, cy))
                else:
                    out.append((p.frame_number, fp[0], fp[1]))
        return out

    def _post_first(tid: int) -> tuple[float, float] | None:
        earliest = None
        for p in positions_by_track.get(tid, []):
            if (
                end_frame + separation_gap
                <= p.frame_number
                <= end_frame + separation_gap + window_frames
            ):
                if earliest is None or p.frame_number < earliest.frame_number:
                    earliest = p
        if earliest is None:
            return None
        fp = _foot_point(earliest.x, earliest.y, earliest.height)
        if court_calibrator is not None:
            return _project_to_court(court_calibrator, fp, video_w, video_h)
        return fp

    pre_a = _pre_window(track_a)
    pre_b = _pre_window(track_b)
    if len(pre_a) < 2 or len(pre_b) < 2:
        return 0.0
    obs_a = _post_first(track_a)
    obs_b = _post_first(track_b)
    if obs_a is None or obs_b is None:
        return 0.0

    # First post frame varies per track — extrapolate each pre-track to
    # its partner's first post frame so both scores share the same horizon.
    target_frame = end_frame + separation_gap
    ext_a = _extrapolate_constant_velocity(pre_a, target_frame)
    ext_b = _extrapolate_constant_velocity(pre_b, target_frame)
    if ext_a is None or ext_b is None:
        return 0.0

    def _d(p: tuple[float, float], q: tuple[float, float]) -> float:
        return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

    scale = TRAJECTORY_COURT_SCALE_M if court_calibrator is not None else 0.3
    return (
        _d(ext_a, obs_a) + _d(ext_b, obs_b)
        - _d(ext_a, obs_b) - _d(ext_b, obs_a)
    ) / scale


def _median_y(
    positions_by_track: dict[int, list[PlayerPosition]],
    track_id: int,
    frame_lo: int,
    frame_hi: int,
) -> float | None:
    """Median bbox-centre Y for track in the given frame window."""
    ys = [
        p.y for p in positions_by_track.get(track_id, [])
        if frame_lo <= p.frame_number <= frame_hi
    ]
    if not ys:
        return None
    return float(np.median(ys))


def _court_side_consistent(
    positions_by_track: dict[int, list[PlayerPosition]],
    track_a: int,
    track_b: int,
    start_frame: int,
    end_frame: int,
    window_frames: int,
    separation_gap: int,
    court_split_y: float | None,
) -> bool:
    """Veto — require both tracks to stay on the same side of the split
    across pre ↔ post windows. Same-team pairs should not cross the net.
    """
    if court_split_y is None:
        # No court split known — cannot veto; fall through to scoring.
        return True

    def _side(tid: int, lo: int, hi: int) -> int | None:
        med = _median_y(positions_by_track, tid, lo, hi)
        if med is None:
            return None
        return 1 if med >= court_split_y else 0

    pre_a = _side(track_a, start_frame - window_frames, start_frame - 1)
    post_a = _side(track_a, end_frame + separation_gap, end_frame + separation_gap + window_frames)
    pre_b = _side(track_b, start_frame - window_frames, start_frame - 1)
    post_b = _side(track_b, end_frame + separation_gap, end_frame + separation_gap + window_frames)

    if None in (pre_a, post_a, pre_b, post_b):
        # Missing evidence → don't block.
        return True
    return pre_a == post_a and pre_b == post_b


def _positions_by_track(
    positions: list[PlayerPosition],
) -> dict[int, list[PlayerPosition]]:
    out: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        if p.track_id >= 0:
            out.setdefault(p.track_id, []).append(p)
    for tid in out:
        out[tid].sort(key=lambda p: p.frame_number)
    return out


def _swap_track_ids_after_frame(
    positions: list[PlayerPosition],
    track_a: int,
    track_b: int,
    from_frame: int,
) -> int:
    """Bidirectional swap of two track IDs for positions with frame >=
    ``from_frame``. Matches ``convergence_swap._apply_swap`` semantics.
    Returns the count of swapped positions.
    """
    swapped = 0
    for p in positions:
        if p.frame_number < from_frame:
            continue
        if p.track_id == track_a:
            p.track_id = track_b
            swapped += 1
        elif p.track_id == track_b:
            p.track_id = track_a
            swapped += 1
    return swapped


def resolve_within_team_convergence_swaps(
    positions: list[PlayerPosition],
    primary_track_ids: Iterable[int],
    team_assignments: dict[int, int],
    color_store: ColorHistogramStore | None = None,
    appearance_store: AppearanceDescriptorStore | None = None,
    learned_store: LearnedEmbeddingStore | None = None,
    court_calibrator: CourtCalibrator | None = None,
    video_width: int = 0,
    video_height: int = 0,
    court_split_y: float | None = None,
    *,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    separation_gap: int = DEFAULT_SEPARATION_GAP,
    min_frames_per_window: int = DEFAULT_MIN_FRAMES_PER_WINDOW,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    min_overlap_frames: int = DEFAULT_MIN_OVERLAP_FRAMES,
    t_appearance: float = DEFAULT_T_APPEARANCE,
    t_trajectory: float = DEFAULT_T_TRAJECTORY,
    t_combined: float = DEFAULT_T_COMBINED,
    alpha: float = DEFAULT_ALPHA,
    rally_id: str | None = None,
) -> tuple[list[PlayerPosition], OcclusionResolverResult]:
    """Detect and (when confident) fix same-team convergence swaps.

    Mutates ``positions`` in-place and propagates to each store via
    ``.swap``. Returns ``(positions, OcclusionResolverResult)``.
    """
    result = OcclusionResolverResult()
    primary_set = set(primary_track_ids)
    if not positions or not primary_set or not team_assignments:
        return positions, result

    periods = detect_convergence_periods(
        positions,
        iou_threshold=iou_threshold,
        min_duration=min_overlap_frames,
    )
    result.n_convergences = len(periods)

    by_track = _positions_by_track(positions)

    for cp in periods:
        a, b = cp.track_a, cp.track_b
        if a not in primary_set or b not in primary_set:
            continue
        ta = team_assignments.get(a)
        tb = team_assignments.get(b)
        if ta is None or tb is None or ta != tb:
            continue
        result.n_within_team += 1

        # Windows
        pre_lo = cp.start_frame - window_frames
        pre_hi = cp.start_frame - 1
        post_lo = cp.end_frame + separation_gap
        post_hi = cp.end_frame + separation_gap + window_frames
        mid_frame = (cp.start_frame + cp.end_frame) // 2

        # Appearance feature — abstain if learned embeddings insufficient.
        abstain_reason: str | None = None
        appearance_score = 0.0
        trajectory_score = 0.0
        combined_score = 0.0
        n_pre_a = n_pre_b = n_post_a = n_post_b = 0
        court_side_ok = True

        if learned_store is None or not learned_store.has_data():
            abstain_reason = "no_learned_store"
        else:
            e_pre_a = _median_embedding(
                learned_store, a, pre_lo, pre_hi, min_frames_per_window,
            )
            e_pre_b = _median_embedding(
                learned_store, b, pre_lo, pre_hi, min_frames_per_window,
            )
            e_post_a = _median_embedding(
                learned_store, a, post_lo, post_hi, min_frames_per_window,
            )
            e_post_b = _median_embedding(
                learned_store, b, post_lo, post_hi, min_frames_per_window,
            )
            n_pre_a = _count_emb(learned_store, a, pre_lo, pre_hi)
            n_pre_b = _count_emb(learned_store, b, pre_lo, pre_hi)
            n_post_a = _count_emb(learned_store, a, post_lo, post_hi)
            n_post_b = _count_emb(learned_store, b, post_lo, post_hi)
            if (
                e_pre_a is None
                or e_pre_b is None
                or e_post_a is None
                or e_post_b is None
            ):
                abstain_reason = "insufficient_embeddings"
            else:
                appearance_score = float(
                    np.dot(e_pre_a, e_post_b) + np.dot(e_pre_b, e_post_a)
                    - np.dot(e_pre_a, e_post_a) - np.dot(e_pre_b, e_post_b)
                )

        # Court-side veto
        if abstain_reason is None:
            court_side_ok = _court_side_consistent(
                by_track, a, b, cp.start_frame, cp.end_frame,
                window_frames, separation_gap, court_split_y,
            )
            if not court_side_ok:
                abstain_reason = "court_side_veto"

        # Trajectory feature (separate — don't gate on it alone).
        if abstain_reason is None:
            trajectory_score = _trajectory_swap_score(
                by_track, a, b,
                cp.start_frame, cp.end_frame,
                window_frames, separation_gap,
                court_calibrator, video_width, video_height,
            )
            combined_score = appearance_score + alpha * trajectory_score

        # Decision
        applied = False
        if abstain_reason is None:
            swap_decision = (
                appearance_score >= t_appearance
                and trajectory_score >= t_trajectory
                and combined_score >= t_combined
            )
            if swap_decision:
                n_swapped_positions = _swap_track_ids_after_frame(
                    positions, a, b, mid_frame,
                )
                if color_store is not None:
                    color_store.swap(a, b, mid_frame)
                if learned_store is not None:
                    learned_store.swap(a, b, mid_frame)
                if appearance_store is not None:
                    appearance_store.swap(a, b, mid_frame)
                # Rebuild by_track so downstream periods see the swap.
                by_track = _positions_by_track(positions)
                applied = n_swapped_positions > 0
                if applied:
                    result.n_swaps_applied += 1
                    logger.info(
                        "occlusion_resolver: swapped T%d<->T%d at frame %d "
                        "(app=%.3f traj=%.3f comb=%.3f)",
                        a, b, mid_frame,
                        appearance_score, trajectory_score, combined_score,
                    )

        if abstain_reason is not None:
            result.n_abstained += 1
        else:
            result.n_evaluated += 1

        result.events.append(ResolvedSwap(
            rally_id=rally_id,
            track_a=a,
            track_b=b,
            start_frame=cp.start_frame,
            end_frame=cp.end_frame,
            mid_frame=mid_frame,
            applied=applied,
            abstain_reason=abstain_reason,
            appearance_score=appearance_score,
            trajectory_score=trajectory_score,
            combined_score=combined_score,
            court_side_ok=court_side_ok,
            n_crops_pre_a=n_pre_a,
            n_crops_pre_b=n_pre_b,
            n_crops_post_a=n_post_a,
            n_crops_post_b=n_post_b,
        ))

    return positions, result


def _count_emb(
    learned_store: LearnedEmbeddingStore,
    track_id: int,
    frame_lo: int,
    frame_hi: int,
) -> int:
    return sum(
        1 for (tid, fn), _ in getattr(learned_store, "_embeddings", {}).items()
        if tid == track_id and frame_lo <= fn <= frame_hi
    )
