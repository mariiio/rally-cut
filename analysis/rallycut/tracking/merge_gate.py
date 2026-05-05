"""Unified merge-gate decisions for player track merging.

Consolidates the gates that decide whether two tracks should be merged
into one (assigned the same canonical ID). Single source of truth for
all merge stages:
  - `tracklet_link.link_tracklets_by_appearance` (`_greedy_merge`,
    `_swap_optimize`)
  - `tracklet_link.relink_spatial_splits`
  - `tracklet_link.relink_primary_fragments`
  - `player_filter.stabilize_track_ids`
  - `global_identity.optimize_global_identity`

Gates (applied in order, cheapest first):
  1. Temporal overlap: tracks must not coexist for more than
     `max_overlap_frames`.
  2. Velocity anomaly: court-plane displacement (when calibrated) or
     image-plane displacement (fallback) must not exceed limits over a
     sliding window around the junction.
  3. Learned ReID cosine veto: if the fine-tuned OSNet head says these
     fragments are different players (cosine similarity < threshold),
     block. Skipped when threshold <= 0.

Per-gate enables let each call site keep its historical behavior while
opting into stricter gates as we tighten the pipeline. Pass a custom
`MergeGateConfig` to disable individual gates or change thresholds.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rallycut.tracking.merge_veto import learned_cosine_veto
from rallycut.tracking.player_tracker import PlayerPosition

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.color_repair import LearnedEmbeddingStore

logger = logging.getLogger(__name__)


# --- Defaults (mirror tracklet_link.py for bit-exact backward compat) ----

DEFAULT_MAX_DISPLACEMENT_IMAGE = 0.20
DEFAULT_MAX_DISPLACEMENT_METERS = float(
    os.environ.get("RALLYCUT_MAX_MERGE_VELOCITY_METERS", 2.5)
)
DEFAULT_VELOCITY_WINDOW = 10
DEFAULT_MAX_OVERLAP_FRAMES = 15
DEFAULT_LEARNED_VETO_COS = float(
    os.environ.get("LEARNED_MERGE_VETO_COS", "0.0")
)

# Margin (metres) outside the court lines within which we still trust the
# homography. See `tracklet_link.COURT_FOOT_TRUST_MARGIN_M` history.
COURT_FOOT_TRUST_MARGIN_M = 2.0


# --- Foot-court projection helpers (moved from tracklet_link.py) ---------


def foot_court_coord(
    calibrator: CourtCalibrator, p: PlayerPosition,
) -> tuple[float, float]:
    """Project a bbox foot point (bottom edge centre) to court coords.

    Foot = (x, y + height/2) — the only image-y that lies on the court
    floor for a standing player. The bbox center for standing players
    sits ABOVE the trapezoid's far baseline and projects to wildly
    extrapolated coordinates; foot-y projections land in the calibrated
    region where the homography is reliable.
    """
    fx, fy = p.x, p.y + p.height / 2
    return calibrator.image_to_court((fx, fy), 0, 0)


def both_feet_in_court(
    calibrator: CourtCalibrator,
    pa: PlayerPosition,
    pb: PlayerPosition,
    margin_m: float = COURT_FOOT_TRUST_MARGIN_M,
) -> bool:
    """True iff both foot projections land within court+margin."""
    return (
        calibrator.is_point_in_court(
            foot_court_coord(calibrator, pa), margin=margin_m,
        )
        and calibrator.is_point_in_court(
            foot_court_coord(calibrator, pb), margin=margin_m,
        )
    )


def court_displacement_meters(
    calibrator: CourtCalibrator,
    pa: PlayerPosition,
    pb: PlayerPosition,
) -> float:
    """Euclidean court-plane distance in metres between two foot projections.

    Caller must verify both points are in the trusted region via
    `both_feet_in_court` — outside that region the homography
    extrapolates and the return value is not physically meaningful.
    """
    cax, cay = foot_court_coord(calibrator, pa)
    cbx, cby = foot_court_coord(calibrator, pb)
    dx, dy = cbx - cax, cby - cay
    return float((dx * dx + dy * dy) ** 0.5)


# --- Gate primitives ------------------------------------------------------


def tracks_overlap_temporally(
    frames_a: set[int] | list[int],
    frames_b: set[int] | list[int],
    max_allowed_overlap: int = 0,
) -> bool:
    """True iff tracks share more than `max_allowed_overlap` frames.

    `max_allowed_overlap=0` is strict (any overlap blocks). Higher values
    tolerate brief Kalman-ghost handoffs (BoT-SORT briefly carries a
    dying track plus a new detection at occlusion exit).
    """
    a = frames_a if isinstance(frames_a, set) else set(frames_a)
    b = frames_b if isinstance(frames_b, set) else set(frames_b)
    return len(a & b) > max_allowed_overlap


def velocity_anomaly(
    positions: list[PlayerPosition],
    tid_a: int,
    tid_b: int,
    max_displacement_image: float = DEFAULT_MAX_DISPLACEMENT_IMAGE,
    window: int = DEFAULT_VELOCITY_WINDOW,
    *,
    calibrator: CourtCalibrator | None = None,
    max_displacement_meters: float = DEFAULT_MAX_DISPLACEMENT_METERS,
) -> bool:
    """True iff merging `tid_a` and `tid_b` would create impossible velocity.

    Examines positions from both tracks near the temporal boundary where
    they meet. If any pair within `window` frames has displacement
    exceeding the threshold, the merge is rejected.

    When a calibrated `calibrator` is passed, displacement is measured
    in court-plane metres (compared to `max_displacement_meters`). When
    either point falls outside the trusted court+margin region, falls
    back to image-plane normalized distance. This is the original
    behaviour of `tracklet_link._would_create_velocity_anomaly` —
    preserved bit-exact.

    Env knobs (preserved):
      - `RALLYCUT_DISABLE_COURT_VELOCITY_GATE=1` — force image-plane
        always.
      - `RALLYCUT_COURT_GATE_ADDITIVE=1` — additive mode: image-plane
        always active; court-plane adds rejections on top when points
        are in-court. Strictly stricter.
    """
    pos_a = sorted(
        [p for p in positions if p.track_id == tid_a],
        key=lambda p: p.frame_number,
    )
    pos_b = sorted(
        [p for p in positions if p.track_id == tid_b],
        key=lambda p: p.frame_number,
    )

    if not pos_a or not pos_b:
        return False

    court_gate_enabled = (
        calibrator is not None
        and calibrator.is_calibrated
        and os.environ.get("RALLYCUT_DISABLE_COURT_VELOCITY_GATE", "0") != "1"
    )
    additive_mode = os.environ.get("RALLYCUT_COURT_GATE_ADDITIVE", "0") == "1"

    def _image_dist(pa: PlayerPosition, pb: PlayerPosition) -> float:
        dx = pb.x - pa.x
        dy = pb.y - pa.y
        return float((dx * dx + dy * dy) ** 0.5)

    def _exceeds_gate(pa: PlayerPosition, pb: PlayerPosition) -> bool:
        if additive_mode and _image_dist(pa, pb) > max_displacement_image:
            return True
        if (
            court_gate_enabled
            and calibrator is not None
            and both_feet_in_court(calibrator, pa, pb)
        ):
            return court_displacement_meters(
                calibrator, pa, pb,
            ) > max_displacement_meters
        return _image_dist(pa, pb) > max_displacement_image

    # Determine temporal order
    if pos_a[-1].frame_number <= pos_b[0].frame_number:
        earlier, later = pos_a, pos_b
    elif pos_b[-1].frame_number <= pos_a[0].frame_number:
        earlier, later = pos_b, pos_a
    else:
        # Overlapping — check positions near the overlap boundary
        overlap_start = max(pos_a[0].frame_number, pos_b[0].frame_number)
        tail = [p for p in pos_a if abs(p.frame_number - overlap_start) <= window]
        head = [p for p in pos_b if abs(p.frame_number - overlap_start) <= window]
        for pa in tail:
            for pb in head:
                frame_gap = abs(pb.frame_number - pa.frame_number)
                if frame_gap > window or frame_gap == 0:
                    continue
                if _exceeds_gate(pa, pb):
                    return True
        return False

    # Endpoint displacement check (regardless of gap size)
    end_pos = earlier[-1]
    start_pos = later[0]
    if _exceeds_gate(end_pos, start_pos):
        return True

    # Sliding window near junction for short-gap merges
    tail = [p for p in earlier if p.frame_number >= earlier[-1].frame_number - window]
    head = [p for p in later if p.frame_number <= later[0].frame_number + window]
    for pa in tail:
        for pb in head:
            frame_gap = abs(pb.frame_number - pa.frame_number)
            if frame_gap > window or frame_gap == 0:
                continue
            if _exceeds_gate(pa, pb):
                return True

    return False


# --- Unified merge-gate API ----------------------------------------------


@dataclass(frozen=True)
class MergeGateConfig:
    """Per-call-site configuration of which merge gates to apply.

    Each gate has an `enable_<name>` flag; threshold knobs let callers
    tune sensitivity without disabling. `learned_veto_cos <= 0` disables
    the learned-ReID gate even if `enable_learned=True`.

    Defaults match `tracklet_link.link_tracklets_by_appearance`'s
    historical behaviour for backward compatibility. Other call sites
    should construct configs that mirror their existing gates plus any
    newly-enabled ones.
    """

    max_displacement_image: float = DEFAULT_MAX_DISPLACEMENT_IMAGE
    max_displacement_meters: float = DEFAULT_MAX_DISPLACEMENT_METERS
    velocity_window: int = DEFAULT_VELOCITY_WINDOW
    max_overlap_frames: int = DEFAULT_MAX_OVERLAP_FRAMES
    learned_veto_cos: float = DEFAULT_LEARNED_VETO_COS

    enable_overlap: bool = True
    enable_velocity: bool = True
    enable_learned: bool = True


@dataclass(frozen=True)
class MergeGateResult:
    """Outcome of `should_block_merge`."""

    blocked: bool
    reason: str = ""  # empty when not blocked


def should_block_merge(
    track_id_a: int,
    track_id_b: int,
    positions: list[PlayerPosition],
    *,
    frames_a: set[int] | list[int] | None = None,
    frames_b: set[int] | list[int] | None = None,
    config: MergeGateConfig | None = None,
    calibrator: CourtCalibrator | None = None,
    learned_store: LearnedEmbeddingStore | None = None,
) -> MergeGateResult:
    """Single-source-of-truth merge decision.

    Args:
        track_id_a, track_id_b: Tracks proposed for merging.
        positions: All player positions (with ORIGINAL track IDs). Used
            by the velocity gate.
        frames_a, frames_b: Optional pre-computed frame sets, used by
            overlap + learned-ReID gates. Derived from `positions` when
            not provided.
        config: Gate configuration. Defaults to module-level DEFAULTS.
        calibrator: Optional court calibrator. Court-plane velocity
            gating activates when calibrated; image-plane fallback
            otherwise.
        learned_store: Optional learned-ReID embedding store. Learned
            veto abstains when None.

    Returns:
        MergeGateResult.blocked = True iff any enabled gate blocks. The
        `reason` field carries the first-blocking gate's name + a short
        explainer for logging.
    """
    cfg = config or MergeGateConfig()

    if frames_a is None or frames_b is None:
        frames_a_derived: set[int] = set()
        frames_b_derived: set[int] = set()
        for p in positions:
            if p.track_id == track_id_a:
                frames_a_derived.add(p.frame_number)
            elif p.track_id == track_id_b:
                frames_b_derived.add(p.frame_number)
        if frames_a is None:
            frames_a = frames_a_derived
        if frames_b is None:
            frames_b = frames_b_derived

    # Gate 1: temporal overlap (cheapest)
    if cfg.enable_overlap:
        if tracks_overlap_temporally(
            frames_a, frames_b, max_allowed_overlap=cfg.max_overlap_frames,
        ):
            a = frames_a if isinstance(frames_a, set) else set(frames_a)
            b = frames_b if isinstance(frames_b, set) else set(frames_b)
            return MergeGateResult(
                blocked=True,
                reason=f"temporal_overlap({len(a & b)}>{cfg.max_overlap_frames})",
            )

    # Gate 2: velocity anomaly
    if cfg.enable_velocity:
        if velocity_anomaly(
            positions, track_id_a, track_id_b,
            max_displacement_image=cfg.max_displacement_image,
            window=cfg.velocity_window,
            calibrator=calibrator,
            max_displacement_meters=cfg.max_displacement_meters,
        ):
            return MergeGateResult(
                blocked=True,
                reason="velocity_anomaly",
            )

    # Gate 3: learned-ReID cosine veto
    if (
        cfg.enable_learned
        and cfg.learned_veto_cos > 0.0
        and learned_store is not None
    ):
        if learned_cosine_veto(
            learned_store,
            track_id_a, frames_a,
            track_id_b, frames_b,
            threshold=cfg.learned_veto_cos,
        ):
            return MergeGateResult(
                blocked=True,
                reason=f"learned_reid(cos<{cfg.learned_veto_cos:.2f})",
            )

    return MergeGateResult(blocked=False)
