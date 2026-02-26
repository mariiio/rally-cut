"""Height-based track swap detection and correction.

Detects cases where BoT-SORT revives a lost track on the wrong player by
looking for complementary bbox height discontinuities across a gap:
  - Track A: big → small  (near-side player assigned to far-side track)
  - Track B: small → big  (far-side player assigned to near-side track)

With a fixed endline camera, bbox height is a reliable proxy for court depth
(near ≈ 0.25, far ≈ 0.10-0.15). When both tracks show a >30% height change
at the same gap and the heights cross-match (pre_A ≈ post_B, pre_B ≈ post_A),
the fix swaps their track IDs from the gap point onward.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore
from rallycut.tracking.color_repair import ColorHistogramStore
from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_HEIGHT_CHANGE_THRESHOLD = 0.30
DEFAULT_CROSS_MATCH_TOLERANCE = 0.30
DEFAULT_MIN_CONTEXT_FRAMES = 5
DEFAULT_WINDOW_FRAMES = 10
DEFAULT_GAP_ALIGNMENT_TOLERANCE = 30
DEFAULT_MIN_GAP_FRAMES = 3


@dataclass
class HeightSwapDetail:
    """Details of a single height-based track swap."""

    track_a: int
    track_b: int
    swap_frame: int
    pre_h_a: float
    post_h_a: float
    pre_h_b: float
    post_h_b: float


@dataclass
class HeightSwapResult:
    """Result of height-based swap detection."""

    swaps: int = 0
    swap_details: list[HeightSwapDetail] = field(default_factory=list)


@dataclass
class _GapDiscontinuity:
    """Internal: a height discontinuity at a gap in a track."""

    track_id: int
    gap_start: int  # first missing frame
    gap_end: int  # last missing frame
    pre_h: float  # avg height before gap
    post_h: float  # avg height after gap
    rel_change: float  # |pre - post| / max(pre, post)
    post_resume_frame: int  # first frame after the gap


def fix_height_swaps(
    positions: list[PlayerPosition],
    color_store: ColorHistogramStore | None = None,
    appearance_store: AppearanceDescriptorStore | None = None,
    height_change_threshold: float = DEFAULT_HEIGHT_CHANGE_THRESHOLD,
    cross_match_tolerance: float = DEFAULT_CROSS_MATCH_TOLERANCE,
    min_context_frames: int = DEFAULT_MIN_CONTEXT_FRAMES,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    gap_alignment_tolerance: int = DEFAULT_GAP_ALIGNMENT_TOLERANCE,
    min_gap_frames: int = DEFAULT_MIN_GAP_FRAMES,
) -> tuple[list[PlayerPosition], HeightSwapResult]:
    """Detect and fix height-based track swaps.

    Scans each track for gaps where bbox height changes dramatically, then
    checks if another track has a complementary discontinuity at the same
    gap. If so, swaps track IDs from the gap onward.

    Args:
        positions: All player positions (modified in place).
        color_store: Optional histogram store (swapped on fix).
        appearance_store: Optional appearance store (swapped on fix).
        height_change_threshold: Min relative height change to flag (0-1).
        cross_match_tolerance: Max relative diff for cross-match (0-1).
        min_context_frames: Min frames before and after gap for analysis.
        window_frames: Number of frames to average height over.
        gap_alignment_tolerance: Max frame distance between two gaps to
            consider them "at the same location".
        min_gap_frames: Minimum gap size (consecutive missing frames) to
            analyze. Gaps smaller than this are normal tracker jitter.

    Returns:
        Tuple of (positions, HeightSwapResult).
    """
    if not positions:
        return positions, HeightSwapResult()

    result = HeightSwapResult()

    # Group positions by track
    tracks = _group_by_track(positions)
    if len(tracks) < 2:
        return positions, result

    # Step 1: Find height discontinuities in each track
    all_discons: list[_GapDiscontinuity] = []
    for track_id, track_pos in tracks.items():
        discons = _find_height_discontinuities(
            track_id,
            track_pos,
            height_change_threshold=height_change_threshold,
            min_context_frames=min_context_frames,
            window_frames=window_frames,
            min_gap_frames=min_gap_frames,
        )
        all_discons.extend(discons)

    logger.debug(
        "Height consistency: %d tracks analyzed, %d discontinuities found",
        len(tracks),
        len(all_discons),
    )
    for d in all_discons:
        logger.debug(
            "  Track %d: gap [%d-%d], h %.3f->%.3f (%.1f%% change)",
            d.track_id,
            d.gap_start,
            d.gap_end,
            d.pre_h,
            d.post_h,
            d.rel_change * 100,
        )

    if not all_discons:
        return positions, result

    # Step 2: Cross-match complementary discontinuities
    # Track which gaps have already been used to prevent cascading
    used_discons: set[int] = set()  # indices into all_discons

    for i, disc_a in enumerate(all_discons):
        if i in used_discons:
            continue

        for j, disc_b in enumerate(all_discons):
            if j in used_discons or j <= i:
                continue
            if disc_a.track_id == disc_b.track_id:
                continue

            # Check gap alignment
            if not _gaps_aligned(disc_a, disc_b, gap_alignment_tolerance):
                continue

            # Check complementary heights
            if not _heights_cross_match(
                disc_a.pre_h,
                disc_a.post_h,
                disc_b.pre_h,
                disc_b.post_h,
                cross_match_tolerance,
            ):
                continue

            # Cross-match confirmed — swap track IDs from the gap onward
            swap_frame = min(
                disc_a.post_resume_frame, disc_b.post_resume_frame
            )

            _apply_swap(
                positions,
                disc_a.track_id,
                disc_b.track_id,
                swap_frame,
                color_store,
                appearance_store,
            )

            detail = HeightSwapDetail(
                track_a=disc_a.track_id,
                track_b=disc_b.track_id,
                swap_frame=swap_frame,
                pre_h_a=disc_a.pre_h,
                post_h_a=disc_a.post_h,
                pre_h_b=disc_b.pre_h,
                post_h_b=disc_b.post_h,
            )
            result.swaps += 1
            result.swap_details.append(detail)

            logger.info(
                f"Height swap: tracks {disc_a.track_id} <-> {disc_b.track_id} "
                f"at frame {swap_frame} "
                f"(A: {disc_a.pre_h:.3f}->{disc_a.post_h:.3f}, "
                f"B: {disc_b.pre_h:.3f}->{disc_b.post_h:.3f})"
            )

            used_discons.add(i)
            used_discons.add(j)
            break  # Move to next disc_a

    if result.swaps:
        logger.info(f"Height consistency: {result.swaps} swap(s) corrected")

    return positions, result


def _find_height_discontinuities(
    track_id: int,
    track_pos: list[PlayerPosition],
    height_change_threshold: float,
    min_context_frames: int,
    window_frames: int,
    min_gap_frames: int,
) -> list[_GapDiscontinuity]:
    """Find height discontinuities across gaps in a single track."""
    if len(track_pos) < 2 * min_context_frames:
        return []

    sorted_pos = sorted(track_pos, key=lambda p: p.frame_number)
    discons: list[_GapDiscontinuity] = []

    for i in range(len(sorted_pos) - 1):
        curr = sorted_pos[i]
        nxt = sorted_pos[i + 1]
        gap_size = nxt.frame_number - curr.frame_number - 1

        if gap_size < min_gap_frames:
            continue

        # Count positions available before and after gap
        positions_before = i + 1
        positions_after = len(sorted_pos) - (i + 1)

        if positions_before < min_context_frames or positions_after < min_context_frames:
            continue

        # Compute average height in window before gap
        pre_start_idx = max(0, i - window_frames + 1)
        pre_heights = [
            p.height for p in sorted_pos[pre_start_idx : i + 1]
        ]

        # Compute average height in window after gap
        post_end_idx = min(len(sorted_pos), i + 1 + window_frames)
        post_heights = [
            p.height for p in sorted_pos[i + 1 : post_end_idx]
        ]

        if not pre_heights or not post_heights:
            continue

        pre_h = sum(pre_heights) / len(pre_heights)
        post_h = sum(post_heights) / len(post_heights)

        # Relative change
        ref = max(pre_h, post_h)
        if ref <= 0:
            continue
        rel_change = abs(pre_h - post_h) / ref

        if rel_change < height_change_threshold:
            continue

        discons.append(
            _GapDiscontinuity(
                track_id=track_id,
                gap_start=curr.frame_number + 1,
                gap_end=nxt.frame_number - 1,
                pre_h=pre_h,
                post_h=post_h,
                rel_change=rel_change,
                post_resume_frame=nxt.frame_number,
            )
        )

    return discons


def _gaps_aligned(
    a: _GapDiscontinuity,
    b: _GapDiscontinuity,
    tolerance: int,
) -> bool:
    """Check if two gaps are at approximately the same temporal location."""
    return (
        a.gap_start <= b.gap_end + tolerance
        and b.gap_start <= a.gap_end + tolerance
    )


def _heights_cross_match(
    pre_a: float,
    post_a: float,
    pre_b: float,
    post_b: float,
    tolerance: float,
) -> bool:
    """Check if two tracks have complementary (swapped) height discontinuities.

    Track A: pre_a → post_a (e.g., 0.26 → 0.13)
    Track B: pre_b → post_b (e.g., 0.10 → 0.22)
    Cross-check: pre_A ≈ post_B AND pre_B ≈ post_A
    """

    def _approx(v1: float, v2: float) -> bool:
        ref = max(v1, v2)
        return ref > 0 and abs(v1 - v2) / ref <= tolerance

    return _approx(pre_a, post_b) and _approx(pre_b, post_a)


def _apply_swap(
    positions: list[PlayerPosition],
    track_a: int,
    track_b: int,
    from_frame: int,
    color_store: ColorHistogramStore | None,
    appearance_store: AppearanceDescriptorStore | None,
) -> None:
    """Swap track IDs between two tracks from a given frame onward.

    Unlike rekey (one-directional), this exchanges IDs in both directions:
    positions with track_a → track_b and vice versa, for frames >= from_frame.
    """
    for p in positions:
        if p.frame_number >= from_frame:
            if p.track_id == track_a:
                p.track_id = track_b
            elif p.track_id == track_b:
                p.track_id = track_a

    if color_store is not None:
        color_store.swap(track_a, track_b, from_frame)
    if appearance_store is not None:
        appearance_store.swap(track_a, track_b, from_frame)


def _group_by_track(
    positions: list[PlayerPosition],
) -> dict[int, list[PlayerPosition]]:
    """Group positions by track_id, excluding negative IDs."""
    tracks: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if p.track_id >= 0:
            tracks[p.track_id].append(p)
    return dict(tracks)
