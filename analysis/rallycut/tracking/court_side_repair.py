"""Court-side post-hoc repair for cross-team ID switches.

After global identity optimization, some cross-team switches may persist
(e.g., when global identity skips because "tracks already clean" or has
insufficient appearance contrast). This module detects tracks that spend
sustained time on the wrong court side, finds complementary swap pairs
from opposite teams, and swaps track IDs to fix the switch.

Key design: the module computes its own court split from the primary
tracks' early-frame Y positions, rather than relying on the upstream
court_split_y which can be wrong (e.g., when ball tracking is poor).
This makes the repair robust to upstream court detection failures.

Detection strategy:
1. Compute court split by clustering primary tracks into 2 groups by
   median Y of early frames, then taking the midpoint.
2. For each track, compute the fraction of classifiable frames on the
   wrong court side. Flag tracks exceeding thresholds.
3. Find swap pairs: two tracks from opposite teams with wrong-side
   presence in overlapping or temporally close time ranges.
4. Validate with appearance (optional): pre-swap A should look like
   post-swap B.

Inserted as Step 4d after global identity and before quality report.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from rallycut.tracking.color_repair import ColorHistogramStore
from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Court split computation
EARLY_FRAMES_WINDOW = 60  # Use first N frames per track for team anchoring
MIN_SPLIT_GAP = 0.03  # Min Y gap between near/far groups to trust the split

# Detection thresholds
MIN_WRONG_SIDE_FRAMES = 15  # Minimum frames on wrong side to flag
NEAR_NET_MARGIN = 0.03  # Ignore positions within this of court_split_y
MIN_WRONG_SIDE_RATIO = 0.25  # Min fraction of classifiable frames on wrong side
APPEARANCE_VALIDATION_THRESHOLD = 0.60  # Max Bhattacharyya for swap validation
MAX_PAIR_GAP_FRAMES = 200  # Max gap between non-overlapping wrong-side periods


@dataclass
class _WrongSideInfo:
    """Summary of a track's wrong-side presence."""

    track_id: int
    team: int  # 0=near, 1=far
    wrong_frames: int  # Total frames on wrong side (excluding near-net)
    total_classifiable: int  # Total frames outside near-net margin
    wrong_ratio: float  # wrong_frames / total_classifiable
    first_wrong_frame: int  # Earliest frame on wrong side
    last_wrong_frame: int  # Latest frame on wrong side
    transition_frame: int  # Best estimate of where right->wrong transition occurs


def _compute_court_split_from_tracks(
    positions: list[PlayerPosition],
    primary_track_ids: list[int],
) -> tuple[float, dict[int, int]] | None:
    """Compute court split and team assignments from primary track positions.

    Uses the first EARLY_FRAMES_WINDOW frames per track to compute median Y,
    then clusters the 4 tracks into 2 groups (near=high Y, far=low Y).

    Returns (split_y, team_assignments) or None if clustering fails.
    """
    if len(primary_track_ids) < 4:
        return None

    # Compute median Y from early frames for each primary track
    track_early: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for p in positions:
        if p.track_id in primary_track_ids:
            track_early[p.track_id].append((p.frame_number, p.y))

    median_ys: dict[int, float] = {}
    for tid in primary_track_ids:
        entries = track_early.get(tid, [])
        if not entries:
            continue
        # Sort by frame, take first EARLY_FRAMES_WINDOW
        entries.sort(key=lambda x: x[0])
        early_ys = [y for _, y in entries[:EARLY_FRAMES_WINDOW]]
        if early_ys:
            early_ys.sort()
            median_ys[tid] = early_ys[len(early_ys) // 2]

    if len(median_ys) < 4:
        return None

    # Sort tracks by median Y (ascending = far-side first, near-side last)
    sorted_tracks = sorted(median_ys.items(), key=lambda x: x[1])

    # Split into 2 groups: far (lower Y) and near (higher Y)
    far_tracks = sorted_tracks[:2]
    near_tracks = sorted_tracks[2:]

    # Check gap between groups is large enough
    max_far_y = far_tracks[-1][1]
    min_near_y = near_tracks[0][1]
    gap = min_near_y - max_far_y

    if gap < MIN_SPLIT_GAP:
        logger.debug(
            f"Court-side repair: Y gap too small ({gap:.3f} < {MIN_SPLIT_GAP}), "
            f"far=[{far_tracks[0][1]:.3f}, {far_tracks[1][1]:.3f}], "
            f"near=[{near_tracks[0][1]:.3f}, {near_tracks[1][1]:.3f}]"
        )
        return None

    split_y = (max_far_y + min_near_y) / 2.0

    team_assignments: dict[int, int] = {}
    for tid, _ in near_tracks:
        team_assignments[tid] = 0  # near
    for tid, _ in far_tracks:
        team_assignments[tid] = 1  # far

    logger.debug(
        f"Court-side repair: self-computed split_y={split_y:.3f} (gap={gap:.3f}), "
        f"near={[t[0] for t in near_tracks]}, far={[t[0] for t in far_tracks]}"
    )

    return split_y, team_assignments


def _analyze_wrong_side(
    positions: list[PlayerPosition],
    team_assignments: dict[int, int],
    court_split_y: float,
) -> list[_WrongSideInfo]:
    """Analyze wrong-side presence for all tracks.

    Computes overall wrong-side statistics and finds the best transition
    frame (the point where a track shifts from predominantly correct side
    to predominantly wrong side).
    """
    track_positions: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if p.track_id >= 0 and p.track_id in team_assignments:
            track_positions[p.track_id].append(p)

    results: list[_WrongSideInfo] = []

    for track_id, track_pos in track_positions.items():
        team = team_assignments[track_id]
        track_pos.sort(key=lambda p: p.frame_number)

        # Classify each frame as right/wrong (skip near-net)
        frame_sides: list[tuple[int, bool]] = []
        for p in track_pos:
            if abs(p.y - court_split_y) < NEAR_NET_MARGIN:
                continue
            on_near_side = p.y > court_split_y
            expected_near = team == 0
            frame_sides.append((p.frame_number, on_near_side != expected_near))

        if not frame_sides:
            continue

        wrong_frames = sum(1 for _, w in frame_sides if w)
        total = len(frame_sides)
        ratio = wrong_frames / total

        if wrong_frames < MIN_WRONG_SIDE_FRAMES or ratio < MIN_WRONG_SIDE_RATIO:
            continue

        wrong_frame_nums = [f for f, w in frame_sides if w]
        first_wrong = wrong_frame_nums[0]
        last_wrong = wrong_frame_nums[-1]

        # Find transition frame using cumulative counts.
        # Best transition = maximizes (right before) * (wrong after).
        transition_frame = first_wrong
        if total >= 20:
            cum_wrong = [0] * (total + 1)
            for i, (_, w) in enumerate(frame_sides):
                cum_wrong[i + 1] = cum_wrong[i] + (1 if w else 0)

            best_score = -1.0
            for i in range(1, total):
                right_before = i - cum_wrong[i]
                wrong_after = cum_wrong[total] - cum_wrong[i]
                score = (right_before / i) * (wrong_after / (total - i))
                if score > best_score:
                    best_score = score
                    transition_frame = frame_sides[i][0]

        results.append(
            _WrongSideInfo(
                track_id=track_id,
                team=team,
                wrong_frames=wrong_frames,
                total_classifiable=total,
                wrong_ratio=ratio,
                first_wrong_frame=first_wrong,
                last_wrong_frame=last_wrong,
                transition_frame=transition_frame,
            )
        )

    return results


def _find_swap_pairs(
    wrong_sides: list[_WrongSideInfo],
) -> list[tuple[_WrongSideInfo, _WrongSideInfo]]:
    """Find complementary swap pairs from opposite teams.

    Matches tracks from opposite teams that both have wrong-side presence.
    Accepts overlapping OR temporally close wrong-side periods (same swap
    event can cause non-overlapping wrong-side runs if one player returns
    to position faster). Prioritizes strongest violations.
    """
    sorted_ws = sorted(wrong_sides, key=lambda w: -w.wrong_ratio)

    pairs: list[tuple[_WrongSideInfo, _WrongSideInfo]] = []
    used: set[int] = set()

    for wa in sorted_ws:
        if wa.track_id in used:
            continue

        best_partner: _WrongSideInfo | None = None
        best_score = -1.0

        for wb in sorted_ws:
            if wb.track_id in used or wb.track_id == wa.track_id:
                continue
            if wa.team == wb.team:
                continue

            overlap_start = max(wa.first_wrong_frame, wb.first_wrong_frame)
            overlap_end = min(wa.last_wrong_frame, wb.last_wrong_frame)

            if overlap_start <= overlap_end:
                proximity = min((overlap_end - overlap_start) / 100.0, 1.0)
            else:
                gap = overlap_start - overlap_end
                if gap > MAX_PAIR_GAP_FRAMES:
                    continue
                proximity = 0.5 * (1.0 - gap / MAX_PAIR_GAP_FRAMES)

            score = wb.wrong_ratio * proximity
            if score > best_score:
                best_score = score
                best_partner = wb

        if best_partner is not None:
            pairs.append((wa, best_partner))
            used.add(wa.track_id)
            used.add(best_partner.track_id)

    return pairs


def _validate_swap_appearance(
    track_a: int,
    track_b: int,
    swap_frame: int,
    color_store: ColorHistogramStore,
) -> bool:
    """Validate swap with appearance: pre-swap A should look like post-swap B.

    Returns True if at least one cross-check passes, or if insufficient data.
    """
    a_hists = color_store.get_track_histograms(track_a)
    b_hists = color_store.get_track_histograms(track_b)

    a_before = [h for fn, h in a_hists if fn < swap_frame]
    a_after = [h for fn, h in a_hists if fn >= swap_frame]
    b_before = [h for fn, h in b_hists if fn < swap_frame]
    b_after = [h for fn, h in b_hists if fn >= swap_frame]

    def _mean_hist(hists: list[np.ndarray]) -> np.ndarray | None:
        if not hists:
            return None
        mean: np.ndarray = np.mean(np.stack(hists), axis=0).astype(np.float32)
        total = mean.sum()
        if total > 0:
            mean /= total
        return mean

    mean_a_before = _mean_hist(a_before)
    mean_a_after = _mean_hist(a_after)
    mean_b_before = _mean_hist(b_before)
    mean_b_after = _mean_hist(b_after)

    checks_passed = 0
    checks_total = 0

    if mean_a_before is not None and mean_b_after is not None:
        dist = cv2.compareHist(
            mean_a_before, mean_b_after, cv2.HISTCMP_BHATTACHARYYA
        )
        checks_total += 1
        if dist < APPEARANCE_VALIDATION_THRESHOLD:
            checks_passed += 1

    if mean_b_before is not None and mean_a_after is not None:
        dist = cv2.compareHist(
            mean_b_before, mean_a_after, cv2.HISTCMP_BHATTACHARYYA
        )
        checks_total += 1
        if dist < APPEARANCE_VALIDATION_THRESHOLD:
            checks_passed += 1

    if checks_total == 0:
        return True
    return checks_passed > 0


def _find_pairs_for_split(
    positions: list[PlayerPosition],
    team_assignments: dict[int, int],
    split_y: float,
    label: str,
) -> list[tuple[_WrongSideInfo, _WrongSideInfo]]:
    """Run wrong-side analysis and pair finding for a given split."""
    wrong_sides = _analyze_wrong_side(positions, team_assignments, split_y)
    if not wrong_sides:
        return []

    logger.debug(
        f"Court-side repair ({label}): {len(wrong_sides)} wrong-side tracks "
        f"({', '.join(f'T{w.track_id}={w.wrong_ratio:.0%}' for w in wrong_sides)})"
    )

    return _find_swap_pairs(wrong_sides)


def repair_cross_team_court_violations(
    positions: list[PlayerPosition],
    primary_track_ids: list[int],
    color_store: ColorHistogramStore | None = None,
    upstream_court_split_y: float | None = None,
    upstream_team_assignments: dict[int, int] | None = None,
) -> tuple[list[PlayerPosition], int]:
    """Detect and repair cross-team ID switches using court-side information.

    Tries two court split strategies and takes the union of detected pairs:
    1. Self-computed split from primary tracks' Y positions (robust to
       upstream court detection failures).
    2. Upstream split from classify_teams (when available; catches cases
       where early-frame clustering fails).

    Args:
        positions: Player positions (modified in place).
        primary_track_ids: The 4 primary player track IDs.
        color_store: Optional color histogram store for appearance validation.
        upstream_court_split_y: Optional upstream court split Y value.
        upstream_team_assignments: Optional upstream team assignments.

    Returns:
        Tuple of (positions, num_repairs). Positions are modified in place.
    """
    # Collect pairs from both split strategies.
    # A track can only appear in one swap to prevent conflicting repairs.
    all_pairs: list[tuple[_WrongSideInfo, _WrongSideInfo]] = []
    used_tracks: set[int] = set()

    def _add_pairs(
        pairs: list[tuple[_WrongSideInfo, _WrongSideInfo]],
    ) -> None:
        for wa, wb in pairs:
            if wa.track_id in used_tracks or wb.track_id in used_tracks:
                continue
            all_pairs.append((wa, wb))
            used_tracks.add(wa.track_id)
            used_tracks.add(wb.track_id)

    # Strategy 1: Self-computed split from track positions
    split_result = _compute_court_split_from_tracks(positions, primary_track_ids)
    if split_result is not None:
        split_y, teams = split_result
        _add_pairs(_find_pairs_for_split(positions, teams, split_y, "self"))

    # Strategy 2: Upstream split (if available)
    if (
        upstream_court_split_y is not None
        and upstream_team_assignments
    ):
        _add_pairs(_find_pairs_for_split(
            positions, upstream_team_assignments,
            upstream_court_split_y, "upstream",
        ))

    if not all_pairs:
        return positions, 0

    # Validate and apply swaps
    num_repairs = 0
    for wa, wb in all_pairs:
        swap_frame = max(wa.transition_frame, wb.transition_frame)

        # Appearance validation (skip if no color store)
        if color_store is not None and color_store.has_data():
            if not _validate_swap_appearance(
                wa.track_id, wb.track_id, swap_frame, color_store
            ):
                logger.debug(
                    f"Court-side repair: skipping T{wa.track_id}<->T{wb.track_id} "
                    f"at frame {swap_frame} — appearance validation failed"
                )
                continue

        # Swap track IDs for all positions at and after swap_frame
        swapped = 0
        for p in positions:
            if p.frame_number >= swap_frame:
                if p.track_id == wa.track_id:
                    p.track_id = wb.track_id
                    swapped += 1
                elif p.track_id == wb.track_id:
                    p.track_id = wa.track_id
                    swapped += 1

        # Keep color_store in sync
        if color_store is not None:
            color_store.swap(wa.track_id, wb.track_id, swap_frame)

        if swapped > 0:
            num_repairs += 1
            logger.info(
                f"Court-side repair: swapped T{wa.track_id}<->T{wb.track_id} "
                f"at frame {swap_frame} ({swapped} positions, "
                f"T{wa.track_id} wrong-side={wa.wrong_ratio:.0%}, "
                f"T{wb.track_id} wrong-side={wb.wrong_ratio:.0%})"
            )

    return positions, num_repairs
