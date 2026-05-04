"""Post-occlusion identity repair using match-level player profiles.

After cross-rally matching builds strong player profiles from 30+ rallies,
this module detects within-rally identity switches by comparing temporal
windows of each track against match-level profiles.

Detection approach: For each primary track, extract appearance from the first
half and second half separately. If the best-matching profile changes between
halves, that indicates a potential switch. Then refine the switch point via
binary search and validate with a confidence gate.

This works because BoT-SORT swaps happen cleanly — the stored positions don't
show bbox overlap at the switch point. The swap is invisible in position data
but visible in appearance when compared against strong match-level profiles.

Pipeline position:
    match-players → repair-identities → remap-track-ids → reattribute-actions
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from rallycut.tracking.player_features import (
    PlayerAppearanceFeatures,
    PlayerAppearanceProfile,
    TrackAppearanceStats,
    compute_appearance_similarity,
    extract_appearance_features,
)
from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Minimum frames in a segment for reliable appearance
MIN_SEGMENT_FRAMES = 15

# Number of appearance samples to extract per segment
NUM_SAMPLES_PER_SEGMENT = 8

# Minimum cost improvement to accept a swap (cross-team)
MIN_IMPROVEMENT_CROSS_TEAM = 0.06

# Minimum cost improvement to accept a swap (same-team — harder)
MIN_IMPROVEMENT_SAME_TEAM = 0.12

# Minimum rally count in profiles to trust them for repair
MIN_PROFILE_RALLY_COUNT = 5

# Number of windows to divide each track into for shift detection
NUM_WINDOWS = 3

# Minimum window size in frames
MIN_WINDOW_FRAMES = 20

# Binary search: minimum segment for refinement
REFINE_MIN_SEGMENT = 10


@dataclass
class SwitchCandidate:
    """A detected appearance shift within a track."""

    track_id: int
    player_id: int  # Current player assignment
    best_profile_first: int  # Best matching profile for first half
    best_profile_second: int  # Best matching profile for second half
    cost_first_current: float  # Cost of first half vs current profile
    cost_second_current: float  # Cost of second half vs current profile
    cost_first_best: float  # Cost of first half vs best profile
    cost_second_best: float  # Cost of second half vs best profile
    approximate_switch_frame: int  # Rough switch point (window boundary)
    refined_switch_frame: int | None = None  # After binary search refinement


@dataclass
class RepairDecision:
    """A validated pair of complementary switches to swap."""

    track_a: int
    track_b: int
    player_a: int  # Current assignment for track_a
    player_b: int  # Current assignment for track_b
    swap_frame: int  # Frame boundary for the swap
    improvement: float
    is_cross_team: bool
    threshold: float
    accepted: bool = False
    is_backward: bool = False  # If True, swap BEFORE swap_frame (not after)


@dataclass
class IdentityRepairResult:
    """Result of identity repair for a rally."""

    rally_id: str
    num_candidates: int = 0
    num_repairs: int = 0
    decisions: list[RepairDecision] = field(default_factory=list)
    candidates: list[SwitchCandidate] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def _is_cross_team(player_a: int, player_b: int) -> bool:
    """Check if two players are on different teams.

    Convention: players 1-2 = team 0 (near), 3-4 = team 1 (far).
    """
    team_a = 0 if player_a <= 2 else 1
    team_b = 0 if player_b <= 2 else 1
    return team_a != team_b


def _extract_segment_appearances(
    cap: cv2.VideoCapture,
    positions: list[PlayerPosition],
    track_id: int,
    frame_range: tuple[int, int],
    start_frame_abs: int,
    frame_width: int,
    frame_height: int,
    num_samples: int = NUM_SAMPLES_PER_SEGMENT,
) -> TrackAppearanceStats | None:
    """Extract appearance features for a track within a frame range.

    Uses an already-opened VideoCapture (seeking per frame).

    Returns:
        TrackAppearanceStats with computed averages, or None if insufficient data.
    """
    track_pos = [
        p for p in positions
        if p.track_id == track_id
        and frame_range[0] <= p.frame_number <= frame_range[1]
    ]
    # Require fewer frames than full segments — windows may be short
    min_window_frames = 5
    if len(track_pos) < min_window_frames:
        return None

    track_pos.sort(key=lambda p: p.frame_number)

    # Sample evenly-spaced frames
    n = len(track_pos)
    if n <= num_samples:
        sample_indices = list(range(n))
    else:
        sample_indices = [
            int(i * (n - 1) / (num_samples - 1)) for i in range(num_samples)
        ]

    stats = TrackAppearanceStats(track_id=track_id)

    for idx in sample_indices:
        p = track_pos[idx]
        abs_frame = start_frame_abs + p.frame_number
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        features = extract_appearance_features(
            np.asarray(frame, dtype=np.uint8),
            track_id, p.frame_number,
            (p.x, p.y, p.width, p.height),
            frame_width, frame_height,
        )
        stats.features.append(features)

    if len(stats.features) < 3:
        return None

    # Filter features with valid histograms (consistent shapes)
    valid_features = [
        f for f in stats.features
        if f.upper_body_hist is not None and f.upper_body_hist.shape == (16, 8)
    ]
    if len(valid_features) < 3:
        return None
    stats.features = valid_features

    stats.compute_averages()
    return stats


def _find_best_profile(
    stats: TrackAppearanceStats,
    profiles: dict[int, PlayerAppearanceProfile],
) -> tuple[int, float]:
    """Find the profile with lowest cost (best match).

    Returns (player_id, cost).
    """
    best_pid = -1
    best_cost = float("inf")
    for pid, profile in profiles.items():
        cost = compute_appearance_similarity(profile, stats)
        if cost < best_cost:
            best_cost = cost
            best_pid = pid
    return best_pid, best_cost


def _refine_switch_frame(
    cap: cv2.VideoCapture,
    positions: list[PlayerPosition],
    track_id: int,
    profile_before: PlayerAppearanceProfile,
    profile_after: PlayerAppearanceProfile,
    search_start: int,
    search_end: int,
    start_frame_abs: int,
    frame_width: int,
    frame_height: int,
) -> int:
    """Binary search to find the exact switch frame.

    Finds the frame where the best-matching profile changes from
    profile_before to profile_after.
    """
    lo, hi = search_start, search_end

    # Get track frames in range
    track_frames = sorted(
        p.frame_number for p in positions
        if p.track_id == track_id and lo <= p.frame_number <= hi
    )
    if len(track_frames) < 4:
        return (lo + hi) // 2

    for _ in range(8):  # Max iterations
        if hi - lo < REFINE_MIN_SEGMENT * 2:
            break

        mid = (lo + hi) // 2

        # Extract appearance for [mid, hi]
        stats_right = _extract_segment_appearances(
            cap, positions, track_id, (mid, hi),
            start_frame_abs, frame_width, frame_height,
            num_samples=6,
        )
        if stats_right is None:
            break

        cost_before = compute_appearance_similarity(profile_before, stats_right)
        cost_after = compute_appearance_similarity(profile_after, stats_right)

        if cost_after < cost_before:
            # Right half matches profile_after → switch is in left half
            hi = mid
        else:
            # Right half still matches profile_before → switch is in right half
            lo = mid

    return (lo + hi) // 2


def repair_rally_identities(
    positions: list[PlayerPosition],
    track_to_player: dict[int, int],
    player_profiles: dict[int, PlayerAppearanceProfile],
    video_path: Path,
    start_ms: int,
    rally_id: str = "",
) -> IdentityRepairResult:
    """Detect and repair identity switches within a rally.

    For each primary track, divides it into temporal windows and compares
    each window against match-level profiles. If the best-matching profile
    shifts mid-track, looks for a complementary shift in another track
    (indicating a mutual swap) and validates with a confidence gate.

    Args:
        positions: All player positions for this rally.
        track_to_player: Mapping from track_id to player_id (1-4).
        player_profiles: Match-level profiles from cross-rally matching.
        video_path: Path to the video file.
        start_ms: Rally start time in milliseconds.
        rally_id: Rally identifier for logging.

    Returns:
        IdentityRepairResult with detected repairs.
    """
    result = IdentityRepairResult(rally_id=rally_id)

    # Validate profiles are strong enough
    min_rally_count = min(
        (p.rally_count for p in player_profiles.values()),
        default=0,
    )
    if min_rally_count < MIN_PROFILE_RALLY_COUNT:
        result.skipped = True
        result.skip_reason = (
            f"Profiles too weak (min rally_count={min_rally_count}, "
            f"need {MIN_PROFILE_RALLY_COUNT})"
        )
        return result

    # Open video once for all extractions
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.skipped = True
        result.skip_reason = f"Could not open video: {video_path}"
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame_abs = int(start_ms / 1000 * fps)

    try:
        result = _detect_and_repair(
            cap, positions, track_to_player, player_profiles,
            start_frame_abs, frame_width, frame_height,
            rally_id,
        )
    finally:
        cap.release()

    return result


def _detect_and_repair(
    cap: cv2.VideoCapture,
    positions: list[PlayerPosition],
    track_to_player: dict[int, int],
    player_profiles: dict[int, PlayerAppearanceProfile],
    start_frame_abs: int,
    frame_width: int,
    frame_height: int,
    rally_id: str,
) -> IdentityRepairResult:
    """Core detection and repair logic (with video already open)."""
    result = IdentityRepairResult(rally_id=rally_id)

    # Group positions by track
    track_positions: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if p.track_id in track_to_player:
            track_positions[p.track_id].append(p)

    # Phase 1: Detect appearance shifts per track
    candidates: list[SwitchCandidate] = []

    for track_id, player_id in track_to_player.items():
        tpos = track_positions.get(track_id, [])
        if len(tpos) < MIN_WINDOW_FRAMES * 2:
            continue

        tpos.sort(key=lambda p: p.frame_number)
        first_frame = tpos[0].frame_number
        last_frame = tpos[-1].frame_number
        total_frames = last_frame - first_frame + 1

        if total_frames < MIN_WINDOW_FRAMES * 2:
            continue

        # Divide into windows
        window_size = total_frames // NUM_WINDOWS
        if window_size < MIN_WINDOW_FRAMES:
            # Fall back to 2 windows (halves)
            mid = first_frame + total_frames // 2
            windows = [(first_frame, mid - 1), (mid, last_frame)]
        else:
            windows = []
            for w in range(NUM_WINDOWS):
                ws = first_frame + w * window_size
                we = ws + window_size - 1 if w < NUM_WINDOWS - 1 else last_frame
                windows.append((ws, we))

        # Extract appearance per window
        window_stats: list[tuple[tuple[int, int], TrackAppearanceStats | None]] = []
        for ws, we in windows:
            stats = _extract_segment_appearances(
                cap, positions, track_id, (ws, we),
                start_frame_abs, frame_width, frame_height,
            )
            window_stats.append(((ws, we), stats))

        # Compare each window against profiles
        window_best: list[tuple[int, float, float]] = []  # (best_pid, best_cost, current_cost)
        current_profile = player_profiles.get(player_id)
        if current_profile is None:
            continue

        for (ws, we), stats in window_stats:
            if stats is None:
                window_best.append((-1, float("inf"), float("inf")))
                continue
            best_pid, best_cost = _find_best_profile(stats, player_profiles)
            current_cost = compute_appearance_similarity(current_profile, stats)
            window_best.append((best_pid, best_cost, current_cost))

        # Check for profile shift between consecutive windows
        for i in range(len(window_best) - 1):
            pid_before, cost_before, ccost_before = window_best[i]
            pid_after, cost_after, ccost_after = window_best[i + 1]

            if pid_before == -1 or pid_after == -1:
                continue
            if pid_before == pid_after:
                continue  # No shift
            if pid_before != player_id and pid_after != player_id:
                continue  # Neither window matches current assignment — unreliable

            # Appearance shift detected
            boundary_frame = windows[i][1]  # End of first window
            candidate = SwitchCandidate(
                track_id=track_id,
                player_id=player_id,
                best_profile_first=pid_before,
                best_profile_second=pid_after,
                cost_first_current=ccost_before,
                cost_second_current=ccost_after,
                cost_first_best=cost_before,
                cost_second_best=cost_after,
                approximate_switch_frame=boundary_frame,
            )
            candidates.append(candidate)
            logger.info(
                f"Rally {rally_id[:8]} track {track_id} (player {player_id}): "
                f"shift at ~frame {boundary_frame} "
                f"profile {pid_before}→{pid_after} "
                f"(current cost {ccost_before:.3f}→{ccost_after:.3f})"
            )

    result.num_candidates = len(candidates)
    result.candidates = candidates

    if not candidates:
        return result

    # Phase 2: Validate each shift and decide whether to swap
    #
    # For each candidate with a clear profile shift:
    #   1. Identify the "wrong" portion (where track matches another player's profile)
    #   2. Find the partner track (the one assigned to the shifted-to profile)
    #   3. Verify the partner track's appearance in the wrong portion also supports swap
    #   4. Verify swapping doesn't make either track worse overall
    #
    # Two directions:
    #   Forward: first=current, second=other → wrong portion is AFTER shift
    #   Backward: first=other, second=current → wrong portion is BEFORE shift
    used_tracks: set[int] = set()

    # Build player→track reverse map
    player_to_track_map = {v: k for k, v in track_to_player.items()}

    for ca in candidates:
        if ca.track_id in used_tracks:
            continue

        # Determine direction
        is_forward = ca.best_profile_first == ca.player_id
        is_backward = ca.best_profile_second == ca.player_id

        if not is_forward and not is_backward:
            continue  # Neither half matches current profile — unreliable

        # The "other" player this track is shifting to/from
        other_player = (
            ca.best_profile_second if is_forward else ca.best_profile_first
        )
        partner_track = player_to_track_map.get(other_player)
        if partner_track is None or partner_track in used_tracks:
            continue

        approx_frame = ca.approximate_switch_frame

        # Compute improvement for the shifting track
        if is_forward:
            # Second half is wrong — improvement from reassigning it
            shift_improvement = ca.cost_second_current - ca.cost_second_best
        else:
            # First half is wrong — improvement from reassigning it
            shift_improvement = ca.cost_first_current - ca.cost_first_best

        # Verify partner track: extract its appearance in the "wrong" portion
        # and check if it matches the shifting track's current profile
        tpos = sorted(track_positions.get(ca.track_id, []),
                      key=lambda p: p.frame_number)
        if not tpos:
            continue
        first_frame = tpos[0].frame_number
        last_frame = tpos[-1].frame_number

        if is_forward:
            wrong_range = (approx_frame, last_frame)
        else:
            wrong_range = (first_frame, approx_frame)

        partner_wrong = _extract_segment_appearances(
            cap, positions, partner_track, wrong_range,
            start_frame_abs, frame_width, frame_height,
        )
        if partner_wrong is None:
            continue

        # Check: does the partner match the shifting track's profile better
        # in the wrong portion? (i.e., the partner also swapped)
        partner_current_profile = player_profiles.get(other_player)
        shifting_profile = player_profiles.get(ca.player_id)
        if partner_current_profile is None or shifting_profile is None:
            continue

        partner_cost_current = compute_appearance_similarity(
            partner_current_profile, partner_wrong,
        )
        partner_cost_swapped = compute_appearance_similarity(
            shifting_profile, partner_wrong,
        )
        partner_improvement = partner_cost_current - partner_cost_swapped

        total_improvement = shift_improvement + partner_improvement

        # Refine switch frame
        if is_forward:
            profile_orig = player_profiles.get(ca.player_id)
            profile_new = player_profiles.get(other_player)
        else:
            profile_orig = player_profiles.get(other_player)
            profile_new = player_profiles.get(ca.player_id)

        refined_frame = approx_frame
        if profile_orig and profile_new:
            search_start = max(0, approx_frame - MIN_WINDOW_FRAMES * 2)
            search_end = min(
                approx_frame + MIN_WINDOW_FRAMES * 2,
                max(p.frame_number for p in positions),
            )
            refined_frame = _refine_switch_frame(
                cap, positions, ca.track_id,
                profile_orig, profile_new,
                search_start, search_end,
                start_frame_abs, frame_width, frame_height,
            )
            ca.refined_switch_frame = refined_frame

        cross_team = _is_cross_team(ca.player_id, other_player)
        threshold = (
            MIN_IMPROVEMENT_CROSS_TEAM if cross_team
            else MIN_IMPROVEMENT_SAME_TEAM
        )

        decision = RepairDecision(
            track_a=ca.track_id,
            track_b=partner_track,
            player_a=ca.player_id,
            player_b=other_player,
            swap_frame=refined_frame,
            improvement=total_improvement,
            is_cross_team=cross_team,
            threshold=threshold,
            is_backward=is_backward,
        )

        # Accept if: total improvement exceeds threshold AND both tracks
        # individually improve (partner_improvement > 0)
        if (
            total_improvement >= threshold
            and shift_improvement > 0
            and partner_improvement > 0
        ):
            decision.accepted = True
            result.num_repairs += 1
            used_tracks.add(ca.track_id)
            used_tracks.add(partner_track)

        result.decisions.append(decision)

        logger.info(
            f"Rally {rally_id[:8]} track {ca.track_id} "
            f"(p{ca.player_id}→p{other_player}) partner={partner_track}: "
            f"{'backward' if is_backward else 'forward'} "
            f"shift={shift_improvement:.3f} partner={partner_improvement:.3f} "
            f"total={total_improvement:.3f} threshold={threshold:.3f} "
            f"cross_team={cross_team} → "
            f"{'SWAP at frame ' + str(refined_frame) if decision.accepted else 'SKIP'}"
        )

    return result


@dataclass
class RallyRepairInput:
    """Input data for one rally in a batch repair run."""

    rally_id: str
    start_ms: int
    positions: list[PlayerPosition]
    track_to_player: dict[int, int]


@dataclass
class _FrameRequest:
    """A frame to extract during the sequential video pass."""

    abs_frame: int
    rally_id: str
    track_id: int
    position: PlayerPosition


def _plan_segment_samples(
    positions: list[PlayerPosition],
    track_id: int,
    frame_range: tuple[int, int],
    start_frame_abs: int,
    num_samples: int = NUM_SAMPLES_PER_SEGMENT,
) -> list[tuple[int, PlayerPosition]]:
    """Determine which frames to sample for a segment (no video I/O).

    Returns list of (abs_frame, position) pairs.
    """
    track_pos = [
        p for p in positions
        if p.track_id == track_id
        and frame_range[0] <= p.frame_number <= frame_range[1]
    ]
    if len(track_pos) < 5:
        return []

    track_pos.sort(key=lambda p: p.frame_number)
    n = len(track_pos)
    if n <= num_samples:
        sample_indices = list(range(n))
    else:
        sample_indices = [
            int(i * (n - 1) / (num_samples - 1)) for i in range(num_samples)
        ]

    return [
        (start_frame_abs + track_pos[idx].frame_number, track_pos[idx])
        for idx in sample_indices
    ]


def _build_stats_from_features(
    track_id: int,
    features: list[PlayerAppearanceFeatures],
) -> TrackAppearanceStats | None:
    """Build TrackAppearanceStats from pre-extracted features."""
    valid = [
        f for f in features
        if f.upper_body_hist is not None and f.upper_body_hist.shape == (16, 8)
    ]
    if len(valid) < 3:
        return None
    stats = TrackAppearanceStats(track_id=track_id)
    stats.features = valid
    stats.compute_averages()
    return stats


def _plan_rally_windows(
    positions: list[PlayerPosition],
    track_to_player: dict[int, int],
) -> dict[int, list[tuple[int, int]]]:
    """Compute temporal windows for each track (no video I/O).

    Returns: {track_id: [(window_start, window_end), ...]}
    """
    track_positions: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if p.track_id in track_to_player:
            track_positions[p.track_id].append(p)

    track_windows: dict[int, list[tuple[int, int]]] = {}
    for track_id in track_to_player:
        tpos = track_positions.get(track_id, [])
        if len(tpos) < MIN_WINDOW_FRAMES * 2:
            continue
        tpos.sort(key=lambda p: p.frame_number)
        first_frame = tpos[0].frame_number
        last_frame = tpos[-1].frame_number
        total_frames = last_frame - first_frame + 1
        if total_frames < MIN_WINDOW_FRAMES * 2:
            continue

        window_size = total_frames // NUM_WINDOWS
        if window_size < MIN_WINDOW_FRAMES:
            mid = first_frame + total_frames // 2
            windows = [(first_frame, mid - 1), (mid, last_frame)]
        else:
            windows = []
            for w in range(NUM_WINDOWS):
                ws = first_frame + w * window_size
                we = ws + window_size - 1 if w < NUM_WINDOWS - 1 else last_frame
                windows.append((ws, we))
        track_windows[track_id] = windows

    return track_windows


def _detect_shifts(
    track_to_player: dict[int, int],
    track_windows: dict[int, list[tuple[int, int]]],
    window_stats: dict[tuple[int, tuple[int, int]], TrackAppearanceStats | None],
    player_profiles: dict[int, PlayerAppearanceProfile],
    rally_id: str,
) -> list[SwitchCandidate]:
    """Detect appearance shifts from pre-computed window stats (no video I/O)."""
    candidates: list[SwitchCandidate] = []

    for track_id, player_id in track_to_player.items():
        windows = track_windows.get(track_id)
        if not windows:
            continue

        current_profile = player_profiles.get(player_id)
        if current_profile is None:
            continue

        window_best: list[tuple[int, float, float]] = []
        for ws, we in windows:
            stats = window_stats.get((track_id, (ws, we)))
            if stats is None:
                window_best.append((-1, float("inf"), float("inf")))
                continue
            best_pid, best_cost = _find_best_profile(stats, player_profiles)
            current_cost = compute_appearance_similarity(current_profile, stats)
            window_best.append((best_pid, best_cost, current_cost))

        for i in range(len(window_best) - 1):
            pid_before, cost_before, ccost_before = window_best[i]
            pid_after, cost_after, ccost_after = window_best[i + 1]

            if pid_before == -1 or pid_after == -1:
                continue
            if pid_before == pid_after:
                continue
            if pid_before != player_id and pid_after != player_id:
                continue

            boundary_frame = windows[i][1]
            candidates.append(SwitchCandidate(
                track_id=track_id,
                player_id=player_id,
                best_profile_first=pid_before,
                best_profile_second=pid_after,
                cost_first_current=ccost_before,
                cost_second_current=ccost_after,
                cost_first_best=cost_before,
                cost_second_best=cost_after,
                approximate_switch_frame=boundary_frame,
            ))
            logger.info(
                f"Rally {rally_id[:8]} track {track_id} (player {player_id}): "
                f"shift at ~frame {boundary_frame} "
                f"profile {pid_before}→{pid_after} "
                f"(current cost {ccost_before:.3f}→{ccost_after:.3f})"
            )

    return candidates


def repair_all_rallies(
    video_path: Path,
    rallies: list[RallyRepairInput],
    player_profiles: dict[int, PlayerAppearanceProfile],
) -> list[IdentityRepairResult]:
    """Batch repair all rallies with a single sequential video pass.

    Opens the video once and reads frames sequentially, extracting
    appearance features for all rallies in one pass.  This avoids the
    O(rallies × tracks × windows × samples) random seeks that make
    per-rally repair slow on compressed video.
    """
    # Early validation
    min_rally_count = min(
        (p.rally_count for p in player_profiles.values()), default=0,
    )
    if min_rally_count < MIN_PROFILE_RALLY_COUNT:
        return [
            IdentityRepairResult(
                rally_id=r.rally_id, skipped=True,
                skip_reason=f"Profiles too weak (min rally_count={min_rally_count})",
            )
            for r in rallies
        ]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [
            IdentityRepairResult(
                rally_id=r.rally_id, skipped=True,
                skip_reason=f"Could not open video: {video_path}",
            )
            for r in rallies
        ]

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        return _batch_detect_and_repair(
            cap, rallies, player_profiles,
            fps, frame_width, frame_height,
        )
    finally:
        cap.release()


def _batch_detect_and_repair(
    cap: cv2.VideoCapture,
    rallies: list[RallyRepairInput],
    player_profiles: dict[int, PlayerAppearanceProfile],
    fps: float,
    frame_width: int,
    frame_height: int,
) -> list[IdentityRepairResult]:
    """Core batch detection: plan → sequential extract → analyse."""
    # Phase 1: Plan all frame extractions across all rallies
    requests: list[_FrameRequest] = []
    rally_windows: dict[str, dict[int, list[tuple[int, int]]]] = {}

    for rally in rallies:
        start_frame_abs = int(rally.start_ms / 1000 * fps)
        windows = _plan_rally_windows(rally.positions, rally.track_to_player)
        rally_windows[rally.rally_id] = windows

        for track_id, track_wins in windows.items():
            for ws, we in track_wins:
                samples = _plan_segment_samples(
                    rally.positions, track_id, (ws, we), start_frame_abs,
                )
                for abs_frame, pos in samples:
                    requests.append(_FrameRequest(
                        abs_frame=abs_frame,
                        rally_id=rally.rally_id,
                        track_id=track_id,
                        position=pos,
                    ))

    if not requests:
        return [IdentityRepairResult(rally_id=r.rally_id) for r in rallies]

    # Phase 2: Sequential video pass — read frames in order
    requests.sort(key=lambda r: r.abs_frame)

    # Group requests by abs_frame so we read each frame at most once
    from itertools import groupby

    # Key: (rally_id, track_id, window) → list of features
    features_by_key: dict[str, dict[int, list[PlayerAppearanceFeatures]]] = defaultdict(
        lambda: defaultdict(list)
    )

    last_frame_num = -1
    last_frame_data: np.ndarray | None = None

    for abs_frame, group in groupby(requests, key=lambda r: r.abs_frame):
        group_list = list(group)

        # Read frame — sequential when possible, seek only when gap
        if abs_frame == last_frame_num and last_frame_data is not None:
            frame_data = last_frame_data
        elif abs_frame == last_frame_num + 1:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_data = np.asarray(frame, dtype=np.uint8)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_data = np.asarray(frame, dtype=np.uint8)

        last_frame_num = abs_frame
        last_frame_data = frame_data

        # Extract features for all requests on this frame
        for req in group_list:
            p = req.position
            feat = extract_appearance_features(
                frame_data, req.track_id, p.frame_number,
                (p.x, p.y, p.width, p.height),
                frame_width, frame_height,
            )
            features_by_key[req.rally_id][req.track_id].append(feat)

    # Phase 3: Build stats and detect shifts per rally
    results: list[IdentityRepairResult] = []

    for rally in rallies:
        windows = rally_windows.get(rally.rally_id, {})
        if not windows:
            results.append(IdentityRepairResult(rally_id=rally.rally_id))
            continue

        start_frame_abs = int(rally.start_ms / 1000 * fps)
        rally_features = features_by_key.get(rally.rally_id, {})

        # Build window stats from sequential features
        # Features were appended in frame order, so we can split by window
        window_stats: dict[tuple[int, tuple[int, int]], TrackAppearanceStats | None] = {}
        for track_id, track_wins in windows.items():
            track_feats = rally_features.get(track_id, [])
            # Features are in frame order; split into windows by frame range
            feat_idx = 0
            for ws, we in track_wins:
                abs_ws = start_frame_abs + ws
                abs_we = start_frame_abs + we
                segment_feats: list[PlayerAppearanceFeatures] = []
                while feat_idx < len(track_feats):
                    f = track_feats[feat_idx]
                    f_abs = start_frame_abs + f.frame_number
                    if f_abs > abs_we:
                        break
                    if f_abs >= abs_ws:
                        segment_feats.append(f)
                    feat_idx += 1
                window_stats[(track_id, (ws, we))] = _build_stats_from_features(
                    track_id, segment_feats,
                )

        # Detect shifts
        candidates = _detect_shifts(
            rally.track_to_player, windows, window_stats,
            player_profiles, rally.rally_id,
        )

        result = IdentityRepairResult(
            rally_id=rally.rally_id,
            num_candidates=len(candidates),
            candidates=candidates,
        )

        if not candidates:
            results.append(result)
            continue

        # Phase 2b: Validate candidates (may need additional frame reads)
        # For the rare case where candidates are found, fall back to
        # per-candidate seeking — this path is almost never hit (0 repairs
        # across 207 rallies in validation).
        result = _validate_candidates_and_repair(
            cap, rally, result, player_profiles,
            start_frame_abs, frame_width, frame_height,
        )
        results.append(result)

    return results


def _validate_candidates_and_repair(
    cap: cv2.VideoCapture,
    rally: RallyRepairInput,
    result: IdentityRepairResult,
    player_profiles: dict[int, PlayerAppearanceProfile],
    start_frame_abs: int,
    frame_width: int,
    frame_height: int,
) -> IdentityRepairResult:
    """Validate switch candidates and decide repairs (rare path).

    Uses per-candidate video seeking — acceptable since this path is
    hit only when appearance shifts are detected (< 5% of rallies).
    """
    track_positions: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in rally.positions:
        if p.track_id in rally.track_to_player:
            track_positions[p.track_id].append(p)

    used_tracks: set[int] = set()
    player_to_track = {v: k for k, v in rally.track_to_player.items()}

    for ca in result.candidates:
        if ca.track_id in used_tracks:
            continue

        is_forward = ca.best_profile_first == ca.player_id
        is_backward = ca.best_profile_second == ca.player_id
        if not is_forward and not is_backward:
            continue

        other_player = (
            ca.best_profile_second if is_forward else ca.best_profile_first
        )
        partner_track = player_to_track.get(other_player)
        if partner_track is None or partner_track in used_tracks:
            continue

        approx_frame = ca.approximate_switch_frame
        if is_forward:
            shift_improvement = ca.cost_second_current - ca.cost_second_best
        else:
            shift_improvement = ca.cost_first_current - ca.cost_first_best

        tpos = sorted(
            track_positions.get(ca.track_id, []),
            key=lambda p: p.frame_number,
        )
        if not tpos:
            continue
        first_frame = tpos[0].frame_number
        last_frame = tpos[-1].frame_number

        wrong_range = (
            (approx_frame, last_frame) if is_forward
            else (first_frame, approx_frame)
        )

        partner_wrong = _extract_segment_appearances(
            cap, rally.positions, partner_track, wrong_range,
            start_frame_abs, frame_width, frame_height,
        )
        if partner_wrong is None:
            continue

        partner_current_profile = player_profiles.get(other_player)
        shifting_profile = player_profiles.get(ca.player_id)
        if partner_current_profile is None or shifting_profile is None:
            continue

        partner_cost_current = compute_appearance_similarity(
            partner_current_profile, partner_wrong,
        )
        partner_cost_swapped = compute_appearance_similarity(
            shifting_profile, partner_wrong,
        )
        partner_improvement = partner_cost_current - partner_cost_swapped
        total_improvement = shift_improvement + partner_improvement

        if is_forward:
            profile_orig = player_profiles.get(ca.player_id)
            profile_new = player_profiles.get(other_player)
        else:
            profile_orig = player_profiles.get(other_player)
            profile_new = player_profiles.get(ca.player_id)

        refined_frame = approx_frame
        if profile_orig and profile_new:
            search_start = max(0, approx_frame - MIN_WINDOW_FRAMES * 2)
            search_end = min(
                approx_frame + MIN_WINDOW_FRAMES * 2,
                max(p.frame_number for p in rally.positions),
            )
            refined_frame = _refine_switch_frame(
                cap, rally.positions, ca.track_id,
                profile_orig, profile_new,
                search_start, search_end,
                start_frame_abs, frame_width, frame_height,
            )
            ca.refined_switch_frame = refined_frame

        cross_team = _is_cross_team(ca.player_id, other_player)
        threshold = (
            MIN_IMPROVEMENT_CROSS_TEAM if cross_team
            else MIN_IMPROVEMENT_SAME_TEAM
        )

        decision = RepairDecision(
            track_a=ca.track_id,
            track_b=partner_track,
            player_a=ca.player_id,
            player_b=other_player,
            swap_frame=refined_frame,
            improvement=total_improvement,
            is_cross_team=cross_team,
            threshold=threshold,
            is_backward=is_backward,
        )

        if (
            total_improvement >= threshold
            and shift_improvement > 0
            and partner_improvement > 0
        ):
            decision.accepted = True
            result.num_repairs += 1
            used_tracks.add(ca.track_id)
            used_tracks.add(partner_track)

        result.decisions.append(decision)

        logger.info(
            f"Rally {rally.rally_id[:8]} track {ca.track_id} "
            f"(p{ca.player_id}→p{other_player}) partner={partner_track}: "
            f"{'backward' if is_backward else 'forward'} "
            f"shift={shift_improvement:.3f} partner={partner_improvement:.3f} "
            f"total={total_improvement:.3f} threshold={threshold:.3f} "
            f"cross_team={cross_team} → "
            f"{'SWAP at frame ' + str(refined_frame) if decision.accepted else 'SKIP'}"
        )

    return result


def apply_repairs(
    positions: list[PlayerPosition],
    decisions: list[RepairDecision],
) -> int:
    """Apply accepted repairs by swapping track IDs after the switch frame.

    For each accepted swap, all positions for track_a after the swap frame
    get track_b's ID and vice versa.

    Args:
        positions: Mutable list of positions to modify in place.
        decisions: Repair decisions (only accepted ones are applied).

    Returns:
        Number of position entries modified.
    """
    accepted = [d for d in decisions if d.accepted]
    if not accepted:
        return 0

    # Sort by swap frame for sequential application
    accepted.sort(key=lambda d: d.swap_frame)

    total_modified = 0
    for decision in accepted:
        swap_frame = decision.swap_frame
        ta = decision.track_a
        tb = decision.track_b

        for p in positions:
            if decision.is_backward:
                # Swap positions BEFORE the swap frame (fix early portion)
                if p.frame_number < swap_frame:
                    if p.track_id == ta:
                        p.track_id = tb
                        total_modified += 1
                    elif p.track_id == tb:
                        p.track_id = ta
                        total_modified += 1
            else:
                # Swap positions AFTER the swap frame (fix late portion)
                if p.frame_number >= swap_frame:
                    if p.track_id == ta:
                        p.track_id = tb
                        total_modified += 1
                    elif p.track_id == tb:
                        p.track_id = ta
                        total_modified += 1

    return total_modified
