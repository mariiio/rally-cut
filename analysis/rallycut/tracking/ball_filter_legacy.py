"""
Legacy ball tracking filter stages, disabled for WASB.

These stages were part of the original ball filter pipeline but are disabled
in the WASB configuration (get_wasb_filter_config). They are kept here for
backward compatibility and potential use with other detectors.

Stages:
- Motion energy filter: removes false positives at stationary positions
- Stationarity filter: removes player lock-on runs (sliding window)
- Oscillation pruning: trims cluster-based A-B-A-B tails
- Blip removal: removes multi-frame trajectory blips
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.ball_filter import BallFilterConfig
    from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)


def motion_energy_filter(
    positions: list["BallPosition"],
    config: "BallFilterConfig",
) -> list["BallPosition"]:
    """Remove false positives with low motion energy.

    A real ball in flight creates high temporal intensity change at its
    position (it wasn't there before, or it leaves after). False positive
    detections at player/court positions have low motion energy because
    those regions are relatively static.

    Zeroes the confidence of positions where motion_energy is below threshold.

    Early-exits if all motion_energy values are 0 (e.g. WASB hardcodes
    motion_energy=0.0), avoiding a wasted iteration.
    """
    from rallycut.tracking.ball_tracker import BallPosition

    # Early exit: if all motion_energy values are 0, skip entirely.
    # WASB hardcodes motion_energy=0.0, so this prevents wasted computation.
    if all(p.motion_energy == 0.0 for p in positions if p.confidence > 0):
        return positions

    threshold = config.motion_energy_threshold
    removed = 0
    result = []

    for p in positions:
        if p.confidence > 0 and p.motion_energy > 0 and p.motion_energy < threshold:
            # Low motion energy = likely false positive at static position
            result.append(BallPosition(
                frame_number=p.frame_number,
                x=p.x,
                y=p.y,
                confidence=0.0,
                motion_energy=p.motion_energy,
            ))
            removed += 1
        else:
            result.append(p)

    if removed > 0:
        logger.info(f"Motion energy filter: zeroed {removed} low-energy positions")

    return result


def remove_stationary_runs(
    positions: list["BallPosition"],
    config: "BallFilterConfig",
) -> list["BallPosition"]:
    """Remove stationary sub-sequences using a sliding window.

    A volleyball is always in motion during a rally. If 12+ consecutive
    confident detections cluster within 0.5% of screen, it's a model
    locked onto a player, not a real ball trajectory. This is source-agnostic:
    it applies to any ball detector because physics overrides
    model trust.

    Uses a sliding window to detect stationary blocks even when they are
    interleaved with real detections in the same contiguous run. WASB can
    intermittently snap back to a player position (e.g., 25 bad frames,
    then 50 good, then 16 bad at the same spot). The window approach
    catches each stationary block independently.

    Tight thresholds (0.5% spread, 12 frames) avoid false positives on
    real slow-ball events: ball apex during sets/tosses (~0.8% spread),
    slow trajectories near the net (~1.5%). True player lock-on has
    spread < 0.2% (random sub-pixel jitter only).
    """
    from rallycut.tracking.ball_tracker import BallPosition

    min_frames = config.stationarity_min_frames
    max_spread = config.stationarity_max_spread

    # Collect confident positions with their indices into positions list
    confident_indices = [
        i for i, p in enumerate(positions) if p.confidence > 0
    ]
    if len(confident_indices) < min_frames:
        return positions

    # Sliding window: check each window of min_frames consecutive
    # confident positions for stationarity
    stationary_indices: set[int] = set()
    n = len(confident_indices)
    # Net displacement threshold: truly stationary positions have near-zero
    # net displacement (random jitter), while slow-moving trajectories show
    # consistent directional movement (net displacement > threshold).
    net_disp_threshold = max_spread / 2

    for start in range(n - min_frames + 1):
        window = confident_indices[start : start + min_frames]
        window_positions = [positions[i] for i in window]

        # Require temporal contiguity: frame gap between first and last
        # position in window must be reasonable (not spanning a huge gap)
        frame_span = (
            window_positions[-1].frame_number
            - window_positions[0].frame_number
        )
        if frame_span > min_frames * 3:
            continue

        cx = float(np.mean([p.x for p in window_positions]))
        cy = float(np.mean([p.y for p in window_positions]))
        spread = float(max(
            np.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2)
            for p in window_positions
        ))

        if spread >= max_spread:
            continue

        # Check net displacement: distance from first to last position.
        # Stationary lock-on has near-zero net displacement (jitter),
        # while slow-moving real trajectories have measurable displacement.
        first, last = window_positions[0], window_positions[-1]
        net_disp = float(np.sqrt(
            (last.x - first.x) ** 2 + (last.y - first.y) ** 2
        ))
        if net_disp > net_disp_threshold:
            continue

        for i in window:
            stationary_indices.add(i)

    if not stationary_indices:
        return positions

    # Log flagged regions grouped into contiguous runs
    flagged_sorted = sorted(stationary_indices)
    runs: list[list[int]] = [[flagged_sorted[0]]]
    for idx in flagged_sorted[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    for run in runs:
        run_positions = [positions[i] for i in run]
        logger.info(
            f"Stationarity filter: zeroed {len(run)} frames "
            f"[{run_positions[0].frame_number}-"
            f"{run_positions[-1].frame_number}] "
            f"(spread<{max_spread})"
        )

    result = []
    for i, p in enumerate(positions):
        if i in stationary_indices:
            result.append(BallPosition(
                frame_number=p.frame_number,
                x=p.x,
                y=p.y,
                confidence=0.0,
                motion_energy=p.motion_energy,
            ))
        else:
            result.append(p)

    return result


def prune_oscillating(
    positions: list["BallPosition"],
    config: "BallFilterConfig",
) -> list["BallPosition"]:
    """Trim sustained oscillation from trajectory tails using cluster detection.

    The ball detector can lock onto two players and alternate between them
    with high confidence after the ball exits the frame. The pattern is
    cluster-based: positions stay near player B for 2-5 frames, jump to
    player A for 1-2 frames, then back. Per-frame displacement is tiny
    within each cluster, so displacement-reversal detection misses this
    pattern entirely.

    Also detects single-cluster hovering: after a large gap (ball exited
    frame), the detector can lock onto a single player position and produce
    many frames within a tiny radius. Detected by checking short segments
    (<=3x window) after a large gap: if the first window positions all lie
    within segment_jump_threshold/4 of their centroid, the segment is dropped.

    Algorithm (cluster transition detection):
    1. Split into contiguous segments (gap > 5 frames)
    2. For each segment after a large gap, check for hovering (all
       positions within a small radius of centroid -> drop entire segment)
    3. For each remaining segment, slide a window across positions:
       a. Find two poles: the pair of positions with maximum distance
       b. If pole distance < min_displacement: skip (jitter, not oscillation)
       c. Assign each position to nearest pole (binary cluster label)
       d. Count transitions (cluster[i] != cluster[i+1])
       e. If transition_rate >= threshold: trim from window start onward
    """
    if len(positions) < config.min_oscillation_frames + 2:
        return positions

    min_pole_dist = config.oscillation_min_displacement
    window = config.min_oscillation_frames
    rate_threshold = config.oscillation_reversal_rate

    # Step 1: Split into contiguous segments (gap > 5 frames)
    segments: list[list["BallPosition"]] = [[positions[0]]]
    for i in range(1, len(positions)):
        if positions[i].frame_number - positions[i - 1].frame_number > 5:
            segments.append([positions[i]])
        else:
            segments[-1].append(positions[i])

    result: list["BallPosition"] = []
    max_gap = config.max_interpolation_gap
    hover_radius = config.segment_jump_threshold / 4
    prev_end_frame: int | None = None

    for seg in segments:
        # Hovering detection: single-player lock-on after ball exits frame.
        # If segment follows a large gap and all positions cluster within
        # a tiny radius, it's the detector locked onto a stationary player.
        # Only flag short segments -- long ones that start slow are likely
        # real (ball gradually accelerating after serve/bounce).
        gap = (
            seg[0].frame_number - prev_end_frame
            if prev_end_frame is not None
            else 0
        )
        if gap > max_gap and window <= len(seg) <= window * 3:
            first_w = seg[:window]
            cx = float(np.mean([p.x for p in first_w]))
            cy = float(np.mean([p.y for p in first_w]))
            max_spread = float(
                max(
                    np.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2)
                    for p in first_w
                )
            )
            if max_spread < hover_radius:
                frame_range = (
                    f"[{seg[0].frame_number}-{seg[-1].frame_number}]"
                )
                logger.info(
                    f"Oscillation pruning: dropped hovering segment "
                    f"{frame_range} ({len(seg)} frames, "
                    f"spread={max_spread:.4f}, gap={gap})"
                )
                prev_end_frame = seg[-1].frame_number
                continue

        if len(seg) < window + 2:
            result.extend(seg)
            prev_end_frame = seg[-1].frame_number
            continue

        n = len(seg)
        trim_idx = None  # Index into seg where trimming starts

        # Slide a window across the segment
        for start in range(n - window + 1):
            w = seg[start : start + window]

            # Find two poles: pair with maximum distance
            max_dist = 0.0
            pole_a = (w[0].x, w[0].y)
            pole_b = (w[0].x, w[0].y)
            for i in range(len(w)):
                for j in range(i + 1, len(w)):
                    d = np.sqrt(
                        (w[i].x - w[j].x) ** 2 + (w[i].y - w[j].y) ** 2
                    )
                    if d > max_dist:
                        max_dist = d
                        pole_a = (w[i].x, w[i].y)
                        pole_b = (w[j].x, w[j].y)

            # Skip if poles are too close (jitter, not oscillation)
            if max_dist < min_pole_dist:
                continue

            # Assign each position to nearest pole, tracking distance
            labels = []
            dists_a: list[float] = []  # distances to pole_a for cluster 0
            dists_b: list[float] = []  # distances to pole_b for cluster 1
            for p in w:
                da = np.sqrt(
                    (p.x - pole_a[0]) ** 2 + (p.y - pole_a[1]) ** 2
                )
                db = np.sqrt(
                    (p.x - pole_b[0]) ** 2 + (p.y - pole_b[1]) ** 2
                )
                if da <= db:
                    labels.append(0)
                    dists_a.append(da)
                else:
                    labels.append(1)
                    dists_b.append(db)

            # Both clusters must have at least 3 positions to be oscillation.
            # A single spike + geometric artifacts won't reach 3.
            if len(dists_a) < 3 or len(dists_b) < 3:
                continue

            # Clusters must be compact: in real oscillation, the detector locks
            # onto fixed player positions so within-cluster spread is tiny
            # (<1% of screen). In a ball bounce passing through the midpoint,
            # cluster members span a wide trajectory arc. Use half the pole
            # distance as the max allowed spread per cluster.
            max_cluster_spread = min_pole_dist / 2
            if (
                max(dists_a) > max_cluster_spread
                or max(dists_b) > max_cluster_spread
            ):
                continue

            # Count transitions between clusters
            transitions = sum(
                1 for k in range(len(labels) - 1) if labels[k] != labels[k + 1]
            )
            rate = transitions / (len(labels) - 1)

            if rate >= rate_threshold:
                trim_idx = start
                break

        if trim_idx is not None:
            trimmed = len(seg) - trim_idx
            result.extend(seg[:trim_idx])
            frame_range = (
                f"[{seg[trim_idx].frame_number}-{seg[-1].frame_number}]"
            )
            logger.info(
                f"Oscillation pruning: trimmed {trimmed} frames {frame_range} "
                f"from segment [{seg[0].frame_number}-{seg[-1].frame_number}]"
            )
        else:
            result.extend(seg)

        prev_end_frame = seg[-1].frame_number

    return result


def _find_blip_context(
    idx: int,
    frames: list[int],
    pos_by_frame: dict[int, "BallPosition"],
    min_dist: int,
    exclude: set[int],
) -> tuple["BallPosition | None", int, "BallPosition | None", int]:
    """Find distant context positions for blip detection.

    Searches backward and forward from idx for the nearest position that is
    at least min_dist frame numbers away, skipping any indices in exclude.

    Returns:
        (prev_ctx, prev_gap, next_ctx, next_gap) where:
        - prev_ctx: nearest previous context position (or None)
        - prev_gap: frame distance to prev_ctx
        - next_ctx: nearest next context position (or None)
        - next_gap: frame distance to next_ctx
    """
    current_frame = frames[idx]
    prev_ctx = None
    prev_gap = 0
    next_ctx = None
    next_gap = 0

    # Search backward for previous context
    for j in range(idx - 1, -1, -1):
        if j in exclude:
            continue
        if current_frame - frames[j] >= min_dist:
            prev_ctx = pos_by_frame[frames[j]]
            prev_gap = current_frame - frames[j]
            break

    # Search forward for next context
    for j in range(idx + 1, len(frames)):
        if j in exclude:
            continue
        if frames[j] - current_frame >= min_dist:
            next_ctx = pos_by_frame[frames[j]]
            next_gap = frames[j] - current_frame
            break

    return prev_ctx, prev_gap, next_ctx, next_gap


def _blip_deviation(
    pos: "BallPosition",
    prev_ctx: "BallPosition",
    next_ctx: "BallPosition",
    prev_gap: int,
    next_gap: int,
) -> float:
    """Compute deviation of a position from the line between two context positions.

    Linearly interpolates between prev_ctx and next_ctx based on frame
    distance ratios, then returns the Euclidean distance from pos to the
    interpolated point.
    """
    total_gap = prev_gap + next_gap
    t = prev_gap / total_gap
    expected_x = prev_ctx.x + t * (next_ctx.x - prev_ctx.x)
    expected_y = prev_ctx.y + t * (next_ctx.y - prev_ctx.y)
    return float(np.sqrt((pos.x - expected_x) ** 2 + (pos.y - expected_y) ** 2))


def remove_trajectory_blips(
    positions: list["BallPosition"],
    config: "BallFilterConfig",
) -> list["BallPosition"]:
    """Remove multi-frame trajectory blips using distant context.

    The ball detector can briefly lock onto a player position for 2-5
    consecutive frames mid-trajectory. Single-frame outlier detection misses
    these because the consecutive false positives validate each other as
    neighbors.

    Two-phase approach to avoid false positives on real bounces:
    1. Flag positions that deviate from distant trajectory context
    2. Only remove CLUSTERS of >=2 consecutive flagged frames with compact
       internal spread -- real bounces have spread, blips are tightly clustered
    """
    if len(positions) < 3:
        return positions

    min_dist = config.blip_context_min_frames
    max_dev = config.blip_max_deviation
    max_ctx_gap = config.blip_max_context_gap
    # Blip cluster must be spatially compact (within 5% of screen).
    # Real bounces spread along a curve; detector player-locking blips
    # cluster tightly at a fixed position (~1% noise).
    max_blip_spread = 0.05

    pos_by_frame = {p.frame_number: p for p in positions}
    frames = sorted(pos_by_frame.keys())

    # Phase 1a: Flag suspect positions deviating from distant context
    suspect_indices: set[int] = set()

    for i, frame in enumerate(frames):
        prev_ctx, prev_gap, next_ctx, next_gap = _find_blip_context(
            i, frames, pos_by_frame, min_dist, exclude=set()
        )
        # Skip if context is too far -- linear interpolation becomes unreliable
        if not prev_ctx or not next_ctx or prev_gap + next_gap > max_ctx_gap:
            continue

        deviation = _blip_deviation(
            pos_by_frame[frame], prev_ctx, next_ctx, prev_gap, next_gap
        )
        conf_factor = 0.5 + 0.5 * pos_by_frame[frame].confidence
        if deviation > max_dev * conf_factor:
            suspect_indices.add(i)

    if not suspect_indices:
        return positions

    # Phase 1b: Re-evaluate suspects with clean context (skip other suspects)
    # Prevents blip positions from contaminating nearby real frame context
    confirmed: set[int] = set()
    for i in suspect_indices:
        prev_ctx, prev_gap, next_ctx, next_gap = _find_blip_context(
            i, frames, pos_by_frame, min_dist, exclude=suspect_indices
        )
        # Skip if context is too far -- linear interpolation becomes unreliable
        if not prev_ctx or not next_ctx or prev_gap + next_gap > max_ctx_gap:
            continue

        deviation = _blip_deviation(
            pos_by_frame[frames[i]], prev_ctx, next_ctx, prev_gap, next_gap
        )
        conf_factor = 0.5 + 0.5 * pos_by_frame[frames[i]].confidence
        if deviation > max_dev * conf_factor:
            confirmed.add(i)

    if not confirmed:
        return positions

    suspect_indices = confirmed

    # Phase 2: Group consecutive suspects into runs, keep only compact clusters
    blip_frames: set[int] = set()
    sorted_suspects = sorted(suspect_indices)
    runs: list[list[int]] = [[sorted_suspects[0]]]
    for idx in sorted_suspects[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])

    for run in runs:
        # Require >=2 consecutive suspect frames (single deviations are
        # real trajectory changes like bounces)
        if len(run) < 2:
            continue

        # Check cluster compactness -- blips are at a fixed player position,
        # real bounces spread along a curve.
        # Scale spread tolerance with cluster length: longer blips have
        # transitional frames as tracker moves to/from wrong position,
        # creating more spread even though Phase 1 confirmed deviation.
        effective_spread = min(
            max_blip_spread + 0.01 * max(0, len(run) - 2),
            3 * max_blip_spread,
        )
        cluster_positions = [pos_by_frame[frames[i]] for i in run]
        cx = float(np.mean([p.x for p in cluster_positions]))
        cy = float(np.mean([p.y for p in cluster_positions]))
        spread = float(max(
            np.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2)
            for p in cluster_positions
        ))

        if spread <= effective_spread:
            for i in run:
                blip_frames.add(frames[i])
            frame_range = f"[{frames[run[0]]}-{frames[run[-1]]}]"
            logger.info(
                f"Blip removal: removed {len(run)} frames {frame_range} "
                f"(spread={spread:.4f})"
            )

    if blip_frames:
        return [
            p for p in positions if p.frame_number not in blip_frames
        ]

    return positions
