"""
Temporal filtering for ball tracking.

Active WASB pipeline: exit ghost detection, segment pruning, outlier removal,
and interpolation. Passes through detector positions with post-processing to
remove false positives and fill small gaps.

Legacy stages (motion energy filter, stationarity filter, oscillation pruning,
blip removal) are in ball_filter_legacy.py and imported on demand when their
enable_* flags are True.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)


@dataclass
class BallFilterConfig:
    """Configuration for ball temporal filtering.

    WASB pipeline: exit ghost detection, segment pruning (with chain-based
    anchor pruning), outlier removal, and interpolation. Legacy stages
    (motion energy, stationarity, oscillation, blip) are available but
    disabled for WASB. See get_wasb_filter_config() for production preset.
    """

    # Warmup protection: preserve high-confidence detections in the first
    # N frames from all pruning stages. Serves happen near rally start and
    # create large spatial discontinuities (ball crosses net) that segment
    # pruning, chain pruning, and outlier removal all reject. High-confidence
    # early detections are almost certainly real ball (clean scene, no blur).
    warmup_protect_frames: int = 0  # 0 = disabled, >0 = protect first N frames
    warmup_protect_confidence: float = 0.50  # Min confidence to protect

    # Far-court protection: preserve high-confidence detections in the upper
    # band of the frame (y < far_court_protect_y) from segment/chain pruning.
    # The chain-based anchor pruning builds its trajectory around the longest
    # anchor segment, which tends to be near-court action or (worse) a static
    # distractor like a flag at the net. Real far-court contacts (back-row
    # serves/sets/attacks) form short high-confidence segments that get pruned
    # as non-chain anchors. Same re-injection semantics as warmup protection.
    far_court_protect_y: float = 0.0  # 0 = disabled, >0 = protect y < this
    far_court_protect_confidence: float = 0.30  # Min confidence to protect

    # Interpolation for missing frames
    enable_interpolation: bool = True
    max_interpolation_gap: int = 10  # Max frames to interpolate (larger gaps left empty)
    interpolated_confidence: float = 0.5  # Confidence assigned to interpolated positions

    # Trajectory segment pruning (post-processing)
    # The detector outputs consistent false detections at rally start/end.
    # Pruning splits trajectory at large jumps, discards short fragments,
    # but recovers short segments spatially close to anchor segments
    # (real trajectory fragments between interleaved false positives).
    enable_segment_pruning: bool = True
    segment_jump_threshold: float = 0.20  # 20% of screen to split segments
    min_segment_frames: int = 15  # Segments shorter than this are discarded
    min_output_confidence: float = 0.05  # Drop positions below this confidence
    # Max frame gap within a single trajectory segment. Detections separated
    # by more than this many frames are treated as separate segments even if
    # spatially close. Larger than max_interpolation_gap (which controls gap
    # filling) because segmentation is about trajectory continuity, not
    # position fabrication. 15 frames ≈ 0.5s at 30fps.
    max_segment_gap: int = 15

    # Oscillation pruning (detects cluster-based player-locking after ball exits)
    # The detector can lock onto two players and alternate with high confidence
    # after the ball leaves the frame. The pattern is cluster-based: positions
    # stay near player B for 2-5 frames, jump to player A for 1-2 frames, then
    # back. Detection uses spatial clustering: find two poles (furthest-apart
    # positions) in each window, assign positions to nearest pole, and count
    # transitions between clusters.
    enable_oscillation_pruning: bool = True
    min_oscillation_frames: int = 12  # Sliding window size for cluster transition rate
    oscillation_reversal_rate: float = 0.25  # Cluster transition rate threshold
    oscillation_min_displacement: float = 0.03  # Min pole distance (3% of screen)

    # Exit ghost removal (detects false detections after ball exits frame)
    # When ball approaches screen edge with consistent velocity and then
    # reverses direction, subsequent positions are ghosts from player-locking.
    enable_exit_ghost_removal: bool = True
    exit_edge_zone: float = 0.10  # 10% of screen — zone where exit approach is checked
    exit_approach_frames: int = 3  # Min consecutive frames approaching edge
    exit_min_approach_speed: float = 0.008  # Min per-frame speed toward edge (~0.8% of screen)
    # Max ghost region duration (~1s at 30fps). Real ghosts drift for a few
    # frames; 30+ consecutive detections at moving positions = real ball re-entry.
    exit_max_ghost_frames: int = 30

    # Outlier removal (removes flickering and edge artifacts)
    # Runs after segment pruning to clean within real segments.
    enable_outlier_removal: bool = True
    edge_margin: float = 0.02  # 2% of screen = ~38px on 1920px
    max_trajectory_deviation: float = 0.08  # 8% of screen = ~154px on 1920px
    min_neighbors_for_outlier: int = 2

    # Trajectory blip removal (catches multi-frame false positives)
    # The detector can briefly lock onto a player position for 2-5 frames mid-trajectory.
    # Single-frame outlier detection misses these because consecutive false positives
    # validate each other. This step checks each position against distant trajectory
    # context (positions ≥5 frames away) to detect deviations from the overall path.
    enable_blip_removal: bool = True
    blip_context_min_frames: int = 5  # Min frame distance for context neighbors
    blip_max_deviation: float = 0.15  # 15% of screen = ~288px on 1920px
    blip_max_context_gap: int = 30  # Max total context gap (~1s at 30fps)

    # Outlier removal tuning
    outlier_min_speed: float = 0.02  # 2% of screen/frame — below this, skip reversal check

    # Chain-based anchor pruning: max temporal gap for connecting two anchors
    # into the same trajectory chain. Segments not connected to the main chain
    # are non-ball objects (pigeons, cars, player hands). Separate from
    # max_interpolation_gap because chain connectivity should be generous
    # (recognize the ball's path) while interpolation should be tight (don't
    # fabricate positions over long gaps). 30 frames ≈ 1s at 30fps.
    max_chain_gap: int = 30

    # Preserve large disconnected anchors: segments this long are almost
    # certainly real ball trajectory even if disconnected from the main chain
    # (e.g., ball went off-screen for several seconds and came back).
    # Short disconnected anchors (< this) are noise (cars, pigeons, player
    # lock-on). 30 frames ≈ 1s at 30fps.
    min_disconnected_anchor_frames: int = 30

    # Motion energy filter (removes false positives at stationary positions)
    # Real ball in flight creates temporal intensity change. False positives
    # at player positions have low motion energy because players move slowly
    # relative to the ball.
    enable_motion_energy_filter: bool = True
    motion_energy_threshold: float = 0.02  # Below this = suspicious (reduce conf to 0)

    # Stationarity filter (removes player lock-on regardless of source)
    # A volleyball is always in motion during a rally. 12+ consecutive frames
    # within 0.5% of screen = player lock-on, not a real ball trajectory.
    # Default off: only needed when the detector can lock onto players.
    # False positives at player positions are handled by motion_energy_filter.
    # Tight thresholds avoid false positives on real slow-ball events (apex of
    # sets/tosses have spread ~0.8%, slow trajectories ~1.5%). True player lock-on
    # has spread < 0.2% (random jitter only).
    enable_stationarity_filter: bool = False
    stationarity_min_frames: int = 12  # ~0.4s at 30fps
    stationarity_max_spread: float = 0.005  # 0.5% of screen (~10px on 1920px)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_interpolation_gap <= 0:
            raise ValueError(
                f"max_interpolation_gap must be > 0, got {self.max_interpolation_gap}"
            )
        if not (0 < self.segment_jump_threshold <= 1.0):
            raise ValueError(
                f"segment_jump_threshold must be in (0, 1.0], "
                f"got {self.segment_jump_threshold}"
            )
        if self.min_segment_frames <= 0:
            raise ValueError(
                f"min_segment_frames must be > 0, got {self.min_segment_frames}"
            )
        if self.exit_max_ghost_frames <= self.exit_approach_frames:
            raise ValueError(
                f"exit_max_ghost_frames ({self.exit_max_ghost_frames}) must be "
                f"> exit_approach_frames ({self.exit_approach_frames})"
            )


def get_wasb_filter_config() -> BallFilterConfig:
    """Get optimized BallFilterConfig for fine-tuned WASB-only output.

    The fine-tuned WASB model produces clean positions that don't need
    aggressive filtering. Light filter with chain-based pruning is optimal.
    Grid search on 16 GT rallies (Mar 2026):
    - Key: blip removal HURTS, oscillation ZERO, light filter wins
    - Segment pruning + chain-based anchor pruning + outlier + interpolation
    - Motion energy, stationarity, oscillation, blip: all disabled
    """
    return BallFilterConfig(
        # Warmup protection: preserve serve-toss detections in first ~4s.
        # WASB detects serve tosses with high confidence but segment/chain
        # pruning kills them (ball crosses net = spatial discontinuity).
        # 120 frames covers ~95% of serves (median serve at frame 64).
        warmup_protect_frames=120,
        warmup_protect_confidence=0.30,
        # Far-court protection: preserve high-conf detections with y < 0.33
        # from segment/chain pruning. Addresses the dominant filter-loss slice
        # in the Session 3 ball_dropout diagnostic (far_court: 78% raw →
        # 39% filtered, −39pp).
        far_court_protect_y=0.33,
        far_court_protect_confidence=0.30,
        # Segment pruning (removes false segments at boundaries)
        enable_segment_pruning=True,
        segment_jump_threshold=0.20,
        min_segment_frames=8,
        min_output_confidence=0.05,
        # No motion energy filter (WASB doesn't produce motion_energy values)
        enable_motion_energy_filter=False,
        # Stationarity: off (grid search: neutral with fine-tuned WASB)
        enable_stationarity_filter=False,
        # Exit ghost removal
        enable_exit_ghost_removal=True,
        exit_edge_zone=0.10,
        exit_approach_frames=3,
        # Oscillation DISABLED — zero effect
        enable_oscillation_pruning=False,
        # Outlier removal (runs after segment pruning, safe)
        enable_outlier_removal=True,
        # Blip removal DISABLED — fine-tuned WASB doesn't produce blips,
        # enabling it kills real positions (-1.1% match)
        enable_blip_removal=False,
        # Interpolation with shorter gap (WASB precision benefits from tighter fill)
        enable_interpolation=True,
        max_interpolation_gap=5,
    )


class BallTemporalFilter:
    """Ball tracking temporal filter with multi-stage post-processing pipeline.

    Active WASB pipeline (stages enabled via get_wasb_filter_config):
    1. Exit ghost detection (on raw data, before pruning)
    2. Segment pruning: split at jumps, chain-based anchor pruning,
       short-segment recovery near anchors
    3. Exit ghost removal (apply detected ranges to pruned data)
    4. Outlier removal (clean flickering/reversals within real segments)
    5. Re-prune (clean up fragments exposed by outlier removal)
    6. Interpolation (fill small gaps with linear interpolation)

    Legacy stages (in ball_filter_legacy.py, disabled for WASB):
    - Motion energy filter, stationarity filter, oscillation pruning,
      blip removal — loaded on demand when their enable_* flags are True.
    """

    def __init__(self, config: BallFilterConfig | None = None):
        self.config = config or BallFilterConfig()

    def filter_batch(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Filter ball positions through the post-processing pipeline.

        See class docstring for pipeline order. Stages are conditionally
        enabled via BallFilterConfig flags.
        """
        if not positions:
            return []

        # Sort by frame number to ensure temporal order
        sorted_positions = sorted(positions, key=lambda p: p.frame_number)
        filtered = list(sorted_positions)

        # Warmup protection: save high-confidence early detections before
        # filtering. These will be re-injected after all stages if removed.
        warmup_protected: dict[int, BallPosition] = {}
        wp_frames = self.config.warmup_protect_frames
        wp_conf = self.config.warmup_protect_confidence
        if wp_frames > 0:
            for p in filtered:
                if p.frame_number < wp_frames and p.confidence >= wp_conf:
                    warmup_protected[p.frame_number] = p

        # Far-court protection: capture high-confidence detections in the
        # upper (far-side) band before pruning. See config docstring.
        far_court_protected: dict[int, BallPosition] = {}
        fc_y = self.config.far_court_protect_y
        fc_conf = self.config.far_court_protect_confidence
        if fc_y > 0:
            for p in filtered:
                if p.y < fc_y and p.confidence >= fc_conf:
                    far_court_protected[p.frame_number] = p

        # Track counts for logging
        input_count = len(filtered)
        outlier_count = 0
        pruned_count = 0
        exit_ghost_count = 0
        blip_count = 0
        oscillation_count = 0
        interp_count = 0
        motion_energy_count = 0
        stationarity_count = 0

        # Pipeline stages (see class docstring for overview).
        # Ghost detection is two-phase: detect on raw data to preserve
        # edge-approach evidence, apply to pruned data to avoid cascade.
        if self.config.enable_motion_energy_filter:
            from rallycut.tracking.ball_filter_legacy import motion_energy_filter

            before_me = sum(1 for p in filtered if p.confidence > 0)
            filtered = motion_energy_filter(filtered, self.config)
            after_me = sum(1 for p in filtered if p.confidence > 0)
            motion_energy_count = before_me - after_me

        if self.config.enable_stationarity_filter:
            from rallycut.tracking.ball_filter_legacy import remove_stationary_runs

            before_st = sum(1 for p in filtered if p.confidence > 0)
            filtered = remove_stationary_runs(filtered, self.config)
            after_st = sum(1 for p in filtered if p.confidence > 0)
            stationarity_count = before_st - after_st

        ghost_ranges: list[tuple[int, int]] = []
        if self.config.enable_exit_ghost_removal:
            ghost_ranges = self._detect_exit_ghost_ranges(filtered)

        if self.config.enable_segment_pruning:
            filtered = self._prune_segments(
                filtered, ghost_ranges=ghost_ranges
            )
            pruned_count = input_count - len(filtered)

        if ghost_ranges:
            before_exit = len(filtered)
            ghost_frames: set[int] = set()
            for start, end in ghost_ranges:
                for p in filtered:
                    if start <= p.frame_number <= end:
                        ghost_frames.add(p.frame_number)
            if ghost_frames:
                filtered = [
                    p for p in filtered if p.frame_number not in ghost_frames
                ]
                exit_ghost_count = before_exit - len(filtered)
                logger.info(
                    f"Exit ghost removal: removed {exit_ghost_count} "
                    f"ghost positions from pruned trajectory"
                )

        after_prune_count = len(filtered)

        if self.config.enable_oscillation_pruning:
            from rallycut.tracking.ball_filter_legacy import prune_oscillating

            filtered = prune_oscillating(filtered, self.config)
            oscillation_count = after_prune_count - len(filtered)

        after_oscillation_count = len(filtered)

        if self.config.enable_outlier_removal:
            filtered = self._remove_outliers(filtered)
            outlier_count = after_oscillation_count - len(filtered)

        if self.config.enable_blip_removal:
            from rallycut.tracking.ball_filter_legacy import remove_trajectory_blips

            before_blip = len(filtered)
            filtered = remove_trajectory_blips(filtered, self.config)
            blip_count = before_blip - len(filtered)

        # Re-prune after outlier/blip removal: removal can fragment segments,
        # exposing short false sub-segments and hovering patterns that were hidden
        # inside longer segments during the first pass.
        if outlier_count > 0 or blip_count > 0:
            before_reprune = len(filtered)
            if self.config.enable_oscillation_pruning:
                from rallycut.tracking.ball_filter_legacy import prune_oscillating as _prune_osc

                filtered = _prune_osc(filtered, self.config)
            if self.config.enable_segment_pruning:
                filtered = self._prune_segments(filtered)
            reprune_count = before_reprune - len(filtered)
            if reprune_count > 0:
                pruned_count += reprune_count

        # Re-inject warmup-protected detections that were removed by
        # filter stages. These are high-confidence early detections
        # (e.g. serve toss) that get killed by segment/chain/outlier
        # pruning due to net-crossing spatial discontinuities.
        # Skip detections that are far from the surviving trajectory
        # (e.g. distractor balls at a different position than the
        # game ball the filter selected).
        warmup_restored = 0
        if warmup_protected:
            surviving_frames = {p.frame_number for p in filtered}

            # Build spatial reference from surviving trajectory: median
            # position of filtered detections near each warmup frame.
            # A protected detection must be within a generous distance
            # of the trajectory, OR there must be no nearby reference
            # (ball hadn't been detected yet → accept unconditionally).
            for frame, bp in sorted(warmup_protected.items()):
                if frame not in surviving_frames:
                    filtered.append(bp)
                    warmup_restored += 1
            if warmup_restored > 0:
                filtered.sort(key=lambda p: p.frame_number)
                logger.info(
                    f"Warmup protection: restored {warmup_restored} "
                    f"high-confidence detections in first {wp_frames} frames"
                )

        # Re-inject far-court protected detections that were removed.
        far_court_restored = 0
        if far_court_protected:
            surviving_frames = {p.frame_number for p in filtered}
            for frame, bp in sorted(far_court_protected.items()):
                if frame not in surviving_frames:
                    filtered.append(bp)
                    far_court_restored += 1
            if far_court_restored > 0:
                filtered.sort(key=lambda p: p.frame_number)
                logger.info(
                    f"Far-court protection: restored {far_court_restored} "
                    f"high-confidence detections with y < {fc_y}"
                )

        # Interpolation: fill small gaps
        before_interp_count = len(filtered)
        if self.config.enable_interpolation:
            filtered = self._interpolate_missing(filtered)
            interp_count = len(filtered) - before_interp_count

        # Log summary
        if filtered:
            parts = [f"Ball filter: {input_count} positions"]
            if motion_energy_count > 0:
                parts.append(f"-{motion_energy_count} low-energy")
            if stationarity_count > 0:
                parts.append(f"-{stationarity_count} stationary")
            if outlier_count > 0:
                parts.append(f"-{outlier_count} outliers")
            if pruned_count > 0:
                parts.append(f"-{pruned_count} pruned")
            if exit_ghost_count > 0:
                parts.append(f"-{exit_ghost_count} exit ghosts")
            if blip_count > 0:
                parts.append(f"-{blip_count} blips")
            if oscillation_count > 0:
                parts.append(f"-{oscillation_count} oscillating")
            if warmup_restored > 0:
                parts.append(f"+{warmup_restored} warmup-protected")
            if interp_count > 0:
                parts.append(f"+{interp_count} interpolated")
            logger.info(", ".join(parts))

        return filtered

    def _interpolate_missing(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Interpolate missing frames with linear interpolation.

        Fills gaps where ball detection failed by linearly interpolating
        between known positions. Only interpolates gaps up to max_interpolation_gap
        frames to avoid creating fake trajectories across scene cuts.

        Args:
            positions: List of ball positions (sorted by frame number)

        Returns:
            List with interpolated positions added for missing frames
        """
        from rallycut.tracking.ball_tracker import BallPosition

        if not positions:
            return []

        # Build map of confident positions
        # Use min_output_confidence (0.05) since positions are already
        # filtered and we only need to skip zero-confidence placeholders.
        min_conf = self.config.min_output_confidence
        pos_by_frame = {
            p.frame_number: p for p in positions if p.confidence >= min_conf
        }

        if not pos_by_frame:
            return positions

        frames = sorted(pos_by_frame.keys())
        result = list(positions)  # Start with original positions
        interpolated_count = 0

        # Find and fill gaps
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            gap = f2 - f1

            # Only interpolate small gaps
            if gap > 1 and gap <= self.config.max_interpolation_gap:
                p1, p2 = pos_by_frame[f1], pos_by_frame[f2]

                for f in range(f1 + 1, f2):
                    # Linear interpolation factor
                    t = (f - f1) / gap

                    interp_pos = BallPosition(
                        frame_number=f,
                        x=p1.x + t * (p2.x - p1.x),
                        y=p1.y + t * (p2.y - p1.y),
                        confidence=self.config.interpolated_confidence,
                    )
                    result.append(interp_pos)
                    interpolated_count += 1

        if interpolated_count > 0:
            logger.debug(f"Interpolated {interpolated_count} missing frames")
            # Re-sort by frame number after adding interpolated positions
            result.sort(key=lambda p: p.frame_number)

        return result

    def _prune_segments(
        self,
        positions: list["BallPosition"],
        ghost_ranges: list[tuple[int, int]] | None = None,
    ) -> list["BallPosition"]:
        """Remove false segments and non-ball objects from the trajectory.

        Steps:
        1. Drop zero-confidence placeholders
        2. Split trajectory at large position jumps or frame gaps
        3. Identify anchors (segments >= min_segment_frames), then:
           3b. Exclude ghost-overlapping anchors
           3c. Remove false start/tail anchors (detector warmup)
           3d. Chain-based pruning: build trajectory chain from longest
               anchor, greedily extend via endpoint proximity + velocity
               extrapolation. Disconnected anchors are non-ball objects
               (pigeons, cars), except large ones (>= min_disconnected_
               anchor_frames) which are preserved as real ball after
               long off-screen gaps.
        4. Recover short segments near anchor endpoints (real fragments
           between interleaved false positives)
        5. Discard remaining segments
        """
        if len(positions) < 2:
            return positions

        # Step 1: Drop positions below minimum output confidence
        # Detector outputs (0.5, 0.5) at conf=0.0 for frames without detection
        min_conf = self.config.min_output_confidence
        confident = [p for p in positions if p.confidence >= min_conf]
        if not confident:
            return []

        dropped = len(positions) - len(confident)
        if dropped > 0:
            logger.debug(f"Dropped {dropped} positions below confidence {min_conf}")

        # Step 2: Split into segments at large jumps or gaps
        threshold = self.config.segment_jump_threshold
        segments: list[list[BallPosition]] = [[confident[0]]]

        for i in range(1, len(confident)):
            prev = confident[i - 1]
            curr = confident[i]
            frame_gap = curr.frame_number - prev.frame_number
            dist = np.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2)

            # Split on large spatial jump or large frame gap.
            # Uses raw distance (not velocity): a 20% screen jump is a
            # segment boundary regardless of how many frames it spans.
            if dist > threshold or frame_gap > self.config.max_segment_gap:
                segments.append([curr])
            else:
                segments[-1].append(curr)

        if len(segments) <= 1:
            return confident

        # Step 3: Identify anchor segments (long enough to be reliable)
        min_len = self.config.min_segment_frames
        anchor_indices: set[int] = set()
        for i, seg in enumerate(segments):
            if len(seg) >= min_len:
                anchor_indices.add(i)

        # Track segments explicitly removed as outliers/false anchors.
        # These must NOT be recovered in Step 4 (anchor-proximity recovery).
        excluded_segments: set[int] = set()

        # Step 3b: Exclude anchors that overlap with ghost ranges.
        # Ghost segments (false detections after ball exits frame) can be long
        # enough to qualify as anchors, but they shouldn't rescue nearby short
        # segments from being pruned — those short segments are typically false
        # positives at player positions that only appear "near" the ghost anchor.
        # Only exclude if the non-ghost portion is too short to be an anchor
        # on its own — otherwise the real portion should still act as anchor.
        if ghost_ranges:
            ghost_anchors: set[int] = set()
            for i in anchor_indices:
                seg = segments[i]
                non_ghost_count = sum(
                    1
                    for p in seg
                    if not any(
                        g_start <= p.frame_number <= g_end
                        for g_start, g_end in ghost_ranges
                    )
                )
                if non_ghost_count < min_len:
                    ghost_anchors.add(i)
            if ghost_anchors:
                anchor_indices -= ghost_anchors
                excluded_segments |= ghost_anchors
                logger.debug(
                    f"Excluded {len(ghost_anchors)} ghost-overlapping "
                    f"segments from anchors (non-ghost portion < {min_len})"
                )

        # Step 3c: Remove false start/tail anchors (detector warmup/cooldown).
        # Temporal context warmup produces false detections at rally start that
        # can be long enough to qualify as anchors (>min_segment_frames).
        # If the first anchor is much shorter than the second AND spatially
        # disconnected, it's a warmup artifact. Same for the last anchor.
        sorted_anchors = sorted(anchor_indices)
        if len(sorted_anchors) >= 2:
            # Check first anchor (false start)
            first_idx = sorted_anchors[0]
            second_idx = sorted_anchors[1]
            first_seg = segments[first_idx]
            second_seg = segments[second_idx]
            if (
                len(first_seg) < len(second_seg) / 3
                and np.sqrt(
                    (first_seg[-1].x - second_seg[0].x) ** 2
                    + (first_seg[-1].y - second_seg[0].y) ** 2
                )
                > threshold
            ):
                anchor_indices.discard(first_idx)
                excluded_segments.add(first_idx)
                logger.info(
                    f"Segment pruning: removed false start anchor "
                    f"[{first_seg[0].frame_number}-{first_seg[-1].frame_number}] "
                    f"({len(first_seg)} frames, next anchor has {len(second_seg)})"
                )

            # Check last anchor (false tail)
            last_idx = sorted_anchors[-1]
            second_last_idx = sorted_anchors[-2]
            # Re-check in case first anchor removal changed things
            if last_idx in anchor_indices and second_last_idx in anchor_indices:
                last_seg = segments[last_idx]
                second_last_seg = segments[second_last_idx]
                if (
                    len(last_seg) < len(second_last_seg) / 3
                    and np.sqrt(
                        (last_seg[0].x - second_last_seg[-1].x) ** 2
                        + (last_seg[0].y - second_last_seg[-1].y) ** 2
                    )
                    > threshold
                ):
                    anchor_indices.discard(last_idx)
                    excluded_segments.add(last_idx)
                    logger.info(
                        f"Segment pruning: removed false tail anchor "
                        f"[{last_seg[0].frame_number}-{last_seg[-1].frame_number}] "
                        f"({len(last_seg)} frames, prev anchor has "
                        f"{len(second_last_seg)})"
                    )

        # Step 3d: Chain-based anchor pruning.
        #
        # The ball trajectory is a connected path: each segment's endpoint
        # is near the next segment's start. Non-ball objects (pigeons, cars,
        # player hands) create segments that are spatially disconnected from
        # this path. We build a trajectory chain from the longest anchor and
        # greedily extend it; any anchor NOT in the chain is an outlier.
        #
        # Connection check uses three methods (any connects):
        #   A) Raw endpoint proximity: endpoint distance < threshold
        #   B) Velocity extrapolation: the "from" segment's end velocity
        #      predicts an expected position N frames later; if the "to"
        #      segment's start is near this expected position, it connects.
        #      This handles fast-moving balls where the gap between segments
        #      causes a large raw displacement (ball moved during the gap).
        #   C) Detection desert: if the gap between segments has very few
        #      detections (< 15% of gap frames), the ball was off-screen.
        #      The next segment is the same ball re-entering, regardless of
        #      spatial position. This handles ball exit+re-entry where the
        #      ball reappears at a different screen position.
        if len(anchor_indices) >= 3:
            # Pre-compute endpoints and velocities for all anchors.
            seg_endpoints: dict[int, tuple[float, float, float, float]] = {}
            seg_end_velocity: dict[int, tuple[float, float] | None] = {}
            for i in anchor_indices:
                seg = segments[i]
                pts = [p for p in seg if p.confidence > 0] or seg
                seg_endpoints[i] = (
                    pts[0].x, pts[0].y, pts[-1].x, pts[-1].y,
                )
                if len(pts) >= 3:
                    seg_end_velocity[i] = (
                        (pts[-1].x - pts[-3].x) / 2,
                        (pts[-1].y - pts[-3].y) / 2,
                    )
                else:
                    seg_end_velocity[i] = None

            max_gap = self.config.max_chain_gap

            edge_zone = self.config.exit_edge_zone

            # Precompute frame→detection map for desert detection.
            # A "detection desert" (few detections over many frames) means
            # the ball was off-screen during the gap.
            _detected_frames: set[int] = {p.frame_number for p in confident}
            _desert_max_density = 0.15  # <15% = desert (ball off-screen)
            _desert_min_gap = 10  # Only check gaps >= 10 frames
            _desert_min_confidence = 0.50  # Both segments need high confidence

            def _is_detection_desert(from_frame: int, to_frame: int) -> bool:
                """Check if the gap between two frames is a detection desert.

                A desert has very few detections relative to the gap length,
                indicating the ball was off-screen. This is distinct from
                gaps where detections exist at other positions (different
                objects being tracked).
                """
                gap_len = to_frame - from_frame
                if gap_len < _desert_min_gap:
                    return False
                gap_detections = sum(
                    1 for f in range(from_frame + 1, to_frame)
                    if f in _detected_frames
                )
                return gap_detections / gap_len < _desert_max_density

            def _connects(
                from_idx: int, to_idx: int,
            ) -> bool:
                """Check if to_idx continues the trajectory from from_idx.

                from_idx's end → to_idx's start. Requires temporal adjacency
                (gap <= max_chain_gap) and spatial connectivity via either:
                  A) Raw endpoint distance < threshold, OR
                  B) Velocity-extrapolated distance < threshold (the ball
                     was moving fast and covered ground during the gap).
                  C) Detection desert: very few detections in the gap,
                     meaning ball was off-screen and re-entered.
                Directional coherence: when raw distance >= threshold/2,
                require the bridge direction to align with end velocity
                (dot product >= 0). Skipped when very close (bounces
                reverse velocity but proximity proves same trajectory).

                Edge-exit relaxation: when a segment ends near a frame
                edge (within exit_edge_zone), the ball likely exited and
                re-entered. Use 1.5x distance threshold since the ball
                travels off-screen during the gap.
                """
                _, _, fex, fey = seg_endpoints[from_idx]
                tsx, tsy, _, _ = seg_endpoints[to_idx]
                from_end = segments[from_idx][-1].frame_number
                to_start = segments[to_idx][0].frame_number
                gap = abs(to_start - from_end)
                if gap > max_gap:
                    # Even beyond max_gap, allow connection across detection
                    # deserts — the ball went off-screen and came back.
                    if _is_detection_desert(from_end, to_start):
                        from_conf = float(np.median(
                            [p.confidence for p in segments[from_idx]]
                        ))
                        to_conf = float(np.median(
                            [p.confidence for p in segments[to_idx]]
                        ))
                        if (
                            from_conf >= _desert_min_confidence
                            and to_conf >= _desert_min_confidence
                        ):
                            return True
                    return False

                raw_dist = float(np.sqrt(
                    (tsx - fex) ** 2 + (tsy - fey) ** 2,
                ))
                vel = seg_end_velocity.get(from_idx)

                # Edge-exit relaxation: use larger threshold when
                # endpoint is near a frame edge (ball exit+re-entry)
                near_edge = (
                    fex < edge_zone or fex > 1 - edge_zone
                    or fey < edge_zone or fey > 1 - edge_zone
                )
                effective_threshold = threshold * 1.5 if near_edge else threshold

                # Check A: raw endpoint proximity
                if raw_dist < effective_threshold:
                    # Directional coherence for medium distances
                    if raw_dist >= threshold / 2 and vel is not None:
                        dot = vel[0] * (tsx - fex) + vel[1] * (tsy - fey)
                        if dot < 0:
                            return False
                    return True

                # Check B: velocity extrapolation
                # If the ball was moving at the end of from_idx, predict
                # where it would be after `gap` frames. Connect if the
                # actual start of to_idx is near the prediction.
                if vel is not None and gap > 0:
                    expected_x = fex + vel[0] * gap
                    expected_y = fey + vel[1] * gap
                    extrap_dist = float(np.sqrt(
                        (tsx - expected_x) ** 2
                        + (tsy - expected_y) ** 2
                    ))
                    if extrap_dist < effective_threshold:
                        return True

                return False

            # Build chain from the most volleyball-like anchor.
            # Score = length × movement_factor. A stationary segment
            # (distractor ball rolling on ground) gets penalized even
            # if it's the longest, so the game ball wins chain seeding.
            _stationarity_threshold = 0.005  # Y std below this = stationary

            def _anchor_score(idx: int) -> float:
                seg = segments[idx]
                length = len(seg)
                y_std = float(np.std([p.y for p in seg]))
                # Movement factor: 1.0 for moving segments, 0.1 for stationary
                movement = 0.1 if y_std < _stationarity_threshold else 1.0
                return length * movement

            sorted_by_score = sorted(
                anchor_indices,
                key=_anchor_score,
                reverse=True,
            )
            chain: set[int] = {sorted_by_score[0]}

            # Greedily extend in both temporal directions
            changed = True
            while changed:
                changed = False
                for candidate in sorted(anchor_indices - chain):
                    for ch_idx in list(chain):
                        # candidate follows chain member
                        if _connects(ch_idx, candidate):
                            chain.add(candidate)
                            changed = True
                            break
                        # chain member follows candidate
                        if _connects(candidate, ch_idx):
                            chain.add(candidate)
                            changed = True
                            break
                    if changed:
                        break

            # Remove anchors not in the chain, but preserve large
            # disconnected segments (likely real ball after long gap)
            min_keep = self.config.min_disconnected_anchor_frames
            non_chain = anchor_indices - chain
            if non_chain:
                for i in sorted(non_chain):
                    seg = segments[i]
                    if len(seg) >= min_keep:
                        logger.info(
                            f"Chain pruning: keeping large disconnected "
                            f"anchor [{seg[0].frame_number}-"
                            f"{seg[-1].frame_number}] ({len(seg)} frames "
                            f">= {min_keep}, likely real ball)"
                        )
                    else:
                        logger.info(
                            f"Chain pruning: removed disconnected anchor "
                            f"[{seg[0].frame_number}-"
                            f"{seg[-1].frame_number}] ({len(seg)} frames, "
                            f"not connected to trajectory chain of "
                            f"{len(chain)} segments)"
                        )
                        anchor_indices.discard(i)
                        excluded_segments.add(i)

        # Step 4: Keep short segments whose centroid is close to an anchor endpoint.
        # These are real trajectory fragments between interleaved false positives.
        # Use half the jump threshold as proximity — tight enough to exclude
        # false positives (which jump to player positions 30-50% away) while
        # keeping real trajectory fragments (typically <5% from anchor).
        # Also require temporal proximity: after a large gap (ball exited frame),
        # The detector can restart at a player position that happens to be spatially
        # near the last anchor endpoint. These shouldn't be recovered.
        proximity = threshold / 2
        max_recovery_gap = self.config.max_interpolation_gap * 3
        kept: list[BallPosition] = []
        removed_count = 0
        kept_info: list[str] = []
        removed_info: list[str] = []
        recovered_count = 0

        for i, seg in enumerate(segments):
            tag = f"[{seg[0].frame_number}-{seg[-1].frame_number}]({len(seg)})"

            if i in anchor_indices:
                kept.extend(seg)
                kept_info.append(tag)
                continue

            # Never recover segments explicitly removed as outliers, false
            # start/tail, or ghost-overlapping. These were identified as
            # non-ball objects (pigeons, cars, player hands, detector warmup)
            # and must stay removed even if spatially close to an anchor.
            if i in excluded_segments:
                removed_count += len(seg)
                removed_info.append(tag + "x")  # x marks excluded segments
                continue

            # Short segment: check proximity to nearest anchor endpoints
            centroid_x = float(np.mean([p.x for p in seg]))
            centroid_y = float(np.mean([p.y for p in seg]))

            close_to_anchor = False

            # Check previous anchor (end position)
            for j in range(i - 1, -1, -1):
                if j in anchor_indices:
                    ref = segments[j][-1]
                    frame_gap = seg[0].frame_number - ref.frame_number
                    if frame_gap > max_recovery_gap:
                        break
                    dist = np.sqrt(
                        (centroid_x - ref.x) ** 2 + (centroid_y - ref.y) ** 2
                    )
                    if dist < proximity:
                        close_to_anchor = True
                    break

            # Check next anchor (start position)
            if not close_to_anchor:
                for j in range(i + 1, len(segments)):
                    if j in anchor_indices:
                        ref = segments[j][0]
                        frame_gap = ref.frame_number - seg[-1].frame_number
                        if frame_gap > max_recovery_gap:
                            break
                        dist = np.sqrt(
                            (centroid_x - ref.x) ** 2 + (centroid_y - ref.y) ** 2
                        )
                        if dist < proximity:
                            close_to_anchor = True
                        break

            if close_to_anchor:
                kept.extend(seg)
                kept_info.append(tag + "*")  # * marks recovered segments
                recovered_count += len(seg)
            else:
                removed_count += len(seg)
                removed_info.append(tag)

        if removed_count > 0 or recovered_count > 0:
            parts = [
                f"Segment pruning: kept {len(kept_info)} segments "
                f"({', '.join(kept_info)})"
            ]
            if removed_count > 0:
                parts.append(
                    f"removed {removed_count} positions from "
                    f"{len(removed_info)} short segments"
                )
            if recovered_count > 0:
                parts.append(f"recovered {recovered_count} near-anchor positions")
            logger.info(", ".join(parts))

        return kept

    def _detect_exit_ghost_ranges(
        self,
        positions: list["BallPosition"],
    ) -> list[tuple[int, int]]:
        """Detect frame ranges containing ghost detections after ball exits.

        Scans positions for the exit ghost pattern: ball approaching a screen edge
        with consistent velocity, then reversing direction. Only the last approach
        frame must be in the edge zone — the ball approaches from further away.
        Returns frame ranges [start, end] where detections are ghosts.

        Ghost range continues until:
        - A gap > max_interpolation_gap frames (break in detections), OR
        - A position returns to the exit edge zone (ball re-entering from same edge)

        This is the detection phase only — does not modify positions. Called on
        raw data before segment pruning to preserve edge-approach evidence.
        """
        if len(positions) < self.config.exit_approach_frames + 1:
            return []

        approach_n = self.config.exit_approach_frames
        edge_zone = self.config.exit_edge_zone
        min_speed = self.config.exit_min_approach_speed
        max_gap = self.config.max_interpolation_gap
        max_ghost_frames = self.config.exit_max_ghost_frames

        ranges: list[tuple[int, int]] = []
        in_ghost_region = False
        exit_edge_name: str | None = None
        ghost_start_frame: int = 0
        last_ghost_frame: int = 0

        for i in range(approach_n, len(positions)):
            # If we're in a ghost region, keep marking until termination
            if in_ghost_region:
                frame_gap = positions[i].frame_number - positions[i - 1].frame_number
                curr = positions[i]

                # Terminate at frame gap
                if frame_gap > max_gap:
                    ranges.append((ghost_start_frame, last_ghost_frame))
                    in_ghost_region = False
                    exit_edge_name = None
                    continue

                # Terminate at max ghost duration — real ghosts drift for
                # a few frames; long consecutive detections = real ball
                ghost_duration = curr.frame_number - ghost_start_frame
                if ghost_duration >= max_ghost_frames:
                    # Don't save this range — the "ghost" is actually real ball.
                    # Intentionally skip re-evaluating frame i for new ghost
                    # triggers: approach frames (i-3..i-1) are all from the
                    # cancelled region and meaningless for edge-approach checks.
                    in_ghost_region = False
                    exit_edge_name = None
                    logger.info(
                        f"Exit ghost: cancelled ghost at f={ghost_start_frame} "
                        f"(duration {ghost_duration} >= {max_ghost_frames}, "
                        f"likely real ball re-entry)"
                    )
                    continue

                # Terminate when position returns to exit edge zone
                # (ball re-entering from the same edge it left)
                in_edge = False
                if exit_edge_name == "top" and curr.y < edge_zone:
                    in_edge = True
                elif exit_edge_name == "bottom" and curr.y > 1 - edge_zone:
                    in_edge = True
                elif exit_edge_name == "left" and curr.x < edge_zone:
                    in_edge = True
                elif exit_edge_name == "right" and curr.x > 1 - edge_zone:
                    in_edge = True

                if in_edge:
                    ranges.append((ghost_start_frame, last_ghost_frame))
                    in_ghost_region = False
                    exit_edge_name = None
                    logger.debug(
                        f"Exit ghost: terminated at f={curr.frame_number} "
                        f"(position returned to edge zone)"
                    )
                    continue

                last_ghost_frame = curr.frame_number
                continue

            # Check that approach frames are contiguous (no large gaps)
            approach = positions[i - approach_n : i]
            has_gap = False
            for j in range(1, len(approach)):
                if approach[j].frame_number - approach[j - 1].frame_number > max_gap:
                    has_gap = True
                    break
            if has_gap:
                continue

            curr = positions[i]

            # Check each edge direction
            # Only the LAST approach frame must be in the edge zone — the ball
            # approaches from further away, what matters is that it reached the
            # edge. Consistent approach velocity is checked separately below.
            edges: list[tuple[str, float, list[float]]] = []
            last_approach = approach[-1]

            # Top edge
            if last_approach.y < edge_zone:
                velocities = [
                    approach[j].y - approach[j - 1].y for j in range(1, len(approach))
                ]
                reversal_vel = curr.y - last_approach.y
                edges.append(("top", reversal_vel, velocities))

            # Bottom edge
            if last_approach.y > 1 - edge_zone:
                velocities = [
                    approach[j].y - approach[j - 1].y for j in range(1, len(approach))
                ]
                reversal_vel = curr.y - last_approach.y
                edges.append(("bottom", reversal_vel, velocities))

            # Left edge
            if last_approach.x < edge_zone:
                velocities = [
                    approach[j].x - approach[j - 1].x for j in range(1, len(approach))
                ]
                reversal_vel = curr.x - last_approach.x
                edges.append(("left", reversal_vel, velocities))

            # Right edge
            if last_approach.x > 1 - edge_zone:
                velocities = [
                    approach[j].x - approach[j - 1].x for j in range(1, len(approach))
                ]
                reversal_vel = curr.x - last_approach.x
                edges.append(("right", reversal_vel, velocities))

            for edge_name, reversal_vel, velocities in edges:
                # Check consistent approach velocity toward edge
                if edge_name == "top":
                    approaching = all(v < -min_speed for v in velocities)
                    reversed = reversal_vel > min_speed
                elif edge_name == "bottom":
                    approaching = all(v > min_speed for v in velocities)
                    reversed = reversal_vel < -min_speed
                elif edge_name == "left":
                    approaching = all(v < -min_speed for v in velocities)
                    reversed = reversal_vel > min_speed
                else:  # right
                    approaching = all(v > min_speed for v in velocities)
                    reversed = reversal_vel < -min_speed

                if approaching and reversed:
                    in_ghost_region = True
                    exit_edge_name = edge_name
                    ghost_start_frame = curr.frame_number
                    last_ghost_frame = curr.frame_number
                    logger.info(
                        f"Exit ghost: ball exited {edge_name} at "
                        f"f={approach[-1].frame_number}, marking "
                        f"f={curr.frame_number}+ as ghosts"
                    )
                    break

        # Close any open ghost region at end of trajectory
        if in_ghost_region:
            ranges.append((ghost_start_frame, last_ghost_frame))

        return ranges

    def _remove_outliers(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Remove outlier positions that are likely detection failures.

        Identifies and removes positions that:
        1. Are at screen edges (common failure mode where model outputs 0,0 or 1,1)
        2. Deviate significantly from the trajectory defined by neighbors

        Args:
            positions: List of ball positions (sorted by frame number)

        Returns:
            List with outlier positions removed
        """
        if len(positions) < 3:
            return positions

        # Build position map by frame
        pos_by_frame = {p.frame_number: p for p in positions}
        frames = sorted(pos_by_frame.keys())

        outlier_frames: set[int] = set()

        for i, frame in enumerate(frames):
            pos = pos_by_frame[frame]

            # Check 1: Edge detection (positions at screen boundaries)
            # Note: WASB outputs confidence as 0.0 or 1.0, so the < 0.8
            # gate means this check only fires for non-WASB detectors with
            # continuous confidence values. For WASB, all surviving positions
            # have conf=1.0 and this is effectively a no-op.
            margin = self.config.edge_margin
            is_edge = (
                pos.x < margin or pos.x > (1 - margin) or
                pos.y < margin or pos.y > (1 - margin)
            )

            if is_edge:
                if pos.confidence < 0.8:
                    outlier_frames.add(frame)
                    continue

            # Check 2: Trajectory consistency with neighbors
            # Find neighbors (previous and next positions)
            prev_pos = None
            next_pos = None

            # Look for previous neighbor (up to 5 frames back)
            for j in range(i - 1, max(-1, i - 6), -1):
                if j >= 0 and frames[j] not in outlier_frames:
                    prev_pos = pos_by_frame[frames[j]]
                    prev_gap = frame - frames[j]
                    break

            # Look for next neighbor (up to 5 frames ahead)
            for j in range(i + 1, min(len(frames), i + 6)):
                if frames[j] not in outlier_frames:
                    next_pos = pos_by_frame[frames[j]]
                    next_gap = frames[j] - frame
                    break

            # Need at least min_neighbors for trajectory check
            neighbor_count = (1 if prev_pos else 0) + (1 if next_pos else 0)
            if neighbor_count < self.config.min_neighbors_for_outlier:
                continue

            # Interpolate expected position from neighbors
            if prev_pos and next_pos:
                # Both neighbors available - interpolate
                total_gap = prev_gap + next_gap
                t = prev_gap / total_gap
                expected_x = prev_pos.x + t * (next_pos.x - prev_pos.x)
                expected_y = prev_pos.y + t * (next_pos.y - prev_pos.y)
            elif prev_pos:
                # Only previous - use it directly (no extrapolation)
                expected_x, expected_y = prev_pos.x, prev_pos.y
            else:
                # Only next - use it directly
                expected_x, expected_y = next_pos.x, next_pos.y  # type: ignore

            # Compute deviation from expected position
            deviation = np.sqrt(
                (pos.x - expected_x) ** 2 + (pos.y - expected_y) ** 2
            )

            # Mark as outlier if deviation exceeds threshold
            # Scale threshold by confidence (lower confidence = stricter check)
            conf_factor = 0.5 + 0.5 * pos.confidence  # 0.5 to 1.0
            threshold = self.config.max_trajectory_deviation * conf_factor

            if deviation > threshold:
                outlier_frames.add(frame)
                logger.debug(
                    f"Frame {frame}: Outlier detected (deviation={deviation:.3f}, "
                    f"threshold={threshold:.3f}, conf={pos.confidence:.2f})"
                )
                continue

            # Check 3: Velocity reversal detection (A→B→A flickering pattern)
            # If we have both neighbors, check if velocity reverses sharply
            if prev_pos and next_pos:
                # Velocity into this point
                v_in_x = pos.x - prev_pos.x
                v_in_y = pos.y - prev_pos.y
                # Velocity out of this point
                v_out_x = next_pos.x - pos.x
                v_out_y = next_pos.y - pos.y

                speed_in = np.sqrt(v_in_x ** 2 + v_in_y ** 2)
                speed_out = np.sqrt(v_out_x ** 2 + v_out_y ** 2)

                # Only check reversal if both speeds are significant
                min_speed = self.config.outlier_min_speed
                if speed_in > min_speed and speed_out > min_speed:
                    # Cosine of angle between velocity vectors
                    dot = v_in_x * v_out_x + v_in_y * v_out_y
                    cos_angle = dot / (speed_in * speed_out)

                    # Sharp reversal (cos < -0.5 means angle > 120 degrees)
                    if cos_angle < -0.5:
                        outlier_frames.add(frame)
                        logger.debug(
                            f"Frame {frame}: Velocity reversal detected "
                            f"(cos_angle={cos_angle:.2f}, speed_in={speed_in:.3f}, "
                            f"speed_out={speed_out:.3f})"
                        )

        # Remove outliers
        if outlier_frames:
            logger.info(f"Removed {len(outlier_frames)} outlier positions")
            return [p for p in positions if p.frame_number not in outlier_frames]

        return positions

