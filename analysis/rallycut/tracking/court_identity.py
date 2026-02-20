"""Court-plane identity resolution for player tracking.

Uses homography-based court projection to detect and correct ID swaps
that occur during net interactions. Players who overlap in 2D image space
are always separated in court space (opposite sides of the net).

The algorithm:
1. Project all player detections to court coordinates (meters)
2. Detect net interactions where cross-team tracks are near the net
3. Score swap/no-swap hypotheses using side-of-net consistency,
   court-plane motion smoothness, bbox size consistency,
   touch grammar (Phase 3), and serve anchoring (Phase 2)
4. Apply corrections where confident, flag uncertainty where ambiguous
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rallycut.court.calibration import COURT_LENGTH, CourtCalibrator
from rallycut.tracking.player_tracker import PlayerPosition

if TYPE_CHECKING:
    from rallycut.tracking.color_repair import ColorHistogramStore
    from rallycut.tracking.contact_detector import ContactSequence
    from rallycut.tracking.identity_anchor import ServeAnchor
    from rallycut.tracking.team_aware_tracker import ActiveFreeze

logger = logging.getLogger(__name__)

# Net is at half court
NET_Y = COURT_LENGTH / 2.0  # 8.0 meters


@dataclass
class CourtIdentityConfig:
    """Configuration for court-plane identity resolution."""

    # Net interaction detection
    net_approach_distance: float = 2.0  # meters from net to count as "near net"

    # Observation windows
    observation_window: int = 30  # frames after interaction to observe side-of-net
    pre_interaction_window: int = 5  # frames before interaction for motion continuity

    # Hypothesis scoring weights
    weight_side_of_net: float = 0.40
    weight_motion_smoothness: float = 0.25
    weight_bbox_size: float = 0.15
    weight_color: float = 0.20

    # Decision thresholds
    confidence_margin: float = 0.15  # swap score must exceed no-swap by this much
    min_observation_frames: int = 10  # minimum post-interaction frames to decide

    # Dead zone around net (ignore positions within this distance of net line)
    net_dead_zone: float = 0.5  # meters

    # Minimum pre-interaction frames for motion template
    min_pre_frames: int = 3

    # Minimum team separation in image Y space to run court identity.
    # When teams are too close (e.g., <0.06), the homography can't reliably
    # distinguish near/far sides and false swaps occur.
    min_team_separation: float = 0.08


@dataclass
class CourtTrack:
    """Court-space positions for a single track."""

    track_id: int
    # frame_number -> (court_x, court_y) in meters
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)


@dataclass
class NetInteraction:
    """A detected net interaction between two tracks."""

    track_a: int  # near-team track
    track_b: int  # far-team track
    start_frame: int
    end_frame: int


@dataclass
class IdentityHypothesis:
    """Swap vs no-swap hypothesis with component scores."""

    interaction: NetInteraction
    # Component scores (higher = more evidence for this hypothesis)
    side_of_net_score: float = 0.0
    motion_smoothness_score: float = 0.0
    bbox_size_score: float = 0.0
    color_score: float = 0.0  # Color histogram consistency
    grammar_score: float = 0.0  # Phase 3: touch grammar
    serve_anchor_score: float = 0.0  # Phase 2: serve anchoring


@dataclass
class SwapDecision:
    """Decision about whether to swap two tracks."""

    interaction: NetInteraction
    should_swap: bool
    swap_score: float
    no_swap_score: float
    confident: bool  # True if margin exceeds threshold

    @property
    def margin(self) -> float:
        return abs(self.swap_score - self.no_swap_score)


class CourtIdentityResolver:
    """Resolves player identity swaps using court-plane projection."""

    def __init__(
        self,
        calibrator: CourtCalibrator,
        config: CourtIdentityConfig | None = None,
        video_width: int = 1920,
        video_height: int = 1080,
    ) -> None:
        self.calibrator = calibrator
        self.config = config or CourtIdentityConfig()
        self.video_width = video_width
        self.video_height = video_height

    def resolve(
        self,
        positions: list[PlayerPosition],
        team_assignments: dict[int, int],
        contact_sequence: ContactSequence | None = None,
        serve_anchor: ServeAnchor | None = None,
        color_store: ColorHistogramStore | None = None,
        frozen_interactions: list[ActiveFreeze] | None = None,
    ) -> tuple[list[PlayerPosition], list[SwapDecision]]:
        """Run court-plane identity resolution.

        Args:
            positions: All player positions (will be modified if swaps found).
            team_assignments: track_id -> team (0=near, 1=far).
            contact_sequence: Optional contact sequence for grammar scoring.
            serve_anchor: Optional serve anchor for identity anchoring.
            color_store: Optional color histogram store for appearance scoring.
            frozen_interactions: Optional list of freeze periods from in-tracker
                detector. Used to add no-swap bonus for freeze-protected interactions.

        Returns:
            Tuple of (positions, list of swap decisions made).
        """
        if not positions or not team_assignments:
            return positions, []

        # Guard: check team separation in image Y space
        if not self._teams_sufficiently_separated(positions, team_assignments):
            return positions, []

        # Step 1: Project all positions to court coordinates
        court_tracks = self._build_court_tracks(positions)

        # Compute effective net Y from actual team positions in court space.
        # The hardcoded NET_Y=8.0 assumes a specific calibration convention
        # (near baseline at court_y=0, far baseline at court_y=16). Many
        # calibrations project both teams to the same side of 8.0m due to
        # corner ordering differences. The midpoint between team medians
        # is a robust estimate of the net position regardless of convention.
        effective_net_y = self._estimate_effective_net_y(
            court_tracks, team_assignments
        )
        if effective_net_y is None:
            return positions, []

        # Step 2: Detect net interactions
        interactions = self._detect_net_interactions(
            court_tracks, team_assignments, effective_net_y
        )

        if not interactions:
            return positions, []

        logger.info(
            f"Court identity: {len(interactions)} net interactions detected"
        )

        # Step 3: Build image-space lookup for bbox scoring
        image_positions = self._build_image_lookup(positions)

        # Step 4: Score each interaction
        decisions: list[SwapDecision] = []
        swaps_applied = 0

        for interaction in interactions:
            decision = self._score_interaction(
                interaction, court_tracks, image_positions, team_assignments,
                effective_net_y=effective_net_y,
                contact_sequence=contact_sequence,
                serve_anchor=serve_anchor,
                color_store=color_store,
                frozen_interactions=frozen_interactions,
            )
            decisions.append(decision)

            if decision.should_swap and decision.confident:
                self._apply_swap(
                    positions, interaction.track_a, interaction.track_b,
                    interaction.end_frame, team_assignments,
                )
                swaps_applied += 1
                logger.info(
                    f"Court identity swap: {interaction.track_a}<->"
                    f"{interaction.track_b} at frame {interaction.end_frame} "
                    f"(margin={decision.margin:.3f})"
                )

        if swaps_applied:
            logger.info(
                f"Court identity: {swaps_applied} swaps applied "
                f"out of {len(interactions)} interactions"
            )

        return positions, decisions

    def _teams_sufficiently_separated(
        self,
        positions: list[PlayerPosition],
        team_assignments: dict[int, int],
    ) -> bool:
        """Check if near/far teams are far enough apart to run court identity.

        Uses image-space separation as the primary check, with court-space
        validation as a fallback when calibration is available. Teams that
        appear close in image Y (shallow camera angle) can still be well-
        separated in court space (opposite sides of the 16m court).
        """
        near_ys: list[float] = []
        far_ys: list[float] = []

        # Use first 60 frames (relative to rally start) for stable positions.
        # Frame numbers may be absolute video frames, not rally-relative.
        min_frame = min((p.frame_number for p in positions), default=0)
        frame_cutoff = min_frame + 60

        for p in positions:
            if p.frame_number > frame_cutoff:
                continue
            team = team_assignments.get(p.track_id, -1)
            if team == 0:
                near_ys.append(p.y)
            elif team == 1:
                far_ys.append(p.y)

        if not near_ys or not far_ys:
            return False

        near_median = sorted(near_ys)[len(near_ys) // 2]
        far_median = sorted(far_ys)[len(far_ys) // 2]
        separation = abs(near_median - far_median)

        # Pass: image-space separation is sufficient
        if separation >= self.config.min_team_separation:
            return True

        # Hard floor: teams truly overlapping in image space (< 3%)
        if separation < 0.03:
            logger.info(
                f"Court identity skipped: team separation {separation:.3f} "
                f"below hard floor 0.03 (near_y={near_median:.3f}, "
                f"far_y={far_median:.3f})"
            )
            return False

        # Fallback: validate separation in court space using calibrator.
        # Teams on opposite sides of the net are always >4m apart in court
        # space, even when they appear close in image Y due to camera angle.
        if self.calibrator.is_calibrated:
            court_sep = self._check_court_space_separation(
                near_median, far_median
            )
            if court_sep is not None:
                return court_sep

        logger.info(
            f"Court identity skipped: team separation {separation:.3f} "
            f"< {self.config.min_team_separation} (near_y={near_median:.3f}, "
            f"far_y={far_median:.3f})"
        )
        return False

    def _check_court_space_separation(
        self,
        near_median_y: float,
        far_median_y: float,
    ) -> bool | None:
        """Validate team separation using court-space projection.

        Returns True if teams are well-separated in court space,
        False if projections are unreliable or teams overlap,
        None if projection fails.
        """
        try:
            near_court = self.calibrator.image_to_court(
                (0.5, near_median_y), self.video_width, self.video_height
            )
            far_court = self.calibrator.image_to_court(
                (0.5, far_median_y), self.video_width, self.video_height
            )
        except (RuntimeError, ValueError):
            return None

        court_sep = abs(near_court[1] - far_court[1])

        # Reject wildly extrapolated projections (> 3x court length from origin).
        # These indicate the homography is unreliable in this image region.
        max_reasonable = COURT_LENGTH * 3  # 48m
        if abs(near_court[1]) > max_reasonable or abs(far_court[1]) > max_reasonable:
            logger.info(
                f"Court identity: court-space projection unreliable "
                f"(near_court_y={near_court[1]:.1f}, "
                f"far_court_y={far_court[1]:.1f})"
            )
            return False

        # Teams must be well-separated in court space (> 4m = quarter court).
        # On a 16m court, same-side players are at most ~8m apart.
        # This check is convention-agnostic: doesn't require teams to be on
        # opposite sides of the hardcoded NET_Y=8.0m, since many calibrations
        # project both teams to the same side due to corner ordering.
        if court_sep > 4.0:
            logger.info(
                f"Court identity: court-space validation passed "
                f"(image sep={abs(near_median_y - far_median_y):.3f}, "
                f"court sep={court_sep:.1f}m, near_court_y={near_court[1]:.1f}, "
                f"far_court_y={far_court[1]:.1f})"
            )
            return True

        logger.info(
            f"Court identity: court-space validation failed — insufficient "
            f"court separation (near_court_y={near_court[1]:.1f}, "
            f"far_court_y={far_court[1]:.1f}, court sep={court_sep:.1f}m)"
        )
        return False

    def _estimate_effective_net_y(
        self,
        court_tracks: dict[int, CourtTrack],
        team_assignments: dict[int, int],
    ) -> float | None:
        """Estimate the net Y position in court space from actual team positions.

        Many calibrations don't follow the standard convention (near=0, far=16),
        so the hardcoded NET_Y=8.0 is unreliable. Instead, compute the midpoint
        between the two teams' median court_y positions.

        Returns effective net Y, or None if insufficient data.
        """
        near_ys: list[float] = []
        far_ys: list[float] = []

        for tid, ct in court_tracks.items():
            team = team_assignments.get(tid, -1)
            if team < 0 or not ct.positions:
                continue
            median_cy = float(np.median([cy for _, cy in ct.positions.values()]))
            if team == 0:
                near_ys.append(median_cy)
            else:
                far_ys.append(median_cy)

        if not near_ys or not far_ys:
            return None

        near_median = float(np.median(near_ys))
        far_median = float(np.median(far_ys))
        effective_net_y = (near_median + far_median) / 2.0

        logger.debug(
            f"Effective net_y={effective_net_y:.1f}m "
            f"(near={near_median:.1f}m, far={far_median:.1f}m, "
            f"hardcoded NET_Y={NET_Y:.1f}m)"
        )
        return effective_net_y

    def _build_court_tracks(
        self,
        positions: list[PlayerPosition],
    ) -> dict[int, CourtTrack]:
        """Project all player positions to court coordinates."""
        tracks: dict[int, CourtTrack] = {}

        for p in positions:
            if p.track_id < 0:
                continue

            # Use bbox foot-point (bottom-center) for court projection
            foot_x = p.x
            foot_y = p.y + p.height / 2.0

            try:
                court_x, court_y = self.calibrator.image_to_court(
                    (foot_x, foot_y), self.video_width, self.video_height
                )
            except (RuntimeError, ValueError):
                continue

            if p.track_id not in tracks:
                tracks[p.track_id] = CourtTrack(track_id=p.track_id)

            tracks[p.track_id].positions[p.frame_number] = (court_x, court_y)

        return tracks

    def _detect_net_interactions(
        self,
        court_tracks: dict[int, CourtTrack],
        team_assignments: dict[int, int],
        effective_net_y: float | None = None,
    ) -> list[NetInteraction]:
        """Detect windows where cross-team tracks are both near the net."""
        interactions: list[NetInteraction] = []
        net_dist = self.config.net_approach_distance
        net_y = effective_net_y if effective_net_y is not None else NET_Y

        # Get cross-team pairs
        near_tracks = [
            tid for tid, team in team_assignments.items()
            if team == 0 and tid in court_tracks
        ]
        far_tracks = [
            tid for tid, team in team_assignments.items()
            if team == 1 and tid in court_tracks
        ]

        for near_tid in near_tracks:
            for far_tid in far_tracks:
                near_ct = court_tracks[near_tid]
                far_ct = court_tracks[far_tid]

                # Find frames where both are near net
                common_frames = sorted(
                    set(near_ct.positions.keys()) & set(far_ct.positions.keys())
                )

                if not common_frames:
                    continue

                # Find streaks of both-near-net frames
                streak_start: int | None = None
                streak_len = 0

                for frame in common_frames:
                    near_pos = near_ct.positions[frame]
                    far_pos = far_ct.positions[frame]

                    near_at_net = abs(near_pos[1] - net_y) < net_dist
                    far_at_net = abs(far_pos[1] - net_y) < net_dist

                    if near_at_net and far_at_net:
                        if streak_start is None:
                            streak_start = frame
                        streak_len += 1
                    else:
                        if (
                            streak_start is not None
                            and streak_len >= 5  # minimum 5 frames
                        ):
                            interactions.append(NetInteraction(
                                track_a=near_tid,
                                track_b=far_tid,
                                start_frame=streak_start,
                                end_frame=frame - 1,
                            ))
                        streak_start = None
                        streak_len = 0

                # Check remaining streak
                if streak_start is not None and streak_len >= 5:
                    interactions.append(NetInteraction(
                        track_a=near_tid,
                        track_b=far_tid,
                        start_frame=streak_start,
                        end_frame=common_frames[-1],
                    ))

        return interactions

    def _build_image_lookup(
        self,
        positions: list[PlayerPosition],
    ) -> dict[int, dict[int, PlayerPosition]]:
        """Build track_id -> frame -> position lookup."""
        lookup: dict[int, dict[int, PlayerPosition]] = defaultdict(dict)
        for p in positions:
            if p.track_id >= 0:
                lookup[p.track_id][p.frame_number] = p
        return dict(lookup)

    def _score_interaction(
        self,
        interaction: NetInteraction,
        court_tracks: dict[int, CourtTrack],
        image_positions: dict[int, dict[int, PlayerPosition]],
        team_assignments: dict[int, int],
        effective_net_y: float = NET_Y,
        contact_sequence: ContactSequence | None = None,
        serve_anchor: ServeAnchor | None = None,
        color_store: ColorHistogramStore | None = None,
        frozen_interactions: list[ActiveFreeze] | None = None,
    ) -> SwapDecision:
        """Score swap vs no-swap hypotheses for one interaction."""
        cfg = self.config
        track_a = interaction.track_a  # near team
        track_b = interaction.track_b  # far team

        ct_a = court_tracks.get(track_a)
        ct_b = court_tracks.get(track_b)

        # Score both hypotheses
        # no-swap: track_a stays near, track_b stays far
        # swap: track_a becomes far, track_b becomes near
        no_swap = IdentityHypothesis(interaction=interaction)
        swap = IdentityHypothesis(interaction=interaction)

        # --- Component 1: Side-of-net consistency ---
        if ct_a and ct_b:
            no_swap.side_of_net_score, swap.side_of_net_score = (
                self._score_side_of_net(
                    interaction, ct_a, ct_b, team_assignments, effective_net_y
                )
            )

        # --- Component 2: Motion smoothness ---
        if ct_a and ct_b:
            no_swap.motion_smoothness_score, swap.motion_smoothness_score = (
                self._score_motion_smoothness(interaction, ct_a, ct_b)
            )

        # --- Component 3: Bbox size consistency ---
        img_a = image_positions.get(track_a, {})
        img_b = image_positions.get(track_b, {})
        if img_a and img_b:
            no_swap.bbox_size_score, swap.bbox_size_score = (
                self._score_bbox_size(interaction, img_a, img_b, team_assignments)
            )

        # --- Component 4: Color histogram consistency ---
        if color_store is not None and color_store.has_data():
            no_swap.color_score, swap.color_score = (
                self._score_color_consistency(interaction, color_store)
            )

        # --- Component 5: Touch grammar (Phase 3) ---
        if contact_sequence and len(contact_sequence.contacts) >= 2:
            from rallycut.tracking.touch_grammar import score_swap_hypothesis

            no_swap.grammar_score, swap.grammar_score = score_swap_hypothesis(
                contact_sequence, team_assignments,
                track_a, track_b, interaction.end_frame,
            )

        # --- Component 6: Serve anchor consistency (Phase 2) ---
        if serve_anchor and serve_anchor.confidence > 0.5:
            no_swap.serve_anchor_score, swap.serve_anchor_score = (
                self._score_serve_anchor(
                    interaction, serve_anchor, team_assignments
                )
            )

        # Compute weighted totals (weights auto-adjust for available components)
        weight_color = cfg.weight_color if (
            color_store is not None and color_store.has_data()
        ) else 0.0
        weight_grammar = 0.10 if contact_sequence else 0.0
        weight_serve = 0.05 if serve_anchor else 0.0
        # Re-normalize base weights to sum to 1.0
        base_weight_sum = (
            cfg.weight_side_of_net
            + cfg.weight_motion_smoothness
            + cfg.weight_bbox_size
            + weight_color
        )
        total_weight = base_weight_sum + weight_grammar + weight_serve
        scale = 1.0 / total_weight if total_weight > 0 else 1.0

        no_swap_total = scale * (
            cfg.weight_side_of_net * no_swap.side_of_net_score
            + cfg.weight_motion_smoothness * no_swap.motion_smoothness_score
            + cfg.weight_bbox_size * no_swap.bbox_size_score
            + weight_color * no_swap.color_score
            + weight_grammar * no_swap.grammar_score
            + weight_serve * no_swap.serve_anchor_score
        )
        swap_total = scale * (
            cfg.weight_side_of_net * swap.side_of_net_score
            + cfg.weight_motion_smoothness * swap.motion_smoothness_score
            + cfg.weight_bbox_size * swap.bbox_size_score
            + weight_color * swap.color_score
            + weight_grammar * swap.grammar_score
            + weight_serve * swap.serve_anchor_score
        )

        # Freeze metadata bonus: if the in-tracker freeze was active for this
        # interaction, identities were preserved during overlap → small no-swap bonus
        if frozen_interactions:
            freeze_bonus = self._compute_freeze_bonus(
                interaction, frozen_interactions
            )
            if freeze_bonus > 0:
                no_swap_total += freeze_bonus
                logger.debug(
                    f"  Freeze bonus +{freeze_bonus:.2f} for "
                    f"{track_a}<->{track_b}"
                )

        should_swap = swap_total > no_swap_total + cfg.confidence_margin
        confident = abs(swap_total - no_swap_total) >= cfg.confidence_margin

        logger.debug(
            f"  Interaction {track_a}<->{track_b} f{interaction.start_frame}-{interaction.end_frame}: "
            f"side={no_swap.side_of_net_score:.2f}/{swap.side_of_net_score:.2f} "
            f"motion={no_swap.motion_smoothness_score:.2f}/{swap.motion_smoothness_score:.2f} "
            f"bbox={no_swap.bbox_size_score:.2f}/{swap.bbox_size_score:.2f} "
            f"color={no_swap.color_score:.2f}/{swap.color_score:.2f} "
            f"-> no_swap={no_swap_total:.3f} swap={swap_total:.3f} "
            f"{'SWAP' if should_swap else 'keep'}"
        )

        return SwapDecision(
            interaction=interaction,
            should_swap=should_swap,
            swap_score=swap_total,
            no_swap_score=no_swap_total,
            confident=confident,
        )

    def _score_side_of_net(
        self,
        interaction: NetInteraction,
        ct_a: CourtTrack,
        ct_b: CourtTrack,
        team_assignments: dict[int, int],
        effective_net_y: float = NET_Y,
    ) -> tuple[float, float]:
        """Score side-of-net consistency for no-swap vs swap.

        Returns (no_swap_score, swap_score) each in [0, 1].
        """
        cfg = self.config
        end = interaction.end_frame
        window_end = end + cfg.observation_window
        dead_zone = cfg.net_dead_zone

        team_a = team_assignments.get(interaction.track_a, 0)
        team_b = team_assignments.get(interaction.track_b, 1)

        # Collect post-interaction positions (outside dead zone)
        a_positions: list[tuple[float, float]] = []
        b_positions: list[tuple[float, float]] = []

        for frame in range(end + 1, window_end + 1):
            if frame in ct_a.positions:
                pos = ct_a.positions[frame]
                if abs(pos[1] - effective_net_y) > dead_zone:
                    a_positions.append(pos)
            if frame in ct_b.positions:
                pos = ct_b.positions[frame]
                if abs(pos[1] - effective_net_y) > dead_zone:
                    b_positions.append(pos)

        if (
            len(a_positions) < cfg.min_observation_frames
            or len(b_positions) < cfg.min_observation_frames
        ):
            # Not enough data — return neutral scores
            return 0.5, 0.5

        # For no-swap: track_a should be on team_a's side, track_b on team_b's
        # Near team (0) → court_y < effective_net_y
        # Far team (1) → court_y > effective_net_y
        a_correct_noswap = self._fraction_on_correct_side(
            a_positions, team_a, dead_zone, effective_net_y
        )
        b_correct_noswap = self._fraction_on_correct_side(
            b_positions, team_b, dead_zone, effective_net_y
        )
        no_swap_score = (a_correct_noswap + b_correct_noswap) / 2.0

        # For swap: track_a would be team_b's side, track_b would be team_a's
        a_correct_swap = self._fraction_on_correct_side(
            a_positions, team_b, dead_zone, effective_net_y
        )
        b_correct_swap = self._fraction_on_correct_side(
            b_positions, team_a, dead_zone, effective_net_y
        )
        swap_score = (a_correct_swap + b_correct_swap) / 2.0

        return no_swap_score, swap_score

    @staticmethod
    def _fraction_on_correct_side(
        positions: list[tuple[float, float]],
        team: int,
        dead_zone: float,
        net_y: float = NET_Y,
    ) -> float:
        """What fraction of positions are on the expected side of net."""
        if not positions:
            return 0.5

        correct = 0
        total = 0
        for _, cy in positions:
            dist_to_net = cy - net_y
            if abs(dist_to_net) <= dead_zone:
                continue  # Skip positions in dead zone
            total += 1
            if team == 0 and dist_to_net < 0:  # Near team → court_y < net_y
                correct += 1
            elif team == 1 and dist_to_net > 0:  # Far team → court_y > net_y
                correct += 1

        return correct / total if total > 0 else 0.5

    def _score_motion_smoothness(
        self,
        interaction: NetInteraction,
        ct_a: CourtTrack,
        ct_b: CourtTrack,
    ) -> tuple[float, float]:
        """Score motion continuity for no-swap vs swap.

        Compares velocity before interaction to position after.
        Lower displacement = smoother = higher score.

        Returns (no_swap_score, swap_score) each in [0, 1].
        """
        cfg = self.config
        start = interaction.start_frame
        end = interaction.end_frame

        # Get pre-interaction tail positions
        pre_a = self._get_positions_in_range(
            ct_a, start - cfg.pre_interaction_window, start - 1
        )
        pre_b = self._get_positions_in_range(
            ct_b, start - cfg.pre_interaction_window, start - 1
        )

        # Get post-interaction head positions
        post_a = self._get_positions_in_range(
            ct_a, end + 1, end + cfg.pre_interaction_window
        )
        post_b = self._get_positions_in_range(
            ct_b, end + 1, end + cfg.pre_interaction_window
        )

        if (
            len(pre_a) < cfg.min_pre_frames
            or len(pre_b) < cfg.min_pre_frames
            or len(post_a) < cfg.min_pre_frames
            or len(post_b) < cfg.min_pre_frames
        ):
            return 0.5, 0.5

        # Compute mean pre and post positions
        pre_a_mean = self._mean_position(pre_a)
        pre_b_mean = self._mean_position(pre_b)
        post_a_mean = self._mean_position(post_a)
        post_b_mean = self._mean_position(post_b)

        # No-swap: pre_a → post_a, pre_b → post_b
        dist_noswap_a = self._court_distance(pre_a_mean, post_a_mean)
        dist_noswap_b = self._court_distance(pre_b_mean, post_b_mean)
        noswap_displacement = dist_noswap_a + dist_noswap_b

        # Swap: pre_a → post_b, pre_b → post_a
        dist_swap_a = self._court_distance(pre_a_mean, post_b_mean)
        dist_swap_b = self._court_distance(pre_b_mean, post_a_mean)
        swap_displacement = dist_swap_a + dist_swap_b

        # Convert to scores: lower displacement = higher score
        # Normalize by typical court-crossing distance (~16m)
        max_dist = COURT_LENGTH
        no_swap_score = max(0.0, 1.0 - noswap_displacement / max_dist)
        swap_score = max(0.0, 1.0 - swap_displacement / max_dist)

        return no_swap_score, swap_score

    def _score_bbox_size(
        self,
        interaction: NetInteraction,
        img_a: dict[int, PlayerPosition],
        img_b: dict[int, PlayerPosition],
        team_assignments: dict[int, int],
    ) -> tuple[float, float]:
        """Score bbox size consistency.

        Near-team players have larger bboxes (closer to camera).
        If a "near" track suddenly has small bboxes after interaction,
        it's likely swapped.

        Returns (no_swap_score, swap_score) each in [0, 1].
        """
        cfg = self.config
        start = interaction.start_frame
        end = interaction.end_frame

        team_a = team_assignments.get(interaction.track_a, 0)

        # Pre-interaction mean heights
        pre_heights_a = [
            p.height for f, p in img_a.items()
            if start - 30 <= f < start
        ]
        pre_heights_b = [
            p.height for f, p in img_b.items()
            if start - 30 <= f < start
        ]

        # Post-interaction mean heights
        post_heights_a = [
            p.height for f, p in img_a.items()
            if end < f <= end + cfg.observation_window
        ]
        post_heights_b = [
            p.height for f, p in img_b.items()
            if end < f <= end + cfg.observation_window
        ]

        if (
            len(pre_heights_a) < 3
            or len(pre_heights_b) < 3
            or len(post_heights_a) < 3
            or len(post_heights_b) < 3
        ):
            return 0.5, 0.5

        pre_h_a = float(np.mean(pre_heights_a))
        pre_h_b = float(np.mean(pre_heights_b))
        post_h_a = float(np.mean(post_heights_a))
        post_h_b = float(np.mean(post_heights_b))

        # No-swap: consistency of height before/after
        noswap_total_change = abs(post_h_a - pre_h_a) + abs(post_h_b - pre_h_b)

        # Swap: pre_a matches post_b, pre_b matches post_a
        swap_total_change = abs(post_h_b - pre_h_a) + abs(post_h_a - pre_h_b)

        # Also check team-size consistency: near team should have larger bbox
        # This is a soft prior
        near_team = 0
        if team_a == near_team:
            # Track A is near team — should be taller
            noswap_size_ok = post_h_a > post_h_b
            swap_size_ok = post_h_b > post_h_a  # after swap, B becomes near
        else:
            noswap_size_ok = post_h_b > post_h_a
            swap_size_ok = post_h_a > post_h_b

        # Convert change to score (lower change = higher score)
        max_change = 0.2  # normalize: 20% height change is max
        no_swap_score = max(0.0, 1.0 - noswap_total_change / max_change)
        swap_score = max(0.0, 1.0 - swap_total_change / max_change)

        # Apply size-order bonus
        if noswap_size_ok:
            no_swap_score = min(1.0, no_swap_score + 0.1)
        if swap_size_ok:
            swap_score = min(1.0, swap_score + 0.1)

        return no_swap_score, swap_score

    @staticmethod
    def _score_serve_anchor(
        interaction: NetInteraction,
        serve_anchor: ServeAnchor,
        team_assignments: dict[int, int],
    ) -> tuple[float, float]:
        """Score serve anchor consistency for no-swap vs swap.

        Returns (no_swap_score, swap_score) each in [0, 1].
        """
        track_a = interaction.track_a
        track_b = interaction.track_b

        # Check if either track is the server
        server_tid = serve_anchor.server_track_id
        if server_tid not in (track_a, track_b):
            return 0.5, 0.5  # Serve anchor doesn't involve these tracks

        # Server should be on the serving team's side
        server_team = serve_anchor.server_team
        current_team = team_assignments.get(server_tid, -1)

        if current_team == server_team:
            # No-swap keeps server on correct team
            return 0.8, 0.2
        else:
            # Swap would put server back on correct team
            return 0.2, 0.8

    @staticmethod
    def _score_color_consistency(
        interaction: NetInteraction,
        color_store: ColorHistogramStore,
        pre_window: int = 30,
        post_window: int = 30,
        min_histograms: int = 3,
    ) -> tuple[float, float]:
        """Score color histogram consistency for no-swap vs swap.

        Compares pre-interaction color templates with post-interaction histograms
        for both normal and swapped assignments. Uses Bhattacharyya distance.

        Returns (no_swap_score, swap_score) each in [0, 1].
        """
        track_a = interaction.track_a
        track_b = interaction.track_b
        start = interaction.start_frame
        end = interaction.end_frame

        # Get pre-interaction histograms for each track
        pre_a = [
            hist for fn, hist
            in color_store.get_track_histograms(track_a)
            if start - pre_window <= fn < start
        ]
        pre_b = [
            hist for fn, hist
            in color_store.get_track_histograms(track_b)
            if start - pre_window <= fn < start
        ]

        # Get post-interaction histograms for each track
        post_a = [
            hist for fn, hist
            in color_store.get_track_histograms(track_a)
            if end < fn <= end + post_window
        ]
        post_b = [
            hist for fn, hist
            in color_store.get_track_histograms(track_b)
            if end < fn <= end + post_window
        ]

        if (
            len(pre_a) < min_histograms
            or len(pre_b) < min_histograms
            or len(post_a) < min_histograms
            or len(post_b) < min_histograms
        ):
            return 0.5, 0.5

        # Compute mean templates
        template_a = np.mean(pre_a, axis=0).astype(np.float32)
        template_b = np.mean(pre_b, axis=0).astype(np.float32)
        post_mean_a = np.mean(post_a, axis=0).astype(np.float32)
        post_mean_b = np.mean(post_b, axis=0).astype(np.float32)

        # Normalize for Bhattacharyya
        for arr in [template_a, template_b, post_mean_a, post_mean_b]:
            total = arr.sum()
            if total > 0:
                arr /= total

        import cv2

        # No-swap: pre_a matches post_a, pre_b matches post_b
        dist_noswap_a = cv2.compareHist(
            template_a, post_mean_a, cv2.HISTCMP_BHATTACHARYYA
        )
        dist_noswap_b = cv2.compareHist(
            template_b, post_mean_b, cv2.HISTCMP_BHATTACHARYYA
        )
        noswap_dist = (dist_noswap_a + dist_noswap_b) / 2.0

        # Swap: pre_a matches post_b, pre_b matches post_a
        dist_swap_a = cv2.compareHist(
            template_a, post_mean_b, cv2.HISTCMP_BHATTACHARYYA
        )
        dist_swap_b = cv2.compareHist(
            template_b, post_mean_a, cv2.HISTCMP_BHATTACHARYYA
        )
        swap_dist = (dist_swap_a + dist_swap_b) / 2.0

        # Convert Bhattacharyya distance to score (lower = better = higher score)
        # Bhattacharyya range is [0, 1], typical same-person distance: 0.1-0.4
        no_swap_score = max(0.0, 1.0 - noswap_dist)
        swap_score = max(0.0, 1.0 - swap_dist)

        logger.debug(
            f"  Color: noswap_dist={noswap_dist:.3f} swap_dist={swap_dist:.3f} "
            f"-> scores {no_swap_score:.2f}/{swap_score:.2f}"
        )

        return no_swap_score, swap_score

    @staticmethod
    def _compute_freeze_bonus(
        interaction: NetInteraction,
        frozen_interactions: list[ActiveFreeze],
        bonus: float = 0.10,
    ) -> float:
        """Check if the in-tracker freeze was active for this interaction.

        When the freeze was active, BoT-SORT was prevented from swapping,
        so the current assignment is more likely correct → no-swap bonus.

        Returns bonus value if freeze overlaps, 0.0 otherwise.
        """
        for freeze in frozen_interactions:
            # Check if freeze overlaps with this court-space interaction
            a_match = {interaction.track_a, interaction.track_b}
            f_match = {freeze.track_a, freeze.track_b}
            if a_match != f_match:
                continue

            # Check temporal overlap
            if (freeze.start_frame <= interaction.end_frame
                    and freeze.last_overlap_frame >= interaction.start_frame):
                return bonus

        return 0.0

    @staticmethod
    def _get_positions_in_range(
        ct: CourtTrack,
        start: int,
        end: int,
    ) -> list[tuple[float, float]]:
        """Get court positions in a frame range."""
        return [
            ct.positions[f]
            for f in range(start, end + 1)
            if f in ct.positions
        ]

    @staticmethod
    def _mean_position(
        positions: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """Compute mean of court positions."""
        if not positions:
            return (0.0, 0.0)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @staticmethod
    def _court_distance(
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        """Euclidean distance in court meters."""
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    @staticmethod
    def _apply_swap(
        positions: list[PlayerPosition],
        track_a: int,
        track_b: int,
        from_frame: int,
        team_assignments: dict[int, int],
    ) -> None:
        """Swap track IDs from from_frame onward."""
        for p in positions:
            if p.frame_number >= from_frame:
                if p.track_id == track_a:
                    p.track_id = track_b
                elif p.track_id == track_b:
                    p.track_id = track_a

        # Swap team assignments
        team_a = team_assignments.get(track_a)
        team_b = team_assignments.get(track_b)
        if team_a is not None:
            team_assignments[track_b] = team_a
        if team_b is not None:
            team_assignments[track_a] = team_b


def resolve_court_identity(
    positions: list[PlayerPosition],
    team_assignments: dict[int, int],
    calibrator: CourtCalibrator,
    video_width: int = 1920,
    video_height: int = 1080,
    config: CourtIdentityConfig | None = None,
    contact_sequence: ContactSequence | None = None,
    serve_anchor: ServeAnchor | None = None,
    color_store: ColorHistogramStore | None = None,
    frozen_interactions: list[ActiveFreeze] | None = None,
) -> tuple[list[PlayerPosition], int, list[SwapDecision]]:
    """Convenience function for court-plane identity resolution.

    Args:
        positions: Player positions (will be modified if swaps found).
        team_assignments: track_id -> team (0=near, 1=far).
        calibrator: Court calibrator with loaded homography.
        video_width: Video width in pixels.
        video_height: Video height in pixels.
        config: Optional configuration overrides.
        contact_sequence: Optional contacts for grammar scoring.
        serve_anchor: Optional serve anchor for identity anchoring.
        color_store: Optional color histogram store for appearance scoring.
        frozen_interactions: Optional freeze metadata from in-tracker detector.

    Returns:
        Tuple of (positions, num_swaps, decisions).
    """
    if not calibrator.is_calibrated:
        return positions, 0, []

    resolver = CourtIdentityResolver(
        calibrator=calibrator,
        config=config,
        video_width=video_width,
        video_height=video_height,
    )

    positions, decisions = resolver.resolve(
        positions, team_assignments,
        contact_sequence=contact_sequence,
        serve_anchor=serve_anchor,
        color_store=color_store,
        frozen_interactions=frozen_interactions,
    )
    num_swaps = sum(1 for d in decisions if d.should_swap and d.confident)

    return positions, num_swaps, decisions
