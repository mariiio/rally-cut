"""Team-aware BoT-SORT association penalty.

Injects team-side knowledge into BoT-SORT's cost matrix so that cross-team
ID swaps become expensive even when IoU is ambiguous during net interactions.

Players never cross the net during a rally, so team side has very low entropy.
When calibration provides a reliable court_split_y, we add a penalty to
associations that would assign a detection on the wrong side of the net to
a track belonging to the other team.

Safety layers:
1. Calibration trust gate: penalty only activates with reliable calibration
2. Net-margin dead band: no penalty within ±net_band of the split line
3. Team separation gate: penalty disabled if teams aren't separated enough
4. Conservative max penalty: capped at max_penalty (default 0.20)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TeamAwareConfig:
    """Configuration for team-aware BoT-SORT association penalty."""

    enabled: bool = False
    bootstrap_frames: int = 30
    net_band: float = 0.05  # Half-width of dead zone around court_split
    penalty_scale: float = 2.0
    max_penalty: float = 0.20
    min_tracks_for_bootstrap: int = 3
    min_team_separation: float = 0.08  # Minimum median Y gap between teams
    min_clearance_from_split: float = 0.05  # Each team's median must be >= this from split_y

    # Net interaction freeze (Tier 1: preventive identity preservation)
    # Disabled by default — single-frame and short overlaps during normal net
    # play can cause track fragmentation. Needs more work on duration gating
    # and scoring before enabling. See retrack_labeled_rallies.py results.
    enable_freeze: bool = False
    freeze_proximity: float = 0.08         # Center distance < this triggers freeze (normalized)
    freeze_net_band: float = 0.10          # Both tracks must be within this of split_y
    freeze_penalty: float = 1e6            # Hard barrier in cost matrix
    freeze_cooldown_frames: int = 10       # Keep freeze N frames after overlap ends
    freeze_min_duration: int = 3           # Min consecutive overlap frames before penalty fires
    freeze_min_team_separation: float = 0.05   # Relaxed vs 0.08 for soft penalty
    freeze_min_clearance: float = 0.03         # Relaxed vs 0.05 for soft penalty


@dataclass
class ActiveFreeze:
    """A currently active identity freeze between two cross-team tracks."""

    track_a: int          # near team track
    track_b: int          # far team track
    start_frame: int
    last_overlap_frame: int


class NetInteractionDetector:
    """Detects overlapping cross-team pairs near the net and manages freeze lifecycle.

    Operates in image space on per-frame bounding boxes. When a cross-team pair
    overlaps near the net, their identities are frozen (hard barrier in cost matrix)
    to prevent BoT-SORT from swapping them during the unobservable window.
    """

    def __init__(
        self,
        config: TeamAwareConfig,
        court_split_y: float,
    ) -> None:
        self.config = config
        self.court_split_y = court_split_y
        self._active_freezes: dict[tuple[int, int], ActiveFreeze] = {}
        self._completed_freezes: list[ActiveFreeze] = []

    def update(
        self,
        positions: list[Any],
        team_assignments: dict[int, int],
        frame_num: int,
    ) -> None:
        """Detect overlapping cross-team pairs near net, manage freeze lifecycle.

        Args:
            positions: List of objects with track_id, x, y, width, height attrs.
            team_assignments: track_id -> team (0=near, 1=far).
            frame_num: Current frame number.
        """
        cfg = self.config

        # Build position lookup for this frame
        pos_by_id: dict[int, Any] = {}
        for p in positions:
            if p.track_id >= 0:
                pos_by_id[p.track_id] = p

        # Find cross-team pairs that overlap near the net
        overlapping_pairs: set[tuple[int, int]] = set()
        near_tracks = [
            tid for tid, team in team_assignments.items()
            if team == 0 and tid in pos_by_id
        ]
        far_tracks = [
            tid for tid, team in team_assignments.items()
            if team == 1 and tid in pos_by_id
        ]

        for near_tid in near_tracks:
            p_near = pos_by_id[near_tid]
            for far_tid in far_tracks:
                p_far = pos_by_id[far_tid]

                # Check if both are near the net
                near_at_net = abs(p_near.y - self.court_split_y) < cfg.freeze_net_band
                far_at_net = abs(p_far.y - self.court_split_y) < cfg.freeze_net_band
                if not (near_at_net and far_at_net):
                    continue

                # Check overlap: IoU > 0 or center distance < proximity threshold
                if self._check_overlap(p_near, p_far):
                    pair_key = (min(near_tid, far_tid), max(near_tid, far_tid))
                    overlapping_pairs.add(pair_key)

        # Update active freezes
        expired_keys: list[tuple[int, int]] = []
        for key, freeze in self._active_freezes.items():
            if key in overlapping_pairs:
                freeze.last_overlap_frame = frame_num
            elif frame_num - freeze.last_overlap_frame > cfg.freeze_cooldown_frames:
                expired_keys.append(key)

        # Expire old freezes
        for key in expired_keys:
            freeze = self._active_freezes.pop(key)
            self._completed_freezes.append(freeze)
            logger.debug(
                "Freeze expired: %d<->%d (f%d-f%d, cooldown at f%d)",
                freeze.track_a, freeze.track_b,
                freeze.start_frame, freeze.last_overlap_frame, frame_num,
            )

        # Activate new freezes
        for pair_key in overlapping_pairs:
            if pair_key not in self._active_freezes:
                # Determine which is near/far
                tid_a, tid_b = pair_key
                team_a = team_assignments.get(tid_a, -1)
                near_tid = tid_a if team_a == 0 else tid_b
                far_tid = tid_b if team_a == 0 else tid_a

                self._active_freezes[pair_key] = ActiveFreeze(
                    track_a=near_tid,
                    track_b=far_tid,
                    start_frame=frame_num,
                    last_overlap_frame=frame_num,
                )
                logger.debug(
                    "Freeze activated: %d<->%d at frame %d",
                    near_tid, far_tid, frame_num,
                )

    def _check_overlap(self, p_a: Any, p_b: Any) -> bool:
        """Check if two positions overlap (IoU > 0 or centers close)."""
        # Center distance check (fast)
        dx = p_a.x - p_b.x
        dy = p_a.y - p_b.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < self.config.freeze_proximity:
            return True

        # IoU check
        a_x1 = p_a.x - p_a.width / 2
        a_y1 = p_a.y - p_a.height / 2
        a_x2 = p_a.x + p_a.width / 2
        a_y2 = p_a.y + p_a.height / 2

        b_x1 = p_b.x - p_b.width / 2
        b_y1 = p_b.y - p_b.height / 2
        b_x2 = p_b.x + p_b.width / 2
        b_y2 = p_b.y + p_b.height / 2

        inter_x1 = max(a_x1, b_x1)
        inter_y1 = max(a_y1, b_y1)
        inter_x2 = min(a_x2, b_x2)
        inter_y2 = min(a_y2, b_y2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            return True  # Any overlap

        return False

    def get_frozen_pairs(self) -> set[tuple[int, int]]:
        """Return currently frozen pairs that have met the minimum duration gate.

        Only returns pairs that have been overlapping for >= freeze_min_duration
        frames. This prevents single-frame bbox overlaps from injecting 1e6
        cost spikes into BoT-SORT, which would fragment tracks and create IDsw.
        """
        min_dur = self.config.freeze_min_duration
        return {
            (f.track_a, f.track_b)
            for f in self._active_freezes.values()
            if (f.last_overlap_frame - f.start_frame) >= min_dur - 1
        }

    def get_all_frozen_interactions(self) -> list[ActiveFreeze]:
        """Return all interactions that were frozen (completed + active)."""
        return self._completed_freezes + list(self._active_freezes.values())


class TeamSideTracker:
    """Tracks which team each track ID belongs to based on Y position.

    After bootstrap_frames, assigns teams using court_split_y from calibration.
    Near team: median Y > court_split_y (closer to camera, bottom of frame).
    Far team: median Y <= court_split_y (top of frame).
    """

    def __init__(
        self,
        court_split_y: float,
        config: TeamAwareConfig | None = None,
    ) -> None:
        self.court_split_y = court_split_y
        self.config = config or TeamAwareConfig(enabled=True)

        # Track Y position history: track_id -> list of normalized Y positions
        self._track_ys: dict[int, list[float]] = {}
        # Team assignments: track_id -> 0 (near) or 1 (far)
        self._team: dict[int, int] = {}
        self._frames_seen = 0
        self._bootstrapped = False
        self._active = False  # Set to True after bootstrap + trust gate passes

        # Net interaction freeze detector (created after bootstrap succeeds)
        self._interaction_detector: NetInteractionDetector | None = None
        self._freeze_active = False  # Freeze can activate with weaker gates

    @property
    def is_active(self) -> bool:
        """Whether the penalty is currently being applied."""
        return self._active

    @property
    def team_assignments(self) -> dict[int, int]:
        """Current team assignments (track_id -> 0=near, 1=far)."""
        return dict(self._team)

    @property
    def interaction_detector(self) -> NetInteractionDetector | None:
        """Access the net interaction detector (available after bootstrap)."""
        return self._interaction_detector

    def update(self, positions: list[Any]) -> None:
        """Feed frame positions to accumulate Y statistics.

        Args:
            positions: List of objects with track_id and y attributes
                (PlayerPosition or similar).
        """
        self._frames_seen += 1

        for p in positions:
            tid = p.track_id
            if tid < 0:
                continue
            if tid not in self._track_ys:
                self._track_ys[tid] = []
            self._track_ys[tid].append(p.y)

        if not self._bootstrapped and self._frames_seen >= self.config.bootstrap_frames:
            self._bootstrap()

        # Assign new tracks appearing after bootstrap
        if self._bootstrapped:
            for p in positions:
                tid = p.track_id
                if tid >= 0 and tid not in self._team and tid in self._track_ys:
                    self._assign_track(tid)

    def _bootstrap(self) -> None:
        """Assign teams based on accumulated Y positions."""
        self._bootstrapped = True

        # Need enough tracks
        qualified = {
            tid: ys for tid, ys in self._track_ys.items()
            if len(ys) >= 10
        }
        if len(qualified) < self.config.min_tracks_for_bootstrap:
            logger.debug(
                "Team-aware: only %d qualified tracks (need %d), staying inactive",
                len(qualified), self.config.min_tracks_for_bootstrap,
            )
            return

        # Assign teams by median Y
        for tid, ys in qualified.items():
            self._assign_track(tid)

        # Check team separation trust gate (strict for soft penalty)
        strict_pass = self._check_separation()

        # Check relaxed gates for freeze only
        relaxed_pass = False
        if not strict_pass and self.config.enable_freeze:
            relaxed_pass = self._check_separation(
                min_separation=self.config.freeze_min_team_separation,
                min_clearance=self.config.freeze_min_clearance,
            )

        if not strict_pass and not relaxed_pass:
            logger.debug("Team-aware: insufficient team separation, staying inactive")
            return

        if strict_pass:
            self._active = True

        # Create freeze detector if enabled (uses relaxed or strict gates)
        if self.config.enable_freeze and (strict_pass or relaxed_pass):
            self._interaction_detector = NetInteractionDetector(
                self.config, self.court_split_y
            )
            self._freeze_active = True
            logger.info(
                "Net interaction freeze enabled (strict_penalty=%s)",
                strict_pass,
            )

        near = [t for t, team in self._team.items() if team == 0]
        far = [t for t, team in self._team.items() if team == 1]
        logger.info(
            "Team-aware tracking active: split_y=%.3f, near=%s, far=%s, "
            "soft_penalty=%s, freeze=%s",
            self.court_split_y, near, far, self._active, self._freeze_active,
        )

    def _assign_track(self, track_id: int) -> None:
        """Assign a single track to near (0) or far (1) team."""
        ys = self._track_ys.get(track_id, [])
        if not ys:
            return
        median_y = float(np.median(ys))
        # Near team: y > split (bottom of frame, closer to camera)
        self._team[track_id] = 0 if median_y > self.court_split_y else 1

    def _check_separation(
        self,
        min_separation: float | None = None,
        min_clearance: float | None = None,
    ) -> bool:
        """Check if near and far teams are sufficiently separated in Y.

        Args:
            min_separation: Override minimum team separation (default: config value).
            min_clearance: Override minimum clearance from split (default: config value).
        """
        if min_separation is None:
            min_separation = self.config.min_team_separation
        if min_clearance is None:
            min_clearance = self.config.min_clearance_from_split

        near_ys: list[float] = []
        far_ys: list[float] = []
        for tid, team in self._team.items():
            ys = self._track_ys.get(tid, [])
            if not ys:
                continue
            median_y = float(np.median(ys))
            if team == 0:
                near_ys.append(median_y)
            else:
                far_ys.append(median_y)

        if not near_ys or not far_ys:
            return False

        near_median = float(np.median(near_ys))
        far_median = float(np.median(far_ys))
        separation = near_median - far_median

        logger.debug(
            "Team separation: near_median=%.3f, far_median=%.3f, gap=%.3f (min=%.3f)",
            near_median, far_median, separation, min_separation,
        )
        if separation < min_separation:
            return False

        # Per-team clearance: each team's median must be safely away from the
        # split line. Without this, cameras with extreme perspective compression
        # (far court squeezed to <10% of frame) pass the inter-team gap check
        # but place one team so close to the split that normal play enters the
        # penalty ramp zone, causing track fragmentation.
        near_clearance = near_median - self.court_split_y
        far_clearance = self.court_split_y - far_median

        if near_clearance < min_clearance:
            logger.debug(
                "Team-aware: near team too close to split "
                "(clearance=%.3f < %.3f), staying inactive",
                near_clearance, min_clearance,
            )
            return False
        if far_clearance < min_clearance:
            logger.debug(
                "Team-aware: far team too close to split "
                "(clearance=%.3f < %.3f), staying inactive",
                far_clearance, min_clearance,
            )
            return False

        return True

    def compute_penalty_matrix(
        self,
        tracks: list[Any],
        detections: list[Any],
        img_height: int,
    ) -> np.ndarray:
        """Compute team-crossing penalty matrix for BoT-SORT cost fusion.

        Combines the soft ramp penalty (for general team-side enforcement)
        with the hard freeze penalty (for specific overlapping pairs at net).

        Args:
            tracks: List of STrack objects from BoT-SORT.
            detections: List of STrack objects (detections).
            img_height: Image height in pixels (for normalization).

        Returns:
            (n_tracks, n_dets) penalty matrix. Zero when no penalty applies.
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        penalty = np.zeros((n_tracks, n_dets), dtype=np.float64)

        if n_tracks == 0 or n_dets == 0:
            return penalty

        # Soft ramp penalty (requires full activation with strict gates)
        if self._active:
            for i, track in enumerate(tracks):
                tid = track.track_id
                if tid not in self._team:
                    continue
                team = self._team[tid]

                for j, det in enumerate(detections):
                    tlwh = det.tlwh
                    det_cy = (tlwh[1] + tlwh[3] / 2) / img_height

                    if team == 0:
                        signed_dist = self.court_split_y - det_cy
                    else:
                        signed_dist = det_cy - self.court_split_y

                    penetration = max(0.0, signed_dist - self.config.net_band)

                    penalty[i, j] = min(
                        penetration * self.config.penalty_scale,
                        self.config.max_penalty,
                    )

        # Hard freeze penalty overlay (uses relaxed gates, targets specific pairs)
        if self._interaction_detector is not None:
            frozen_pairs = self._interaction_detector.get_frozen_pairs()
            if frozen_pairs:
                self._apply_freeze_penalty(
                    penalty, tracks, detections, img_height, frozen_pairs
                )

        return penalty

    def _apply_freeze_penalty(
        self,
        penalty: np.ndarray,
        tracks: list[Any],
        detections: list[Any],
        img_height: int,
        frozen_pairs: set[tuple[int, int]],
    ) -> None:
        """Overlay hard freeze penalty for active net interaction pairs.

        For frozen pairs, penalizes associations that would swap identities:
        if track A (near team) is frozen with track B (far team), penalize
        A being matched to detections on B's side and vice versa.
        """
        freeze_pen = self.config.freeze_penalty

        for i, track in enumerate(tracks):
            tid = track.track_id
            # Check if this track is in any frozen pair
            partner_tid = None
            for fa, fb in frozen_pairs:
                if tid == fa:
                    partner_tid = fb
                    break
                elif tid == fb:
                    partner_tid = fa
                    break

            if partner_tid is None:
                continue

            partner_team = self._team.get(partner_tid, -1)
            if partner_team < 0:
                continue

            for j, det in enumerate(detections):
                tlwh = det.tlwh
                det_cy = (tlwh[1] + tlwh[3] / 2) / img_height

                # Penalize this track matching detections on the partner's side
                if partner_team == 0 and det_cy > self.court_split_y:
                    # Partner is near team → detection on near side is partner's
                    penalty[i, j] = max(penalty[i, j], freeze_pen)
                elif partner_team == 1 and det_cy < self.court_split_y:
                    # Partner is far team → detection on far side is partner's
                    penalty[i, j] = max(penalty[i, j], freeze_pen)


def get_court_split_y_from_calibration(
    calibrator: Any,
) -> float | None:
    """Get the net line Y position in normalized image coordinates.

    Projects the net center point (court midpoint) to image space.

    Args:
        calibrator: CourtCalibrator instance.

    Returns:
        Normalized Y coordinate of the net line (0-1), or None on failure.
    """
    from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH

    net_court_x = COURT_WIDTH / 2  # Center of net
    net_court_y = COURT_LENGTH / 2  # Net is at midpoint (8m)

    try:
        # court_to_image returns normalized coords when img dims are (1, 1)
        _, net_y = calibrator.court_to_image(
            (net_court_x, net_court_y), 1, 1
        )
        if 0.1 < net_y < 0.9:
            return float(net_y)
        logger.warning(
            "Net Y projection out of reasonable range: %.3f", net_y
        )
        return None
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        logger.warning("Failed to project net to image: %s", e)
        return None


def patch_tracker_with_team_awareness(
    botsort_instance: Any,
    team_tracker: TeamSideTracker,
    img_height: int,
) -> None:
    """Monkey-patch BoT-SORT's get_dists to add team crossing penalty.

    Stores original get_dists on the instance and wraps it with penalty
    addition. Safe: if the original method doesn't exist, does nothing.

    Args:
        botsort_instance: A BOTSORT tracker instance from ultralytics.
        team_tracker: TeamSideTracker with team assignments.
        img_height: Image height in pixels for Y normalization.
    """
    if not hasattr(botsort_instance, "get_dists"):
        logger.warning("BoT-SORT instance has no get_dists method, skipping patch")
        return

    original_get_dists = botsort_instance.get_dists

    def patched_get_dists(
        tracks: list[Any], detections: list[Any]
    ) -> Any:
        dists = original_get_dists(tracks, detections)
        penalty = team_tracker.compute_penalty_matrix(
            tracks, detections, img_height
        )
        return dists + penalty

    botsort_instance.get_dists = patched_get_dists
    logger.debug("Patched BoT-SORT get_dists with team-aware penalty")
