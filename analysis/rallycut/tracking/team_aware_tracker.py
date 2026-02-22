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
from dataclasses import dataclass
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

    @property
    def is_active(self) -> bool:
        """Whether the penalty is currently being applied."""
        return self._active

    @property
    def team_assignments(self) -> dict[int, int]:
        """Current team assignments (track_id -> 0=near, 1=far)."""
        return dict(self._team)

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

        # Check team separation trust gate
        if not self._check_separation():
            logger.debug("Team-aware: insufficient team separation, staying inactive")
            return

        self._active = True

        near = [t for t, team in self._team.items() if team == 0]
        far = [t for t, team in self._team.items() if team == 1]
        logger.info(
            "Team-aware tracking active: split_y=%.3f, near=%s, far=%s",
            self.court_split_y, near, far,
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

        Applies a soft ramp penalty for general team-side enforcement.

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

        return penalty


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
