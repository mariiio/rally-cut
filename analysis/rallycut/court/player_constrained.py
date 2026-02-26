"""Player-constrained court homography optimization.

Uses tracked player positions as geometric constraints to estimate the court
homography when line detection fails. Key insight: near-side players constrain
near corners, which line detection cannot see (near baseline is often off-screen).

Primary use case (estimate_from_players): when no lines are detected, build an
initial court estimate from player spread and team separation, then optimize.

Secondary use case (refine_corners): refine existing corners — currently disabled
in production because team assignments are often inaccurate, causing regressions.
Available for eval comparison via scripts/eval_court_detection.py --compare.

Cost function terms:
- Side constraint: near-team → court y ∈ [0, 8], far-team → y ∈ [8, 16]
- Bounds: all players within court ± margin
- Spread: within-team pairwise distance 1-7m
- Geometric: convex, perspective-correct (near wider than far)
- Line evidence: stay near initial corner positions
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH
from rallycut.court.line_geometry import COURT_MODEL_CORNERS

logger = logging.getLogger(__name__)


@dataclass
class PlayerConstrainedConfig:
    """Configuration for player-constrained optimization."""

    # Court margins for player positions (meters)
    sideline_margin: float = 2.0  # players can be up to 2m outside sidelines
    baseline_margin: float = 4.0  # players can be up to 4m behind baselines

    # Within-team spread constraints (meters)
    min_team_spread: float = 1.0  # players at least 1m apart
    max_team_spread: float = 7.0  # players at most 7m apart

    # Player sampling
    max_player_samples: int = 500  # max foot positions to use
    min_player_samples: int = 20  # need at least this many

    # Optimization
    max_iterations: int = 200
    near_corner_search_range: float = 0.5  # search ±50% of frame around initial

    # Cost weights
    weight_side_constraint: float = 10.0  # near-team on near side, far-team on far side
    weight_bounds_constraint: float = 5.0  # all players within court + margin
    weight_spread_constraint: float = 2.0  # within-team spread
    weight_geometric_reg: float = 3.0  # convex quad, perspective-correct
    weight_line_evidence: float = 5.0  # stay near line-detected corners


@dataclass
class PlayerFootPosition:
    """A player foot position with team assignment."""

    x: float  # normalized image x (0-1)
    y: float  # normalized image y (0-1), foot = bbox bottom
    team: int  # 0=near, 1=far


class PlayerConstrainedOptimizer:
    """Optimize court corners using player position constraints."""

    def __init__(self, config: PlayerConstrainedConfig | None = None) -> None:
        self.config = config or PlayerConstrainedConfig()

    def refine_corners(
        self,
        initial_corners: list[dict[str, float]],
        player_feet: list[PlayerFootPosition],
        fix_far_corners: bool = True,
    ) -> list[dict[str, float]] | None:
        """Refine court corners using player positions.

        Args:
            initial_corners: 4 corners [near-left, near-right, far-right, far-left]
                in normalized image coords.
            player_feet: Player foot positions with team assignments.
            fix_far_corners: If True, only optimize near corners (far are well-detected).

        Returns:
            Refined corners in same format, or None if optimization failed.
        """
        cfg = self.config

        if len(player_feet) < cfg.min_player_samples:
            logger.info(
                f"Player-constrained: insufficient samples "
                f"({len(player_feet)} < {cfg.min_player_samples})"
            )
            return None

        if len(initial_corners) != 4:
            return None

        # Sample if too many
        feet = player_feet
        if len(feet) > cfg.max_player_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(feet), cfg.max_player_samples, replace=False)
            feet = [player_feet[i] for i in indices]

        # Separate by team
        near_feet = [(f.x, f.y) for f in feet if f.team == 0]
        far_feet = [(f.x, f.y) for f in feet if f.team == 1]

        if len(near_feet) < 5 or len(far_feet) < 5:
            logger.info(
                f"Player-constrained: insufficient per-team samples "
                f"(near={len(near_feet)}, far={len(far_feet)})"
            )
            return None

        near_feet_arr = np.array(near_feet, dtype=np.float64)
        far_feet_arr = np.array(far_feet, dtype=np.float64)

        # Extract initial corners
        nl = (initial_corners[0]["x"], initial_corners[0]["y"])
        nr = (initial_corners[1]["x"], initial_corners[1]["y"])
        fr = (initial_corners[2]["x"], initial_corners[2]["y"])
        fl = (initial_corners[3]["x"], initial_corners[3]["y"])

        if fix_far_corners:
            # Optimize only near corners (4 params: nl_x, nl_y, nr_x, nr_y)
            x0 = np.array([nl[0], nl[1], nr[0], nr[1]], dtype=np.float64)

            # Bounds: near corners within search range of initial
            r = cfg.near_corner_search_range
            bounds = [
                (nl[0] - r, nl[0] + r),  # nl_x
                (nl[1] - r, nl[1] + r),  # nl_y
                (nr[0] - r, nr[0] + r),  # nr_x
                (nr[1] - r, nr[1] + r),  # nr_y
            ]

            def cost_fn(params: np.ndarray) -> float:
                nl_opt = (params[0], params[1])
                nr_opt = (params[2], params[3])
                return self._compute_cost(
                    nl_opt, nr_opt, fr, fl,
                    near_feet_arr, far_feet_arr,
                    nl, nr,  # initial for line evidence
                )

        else:
            # Optimize all 4 corners (8 params)
            x0 = np.array([
                nl[0], nl[1], nr[0], nr[1],
                fr[0], fr[1], fl[0], fl[1],
            ], dtype=np.float64)

            r = cfg.near_corner_search_range
            bounds = [
                (nl[0] - r, nl[0] + r),
                (nl[1] - r, nl[1] + r),
                (nr[0] - r, nr[0] + r),
                (nr[1] - r, nr[1] + r),
                (fr[0] - r, fr[0] + r),
                (fr[1] - r, fr[1] + r),
                (fl[0] - r, fl[0] + r),
                (fl[1] - r, fl[1] + r),
            ]

            def cost_fn(params: np.ndarray) -> float:
                nl_opt = (params[0], params[1])
                nr_opt = (params[2], params[3])
                fr_opt = (params[4], params[5])
                fl_opt = (params[6], params[7])
                return self._compute_cost(
                    nl_opt, nr_opt, fr_opt, fl_opt,
                    near_feet_arr, far_feet_arr,
                    nl, nr,
                )

        initial_cost = cost_fn(x0)

        result = minimize(
            cost_fn, x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": cfg.max_iterations, "ftol": 1e-8},
        )

        if not result.success and result.fun > initial_cost:
            logger.info(
                f"Player-constrained: optimization did not improve "
                f"(initial={initial_cost:.3f}, final={result.fun:.3f})"
            )
            return None

        # Extract optimized corners
        if fix_far_corners:
            opt_nl = (result.x[0], result.x[1])
            opt_nr = (result.x[2], result.x[3])
            opt_fr = fr
            opt_fl = fl
        else:
            opt_nl = (result.x[0], result.x[1])
            opt_nr = (result.x[2], result.x[3])
            opt_fr = (result.x[4], result.x[5])
            opt_fl = (result.x[6], result.x[7])

        # Validate: quad must be convex and perspective-correct
        if not self._is_valid_court_quad(opt_nl, opt_nr, opt_fr, opt_fl):
            logger.info("Player-constrained: optimized quad is invalid")
            return None

        # Check improvement: project players and verify team separation
        H = self._corners_to_homography(opt_nl, opt_nr, opt_fr, opt_fl)  # noqa: N806
        if H is None:
            return None

        near_correct, far_correct, total = self._evaluate_team_separation(
            H, near_feet_arr, far_feet_arr,
        )
        accuracy = (near_correct + far_correct) / max(1, total)

        if accuracy < 0.6:
            logger.info(
                f"Player-constrained: poor team separation "
                f"({accuracy:.1%}, near={near_correct}, far={far_correct})"
            )
            return None

        logger.info(
            f"Player-constrained: optimized with {accuracy:.1%} team separation "
            f"(near={near_correct}/{len(near_feet_arr)}, "
            f"far={far_correct}/{len(far_feet_arr)}, cost={result.fun:.3f})"
        )

        return [
            {"x": opt_nl[0], "y": opt_nl[1]},
            {"x": opt_nr[0], "y": opt_nr[1]},
            {"x": opt_fr[0], "y": opt_fr[1]},
            {"x": opt_fl[0], "y": opt_fl[1]},
        ]

    def estimate_from_players(
        self,
        player_feet: list[PlayerFootPosition],
        net_y_image: float,
    ) -> list[dict[str, float]] | None:
        """Estimate court corners purely from player positions.

        For videos where line detection fails completely. Uses:
        - net_y from team separation
        - far-team player X spread (compressed by perspective)
        - geometric priors (8x16m court)

        Args:
            player_feet: Player foot positions with team assignments.
            net_y_image: Y coordinate of the net in normalized image space.

        Returns:
            Estimated corners, or None if insufficient data.
        """
        cfg = self.config

        if len(player_feet) < cfg.min_player_samples:
            return None

        near_feet = np.array(
            [(f.x, f.y) for f in player_feet if f.team == 0],
            dtype=np.float64,
        )
        far_feet = np.array(
            [(f.x, f.y) for f in player_feet if f.team == 1],
            dtype=np.float64,
        )

        if len(near_feet) < 5 or len(far_feet) < 5:
            return None

        # Estimate far corners from far-team player spread
        far_x_min = float(np.percentile(far_feet[:, 0], 5))
        far_x_max = float(np.percentile(far_feet[:, 0], 95))
        far_y = float(np.percentile(far_feet[:, 1], 10))  # far side = top of frame

        # Far-team spread in image gives sideline positions
        # Add margin: players are typically 0-1m inside sidelines
        far_margin_x = (far_x_max - far_x_min) * 0.15
        fl_x = far_x_min - far_margin_x
        fr_x = far_x_max + far_margin_x

        # Far Y: slightly above the highest far-team player
        far_margin_y = (net_y_image - far_y) * 0.3
        far_y_corner = max(0.0, far_y - far_margin_y)

        # Near corners: use perspective expansion
        # In perspective, near side is wider than far side
        near_x_min = float(np.percentile(near_feet[:, 0], 5))
        near_x_max = float(np.percentile(near_feet[:, 0], 95))
        near_y = float(np.percentile(near_feet[:, 1], 90))

        near_margin_x = (near_x_max - near_x_min) * 0.15
        nl_x = near_x_min - near_margin_x
        nr_x = near_x_max + near_margin_x

        near_margin_y = (near_y - net_y_image) * 0.3
        near_y_corner = min(1.5, near_y + near_margin_y)  # can be off-screen

        # Initial estimate
        initial_corners = [
            {"x": nl_x, "y": near_y_corner},
            {"x": nr_x, "y": near_y_corner},
            {"x": fr_x, "y": far_y_corner},
            {"x": fl_x, "y": far_y_corner},
        ]

        # Refine with optimizer (optimize all 4 corners)
        refined = self.refine_corners(
            initial_corners, player_feet, fix_far_corners=False,
        )

        if refined is not None:
            logger.info("Player-only estimation: refined with optimizer")
            return refined

        # Fallback: return the geometric estimate
        logger.info(
            "Player-only estimation: returning geometric estimate "
            "(optimizer did not improve)"
        )
        if self._is_valid_court_quad(
            (nl_x, near_y_corner), (nr_x, near_y_corner),
            (fr_x, far_y_corner), (fl_x, far_y_corner),
        ):
            return initial_corners

        return None

    def _compute_cost(
        self,
        nl: tuple[float, float],
        nr: tuple[float, float],
        fr: tuple[float, float],
        fl: tuple[float, float],
        near_feet: np.ndarray,
        far_feet: np.ndarray,
        initial_nl: tuple[float, float],
        initial_nr: tuple[float, float],
    ) -> float:
        """Compute total cost for a set of corners."""
        cfg = self.config

        # Compute homography from corners
        H = self._corners_to_homography(nl, nr, fr, fl)  # noqa: N806
        if H is None:
            return 1e6

        cost = 0.0

        # 1. Side constraint: players should be on correct team half
        cost += cfg.weight_side_constraint * self._side_constraint_cost(
            H, near_feet, far_feet,
        )

        # 2. Bounds constraint: all players within court + margin
        cost += cfg.weight_bounds_constraint * self._bounds_constraint_cost(
            H, near_feet, far_feet,
        )

        # 3. Spread constraint: within-team spread should be reasonable
        cost += cfg.weight_spread_constraint * self._spread_constraint_cost(
            H, near_feet, far_feet,
        )

        # 4. Geometric regularization: convex, perspective-correct
        cost += cfg.weight_geometric_reg * self._geometric_reg_cost(
            nl, nr, fr, fl,
        )

        # 5. Line evidence: stay near initial corners
        cost += cfg.weight_line_evidence * self._line_evidence_cost(
            nl, nr, initial_nl, initial_nr,
        )

        return cost

    def _side_constraint_cost(
        self,
        H: np.ndarray,  # noqa: N803
        near_feet: np.ndarray,
        far_feet: np.ndarray,
    ) -> float:
        """Cost for players being on the wrong side of the net.

        Convention (matching classify_teams in player_filter.py):
        - team=0 (near): higher image-Y (closer to camera) → court y ∈ [0, 8]
        - team=1 (far):  lower image-Y  (farther from camera) → court y ∈ [8, 16]

        Court model: COURT_MODEL_CORNERS maps near-left to (0,0), far-left to (0,16).
        """
        net_y = COURT_LENGTH / 2.0  # 8.0m

        cost = 0.0
        n_total = len(near_feet) + len(far_feet)
        if n_total == 0:
            return 0.0

        # Near team: penalize if y > 8 (on far side)
        if len(near_feet) > 0:
            near_court = self._project_points(H, near_feet)
            violations = np.maximum(0.0, near_court[:, 1] - net_y)
            cost += float(np.mean(violations ** 2))

        # Far team: penalize if y < 8 (on near side)
        if len(far_feet) > 0:
            far_court = self._project_points(H, far_feet)
            violations = np.maximum(0.0, net_y - far_court[:, 1])
            cost += float(np.mean(violations ** 2))

        return cost

    def _bounds_constraint_cost(
        self,
        H: np.ndarray,  # noqa: N803
        near_feet: np.ndarray,
        far_feet: np.ndarray,
    ) -> float:
        """Cost for players being too far outside the court."""
        cfg = self.config
        all_feet = np.vstack([near_feet, far_feet])
        court_pts = self._project_points(H, all_feet)

        cost = 0.0
        # X bounds: [-margin, 8+margin]
        x_low = -cfg.sideline_margin
        x_high = COURT_WIDTH + cfg.sideline_margin
        x_violations = np.maximum(0.0, x_low - court_pts[:, 0]) + np.maximum(
            0.0, court_pts[:, 0] - x_high,
        )
        cost += float(np.mean(x_violations ** 2))

        # Y bounds: [-margin, 16+margin]
        y_low = -cfg.baseline_margin
        y_high = COURT_LENGTH + cfg.baseline_margin
        y_violations = np.maximum(0.0, y_low - court_pts[:, 1]) + np.maximum(
            0.0, court_pts[:, 1] - y_high,
        )
        cost += float(np.mean(y_violations ** 2))

        return cost

    def _spread_constraint_cost(
        self,
        H: np.ndarray,  # noqa: N803
        near_feet: np.ndarray,
        far_feet: np.ndarray,
    ) -> float:
        """Cost for within-team player spread being unreasonable."""
        cfg = self.config
        cost = 0.0

        for feet in [near_feet, far_feet]:
            if len(feet) < 2:
                continue
            court_pts = self._project_points(H, feet)

            # Sample to avoid O(N²) on large sets; note this uses cross-frame
            # pairwise distances (not per-frame), which biases toward zero when
            # players stand still. Kept at low weight to limit impact.
            if len(court_pts) > 50:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(court_pts), 50, replace=False)
                court_pts = court_pts[idx]

            dists = np.sqrt(
                np.sum((court_pts[:, np.newaxis] - court_pts[np.newaxis, :]) ** 2, axis=2)
            )
            # Only upper triangle (avoid double-counting)
            upper_idx = np.triu_indices(len(court_pts), k=1)
            pair_dists = dists[upper_idx]

            if len(pair_dists) == 0:
                continue

            # Penalize distances outside [min, max]
            too_close = np.maximum(0.0, cfg.min_team_spread - pair_dists)
            too_far = np.maximum(0.0, pair_dists - cfg.max_team_spread)
            cost += float(np.mean(too_close ** 2) + np.mean(too_far ** 2))

        return cost

    def _geometric_reg_cost(
        self,
        nl: tuple[float, float],
        nr: tuple[float, float],
        fr: tuple[float, float],
        fl: tuple[float, float],
    ) -> float:
        """Cost for non-convex or non-perspective-correct quads."""
        cost = 0.0

        # Near side should be wider than far side (perspective)
        near_width = nr[0] - nl[0]
        far_width = fr[0] - fl[0]
        if near_width <= far_width:
            cost += (far_width - near_width + 0.01) ** 2

        # Near side should be below far side
        near_mid_y = (nl[1] + nr[1]) / 2
        far_mid_y = (fl[1] + fr[1]) / 2
        if near_mid_y <= far_mid_y:
            cost += (far_mid_y - near_mid_y + 0.1) ** 2

        # Minimum widths
        if near_width < 0.1:
            cost += (0.1 - near_width) ** 2
        if far_width < 0.05:
            cost += (0.05 - far_width) ** 2

        # Convexity check via cross products
        corners = [nl, nr, fr, fl]
        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]
            p2 = corners[(i + 2) % 4]
            cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            if cross > 0:  # should be negative for clockwise
                cost += cross ** 2

        return cost

    def _line_evidence_cost(
        self,
        nl: tuple[float, float],
        nr: tuple[float, float],
        initial_nl: tuple[float, float],
        initial_nr: tuple[float, float],
    ) -> float:
        """Cost for deviating from line-detected near corner positions."""
        dl = (nl[0] - initial_nl[0]) ** 2 + (nl[1] - initial_nl[1]) ** 2
        dr = (nr[0] - initial_nr[0]) ** 2 + (nr[1] - initial_nr[1]) ** 2
        return dl + dr

    @staticmethod
    def _corners_to_homography(
        nl: tuple[float, float],
        nr: tuple[float, float],
        fr: tuple[float, float],
        fl: tuple[float, float],
    ) -> np.ndarray | None:
        """Compute homography from image corners to court model.

        Maps image corners → court model corners (image→court).
        """
        img_pts = np.array([nl, nr, fr, fl], dtype=np.float64)
        court_pts = np.array(COURT_MODEL_CORNERS, dtype=np.float64)

        try:
            H, _ = cv2.findHomography(img_pts, court_pts)  # noqa: N806
        except cv2.error:
            return None

        if H is None:
            return None

        return H

    @staticmethod
    def _project_points(
        H: np.ndarray,  # noqa: N803
        image_pts: np.ndarray,
    ) -> np.ndarray:
        """Project image points to court space using homography H (image→court)."""
        pts = np.hstack([image_pts, np.ones((len(image_pts), 1))]).T  # (3, N)
        result = H @ pts  # (3, N)
        # Normalize by homogeneous coordinate
        w = result[2, :]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        return np.asarray((result[:2, :] / w).T)  # (N, 2)

    @staticmethod
    def _is_valid_court_quad(
        nl: tuple[float, float],
        nr: tuple[float, float],
        fr: tuple[float, float],
        fl: tuple[float, float],
    ) -> bool:
        """Check if the quad is a valid court (convex, perspective-correct)."""
        # Near side wider than far side
        near_width = nr[0] - nl[0]
        far_width = fr[0] - fl[0]
        if near_width < 0.05 or far_width < 0.02:
            return False

        # Near below far
        near_mid_y = (nl[1] + nr[1]) / 2
        far_mid_y = (fl[1] + fr[1]) / 2
        if near_mid_y <= far_mid_y:
            return False

        # Basic convexity (all cross products same sign)
        corners = [nl, nr, fr, fl]
        crosses = []
        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]
            p2 = corners[(i + 2) % 4]
            cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            crosses.append(cross)

        # All should be same sign (negative for clockwise)
        if not (all(c < 0 for c in crosses) or all(c > 0 for c in crosses)):
            return False

        return True

    def _evaluate_team_separation(
        self,
        H: np.ndarray,  # noqa: N803
        near_feet: np.ndarray,
        far_feet: np.ndarray,
    ) -> tuple[int, int, int]:
        """Evaluate how well the homography separates teams.

        Returns (near_correct, far_correct, total).
        """
        net_y = COURT_LENGTH / 2.0
        near_correct = 0
        far_correct = 0

        if len(near_feet) > 0:
            near_court = self._project_points(H, near_feet)
            near_correct = int(np.sum(near_court[:, 1] < net_y))

        if len(far_feet) > 0:
            far_court = self._project_points(H, far_feet)
            far_correct = int(np.sum(far_court[:, 1] >= net_y))

        total = len(near_feet) + len(far_feet)
        return near_correct, far_correct, total


def extract_player_feet(
    player_positions: Sequence[object],
    team_assignments: dict[int, int],
) -> list[PlayerFootPosition]:
    """Extract player foot positions from tracking data.

    Args:
        player_positions: List of PlayerPosition objects.
        team_assignments: Map of track_id → team (0=near, 1=far).

    Returns:
        List of PlayerFootPosition with team assignments.
    """
    feet: list[PlayerFootPosition] = []
    for pos in player_positions:
        track_id = getattr(pos, "track_id", -1)
        if track_id not in team_assignments:
            continue

        x = getattr(pos, "x", 0.0)
        y = getattr(pos, "y", 0.0)
        height = getattr(pos, "height", 0.0)

        # Foot position = bbox bottom center
        foot_y = y + height / 2.0

        feet.append(PlayerFootPosition(
            x=x, y=foot_y,
            team=team_assignments[track_id],
        ))

    return feet
