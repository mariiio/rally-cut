"""3D trajectory fitting for volleyball ball flight arcs.

Fits physics-constrained parabolic trajectories to 2D ball detections
given a calibrated camera model.  Between any two contacts the ball is
in free flight under gravity:

    X(t) = x0 + vx0·t
    Y(t) = y0 + vy0·t
    Z(t) = z0 + vz0·t − 0.5·g·t²

Each 2D observation constrains the 3D point to lie on a camera ray.
We minimise reprojection error with ``scipy.optimize.least_squares``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from rallycut.court.camera_model import CameraModel, project_3d_to_image

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rallycut.tracking.action_classifier import ClassifiedAction
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.contact_detector import ContactSequence

logger = logging.getLogger(__name__)

# Gravity in m/s².
GRAVITY = 9.81

# Minimum observations for a valid arc fit.
_MIN_OBS = 5

# Net position in court Y coordinate (metres).
_NET_Y = 8.0

# Net height (metres) — women's beach volleyball (default for our dataset).
_NET_HEIGHT = 2.24

# Maximum plausible ball height at net crossing (metres).
_NET_CEILING = 4.5

# Penalty weight for net-crossing constraint.
# The net constraint residual is normalised: deviation / half_range, so
# it's O(1) at the boundary.  Reprojection residuals are ~0.01–0.03 in
# normalised coords.  A weight of 0.05 means a boundary violation is
# comparable to ~5 px of reprojection error — a moderate steering force.
_W_NET = 0.05

# Penalty weight for landing constraint (weaker — less certain).
_W_LAND = 0.02


@dataclass
class FittedArc:
    """Result of fitting a single free-flight arc."""

    arc_index: int
    start_frame: int
    end_frame: int
    num_observations: int
    num_inliers: int

    # Fitted 3D trajectory.
    initial_position: NDArray[np.float64]   # [x0, y0, z0] metres
    initial_velocity: NDArray[np.float64]   # [vx0, vy0, vz0] m/s
    speed_at_start: float                   # |v0| m/s

    # Derived quantities.
    peak_height: float                      # max Z in metres
    net_crossing_height: float | None       # Z at Y=NET_Y, if arc crosses net
    landing_position: tuple[float, float] | None  # (X, Y) where Z=0

    # Quality.
    reprojection_rmse: float                # pixels
    gravity_residual: float | None          # fitted g / 9.81 − 1 (from free-g fit)
    is_valid: bool


@dataclass
class TrajectoryResult:
    """3D trajectory fitting result for a full rally."""

    rally_id: str
    video_id: str
    camera: CameraModel
    arcs: list[FittedArc] = field(default_factory=list)
    serve_speed_mps: float | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_arc(
    camera: CameraModel,
    ball_positions: list[BallPosition],
    start_frame: int,
    end_frame: int,
    fps: float,
    arc_index: int = 0,
    z0_prior: float = 2.0,
    net_height: float = _NET_HEIGHT,
    landing: bool = False,
) -> FittedArc | None:
    """Fit a 3D parabola to a single free-flight arc.

    Args:
        camera: Calibrated camera model.
        ball_positions: Full rally ball positions (will be filtered to arc range).
        start_frame: First frame of this arc (contact frame).
        end_frame: Last frame of this arc (next contact frame).
        fps: Video frame rate.
        arc_index: 0-based index within the rally.
        z0_prior: Initial height guess in metres.
        net_height: Minimum ball height at net crossing (metres).
        landing: If True, penalise positive Z at the arc's end.

    Returns:
        ``FittedArc`` or ``None`` if the fit fails or there are too few observations.
    """
    obs_arr = _collect_observations(ball_positions, start_frame, end_frame, fps)
    if obs_arr is None:
        return None

    times = obs_arr[:, 0]
    uv = obs_arr[:, 1:3]
    weights = obs_arr[:, 3]

    # Build constraint parameters.
    net_constraint: tuple[float, float, float] | None = (
        _NET_Y, net_height, _NET_CEILING,
    )

    # --- Multi-start optimisation ---------------------------------------------
    # Try several initial guesses spanning plausible court positions and
    # heights.  Pick the fit with the lowest RMSE.
    best_result: tuple[NDArray[np.float64], float] | None = None

    for p0 in _multi_start_guesses(camera, uv, times, z0_prior):
        result = _fit_parabola(
            camera, times, uv, weights, p0, free_gravity=False,
            net_constraint=net_constraint, landing_constraint=landing,
        )
        if result is None:
            continue
        if best_result is None or result[1] < best_result[1]:
            best_result = result

    if best_result is None:
        return None

    params, rmse = best_result

    # --- Optional: free-gravity refit for diagnostics -----------------------
    # NOTE: no constraints here — this is a physics consistency check.
    g_residual: float | None = None
    result_free = _fit_parabola(
        camera, times, uv, weights, params[:6], free_gravity=True,
    )
    if result_free is not None:
        params_free, _ = result_free
        fitted_g = params_free[6]
        g_residual = fitted_g / GRAVITY - 1.0

    # --- Extract results ----------------------------------------------------
    pos0 = params[:3]
    vel0 = params[3:6]
    speed = float(np.linalg.norm(vel0))

    # Peak height: solve dZ/dt = 0 → t_peak = vz0/g.
    t_peak = vel0[2] / GRAVITY if vel0[2] > 0 else 0.0
    peak_z = float(pos0[2] + vel0[2] * t_peak - 0.5 * GRAVITY * t_peak**2)
    peak_height = max(float(pos0[2]), peak_z)

    # Net crossing.
    net_crossing = _compute_net_crossing(pos0, vel0, times[-1])

    # Landing position (Z = 0 intercept).
    landing_pos = _compute_landing(pos0, vel0)

    # Count inliers (reprojection error < 5 px).
    inlier_thresh_norm = 5.0 / max(camera.image_size)
    n_inliers = 0
    for i in range(len(times)):
        pt3d = _eval_trajectory(params[:3], params[3:6], times[i])
        u_hat, v_hat = project_3d_to_image(camera, pt3d)
        err = math.sqrt((u_hat - uv[i, 0]) ** 2 + (v_hat - uv[i, 1]) ** 2)
        if err < inlier_thresh_norm:
            n_inliers += 1

    # Convert RMSE from normalised to pixel units.
    rmse_px = rmse * max(camera.image_size)

    is_valid = (
        rmse_px < 10.0
        and 0.0 <= pos0[2] <= 6.0
        and speed < 40.0
    )

    return FittedArc(
        arc_index=arc_index,
        start_frame=start_frame,
        end_frame=end_frame,
        num_observations=len(obs_arr),
        num_inliers=n_inliers,
        initial_position=pos0,
        initial_velocity=vel0,
        speed_at_start=speed,
        peak_height=peak_height,
        net_crossing_height=net_crossing,
        landing_position=landing_pos,
        reprojection_rmse=rmse_px,
        gravity_residual=g_residual,
        is_valid=is_valid,
    )


def fit_rally(
    camera: CameraModel,
    contact_sequence: ContactSequence,
    classified_actions: list[ClassifiedAction] | None,
    fps: float,
    rally_id: str = "",
    video_id: str = "",
    net_height: float = _NET_HEIGHT,
    joint: bool = False,
) -> TrajectoryResult:
    """Fit 3D trajectories for all free-flight arcs in a rally.

    Segments the ball trajectory at contact frames and fits each arc
    independently, with geometric constraints (net crossing, landing).
    """
    result = TrajectoryResult(
        rally_id=rally_id,
        video_id=video_id,
        camera=camera,
    )

    contacts = contact_sequence.contacts
    ball_positions = contact_sequence.ball_positions

    if not contacts or not ball_positions:
        return result

    # Build (start_frame, end_frame, z0_prior, is_landing) for each arc.
    arc_specs: list[tuple[int, int, float, bool]] = []

    for i in range(len(contacts) - 1):
        sf = contacts[i].frame
        ef = contacts[i + 1].frame
        z0 = _z0_for_action(classified_actions, i) if classified_actions else 2.0
        is_landing = _is_landing_arc(classified_actions, i + 1)
        arc_specs.append((sf, ef, z0, is_landing))

    # Also fit the arc after the last contact (ball flight until end of tracking).
    last_frame = max(bp.frame_number for bp in ball_positions)
    if last_frame > contacts[-1].frame + _MIN_OBS:
        z0 = _z0_for_action(classified_actions, len(contacts) - 1) if classified_actions else 2.0
        # Last arc in rally — ball likely lands.
        arc_specs.append((contacts[-1].frame, last_frame, z0, True))

    fitted_arc_specs: list[tuple[int, int, float, bool]] = []
    for idx, (sf, ef, z0, is_landing) in enumerate(arc_specs):
        arc = fit_arc(
            camera, ball_positions, sf, ef, fps,
            arc_index=idx, z0_prior=z0,
            net_height=net_height, landing=is_landing,
        )
        if arc is not None:
            result.arcs.append(arc)
            fitted_arc_specs.append((sf, ef, z0, is_landing))

    # --- Joint multi-arc refinement -------------------------------------------
    if joint and len(result.arcs) >= 2:
        joint_arcs = _fit_rally_joint(
            camera, result.arcs, fitted_arc_specs, ball_positions, fps,
            net_height=net_height,
        )
        if joint_arcs is not None:
            result.arcs = joint_arcs

    # Serve speed = speed at the first arc (if it's a serve).
    if result.arcs:
        result.serve_speed_mps = result.arcs[0].speed_at_start

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_rally_joint(
    camera: CameraModel,
    independent_arcs: list[FittedArc],
    arc_specs: list[tuple[int, int, float, bool]],
    ball_positions: list[BallPosition],
    fps: float,
    net_height: float = _NET_HEIGHT,
) -> list[FittedArc] | None:
    """Refit arcs jointly so consecutive arcs share their contact point.

    Parameterisation: arc 0 has 6 params (pos0, vel0); arc i>0 has 3 params
    (vel0 only — pos0 is the endpoint of the previous arc).  Total params =
    3N + 3 for N arcs.

    Returns a new list of FittedArc, or None if joint fit is worse.
    """
    n_arcs = len(independent_arcs)
    if n_arcs < 2:
        return None

    # Collect per-arc observation arrays.
    arc_data: list[tuple[NDArray[np.float64], NDArray[np.float64],
                         NDArray[np.float64], float, bool]] = []
    for arc, (sf, ef, _z0, is_landing) in zip(independent_arcs, arc_specs):
        arr = _collect_observations(ball_positions, sf, ef, fps)
        if arr is None:
            return None  # Can't joint-fit if any arc is missing data.
        duration = (ef - sf) / fps
        arc_data.append((arr[:, 0], arr[:, 1:3], arr[:, 3], duration, is_landing))

    net_constraint = (_NET_Y, net_height, _NET_CEILING)

    # Build initial guess from independent fits: [pos0_0, vel0_0, vel0_1, ..., vel0_{N-1}].
    x0 = np.concatenate([
        independent_arcs[0].initial_position,
        independent_arcs[0].initial_velocity,
    ] + [arc.initial_velocity for arc in independent_arcs[1:]])

    n_params = 6 + 3 * (n_arcs - 1)
    assert len(x0) == n_params

    # Bounds.
    lb = np.array([-10, -10, 0.0, -35, -35, -20] + [-35, -35, -20] * (n_arcs - 1))
    ub = np.array([18, 26, 6.0, 35, 35, 20] + [35, 35, 20] * (n_arcs - 1))
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)

    def joint_residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        all_residuals: list[NDArray[np.float64]] = []
        pos0 = params[:3].copy()
        vel0 = params[3:6].copy()

        for i, (times, uv, weights, duration, is_landing) in enumerate(arc_data):
            if i > 0:
                vel0 = params[6 + 3 * (i - 1):6 + 3 * i].copy()

            # Reprojection residuals for this arc.
            arc_params = np.concatenate([pos0, vel0])
            arc_resid = _residuals(
                arc_params, camera, times, uv, weights,
                free_gravity=False,
                net_constraint=net_constraint,
                landing_constraint=is_landing,
            )
            all_residuals.append(arc_resid)

            # Compute endpoint for chaining to next arc.
            t_end = duration
            pos0 = np.array([
                pos0[0] + vel0[0] * t_end,
                pos0[1] + vel0[1] * t_end,
                pos0[2] + vel0[2] * t_end - 0.5 * GRAVITY * t_end ** 2,
            ])

        return np.concatenate(all_residuals)

    try:
        result = least_squares(
            joint_residuals, x0, bounds=(lb, ub),
            loss="soft_l1", f_scale=0.01, max_nfev=5000,
        )
    except Exception:
        return None

    if not result.success and result.status not in (1, 2, 3, 4):
        return None

    # Extract per-arc results and compare with independent fits.
    joint_params = result.x
    pos0 = joint_params[:3].copy()
    vel0 = joint_params[3:6].copy()

    new_arcs: list[FittedArc] = []
    total_joint_rmse = 0.0
    total_indep_rmse = 0.0

    for i, (times, uv, weights, duration, is_landing) in enumerate(arc_data):
        if i > 0:
            vel0 = joint_params[6 + 3 * (i - 1):6 + 3 * i].copy()

        arc_params = np.concatenate([pos0, vel0])
        old_arc = independent_arcs[i]

        # Compute reprojection RMSE (without constraint penalties).
        raw = _residuals(arc_params, camera, times, uv, np.ones(len(times)), False)
        rmse_norm = float(np.sqrt(np.mean(raw ** 2)))
        rmse_px = rmse_norm * max(camera.image_size)

        total_joint_rmse += rmse_px
        total_indep_rmse += old_arc.reprojection_rmse

        speed = float(np.linalg.norm(vel0))
        t_peak = vel0[2] / GRAVITY if vel0[2] > 0 else 0.0
        peak_z = float(pos0[2] + vel0[2] * t_peak - 0.5 * GRAVITY * t_peak ** 2)
        peak_height = max(float(pos0[2]), peak_z)
        net_crossing = _compute_net_crossing(pos0, vel0, times[-1])
        landing_pos = _compute_landing(pos0, vel0)

        # Count inliers.
        inlier_thresh = 5.0 / max(camera.image_size)
        n_inliers = 0
        for j in range(len(times)):
            pt3d = _eval_trajectory(pos0, vel0, float(times[j]))
            u_hat, v_hat = project_3d_to_image(camera, pt3d)
            err = math.sqrt((u_hat - uv[j, 0]) ** 2 + (v_hat - uv[j, 1]) ** 2)
            if err < inlier_thresh:
                n_inliers += 1

        is_valid = rmse_px < 10.0 and 0.0 <= pos0[2] <= 6.0 and speed < 40.0

        new_arcs.append(FittedArc(
            arc_index=old_arc.arc_index,
            start_frame=old_arc.start_frame,
            end_frame=old_arc.end_frame,
            num_observations=old_arc.num_observations,
            num_inliers=n_inliers,
            initial_position=pos0.copy(),
            initial_velocity=vel0.copy(),
            speed_at_start=speed,
            peak_height=peak_height,
            net_crossing_height=net_crossing,
            landing_position=landing_pos,
            reprojection_rmse=rmse_px,
            gravity_residual=old_arc.gravity_residual,
            is_valid=is_valid,
        ))

        # Chain endpoint.
        t_end = duration
        pos0 = np.array([
            pos0[0] + vel0[0] * t_end,
            pos0[1] + vel0[1] * t_end,
            pos0[2] + vel0[2] * t_end - 0.5 * GRAVITY * t_end ** 2,
        ])

    # Keep joint fit only if total RMSE improves (or is within 5% tolerance).
    if total_joint_rmse <= total_indep_rmse * 1.05:
        return new_arcs
    return None


def _collect_observations(
    ball_positions: list[BallPosition],
    start_frame: int,
    end_frame: int,
    fps: float,
) -> NDArray[np.float64] | None:
    """Collect ball observations in an arc as (t_sec, u, v, confidence) rows.

    Returns an (N, 4) array, or None if fewer than ``_MIN_OBS`` are found.
    """
    obs: list[tuple[float, float, float, float]] = []
    for bp in ball_positions:
        if bp.frame_number < start_frame or bp.frame_number > end_frame:
            continue
        if bp.confidence < 0.1:
            continue
        t = (bp.frame_number - start_frame) / fps
        obs.append((t, bp.x, bp.y, bp.confidence))
    if len(obs) < _MIN_OBS:
        return None
    return np.array(obs, dtype=np.float64)


def _eval_trajectory(
    pos0: NDArray[np.float64],
    vel0: NDArray[np.float64],
    t: float,
    g: float = GRAVITY,
) -> NDArray[np.float64]:
    """Evaluate the 3D trajectory at time *t*."""
    return np.array([
        pos0[0] + vel0[0] * t,
        pos0[1] + vel0[1] * t,
        pos0[2] + vel0[2] * t - 0.5 * g * t**2,
    ])


def _residuals(
    params: NDArray[np.float64],
    camera: CameraModel,
    times: NDArray[np.float64],
    uv: NDArray[np.float64],
    weights: NDArray[np.float64],
    free_gravity: bool,
    net_constraint: tuple[float, float, float] | None = None,
    landing_constraint: bool = False,
) -> NDArray[np.float64]:
    """Reprojection residuals for the parabolic fit (vectorised).

    Args:
        net_constraint: (net_y, z_min, z_max) — when the trajectory crosses
            Y=net_y, penalise Z outside [z_min, z_max].
        landing_constraint: If True, penalise positive Z at the last time step.
    """
    pos0 = params[:3]
    vel0 = params[3:6]
    g = float(params[6]) if free_gravity else GRAVITY

    n = len(times)
    # Evaluate 3D positions for all time steps at once.
    # pts3d shape: (n, 3)
    t = times[:, np.newaxis]  # (n, 1)
    pts3d = pos0 + vel0 * t
    pts3d[:, 2] -= 0.5 * g * times**2

    # Batch projection: P @ [X, Y, Z, 1]^T for all points.
    # pts_h shape: (n, 4)
    pts_h = np.column_stack([pts3d, np.ones(n)])
    # proj shape: (n, 3)
    proj = (camera.projection_matrix @ pts_h.T).T
    w_img, h_img = camera.image_size
    pred_u = (proj[:, 0] / proj[:, 2]) / w_img
    pred_v = (proj[:, 1] / proj[:, 2]) / h_img

    w_sqrt = np.sqrt(weights)
    reproj = np.empty(2 * n)
    reproj[0::2] = w_sqrt * (pred_u - uv[:, 0])
    reproj[1::2] = w_sqrt * (pred_v - uv[:, 1])

    # --- Geometric constraint penalties ----------------------------------------
    penalties: list[float] = []

    # Net-crossing height constraint: continuous pull toward the midpoint
    # of [z_min, z_max].  The residual is (z_net - midpoint) / half_range,
    # which is 0 at the midpoint, ±1 at the boundaries.  least_squares
    # squares this, so the cost is quadratic — a smooth, always-active
    # attraction toward the expected height range.
    #
    # Design choice: centering pull (not dead-zone).  A dead-zone penalty
    # (zero inside the range) provides no gradient when the height is
    # already valid, and measured 66% vs 85% pass rate.  The centering
    # pull actively steers trajectories toward plausible net heights even
    # when the unconstrained fit is technically in-range but near a
    # boundary — this is important because the depth axis is poorly
    # conditioned from low cameras.
    if net_constraint is not None:
        net_y, z_min, z_max = net_constraint
        if abs(vel0[1]) > 1e-6:
            t_net = (net_y - pos0[1]) / vel0[1]
            duration = float(times[-1])
            if 0 < t_net < duration * 1.5:
                z_net = pos0[2] + vel0[2] * t_net - 0.5 * g * t_net ** 2
                z_target = 0.5 * (z_min + z_max)
                z_range = 0.5 * (z_max - z_min) if z_max > z_min else 1.0
                deviation = (z_net - z_target) / z_range
                penalties.append(_W_NET * deviation)

    # Landing constraint: ball should be near ground at arc end.
    if landing_constraint:
        z_end = pos0[2] + vel0[2] * times[-1] - 0.5 * g * times[-1] ** 2
        penalties.append(_W_LAND * max(0.0, z_end - 0.5))

    if penalties:
        return np.concatenate([reproj, np.array(penalties)])
    return reproj


def _fit_parabola(
    camera: CameraModel,
    times: NDArray[np.float64],
    uv: NDArray[np.float64],
    weights: NDArray[np.float64],
    p0: NDArray[np.float64],
    free_gravity: bool,
    net_constraint: tuple[float, float, float] | None = None,
    landing_constraint: bool = False,
) -> tuple[NDArray[np.float64], float] | None:
    """Run least-squares optimisation.

    Returns (params, rmse_normalised) or None on failure.
    """
    if free_gravity:
        x0 = np.append(p0, GRAVITY)
        lb = np.array([-10, -10, 0.0, -35, -35, -20, 5.0])
        ub = np.array([18, 26, 6.0, 35, 35, 20, 15.0])
    else:
        x0 = p0.copy()
        lb = np.array([-10, -10, 0.0, -35, -35, -20])
        ub = np.array([18, 26, 6.0, 35, 35, 20])

    # Clamp initial guess to bounds.
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)

    try:
        result = least_squares(
            _residuals,
            x0,
            args=(camera, times, uv, weights, free_gravity,
                  net_constraint, landing_constraint),
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=0.01,  # ~10px / 1000px normalisation
            max_nfev=2000,
        )
    except Exception:
        return None

    if not result.success and result.status not in (1, 2, 3, 4):
        return None

    # RMSE in normalised image coordinates (exclude penalty residuals).
    n = len(times)
    raw_residuals = _residuals(
        result.x, camera, times, uv, np.ones(n), free_gravity,
    )
    rmse = float(np.sqrt(np.mean(raw_residuals**2)))

    return result.x, rmse


def _initialise_params(
    camera: CameraModel,
    uv_first: NDArray[np.float64],
    uv_last: NDArray[np.float64],
    dt: float,
    z0_prior: float,
) -> NDArray[np.float64] | None:
    """Compute initial parameter guess for the trajectory.

    Uses a multi-strategy approach because ray-plane intersection fails
    for low cameras where ball positions are near the horizon.
    """
    if dt < 1e-6:
        return None

    # Strategy 1: intersect rays with z=z0_prior plane.
    pos = _ray_intersect_z(camera, uv_first, z0_prior)
    pos_end = _ray_intersect_z(camera, uv_last, z0_prior)

    if pos is not None and pos_end is not None:
        # Sanity: both should be within a generous court region.
        if _in_court_region(pos) and _in_court_region(pos_end):
            vx = (pos_end[0] - pos[0]) / dt
            vy = (pos_end[1] - pos[1]) / dt
            vz = 0.5 * GRAVITY * dt
            return np.array([pos[0], pos[1], z0_prior, vx, vy, vz])

    # Strategy 2: use image X position to estimate court X, and use the
    # net_y / court_side to estimate court Y.  This doesn't depend on
    # ray intersection and works for any camera geometry.
    #
    # Rough mapping: image x ∈ [0,1] maps to court x ∈ [0,8] (linear approx).
    # Ball image y above center → far court (y > 8), below → near court (y < 8).
    x_start = float(uv_first[0]) * 8.0  # crude but bounded
    x_end = float(uv_last[0]) * 8.0

    # Estimate Y from image position relative to image midpoint.
    # Ball near top of frame → far from camera → large Y.
    # Ball near bottom → near camera → small Y.
    mid_v = 0.55  # approximate court center in image Y
    y_start = 8.0 + (mid_v - float(uv_first[1])) * 20.0  # scale factor
    y_end = 8.0 + (mid_v - float(uv_last[1])) * 20.0
    y_start = np.clip(y_start, -2.0, 18.0)
    y_end = np.clip(y_end, -2.0, 18.0)

    vx = (x_end - x_start) / dt
    vy = (y_end - y_start) / dt
    vz = 0.5 * GRAVITY * dt

    return np.array([x_start, y_start, z0_prior, vx, vy, vz])


def _multi_start_guesses(
    camera: CameraModel,
    uv: NDArray[np.float64],
    times: NDArray[np.float64],
    z0_prior: float,
) -> list[NDArray[np.float64]]:
    """Generate diverse initial guesses for the trajectory optimiser.

    Tries ray-based initialisation first, then generates a grid of
    physics-informed starting points spanning plausible court positions,
    heights, and speeds.
    """
    dt = float(times[-1])
    if dt < 1e-6:
        return []

    guesses: list[NDArray[np.float64]] = []

    # --- Strategy 1: ray intersection (works for high cameras) ----------------
    p0 = _initialise_params(camera, uv[0], uv[-1], dt, z0_prior)
    if p0 is not None:
        guesses.append(p0)

    # --- Strategy 2: focused grid over court positions -----------------------
    x_start = float(uv[0, 0]) * 8.0
    x_end = float(uv[-1, 0]) * 8.0
    vx = (x_end - x_start) / dt
    vz = 0.5 * GRAVITY * dt

    # 8 guesses: near-court and far-court starts, crossing and same-side,
    # at two heights.
    for y_start, y_end in [(3.0, 13.0), (13.0, 3.0), (4.0, 6.0), (12.0, 10.0)]:
        for z0 in [1.5, 3.0]:
            vy = (y_end - y_start) / dt
            guesses.append(np.array([x_start, y_start, z0, vx, vy, vz]))

    return guesses


def _ray_intersect_z(
    camera: CameraModel,
    uv: NDArray[np.float64],
    z: float,
) -> NDArray[np.float64] | None:
    """Intersect camera ray with horizontal plane at height *z*."""
    from rallycut.court.camera_model import image_ray

    origin, direction = image_ray(camera, (float(uv[0]), float(uv[1])))
    if abs(direction[2]) < 1e-10:
        return None
    t_param = (z - origin[2]) / direction[2]
    if t_param < 0:
        return None
    hit: NDArray[np.float64] = origin + t_param * direction
    return hit


def _in_court_region(pt: NDArray[np.float64]) -> bool:
    """Check if a 3D point is in a generous region around the court."""
    return bool(-10 < pt[0] < 18 and -10 < pt[1] < 26)


def _z0_for_action(
    actions: list[ClassifiedAction] | None,
    contact_idx: int,
) -> float:
    """Initial height guess based on the action type at a contact."""
    if not actions or contact_idx >= len(actions):
        return 2.0

    action = actions[contact_idx].action_type
    # Import here to avoid circular imports at module level.
    from rallycut.tracking.action_classifier import ActionType

    return {
        ActionType.SERVE: 3.0,
        ActionType.RECEIVE: 1.0,
        ActionType.SET: 2.5,
        ActionType.ATTACK: 3.0,
        ActionType.BLOCK: 2.5,
        ActionType.DIG: 0.5,
    }.get(action, 2.0)


def _is_landing_arc(
    actions: list[ClassifiedAction] | None,
    next_contact_idx: int,
) -> bool:
    """Determine if an arc ends in a ground contact (receive, dig)."""
    if not actions or next_contact_idx >= len(actions):
        return False
    from rallycut.tracking.action_classifier import ActionType

    return actions[next_contact_idx].action_type in (
        ActionType.RECEIVE, ActionType.DIG,
    )


def _compute_net_crossing(
    pos0: NDArray[np.float64],
    vel0: NDArray[np.float64],
    max_t: float,
) -> float | None:
    """Compute ball height at net crossing (Y = NET_Y)."""
    if abs(vel0[1]) < 1e-6:
        return None

    t_net = (_NET_Y - pos0[1]) / vel0[1]
    if t_net < 0 or t_net > max_t * 1.5:
        return None

    z_net = float(pos0[2] + vel0[2] * t_net - 0.5 * GRAVITY * t_net**2)
    return z_net


def _compute_landing(
    pos0: NDArray[np.float64],
    vel0: NDArray[np.float64],
) -> tuple[float, float] | None:
    """Compute where the ball hits the ground (Z = 0).

    Solves z0 + vz0·t − 0.5·g·t² = 0 for the positive root.
    """
    a = -0.5 * GRAVITY
    b = vel0[2]
    c = pos0[2]

    disc = b**2 - 4 * a * c
    if disc < 0:
        return None

    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    # Pick the positive root that's in the future.
    t_land = None
    for t in (t1, t2):
        if t > 0.01:  # avoid t≈0 (the start)
            if t_land is None or t < t_land:
                t_land = t

    if t_land is None:
        return None

    x_land = float(pos0[0] + vel0[0] * t_land)
    y_land = float(pos0[1] + vel0[1] * t_land)
    return (x_land, y_land)
