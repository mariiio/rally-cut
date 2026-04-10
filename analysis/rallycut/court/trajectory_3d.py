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

# Net height (metres) — men's beach volleyball.
_NET_HEIGHT = 2.43


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

    Returns:
        ``FittedArc`` or ``None`` if the fit fails or there are too few observations.
    """
    # Collect observations within the arc.
    obs: list[tuple[float, float, float, float]] = []  # (t_sec, u, v, confidence)
    for bp in ball_positions:
        if bp.frame_number < start_frame or bp.frame_number > end_frame:
            continue
        if bp.confidence < 0.1:
            continue
        t = (bp.frame_number - start_frame) / fps
        obs.append((t, bp.x, bp.y, bp.confidence))

    if len(obs) < _MIN_OBS:
        return None

    obs_arr = np.array(obs, dtype=np.float64)
    times = obs_arr[:, 0]
    uv = obs_arr[:, 1:3]
    weights = obs_arr[:, 3]

    # --- Multi-start optimisation ---------------------------------------------
    # Try several initial guesses spanning plausible court positions and
    # heights.  Pick the fit with the lowest RMSE.
    best_result: tuple[NDArray[np.float64], float] | None = None

    for p0 in _multi_start_guesses(camera, uv, times, z0_prior):
        result = _fit_parabola(camera, times, uv, weights, p0, free_gravity=False)
        if result is None:
            continue
        if best_result is None or result[1] < best_result[1]:
            best_result = result

    if best_result is None:
        return None

    params, rmse = best_result

    # --- Optional: free-gravity refit for diagnostics -----------------------
    g_residual: float | None = None
    result_free = _fit_parabola(camera, times, uv, weights, params[:6], free_gravity=True)
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
    landing = _compute_landing(pos0, vel0)

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
        num_observations=len(obs),
        num_inliers=n_inliers,
        initial_position=pos0,
        initial_velocity=vel0,
        speed_at_start=speed,
        peak_height=peak_height,
        net_crossing_height=net_crossing,
        landing_position=landing,
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
) -> TrajectoryResult:
    """Fit 3D trajectories for all free-flight arcs in a rally.

    Segments the ball trajectory at contact frames and fits each arc
    independently.
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

    # Build (start_frame, end_frame, z0_prior) for each arc.
    arc_specs: list[tuple[int, int, float]] = []

    for i in range(len(contacts) - 1):
        sf = contacts[i].frame
        ef = contacts[i + 1].frame
        z0 = _z0_for_action(classified_actions, i) if classified_actions else 2.0
        arc_specs.append((sf, ef, z0))

    # Also fit the arc after the last contact (ball flight until end of tracking).
    last_frame = max(bp.frame_number for bp in ball_positions)
    if last_frame > contacts[-1].frame + _MIN_OBS:
        z0 = _z0_for_action(classified_actions, len(contacts) - 1) if classified_actions else 2.0
        arc_specs.append((contacts[-1].frame, last_frame, z0))

    for idx, (sf, ef, z0) in enumerate(arc_specs):
        arc = fit_arc(camera, ball_positions, sf, ef, fps, arc_index=idx, z0_prior=z0)
        if arc is not None:
            result.arcs.append(arc)

    # Serve speed = speed at the first arc (if it's a serve).
    if result.arcs:
        result.serve_speed_mps = result.arcs[0].speed_at_start

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
) -> NDArray[np.float64]:
    """Reprojection residuals for the parabolic fit (vectorised)."""
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
    residuals = np.empty(2 * n)
    residuals[0::2] = w_sqrt * (pred_u - uv[:, 0])
    residuals[1::2] = w_sqrt * (pred_v - uv[:, 1])

    return residuals


def _fit_parabola(
    camera: CameraModel,
    times: NDArray[np.float64],
    uv: NDArray[np.float64],
    weights: NDArray[np.float64],
    p0: NDArray[np.float64],
    free_gravity: bool,
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
            args=(camera, times, uv, weights, free_gravity),
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=0.01,  # ~10px / 1000px normalisation
            max_nfev=2000,
        )
    except Exception:
        return None

    if not result.success and result.status not in (1, 2, 3, 4):
        return None

    # RMSE in normalised image coordinates.
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
