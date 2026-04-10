# ruff: noqa: N803, N806
"""Pinhole camera model from court calibration.

Recovers camera intrinsics (K) and extrinsics (R, t) from 4 known court
corner correspondences.  Uses Zhang's single-image intrinsic estimation
to get focal length from the homography, then cv2.solvePnP for extrinsics.
This enables 3D reconstruction of airborne objects that a planar homography
cannot handle.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Reprojection error threshold (pixels) for a valid camera model.
# With focal length estimated from a single homography (4 coplanar points),
# 8px is a practical threshold — tighter requires better intrinsics.
_MAX_REPROJ_ERROR_PX = 8.0

# Camera height bounds (metres).
# Beach volleyball cameras range from low sideline (~1.5m) to elevated
# behind-baseline (~8m) or overhead (~15m).
_MIN_CAM_HEIGHT = 1.0
_MAX_CAM_HEIGHT = 20.0

# Focal-length bounds in pixels (at 1920px width).
# f=500 → ~125° FOV, f=5000 → ~22° FOV.
_MIN_FOCAL_PX_AT_1920 = 500.0
_MAX_FOCAL_PX_AT_1920 = 5000.0


@dataclass
class CameraModel:
    """Full pinhole camera extracted from court-corner correspondences."""

    intrinsic_matrix: NDArray[np.float64]   # 3×3 K
    rotation: NDArray[np.float64]           # 3×3 R  (world → camera)
    translation: NDArray[np.float64]        # 3×1 t
    projection_matrix: NDArray[np.float64]  # 3×4 P = K·[R|t]
    camera_position: NDArray[np.float64]    # 3×1 world pos (−R^T·t)
    focal_length_px: float                  # focal length in pixels
    reprojection_error: float               # mean corner error in pixels
    image_size: tuple[int, int]             # (width, height)
    is_valid: bool


def calibrate_camera(
    image_corners: list[tuple[float, float]],
    court_corners: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> CameraModel | None:
    """Recover a pinhole camera from 4 court-corner correspondences.

    Uses a two-stage approach:
    1. Compute homography and estimate focal length via Zhang's IAC constraints.
    2. Use ``cv2.solvePnP`` with the estimated K to recover (R, t).

    Args:
        image_corners: 4 corners in **normalised** image coordinates (0-1).
        court_corners: 4 corners in court coordinates (metres), on z=0.
        image_width:  Video frame width in pixels.
        image_height: Video frame height in pixels.

    Returns:
        A validated ``CameraModel``, or ``None`` if calibration fails.
    """
    if len(image_corners) != 4 or len(court_corners) != 4:
        return None

    # --- Convert to pixel coordinates -----------------------------------------
    img_pts_px = np.array(
        [[x * image_width, y * image_height] for x, y in image_corners],
        dtype=np.float64,
    )
    court_pts = np.array(court_corners, dtype=np.float64)
    obj_pts_3d = np.array(
        [[cx, cy, 0.0] for cx, cy in court_corners], dtype=np.float64
    )

    # --- Stage 1: Estimate focal length from homography -----------------------
    H_raw, _ = cv2.findHomography(court_pts, img_pts_px)
    if H_raw is None:
        return None
    H = np.asarray(H_raw, dtype=np.float64)

    cx_px = image_width / 2.0
    cy_px = image_height / 2.0
    focal = _estimate_focal_from_homography(H, cx_px, cy_px)

    # Scale focal-length bounds to current image width.
    scale = image_width / 1920.0
    min_f = _MIN_FOCAL_PX_AT_1920 * scale
    max_f = _MAX_FOCAL_PX_AT_1920 * scale

    if focal is None or focal < min_f or focal > max_f:
        # Fallback: try a grid of focal lengths.
        focal = _grid_search_focal(obj_pts_3d, img_pts_px, cx_px, cy_px, min_f, max_f)
        if focal is None:
            return None

    K = np.array(
        [[focal, 0.0, cx_px], [0.0, focal, cy_px], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # --- Stage 2: Solve for R, t with known K ---------------------------------
    best: CameraModel | None = None

    for method in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts_3d, img_pts_px, K, distCoeffs=np.zeros(4), flags=method,
            )
        except cv2.error:
            continue
        if not ok:
            continue

        cand = _build_model(
            K, np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64),
            obj_pts_3d, img_pts_px, image_width, image_height,
        )
        if cand is not None and (best is None or cand.reprojection_error < best.reprojection_error):
            best = cand

    # --- Always do a grid search to find optimal focal length ----------------
    grid = _grid_search_focal(
        obj_pts_3d, img_pts_px, cx_px, cy_px, min_f, max_f, steps=40,
    )
    if grid is not None:
        K_grid = np.array(
            [[grid, 0.0, cx_px], [0.0, grid, cy_px], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        for method in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts_3d, img_pts_px, K_grid, distCoeffs=np.zeros(4), flags=method,
                )
            except cv2.error:
                continue
            if not ok:
                continue
            cand = _build_model(
                K_grid, np.asarray(rvec, dtype=np.float64),
                np.asarray(tvec, dtype=np.float64),
                obj_pts_3d, img_pts_px, image_width, image_height,
            )
            if cand is not None and (best is None or cand.reprojection_error < best.reprojection_error):
                best = cand

    # --- Refinement: fine-grid around the best focal length ------------------
    if best is not None and best.reprojection_error > 1.0:
        refined = _refine_focal(
            best, obj_pts_3d, img_pts_px, cx_px, cy_px, image_width, image_height,
        )
        if refined is not None and refined.reprojection_error < best.reprojection_error:
            best = refined

    return best


# Net height (metres) — men's beach volleyball.
_NET_HEIGHT_M = 2.43

# Court midpoint at net (metres).
_NET_MID_COURT = (COURT_WIDTH / 2.0, COURT_LENGTH / 2.0)


def calibrate_camera_with_net(
    image_corners: list[tuple[float, float]],
    court_corners: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    net_y_image: float,
    net_height: float = _NET_HEIGHT_M,
) -> CameraModel | None:
    """Recover a pinhole camera using 4 ground corners + net-height constraint.

    Calibrates with the 4 ground corners (standard), then selects the
    focal length whose predicted net-top image position best matches the
    observed ``net_y_image`` from ball trajectory analysis.  This breaks
    the focal-length / camera-height ambiguity without requiring the
    net point to participate in solvePnP (which inflates reproj error).

    Args:
        image_corners: 4 corners in normalised image coordinates (0-1).
        court_corners: 4 corners in court coordinates (metres), z=0.
        image_width:  Frame width in pixels.
        image_height: Frame height in pixels.
        net_y_image: Normalised image Y of the net top (0-1).
        net_height: Physical height of the net top in metres (default 2.43).

    Returns:
        A validated ``CameraModel``, or ``None`` on failure.
    """
    if len(image_corners) != 4 or len(court_corners) != 4:
        return None

    img_pts_px = np.array(
        [[x * image_width, y * image_height] for x, y in image_corners],
        dtype=np.float64,
    )
    obj_pts_3d = np.array(
        [[cx, cy, 0.0] for cx, cy in court_corners], dtype=np.float64
    )

    cx_px = image_width / 2.0
    cy_px = image_height / 2.0

    scale = image_width / 1920.0
    min_f = _MIN_FOCAL_PX_AT_1920 * scale
    max_f = _MAX_FOCAL_PX_AT_1920 * scale

    # Net-top 3D point.
    net_3d = np.array([_NET_MID_COURT[0], _NET_MID_COURT[1], net_height])

    # --- Grid search: pick the focal length where predicted net Y ≈ observed --
    best: CameraModel | None = None
    best_net_err = float("inf")

    for i in range(81):
        f = min_f + (max_f - min_f) * i / 80
        K = np.array([[f, 0, cx_px], [0, f, cy_px], [0, 0, 1]], dtype=np.float64)

        for method in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts_3d, img_pts_px, K, distCoeffs=np.zeros(4), flags=method,
                )
            except cv2.error:
                continue
            if not ok:
                continue

            cand = _build_model(
                K, np.asarray(rvec, dtype=np.float64),
                np.asarray(tvec, dtype=np.float64),
                obj_pts_3d, img_pts_px, image_width, image_height,
            )
            if cand is None:
                continue

            # Score: how well does this camera predict the net-top Y?
            _, pred_net_v = project_3d_to_image(cand, net_3d)
            net_err = abs(pred_net_v - net_y_image)

            # Accept candidates with reasonable corner reproj AND best net match.
            if net_err < best_net_err:
                best_net_err = net_err
                best = cand

    # --- Fine-grid refinement around the best --------------------------------
    if best is not None:
        f0 = best.focal_length_px
        for i in range(41):
            f = f0 * (0.8 + i * 0.01)
            K = np.array([[f, 0, cx_px], [0, f, cy_px], [0, 0, 1]], dtype=np.float64)
            for method in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
                try:
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_pts_3d, img_pts_px, K, distCoeffs=np.zeros(4), flags=method,
                    )
                except cv2.error:
                    continue
                if not ok:
                    continue
                cand = _build_model(
                    K, np.asarray(rvec, dtype=np.float64),
                    np.asarray(tvec, dtype=np.float64),
                    obj_pts_3d, img_pts_px, image_width, image_height,
                )
                if cand is None:
                    continue
                _, pred_net_v = project_3d_to_image(cand, net_3d)
                net_err = abs(pred_net_v - net_y_image)
                if net_err < best_net_err:
                    best_net_err = net_err
                    best = cand

    return best


def project_3d_to_image(
    camera: CameraModel,
    world_point: NDArray[np.float64],
) -> tuple[float, float]:
    """Project a 3D world point to **normalised** image coordinates (0-1)."""
    pt_h = np.array([world_point[0], world_point[1], world_point[2], 1.0])
    proj = camera.projection_matrix @ pt_h
    if abs(proj[2]) < 1e-10:
        return (0.0, 0.0)
    u_px = proj[0] / proj[2]
    v_px = proj[1] / proj[2]
    w, h = camera.image_size
    return (float(u_px / w), float(v_px / h))


def image_ray(
    camera: CameraModel,
    image_point: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(origin, direction)`` of the camera ray through *image_point*.

    *image_point* is in normalised (0-1) coordinates.  *origin* is the camera
    world position and *direction* is a unit-length vector in world coordinates.
    """
    w, h = camera.image_size
    u_px = image_point[0] * w
    v_px = image_point[1] * h

    K_inv = np.linalg.inv(camera.intrinsic_matrix)
    ray_cam = K_inv @ np.array([u_px, v_px, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)

    ray_world = camera.rotation.T @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    return camera.camera_position.copy(), ray_world



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_focal_from_homography(
    H: NDArray[np.float64],
    cx: float,
    cy: float,
) -> float | None:
    """Zhang's single-homography focal-length estimation.

    H maps court (2D) → image (pixels).  The columns of ``K⁻¹ · H`` should
    satisfy orthogonality and equal-norm constraints on the first two rotation
    columns, yielding a quadratic in f².
    """
    h1 = H[:, 0]
    h2 = H[:, 1]

    def _ab(hi: NDArray, hj: NDArray) -> tuple[float, float]:
        a = (
            hi[0] * hj[0]
            + hi[1] * hj[1]
            - cx * (hi[0] * hj[2] + hi[2] * hj[0])
            - cy * (hi[1] * hj[2] + hi[2] * hj[1])
            + (cx**2 + cy**2) * hi[2] * hj[2]
        )
        b = hi[2] * hj[2]
        return float(a), float(b)

    # Orthogonality: A12/f² + B12 = 0 → f² = −A12/B12
    a12, b12 = _ab(h1, h2)
    f2_orth: float | None = None
    if abs(b12) > 1e-12:
        f2_orth = -a12 / b12

    # Equal norm: (A11 − A22)/f² = B22 − B11 → f² = (A11−A22)/(B22−B11)
    a11, b11 = _ab(h1, h1)
    a22, b22 = _ab(h2, h2)
    f2_norm: float | None = None
    denom = b22 - b11
    if abs(denom) > 1e-12:
        f2_norm = (a11 - a22) / denom

    # Pick best.
    if f2_orth is not None and f2_orth > 0:
        f2 = f2_orth
        if f2_norm is not None and f2_norm > 0:
            ratio = f2_orth / f2_norm
            if 0.5 < ratio < 2.0:
                f2 = math.exp((math.log(f2_orth) + math.log(f2_norm)) / 2.0)
    elif f2_norm is not None and f2_norm > 0:
        f2 = f2_norm
    else:
        return None

    return math.sqrt(f2)


def _grid_search_focal(
    obj_pts: NDArray[np.float64],
    img_pts: NDArray[np.float64],
    cx: float,
    cy: float,
    min_f: float,
    max_f: float,
    steps: int = 20,
) -> float | None:
    """Brute-force search for the focal length that minimises reprojection."""
    best_f: float | None = None
    best_err = float("inf")

    for i in range(steps + 1):
        f = min_f + (max_f - min_f) * i / steps
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, distCoeffs=np.zeros(4), flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            continue
        if not ok:
            continue
        R = np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64)
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        cam_pos: NDArray[np.float64] = -R.T @ t
        if cam_pos[2] < _MIN_CAM_HEIGHT or cam_pos[2] > _MAX_CAM_HEIGHT:
            continue
        P: NDArray[np.float64] = K @ np.hstack([R, t.reshape(3, 1)])
        err = _reproj_err_px(P, obj_pts, img_pts)
        if err < best_err:
            best_err = err
            best_f = f

    return best_f


def _refine_focal(
    base: CameraModel,
    obj_pts: NDArray[np.float64],
    img_pts: NDArray[np.float64],
    cx: float,
    cy: float,
    w: int,
    h: int,
    steps: int = 20,
) -> CameraModel | None:
    """Refine focal length around the current estimate."""
    f0 = base.focal_length_px
    lo = f0 * 0.7
    hi = f0 * 1.3
    best: CameraModel | None = None

    for i in range(steps + 1):
        f = lo + (hi - lo) * i / steps
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, distCoeffs=np.zeros(4), flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            continue
        if not ok:
            continue
        cand = _build_model(
            K, np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64),
            obj_pts, img_pts, w, h,
        )
        if cand is not None and (best is None or cand.reprojection_error < best.reprojection_error):
            best = cand

    return best


def _build_model(
    K: NDArray[np.float64],
    rvec: NDArray[np.float64],
    tvec: NDArray[np.float64],
    obj_pts: NDArray[np.float64],
    img_pts: NDArray[np.float64],
    image_width: int,
    image_height: int,
) -> CameraModel | None:
    """Validate a solvePnP solution and build a ``CameraModel``."""
    R = np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    cam_pos: NDArray[np.float64] = -R.T @ t

    if cam_pos[2] < _MIN_CAM_HEIGHT or cam_pos[2] > _MAX_CAM_HEIGHT:
        return None

    # Court centre must be in front of the camera.
    court_centre = np.array([COURT_WIDTH / 2.0, COURT_LENGTH / 2.0, 0.0])
    depth = float((R @ court_centre + t)[2])
    if depth <= 0:
        return None

    P: NDArray[np.float64] = K @ np.hstack([R, t.reshape(3, 1)])
    err = _reproj_err_px(P, obj_pts, img_pts)

    return CameraModel(
        intrinsic_matrix=K.copy(),
        rotation=R,
        translation=t,
        projection_matrix=P,
        camera_position=cam_pos,
        focal_length_px=float(K[0, 0]),
        reprojection_error=err,
        image_size=(image_width, image_height),
        is_valid=err < _MAX_REPROJ_ERROR_PX,
    )


def _reproj_err_px(
    P: NDArray[np.float64],
    obj_pts: NDArray[np.float64],
    img_pts: NDArray[np.float64],
) -> float:
    """Mean reprojection error in pixels."""
    total = 0.0
    n = len(obj_pts)
    for i in range(n):
        pt = obj_pts[i]
        h = P @ np.array([pt[0], pt[1], pt[2], 1.0])
        if abs(h[2]) < 1e-10:
            return float("inf")
        total += math.sqrt((h[0] / h[2] - img_pts[i, 0]) ** 2 + (h[1] / h[2] - img_pts[i, 1]) ** 2)
    return total / n
