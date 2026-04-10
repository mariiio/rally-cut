"""Unit tests for camera model extraction from court corners."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rallycut.court.camera_model import (
    CameraModel,
    calibrate_camera,
    image_ray,
    project_3d_to_image,
)


def _make_synthetic_camera(
    cam_pos: tuple[float, float, float] = (4.0, -5.0, 5.0),
    look_at: tuple[float, float, float] = (4.0, 8.0, 0.0),
    focal_length_px: float = 1800.0,
    image_size: tuple[int, int] = (1920, 1080),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic K, R, t looking at *look_at* from *cam_pos*.

    Returns (K, R, t) where R is world→camera rotation and t is the
    translation such that ``x_cam = R·x_world + t``.
    """
    w, h = image_size
    K = np.array(
        [[focal_length_px, 0, w / 2.0], [0, focal_length_px, h / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )

    # Camera coordinate system
    cam = np.array(cam_pos, dtype=np.float64)
    target = np.array(look_at, dtype=np.float64)
    forward = target - cam
    forward = forward / np.linalg.norm(forward)

    # Use world-up (0,0,1) to derive right and camera-up.
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # R maps world → camera.  Camera axes: right=X, down=Y, forward=Z.
    R = np.array([right, -up, forward], dtype=np.float64)
    t = -R @ cam

    return K, R, t


def _project(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, pt3d: np.ndarray
) -> tuple[float, float]:
    """Project a 3D point to pixel coordinates."""
    P = K @ np.hstack([R, t.reshape(3, 1)])
    h = P @ np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
    return (float(h[0] / h[2]), float(h[1] / h[2]))


# Beach volleyball court corners (metres, z=0).
COURT_CORNERS = [(0.0, 0.0), (8.0, 0.0), (8.0, 16.0), (0.0, 16.0)]


class TestCalibrateCamera:
    """Test camera recovery from synthetic court-corner projections."""

    def test_recovers_camera_behind_baseline(self) -> None:
        """Camera behind near baseline, looking at court centre."""
        K, R, t = _make_synthetic_camera(cam_pos=(4.0, -5.0, 5.0))
        w, h = 1920, 1080

        # Project court corners to normalised image coords.
        image_corners: list[tuple[float, float]] = []
        for cx, cy in COURT_CORNERS:
            px, py = _project(K, R, t, np.array([cx, cy, 0.0]))
            image_corners.append((px / w, py / h))

        cam = calibrate_camera(image_corners, COURT_CORNERS, w, h)

        assert cam is not None
        assert cam.is_valid
        assert cam.reprojection_error < 2.0  # pixels

        # Camera height should be close to ground truth.
        assert abs(cam.camera_position[2] - 5.0) < 1.0

    def test_recovers_camera_from_side(self) -> None:
        """Camera to the side of the court."""
        K, R, t = _make_synthetic_camera(cam_pos=(-6.0, 8.0, 4.0))
        w, h = 1920, 1080

        image_corners: list[tuple[float, float]] = []
        for cx, cy in COURT_CORNERS:
            px, py = _project(K, R, t, np.array([cx, cy, 0.0]))
            image_corners.append((px / w, py / h))

        cam = calibrate_camera(image_corners, COURT_CORNERS, w, h)

        assert cam is not None
        assert cam.is_valid
        assert cam.reprojection_error < 2.0

    def test_returns_none_for_degenerate_input(self) -> None:
        """All corners at the same pixel should fail."""
        corners = [(0.5, 0.5)] * 4
        result = calibrate_camera(corners, COURT_CORNERS, 1920, 1080)
        # Should either return None or an invalid model.
        assert result is None or not result.is_valid

    def test_returns_none_for_wrong_count(self) -> None:
        assert calibrate_camera([(0.1, 0.1)], COURT_CORNERS, 1920, 1080) is None
        assert calibrate_camera([], [], 1920, 1080) is None


class TestProject3dToImage:
    """Test forward projection through the recovered camera."""

    def test_court_corner_roundtrip(self) -> None:
        """Project a court corner through the recovered camera and check it
        matches the original image position."""
        K, R, t = _make_synthetic_camera()
        w, h = 1920, 1080

        image_corners: list[tuple[float, float]] = []
        for cx, cy in COURT_CORNERS:
            px, py = _project(K, R, t, np.array([cx, cy, 0.0]))
            image_corners.append((px / w, py / h))

        cam = calibrate_camera(image_corners, COURT_CORNERS, w, h)
        assert cam is not None and cam.is_valid

        for i, (cx, cy) in enumerate(COURT_CORNERS):
            u, v = project_3d_to_image(cam, np.array([cx, cy, 0.0]))
            orig_u, orig_v = image_corners[i]
            assert abs(u - orig_u) < 0.005, f"Corner {i}: u={u}, expected={orig_u}"
            assert abs(v - orig_v) < 0.005, f"Corner {i}: v={v}, expected={orig_v}"

    def test_above_ground_projection(self) -> None:
        """A point above the court should project to a different image
        position than its ground shadow."""
        K, R, t = _make_synthetic_camera()
        w, h = 1920, 1080

        image_corners: list[tuple[float, float]] = []
        for cx, cy in COURT_CORNERS:
            px, py = _project(K, R, t, np.array([cx, cy, 0.0]))
            image_corners.append((px / w, py / h))

        cam = calibrate_camera(image_corners, COURT_CORNERS, w, h)
        assert cam is not None

        ground = project_3d_to_image(cam, np.array([4.0, 8.0, 0.0]))
        above = project_3d_to_image(cam, np.array([4.0, 8.0, 3.0]))

        # Ball above ground should project higher in the image (smaller v).
        assert above[1] < ground[1], "Above-ground point should appear higher"


class TestImageRay:
    """Test ray casting through the camera model."""

    def test_ray_hits_ground_near_court(self) -> None:
        """A ray through a court corner's image position should intersect
        the ground plane near that corner."""
        K, R, t = _make_synthetic_camera()
        w, h = 1920, 1080

        image_corners: list[tuple[float, float]] = []
        for cx, cy in COURT_CORNERS:
            px, py = _project(K, R, t, np.array([cx, cy, 0.0]))
            image_corners.append((px / w, py / h))

        cam = calibrate_camera(image_corners, COURT_CORNERS, w, h)
        assert cam is not None

        for i, (cx, cy) in enumerate(COURT_CORNERS):
            origin, direction = image_ray(cam, image_corners[i])

            # Intersect ray with z=0 plane: origin + t*direction, solve for z=0.
            if abs(direction[2]) < 1e-10:
                continue  # Ray parallel to ground — skip
            t_param = -origin[2] / direction[2]
            hit = origin + t_param * direction

            assert abs(hit[0] - cx) < 0.5, f"Corner {i}: x={hit[0]}, expected={cx}"
            assert abs(hit[1] - cy) < 0.5, f"Corner {i}: y={hit[1]}, expected={cy}"

    def test_ray_origin_is_camera_position(self) -> None:
        K, R, t = _make_synthetic_camera()
        w, h = 1920, 1080

        image_corners: list[tuple[float, float]] = []
        for cx, cy in COURT_CORNERS:
            px, py = _project(K, R, t, np.array([cx, cy, 0.0]))
            image_corners.append((px / w, py / h))

        cam = calibrate_camera(image_corners, COURT_CORNERS, w, h)
        assert cam is not None

        origin, _ = image_ray(cam, (0.5, 0.5))
        np.testing.assert_allclose(origin, cam.camera_position, atol=1e-6)
