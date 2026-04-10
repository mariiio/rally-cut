"""Unit tests for 3D trajectory fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from rallycut.court.camera_model import CameraModel, calibrate_camera, project_3d_to_image
from rallycut.court.trajectory_3d import (
    GRAVITY,
    FittedArc,
    fit_arc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COURT_CORNERS = [(0.0, 0.0), (8.0, 0.0), (8.0, 16.0), (0.0, 16.0)]


def _make_camera(
    cam_pos: np.ndarray | None = None,
    focal: float = 1800.0,
) -> CameraModel:
    """Build a synthetic camera behind the near baseline."""
    import cv2

    w, h = 1920, 1080
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)

    if cam_pos is None:
        cam_pos = np.array([4.0, -5.0, 5.0])
    target = np.array([4.0, 8.0, 0.0])
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    R = np.array([right, -up, fwd], dtype=np.float64)
    t = -R @ cam_pos
    P = K @ np.hstack([R, t.reshape(3, 1)])

    # Project court corners to normalised coords.
    img_corners: list[tuple[float, float]] = []
    for cx, cy in COURT_CORNERS:
        pt = P @ np.array([cx, cy, 0.0, 1.0])
        img_corners.append((float(pt[0] / pt[2] / w), float(pt[1] / pt[2] / h)))

    cam = calibrate_camera(img_corners, COURT_CORNERS, w, h)
    assert cam is not None and cam.is_valid
    return cam


@dataclass
class FakeBallPosition:
    """Minimal stand-in for BallPosition in tests."""

    frame_number: int
    x: float
    y: float
    confidence: float
    motion_energy: float = 0.0


def _generate_arc_observations(
    camera: CameraModel,
    pos0: np.ndarray,
    vel0: np.ndarray,
    fps: float,
    n_frames: int,
    noise_px: float = 0.0,
) -> list[FakeBallPosition]:
    """Generate synthetic 2D observations from a known 3D parabola."""
    obs: list[FakeBallPosition] = []
    rng = np.random.RandomState(42)

    for i in range(n_frames):
        t = i / fps
        pt3d = np.array([
            pos0[0] + vel0[0] * t,
            pos0[1] + vel0[1] * t,
            pos0[2] + vel0[2] * t - 0.5 * GRAVITY * t**2,
        ])
        u, v = project_3d_to_image(camera, pt3d)

        if noise_px > 0:
            w, h = camera.image_size
            u += rng.normal(0, noise_px / w)
            v += rng.normal(0, noise_px / h)

        obs.append(FakeBallPosition(
            frame_number=i,
            x=float(u),
            y=float(v),
            confidence=0.9,
        ))

    return obs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFitArc:
    """Test parabolic arc fitting with synthetic data."""

    def test_recovers_serve_speed(self) -> None:
        """Fit a serve arc and check the recovered speed is close to truth."""
        camera = _make_camera()
        fps = 30.0

        # Serve: starting at (1, 1, 3), going toward far court at 20 m/s.
        pos0 = np.array([1.0, 1.0, 3.0])
        vel0 = np.array([2.0, 18.0, 3.0])  # ~18.6 m/s total
        true_speed = float(np.linalg.norm(vel0))

        observations = _generate_arc_observations(camera, pos0, vel0, fps, n_frames=30)
        arc = fit_arc(camera, observations, 0, 29, fps, z0_prior=3.0)  # type: ignore[arg-type]

        assert arc is not None
        assert arc.is_valid
        # Speed should be within 20% of truth.
        assert abs(arc.speed_at_start - true_speed) / true_speed < 0.20, (
            f"Speed {arc.speed_at_start:.1f} vs truth {true_speed:.1f}"
        )

    def test_recovers_with_noise(self) -> None:
        """Fit should be robust to moderate pixel noise."""
        camera = _make_camera()
        fps = 30.0

        pos0 = np.array([4.0, 3.0, 2.5])
        vel0 = np.array([1.0, 10.0, 5.0])
        true_speed = float(np.linalg.norm(vel0))

        observations = _generate_arc_observations(
            camera, pos0, vel0, fps, n_frames=45, noise_px=2.0,
        )
        arc = fit_arc(camera, observations, 0, 44, fps, z0_prior=2.5)  # type: ignore[arg-type]

        assert arc is not None
        assert arc.is_valid
        assert abs(arc.speed_at_start - true_speed) / true_speed < 0.30

    def test_peak_height_plausible(self) -> None:
        """Peak height should be close to the analytical value."""
        camera = _make_camera()
        fps = 30.0

        pos0 = np.array([4.0, 4.0, 2.0])
        vel0 = np.array([0.5, 8.0, 6.0])
        # Analytical peak: z0 + vz0²/(2g) = 2.0 + 36/(2*9.81) ≈ 3.84
        true_peak = 2.0 + 6.0**2 / (2 * GRAVITY)

        observations = _generate_arc_observations(camera, pos0, vel0, fps, n_frames=40)
        arc = fit_arc(camera, observations, 0, 39, fps, z0_prior=2.0)  # type: ignore[arg-type]

        assert arc is not None
        assert abs(arc.peak_height - true_peak) < 1.5, (
            f"Peak {arc.peak_height:.1f} vs truth {true_peak:.1f}"
        )

    def test_too_few_observations_returns_none(self) -> None:
        camera = _make_camera()
        obs = [FakeBallPosition(i, 0.5, 0.5, 0.9) for i in range(3)]
        arc = fit_arc(camera, obs, 0, 2, 30.0)  # type: ignore[arg-type]
        assert arc is None

    def test_gravity_residual_near_zero(self) -> None:
        """Free-gravity refit should recover g ≈ 9.81."""
        camera = _make_camera()
        fps = 30.0

        pos0 = np.array([4.0, 3.0, 2.5])
        vel0 = np.array([1.0, 12.0, 4.0])

        observations = _generate_arc_observations(camera, pos0, vel0, fps, n_frames=45)
        arc = fit_arc(camera, observations, 0, 44, fps, z0_prior=2.5)  # type: ignore[arg-type]

        assert arc is not None
        assert arc.gravity_residual is not None
        assert abs(arc.gravity_residual) < 0.3, (
            f"Gravity residual {arc.gravity_residual:.2f} (expected near 0)"
        )

    def test_net_constraint_improves_low_camera(self) -> None:
        """Net-crossing constraint should improve serve speed recovery from low camera."""
        # Low camera at 1.5m — typical amateur sideline setup.
        camera = _make_camera(cam_pos=np.array([4.0, -3.0, 1.5]))
        fps = 30.0

        # Serve crossing the net: start near baseline, travel to far court.
        pos0 = np.array([2.0, 1.0, 3.0])
        vel0 = np.array([1.5, 16.0, 2.0])  # ~16.3 m/s
        true_speed = float(np.linalg.norm(vel0))

        observations = _generate_arc_observations(
            camera, pos0, vel0, fps, n_frames=25, noise_px=1.0,
        )

        # With net constraint (default): should recover speed reasonably.
        arc_constrained = fit_arc(
            camera, observations, 0, 24, fps,  # type: ignore[arg-type]
            z0_prior=3.0, net_height=2.24,
        )
        assert arc_constrained is not None
        assert arc_constrained.is_valid

        # Net crossing height should be plausible (ball must clear 2.24m net).
        if arc_constrained.net_crossing_height is not None:
            assert arc_constrained.net_crossing_height >= 1.5, (
                f"Net crossing {arc_constrained.net_crossing_height:.1f}m too low"
            )

        # Speed should be within 30% of truth.
        speed_err = abs(arc_constrained.speed_at_start - true_speed) / true_speed
        assert speed_err < 0.30, (
            f"Constrained speed {arc_constrained.speed_at_start:.1f} vs "
            f"truth {true_speed:.1f} ({speed_err:.0%} error)"
        )

    def test_landing_constraint(self) -> None:
        """Landing constraint should push end-of-arc height toward ground."""
        camera = _make_camera()
        fps = 30.0

        # Attack arc that should land on the court (z reaches 0).
        pos0 = np.array([4.0, 10.0, 3.0])
        vel0 = np.array([1.0, 4.0, -2.0])  # descending
        # Time to hit z=0: solve 3 - 2t - 0.5*9.81*t^2 = 0 → t ≈ 0.40s
        n_frames = 12  # ~0.4s at 30fps

        observations = _generate_arc_observations(
            camera, pos0, vel0, fps, n_frames=n_frames,
        )
        arc = fit_arc(
            camera, observations, 0, n_frames - 1, fps,  # type: ignore[arg-type]
            z0_prior=3.0, landing=True,
        )
        assert arc is not None
