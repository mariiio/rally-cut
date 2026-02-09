"""Court calibration for image-to-court coordinate projection.

Provides homography-based projection from image coordinates to real-world
court coordinates (in meters).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Beach volleyball court dimensions (in meters)
COURT_WIDTH = 8.0  # meters
COURT_LENGTH = 16.0  # meters (8m per side)


class CourtType(Enum):
    """Type of volleyball court (beach only for now)."""

    BEACH = "beach"

    def __init__(self, value: str) -> None:
        self._value_ = value.lower()

    @property
    def width(self) -> float:
        """Court width in meters."""
        return COURT_WIDTH

    @property
    def length(self) -> float:
        """Court length in meters."""
        return COURT_LENGTH


@dataclass
class HomographyResult:
    """Result of homography calibration."""

    homography: NDArray[np.float64]  # 3x3 transformation matrix
    image_corners: list[tuple[float, float]]  # 4 corners in image coords (normalized 0-1)
    court_corners: list[tuple[float, float]]  # 4 corners in court coords (meters)
    error: float = 0.0  # Reprojection error
    is_valid: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HomographyResult:
        """Create from dictionary (e.g., from JSON)."""
        matrix = np.array(data["matrix"], dtype=np.float64).reshape(3, 3)
        return cls(
            homography=matrix,
            image_corners=data.get("image_corners", []),
            court_corners=data.get("court_corners", []),
            error=data.get("reprojectionError", 0.0),
            is_valid=data.get("isValid", True),
        )


class CourtCalibrator:
    """
    Court calibration for projecting image coordinates to court coordinates.

    Uses a homography transformation between image space and real-world court space.
    Calibration requires 4 corner points of the court visible in the image.

    Beach volleyball court is 8m x 16m (8m per side, split by net).
    """

    def __init__(self, court_type: CourtType = CourtType.BEACH) -> None:
        self._homography: HomographyResult | None = None
        self.court_type = court_type

    def load_calibration(self, homography: HomographyResult) -> None:
        """Load a pre-computed homography result."""
        self._homography = homography

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration has been performed."""
        return self._homography is not None

    @property
    def homography(self) -> HomographyResult | None:
        """Get the current homography result."""
        return self._homography

    def calibrate(
        self,
        image_corners: list[tuple[float, float]],
        court_corners: list[tuple[float, float]] | None = None,
    ) -> HomographyResult:
        """
        Calibrate using 4 court corners.

        Args:
            image_corners: 4 corner points in normalized image coordinates (0-1).
                          Order: top-left, top-right, bottom-right, bottom-left
            court_corners: Optional court corner coordinates in meters.
                          If not provided, uses standard beach volleyball court.

        Returns:
            HomographyResult with transformation matrix and corners.
        """
        if len(image_corners) != 4:
            raise ValueError(f"Expected 4 corners, got {len(image_corners)}")

        # Default to standard beach volleyball court
        if court_corners is None:
            court_corners = [
                (0.0, 0.0),  # top-left (near sideline)
                (COURT_WIDTH, 0.0),  # top-right
                (COURT_WIDTH, COURT_LENGTH),  # bottom-right
                (0.0, COURT_LENGTH),  # bottom-left
            ]

        # Convert to numpy arrays
        src_pts = np.array(image_corners, dtype=np.float64)
        dst_pts = np.array(court_corners, dtype=np.float64)

        # Compute homography
        try:
            import cv2
            h_matrix, _ = cv2.findHomography(src_pts, dst_pts)
            if h_matrix is None:
                raise ValueError("Failed to compute homography")
            homography = np.array(h_matrix, dtype=np.float64)
        except ImportError:
            # Fallback: simple linear interpolation (less accurate)
            homography = np.eye(3, dtype=np.float64)

        self._homography = HomographyResult(
            homography=homography,
            image_corners=list(image_corners),
            court_corners=list(court_corners),
        )

        return self._homography

    def image_to_court(
        self,
        image_point: tuple[float, float],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        """
        Project an image point to court coordinates.

        Args:
            image_point: Point in normalized image coordinates (0-1).
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            Point in court coordinates (meters).

        Raises:
            RuntimeError: If not calibrated.
        """
        if not self.is_calibrated or self._homography is None:
            raise RuntimeError("Calibrator not calibrated")

        # Normalize point
        x, y = image_point

        # Apply homography
        pt = np.array([x, y, 1.0], dtype=np.float64)
        result = self._homography.homography @ pt
        result = result / result[2]  # Normalize by homogeneous coordinate

        return (float(result[0]), float(result[1]))

    def court_to_image(
        self,
        court_point: tuple[float, float],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        """
        Project a court point to image coordinates.

        Args:
            court_point: Point in court coordinates (meters).
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            Point in normalized image coordinates (0-1).

        Raises:
            RuntimeError: If not calibrated.
        """
        if not self.is_calibrated or self._homography is None:
            raise RuntimeError("Calibrator not calibrated")

        # Compute inverse homography
        h_inv = np.linalg.inv(self._homography.homography)

        # Apply inverse homography
        x, y = court_point
        pt = np.array([x, y, 1.0], dtype=np.float64)
        result = h_inv @ pt
        result = result / result[2]  # Normalize by homogeneous coordinate

        return (float(result[0]), float(result[1]))

    def is_point_in_court(
        self,
        court_point: tuple[float, float],
        margin: float = 0.0,
    ) -> bool:
        """
        Check if a court point is within court bounds.

        Args:
            court_point: Point in court coordinates (meters).
            margin: Extra margin around court (positive = outside allowed).

        Returns:
            True if point is within court bounds (with margin).
        """
        x, y = court_point
        return (
            -margin <= x <= COURT_WIDTH + margin
            and -margin <= y <= COURT_LENGTH + margin
        )
