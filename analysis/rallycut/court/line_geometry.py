"""Geometric utilities for line detection and court fitting.

Functions for line intersection, vanishing point computation,
harmonic conjugate (projective midpoint), parametric line
representations, and court model mapping used by the court detector.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

# ── Beach Volleyball Court Model ────────────────────────────────────────
# Court dimensions: 8m wide × 16m long (8m per side).
# Court-space coordinates: origin at top-left corner (0,0),
# x-axis along width (0→8), y-axis along length (0→16).
# Lines: far_baseline (y=16), near_baseline (y=0),
#         left_sideline (x=0), right_sideline (x=8),
#         center_line (y=8).

COURT_MODEL_CORNERS: list[tuple[float, float]] = [
    (0.0, 0.0),   # near-left
    (8.0, 0.0),   # near-right
    (8.0, 16.0),  # far-right
    (0.0, 16.0),  # far-left
]

# Map each pair of court line labels to their known court-space intersection.
# Only includes intersections that exist on the real court.
COURT_MODEL_INTERSECTIONS: dict[frozenset[str], tuple[float, float]] = {
    frozenset({"far_baseline", "left_sideline"}): (0.0, 16.0),    # far-left
    frozenset({"far_baseline", "right_sideline"}): (8.0, 16.0),   # far-right
    frozenset({"near_baseline", "left_sideline"}): (0.0, 0.0),    # near-left
    frozenset({"near_baseline", "right_sideline"}): (8.0, 0.0),   # near-right
    frozenset({"center_line", "left_sideline"}): (0.0, 8.0),      # center-left
    frozenset({"center_line", "right_sideline"}): (8.0, 8.0),     # center-right
}


def line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Compute intersection of two lines defined by point pairs.

    Line 1 passes through p1 and p2, line 2 passes through p3 and p4.
    Uses homogeneous coordinates for numerical stability.

    Returns:
        Intersection point (x, y), or None if lines are parallel.
    """
    # Convert to homogeneous coordinates
    l1 = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    l2 = np.cross([p3[0], p3[1], 1.0], [p4[0], p4[1], 1.0])

    # Intersection is cross product of lines
    pt = np.cross(l1, l2)

    if abs(pt[2]) < 1e-10:
        return None  # Parallel lines

    return (float(pt[0] / pt[2]), float(pt[1] / pt[2]))


def point_line_distance(
    point: tuple[float, float],
    line_p1: tuple[float, float],
    line_p2: tuple[float, float],
) -> float:
    """Perpendicular distance from point to line defined by two points.

    Args:
        point: Query point (x, y).
        line_p1: First point on line.
        line_p2: Second point on line.

    Returns:
        Unsigned perpendicular distance.
    """
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1e-10:
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / length


def compute_vanishing_point(
    lines: list[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[float, float] | None:
    """Compute vanishing point as least-squares intersection of multiple lines.

    Each line is given as a pair of points ((x1,y1), (x2,y2)).

    Returns:
        Vanishing point (x, y), or None if fewer than 2 lines.
    """
    if len(lines) < 2:
        return None

    # Build system Ax = b where each line contributes one equation:
    # (y2-y1)*x - (x2-x1)*y = x1*(y2-y1) - y1*(x2-x1)
    # Equivalently: a*x + b*y = c with a=dy, b=-dx, c=x1*dy - y1*dx
    a_rows: list[list[float]] = []
    b_rows: list[float] = []

    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            continue
        a = dy
        b = -dx
        c = x1 * dy - y1 * dx
        a_rows.append([a, b])
        b_rows.append(c)

    if len(a_rows) < 2:
        return None

    a_mat = np.array(a_rows, dtype=np.float64)
    b_vec = np.array(b_rows, dtype=np.float64)

    # Least-squares solution
    try:
        result, residuals, rank, sv = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        if rank < 2:
            return None
        return (float(result[0]), float(result[1]))
    except np.linalg.LinAlgError:
        return None


def harmonic_conjugate(
    far_pt: tuple[float, float],
    center_pt: tuple[float, float],
    vanishing_pt: tuple[float, float],
) -> tuple[float, float]:
    """Compute the harmonic conjugate of far_pt w.r.t. center_pt and vanishing_pt.

    In projective geometry, if center_pt divides far_pt and the result
    in a harmonic range with vanishing_pt, the result is the "opposite"
    point such that center_pt is the projective midpoint.

    For a court: far_pt is a far corner, center_pt is where the center line
    intersects the sideline, vanishing_pt is the vanishing point. The result
    is the near corner (the center line divides the court exactly in half).

    Uses the cross-ratio relation: (A,B;C,D) = -1 where A=far, B=near,
    C=center, D=vanishing. Solving for B given A, C, D.
    """
    # Use homogeneous coordinates on the line
    # Points on the line parameterized as: P = far + t * (vanishing - far)
    # Find parameter for center_pt
    fx, fy = far_pt
    cx, cy = center_pt
    vx, vy = vanishing_pt

    dx = vx - fx
    dy = vy - fy

    # Parameter t_c for center point: center = far + t_c * (vanishing - far)
    if abs(dx) > abs(dy):
        t_c = (cx - fx) / dx if abs(dx) > 1e-10 else 0.0
    else:
        t_c = (cy - fy) / dy if abs(dy) > 1e-10 else 0.0

    # Vanishing point has t_v = 1.0
    # Far point has t_f = 0.0
    # Harmonic conjugate: t_near such that (t_f, t_near; t_c, t_v) = -1
    # Cross-ratio: (0 - t_c)/(0 - 1) * (t_n - 1)/(t_n - t_c) = -1
    # => t_c * (t_n - 1) / (t_n - t_c) = -1
    # => t_c * t_n - t_c = -t_n + t_c
    # => t_n * (t_c + 1) = 2 * t_c
    # => t_n = 2 * t_c / (t_c + 1)

    if abs(t_c + 1.0) < 1e-10:
        # Degenerate: center is at the vanishing point complement
        return far_pt

    t_n = 2.0 * t_c / (t_c + 1.0)

    near_x = fx + t_n * dx
    near_y = fy + t_n * dy

    return (near_x, near_y)


def segment_to_rho_theta(
    x1: float, y1: float, x2: float, y2: float,
) -> tuple[float, float]:
    """Convert a line segment to (rho, theta) parametric representation.

    Uses the midpoint and angle of the segment. Theta is the angle
    of the line normal (perpendicular to segment direction), in radians.
    Rho is the signed perpendicular distance from origin to the line.

    Args:
        x1, y1, x2, y2: Segment endpoints in normalized coordinates.

    Returns:
        (rho, theta) where rho is distance from origin, theta in [-pi, pi].
    """
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0

    # Angle of the segment direction
    dx = x2 - x1
    dy = y2 - y1
    segment_angle = math.atan2(dy, dx)

    # Normal angle (perpendicular to segment)
    theta = segment_angle + math.pi / 2.0
    # Normalize to [-pi, pi]
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi

    # Rho = perpendicular distance from origin to the line
    rho = mx * math.cos(theta) + my * math.sin(theta)

    # Ensure rho >= 0 by flipping if needed
    if rho < 0:
        rho = -rho
        theta = theta + math.pi
        if theta > math.pi:
            theta -= 2 * math.pi

    return (rho, theta)


def cross2d(
    o: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """2D cross product of vectors OA and OB. Positive if counter-clockwise."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segment_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Angle of segment from horizontal, in degrees [0, 180)."""
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    return angle


def segment_length(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean length of a segment."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def segments_to_median_line(
    segments: list[tuple[float, float, float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Compute median line from a cluster of segments.

    Returns the line as two points spanning a reasonable extent.

    Args:
        segments: List of (x1, y1, x2, y2) in normalized coords.

    Returns:
        Two-point representation ((x1,y1), (x2,y2)), or None if empty.
    """
    if not segments:
        return None

    # Collect all endpoints sorted by X
    all_points: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in segments:
        all_points.append((x1, y1))
        all_points.append((x2, y2))

    # Use rho/theta median for the line direction
    rho_vals: list[float] = []
    theta_vals: list[float] = []
    for x1, y1, x2, y2 in segments:
        rho, theta = segment_to_rho_theta(x1, y1, x2, y2)
        rho_vals.append(rho)
        theta_vals.append(theta)

    med_rho = float(np.median(rho_vals))
    med_theta = float(np.median(theta_vals))

    # Reconstruct line from rho/theta
    cos_t = math.cos(med_theta)
    sin_t = math.sin(med_theta)

    # Point on the line closest to origin
    px = med_rho * cos_t
    py = med_rho * sin_t

    # Direction along the line (perpendicular to normal)
    dx = -sin_t
    dy = cos_t

    # Extend line across [0,1] range
    # Project all endpoints onto the line direction to find extent
    projections = [(p[0] - px) * dx + (p[1] - py) * dy for p in all_points]
    min_t = min(projections)
    max_t = max(projections)

    p1 = (px + min_t * dx, py + min_t * dy)
    p2 = (px + max_t * dx, py + max_t * dy)

    return (p1, p2)


# ── Court Model Correspondence Functions ─────────────────────────────────


def collect_court_correspondences(
    identified_lines: Mapping[str, Any],
    bounds: tuple[float, float, float, float] = (-1.0, -1.0, 2.0, 3.0),
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Collect image↔court point correspondences from identified court lines.

    For each pair of identified lines that have a known court-space intersection
    in COURT_MODEL_INTERSECTIONS, compute their image-space intersection and
    pair it with the court-space coordinate.

    Args:
        identified_lines: Dict mapping line labels to (DetectedLine, segments) tuples.
            DetectedLine must have .p1 and .p2 attributes.
        bounds: (min_x, min_y, max_x, max_y) filter for image-space intersections.
            Wide bounds allow off-screen near corners (common in beach volleyball
            where near baseline is below the frame).

    Returns:
        List of (image_point, court_point) pairs.
    """
    min_x, min_y, max_x, max_y = bounds
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]] = []

    labels = list(identified_lines.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            key = frozenset({labels[i], labels[j]})
            court_pt = COURT_MODEL_INTERSECTIONS.get(key)
            if court_pt is None:
                continue

            line_a = identified_lines[labels[i]]
            line_b = identified_lines[labels[j]]
            # Extract DetectedLine (first element of the tuple)
            dl_a = line_a[0] if isinstance(line_a, tuple) else line_a
            dl_b = line_b[0] if isinstance(line_b, tuple) else line_b

            img_pt = line_intersection(dl_a.p1, dl_a.p2, dl_b.p1, dl_b.p2)
            if img_pt is None:
                continue

            # Filter out intersections far outside the frame
            if not (min_x <= img_pt[0] <= max_x and min_y <= img_pt[1] <= max_y):
                continue

            correspondences.append((img_pt, court_pt))

    return correspondences


def project_court_corners(
    h_court_to_image: np.ndarray,
    court_corners: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Project court-space corners through a homography to image space.

    Args:
        h_court_to_image: 3×3 homography mapping court coords → image coords.
        court_corners: Court corners in court space. Defaults to COURT_MODEL_CORNERS.

    Returns:
        List of 4 image-space points (x, y) in same order as input corners.
    """
    if court_corners is None:
        court_corners = COURT_MODEL_CORNERS

    pts = np.array(court_corners, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, h_court_to_image)
    return [(float(p[0][0]), float(p[0][1])) for p in projected]
