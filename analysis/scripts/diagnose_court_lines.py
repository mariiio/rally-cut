#!/usr/bin/env python3
"""Diagnose court line visibility and Hough detectability for 4 videos.

For each of veve, papa, pipi, pepe:
  1. Load video from DB, sample middle frame
  2. Run Hough line detection, classify lines by type
  3. Run YOLO keypoint model for corner predictions
  4. Load GT/manual calibration from DB
  5. Save annotated visualization
  6. Print findings table

Line types classified:
  - far_baseline: near-horizontal, near the far corners
  - left_sideline: diagonal, passes near far-left corner
  - right_sideline: diagonal, passes near far-right corner
  - center_line: horizontal, between far baseline and net level
  - unknown: unclassified

Usage:
    cd analysis
    uv run python scripts/diagnose_court_lines.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.court.keypoint_detector import CourtKeypointDetector
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path, load_court_calibration

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "court_lines"
TARGET_NAMES = ["veve", "papa", "pipi", "pepe"]

CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]

# Colors (BGR)
COLOR_GT = (0, 200, 0)        # green — GT/manual corners
COLOR_YOLO = (0, 120, 255)    # orange — YOLO predicted corners
COLOR_FAR_BASELINE = (255, 0, 0)    # blue
COLOR_LEFT_SIDE = (0, 0, 255)       # red
COLOR_RIGHT_SIDE = (255, 0, 200)    # magenta
COLOR_CENTER_LINE = (0, 255, 255)   # yellow
COLOR_UNKNOWN = (150, 150, 150)     # gray


@dataclass
class LineInfo:
    """A detected Hough line with classification."""
    x1: int
    y1: int
    x2: int
    y2: int
    line_type: str  # far_baseline | left_sideline | right_sideline | center_line | unknown
    angle_deg: float   # degrees from horizontal, [-90, 90)
    r: float           # rho in Hough space
    theta: float       # theta in Hough space (radians)


@dataclass
class VideoFindings:
    """Findings for a single video."""
    name: str
    video_id: str
    has_gt: bool
    gt_corners: list[dict[str, float]] | None
    yolo_corners: list[dict[str, float]] | None
    yolo_confidence: float
    lines_by_type: dict[str, list[LineInfo]] = field(default_factory=dict)
    frame_shape: tuple[int, int, int] = (1080, 1920, 3)
    error: str | None = None


def query_videos_by_name(names: list[str]) -> dict[str, str]:
    """Query DB for video IDs by name patterns.

    Returns dict: {name -> video_id} for each found video.
    """
    result: dict[str, str] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            for name in names:
                cur.execute(
                    "SELECT id, filename FROM videos WHERE filename ILIKE %s LIMIT 1",
                    [f"%{name}%"],
                )
                row = cur.fetchone()
                if row:
                    video_id, source_file = str(row[0]), str(row[1])
                    result[name] = video_id
                    print(f"  Found {name}: id={video_id}, file={source_file}")
                else:
                    print(f"  NOT FOUND: {name}")
    return result


def load_gt_calibration_by_id(video_id: str) -> list[dict[str, float]] | None:
    """Load GT calibration corners from DB."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT court_calibration_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            cal = row[0]
            if isinstance(cal, str):
                cal = json.loads(cal)
            if isinstance(cal, list) and len(cal) == 4:
                return cal  # type: ignore[return-value]
            return None


def sample_middle_frame(video_path: Path) -> np.ndarray | None:
    """Sample the middle frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    mid = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def run_hough_lines(frame: np.ndarray) -> list[tuple[float, float]]:
    """Run HoughLines on a frame, return list of (rho, theta) pairs."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)
    if lines is None:
        return []
    return [(float(l[0][0]), float(l[0][1])) for l in lines]


def angle_from_theta(theta: float) -> float:
    """Convert Hough theta (radians) to angle from horizontal in degrees [-90, 90)."""
    # theta in [0, pi): normal to the line
    # angle of the line = theta - pi/2 (perpendicular to normal)
    angle = np.degrees(theta) - 90.0
    # Normalize to [-90, 90)
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return float(angle)


def line_endpoints(rho: float, theta: float, h: int, w: int) -> tuple[int, int, int, int]:
    """Convert (rho, theta) Hough line to pixel endpoints clipped to frame."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x0 = cos_t * rho
    y0 = sin_t * rho
    # Direction vector along the line (perpendicular to normal)
    dx = -sin_t
    dy = cos_t
    # Extend far enough to cover the frame
    scale = max(w, h) * 2
    x1 = int(round(x0 + scale * dx))
    y1 = int(round(y0 + scale * dy))
    x2 = int(round(x0 - scale * dx))
    y2 = int(round(y0 - scale * dy))

    # Clip to frame using Liang-Barsky or just let OpenCV handle it
    # We'll let the drawing handle clipping; just return raw points
    return x1, y1, x2, y2


def point_on_line(rho: float, theta: float, x: float, y: float) -> float:
    """Distance from point (x, y) to the Hough line defined by (rho, theta)."""
    # rho = x*cos(theta) + y*sin(theta) for points ON the line
    return abs(x * np.cos(theta) + y * np.sin(theta) - rho)


def classify_line(
    rho: float,
    theta: float,
    gt_corners: list[dict[str, float]] | None,
    h: int,
    w: int,
) -> str:
    """Classify a Hough line given GT corners (if available).

    Classification rules (fallback when no GT):
    - |angle| < 8 deg AND y-intercept near top 40% of frame -> far_baseline
    - |angle| < 8 deg AND y-intercept in mid 40-65% of frame -> center_line
    - angle significantly tilted (|angle| 10-50 deg) -> left or right sideline

    When GT corners available, use proximity to corner intersections.
    """
    angle = angle_from_theta(theta)

    if gt_corners and len(gt_corners) == 4:
        return _classify_with_gt(rho, theta, angle, gt_corners, h, w)
    else:
        return _classify_geometric(rho, theta, angle, h, w)


def _classify_with_gt(
    rho: float,
    theta: float,
    angle: float,
    gt_corners: list[dict[str, float]],
    h: int,
    w: int,
) -> str:
    """Classify using GT corners as anchors."""
    # GT corners: [near-left, near-right, far-right, far-left]
    # Convert to pixel coords
    corners_px = [(c["x"] * w, c["y"] * h) for c in gt_corners]
    near_left, near_right, far_right, far_left = corners_px

    # Distance thresholds
    thr_pt = max(h, w) * 0.06   # 6% of frame for point proximity
    thr_line = max(h, w) * 0.04  # 4% for point-to-line

    abs_angle = abs(angle)

    # Far baseline: near-horizontal passing near far-right and far-left
    if abs_angle < 15:
        d_far_right = point_on_line(rho, theta, far_right[0], far_right[1])
        d_far_left = point_on_line(rho, theta, far_left[0], far_left[1])
        if d_far_right < thr_line * 3 and d_far_left < thr_line * 3:
            return "far_baseline"

    # Left sideline: passes near far-left and near-left
    if abs_angle > 3:
        d_fl = point_on_line(rho, theta, far_left[0], far_left[1])
        d_nl = point_on_line(rho, theta, near_left[0], near_left[1])
        if d_fl < thr_line * 2 and d_nl < thr_line * 2:
            return "left_sideline"

    # Right sideline: passes near far-right and near-right
    if abs_angle > 3:
        d_fr = point_on_line(rho, theta, far_right[0], far_right[1])
        d_nr = point_on_line(rho, theta, near_right[0], near_right[1])
        if d_fr < thr_line * 2 and d_nr < thr_line * 2:
            return "right_sideline"

    # Fall back to geometry
    return _classify_geometric(rho, theta, angle, h, w)


def _classify_geometric(
    rho: float,
    theta: float,
    angle: float,
    h: int,
    w: int,
) -> str:
    """Classify using frame geometry only."""
    abs_angle = abs(angle)

    # Compute approximate y-intercept at frame center x
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # At x = w/2: y = (rho - (w/2)*cos_t) / sin_t  (if sin_t != 0)
    if abs(sin_t) > 1e-3:
        y_at_center = (rho - (w / 2) * cos_t) / sin_t
        y_norm = y_at_center / h
    else:
        # Vertical-ish line
        return "unknown"

    # Near-horizontal lines
    if abs_angle < 8:
        if 0.25 <= y_norm <= 0.55:
            return "far_baseline"
        if 0.55 <= y_norm <= 0.72:
            return "center_line"
        return "unknown"

    # Tilted lines (sidelines)
    if 10 <= abs_angle <= 60:
        # Check which side of frame the line passes through
        # x-intercept at y = h (near side)
        if abs(cos_t) > 1e-3:
            x_at_bottom = (rho - h * sin_t) / cos_t
            if x_at_bottom < w * 0.4:
                return "left_sideline"
            elif x_at_bottom > w * 0.6:
                return "right_sideline"

    return "unknown"


def classify_all_lines(
    raw_lines: list[tuple[float, float]],
    gt_corners: list[dict[str, float]] | None,
    h: int,
    w: int,
    max_lines: int = 60,
) -> list[LineInfo]:
    """Classify all Hough lines."""
    result: list[LineInfo] = []
    for rho, theta in raw_lines[:max_lines]:
        angle = angle_from_theta(theta)
        x1, y1, x2, y2 = line_endpoints(rho, theta, h, w)
        line_type = classify_line(rho, theta, gt_corners, h, w)
        result.append(LineInfo(
            x1=x1, y1=y1, x2=x2, y2=y2,
            line_type=line_type,
            angle_deg=angle,
            r=rho,
            theta=theta,
        ))
    return result


def deduplicate_lines(
    lines: list[LineInfo],
    angle_tol: float = 3.0,
    rho_tol_frac: float = 0.04,
    frame_diag: float = 1920.0,
) -> list[LineInfo]:
    """Deduplicate near-identical lines (same type, similar angle + rho)."""
    kept: list[LineInfo] = []
    rho_tol = rho_tol_frac * frame_diag
    for line in lines:
        duplicate = False
        for k in kept:
            if k.line_type != line.line_type:
                continue
            if abs(k.angle_deg - line.angle_deg) < angle_tol and abs(k.r - line.r) < rho_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
    return kept


def compute_line_corner_intersections(
    lines: list[LineInfo],
    gt_corners: list[dict[str, float]],
    h: int,
    w: int,
    thr_px: float = 60.0,
) -> list[dict[str, Any]]:
    """Find where detected lines intersect with each other near GT corners.

    For each pair of lines of different 'structural' types (e.g., sideline + baseline),
    compute intersection and check proximity to GT corners.

    Returns list of {corner_name, intersection_px, closest_gt_px, dist_px}.
    """
    from rallycut.court.line_geometry import line_intersection

    corner_names = CORNER_NAMES
    corners_px = [(c["x"] * w, c["y"] * h) for c in gt_corners]

    # Find sidelines and baseline
    left_side = [l for l in lines if l.line_type == "left_sideline"]
    right_side = [l for l in lines if l.line_type == "right_sideline"]
    baseline = [l for l in lines if l.line_type == "far_baseline"]

    results: list[dict[str, Any]] = []

    def hough_to_ab(ln: LineInfo) -> tuple[float, float, float] | None:
        """Convert Hough line to ax + by + c = 0 form."""
        cos_t = np.cos(ln.theta)
        sin_t = np.sin(ln.theta)
        # rho = x*cos + y*sin  =>  x*cos + y*sin - rho = 0
        return (cos_t, sin_t, -ln.r)

    def intersect_lines(
        l1: LineInfo, l2: LineInfo,
    ) -> tuple[float, float] | None:
        """Find pixel intersection of two Hough lines."""
        # Use two-point form
        p1 = np.array([l1.x1, l1.y1], dtype=float)
        p2 = np.array([l1.x2, l1.y2], dtype=float)
        p3 = np.array([l2.x1, l2.y1], dtype=float)
        p4 = np.array([l2.x2, l2.y2], dtype=float)

        d1 = p2 - p1
        d2 = p4 - p3
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-6:
            return None  # Parallel

        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
        ix = p1[0] + t * d1[0]
        iy = p1[1] + t * d1[1]
        return (float(ix), float(iy))

    # far-left corner: left sideline + far baseline
    for ls in left_side:
        for bl in baseline:
            pt = intersect_lines(ls, bl)
            if pt is None:
                continue
            gt_px = corners_px[3]  # far-left
            dist = np.hypot(pt[0] - gt_px[0], pt[1] - gt_px[1])
            results.append({
                "corner": "far-left",
                "intersection_px": pt,
                "gt_px": gt_px,
                "dist_px": float(dist),
                "lines": ("left_sideline", "far_baseline"),
            })

    # far-right corner: right sideline + far baseline
    for rs in right_side:
        for bl in baseline:
            pt = intersect_lines(rs, bl)
            if pt is None:
                continue
            gt_px = corners_px[2]  # far-right
            dist = np.hypot(pt[0] - gt_px[0], pt[1] - gt_px[1])
            results.append({
                "corner": "far-right",
                "intersection_px": pt,
                "gt_px": gt_px,
                "dist_px": float(dist),
                "lines": ("right_sideline", "far_baseline"),
            })

    # near-left: left sideline extrapolated to near edge
    for ls in left_side:
        # Y intercept near bottom (y = 0.85*h)
        y_near = 0.85 * h
        sin_t = np.sin(ls.theta)
        cos_t = np.cos(ls.theta)
        if abs(sin_t) > 1e-3:
            x_near = (ls.r - y_near * sin_t) / cos_t if abs(cos_t) > 1e-3 else ls.r / cos_t
            gt_px = corners_px[0]
            dist = np.hypot(x_near - gt_px[0], y_near - gt_px[1])
            results.append({
                "corner": "near-left (extrapolated)",
                "intersection_px": (float(x_near), float(y_near)),
                "gt_px": gt_px,
                "dist_px": float(dist),
                "lines": ("left_sideline", "near_y_extrapolation"),
            })

    # near-right: right sideline extrapolated to near edge
    for rs in right_side:
        y_near = 0.85 * h
        sin_t = np.sin(rs.theta)
        cos_t = np.cos(rs.theta)
        if abs(sin_t) > 1e-3:
            x_near = (rs.r - y_near * sin_t) / cos_t if abs(cos_t) > 1e-3 else rs.r / cos_t
            gt_px = corners_px[1]
            dist = np.hypot(x_near - gt_px[0], y_near - gt_px[1])
            results.append({
                "corner": "near-right (extrapolated)",
                "intersection_px": (float(x_near), float(y_near)),
                "gt_px": gt_px,
                "dist_px": float(dist),
                "lines": ("right_sideline", "near_y_extrapolation"),
            })

    # Sort by distance and deduplicate by corner (keep best)
    results.sort(key=lambda x: x["dist_px"])
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in results:
        if r["corner"] not in seen:
            seen.add(r["corner"])
            deduped.append(r)

    return deduped


def draw_visualization(
    frame: np.ndarray,
    findings: VideoFindings,
    save_path: Path,
) -> None:
    """Draw annotated frame and save."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    type_colors = {
        "far_baseline": COLOR_FAR_BASELINE,
        "left_sideline": COLOR_LEFT_SIDE,
        "right_sideline": COLOR_RIGHT_SIDE,
        "center_line": COLOR_CENTER_LINE,
        "unknown": COLOR_UNKNOWN,
    }

    # Draw lines
    all_lines: list[LineInfo] = []
    for lines in findings.lines_by_type.values():
        all_lines.extend(lines)

    for line in all_lines:
        color = type_colors.get(line.line_type, COLOR_UNKNOWN)
        cv2.line(vis, (line.x1, line.y1), (line.x2, line.y2), color, 2, cv2.LINE_AA)

    # Draw GT corners (green filled circles + label)
    if findings.gt_corners:
        for i, (name, corner) in enumerate(zip(CORNER_NAMES, findings.gt_corners)):
            px = int(corner["x"] * w)
            py = int(corner["y"] * h)
            cv2.circle(vis, (px, py), 14, COLOR_GT, -1)
            cv2.circle(vis, (px, py), 14, (255, 255, 255), 2)
            label = name.replace("-", "\n").split("\n")
            tag = name[:2]  # NL, NR, FR, FL
            cv2.putText(vis, tag, (px + 16, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT, 2)

    # Draw YOLO predicted corners (orange circles)
    if findings.yolo_corners:
        for i, (name, corner) in enumerate(zip(CORNER_NAMES, findings.yolo_corners)):
            px = int(corner["x"] * w)
            py = int(corner["y"] * h)
            cv2.circle(vis, (px, py), 10, COLOR_YOLO, -1)
            cv2.circle(vis, (px, py), 10, (255, 255, 255), 2)

    # Legend
    legend_items = [
        ("GT corners (green)", COLOR_GT),
        ("YOLO corners (orange)", COLOR_YOLO),
        ("Far baseline", COLOR_FAR_BASELINE),
        ("Left sideline", COLOR_LEFT_SIDE),
        ("Right sideline", COLOR_RIGHT_SIDE),
        ("Center line", COLOR_CENTER_LINE),
        ("Unknown", COLOR_UNKNOWN),
    ]
    y_off = 30
    for label, color in legend_items:
        cv2.rectangle(vis, (10, y_off - 12), (28, y_off + 4), color, -1)
        cv2.putText(vis, label, (35, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_off += 22

    # Video name overlay
    cv2.putText(vis, findings.name, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2)

    # Line count summary
    count_strs = []
    for ltype, lines in sorted(findings.lines_by_type.items()):
        if ltype != "unknown" and lines:
            count_strs.append(f"{ltype.split('_')[0]}:{len(lines)}")
    cv2.putText(vis, "  ".join(count_strs), (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    cv2.imwrite(str(save_path), vis)
    print(f"  Saved: {save_path}")


def print_findings_table(all_findings: list[VideoFindings]) -> None:
    """Print a formatted table of findings."""
    print("\n" + "=" * 100)
    print("COURT LINE DETECTION FINDINGS")
    print("=" * 100)

    for f in all_findings:
        print(f"\n{'─'*80}")
        print(f"VIDEO: {f.name}  (id={f.video_id})")
        print(f"{'─'*80}")

        if f.error:
            print(f"  ERROR: {f.error}")
            continue

        h, w = f.frame_shape[:2]

        # GT status
        gt_status = "YES" if f.has_gt else "NO"
        print(f"  GT calibration: {gt_status}")
        if f.gt_corners:
            for name, c in zip(CORNER_NAMES, f.gt_corners):
                py_pct = c["y"] * 100
                px_pct = c["x"] * 100
                print(f"    {name:12s}: x={px_pct:.1f}%  y={py_pct:.1f}%")

        # YOLO predictions
        yolo_status = "YES" if f.yolo_corners else "NO"
        print(f"\n  YOLO keypoint prediction: {yolo_status}  (conf={f.yolo_confidence:.3f})")
        if f.yolo_corners:
            for name, c in zip(CORNER_NAMES, f.yolo_corners):
                py_pct = c["y"] * 100
                px_pct = c["x"] * 100
                # Compare to GT if available
                if f.gt_corners:
                    gt_c = next((g for gn, g in zip(CORNER_NAMES, f.gt_corners) if gn == name), None)
                    if gt_c:
                        dist = ((c["x"] - gt_c["x"])**2 + (c["y"] - gt_c["y"])**2) ** 0.5
                        dist_px = dist * ((w**2 + h**2) ** 0.5)
                        print(f"    {name:12s}: x={px_pct:.1f}%  y={py_pct:.1f}%  "
                              f"(GT error: {dist:.4f} = {dist_px:.0f}px)")
                    else:
                        print(f"    {name:12s}: x={px_pct:.1f}%  y={py_pct:.1f}%")
                else:
                    print(f"    {name:12s}: x={px_pct:.1f}%  y={py_pct:.1f}%")

        # Hough line counts by type
        print(f"\n  Hough line detection (after dedup):")
        total_lines = sum(len(v) for v in f.lines_by_type.values())
        print(f"    Total unique lines: {total_lines}")
        for ltype in ["far_baseline", "left_sideline", "right_sideline", "center_line", "unknown"]:
            lines = f.lines_by_type.get(ltype, [])
            if lines:
                angles = [l.angle_deg for l in lines]
                angle_str = f"  angles: {', '.join(f'{a:.1f}°' for a in angles[:5])}"
                print(f"    {ltype:20s}: {len(lines):2d} lines{angle_str}")
            else:
                print(f"    {ltype:20s}:  0 lines")

        # Line-corner intersections
        if f.gt_corners and f.lines_by_type:
            all_lines: list[LineInfo] = []
            for lines in f.lines_by_type.values():
                all_lines.extend(lines)
            intersections = compute_line_corner_intersections(
                all_lines, f.gt_corners, h, w,
            )
            if intersections:
                print(f"\n  Best line intersections vs GT corners:")
                for inter in intersections:
                    ix, iy = inter["intersection_px"]
                    gx, gy = inter["gt_px"]
                    dist = inter["dist_px"]
                    lines_used = "+".join(inter["lines"])
                    print(f"    {inter['corner']:30s}: intersection=({ix:.0f},{iy:.0f})  "
                          f"GT=({gx:.0f},{gy:.0f})  dist={dist:.0f}px  [{lines_used}]")
            else:
                print(f"\n  No line intersections computable (missing line types)")

        # Conclusion
        has_baseline = bool(f.lines_by_type.get("far_baseline"))
        has_left = bool(f.lines_by_type.get("left_sideline"))
        has_right = bool(f.lines_by_type.get("right_sideline"))
        print(f"\n  Summary:")
        print(f"    Far baseline detectable: {'YES' if has_baseline else 'NO'}")
        print(f"    Left sideline detectable: {'YES' if has_left else 'NO'}")
        print(f"    Right sideline detectable: {'YES' if has_right else 'NO'}")


def process_video(name: str, video_id: str, kp_detector: CourtKeypointDetector) -> tuple[VideoFindings, np.ndarray | None]:
    """Process a single video. Returns (findings, frame)."""
    print(f"\n[{name}] Processing video_id={video_id}...")

    findings = VideoFindings(name=name, video_id=video_id, has_gt=False,
                              gt_corners=None, yolo_corners=None, yolo_confidence=0.0)

    # Load GT calibration
    gt = load_gt_calibration_by_id(video_id)
    findings.gt_corners = gt
    findings.has_gt = gt is not None
    print(f"  GT calibration: {'found' if gt else 'none'}")

    # Resolve video path
    video_path = get_video_path(video_id)
    if video_path is None:
        findings.error = "Could not resolve video path"
        return findings, None

    print(f"  Video path: {video_path}")

    # Sample middle frame
    frame = sample_middle_frame(video_path)
    if frame is None:
        findings.error = "Could not sample frame from video"
        return findings, None

    h, w = frame.shape[:2]
    findings.frame_shape = frame.shape

    print(f"  Frame: {w}x{h}")

    # Run YOLO keypoint detection on this frame
    try:
        kp_result = kp_detector.detect_from_frame(frame)
        if kp_result.corners:
            findings.yolo_corners = kp_result.corners
            findings.yolo_confidence = kp_result.confidence
            print(f"  YOLO: {len(kp_result.corners)} corners detected, conf={kp_result.confidence:.3f}")
        else:
            print(f"  YOLO: no detection")
    except Exception as e:
        print(f"  YOLO error: {e}")

    # Run Hough line detection
    raw_lines = run_hough_lines(frame)
    print(f"  Hough: {len(raw_lines)} raw lines detected")

    # Classify lines
    classified = classify_all_lines(raw_lines, gt, h, w, max_lines=80)

    # Deduplicate
    deduped = deduplicate_lines(classified, frame_diag=float(np.hypot(w, h)))
    print(f"  After dedup: {len(deduped)} unique lines")

    # Group by type
    by_type: dict[str, list[LineInfo]] = {}
    for line in deduped:
        by_type.setdefault(line.line_type, []).append(line)
    findings.lines_by_type = by_type

    counts = {t: len(v) for t, v in by_type.items()}
    print(f"  Line types: {counts}")

    return findings, frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COURT LINE DIAGNOSIS: veve, papa, pipi, pepe")
    print("=" * 80)

    # Query video IDs
    print("\nQuerying video IDs from DB...")
    name_to_id = query_videos_by_name(TARGET_NAMES)

    if not name_to_id:
        print("ERROR: No videos found. Check DB connection.")
        sys.exit(1)

    # Load YOLO keypoint detector
    print("\nLoading YOLO keypoint detector...")
    kp_detector = CourtKeypointDetector()
    if not kp_detector.model_exists:
        print("WARNING: Keypoint model not found — YOLO predictions will be skipped.")

    # Process each video
    all_findings: list[VideoFindings] = []
    frames: dict[str, np.ndarray] = {}

    for name in TARGET_NAMES:
        if name not in name_to_id:
            f = VideoFindings(name=name, video_id="N/A", has_gt=False,
                               gt_corners=None, yolo_corners=None, yolo_confidence=0.0)
            f.error = "Video not found in DB"
            all_findings.append(f)
            continue

        video_id = name_to_id[name]
        findings, frame = process_video(name, video_id, kp_detector)
        all_findings.append(findings)
        if frame is not None:
            frames[name] = frame

    # Print table
    print_findings_table(all_findings)

    # Save visualizations
    print("\n" + "=" * 80)
    print("SAVING VISUALIZATIONS")
    print("=" * 80)
    for findings in all_findings:
        if findings.error or findings.name not in frames:
            print(f"  Skipping {findings.name}: {findings.error or 'no frame'}")
            continue
        frame = frames[findings.name]
        save_path = OUTPUT_DIR / f"{findings.name}_court_lines.jpg"
        draw_visualization(frame, findings, save_path)

    print(f"\nDone. Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
