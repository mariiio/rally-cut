#!/usr/bin/env python3
"""Evaluate automatic court detection against manually calibrated ground truth.

Loads all videos with court_calibration_json from the DB, runs CourtDetector
on each, and computes per-video and aggregate metrics.

Metrics:
- MCD: Mean Corner Distance (normalized + pixels)
- Per-corner MCD: near-left, near-right, far-right, far-left
- Court Polygon IoU: rasterized overlap
- Reprojection Error: court center projected through both homographies
- Detection Success Rate: % with MCD < 5% frame diagonal
- Player Projection Accuracy: % of player positions mapping to correct team half

Usage:
    uv run python scripts/eval_court_detection.py
    uv run python scripts/eval_court_detection.py --debug         # Save debug images
    uv run python scripts/eval_court_detection.py --with-players  # Player-constrained refinement
    uv run python scripts/eval_court_detection.py --compare       # A/B comparison (line vs player)
    uv run python scripts/eval_court_detection.py -o results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]


def point_in_polygon(
    px: float, py: float, polygon: list[tuple[float, float]],
) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def rasterized_iou(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
    grid_size: int = 200,
) -> float:
    """Compute IoU between two polygons by rasterizing on a grid.

    Supports coordinates outside [0,1] by expanding the grid bounds.
    """
    # Find bounding box of both polygons
    all_pts = poly_a + poly_b
    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)

    # Add margin
    margin = 0.05
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    range_x = max_x - min_x
    range_y = max_y - min_y

    if range_x <= 0 or range_y <= 0:
        return 0.0

    intersection = 0
    union = 0
    step_x = range_x / grid_size
    step_y = range_y / grid_size

    for gy in range(grid_size):
        py = min_y + (gy + 0.5) * step_y
        for gx in range(grid_size):
            px = min_x + (gx + 0.5) * step_x
            in_a = point_in_polygon(px, py, poly_a)
            in_b = point_in_polygon(px, py, poly_b)
            if in_a or in_b:
                union += 1
            if in_a and in_b:
                intersection += 1

    return intersection / union if union > 0 else 0.0


def mean_corner_distance(
    detected: list[dict[str, float]],
    gt: list[dict[str, float]],
) -> float:
    """Mean Euclidean distance between corresponding corners (normalized coords)."""
    if len(detected) != 4 or len(gt) != 4:
        return float("inf")

    total = 0.0
    for d, g in zip(detected, gt):
        dx = d["x"] - g["x"]
        dy = d["y"] - g["y"]
        total += math.sqrt(dx * dx + dy * dy)

    return total / 4.0


def per_corner_distances(
    detected: list[dict[str, float]],
    gt: list[dict[str, float]],
) -> list[float]:
    """Per-corner Euclidean distances (normalized coords).

    Returns [near-left, near-right, far-right, far-left] distances.
    """
    if len(detected) != 4 or len(gt) != 4:
        return [float("inf")] * 4

    dists = []
    for d, g in zip(detected, gt):
        dx = d["x"] - g["x"]
        dy = d["y"] - g["y"]
        dists.append(math.sqrt(dx * dx + dy * dy))

    return dists


def reprojection_error_meters(
    detected: list[dict[str, float]],
    gt: list[dict[str, float]],
) -> float | None:
    """Reprojection error at court center through both homographies, in meters."""
    try:
        from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH, CourtCalibrator

        # Build calibrator from GT
        gt_cal = CourtCalibrator()
        gt_corners = [(c["x"], c["y"]) for c in gt]
        gt_cal.calibrate(gt_corners)

        # Build calibrator from detected
        det_cal = CourtCalibrator()
        det_corners = [(c["x"], c["y"]) for c in detected]
        det_cal.calibrate(det_corners)

        # Project court center (4, 8) through both
        court_center = (COURT_WIDTH / 2.0, COURT_LENGTH / 2.0)

        gt_img = gt_cal.court_to_image(court_center, 1920, 1080)
        det_court = det_cal.image_to_court(gt_img, 1920, 1080)

        dx = det_court[0] - court_center[0]
        dy = det_court[1] - court_center[1]
        return math.sqrt(dx * dx + dy * dy)

    except (np.linalg.LinAlgError, ValueError, AttributeError):
        logging.debug("Reprojection error computation failed", exc_info=True)
        return None


def player_projection_accuracy(
    corners: list[dict[str, float]],
    player_positions: list[Any],
    team_assignments: dict[int, int],
) -> float | None:
    """Compute % of player positions that project to the correct team half.

    Args:
        corners: 4 court corners in normalized image coords.
        player_positions: PlayerPosition objects.
        team_assignments: track_id → team (0=near, 1=far).

    Returns:
        Accuracy 0-1, or None if insufficient data.
    """
    try:
        from rallycut.court.calibration import COURT_LENGTH, CourtCalibrator

        cal = CourtCalibrator()
        cal.calibrate([(c["x"], c["y"]) for c in corners])

        net_y = COURT_LENGTH / 2.0
        correct = 0
        total = 0

        for pos in player_positions:
            track_id = getattr(pos, "track_id", -1)
            if track_id not in team_assignments:
                continue

            x = getattr(pos, "x", 0.0)
            y = getattr(pos, "y", 0.0)
            height = getattr(pos, "height", 0.0)
            foot_y = y + height / 2.0

            try:
                # Corners and positions are both in normalized (0-1) coords
                court_pt = cal.image_to_court((x, foot_y), 1, 1)
            except (RuntimeError, np.linalg.LinAlgError):
                continue

            team = team_assignments[track_id]
            # Near team (0) should be y < 8, far team (1) should be y >= 8
            if team == 0 and court_pt[1] < net_y:
                correct += 1
            elif team == 1 and court_pt[1] >= net_y:
                correct += 1
            total += 1

        return correct / total if total > 20 else None

    except Exception:
        return None


def load_player_data(
    video_id: str,
) -> tuple[list[Any], dict[int, int]] | None:
    """Load player positions and team assignments from DB for a video.

    Uses court_split_y to classify teams by image Y position, avoiding
    dependency on the often-inaccurate teamAssignments from actions_json.

    Track IDs are rally-local, so we offset them per rally to avoid collisions.

    Returns (player_positions, team_assignments) or None if not available.
    """
    try:
        from collections import defaultdict

        from rallycut.evaluation.db import get_connection
        from rallycut.tracking.player_tracker import PlayerPosition

        query = """
            SELECT pt.positions_json, pt.court_split_y, pt.primary_track_ids
            FROM player_tracks pt
            JOIN rallies r ON r.id = pt.rally_id
            WHERE r.video_id = %s
              AND pt.positions_json IS NOT NULL
            LIMIT 10
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [video_id])
                rows = cur.fetchall()

        if not rows:
            return None

        all_positions: list[Any] = []
        combined_teams: dict[int, int] = {}
        track_id_offset = 0

        for positions_json, court_split_y, primary_track_ids in rows:
            if not positions_json:
                continue

            # Parse positions (bare list of dicts)
            if isinstance(positions_json, str):
                positions_list = json.loads(positions_json)
            else:
                positions_list = positions_json

            if not isinstance(positions_list, list) or not positions_list:
                continue

            # Determine team split Y
            split_y_raw = court_split_y
            split_y: float | None = (
                float(split_y_raw)  # type: ignore[arg-type]
                if split_y_raw is not None
                else None
            )
            if split_y is None:
                # Fall back to median Y of all positions
                all_y = [p.get("y", 0.0) for p in positions_list if isinstance(p, dict)]
                if all_y:
                    split_y = sorted(all_y)[len(all_y) // 2]
                else:
                    continue

            # Filter to primary tracks if available
            primary_set: set[int] | None = None
            if primary_track_ids and isinstance(primary_track_ids, list):
                primary_set = set(int(tid) for tid in primary_track_ids)

            # Compute average Y per track to classify teams
            track_ys: dict[int, list[float]] = defaultdict(list)
            for p in positions_list:
                if not isinstance(p, dict):
                    continue
                tid = p.get("trackId", -1)
                if primary_set and tid not in primary_set:
                    continue
                track_ys[tid].append(p.get("y", 0.0))

            # Classify: avg Y > split_y → near (team 0), else far (team 1)
            rally_teams: dict[int, int] = {}
            for tid, ys in track_ys.items():
                if len(ys) < 5:
                    continue
                avg_y = sum(ys) / len(ys)
                offset_tid = tid + track_id_offset
                rally_teams[offset_tid] = 0 if avg_y > split_y else 1

            if not rally_teams:
                continue

            # Create positions with offset track IDs
            for p in positions_list:
                if not isinstance(p, dict):
                    continue
                orig_tid = p.get("trackId", -1)
                offset_tid = orig_tid + track_id_offset
                if offset_tid not in rally_teams:
                    continue
                all_positions.append(PlayerPosition(
                    frame_number=p.get("frameNumber", 0),
                    track_id=offset_tid,
                    x=p.get("x", 0.0),
                    y=p.get("y", 0.0),
                    width=p.get("width", 0.0),
                    height=p.get("height", 0.0),
                    confidence=p.get("confidence", 0.0),
                ))

            combined_teams.update(rally_teams)
            track_id_offset += 10000

        if len(all_positions) < 20 or not combined_teams:
            return None

        return all_positions, combined_teams

    except Exception:
        return None


def _run_player_constrained(
    line_result: Any,
    player_data: tuple[list[Any], dict[int, int]] | None,
) -> list[dict[str, float]] | None:
    """Run player-constrained refinement on a line detection result.

    Returns refined corners or None.
    """
    if player_data is None:
        return None

    player_positions, team_assignments = player_data

    from rallycut.court.player_constrained import (
        PlayerConstrainedOptimizer,
        extract_player_feet,
    )

    feet = extract_player_feet(player_positions, team_assignments)
    if len(feet) < 20:
        return None

    optimizer = PlayerConstrainedOptimizer()
    initial_corners = getattr(line_result, "corners", [])

    if len(initial_corners) == 4:
        # Refine near corners
        return optimizer.refine_corners(initial_corners, feet, fix_far_corners=True)
    else:
        # Estimate from players only
        near_feet_y = [f.y for f in feet if f.team == 0]
        far_feet_y = [f.y for f in feet if f.team == 1]
        if near_feet_y and far_feet_y:
            net_y = (min(near_feet_y) + max(far_feet_y)) / 2.0
            return optimizer.estimate_from_players(feet, net_y)

    return None


def evaluate_corners(
    detected: list[dict[str, float]],
    gt_corners: list[dict[str, float]],
    vid_width: int,
    vid_height: int,
) -> dict[str, Any]:
    """Compute all metrics for a set of detected corners vs GT."""
    mcd = mean_corner_distance(detected, gt_corners)
    mcd_px = mcd * math.sqrt(vid_width ** 2 + vid_height ** 2)
    is_success = mcd < 0.05

    gt_poly = [(c["x"], c["y"]) for c in gt_corners]
    det_poly = [(c["x"], c["y"]) for c in detected]
    iou = rasterized_iou(det_poly, gt_poly)

    reproj = reprojection_error_meters(detected, gt_corners)
    corner_dists = per_corner_distances(detected, gt_corners)

    return {
        "mcd_norm": mcd,
        "mcd_px": mcd_px,
        "iou": iou,
        "reprojection_error_m": reproj,
        "success": is_success,
        "per_corner_dist": corner_dists,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate court detection")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("-o", "--output", type=str, help="Output JSON path")
    parser.add_argument("--video-id", type=str, help="Evaluate single video")
    parser.add_argument(
        "--config", type=str,
        help='Config overrides as JSON, e.g. \'{"dbscan_eps": 0.05}\'',
    )
    parser.add_argument(
        "--with-players", action="store_true",
        help="Refine court detection using player tracking data from DB",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Show A/B comparison: line-only vs player-constrained",
    )
    parser.add_argument(
        "--keypoint", action="store_true",
        help="Use YOLO-pose keypoint model for court detection",
    )
    parser.add_argument(
        "--compare-keypoint", action="store_true",
        help="Show A/B comparison: classical vs keypoint model",
    )
    parser.add_argument(
        "--keypoint-model", type=str, default=None,
        help="Path to keypoint model weights (default: weights/court_keypoint/court_keypoint_best.pt)",
    )
    args = parser.parse_args()

    from rallycut.court.detector import CourtDetectionConfig, CourtDetector
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    # Load all videos with court calibration
    query = """
        SELECT id, court_calibration_json, width, height
        FROM videos
        WHERE court_calibration_json IS NOT NULL
    """
    params: list[str] = []
    if args.video_id:
        query += " AND id = %s"
        params.append(args.video_id)

    videos: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                vid_id, cal_json, width, height = row
                if isinstance(cal_json, list) and len(cal_json) == 4:
                    videos.append({
                        "video_id": str(vid_id),
                        "gt_corners": cal_json,
                        "width": width or 1920,
                        "height": height or 1080,
                    })

    print(f"Found {len(videos)} video(s) with court calibration GT\n")

    if not videos:
        print("No calibrated videos found.")
        return

    # Debug output directory
    debug_dir = Path("debug_court_detection")
    if args.debug:
        debug_dir.mkdir(exist_ok=True)

    config_overrides: dict[str, Any] = {}
    if args.config:
        config_overrides = json.loads(args.config)
        print(f"Config overrides: {config_overrides}\n")

    # Keypoint detector setup
    kp_detector = None
    use_keypoint = args.keypoint or args.compare_keypoint
    if use_keypoint:
        from rallycut.court.keypoint_detector import CourtKeypointDetector

        kp_detector = CourtKeypointDetector(model_path=args.keypoint_model)
        if not kp_detector.model_exists:
            print("Keypoint model not found. Train it first:")
            print("  uv run python scripts/train_court_keypoint_model.py")
            if args.keypoint and not args.compare_keypoint:
                return
            print("Continuing with classical detection only.\n")
            kp_detector = None

    if args.keypoint and kp_detector is not None:
        # Use keypoint model as primary (no classical fallback in eval)
        detector = CourtDetector(CourtDetectionConfig(**config_overrides))
    else:
        # Force classical-only: pass nonexistent path to prevent keypoint auto-detection
        detector = CourtDetector(
            CourtDetectionConfig(**config_overrides),
            keypoint_model_path="/nonexistent/disabled.pt",
        )

    use_players = args.with_players or args.compare

    results: list[dict[str, Any]] = []
    success_count = 0
    total_mcd = 0.0
    total_iou = 0.0
    evaluated = 0

    # Per-corner accumulation
    corner_mcd_sums = [0.0, 0.0, 0.0, 0.0]
    corner_count = 0

    # Player data cache (avoid double DB queries)
    player_data_cache: dict[str, tuple[list[Any], dict[int, int]]] = {}

    # Player-constrained tracking
    pc_success_count = 0
    pc_total_mcd = 0.0
    pc_total_iou = 0.0
    pc_evaluated = 0

    # Keypoint tracking
    kp_success_count = 0
    kp_total_mcd = 0.0
    kp_total_iou = 0.0
    kp_evaluated = 0
    kp_corner_mcd_sums = [0.0, 0.0, 0.0, 0.0]

    # Header
    header = (
        f"{'Video':>12s}  {'MCD':>7s}  {'MCD px':>7s}  {'IoU':>5s}  {'Reproj':>7s}  "
        f"{'Conf':>5s}  {'Lines':>5s}  {'Method':>20s}  {'Corr':>4s}  {'RpErr':>6s}"
    )
    if args.compare:
        header += f"  {'PC MCD':>7s}  {'PC IoU':>5s}  {'PC Meth':>15s}"
    if args.compare_keypoint:
        header += f"  {'KP MCD':>7s}  {'KP IoU':>5s}  {'KP Conf':>7s}"
    print(header)
    print("-" * (160 if args.compare_keypoint else (140 if args.compare else 120)))

    total_videos = len(videos)
    for video_idx, video_info in enumerate(videos):
        t_start = time.monotonic()
        vid_id = video_info["video_id"]
        gt_corners = video_info["gt_corners"]
        vid_width = video_info["width"]
        vid_height = video_info["height"]

        counter_prefix = f"[{video_idx + 1}/{total_videos}]"

        # Resolve video path
        video_path = get_video_path(vid_id)
        if video_path is None:
            print(f"{counter_prefix} {vid_id[:12]:>12s}  {'SKIP':>7s}  (video not found)")
            continue

        # Run detection (keypoint-only or classical)
        try:
            if args.keypoint and kp_detector is not None:
                result = kp_detector.detect(video_path)
            else:
                result = detector.detect(video_path)
        except Exception as e:
            print(f"{counter_prefix} {vid_id[:12]:>12s}  {'ERROR':>7s}  {e}")
            continue

        # Keypoint comparison (run keypoint alongside classical)
        kp_corners = None
        kp_conf = 0.0
        if args.compare_keypoint and kp_detector is not None:
            try:
                kp_result = kp_detector.detect(video_path)
                if len(kp_result.corners) == 4:
                    kp_corners = kp_result.corners
                    kp_conf = kp_result.confidence
            except Exception as e:
                logging.debug(f"Keypoint detection failed for {vid_id[:12]}: {e}")

        # Player-constrained refinement
        pc_corners = None
        pc_method = ""
        cached_player_data = None
        if use_players:
            cached_player_data = load_player_data(vid_id)
            if cached_player_data is not None:
                player_data_cache[vid_id] = cached_player_data
            pc_corners = _run_player_constrained(result, cached_player_data)
            if pc_corners is not None:
                if len(result.corners) == 4:
                    pc_method = result.fitting_method + "+player"
                else:
                    pc_method = "player_only"

        # Use player-constrained if --with-players (not just --compare)
        detected = result.corners
        fitting_method = getattr(result, "fitting_method", "legacy")
        if args.with_players and pc_corners is not None:
            detected = pc_corners
            fitting_method = pc_method

        lines_found = len(result.detected_lines)
        n_correspondences = getattr(result, "n_correspondences", 0)
        reproj_err = getattr(result, "reprojection_error", 0.0)

        result_data: dict[str, Any] = {
            "video_id": vid_id,
            "confidence": result.confidence,
            "lines_found": lines_found,
            "fitting_method": fitting_method,
            "n_correspondences": n_correspondences,
            "reproj_error": reproj_err,
            "warnings": result.warnings,
        }

        if detected and len(detected) == 4:
            metrics = evaluate_corners(detected, gt_corners, vid_width, vid_height)
            mcd = metrics["mcd_norm"]
            mcd_px = metrics["mcd_px"]
            iou = metrics["iou"]
            is_success = metrics["success"]
            reproj = metrics["reprojection_error_m"]
            corner_dists = metrics["per_corner_dist"]

            reproj_str = f"{reproj:.2f}m" if reproj is not None else "N/A"

            elapsed = time.monotonic() - t_start
            line = (
                f"{counter_prefix} {vid_id[:12]:>12s}  "
                f"{mcd:.4f}  "
                f"{mcd_px:6.1f}  "
                f"{iou:.3f}  "
                f"{reproj_str:>7s}  "
                f"{result.confidence:.3f}  "
                f"{lines_found:>5d}  "
                f"{fitting_method:>20s}  "
                f"{n_correspondences:>4d}  "
                f"{reproj_err:>6.4f}  "
                f"({elapsed:.1f}s)"
            )

            if is_success:
                success_count += 1
            total_mcd += mcd
            total_iou += iou
            evaluated += 1

            for i, cd in enumerate(corner_dists):
                corner_mcd_sums[i] += cd
            corner_count += 1

            result_data.update({
                "mcd_norm": mcd,
                "mcd_px": mcd_px,
                "iou": iou,
                "reprojection_error_m": reproj,
                "success": is_success,
                "per_corner_dist": {
                    CORNER_NAMES[i]: round(corner_dists[i], 5)
                    for i in range(4)
                },
                "detected_corners": detected,
                "gt_corners": gt_corners,
            })

            # Player-constrained comparison
            if args.compare and pc_corners is not None:
                pc_metrics = evaluate_corners(
                    pc_corners, gt_corners, vid_width, vid_height,
                )
                line += (
                    f"  {pc_metrics['mcd_norm']:.4f}  "
                    f"{pc_metrics['iou']:.3f}  "
                    f"{pc_method:>15s}"
                )
                if pc_metrics["success"]:
                    pc_success_count += 1
                pc_total_mcd += pc_metrics["mcd_norm"]
                pc_total_iou += pc_metrics["iou"]
                pc_evaluated += 1
                result_data["player_constrained"] = {
                    "corners": pc_corners,
                    "mcd_norm": pc_metrics["mcd_norm"],
                    "iou": pc_metrics["iou"],
                    "method": pc_method,
                }
            elif args.compare:
                line += f"  {'N/A':>7s}  {'N/A':>5s}  {'no data':>15s}"

            # Keypoint comparison
            if args.compare_keypoint and kp_corners is not None:
                kp_metrics = evaluate_corners(
                    kp_corners, gt_corners, vid_width, vid_height,
                )
                line += (
                    f"  {kp_metrics['mcd_norm']:.4f}  "
                    f"{kp_metrics['iou']:.3f}  "
                    f"{kp_conf:>7.3f}"
                )
                if kp_metrics["success"]:
                    kp_success_count += 1
                kp_total_mcd += kp_metrics["mcd_norm"]
                kp_total_iou += kp_metrics["iou"]
                kp_evaluated += 1
                for i, cd in enumerate(kp_metrics["per_corner_dist"]):
                    kp_corner_mcd_sums[i] += cd
                result_data["keypoint"] = {
                    "corners": kp_corners,
                    "mcd_norm": kp_metrics["mcd_norm"],
                    "iou": kp_metrics["iou"],
                    "confidence": kp_conf,
                }
            elif args.compare_keypoint:
                line += f"  {'N/A':>7s}  {'N/A':>5s}  {'N/A':>7s}"

            print(line)

        else:
            elapsed = time.monotonic() - t_start
            line = (
                f"{counter_prefix} {vid_id[:12]:>12s}  {'FAIL':>7s}  "
                f"conf={result.confidence:.3f}  "
                f"lines={lines_found}  "
                f"method={fitting_method}  "
                f"{'; '.join(result.warnings[:2])}  "
                f"({elapsed:.1f}s)"
            )

            # Player-only can still produce results
            if args.compare and pc_corners is not None:
                pc_metrics = evaluate_corners(
                    pc_corners, gt_corners, vid_width, vid_height,
                )
                line += (
                    f"  PC: MCD={pc_metrics['mcd_norm']:.4f} "
                    f"IoU={pc_metrics['iou']:.3f}"
                )
                if pc_metrics["success"]:
                    pc_success_count += 1
                pc_total_mcd += pc_metrics["mcd_norm"]
                pc_total_iou += pc_metrics["iou"]
                pc_evaluated += 1
                result_data["player_constrained"] = {
                    "corners": pc_corners,
                    "mcd_norm": pc_metrics["mcd_norm"],
                    "iou": pc_metrics["iou"],
                    "method": pc_method,
                }

            # Keypoint can still produce results even if classical fails
            if args.compare_keypoint and kp_corners is not None:
                kp_metrics = evaluate_corners(
                    kp_corners, gt_corners, vid_width, vid_height,
                )
                line += (
                    f"  KP: MCD={kp_metrics['mcd_norm']:.4f} "
                    f"IoU={kp_metrics['iou']:.3f}"
                )
                if kp_metrics["success"]:
                    kp_success_count += 1
                kp_total_mcd += kp_metrics["mcd_norm"]
                kp_total_iou += kp_metrics["iou"]
                kp_evaluated += 1
                for i, cd in enumerate(kp_metrics["per_corner_dist"]):
                    kp_corner_mcd_sums[i] += cd
                result_data["keypoint"] = {
                    "corners": kp_corners,
                    "mcd_norm": kp_metrics["mcd_norm"],
                    "iou": kp_metrics["iou"],
                    "confidence": kp_conf,
                }

            print(line)
            result_data.update({
                "mcd_norm": None,
                "iou": None,
                "success": False,
            })

        results.append(result_data)

        # Debug images
        if args.debug and video_path:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_f // 2)
                ret, frame = cap.read()
                if ret and frame is not None:
                    debug_frame = detector.create_debug_image(frame, result)
                    out_path = debug_dir / f"{vid_id[:12]}_court_debug.jpg"
                    cv2.imwrite(str(out_path), debug_frame)
                cap.release()

    # Summary
    print("-" * (160 if args.compare_keypoint else (140 if args.compare else 120)))
    if evaluated > 0:
        avg_mcd = total_mcd / evaluated
        avg_iou = total_iou / evaluated
        rate = success_count / evaluated * 100
        method_label = "Keypoint" if args.keypoint else "Line Detection"
        print(f"\n{method_label} ({evaluated} videos):")
        print(f"  Mean MCD:      {avg_mcd:.4f} (norm)")
        print(f"  Mean IoU:      {avg_iou:.3f}")
        print(f"  Success Rate:  {success_count}/{evaluated} ({rate:.1f}%)")

        # Print quality diagnostics for keypoint model
        if args.keypoint and kp_detector is not None and kp_detector.last_diagnostics is not None:
            diag = kp_detector.last_diagnostics
            print(f"\n  Quality Diagnostics (last video):")
            print(f"    Detection rate: {diag.detection_rate:.0%}")
            print(f"    Perspective ratio: {diag.perspective_ratio:.2f}")
            if diag.off_screen_corners:
                print(f"    Off-screen corners: {', '.join(diag.off_screen_corners)}")
            if diag.warnings:
                for w in diag.warnings:
                    print(f"    Warning: {w}")

        # Per-corner breakdown
        if corner_count > 0:
            print("  Per-corner MCD:")
            for i, name in enumerate(CORNER_NAMES):
                avg = corner_mcd_sums[i] / corner_count
                print(f"    {name:>12s}: {avg:.4f}")

        # Method distribution
        from collections import Counter
        methods = Counter(r.get("fitting_method", "unknown") for r in results)
        print(f"  Fitting methods: {dict(methods)}")

    if args.compare and pc_evaluated > 0:
        pc_avg_mcd = pc_total_mcd / pc_evaluated
        pc_avg_iou = pc_total_iou / pc_evaluated
        pc_rate = pc_success_count / pc_evaluated * 100
        print(f"\nPlayer-Constrained ({pc_evaluated} videos with player data):")
        print(f"  Mean MCD:      {pc_avg_mcd:.4f} (norm)")
        print(f"  Mean IoU:      {pc_avg_iou:.3f}")
        print(f"  Success Rate:  {pc_success_count}/{pc_evaluated} ({pc_rate:.1f}%)")

        if evaluated > 0:
            print(f"\n  Delta MCD:  {pc_avg_mcd - avg_mcd:+.4f}")
            print(f"  Delta IoU:  {pc_avg_iou - avg_iou:+.3f}")

    if args.compare_keypoint and kp_evaluated > 0:
        kp_avg_mcd = kp_total_mcd / kp_evaluated
        kp_avg_iou = kp_total_iou / kp_evaluated
        kp_rate = kp_success_count / kp_evaluated * 100
        print(f"\nKeypoint Model ({kp_evaluated} videos):")
        print(f"  Mean MCD:      {kp_avg_mcd:.4f} (norm)")
        print(f"  Mean IoU:      {kp_avg_iou:.3f}")
        print(f"  Success Rate:  {kp_success_count}/{kp_evaluated} ({kp_rate:.1f}%)")

        # Per-corner breakdown
        print("  Per-corner MCD:")
        for i, name in enumerate(CORNER_NAMES):
            avg = kp_corner_mcd_sums[i] / kp_evaluated
            print(f"    {name:>12s}: {avg:.4f}")

        if evaluated > 0:
            print(f"\n  Delta MCD vs classical:  {kp_avg_mcd - avg_mcd:+.4f}")
            print(f"  Delta IoU vs classical:  {kp_avg_iou - avg_iou:+.3f}")

    elif evaluated == 0:
        print("\nNo videos evaluated.")

    # Player projection accuracy (uses cached data from main loop)
    if use_players and player_data_cache:
        print("\nPlayer Projection Accuracy:")
        for r in results:
            vid_id = r["video_id"]
            corners = r.get("detected_corners")
            if corners is None:
                pc_data = r.get("player_constrained", {})
                corners = pc_data.get("corners")
            if corners is None:
                continue

            pdata = player_data_cache.get(vid_id)
            if pdata is None:
                continue

            positions, teams = pdata
            acc = player_projection_accuracy(corners, positions, teams)
            if acc is not None:
                print(f"  {vid_id[:12]:>12s}: {acc:.1%}")

    # Save results
    if args.output:
        aggregate: dict[str, Any] = {
            "evaluated": evaluated,
            "success_count": success_count,
            "mean_mcd_norm": total_mcd / evaluated if evaluated else None,
            "mean_iou": total_iou / evaluated if evaluated else None,
            "success_rate": success_count / evaluated if evaluated else None,
        }
        if corner_count > 0:
            aggregate["per_corner_mcd"] = {
                CORNER_NAMES[i]: round(corner_mcd_sums[i] / corner_count, 5)
                for i in range(4)
            }
        if args.compare and pc_evaluated > 0:
            aggregate["player_constrained"] = {
                "evaluated": pc_evaluated,
                "success_count": pc_success_count,
                "mean_mcd_norm": pc_total_mcd / pc_evaluated,
                "mean_iou": pc_total_iou / pc_evaluated,
            }
        if args.compare_keypoint and kp_evaluated > 0:
            aggregate["keypoint"] = {
                "evaluated": kp_evaluated,
                "success_count": kp_success_count,
                "mean_mcd_norm": kp_total_mcd / kp_evaluated,
                "mean_iou": kp_total_iou / kp_evaluated,
                "per_corner_mcd": {
                    CORNER_NAMES[i]: round(kp_corner_mcd_sums[i] / kp_evaluated, 5)
                    for i in range(4)
                },
            }

        with open(args.output, "w") as f:
            json.dump({"per_video": results, "aggregate": aggregate}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
