"""Probe 1 — net-line detection reliability.

For docs/superpowers/plans/2026-04-22-game-semantics-scoping.md.

Picks 20 rallies stratified across the corpus (4 hardest: dark / low-res;
4 extreme camera angles; 12 regular brightness bins), renders the first
usable rally frame with net-base (cyan) + net-top (yellow) lines overlaid,
and a per-rally CSV of endpoint coordinates. Visual inspection by a human
counts pass/fail per rally.

Kill gate: < 16/20 rallies pass visual check (net-base on net's base ±5px
AND net-top within net's top rope ±10px) → workstream dies.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.court.camera_model import calibrate_camera, project_3d_to_image
from rallycut.court.keypoint_detector import CourtKeypointDetector
from rallycut.court.net_line_estimator import estimate_net_line_from_s3
from rallycut.evaluation.db import get_connection

COURT_WIDTH_M = 8.0
COURT_LENGTH_M = 16.0
NET_Y_M = 8.0
NET_HEIGHT_M = 2.43

DEFAULT_COURT_CORNERS = [
    (0.0, 0.0),
    (COURT_WIDTH_M, 0.0),
    (COURT_WIDTH_M, COURT_LENGTH_M),
    (0.0, COURT_LENGTH_M),
]


def _pick_stratified_videos(target: int = 20) -> list[dict]:
    """Pick videos stratified across brightness / resolution / angle.

    Aim for 4 hard-night (dark + night-looking), 4 low-res (< 1920),
    4 extreme angle (high perspective ratio), 8 regular.
    """
    q = """
        SELECT v.id, v.width, v.height, v.quality_report_json,
               v.court_calibration_json
        FROM videos v
        WHERE v.court_calibration_json IS NOT NULL
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(q)
        rows = cur.fetchall()

    def bmean(qr):
        if not isinstance(qr, dict):
            return None
        b = qr.get("brightness")
        if not isinstance(b, dict):
            return None
        return b.get("mean")

    videos = []
    for row in rows:
        vid, w, h, qr, cal = row
        videos.append({
            "video_id": str(vid),
            "width": w,
            "height": h,
            "brightness": bmean(qr),
            "calibration": [(c["x"], c["y"]) for c in cal] if isinstance(cal, list) else None,
        })

    # Perspective ratio proxy = (near width) / (far width) where near = bottom
    # two corners, far = top two corners, after normalization.
    for v in videos:
        cal = v["calibration"]
        if cal and len(cal) == 4:
            top_w = abs(cal[1][0] - cal[0][0])
            bot_w = abs(cal[2][0] - cal[3][0])
            v["persp"] = bot_w / top_w if top_w > 0 else 0.0
        else:
            v["persp"] = 0.0

    picked: list[dict] = []
    seen: set[str] = set()

    # 4 darkest (including night/dark category). Prefer those with non-null brightness.
    dark_sorted = sorted(
        [v for v in videos if v["brightness"] is not None],
        key=lambda v: v["brightness"],
    )
    for v in dark_sorted[:4]:
        if v["video_id"] not in seen:
            v["strat"] = "hard_dark"
            picked.append(v)
            seen.add(v["video_id"])

    # 4 low-res (no width reported OR width < 1920)
    low_res = [v for v in videos if v["video_id"] not in seen
               and (v["width"] is None or (v["width"] and v["width"] < 1920))]
    for v in low_res[:4]:
        v["strat"] = "low_res"
        picked.append(v)
        seen.add(v["video_id"])

    # 4 extreme camera angle (largest persp ratio)
    extreme = sorted(
        [v for v in videos if v["video_id"] not in seen],
        key=lambda v: v["persp"],
        reverse=True,
    )
    for v in extreme[:4]:
        v["strat"] = "extreme_angle"
        picked.append(v)
        seen.add(v["video_id"])

    # Remaining slots: regular — spread across brightness bins
    remaining = [v for v in videos if v["video_id"] not in seen
                 and v["brightness"] is not None]
    remaining.sort(key=lambda v: v["brightness"])
    n_need = target - len(picked)
    if n_need > 0 and remaining:
        step = max(1, len(remaining) // n_need)
        for i in range(n_need):
            idx = min(i * step, len(remaining) - 1)
            v = remaining[idx]
            if v["video_id"] not in seen:
                v["strat"] = "regular"
                picked.append(v)
                seen.add(v["video_id"])

    return picked[:target]


def _pick_rally_for_video(video_id: str) -> dict | None:
    """Pick the first rally in the video with calibration + ball data."""
    q = """
        SELECT r.id, r.start_ms, r.end_ms, pt.fps,
               v.width, v.height, v.s3_key, v.processed_s3_key, v.proxy_s3_key,
               v.court_calibration_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE v.id = %s AND pt.ball_positions_json IS NOT NULL
        ORDER BY r.start_ms
        LIMIT 1
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(q, (video_id,))
        row = cur.fetchone()
    if row is None:
        return None
    cal_json = row[9]
    cal = None
    if isinstance(cal_json, list) and len(cal_json) == 4:
        cal = [(float(c["x"]), float(c["y"])) for c in cal_json]
    return {
        "rally_id": str(row[0]),
        "video_id_full": video_id,
        "start_ms": int(row[1]),
        "end_ms": int(row[2]),
        "fps": float(row[3] or 30.0),
        "width": int(row[4] or 1920),
        "height": int(row[5] or 1080),
        "s3_key": row[6],
        "processed_s3_key": row[7],
        "proxy_s3_key": row[8],
        "calibration": cal,
    }


def _s3_presigned_url(s3_key: str) -> str:
    import boto3

    endpoint = os.environ.get("S3_ENDPOINT") or "http://localhost:9000"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    bucket = os.environ.get("S3_BUCKET_NAME", "rallycut-dev")
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600,
    )


def _extract_frame(s3_key: str, time_s: float, out: Path) -> bool:
    url = _s3_presigned_url(s3_key)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{time_s:.3f}",
        "-i", url,
        "-frames:v", "1",
        "-q:v", "2",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg error: {r.stderr.strip()}")
        return False
    return out.exists() and out.stat().st_size > 0


def _hough_refine_net_top(
    frame_img: np.ndarray,
    predicted_top_l: tuple[float, float],
    predicted_top_r: tuple[float, float],
    predicted_base_l: tuple[float, float],
    predicted_base_r: tuple[float, float],
    search_band_frac: float = 0.35,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Snap the predicted net-top line to the strongest near-horizontal edge.

    Strategy:
      1. Build a vertical search strip spanning
         `predicted_top_y ± search_band_frac · (base_y - top_y)` at each sideline.
      2. Run Canny + probabilistic Hough within the strip to find candidate line
         segments whose slope matches the predicted line (±3°) and whose length
         exceeds 40 % of the image width.
      3. If multiple candidates qualify, pick the one whose predicted-top Y
         deviation is smallest AND whose average intensity along the line is
         darkest (net rope is darker than sky/background in most shots).
      4. Return refined (top_l_norm, top_r_norm) OR None if no confident line.
    """
    h, w = frame_img.shape[:2]
    gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)

    # Predicted top line image-y at each sideline (normalized).
    pred_top_l_y = predicted_top_l[1] * h
    pred_top_r_y = predicted_top_r[1] * h
    pred_base_l_y = predicted_base_l[1] * h
    pred_base_r_y = predicted_base_r[1] * h

    # Search band around the predicted top — scale with the base-to-top distance.
    top_to_base = max(abs(pred_base_l_y - pred_top_l_y), abs(pred_base_r_y - pred_top_r_y))
    if top_to_base < 20.0:
        return None
    half = max(12.0, search_band_frac * top_to_base)
    y_min = max(0, int(min(pred_top_l_y, pred_top_r_y) - half))
    y_max = min(h, int(max(pred_top_l_y, pred_top_r_y) + half))
    if y_max - y_min < 20:
        return None

    strip = gray[y_min:y_max]
    edges = cv2.Canny(strip, 30, 90)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=int(0.25 * w),
        minLineLength=int(0.40 * w),
        maxLineGap=int(0.05 * w),
    )
    if lines is None:
        return None

    # Predicted slope in the strip frame (offset by y_min).
    pred_slope = (pred_top_r_y - pred_top_l_y) / max(1.0, (predicted_top_r[0] - predicted_top_l[0]) * w)
    pred_slope_deg = np.degrees(np.arctan(pred_slope))

    candidates: list[tuple[float, float, float]] = []  # (deviation, avg_intensity, y_at_mid_x)
    best = None
    best_score = float("inf")

    for seg in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in seg]
        # Convert strip y back to full-frame y.
        y1 += y_min
        y2 += y_min
        dx = x2 - x1
        if abs(dx) < 1e-3:
            continue
        slope = (y2 - y1) / dx
        slope_deg = np.degrees(np.arctan(slope))
        if abs(slope_deg - pred_slope_deg) > 3.0:
            continue
        # Extrapolate to sideline X (0 and w) of the full frame.
        b = y1 - slope * x1
        y_at_0 = b
        y_at_w = slope * w + b
        # Deviation from predicted at sidelines
        dev = 0.5 * (abs(y_at_0 - pred_top_l_y) + abs(y_at_w - pred_top_r_y))
        if dev > half:
            continue
        # Sample intensity along the line
        n_samples = 40
        xs = np.linspace(max(0, x1), min(w - 1, x2), n_samples).astype(int)
        ys = np.clip((slope * xs + b).astype(int), 0, h - 1)
        avg_int = float(gray[ys, xs].mean())
        score = dev + 0.3 * (avg_int / 255.0) * h  # prefer darker lines
        candidates.append((dev, avg_int, (y_at_0 + y_at_w) * 0.5))
        if score < best_score:
            best_score = score
            best = (y_at_0 / h, y_at_w / h)

    if best is None:
        return None
    refined_l = (predicted_top_l[0], float(best[0]))
    refined_r = (predicted_top_r[0], float(best[1]))
    return refined_l, refined_r


def _detect_keypoints(
    detector: CourtKeypointDetector,
    frame_img: np.ndarray,
) -> dict | None:
    """Run keypoint detector on a single frame and return corners + center points.

    Returns dict with 'corners' (4 pts in near-left/near-right/far-right/far-left
    order, normalised xy) + 'center_left' / 'center_right' (normalised xy on net-base
    at sidelines) + 'corner_confs' + 'center_confs'. None if detection fails.
    """
    fk = detector._detect_frame(frame_img)  # noqa: SLF001 — intentional direct call
    if fk is None or fk.center_points is None or len(fk.center_points) != 2:
        return None
    return {
        "corners": [(float(c["x"]), float(c["y"])) for c in fk.corners],
        "center_left": (
            float(fk.center_points[0]["x"]),
            float(fk.center_points[0]["y"]),
        ),
        "center_right": (
            float(fk.center_points[1]["x"]),
            float(fk.center_points[1]["y"]),
        ),
        "corner_confs": list(fk.kpt_confidences),
        "center_confs": list(fk.center_confidences) if fk.center_confidences else [0.0, 0.0],
    }


def _render_frame(
    rally: dict,
    video_strat: str,
    frame_img: np.ndarray,
    out_png: Path,
    mode: str,
    keypoint_detector: CourtKeypointDetector | None = None,
) -> dict:
    """Render net-base + net-top overlay.

    mode='homography': use DB corners + calibrate_camera (the original probe).
    mode='keypoints_net': use keypoint-detected corners for the camera AND
                          keypoint center-left/right directly as net-base.
    """
    w = int(rally["width"])
    h = int(rally["height"])

    frame = frame_img.copy()
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h))

    NET_BASE_COLOR = (255, 255, 0)   # cyan  # noqa: N806
    NET_TOP_COLOR = (0, 255, 255)    # yellow (BGR)  # noqa: N806
    CORNER_COLOR = (0, 255, 0)       # green  # noqa: N806
    CENTER_COLOR = (255, 0, 255)     # magenta  # noqa: N806

    if mode == "homography":
        cal = rally["calibration"]
        if cal is None:
            return {"ok": False, "reason": "no_calibration"}

        cam = calibrate_camera(cal, DEFAULT_COURT_CORNERS, w, h)
        if cam is None:
            print("  [skip] homography camera calibration failed")
            return {"ok": False, "reason": "homography_camera_failed"}

        base_l = project_3d_to_image(cam, np.array([0.0, NET_Y_M, 0.0]))
        base_r = project_3d_to_image(cam, np.array([COURT_WIDTH_M, NET_Y_M, 0.0]))
        top_l = project_3d_to_image(cam, np.array([0.0, NET_Y_M, NET_HEIGHT_M]))
        top_r = project_3d_to_image(cam, np.array([COURT_WIDTH_M, NET_Y_M, NET_HEIGHT_M]))

        for cx, cy in cal:
            cv2.circle(frame, (int(cx * w), int(cy * h)), 6, CORNER_COLOR, -1)

        bl = (int(base_l[0] * w), int(base_l[1] * h))
        br = (int(base_r[0] * w), int(base_r[1] * h))
        tl = (int(top_l[0] * w), int(top_l[1] * h))
        tr = (int(top_r[0] * w), int(top_r[1] * h))

        cv2.line(frame, bl, br, NET_BASE_COLOR, 3)
        cv2.line(frame, tl, tr, NET_TOP_COLOR, 3)
        meta = (
            f"rally={rally['rally_id'][:8]} strat={video_strat} MODE=homography "
            f"reproj={cam.reprojection_error:.1f}px focal={cam.focal_length_px:.0f}"
        )

        result_extra = {
            "mode": "homography",
            "reproj_err_px": cam.reprojection_error,
            "focal_px": cam.focal_length_px,
            "keypoint_corner_confs": None,
            "keypoint_center_confs": None,
        }

    elif mode == "tier1":
        # Multi-frame aggregation + cache + sanity. One detect per VIDEO (not rally).
        s3_key = rally.get("processed_s3_key") or rally.get("proxy_s3_key") or rally.get("s3_key")
        if not s3_key:
            return {"ok": False, "reason": "no_s3_key"}

        nl = estimate_net_line_from_s3(
            s3_key,
            video_id=rally["video_id_full"],
            image_width=w,
            image_height=h,
            detector=keypoint_detector,
            n_frames=30,
            duration_s=30.0,
        )
        if nl is None:
            print("  [skip] tier1 net-line estimation failed")
            return {"ok": False, "reason": "tier1_estimation_failed"}

        base_l = nl.base_left_xy
        base_r = nl.base_right_xy
        top_l = nl.top_left_xy
        top_r = nl.top_right_xy

        cv2.circle(frame, (int(base_l[0] * w), int(base_l[1] * h)), 8, CENTER_COLOR, -1)
        cv2.circle(frame, (int(base_r[0] * w), int(base_r[1] * h)), 8, CENTER_COLOR, -1)

        bl = (int(base_l[0] * w), int(base_l[1] * h))
        br = (int(base_r[0] * w), int(base_r[1] * h))
        tl = (int(top_l[0] * w), int(top_l[1] * h))
        tr = (int(top_r[0] * w), int(top_r[1] * h))

        cv2.line(frame, bl, br, NET_BASE_COLOR, 3)
        cv2.line(frame, tl, tr, NET_TOP_COLOR, 3)
        warn_str = ",".join(nl.warnings) if nl.warnings else "-"
        meta = (
            f"rally={rally['rally_id'][:8]} strat={video_strat} MODE=tier1 "
            f"n_frames={nl.n_frames_used} conf={nl.confidence:.2f} "
            f"L={nl.left_source} R={nl.right_source} warn={warn_str}"
        )

        result_extra = {
            "mode": "tier1",
            "reproj_err_px": nl.reproj_err_px,
            "focal_px": nl.focal_px,
            "keypoint_corner_confs": None,
            "keypoint_center_confs": [round(nl.confidence, 3)],
            "n_frames_used": nl.n_frames_used,
            "warnings": nl.warnings,
            "left_source": nl.left_source,
            "right_source": nl.right_source,
        }

    elif mode in ("keypoints_net", "keypoints_hough"):
        assert keypoint_detector is not None
        kpts = _detect_keypoints(keypoint_detector, frame_img)
        if kpts is None:
            print("  [skip] keypoint detection failed")
            return {"ok": False, "reason": "keypoint_detection_failed"}

        cam = calibrate_camera(kpts["corners"], DEFAULT_COURT_CORNERS, w, h)
        if cam is None:
            print("  [skip] keypoint camera calibration failed")
            return {"ok": False, "reason": "keypoint_camera_failed"}

        # Net-BASE directly from keypoint center-left / center-right.
        base_l = kpts["center_left"]
        base_r = kpts["center_right"]

        hom_base_l = project_3d_to_image(cam, np.array([0.0, NET_Y_M, 0.0]))
        hom_base_r = project_3d_to_image(cam, np.array([COURT_WIDTH_M, NET_Y_M, 0.0]))
        hom_top_l = project_3d_to_image(cam, np.array([0.0, NET_Y_M, NET_HEIGHT_M]))
        hom_top_r = project_3d_to_image(cam, np.array([COURT_WIDTH_M, NET_Y_M, NET_HEIGHT_M]))

        # Y shift so observed net-base matches homography-predicted net-base.
        shift_l = base_l[1] - hom_base_l[1]
        shift_r = base_r[1] - hom_base_r[1]
        top_l_predicted = (hom_top_l[0], hom_top_l[1] + shift_l)
        top_r_predicted = (hom_top_r[0], hom_top_r[1] + shift_r)

        # Hough refinement (only in keypoints_hough mode).
        hough_snapped = False
        if mode == "keypoints_hough":
            refined = _hough_refine_net_top(
                frame_img, top_l_predicted, top_r_predicted, base_l, base_r,
            )
            if refined is not None:
                top_l, top_r = refined
                hough_snapped = True
            else:
                top_l, top_r = top_l_predicted, top_r_predicted
        else:
            top_l, top_r = top_l_predicted, top_r_predicted

        # Draw keypoint corners (green) and center points (magenta).
        for (cx, cy) in kpts["corners"]:
            cv2.circle(frame, (int(cx * w), int(cy * h)), 6, CORNER_COLOR, -1)
        cv2.circle(frame, (int(base_l[0] * w), int(base_l[1] * h)), 8, CENTER_COLOR, -1)
        cv2.circle(frame, (int(base_r[0] * w), int(base_r[1] * h)), 8, CENTER_COLOR, -1)

        bl = (int(base_l[0] * w), int(base_l[1] * h))
        br = (int(base_r[0] * w), int(base_r[1] * h))
        tl = (int(top_l[0] * w), int(top_l[1] * h))
        tr = (int(top_r[0] * w), int(top_r[1] * h))

        cv2.line(frame, bl, br, NET_BASE_COLOR, 3)
        cv2.line(frame, tl, tr, NET_TOP_COLOR, 3)

        # If Hough was attempted, also draw the pre-refinement prediction for visual compare.
        if mode == "keypoints_hough":
            tl_pre = (int(top_l_predicted[0] * w), int(top_l_predicted[1] * h))
            tr_pre = (int(top_r_predicted[0] * w), int(top_r_predicted[1] * h))
            cv2.line(frame, tl_pre, tr_pre, (60, 60, 200), 1, lineType=cv2.LINE_AA)

        snap_tag = "SNAP" if hough_snapped else "KEEP"
        meta = (
            f"rally={rally['rally_id'][:8]} strat={video_strat} MODE={mode} "
            f"reproj={cam.reprojection_error:.1f}px focal={cam.focal_length_px:.0f} "
            f"kpt_conf=[{kpts['center_confs'][0]:.2f},{kpts['center_confs'][1]:.2f}] "
            + (snap_tag if mode == "keypoints_hough" else "")
        )

        result_extra = {
            "mode": mode,
            "reproj_err_px": cam.reprojection_error,
            "focal_px": cam.focal_length_px,
            "keypoint_corner_confs": [round(c, 3) for c in kpts["corner_confs"]],
            "keypoint_center_confs": [round(c, 3) for c in kpts["center_confs"]],
            "hough_snapped": hough_snapped,
        }

    else:
        raise ValueError(f"unknown mode {mode!r}")

    cv2.putText(
        frame, "NET BASE (cyan)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, NET_BASE_COLOR, 2,
    )
    cv2.putText(
        frame, "NET TOP (yellow)", (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, NET_TOP_COLOR, 2,
    )
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, meta, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), frame)

    return {
        "ok": True,
        "base_l_x_norm": base_l[0],
        "base_l_y_norm": base_l[1],
        "base_r_x_norm": base_r[0],
        "base_r_y_norm": base_r[1],
        "top_l_x_norm": top_l[0],
        "top_l_y_norm": top_l[1],
        "top_r_x_norm": top_r[0],
        "top_r_y_norm": top_r[1],
        "net_base_to_top_px": (base_l[1] - top_l[1]) * h,
        **result_extra,
    }


def _build_comparison_png(
    panel_paths: list[tuple[str, Path]],
    out_path: Path,
    label: str,
) -> None:
    """Stack N mode outputs side-by-side for rapid visual comparison."""
    panels: list[np.ndarray] = []
    titles: list[str] = []
    ref_shape = None

    for title, path in panel_paths:
        img = cv2.imread(str(path)) if path.exists() else None
        panels.append(img)
        titles.append(title)
        if img is not None and ref_shape is None:
            ref_shape = img.shape

    if ref_shape is None:
        return

    # Fill FAIL placeholders
    for i, img in enumerate(panels):
        if img is None:
            ph = np.zeros(ref_shape, dtype=np.uint8)
            cv2.putText(ph, f"{titles[i]} FAIL", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            panels[i] = ph

    # Resize all to same height
    h = min(p.shape[0] for p in panels)

    def _scale(img: np.ndarray) -> np.ndarray:
        if img.shape[0] != h:
            scale = h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), h))
        return img

    panels = [_scale(p) for p in panels]

    banner_h = 40

    def _banner(text: str, w: int) -> np.ndarray:
        b = np.zeros((banner_h, w, 3), dtype=np.uint8)
        cv2.putText(b, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return b

    banners = [
        _banner(f"[{chr(ord('A') + i)}] {titles[i]}", p.shape[1])
        for i, p in enumerate(panels)
    ]
    stacked = [np.vstack([b, p]) for b, p in zip(banners, panels)]
    combined = np.hstack(stacked)
    title = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    combined = np.vstack([title, combined])
    cv2.imwrite(str(out_path), combined)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/net_line_probe"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument(
        "--mode",
        choices=["homography", "keypoints_net", "keypoints_hough", "tier1", "both", "three", "vs_tier1"],
        default="homography",
        help=(
            "Net-line mode. 'both' = homography + keypoints_net (2-up compare). "
            "'three' = homography + keypoints_net + keypoints_hough (3-up compare). "
            "'tier1' = multi-frame keypoint-aggregated estimator only. "
            "'vs_tier1' = keypoints_net vs tier1 side-by-side."
        ),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    videos = _pick_stratified_videos(args.n)
    print(f"Selected {len(videos)} videos for Probe 1 ({args.mode}):")
    strats = {}
    for v in videos:
        strats[v["strat"]] = strats.get(v["strat"], 0) + 1
    for s, n in strats.items():
        print(f"  {s}: {n}")

    keypoint_detector: CourtKeypointDetector | None = None
    if args.mode in (
        "keypoints_net", "keypoints_hough", "tier1",
        "both", "three", "vs_tier1",
    ):
        keypoint_detector = CourtKeypointDetector()
        if not keypoint_detector.model_exists:
            print(f"ERROR: keypoint model not found at {keypoint_detector._model_path}")
            return 2

    csv_rows: list[dict] = []

    if args.mode == "both":
        modes_to_run = ["homography", "keypoints_net"]
    elif args.mode == "three":
        modes_to_run = ["homography", "keypoints_net", "keypoints_hough"]
    elif args.mode == "vs_tier1":
        modes_to_run = ["keypoints_net", "tier1"]
    else:
        modes_to_run = [args.mode]

    for i, video in enumerate(videos, 1):
        vid = video["video_id"]
        print(f"\n[{i}/{len(videos)}] video={vid[:8]} strat={video['strat']} "
              f"brightness={video['brightness']} persp={video['persp']:.2f}")

        rally = _pick_rally_for_video(vid)
        if rally is None:
            print("  [skip] no rally with ball data")
            continue

        s3_key = rally["processed_s3_key"] or rally["proxy_s3_key"] or rally["s3_key"]
        if not s3_key:
            print("  [skip] no s3 key")
            continue
        time_s = rally["start_ms"] / 1000.0
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if not _extract_frame(s3_key, time_s, tmp_path):
                print("  [skip] frame extraction failed")
                continue

            frame_img = cv2.imread(str(tmp_path))
            if frame_img is None:
                print("  [skip] could not read extracted frame")
                continue

            mode_results: dict[str, dict] = {}
            for m in modes_to_run:
                out_png = args.out_dir / f"{vid[:8]}_{video['strat']}_{m}.png"
                result = _render_frame(
                    rally, video["strat"], frame_img, out_png, m, keypoint_detector,
                )
                mode_results[m] = result
                if result.get("ok"):
                    print(f"  [{m}] reproj={result['reproj_err_px']:.1f}px "
                          f"net_base_y_norm=[{result['base_l_y_norm']:.3f}, {result['base_r_y_norm']:.3f}] "
                          f"net_top_y_norm=[{result['top_l_y_norm']:.3f}, {result['top_r_y_norm']:.3f}]")
                    print(f"  [{m}] wrote {out_png}")
                else:
                    print(f"  [{m}] FAILED: {result.get('reason', 'unknown')}")

            # Emit side-by-side comparison when multiple modes run
            if args.mode in ("both", "three", "vs_tier1"):
                cmp_path = args.out_dir / f"{vid[:8]}_{video['strat']}_COMPARE.png"
                panel_paths = [
                    (m, args.out_dir / f"{vid[:8]}_{video['strat']}_{m}.png")
                    for m in modes_to_run
                ]
                label = (
                    f"{vid[:8]}  strat={video['strat']}  "
                    f"brightness={video['brightness']}  persp={video['persp']:.2f}"
                )
                _build_comparison_png(panel_paths, cmp_path, label)
                print(f"  wrote COMPARE: {cmp_path}")

            # CSV row (primary mode)
            primary_mode = modes_to_run[-1]  # keypoints if 'both', else the single mode
            primary = mode_results.get(primary_mode, {})
            if primary.get("ok"):
                csv_rows.append({
                    "video_id": vid[:8],
                    "rally_id": rally["rally_id"][:8],
                    "strat": video["strat"],
                    "brightness": video["brightness"],
                    "mode": primary_mode,
                    "reproj_err_px": round(primary["reproj_err_px"], 2),
                    "focal_px": round(primary["focal_px"], 0),
                    "base_l_y_norm": round(primary["base_l_y_norm"], 4),
                    "base_r_y_norm": round(primary["base_r_y_norm"], 4),
                    "top_l_y_norm": round(primary["top_l_y_norm"], 4),
                    "top_r_y_norm": round(primary["top_r_y_norm"], 4),
                    "net_base_to_top_px": round(primary["net_base_to_top_px"], 1),
                    "keypoint_corner_confs": primary.get("keypoint_corner_confs"),
                    "keypoint_center_confs": primary.get("keypoint_center_confs"),
                    "net_base_ok": "",
                    "net_top_ok": "",
                    "notes": "",
                })

        finally:
            tmp_path.unlink(missing_ok=True)

    csv_path = args.out_dir / f"probe1_results_{args.mode}.csv"
    if csv_rows:
        with csv_path.open("w", newline="") as f:
            fieldnames = list(csv_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        print(f"\nwrote {csv_path}")
    print(f"Probe 1 done: {len(csv_rows)}/{len(videos)} rendered ({args.mode}).")
    print(f"VISUAL CHECK needed on PNGs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
