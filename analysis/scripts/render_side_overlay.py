"""Per-rally ball-side visual overlay — game-semantics Probe scoping (2026-04-22).

Emits:
  - MP4 with net-base (cyan) + net-top (yellow) lines drawn + ball colored by
    predicted side (red = near, blue = far, magenta = ambiguous).
  - Per-frame text: `f=<frame> side=<near|far|ambiguous>` + a flip indicator.

Required by docs/superpowers/plans/2026-04-22-game-semantics-scoping.md:
  "Before trusting any automated net-cross signal we need humans in the loop
   to sanity-check that the signal matches perception."

Consumes:
  - videos.court_calibration_json (4 normalised corners)
  - player_tracks.ball_positions_json (per-frame ball x,y in normalised coords)
  - processed_s3_key (where tracking ran — frame indices align)

Usage:
    uv run python analysis/scripts/render_side_overlay.py \
        --rally-id <full-or-prefix-id> \
        --out reports/session_game_semantics/overlay_<prefix>.mp4

    uv run python analysis/scripts/render_side_overlay.py \
        --video-id <full-video-id> --limit 5 \
        --out-dir reports/session_game_semantics/overlays
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.court.camera_model import CameraModel, calibrate_camera, project_3d_to_image
from rallycut.court.keypoint_detector import CourtKeypointDetector
from rallycut.court.net_line_estimator import estimate_net_line_from_s3
from rallycut.evaluation.db import get_connection

COURT_WIDTH_M = 8.0
COURT_LENGTH_M = 16.0
NET_Y_M = COURT_LENGTH_M / 2.0  # y=8 — net at the midline
NET_HEIGHT_M = 2.43  # men's beach (women: 2.24)

# Ambiguity band around net-top line (normalized image-Y).
# Within this band we flag "ambiguous" rather than commit to near/far.
AMBIG_BAND_NORM_Y = 0.008  # ~9px at 1080p, ~6px at 720p

DEFAULT_COURT_CORNERS = [
    (0.0, 0.0),
    (COURT_WIDTH_M, 0.0),
    (COURT_WIDTH_M, COURT_LENGTH_M),
    (0.0, COURT_LENGTH_M),
]


@dataclass
class RallyData:
    rally_id: str
    video_id: str
    start_ms: int
    end_ms: int
    fps: float
    width: int
    height: int
    s3_key: str
    processed_s3_key: str | None
    proxy_s3_key: str | None
    calibration: list[tuple[float, float]] | None
    ball_positions: list[dict]


def _load_rally(rally_ref: str) -> RallyData | None:
    like = rally_ref + "%"
    q = """
        SELECT r.id, r.video_id, r.start_ms, r.end_ms, pt.fps,
               v.width, v.height, v.s3_key, v.processed_s3_key, v.proxy_s3_key,
               v.court_calibration_json, pt.ball_positions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE r.id LIKE %s LIMIT 1
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(q, (like,))
        row = cur.fetchone()
    if row is None:
        return None
    cal_json = row[10]
    cal = None
    if isinstance(cal_json, list) and len(cal_json) == 4:
        cal = [(float(c["x"]), float(c["y"])) for c in cal_json]
    ball_raw = row[11] or []
    ball_list = ball_raw if isinstance(ball_raw, list) else json.loads(ball_raw)
    return RallyData(
        rally_id=str(row[0]),
        video_id=str(row[1]),
        start_ms=int(row[2]),
        end_ms=int(row[3]),
        fps=float(row[4] or 30.0),
        width=int(row[5] or 1920),
        height=int(row[6] or 1080),
        s3_key=str(row[7] or ""),
        processed_s3_key=row[8],
        proxy_s3_key=row[9],
        calibration=cal,
        ball_positions=list(ball_list),
    )


def _load_rallies_for_video(video_id: str, limit: int | None) -> list[str]:
    q = """
        SELECT r.id
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE v.id = %s
          AND pt.ball_positions_json IS NOT NULL
        ORDER BY r.start_ms
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(q, (video_id,))
        rows = cur.fetchall()
    ids = [str(r[0]) for r in rows]
    return ids[:limit] if limit else ids


def _s3_presigned_url(s3_key: str) -> str:
    # Match existing pattern from extract_serve_debug_clips.py
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


def _download_clip(s3_key: str, start_s: float, duration_s: float, out: Path) -> bool:
    url = _s3_presigned_url(s3_key)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", url,
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration_s:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-an", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg error: {r.stderr.strip()}")
        return False
    return out.exists() and out.stat().st_size > 0


def _compute_net_endpoints_tier1(
    rally: RallyData,
    detector: CourtKeypointDetector,
) -> tuple[str, float, tuple[float, float], tuple[float, float],
           tuple[float, float], tuple[float, float]] | None:
    """Run the Tier 1 cached estimator. Returns
    (provenance, confidence, base_l, base_r, top_l, top_r) or None on failure."""
    s3_key = rally.processed_s3_key or rally.proxy_s3_key or rally.s3_key
    if not s3_key:
        return None
    nl = estimate_net_line_from_s3(
        s3_key,
        video_id=rally.video_id,
        image_width=rally.width,
        image_height=rally.height,
        detector=detector,
        n_frames=30,
        duration_s=30.0,
    )
    if nl is None:
        return None
    return (
        f"tier1 n={nl.n_frames_used} L={nl.left_source} R={nl.right_source} "
        f"warn={','.join(nl.warnings) or '-'}",
        nl.confidence,
        nl.base_left_xy,
        nl.base_right_xy,
        nl.top_left_xy,
        nl.top_right_xy,
    )


def _compute_net_endpoints_homography(
    rally: RallyData,
) -> tuple[CameraModel | None, tuple[float, float], tuple[float, float],
           tuple[float, float], tuple[float, float]]:
    """Legacy homography-based net-line. Kept as a fallback."""
    if rally.calibration is None:
        return None, (0, 0), (0, 0), (0, 0), (0, 0)

    cam = calibrate_camera(rally.calibration, DEFAULT_COURT_CORNERS, rally.width, rally.height)
    if cam is None:
        return None, (0, 0), (0, 0), (0, 0), (0, 0)

    base_l = project_3d_to_image(cam, np.array([0.0, NET_Y_M, 0.0]))
    base_r = project_3d_to_image(cam, np.array([COURT_WIDTH_M, NET_Y_M, 0.0]))
    top_l = project_3d_to_image(cam, np.array([0.0, NET_Y_M, NET_HEIGHT_M]))
    top_r = project_3d_to_image(cam, np.array([COURT_WIDTH_M, NET_Y_M, NET_HEIGHT_M]))
    return cam, base_l, base_r, top_l, top_r


def _net_top_y_at_x(
    x_norm: float,
    top_l: tuple[float, float],
    top_r: tuple[float, float],
) -> float:
    """Interpolate net-top line at ball's image-x — linear in normalized coords."""
    x1, y1 = top_l
    x2, y2 = top_r
    if abs(x2 - x1) < 1e-8:
        return (y1 + y2) / 2.0
    # Clamp to line segment slope (extend linearly beyond endpoints for balls
    # that project outside sideline-to-sideline range).
    t = (x_norm - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def _classify_side(
    ball_y_norm: float,
    net_top_y_at_x: float,
    ambig_band: float = AMBIG_BAND_NORM_Y,
) -> str:
    """Net-top decisive: image-Y smaller than net-top-line => 'far'; larger => 'near'."""
    delta = ball_y_norm - net_top_y_at_x
    if abs(delta) < ambig_band:
        return "ambiguous"
    return "near" if delta > 0 else "far"


def _render_rally(
    rally: RallyData,
    out_path: Path,
    *,
    net_mode: str = "tier1",
    detector: CourtKeypointDetector | None = None,
) -> bool:
    """Download + overlay one rally. Returns True on success.

    net_mode:
      - 'tier1' (default): use the multi-frame cached keypoint estimator.
      - 'homography': fall back to manual-calibration homography (legacy).
    """
    if net_mode == "tier1":
        if detector is None:
            detector = CourtKeypointDetector()
            if not detector.model_exists:
                print("  [skip] keypoint model not available")
                return False
        tier1 = _compute_net_endpoints_tier1(rally, detector)
        if tier1 is None:
            print(f"  [skip] rally {rally.rally_id[:8]} tier1 net-line estimation failed")
            return False
        provenance, confidence, base_l, base_r, top_l, top_r = tier1
        print(f"  {provenance}  conf={confidence:.2f}")
    else:
        if rally.calibration is None:
            print(f"  [skip] rally {rally.rally_id[:8]} has no calibration")
            return False
        cam, base_l, base_r, top_l, top_r = _compute_net_endpoints_homography(rally)
        if cam is None:
            print(f"  [skip] rally {rally.rally_id[:8]} homography calibration failed")
            return False
        print(f"  homography reproj_err={cam.reprojection_error:.2f}px focal={cam.focal_length_px:.0f}")

    print(f"  net_base image-y: L={base_l[1]:.3f} R={base_r[1]:.3f}")
    print(f"  net_top  image-y: L={top_l[1]:.3f} R={top_r[1]:.3f}")

    # Choose which s3 key: processed_s3_key for tracking-aligned frames;
    # fall back to proxy or original. We rely on FPS metadata for frame alignment.
    s3_key = rally.processed_s3_key or rally.proxy_s3_key or rally.s3_key
    if not s3_key:
        print(f"  [skip] rally {rally.rally_id[:8]} has no s3 key")
        return False

    duration_s = max(0.5, (rally.end_ms - rally.start_ms) / 1000.0)
    start_s = rally.start_ms / 1000.0

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if not _download_clip(s3_key, start_s, duration_s + 0.5, tmp_path):
            print(f"  [skip] download failed for rally {rally.rally_id[:8]}")
            return False

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            print(f"  [skip] cannot open clip for rally {rally.rally_id[:8]}")
            return False
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or rally.fps
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or rally.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or rally.height
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(duration_s * vid_fps)

        ball_by_frame: dict[int, dict] = {}
        for bp in rally.ball_positions:
            fn = bp.get("frameNumber", -1)
            if fn >= 0:
                ball_by_frame[fn] = bp

        # Line endpoints in pixel coords.
        bl = (int(base_l[0] * w), int(base_l[1] * h))
        br = (int(base_r[0] * w), int(base_r[1] * h))
        tl = (int(top_l[0] * w), int(top_l[1] * h))
        tr = (int(top_r[0] * w), int(top_r[1] * h))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, vid_fps, (w, h))

        NET_BASE_COLOR = (255, 255, 0)   # cyan  # noqa: N806
        NET_TOP_COLOR = (0, 255, 255)     # yellow (BGR)  # noqa: N806
        NEAR_COLOR = (0, 0, 255)          # red  # noqa: N806
        FAR_COLOR = (255, 0, 0)           # blue  # noqa: N806
        AMB_COLOR = (255, 0, 255)         # magenta  # noqa: N806

        prev_side: str | None = None
        flip_count = 0

        for frame_i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            cv2.line(frame, bl, br, NET_BASE_COLOR, 2)
            cv2.line(frame, tl, tr, NET_TOP_COLOR, 2)

            rally_frame = frame_i
            bp = ball_by_frame.get(rally_frame)
            side = "no_ball"
            if bp is not None and (bp.get("x", 0) > 0 or bp.get("y", 0) > 0):
                bx = float(bp["x"])
                by = float(bp["y"])
                net_top_y_here = _net_top_y_at_x(bx, top_l, top_r)
                side = _classify_side(by, net_top_y_here)
                color = (
                    NEAR_COLOR if side == "near"
                    else FAR_COLOR if side == "far"
                    else AMB_COLOR
                )
                cv2.circle(frame, (int(bx * w), int(by * h)), 8, color, -1)
                cv2.circle(frame, (int(bx * w), int(by * h)), 10, (255, 255, 255), 1)

            flipped = (
                prev_side is not None
                and prev_side in ("near", "far")
                and side in ("near", "far")
                and prev_side != side
            )
            if flipped:
                flip_count += 1
                cv2.putText(
                    frame, "FLIP", (w - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
            if side in ("near", "far"):
                prev_side = side

            label = f"f={rally_frame} side={side} flips={flip_count}"
            cv2.rectangle(frame, (5, 5), (420, 30), (0, 0, 0), -1)
            cv2.putText(
                frame, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

            writer.write(frame)

        writer.release()
        cap.release()
        print(f"  wrote {out_path} ({n_frames} frames, {flip_count} side-flips)")
        return True
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rally-id", help="Rally ID (full or 8-char prefix)")
    ap.add_argument("--video-id", help="Video ID — renders all rallies in the video")
    ap.add_argument("--limit", type=int, help="Max rallies when using --video-id")
    ap.add_argument("--out", type=Path, help="Output mp4 (with --rally-id)")
    ap.add_argument("--out-dir", type=Path, help="Output dir (with --video-id)")
    ap.add_argument(
        "--net-mode", choices=["tier1", "homography"], default="tier1",
        help="Net-line estimator: tier1 (auto, multi-frame keypoints) or homography (manual calibration).",
    )
    args = ap.parse_args()

    if not args.rally_id and not args.video_id:
        ap.error("one of --rally-id or --video-id required")

    # Shared detector instance keeps weights in memory across calls; cache handles reuse.
    detector = CourtKeypointDetector() if args.net_mode == "tier1" else None

    if args.rally_id:
        if args.out is None:
            ap.error("--out required with --rally-id")
        rally = _load_rally(args.rally_id)
        if rally is None:
            print(f"rally {args.rally_id!r} not found")
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        ok = _render_rally(rally, args.out, net_mode=args.net_mode, detector=detector)
        return 0 if ok else 1

    assert args.video_id is not None
    if args.out_dir is None:
        ap.error("--out-dir required with --video-id")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rids = _load_rallies_for_video(args.video_id, args.limit)
    print(f"rendering {len(rids)} rallies ({args.net_mode})")
    wins = 0
    for i, rid in enumerate(rids, 1):
        print(f"[{i}/{len(rids)}] rally {rid[:8]}")
        rally = _load_rally(rid)
        if rally is None:
            print("  [skip] missing rally row")
            continue
        out = args.out_dir / f"{rid[:8]}.mp4"
        if _render_rally(rally, out, net_mode=args.net_mode, detector=detector):
            wins += 1
    print(f"done: {wins}/{len(rids)} rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
