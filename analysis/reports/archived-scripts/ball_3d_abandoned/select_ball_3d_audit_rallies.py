"""Select the Phase C audit rallies for the ball 3D verification rig.

Stratifies by:
    - Camera height tier (low <1.3m / mid 1.3-1.7m / high >=1.7m)
    - Ball GT availability (prefer rallies with dense 2D ball GT)
    - Rally must have serve + receive action GT (for the flight-time anchor)
    - Rally must be on a calibrated video (for camera model)

Writes ``analysis/outputs/ball_3d_rig/audit_rallies.json`` — the canonical
list of audit rally IDs used by the rig, flight-time anchor, and audit report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH
from rallycut.court.camera_model import calibrate_camera, calibrate_camera_with_net
from rallycut.evaluation.db import get_connection
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts

COURT_CORNERS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (COURT_WIDTH, 0.0),
    (COURT_WIDTH, COURT_LENGTH),
    (0.0, COURT_LENGTH),
]

# Stratification.
LOW_MAX = 1.3
MID_MAX = 1.7
TIER_TARGETS = {"low": 7, "mid": 7, "high": 6}  # total 20

OUTPUT = Path("outputs/ball_3d_rig/audit_rallies.json")


def _load_video_heights() -> dict[str, dict]:
    """Per calibrated video, estimate camera height via net-constrained PnP."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, court_calibration_json, width, height
            FROM videos
            WHERE court_calibration_json IS NOT NULL
        """)
        videos: dict[str, dict] = {}
        for vid_id, cal_json, w, h in cur.fetchall():
            if not isinstance(cal_json, list) or len(cal_json) != 4:
                continue
            videos[str(vid_id)] = {
                "corners": [(c["x"], c["y"]) for c in cal_json],
                "width": w or 1920,
                "height": h or 1080,
            }

    # Estimate net_y per video using the same approach as eval_ball_3d.
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.video_id, pt.ball_positions_json, pt.positions_json
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id = ANY(%s)
              AND pt.ball_positions_json IS NOT NULL
            ORDER BY r.video_id, r.start_ms
        """, (list(videos.keys()),))
        rallies_by_video: dict[str, list[tuple[str, object, object]]] = {}
        for rid, vid, bp, pp in cur.fetchall():
            rallies_by_video.setdefault(str(vid), []).append((str(rid), bp, pp))

    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

    net_y_per_video: dict[str, float] = {}
    for vid_id, rallies in rallies_by_video.items():
        calibrator = CourtCalibrator()
        calibrator.calibrate(videos[vid_id]["corners"])
        net_ys: list[float] = []
        for _, bp_json, pp_json in rallies[:15]:
            positions = bp_json.get("positions", []) if isinstance(bp_json, dict) else bp_json
            bp_list = [
                BallPosition(
                    frame_number=p.get("frameNumber", p.get("frame_number", 0)),
                    x=p.get("x", 0.0),
                    y=p.get("y", 0.0),
                    confidence=p.get("confidence", 0.5),
                )
                for p in (positions or [])
            ]
            if len(bp_list) < 20:
                continue
            pp_list = [
                PlayerPosition(
                    frame_number=p.get("frameNumber", p.get("frame_number", 0)),
                    track_id=p.get("trackId", p.get("track_id", 0)),
                    x=p.get("x", 0.0),
                    y=p.get("y", 0.0),
                    width=p.get("width", 0.0),
                    height=p.get("height", 0.0),
                    confidence=p.get("confidence", 0.5),
                )
                for p in (pp_json or [])
            ]
            cs = detect_contacts(
                ball_positions=bp_list,
                player_positions=pp_list,
                config=ContactDetectionConfig(),
                court_calibrator=calibrator,
            )
            if 0.1 < cs.net_y < 0.9:
                net_ys.append(cs.net_y)
        if net_ys:
            net_y_per_video[vid_id] = float(np.median(net_ys))

    # Calibrate cameras and extract height.
    result: dict[str, dict] = {}
    for vid_id, v in videos.items():
        cam = None
        if vid_id in net_y_per_video:
            cam = calibrate_camera_with_net(
                v["corners"], COURT_CORNERS, v["width"], v["height"],
                net_y_image=net_y_per_video[vid_id],
            )
        if cam is None or not cam.is_valid:
            cam = calibrate_camera(v["corners"], COURT_CORNERS, v["width"], v["height"])
        if cam is not None and cam.is_valid:
            result[vid_id] = {
                "camera_height_m": float(cam.camera_position[2]),
                "focal_px": float(cam.focal_length_px),
                "reproj_px": float(cam.reprojection_error),
                "width": v["width"],
                "height": v["height"],
            }
    return result


def _load_rally_metadata(video_ids: list[str]) -> list[dict]:
    """Load rally-level metadata: action GT, ball GT, serve+receive presence."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                r.id, r.video_id, pt.fps, pt.frame_count,
                pt.action_ground_truth_json,
                pt.ground_truth_json IS NOT NULL as has_ball_gt
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id = ANY(%s)
              AND pt.action_ground_truth_json IS NOT NULL
              AND pt.ball_positions_json IS NOT NULL
        """, (video_ids,))
        rows = cur.fetchall()

    # Load ball GT counts for those that have it.
    ball_gt_rallies = [r[0] for r in rows if r[5]]
    ball_gt_counts: dict[str, int] = {}
    if ball_gt_rallies:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT r.id, pt.ground_truth_json
                FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.id = ANY(%s)
            """, (ball_gt_rallies,))
            for rid, gt in cur.fetchall():
                if not gt:
                    continue
                positions = gt.get("positions", [])
                ball_gt_counts[str(rid)] = sum(
                    1 for p in positions
                    if isinstance(p, dict)
                    and (p.get("label") or "").lower() == "ball"
                )

    result: list[dict] = []
    for rid, vid, fps, frames, action_gt, has_ball_gt in rows:
        actions = [lab.get("action") for lab in (action_gt or [])]
        has_serve = "serve" in actions
        has_receive = "receive" in actions
        serve_frame = next((l["frame"] for l in action_gt if l.get("action") == "serve"), None)
        receive_frame = next((l["frame"] for l in action_gt if l.get("action") == "receive"), None)
        if not (has_serve and has_receive):
            continue
        flight_frames = receive_frame - serve_frame if (serve_frame is not None and receive_frame is not None) else None
        flight_seconds = flight_frames / fps if (flight_frames and fps) else None
        # Filter unphysical flights (<0.2s or >3.0s).
        if flight_seconds is None or not (0.2 <= flight_seconds <= 3.0):
            continue
        result.append({
            "rally_id": str(rid),
            "video_id": str(vid),
            "fps": float(fps or 30.0),
            "frame_count": int(frames or 0),
            "serve_frame": int(serve_frame),
            "receive_frame": int(receive_frame),
            "flight_seconds": float(flight_seconds),
            "action_gt_count": len(action_gt or []),
            "ball_gt_count": ball_gt_counts.get(str(rid), 0),
            "has_dense_ball_gt": ball_gt_counts.get(str(rid), 0) >= 50,
        })
    return result


def _tier_of(height_m: float) -> str:
    if height_m < LOW_MAX:
        return "low"
    if height_m < MID_MAX:
        return "mid"
    return "high"


def main() -> None:
    print("Loading calibrated video heights...")
    videos = _load_video_heights()
    print(f"  {len(videos)} calibrated videos")

    print("Loading rally metadata (serve+receive GT, ball GT)...")
    rallies = _load_rally_metadata(list(videos.keys()))
    print(f"  {len(rallies)} candidate rallies with serve+receive")

    # Attach tier.
    for r in rallies:
        v = videos.get(r["video_id"])
        if not v:
            r["tier"] = None
            continue
        r["camera_height_m"] = v["camera_height_m"]
        r["tier"] = _tier_of(v["camera_height_m"])

    rallies = [r for r in rallies if r.get("tier")]

    tier_counts = {t: sum(1 for r in rallies if r["tier"] == t) for t in ("low", "mid", "high")}
    print(f"  Tier breakdown (candidates): {tier_counts}")

    # Selection: prefer dense ball GT, then spread across tiers and videos.
    selected: list[dict] = []
    rallies_sorted = sorted(
        rallies,
        key=lambda r: (not r["has_dense_ball_gt"], -r["ball_gt_count"], r["video_id"], r["serve_frame"]),
    )

    used_videos_per_tier: dict[str, set[str]] = {"low": set(), "mid": set(), "high": set()}
    tier_counts_sel: dict[str, int] = {"low": 0, "mid": 0, "high": 0}

    # Pass 1: pick rallies with dense ball GT, distributing across videos.
    for r in rallies_sorted:
        tier = r["tier"]
        if tier_counts_sel[tier] >= TIER_TARGETS[tier]:
            continue
        if r["video_id"] in used_videos_per_tier[tier]:
            continue  # spread across videos first
        if not r["has_dense_ball_gt"]:
            continue
        selected.append(r)
        used_videos_per_tier[tier].add(r["video_id"])
        tier_counts_sel[tier] += 1

    # Pass 2: fill remaining slots allowing duplicate videos per tier.
    for r in rallies_sorted:
        tier = r["tier"]
        if tier_counts_sel[tier] >= TIER_TARGETS[tier]:
            continue
        if r in selected:
            continue
        if not r["has_dense_ball_gt"]:
            continue
        selected.append(r)
        tier_counts_sel[tier] += 1

    # Pass 3: if still short, take non-dense-ball-GT rallies.
    for r in rallies_sorted:
        tier = r["tier"]
        if tier_counts_sel[tier] >= TIER_TARGETS[tier]:
            continue
        if r in selected:
            continue
        selected.append(r)
        tier_counts_sel[tier] += 1

    print(f"\nSelected: {len(selected)} rallies")
    print(f"  Tier breakdown: {tier_counts_sel}")
    dense_count = sum(1 for r in selected if r["has_dense_ball_gt"])
    print(f"  With dense ball GT: {dense_count}/{len(selected)}")

    # Summarize heights.
    heights = [r["camera_height_m"] for r in selected]
    print(f"  Heights: min={min(heights):.2f} median={np.median(heights):.2f} max={max(heights):.2f} m")

    # Print per-rally.
    print("\nPer-rally:")
    for r in sorted(selected, key=lambda r: (r["tier"], r["camera_height_m"], r["rally_id"])):
        dense = "★" if r["has_dense_ball_gt"] else "·"
        print(
            f"  [{r['tier']:4s}] h={r['camera_height_m']:.2f}m "
            f"{r['rally_id'][:8]} vid={r['video_id'][:8]} "
            f"flight={r['flight_seconds']:.2f}s "
            f"ball_gt={r['ball_gt_count']:4d} {dense}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "selection_criteria": {
            "tier_low_max": LOW_MAX,
            "tier_mid_max": MID_MAX,
            "tier_targets": TIER_TARGETS,
        },
        "video_heights": {
            vid: v["camera_height_m"] for vid, v in videos.items()
        },
        "audit_rallies": selected,
    }, indent=2))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
