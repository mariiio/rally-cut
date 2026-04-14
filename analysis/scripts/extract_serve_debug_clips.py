"""Extract short debug clips of serve formation for visual inspection.

Dynamically enumerates ALL formation errors by running the formation
predictor against GT labels, then extracts annotated clips for each.

Overlays player bounding boxes (colored by near/far side), ball detections,
court split line, formation prediction, and late-arriving track indicators.

Usage:
    cd analysis
    uv run python scripts/extract_serve_debug_clips.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _compute_auto_split_y,
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

OUTPUT_DIR = Path("outputs/serve_debug_clips")


# ── Error enumeration ────────────────────────────────────────────────────


@dataclass
class FormationError:
    rally_id: str
    video_id: str
    gt_side: str  # "near" / "far"
    pred_side: str | None  # "near" / "far" / None
    confidence: float
    n_tracks: int
    n_near: int
    n_far: int
    video_name: str = ""


def _gt_physical_side(
    gt_serving_team: str, side_flipped: bool, initial_near_is_a: bool,
) -> str:
    near_is_a = initial_near_is_a != side_flipped  # XOR
    if gt_serving_team == "A":
        return "near" if near_is_a else "far"
    else:
        return "far" if near_is_a else "near"


EXCLUDED_VIDEOS = {
    "0a383519",  # yoyo / IMG_2313 — low camera angle
    "627c1add",  # caca — tilted camera
}


def _enumerate_formation_errors() -> list[FormationError]:
    """Find all rallies where formation prediction != GT physical side."""
    from scripts.eval_score_tracking import RallyData, load_score_gt

    video_rallies = load_score_gt()
    # Filter out non-target videos
    video_rallies = {
        vid: rallies for vid, rallies in video_rallies.items()
        if not any(vid.startswith(ex) for ex in EXCLUDED_VIDEOS)
    }
    print(f"Loaded {sum(len(v) for v in video_rallies.values())} rallies "
          f"from {len(video_rallies)} videos (excluded {len(EXCLUDED_VIDEOS)} non-target)")

    # Load video names (s3_key stem)
    video_names: dict[str, str] = {}
    with get_connection() as conn, conn.cursor() as cur:
        vids = list(video_rallies.keys())
        placeholders = ", ".join(["%s"] * len(vids))
        cur.execute(
            f"SELECT id, s3_key FROM videos WHERE id IN ({placeholders})", vids,
        )
        for vid, s3_key in cur.fetchall():
            video_names[vid] = Path(s3_key).stem if s3_key else vid[:8]

    # Per-video convention calibration (majority vote)
    initial_near_is_a: dict[str, bool] = {}
    for vid, rallies in video_rallies.items():
        positions_by_rally: dict[str, list[PlayerPosition]] = {}
        net_ys: dict[str, float] = {}
        for r in rallies:
            positions_by_rally[r.rally_id] = _parse_positions(r.positions)
            net_ys[r.rally_id] = r.court_split_y or 0.5

        # Majority vote
        votes_true = 0
        votes_false = 0
        for r in rallies:
            pos = positions_by_rally[r.rally_id]
            ny = net_ys[r.rally_id]
            pred_side, _ = _find_serving_side_by_formation(pos, net_y=ny, start_frame=0)
            if pred_side is None:
                continue
            gt_if_true = _gt_physical_side(r.gt_serving_team, r.side_flipped, True)
            gt_if_false = _gt_physical_side(r.gt_serving_team, r.side_flipped, False)
            if pred_side == gt_if_true:
                votes_true += 1
            if pred_side == gt_if_false:
                votes_false += 1
        initial_near_is_a[vid] = votes_true >= votes_false

    # Find errors
    errors: list[FormationError] = []
    for vid, rallies in video_rallies.items():
        near_is_a = initial_near_is_a[vid]
        for r in rallies:
            positions = _parse_positions(r.positions)
            net_y = r.court_split_y or 0.5
            pred_side, conf = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            gt_phys = _gt_physical_side(r.gt_serving_team, r.side_flipped, near_is_a)

            if pred_side != gt_phys:
                # Count tracks per side
                by_track: dict[int, list[float]] = defaultdict(list)
                for p in positions:
                    if p.track_id >= 0 and p.frame_number < 120:
                        by_track[p.track_id].append(p.y + p.height / 2.0)
                track_medians = {
                    tid: sum(ys) / len(ys) for tid, ys in by_track.items()
                }
                effective_split = net_y
                n_near = sum(1 for y in track_medians.values() if y > effective_split)
                n_far = len(track_medians) - n_near
                if n_near == 0 or n_far == 0:
                    auto = _compute_auto_split_y(
                        [p for p in positions if p.track_id >= 0 and p.frame_number < 120]
                    )
                    if auto is not None:
                        effective_split = auto
                        n_near = sum(
                            1 for y in track_medians.values() if y > effective_split
                        )
                        n_far = len(track_medians) - n_near

                errors.append(FormationError(
                    rally_id=r.rally_id,
                    video_id=vid,
                    gt_side=gt_phys,
                    pred_side=pred_side,
                    confidence=conf,
                    n_tracks=len(track_medians),
                    n_near=n_near,
                    n_far=n_far,
                    video_name=video_names.get(vid, vid[:8]),
                ))

    return errors


# ── Clip extraction helpers ──────────────────────────────────────────────


def _parse_positions(raw: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=p["frameNumber"],
            track_id=p["trackId"],
            x=p["x"],
            y=p["y"],
            width=p.get("width", 0.05),
            height=p.get("height", 0.10),
            confidence=p.get("confidence", 1.0),
            keypoints=p.get("keypoints"),
        )
        for p in raw
    ]


def _get_video_url(s3_key: str) -> str:
    """Build presigned MinIO URL for local dev (OpenCV needs this)."""
    import boto3
    from botocore.config import Config
    client = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": "rallycut-dev", "Key": s3_key},
        ExpiresIn=3600,
    )
    return url


def load_rally_data(rally_ids: list[str]) -> list[dict]:
    """Load rally + player_track + video data for specified rallies."""
    if not rally_ids:
        return []
    placeholders = ", ".join(["%s"] * len(rally_ids))
    query = f"""
        SELECT r.id, r.video_id, r.start_ms, r.gt_serving_team,
               pt.positions_json, pt.ball_positions_json,
               pt.court_split_y, pt.fps, pt.contacts_json,
               v.s3_key, v.processed_s3_key, v.width, v.height,
               v.proxy_s3_key, v.court_calibration_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE r.id IN ({placeholders})
    """
    results = []
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, rally_ids)
        for row in cur.fetchall():
            # Compute calibrated net Y from court calibration
            cal_net_y = None
            court_cal = row[14]
            if court_cal and isinstance(court_cal, list) and len(court_cal) == 4:
                try:
                    from rallycut.court.calibration import CourtCalibrator
                    cal = CourtCalibrator()
                    cal.calibrate([(c["x"], c["y"]) for c in court_cal])
                    if cal.is_calibrated:
                        _, cy = cal.court_to_image((4, 8), 1, 1)
                        cal_net_y = cy
                except Exception:
                    pass

            # Find first contact/ball frame for timing anchor
            first_ball_frame = None
            ball_json = row[5] or []
            for bp in ball_json:
                fn = bp.get("frameNumber", -1)
                if fn >= 0 and (bp.get("x", 0) > 0 or bp.get("y", 0) > 0):
                    if first_ball_frame is None or fn < first_ball_frame:
                        first_ball_frame = fn
                    break  # ball_positions should be sorted

            first_contact_frame = None
            contacts_json = row[8]
            if contacts_json and isinstance(contacts_json, dict):
                contacts_list = contacts_json.get("contacts", [])
                if contacts_list:
                    first_contact_frame = contacts_list[0].get("frame")

            results.append({
                "rally_id": row[0],
                "video_id": row[1],
                "start_ms": row[2],
                "gt_serving_team": row[3],
                "positions_json": row[4] or [],
                "ball_positions_json": ball_json,
                "court_split_y": row[6],
                "fps": row[7] or 30.0,
                "s3_key": row[9],
                "processed_s3_key": row[10],
                "width": row[11],
                "height": row[12],
                "proxy_s3_key": row[13],
                "cal_net_y": cal_net_y,
                "first_ball_frame": first_ball_frame,
                "first_contact_frame": first_contact_frame,
            })
    return results


def _download_clip(s3_key: str, start_s: float, duration_s: float, tmp_path: Path) -> bool:
    """Download a clip segment using ffmpeg from presigned URL."""
    import subprocess
    url = _get_video_url(s3_key)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", url,
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration_s:.1f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr.strip()}")
        return False
    return tmp_path.exists() and tmp_path.stat().st_size > 0


def extract_clip(
    rally: dict,
    output_path: Path,
    duration_s: float = 5.0,
    gt_physical_side: str | None = None,
) -> bool:
    """Extract a debug clip with tracking overlay."""
    import tempfile

    # Use processed video (same fps as tracking), not proxy (may be resampled)
    s3_key = rally["processed_s3_key"] or rally["s3_key"]
    # Always start exactly at rally start_ms so overlay frame 0 = position frame 0
    start_s = rally["start_ms"] / 1000.0

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    if not _download_clip(s3_key, start_s, duration_s + 1.0, tmp_path):
        tmp_path.unlink(missing_ok=True)
        return False

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        print(f"  Cannot open downloaded clip")
        tmp_path.unlink(missing_ok=True)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or rally["fps"]
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = int(duration_s * fps)

    # Parse tracking data
    positions = _parse_positions(rally["positions_json"])
    ball_positions = rally["ball_positions_json"]
    net_y = rally["court_split_y"] if rally["court_split_y"] else 0.5

    # Run formation prediction
    pred_side, pred_conf = _find_serving_side_by_formation(
        positions, net_y=net_y, start_frame=0,
    )

    # Group positions by frame (rally-relative)
    pos_by_frame: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        pos_by_frame.setdefault(p.frame_number, []).append(p)

    # Compute per-track first frame for late-entry annotation
    track_first_frame: dict[int, int] = {}
    track_first_pos: dict[int, tuple[float, float]] = {}
    for p in positions:
        if p.track_id < 0:
            continue
        if p.track_id not in track_first_frame or p.frame_number < track_first_frame[p.track_id]:
            track_first_frame[p.track_id] = p.frame_number
            track_first_pos[p.track_id] = (p.x, p.y + p.height / 2.0)

    ball_by_frame: dict[int, dict] = {}
    for bp in ball_positions:
        fn = bp.get("frameNumber", -1)
        if fn >= 0:
            ball_by_frame[fn] = bp

    # Output video (clip already starts at rally start)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = min(fps, 30.0)  # Cap output at 30fps
    frame_skip = max(1, int(fps / out_fps))
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))

    # Colors
    NEAR_COLOR = (0, 200, 0)  # Green = near side
    FAR_COLOR = (200, 0, 0)   # Blue = far side
    LATE_COLOR = (0, 165, 255)  # Orange = late-arriving track
    BALL_COLOR = (0, 255, 255)  # Yellow = ball
    NET_COLOR = (255, 255, 0)  # Cyan = net line
    TEXT_BG = (0, 0, 0)

    for frame_i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i % frame_skip != 0:
            continue

        rally_frame = frame_i

        # Draw court_split_y line (player-feet-based)
        net_y_px = int(net_y * h)
        cv2.line(frame, (0, net_y_px), (w, net_y_px), NET_COLOR, 2)
        cv2.putText(frame, f"court_split_y={net_y:.3f}", (10, net_y_px - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, NET_COLOR, 1)

        # Draw calibrated net position (from homography)
        cal_net_y = rally.get("cal_net_y")
        if cal_net_y is not None:
            cal_px = int(cal_net_y * h)
            CAL_COLOR = (0, 0, 255)  # Red = calibrated net
            cv2.line(frame, (0, cal_px), (w, cal_px), CAL_COLOR, 2)
            cv2.putText(frame, f"cal_net={cal_net_y:.3f}", (10, cal_px + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CAL_COLOR, 1)

        # Draw player bboxes
        frame_positions = pos_by_frame.get(rally_frame, [])
        for p in frame_positions:
            px = int(p.x * w)
            py = int(p.y * h)
            pw = int(p.width * w)
            ph = int(p.height * h)
            x1 = px - pw // 2
            y1 = py - ph // 2
            x2 = px + pw // 2
            y2 = py + ph // 2

            foot_y = p.y + p.height / 2.0
            side = "near" if foot_y > net_y else "far"
            color = NEAR_COLOR if side == "near" else FAR_COLOR

            # Check if this track arrived late (potential server-enters-frame)
            first_f = track_first_frame.get(p.track_id, 0)
            is_late = first_f > 15
            if is_late:
                fx, fy = track_first_pos.get(p.track_id, (0.5, 0.5))
                at_edge = fx < 0.05 or fx > 0.95 or fy > 0.85
                # Orange for edge-entry, yellow for non-edge late
                box_color = LATE_COLOR if at_edge else (0, 255, 255)
            else:
                box_color = color

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"T{p.track_id} {side}"
            if is_late:
                edge_tag = "EDGE" if (track_first_pos.get(p.track_id, (0.5, 0.5))[0] < 0.05
                    or track_first_pos.get(p.track_id, (0.5, 0.5))[0] > 0.95
                    or track_first_pos.get(p.track_id, (0.5, 0.5))[1] > 0.85) else "MID"
                label += f" LATE(f={first_f},{edge_tag})"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Draw ball
        bp = ball_by_frame.get(rally_frame)
        if bp:
            bx = int(bp["x"] * w)
            by = int(bp["y"] * h)
            cv2.circle(frame, (bx, by), 8, BALL_COLOR, -1)
            cv2.circle(frame, (bx, by), 10, (0, 0, 0), 2)

        # Info overlay
        first_ball = rally.get("first_ball_frame", "?")
        first_contact = rally.get("first_contact_frame", "?")
        # Use convention-aware GT physical side if provided
        if gt_physical_side is not None:
            verdict = ("CORRECT" if pred_side == gt_physical_side
                       else "WRONG" if pred_side else "ABSTAIN")
            gt_label = f"GT: team {rally['gt_serving_team']} = {gt_physical_side} side"
        else:
            verdict = "?"
            gt_label = f"GT: {rally['gt_serving_team']} serves"
        info_lines = [
            f"Rally: {rally['rally_id'][:8]}  Frame: {rally_frame}  "
            f"1st ball: {first_ball}  1st contact: {first_contact}",
            gt_label,
            f"Formation pred: {pred_side or 'None'} (conf={pred_conf:.2f})",
            verdict,
        ]
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (12, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_BG, 3)
            color = (0, 255, 0) if "CORRECT" in line else (
                (0, 0, 255) if "WRONG" in line else (255, 255, 255))
            cv2.putText(frame, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(frame)

    writer.release()
    cap.release()
    tmp_path.unlink(missing_ok=True)
    return True


def main() -> int:
    # Always start fresh — remove old clips to avoid confusion
    if OUTPUT_DIR.exists():
        for old in OUTPUT_DIR.glob("*.mp4"):
            old.unlink()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Enumerate all formation errors dynamically
    print("=== Enumerating formation errors ===")
    errors = _enumerate_formation_errors()
    print(f"\nFound {len(errors)} formation errors")

    if not errors:
        print("No errors to extract clips for.")
        return 0

    # Summary table
    print(f"\n{'rally':>10s}  {'video':<20s}  {'n/f':>5s}  {'gt':>5s}  "
          f"{'pred':>5s}  {'conf':>5s}")
    print("-" * 75)
    for e in sorted(errors, key=lambda x: (x.video_name, x.rally_id)):
        print(f"{e.rally_id[:10]:>10s}  {e.video_name:<20s}  "
              f"{e.n_near}v{e.n_far:>1d}  {e.gt_side:>5s}  "
              f"{(e.pred_side or 'None'):>5s}  {e.confidence:>5.2f}")

    # Step 2: Load clip data for all error rallies
    print(f"\n=== Loading rally data for {len(errors)} errors ===")
    rally_ids = [e.rally_id for e in errors]
    rally_data = load_rally_data(rally_ids)
    data_by_id = {r["rally_id"]: r for r in rally_data}
    print(f"Loaded data for {len(rally_data)} rallies")

    # Step 3: Extract clips
    print(f"\n=== Extracting debug clips ===")
    ok_count = 0
    for e in sorted(errors, key=lambda x: (x.video_name, x.rally_id)):
        if e.rally_id not in data_by_id:
            print(f"  Skipping {e.rally_id[:8]} (no clip data)")
            continue

        rally = data_by_id[e.rally_id]
        filename = (
            f"{e.video_name}_{e.n_near}v{e.n_far}"
            f"_gt{e.gt_side}_{e.rally_id[:8]}.mp4"
        )
        output_path = OUTPUT_DIR / filename

        print(f"  [{ok_count + 1}/{len(errors)}] {filename}...")
        ok = extract_clip(rally, output_path, gt_physical_side=e.gt_side)
        if ok:
            ok_count += 1
            print(f"    -> {output_path}")
        else:
            print(f"    FAILED")

    # Per-video error summary
    print(f"\n=== Per-video summary ===")
    from collections import Counter
    vid_counts: Counter[str] = Counter()
    vid_totals: dict[str, int] = {}
    for e in errors:
        vid_counts[e.video_name] += 1
    # Get total rallies per video from the enumeration data
    from scripts.eval_score_tracking import load_score_gt
    all_rallies = load_score_gt()
    video_names_map = {}
    with get_connection() as conn, conn.cursor() as cur:
        vids = list(all_rallies.keys())
        placeholders = ", ".join(["%s"] * len(vids))
        cur.execute(
            f"SELECT id, s3_key FROM videos WHERE id IN ({placeholders})", vids,
        )
        for vid, s3_key in cur.fetchall():
            video_names_map[vid] = Path(s3_key).stem if s3_key else vid[:8]
    for vid, rallies in all_rallies.items():
        vname = video_names_map.get(vid, vid[:8])
        vid_totals[vname] = len(rallies)

    print(f"{'video':<20s}  {'errors':>6s}  {'total':>5s}  {'accuracy':>8s}")
    print("-" * 45)
    for vname in sorted(vid_totals.keys()):
        n_err = vid_counts.get(vname, 0)
        n_tot = vid_totals[vname]
        acc = (n_tot - n_err) / n_tot * 100 if n_tot > 0 else 100.0
        if n_err > 0:
            print(f"{vname:<20s}  {n_err:>6d}  {n_tot:>5d}  {acc:>7.1f}%")

    print(f"\nExtracted {ok_count}/{len(errors)} clips to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
