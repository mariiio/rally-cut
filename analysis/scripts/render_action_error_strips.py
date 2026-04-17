"""Render diagnostic video clips for each action detection error.

For each error in corpus_annotated.jsonl, produces a short MP4 clip (~2-3 seconds)
centered on the GT contact frame with overlays:
- Ball trajectory trail (yellow, fading)
- All player bboxes colored by track ID
- GT player highlighted green, predicted player red
- Contact frame flash / marker
- Text HUD: action, error class, FN category, confidence
- Net line

Usage:
    cd analysis
    uv run python scripts/render_action_error_strips.py
    uv run python scripts/render_action_error_strips.py --max-errors 50
    uv run python scripts/render_action_error_strips.py --rally <id>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.contact_detector import _CONFIDENCE_THRESHOLD

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "action_errors"
CLIPS_DIR = OUTPUT_DIR / "clips"
CORPUS_PATH = OUTPUT_DIR / "corpus_annotated.jsonl"

WINDOW_BEFORE = 30
WINDOW_AFTER = 15
CLIP_FPS = 15
CLIP_W = 960
CLIP_H = 540

TRACK_COLORS = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
    (100, 255, 255),
    (200, 200, 200),
    (180, 130, 70),
]


def get_track_color(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def load_corpus(path: Path, rally_filter: str | None = None) -> list[dict]:
    errors = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rally_filter and rec["rally_id"] != rally_filter:
                continue
            errors.append(rec)
    return errors


def load_rally_data(rally_ids: set[str]) -> dict[str, dict]:
    if not rally_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(rally_ids))
    query = f"""
        SELECT r.id, r.video_id, r.start_ms,
               pt.ball_positions_json, pt.positions_json,
               pt.fps, pt.frame_count, pt.court_split_y
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.id IN ({placeholders})
    """
    result = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(rally_ids))
            for row in cur.fetchall():
                (rally_id, video_id, start_ms,
                 ball_json, positions_json, fps, frame_count, court_split_y) = row
                result[rally_id] = {
                    "video_id": video_id,
                    "start_ms": start_ms or 0,
                    "ball_positions": ball_json or [],
                    "positions": positions_json or [],
                    "fps": fps or 30.0,
                    "frame_count": frame_count or 0,
                    "court_split_y": court_split_y,
                }

    # Load trackToPlayer mappings and invert them (player_id → raw_track_id)
    video_ids = {d["video_id"] for d in result.values()}
    if video_ids:
        vid_placeholders = ", ".join(["%s"] * len(video_ids))
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, match_analysis_json FROM videos WHERE id IN ({vid_placeholders})",
                    list(video_ids),
                )
                for vid, ma_json in cur.fetchall():
                    if not ma_json:
                        continue
                    for entry in ma_json.get("rallies", []):
                        rid = entry.get("rallyId", entry.get("rally_id", ""))
                        t2p = entry.get("trackToPlayer", entry.get("track_to_player", {}))
                        if rid in result and t2p:
                            # Invert: player_id → raw_track_id
                            result[rid]["player_to_track"] = {
                                int(v): int(k) for k, v in t2p.items()
                            }

    return result


def draw_hud(
    frame: np.ndarray,
    error: dict,
    current_rally_frame: int,
    contact_frame: int,
    is_contact: bool,
    frame_w: int,
    frame_h: int,
) -> None:
    """Draw the text HUD overlay on the frame."""
    ec = error.get("error_class", "?")
    fn_cat = error.get("fn_subcategory") or ""
    gt_a = error.get("gt_action", "?")
    pred_a = error.get("pred_action") or "MISS"
    cls_conf = error.get("classifier_conf", 0.0)
    ball_cov = error.get("rally_quality", {}).get("ball_coverage_pct", "?")

    hud_bg = np.zeros((80, frame_w, 3), dtype=np.uint8)
    hud_bg[:] = (30, 30, 30)

    line1 = f"GT: {gt_a}  Pred: {pred_a}  |  {ec}"
    if fn_cat:
        line1 += f"  ({fn_cat})"

    frames_to_contact = current_rally_frame - contact_frame
    if frames_to_contact < 0:
        timing = f"contact in {-frames_to_contact}f"
    elif frames_to_contact == 0:
        timing = ">> CONTACT <<"
    else:
        timing = f"contact +{frames_to_contact}f"

    line2 = f"f:{current_rally_frame}  {timing}  |  conf:{cls_conf:.3f}  ball_cov:{ball_cov}%"

    color1 = (0, 255, 255) if is_contact else (220, 220, 220)
    cv2.putText(hud_bg, line1, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color1, 1, cv2.LINE_AA)
    cv2.putText(hud_bg, line2, (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    gt_pid = error.get("gt_player_track_id", -1)
    pred_pid = error.get("pred_player_track_id")
    player_info = f"GT:P{gt_pid}"
    if pred_pid is not None:
        player_info += f"  Pred:P{pred_pid}"
    cv2.putText(hud_bg, player_info, (frame_w - 250, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1, cv2.LINE_AA)

    frame[:80] = cv2.addWeighted(frame[:80], 0.4, hud_bg, 0.6, 0)


def render_clip(
    cap: cv2.VideoCapture,
    src_fps: float,
    frame_width: int,
    frame_height: int,
    rally_start_frame: int,
    contact_frame: int,
    ball_positions: list[dict],
    player_positions: list[dict],
    court_split_y: float | None,
    error: dict,
    output_path: Path,
    player_to_track: dict[int, int] | None = None,
) -> bool:
    """Render a diagnostic video clip for one error."""
    ball_by_frame: dict[int, dict] = {}
    for bp in ball_positions:
        if bp.get("confidence", 1.0) >= _CONFIDENCE_THRESHOLD:
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0:
                ball_by_frame[bp["frameNumber"]] = bp

    pos_by_frame: dict[int, list[dict]] = defaultdict(list)
    for pp in player_positions:
        fn = pp.get("frameNumber", pp.get("frame_number", 0))
        pos_by_frame[fn].append(pp)

    # GT stores player IDs (1-4), positions use raw track IDs.
    # Map GT player ID → raw track ID for correct bbox overlay.
    p2t = player_to_track or {}
    gt_player_id = error.get("gt_player_track_id", -1)
    gt_tid = p2t.get(gt_player_id, gt_player_id)
    pred_player_id = error.get("pred_player_track_id")
    pred_tid = p2t.get(pred_player_id, pred_player_id) if pred_player_id is not None else None

    start_frame = max(0, contact_frame - WINDOW_BEFORE)
    end_frame = contact_frame + WINDOW_AFTER

    # Subsample: take every Nth frame to hit target clip FPS
    step = max(1, round(src_fps / CLIP_FPS))

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    tmp_path = output_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, CLIP_FPS, (CLIP_W, CLIP_H))
    if not writer.isOpened():
        return False

    frames_written = 0

    for rally_frame in range(start_frame, end_frame + 1, step):
        abs_frame = rally_start_frame + rally_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        is_contact = abs(rally_frame - contact_frame) <= 1

        # Ball trail (last 20 frames)
        trail_frames = sorted(
            f for f in ball_by_frame
            if rally_frame - 20 <= f <= rally_frame
        )
        if len(trail_frames) >= 2:
            pts = []
            for bf in trail_frames:
                bp = ball_by_frame[bf]
                px = int(bp["x"] * frame_width)
                py = int(bp["y"] * frame_height)
                pts.append((px, py))
            for j in range(1, len(pts)):
                alpha = j / len(pts)
                thickness = max(1, int(3 * alpha))
                color = (0, int(200 * alpha), int(255 * alpha))
                cv2.line(frame, pts[j - 1], pts[j], color, thickness, cv2.LINE_AA)

        # Ball at current frame
        ball_at = ball_by_frame.get(rally_frame)
        if ball_at:
            bx = int(ball_at["x"] * frame_width)
            by = int(ball_at["y"] * frame_height)
            radius = 14 if is_contact else 8
            cv2.circle(frame, (bx, by), radius, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # Player bboxes
        players = pos_by_frame.get(rally_frame, [])
        for pp in players:
            tid = pp.get("trackId", pp.get("track_id", 0))
            px = pp.get("x", 0.0)
            py_pos = pp.get("y", 0.0)
            pw = pp.get("width", 0.0)
            ph = pp.get("height", 0.0)

            x1 = int((px - pw / 2) * frame_width)
            y1 = int((py_pos - ph / 2) * frame_height)
            x2 = int((px + pw / 2) * frame_width)
            y2 = int((py_pos + ph / 2) * frame_height)

            if tid == gt_tid:
                color = (0, 255, 0)
                thickness = 3
                label = f"GT:T{tid}"
            elif pred_tid is not None and tid == pred_tid:
                color = (0, 0, 255)
                thickness = 3
                label = f"Pred:T{tid}"
            else:
                color = get_track_color(tid)
                thickness = 1
                label = f"T{tid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Court split line
        if court_split_y is not None:
            net_y_px = int(court_split_y * frame_height)
            cv2.line(frame, (0, net_y_px), (frame_width, net_y_px),
                     (128, 128, 0), 1, cv2.LINE_AA)

        # Contact frame flash
        if is_contact:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, frame_height),
                          (0, 255, 255), 6)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # HUD
        draw_hud(frame, error, rally_frame, contact_frame, is_contact,
                 frame_width, frame_height)

        resized = cv2.resize(frame, (CLIP_W, CLIP_H))
        writer.write(resized)
        frames_written += 1

    writer.release()

    if frames_written == 0:
        tmp_path.unlink(missing_ok=True)
        return False

    # Re-encode with ffmpeg for browser compatibility (H.264)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(tmp_path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "28",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ],
            capture_output=True,
            timeout=30,
        )
        tmp_path.unlink(missing_ok=True)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # If ffmpeg fails, keep the raw mp4v file
        tmp_path.rename(output_path)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Render action error diagnostic clips")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=CLIPS_DIR)
    parser.add_argument("--max-errors", type=int, default=0, help="0 = all")
    parser.add_argument("--rally", type=str, help="Filter to specific rally")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Corpus not found at {args.corpus}")
        print("Run build_action_error_corpus.py first.")
        sys.exit(1)

    errors = load_corpus(args.corpus, args.rally)
    if args.max_errors > 0:
        errors = errors[:args.max_errors]
    print(f"Loaded {len(errors)} errors to render")

    if not errors:
        print("No errors to render.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rally_ids = {e["rally_id"] for e in errors}
    print(f"Loading data for {len(rally_ids)} rallies...")
    rally_data = load_rally_data(rally_ids)

    errors_by_video: dict[str, list[dict]] = defaultdict(list)
    for e in errors:
        errors_by_video[e["video_id"]].append(e)

    rendered = 0
    skipped = 0

    for video_id, video_errors in errors_by_video.items():
        video_path = get_video_path(video_id)
        if video_path is None:
            print(f"  Video {video_id[:8]} not available — skipping {len(video_errors)} errors")
            skipped += len(video_errors)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Cannot open {video_path} — skipping {len(video_errors)} errors")
            skipped += len(video_errors)
            continue

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        for e in video_errors:
            rid = e["rally_id"]
            rd = rally_data.get(rid)
            if rd is None:
                skipped += 1
                continue

            rally_start_frame = int(rd["start_ms"] / 1000.0 * vid_fps)
            gt_frame = e["gt_frame"]

            out_path = args.output_dir / f"{rid}_{gt_frame}.mp4"

            ok = render_clip(
                cap=cap,
                src_fps=vid_fps,
                frame_width=fw,
                frame_height=fh,
                rally_start_frame=rally_start_frame,
                contact_frame=gt_frame,
                ball_positions=rd["ball_positions"],
                player_positions=rd["positions"],
                court_split_y=rd.get("court_split_y"),
                error=e,
                output_path=out_path,
                player_to_track=rd.get("player_to_track"),
            )

            if ok:
                rendered += 1
                if rendered % 10 == 0:
                    print(f"  [{rendered}/{len(errors)}] clips rendered...")
            else:
                skipped += 1

        cap.release()

    print(f"\nDone: {rendered} clips rendered, {skipped} skipped")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
