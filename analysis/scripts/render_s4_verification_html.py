"""Render an annotated-frame verification page for the 28 S4 labeling picks.

For each pick, draws on the contact frame:
  - All same-team candidate bboxes (gray)
  - Pipeline pick halo (cyan)
  - S4 proposed pick halo (green)
  - Previous-toucher (if it sits in the contact's candidates) halo (red)
  - Ball trajectory polyline over the K=10 pre-window frames (yellow)
  - Ball position at contact frame (yellow circle)

Outputs:
  analysis/reports/probe_b_sequence_aware/2026_05_14/s4_verification_frames/<i>_<vid>_<rally>_f<frame>.jpg
  analysis/reports/probe_b_sequence_aware/2026_05_14/s4_verification.html

Usage:
    cd analysis
    uv run python scripts/render_s4_verification_html.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from rallycut.evaluation.tracking.db import get_connection
from rallycut.evaluation.video_resolver import VideoResolver

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE.parent / "reports" / "probe_b_sequence_aware" / "2026_05_14"
FRAMES_DIR = REPORT_DIR / "s4_verification_frames"
DEFAULT_FLIPS = REPORT_DIR / "s4_fleet_candidates.json"
DEFAULT_CHECKLIST = REPORT_DIR / "s4_labeling_checklist.md"
DEFAULT_HTML = REPORT_DIR / "s4_verification.html"

K_PRE = 10

# Colors (BGR)
COLOR_PIPELINE = (255, 200, 60)    # cyan-ish
COLOR_S4 = (60, 220, 60)            # green
COLOR_PREV = (60, 60, 255)          # red
COLOR_OTHER = (160, 160, 160)       # gray
COLOR_BALL = (0, 255, 255)
COLOR_TRAJ = (80, 200, 255)


@dataclass
class PickRow:
    idx: int
    bucket: str
    flip: dict[str, Any]
    rally_uuid: str
    source_time: str
    rally_order: int


def _parse_checklist_uuids(md_path: Path) -> list[tuple[int, str, str, int]]:
    """Returns [(idx, bucket, rally_uuid, frame), ...] from the checklist md."""
    rows: list[tuple[int, str, str, int]] = []
    for line in md_path.read_text().splitlines():
        if not line.startswith("| "):
            continue
        # Skip separator row
        if "--:" in line or ":---" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        # Header row contains "Bucket"
        if "Bucket" in parts:
            continue
        # Schema (14 cells with empty boundaries):
        #   '' | idx | bucket | video | '#order' | '**time**' | frame |
        #   action | prev | pl_pick | s4_pick | `uuid` | verdict | ''
        try:
            idx = int(parts[1])
        except (ValueError, IndexError):
            continue
        if len(parts) < 12:
            continue
        bucket = parts[2]
        frame_str = parts[6]
        try:
            frame = int(frame_str)
        except ValueError:
            continue
        uuid_part = parts[11].strip("`")
        rows.append((idx, bucket, uuid_part, frame))
    return rows


def _fetch_video_meta(video_id: str) -> dict[str, Any]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, name, filename, fps, width, height,
                      s3_key, proxy_s3_key, processed_s3_key, content_hash
               FROM videos WHERE id = %s""",
            (video_id,),
        )
        r = cur.fetchone()
    if not r:
        return {}
    return {
        "id": r[0], "name": r[1], "filename": r[2],
        "fps": float(r[3]) if r[3] is not None else 30.0,
        "width": int(r[4] or 0), "height": int(r[5] or 0),
        "s3_key": r[6], "proxy_s3_key": r[7],
        "processed_s3_key": r[8], "content_hash": r[9],
    }


def _resolve_video(resolver: VideoResolver, vm: dict[str, Any]) -> Path | None:
    for key_name in ("proxy_s3_key", "s3_key", "processed_s3_key"):
        sk = vm.get(key_name)
        if not sk:
            continue
        try:
            return resolver.resolve(sk, vm["content_hash"])
        except Exception:
            continue
    return None


def _fetch_pre_ball_and_positions(
    rally_id: str, pl_frame: int,
) -> tuple[
    list[tuple[int, float, float]],
    dict[int, tuple[float, float]],
    dict[int, tuple[float, float, float, float]],
    tuple[float, float],
]:
    """Returns (pre_ball, candidate_at_contact, candidate_bboxes_at_contact, ball_at_contact)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ball_positions_json, positions_json, actions_json, contacts_json "
            "FROM player_tracks WHERE rally_id = %s",
            (rally_id,),
        )
        r = cur.fetchone()
    if not r:
        return [], {}, {}, (0.0, 0.0)
    ball_json, positions_json, actions_json, contacts_json = r
    ball_by_frame: dict[int, tuple[float, float]] = {}
    for bp in ball_json or []:
        try:
            f = int(bp["frameNumber"])
            bx = float(bp.get("x", 0.0)); by = float(bp.get("y", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if bx <= 0 and by <= 0:
            continue
        ball_by_frame[f] = (bx, by)
    pre_ball: list[tuple[int, float, float]] = []
    for f in range(pl_frame - K_PRE, pl_frame):
        if f in ball_by_frame:
            pre_ball.append((f, ball_by_frame[f][0], ball_by_frame[f][1]))
    ball_at = ball_by_frame.get(pl_frame, (0.0, 0.0))
    # Ball at contact may not be present (interpolation); use closest if missing
    if ball_at == (0.0, 0.0) and pre_ball:
        ball_at = (pre_ball[-1][1], pre_ball[-1][2])

    cand_pos: dict[int, tuple[float, float]] = {}
    cand_bbox: dict[int, tuple[float, float, float, float]] = {}
    for p in positions_json or []:
        try:
            f = int(p.get("frameNumber", -1))
            tid = int(p.get("trackId", -1))
        except (TypeError, ValueError):
            continue
        if f != pl_frame:
            continue
        cand_pos[tid] = (float(p.get("x", 0.0)), float(p.get("y", 0.0)))
        cand_bbox[tid] = (
            float(p.get("x", 0.0)),
            float(p.get("y", 0.0)),
            float(p.get("width", 0.0)),
            float(p.get("height", 0.0)),
        )
    return pre_ball, cand_pos, cand_bbox, ball_at


def _annotate(
    *,
    out_path: Path,
    idx: int,
    flip: dict[str, Any],
    video_path: Path,
    rally_start_ms: int,
    fps: float,
) -> Path | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rally_start_frame = int(round(rally_start_ms / 1000.0 * fps))
    source_frame = rally_start_frame + flip["pl_frame"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    img = frame.copy()

    pre_ball, _cand_pos, cand_bbox, ball_at = _fetch_pre_ball_and_positions(
        flip["rally_id"], flip["pl_frame"],
    )

    pipeline_pid = flip["pipeline_pid"]
    s4_pid = flip["s4_pid"]
    prev_pid = flip.get("prev_toucher_pid")

    # Draw all candidate bboxes (gray base).
    for tid, bb in cand_bbox.items():
        cx, cy, bw, bh = bb
        if bw <= 0 or bh <= 0:
            continue
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OTHER, 1)

        # Halos (stacked, larger ring outermost)
        offset = -4
        halos = []
        if tid == pipeline_pid:
            halos.append(("PL", COLOR_PIPELINE))
        if tid == s4_pid:
            halos.append(("S4", COLOR_S4))
        if prev_pid is not None and tid == prev_pid:
            halos.append(("PREV", COLOR_PREV))
        for _name, color in halos:
            cv2.rectangle(img, (x1 + offset, y1 + offset),
                          (x2 - offset, y2 - offset), color, 2)
            offset -= 3

        # Label
        label_parts = [f"p{tid}"]
        for n, _ in halos:
            label_parts.append(n)
        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(y1 - 6, 14)
        cv2.rectangle(img, (x1 - 2, ly - th - 4), (x1 + tw + 4, ly + 2),
                      (0, 0, 0), -1)
        cv2.putText(img, label, (x1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw pre-ball trajectory polyline.
    pre_pts = [(int(x * w), int(y * h)) for _f, x, y in pre_ball]
    for i in range(1, len(pre_pts)):
        cv2.line(img, pre_pts[i - 1], pre_pts[i], COLOR_TRAJ, 2, cv2.LINE_AA)
    for p in pre_pts:
        cv2.circle(img, p, 3, COLOR_TRAJ, -1)
    # Arrow from second-to-last to last (or first to last)
    if len(pre_pts) >= 2:
        cv2.arrowedLine(img, pre_pts[-2], pre_pts[-1], COLOR_TRAJ,
                        2, cv2.LINE_AA, tipLength=0.4)

    # Ball at contact frame.
    bx_n, by_n = ball_at
    if bx_n > 0 or by_n > 0:
        bx, by = int(bx_n * w), int(by_n * h)
        cv2.circle(img, (bx, by), 14, COLOR_BALL, 3)
        cv2.circle(img, (bx, by), 2, COLOR_BALL, -1)

    # Banner.
    banner_h = 70
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    title = (f"#{idx} {flip['video_name']}/{flip['rally_short']} "
             f"{flip['action_type']} f={flip['pl_frame']}")
    sub1 = (f"PL=p{pipeline_pid}({flip['pipeline_team']})  "
            f"S4=p{s4_pid}({flip['s4_team']})  "
            f"PREV=p{prev_pid}({flip.get('prev_toucher_team')})  "
            f"prev_action={flip['prev_action_type']}@f{flip['prev_action_frame']}")
    sub2 = f"n_pre_ball={flip['n_pre_ball']}  cands={flip['same_team_cands']}"
    cv2.putText(img, title, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub1, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, sub2, (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return out_path


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>S4 verification — 2026-05-14</title>
<style>
  body { background:#0f0f10; color:#eaeaea; font-family:-apple-system,sans-serif; margin:0; padding:24px; }
  h1 { margin:0 0 8px 0; }
  .legend { background:#1a1a1c; border:1px solid #2a2a2e; border-radius:8px; padding:10px 18px; margin-bottom:18px; font-size:13px; }
  .legend span.swatch { display:inline-block; width:12px; height:12px; vertical-align:middle; margin-right:4px; border-radius:2px; }
  .case { background:#1a1a1c; border:1px solid #2a2a2e; border-radius:8px; padding:14px 18px; margin-bottom:14px; }
  .case h3 { margin:0 0 8px 0; font-size:16px; }
  .case img { max-width:100%; max-height:75vh; display:block; margin:10px 0; cursor:zoom-in; background:#000; }
  .meta { color:#aaa; font-size:12px; }
  .bucket { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; background:#222; color:#ddd; }
  #zoom-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.94); z-index:99; align-items:center; justify-content:center; cursor:zoom-out; }
  #zoom-overlay.open { display:flex; }
  #zoom-overlay img { max-width:96vw; max-height:96vh; }
</style></head><body>
<h1>S4 verification — 28 fleet picks (2026-05-14)</h1>
<div class="legend">
  <span class="swatch" style="background:rgb(60,200,255)"></span>Pipeline pick (cyan)
  &nbsp;&nbsp;<span class="swatch" style="background:rgb(60,220,60)"></span>S4 proposed (green)
  &nbsp;&nbsp;<span class="swatch" style="background:rgb(255,60,60)"></span>Previous toucher (red)
  &nbsp;&nbsp;<span class="swatch" style="background:rgb(255,255,0)"></span>Ball + trajectory arrow
</div>
<div id="cases"></div>
<div id="zoom-overlay"><img id="zoom-img"></div>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
function caseHtml(r) {
  return `
    <div class="case">
      <h3>#${r.idx} <span class="bucket">${r.bucket}</span> ${r.video_name}/${r.rally_short} ${r.action_type} f=${r.pl_frame}</h3>
      <div class="meta">source time: ${r.source_time} | rally order: #${r.rally_order} | uuid: ${r.rally_id}</div>
      <div class="meta">PL=p${r.pipeline_pid}(${r.pipeline_team}) → S4=p${r.s4_pid}(${r.s4_team}) | prev: f${r.prev_action_frame} ${r.prev_action_type} p${r.prev_toucher_pid}(${r.prev_toucher_team})</div>
      <img src="${r.img}" onclick="zoom(this.src)">
    </div>`;
}
document.getElementById('cases').innerHTML = DATA.map(caseHtml).join('');
function zoom(s){const o=document.getElementById('zoom-overlay');document.getElementById('zoom-img').src=s;o.classList.add('open');}
document.getElementById('zoom-overlay').addEventListener('click', () => document.getElementById('zoom-overlay').classList.remove('open'));
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render S4 verification HTML")
    parser.add_argument("--flips", type=str, default=str(DEFAULT_FLIPS))
    parser.add_argument("--checklist", type=str, default=str(DEFAULT_CHECKLIST))
    parser.add_argument("--html", type=str, default=str(DEFAULT_HTML))
    parser.add_argument("--limit", type=int, default=0,
                        help="Render at most N picks (0 = all 28)")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.flips).read_text())
    flips = data["flips"]
    # Index by (rally_id, pl_frame) for O(1) lookup
    flip_by_key: dict[tuple[str, int], dict[str, Any]] = {
        (f["rally_id"], f["pl_frame"]): f for f in flips
    }

    rows = _parse_checklist_uuids(Path(args.checklist))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Rendering {len(rows)} S4 verification frames...")

    resolver = VideoResolver()
    video_meta_cache: dict[str, dict[str, Any]] = {}
    video_path_cache: dict[str, Path | None] = {}
    payload: list[dict[str, Any]] = []

    for idx, bucket, uuid, frame in rows:
        flip = flip_by_key.get((uuid, frame))
        if not flip:
            print(f"  [{idx}] miss: ({uuid[:8]}, {frame})")
            continue
        vid = flip["video_id"]
        if vid not in video_meta_cache:
            video_meta_cache[vid] = _fetch_video_meta(vid)
            video_path_cache[vid] = _resolve_video(resolver, video_meta_cache[vid]) \
                if video_meta_cache[vid] else None
        vm = video_meta_cache[vid]
        vpath = video_path_cache[vid]
        if not vm or not vpath:
            print(f"  [{idx}] no video for {flip['video_name']}/{flip['rally_short']}")
            continue

        out_path = FRAMES_DIR / (
            f"{idx:02d}_{flip['video_name']}_{flip['rally_short']}_f{flip['pl_frame']}.jpg"
        )
        annotated = _annotate(
            out_path=out_path,
            idx=idx,
            flip=flip,
            video_path=vpath,
            rally_start_ms=flip["rally_start_ms"],
            fps=vm["fps"],
        )
        if not annotated:
            print(f"  [{idx}] failed to annotate")
            continue

        rally_start_s = flip["rally_start_ms"] / 1000.0
        frame_offset_s = flip["pl_frame"] / max(flip["fps"], 1e-6)
        total_s = rally_start_s + frame_offset_s
        m = int(total_s // 60)
        s = total_s - m * 60
        source_time = f"{m}:{s:06.3f}"

        payload.append({
            "idx": idx,
            "bucket": bucket,
            "video_name": flip["video_name"],
            "rally_short": flip["rally_short"],
            "rally_id": flip["rally_id"],
            "rally_order": flip["rally_order"],
            "source_time": source_time,
            "action_type": flip["action_type"],
            "pl_frame": flip["pl_frame"],
            "pipeline_pid": flip["pipeline_pid"],
            "pipeline_team": flip["pipeline_team"],
            "s4_pid": flip["s4_pid"],
            "s4_team": flip["s4_team"],
            "prev_action_frame": flip["prev_action_frame"],
            "prev_action_type": flip["prev_action_type"],
            "prev_toucher_pid": flip["prev_toucher_pid"],
            "prev_toucher_team": flip["prev_toucher_team"],
            "img": str(annotated.relative_to(REPORT_DIR)),
        })
        print(f"  [{idx}] wrote: {annotated.name}")

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, indent=2))
    Path(args.html).write_text(html)
    print()
    print(f"Wrote HTML: {args.html}")
    print(f"Open: open {args.html}")


if __name__ == "__main__":
    main()
