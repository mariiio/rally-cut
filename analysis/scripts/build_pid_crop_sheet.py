"""Per-rally PID crop sheet — visual debugging for cross-rally identity.

Renders a single HTML file: rows = rallies, columns = PID 1/2/3/4. Each
cell shows an actual frame crop of the player MatchSolver assigned that
PID in that rally. Mis-assignments show up immediately as visual
mismatches across a column.

Usage:
    uv run python scripts/build_pid_crop_sheet.py <video_id>
    uv run python scripts/build_pid_crop_sheet.py 5c756c41-1cc1-4486-a95c-97398912cfbe

Output: analysis/reports/visual_debug/<short_id>_pid_sheet.html
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_video_path,
    load_rallies_for_video,
)


def _frame_at(cap: cv2.VideoCapture, ms: int) -> cv2.Mat | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, ms)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _crop_for_pid(
    cap: cv2.VideoCapture,
    rally_start_ms: int,
    fps: float,
    fw: int,
    fh: int,
    positions_for_pid: list[dict],
    target_progress: float,
) -> bytes | None:
    """Extract a single bbox crop for this PID at ~target_progress through the rally.

    target_progress: 0.0 = start, 0.5 = midpoint, 1.0 = end of the PID's
    visible frames. We pick a frame near that progress where the bbox is
    confident and well-sized.
    """
    if not positions_for_pid:
        return None
    sorted_pos = sorted(positions_for_pid, key=lambda p: p.get("frameNumber", 0))
    target_idx = max(0, min(len(sorted_pos) - 1, int(len(sorted_pos) * target_progress)))
    # Try a window around the target idx, preferring high-confidence + reasonable bbox
    best: dict | None = None
    for offset in range(0, 30):
        for sign in (1, -1):
            i = target_idx + offset * sign
            if not (0 <= i < len(sorted_pos)):
                continue
            p = sorted_pos[i]
            cx, cy = float(p.get("x", 0)), float(p.get("y", 0))
            w, h = float(p.get("width", 0)), float(p.get("height", 0))
            conf = float(p.get("confidence", 0))
            if w <= 0.02 or h <= 0.04:  # too small
                continue
            if conf < 0.3:
                continue
            best = p
            break
        if best:
            break
    if best is None:
        best = sorted_pos[target_idx]
    cx, cy = float(best.get("x", 0)), float(best.get("y", 0))
    w, h = float(best.get("width", 0)), float(best.get("height", 0))
    frame_no = int(best.get("frameNumber", 0))
    ms = rally_start_ms + (frame_no * 1000 / fps)
    frame = _frame_at(cap, int(ms))
    if frame is None:
        return None
    x1 = max(0, int((cx - w / 2) * fw))
    y1 = max(0, int((cy - h / 2) * fh))
    x2 = min(fw, int((cx + w / 2) * fw))
    y2 = min(fh, int((cy + h / 2) * fh))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    # Resize to a uniform height so columns align in the HTML
    target_h = 160
    aspect = crop.shape[1] / max(1, crop.shape[0])
    target_w = max(40, int(target_h * aspect))
    crop = cv2.resize(crop, (target_w, target_h))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")  # type: ignore[return-value]


def _b64_data_uri(b64_or_none: bytes | None) -> str:
    if b64_or_none is None:
        return ""
    return f"data:image/jpeg;base64,{b64_or_none}"


def build_sheet(video_id: str, output_path: Path) -> None:
    rallies = load_rallies_for_video(video_id)
    print(f"Loaded {len(rallies)} rallies for {video_id[:8]}")

    video_path = get_video_path(video_id)
    if video_path is None:
        sys.exit("video path not found")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"failed to open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Fetch positions_json per rally (post-remap; trackId == pid).
    rally_positions: dict[str, list[dict]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT r.id, pt.positions_json FROM rallies r
                   JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.video_id = %s""",
                [video_id],
            )
            for rid, pos in cur.fetchall():
                if pos is None:
                    continue
                if isinstance(pos, str):
                    import json
                    pos = json.loads(pos)
                rally_positions[str(rid)] = pos

    rows_html: list[str] = []
    for ri, rally in enumerate(rallies):
        pos = rally_positions.get(rally.rally_id) or []
        by_pid: dict[int, list[dict]] = {}
        for p in pos:
            pid = p.get("trackId")
            if pid in (1, 2, 3, 4):
                by_pid.setdefault(pid, []).append(p)
        # Three crops per PID — start, mid, end of that PID's frames
        cells_html: list[str] = []
        for pid in (1, 2, 3, 4):
            pid_positions = by_pid.get(pid, [])
            if not pid_positions:
                cells_html.append('<td class="empty">—</td>')
                continue
            crops = []
            for progress in (0.1, 0.5, 0.9):
                b64 = _crop_for_pid(
                    cap, rally.start_ms, fps, fw, fh, pid_positions, progress,
                )
                crops.append(_b64_data_uri(b64))
            n_frames = len(pid_positions)
            crops_html = "".join(
                f'<img src="{c}" />' if c else '<span class="missing">?</span>'
                for c in crops
            )
            cells_html.append(
                f'<td>{crops_html}<div class="meta">{n_frames}f</div></td>'
            )
        rows_html.append(
            f"<tr><th>r{ri:02d}<br/><small>{rally.rally_id[:8]}</small></th>"
            + "".join(cells_html)
            + "</tr>"
        )

    cap.release()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PID crop sheet — {video_id[:8]}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #1a1a1a; color: #ddd; margin: 0; padding: 16px; }}
  h1 {{ font-size: 14px; margin: 0 0 12px; }}
  .legend {{ font-size: 12px; color: #888; margin-bottom: 12px; }}
  table {{ border-collapse: collapse; }}
  th {{ background: #2a2a2a; padding: 8px; text-align: center; font-size: 11px; }}
  th small {{ font-weight: normal; color: #888; }}
  td {{ background: #252525; padding: 4px; vertical-align: top; border: 1px solid #333; }}
  td.empty {{ background: #1a1a1a; color: #666; text-align: center; padding: 20px; }}
  td img {{ height: 100px; margin: 0 1px; vertical-align: top; border-radius: 2px; }}
  td .meta {{ font-size: 9px; color: #666; text-align: center; margin-top: 2px; }}
  .missing {{ display: inline-block; width: 50px; height: 100px; background: #333; color: #888; text-align: center; line-height: 100px; }}
  .pid1 {{ background: #1e3a5f !important; }}
  .pid2 {{ background: #5a3a1e !important; }}
  .pid3 {{ background: #3a5a1e !important; }}
  .pid4 {{ background: #5a1e3a !important; }}
</style></head>
<body>
<h1>PID crop sheet — {video_id} ({len(rallies)} rallies)</h1>
<div class="legend">
  Each cell shows three frames (start/mid/end) of the bbox MatchSolver
  assigned the listed PID in that rally. Visually mismatched columns =
  cross-rally identity errors. Same-shirt-color column = correct
  assignment. <code>n f</code> = total frames assigned.
</div>
<table>
<tr><th></th>
  <th class="pid1">PID 1</th>
  <th class="pid2">PID 2</th>
  <th class="pid3">PID 3</th>
  <th class="pid4">PID 4</th>
</tr>
{"".join(rows_html)}
</table>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"wrote {output_path} ({len(html) // 1024} KB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video_id")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    short = args.video_id[:8]
    out = args.out or Path(__file__).resolve().parents[1] / "reports" / "visual_debug" / f"{short}_pid_sheet.html"
    build_sheet(args.video_id, out)


if __name__ == "__main__":
    main()
