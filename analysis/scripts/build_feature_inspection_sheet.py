"""Visualize what `extract_appearance_features` actually feeds the matcher.

For each rally × each track in `top_tracks`, render 3 versions of the
same crop side-by-side:

  1. Raw bbox crop (what the camera sees).
  2. Region overlay: head (red 0-15%), upper body (cyan 20-55%), lower
     body (green 50-78%). Shows the FIXED bbox-fraction regions before
     clothing-mask filtering.
  3. Clothing-mask overlay: pixels DROPPED by `_build_clothing_mask`
     (sand-warm-hue, skin, central-width margin) shown semi-transparent
     red. The pixels that REMAIN are what go into the HSV histograms.

If the green/cyan regions in (2) cover sand or other players, the bbox-
fraction approach is wrong. If (3) drops the actual shirt/shorts
pixels (e.g., shirt color falls in the sand-hue range), the
clothing_mask is too aggressive.

Usage:
    uv run python scripts/build_feature_inspection_sheet.py <video_id>
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_video_path,
    load_rallies_for_video,
)
from rallycut.tracking.player_features import _build_clothing_mask


def _b64_from_bgr(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


def _resize_h(img: np.ndarray, target_h: int = 200) -> np.ndarray:
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img
    aspect = img.shape[1] / img.shape[0]
    target_w = max(40, int(target_h * aspect))
    return cv2.resize(img, (target_w, target_h))


def _get_pid_track_for_rally(positions: list[dict]) -> dict[int, dict]:
    """Pick a representative position per pid (mid-rally, high confidence)."""
    by_pid: dict[int, list[dict]] = {}
    for p in positions:
        tid = p.get("trackId")
        if tid in (1, 2, 3, 4):
            by_pid.setdefault(tid, []).append(p)
    out: dict[int, dict] = {}
    for pid, pts in by_pid.items():
        pts_sorted = sorted(pts, key=lambda q: q.get("frameNumber", 0))
        # Pick mid-rally position with highest confidence among nearby frames
        target = pts_sorted[len(pts_sorted) // 2]
        # Try a window of 10 frames around mid for better confidence
        window = pts_sorted[max(0, len(pts_sorted) // 2 - 5): len(pts_sorted) // 2 + 5]
        if window:
            target = max(window, key=lambda q: q.get("confidence", 0))
        out[pid] = target
    return out


def _draw_regions(crop: np.ndarray) -> np.ndarray:
    """Overlay the bbox-fraction regions matching extract_appearance_features."""
    h = crop.shape[0]
    out = crop.copy()
    overlay = out.copy()
    # head: 0-15% (red)
    cv2.rectangle(overlay, (0, 0), (out.shape[1], int(h * 0.15)),
                  (0, 0, 200), -1)
    # upper body: 20-55% (cyan)
    cv2.rectangle(overlay, (0, int(h * 0.20)), (out.shape[1], int(h * 0.55)),
                  (200, 200, 0), -1)
    # lower body: 50-78% (green)
    cv2.rectangle(overlay, (0, int(h * 0.50)), (out.shape[1], int(h * 0.78)),
                  (0, 200, 0), -1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    # Border lines for clarity
    for frac, color in [(0.15, (0, 0, 255)), (0.20, (255, 200, 0)),
                        (0.55, (255, 200, 0)), (0.50, (0, 255, 0)),
                        (0.78, (0, 255, 0))]:
        y = int(h * frac)
        cv2.line(out, (0, y), (out.shape[1], y), color, 1)
    return out


def _draw_clothing_mask(crop: np.ndarray) -> np.ndarray:
    """Overlay clothing_mask: dropped pixels in semi-red."""
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = _build_clothing_mask(hsv)  # 255 = kept, 0 = dropped
    dropped = cv2.bitwise_not(mask)
    overlay = crop.copy()
    overlay[dropped > 0] = (0, 0, 255)  # red over dropped pixels
    out = crop.copy()
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
    # Add a textual indicator of how much was kept
    kept_pct = float(mask.sum()) / max(1.0, mask.size * 255) * 100
    cv2.putText(out, f"{kept_pct:.0f}%", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return out


def build_sheet(video_id: str, output_path: Path) -> None:
    rallies = load_rallies_for_video(video_id)
    print(f"Loaded {len(rallies)} rallies for {video_id[:8]}")

    video_path = get_video_path(video_id)
    if video_path is None:
        sys.exit("video path not found")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit("video open failed")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
                    pos = json.loads(pos)
                rally_positions[str(rid)] = pos

    rows_html: list[str] = []
    for ri, rally in enumerate(rallies):
        pos = rally_positions.get(rally.rally_id) or []
        rep = _get_pid_track_for_rally(pos)
        cells_html: list[str] = []
        for pid in (1, 2, 3, 4):
            p = rep.get(pid)
            if p is None:
                cells_html.append('<td class="empty">—</td>')
                continue
            cx, cy = float(p.get("x", 0)), float(p.get("y", 0))
            w, h = float(p.get("width", 0)), float(p.get("height", 0))
            frame_no = int(p.get("frameNumber", 0))
            ms = rally.start_ms + (frame_no * 1000 / fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, int(ms))
            ok, frame = cap.read()
            if not ok or frame is None:
                cells_html.append('<td class="empty">!</td>')
                continue
            x1 = max(0, int((cx - w / 2) * fw))
            y1 = max(0, int((cy - h / 2) * fh))
            x2 = min(fw, int((cx + w / 2) * fw))
            y2 = min(fh, int((cy + h / 2) * fh))
            if x2 <= x1 or y2 <= y1:
                cells_html.append('<td class="empty">!</td>')
                continue
            crop = frame[y1:y2, x1:x2]
            raw = _resize_h(crop)
            regions = _resize_h(_draw_regions(crop))
            clothing = _resize_h(_draw_clothing_mask(crop))
            triple = (
                f'<img src="{_b64_from_bgr(raw)}" /> '
                f'<img src="{_b64_from_bgr(regions)}" /> '
                f'<img src="{_b64_from_bgr(clothing)}" />'
            )
            cells_html.append(f'<td>{triple}</td>')
        rows_html.append(
            f'<tr><th>r{ri:02d}<br/><small>{rally.rally_id[:8]}</small></th>'
            + "".join(cells_html) + "</tr>"
        )

    cap.release()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Feature inspection — {video_id[:8]}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #1a1a1a; color: #ddd; margin: 0; padding: 16px; }}
  h1 {{ font-size: 14px; margin: 0 0 8px; }}
  .legend {{ font-size: 12px; color: #aaa; margin-bottom: 12px; line-height: 1.5; }}
  .legend code {{ background: #333; padding: 1px 5px; border-radius: 3px; }}
  table {{ border-collapse: collapse; }}
  th {{ background: #2a2a2a; padding: 8px; text-align: center; font-size: 11px; }}
  th small {{ font-weight: normal; color: #888; }}
  td {{ background: #252525; padding: 6px; vertical-align: top; border: 1px solid #333; white-space: nowrap; }}
  td.empty {{ background: #1a1a1a; color: #666; text-align: center; padding: 20px; min-width: 100px; }}
  td img {{ height: 200px; vertical-align: top; }}
  .pid1 {{ background: #1e3a5f !important; }}
  .pid2 {{ background: #5a3a1e !important; }}
  .pid3 {{ background: #3a5a1e !important; }}
  .pid4 {{ background: #5a1e3a !important; }}
</style></head>
<body>
<h1>Feature inspection — {video_id} ({len(rallies)} rallies)</h1>
<div class="legend">
  Each cell shows three images, mid-rally crop for that PID:
  <b>1) raw crop</b> — what the camera sees.
  <b>2) region overlay</b> — fixed bbox-fraction regions:
    <span style="color:#f44">red 0-15% = head_hist</span>,
    <span style="color:#0ff">cyan 20-55% = upper_body_hist</span>,
    <span style="color:#0f0">green 50-78% = lower_body_hist</span>.
  <b>3) clothing mask</b> — red pixels are <b>DROPPED</b> by
  <code>_build_clothing_mask</code> (sand-warm-hue + skin + outer-25% margin);
  remaining pixels go into the HSV histograms. The <code>%</code> overlay shows
  the fraction of the full bbox kept.
  <br/>
  <b>What to look for:</b>
  (a) cyan/green regions covering sand or other players → bbox-fraction is wrong;
  (b) clothing-mask dropping the actual shirt/shorts pixels → mask too aggressive;
  (c) clothing-mask keeping mostly sand or skin → mask too lenient.
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
    out = args.out or (
        Path(__file__).resolve().parents[1]
        / "reports" / "visual_debug" / f"{short}_feature_inspection.html"
    )
    build_sheet(args.video_id, out)


if __name__ == "__main__":
    main()
