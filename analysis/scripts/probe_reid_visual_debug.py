"""Visual debugger for the learned ReID head's view of a rally.

For a target rally, extracts the actual bbox crops the ReID head sees
at sampled frames per filtered track, saves them to disk, computes the
pairwise learned-cosine matrix, and writes an HTML page that shows
each crop alongside its frame number, raw BoT-SORT track id, bbox
dimensions, and pairwise cosine similarities.

Use this when you want to SEE what the model is seeing — to diagnose
whether failed merges are due to:
  - Bad tracking (bboxes mis-cropped, capturing only partial body /
    background / wrong person)
  - Occlusion artifacts in specific crops
  - Lighting / scale changes that the head can't generalize over
  - A different physical player after all (cross-player merge)

Usage:
    cd analysis
    uv run python scripts/probe_reid_visual_debug.py \
        --rally 533bcdd5-11af-442e-9ecd-3ec467e038c7 \
        --max-frames 12

Output: `analysis/reports/reid_visual_debug/<rally_short>/`
  - crops/<filtered_track>_<frame>_raw<rawtid>.jpg
  - cosines.json (full pairwise + per-track summaries)
  - index.html (browse in Chrome, drag-and-drop into the file viewer)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from rallycut.evaluation.tracking.db import get_connection, get_video_path
from rallycut.tracking.player_features import extract_bbox_crop
from rallycut.tracking.reid_embeddings import (
    _default_device,
    extract_learned_embeddings,
)

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ANALYSIS_ROOT / "reports" / "reid_visual_debug"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _load_rally(rally_id: str) -> dict[str, Any]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                r.id, r.video_id, r.start_ms, r.end_ms,
                pt.positions_json, pt.raw_positions_json,
                pt.primary_track_ids, pt.fps, pt.frame_count,
                v.match_analysis_json,
                v.player_matching_gt_json
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE r.id = %s
            """,
            [rally_id],
        )
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"rally {rally_id} not found")
    (
        rid, video_id, start_ms, end_ms,
        pos, raw_pos, primary, fps, frame_count,
        match_analysis, gt_json,
    ) = row
    pos = pos if isinstance(pos, list) else (json.loads(cast(str, pos)) if pos else [])
    raw_pos = raw_pos if isinstance(raw_pos, list) else (json.loads(cast(str, raw_pos)) if raw_pos else [])
    if isinstance(match_analysis, str):
        match_analysis = json.loads(match_analysis)
    if isinstance(gt_json, str):
        gt_json = json.loads(gt_json)
    rally_entry = None
    afm = None
    if match_analysis:
        for entry in (match_analysis.get("rallies", []) or []):
            if entry.get("rallyId") == rid:
                rally_entry = entry
                afm = entry.get("appliedFullMapping") or entry.get("trackToPlayer") or {}
                break
    gt_labels = []
    if gt_json:
        gt_rallies = gt_json.get("rallies", {}) or {}
        if rid in gt_rallies:
            gt_labels = gt_rallies[rid].get("labels", [])
    return {
        "rally_id": rid,
        "video_id": video_id,
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "positions": pos,
        "raw_positions": raw_pos,
        "primary_track_ids": primary or [],
        "fps": float(fps) if fps else 30.0,
        "frame_count": int(frame_count) if frame_count else 0,
        "rally_entry": rally_entry,
        "applied_full_mapping": afm or {},
        "gt_labels": gt_labels,
    }


def _sample_frames(positions: list[dict[str, Any]], track_id: int, n: int) -> list[dict[str, Any]]:
    track_pos = sorted(
        [p for p in positions if int(p.get("trackId", -1)) == track_id],
        key=lambda p: int(p.get("frameNumber", 0)),
    )
    if not track_pos:
        return []
    if len(track_pos) <= n:
        return track_pos
    step = len(track_pos) / n
    return [track_pos[int(i * step)] for i in range(n)]


def _closest_raw_track(
    pos: dict[str, Any], raw_by_frame: dict[int, list[dict[str, Any]]],
) -> int | None:
    """Find the BoT-SORT raw track id whose bbox is closest to this position at this frame."""
    fnum = int(pos.get("frameNumber", -1))
    px = float(pos.get("x", 0)) + float(pos.get("width", 0)) / 2
    py = float(pos.get("y", 0)) + float(pos.get("height", 0)) / 2
    best_tid: int | None = None
    best_d = float("inf")
    for rp in raw_by_frame.get(fnum, []):
        rx = float(rp.get("x", 0)) + float(rp.get("width", 0)) / 2
        ry = float(rp.get("y", 0)) + float(rp.get("height", 0)) / 2
        d = (rx - px) ** 2 + (ry - py) ** 2
        if d < best_d:
            best_d = d
            best_tid = int(rp.get("trackId", -1))
    return best_tid if best_d < 0.05 ** 2 else None  # 5% match radius


def _extract_and_save_crop(
    cap: cv2.VideoCapture,
    rally_start_ms: int,
    video_fps: float,
    pos: dict[str, Any],
    out_path: Path,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Read frame, extract bbox crop, save jpg. Returns (crop, meta) or (None, meta)."""
    f_in_rally = int(pos.get("frameNumber", 0))
    f_in_video = int(rally_start_ms / 1000.0 * video_fps) + f_in_rally
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_in_video)
    ret, frame = cap.read()
    meta = {
        "rally_frame": f_in_rally,
        "video_frame": f_in_video,
        "x": pos.get("x"),
        "y": pos.get("y"),
        "width": pos.get("width"),
        "height": pos.get("height"),
        "confidence": pos.get("confidence"),
    }
    if not ret or frame is None:
        meta["error"] = "frame read failed"
        return None, meta
    fh, fw = frame.shape[:2]
    crop = extract_bbox_crop(
        frame,
        (
            pos["x"] + pos["width"] / 2,
            pos["y"] + pos["height"] / 2,
            pos["width"],
            pos["height"],
        ),
        fw, fh,
    )
    if crop is None:
        meta["error"] = "crop too small or invalid"
        return None, meta
    meta["crop_w"] = int(crop.shape[1])
    meta["crop_h"] = int(crop.shape[0])
    cv2.imwrite(str(out_path), crop)
    return crop, meta


def _format_html(
    rally_id: str,
    rally_info: dict[str, Any],
    track_records: dict[int, list[dict[str, Any]]],
    track_pair_cosines: dict[tuple[int, int], dict[str, Any]],
    out_dir: Path,
) -> None:
    html: list[str] = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append(f"<title>ReID visual debug — {rally_id[:8]}</title>")
    html.append("""
<style>
body { font-family: -apple-system, sans-serif; max-width: 1600px; margin: 1em auto; padding: 0 1em; }
h1, h2 { font-weight: 600; }
.track-section { margin-bottom: 2em; border-top: 2px solid #333; padding-top: 1em; }
.crop-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.crop-card { border: 1px solid #ccc; padding: 4px; background: #f9f9f9; font-size: 11px; line-height: 1.3; }
.crop-card img { display: block; max-height: 220px; max-width: 140px; }
.crop-card .meta { margin-top: 4px; color: #555; }
table { border-collapse: collapse; margin: 1em 0; font-size: 13px; }
table th, table td { border: 1px solid #ccc; padding: 4px 8px; text-align: center; }
table th { background: #eee; font-weight: 600; }
.cos-high { background: #fee; color: #c00; font-weight: bold; }
.cos-mid { background: #ffd; }
.cos-low { background: #efe; color: #060; font-weight: bold; }
.legend { font-size: 12px; color: #666; margin-top: 0.5em; }
</style>
</head><body>
""")
    html.append(f"<h1>ReID visual debug — rally {rally_id[:8]}</h1>")
    html.append("<p><b>Goal:</b> see what the learned ReID head sees on this rally so we can identify why it doesn't recognize same-player fragments.</p>")
    html.append("<h2>Rally summary</h2>")
    html.append("<ul>")
    html.append(f"<li>video_id: {rally_info['video_id']}</li>")
    html.append(f"<li>frame_count: {rally_info['frame_count']}, fps: {rally_info['fps']}</li>")
    html.append(f"<li>primary_track_ids (post-remap): {rally_info['primary_track_ids']}</li>")
    html.append(f"<li>appliedFullMapping (pre→post): {rally_info['applied_full_mapping']}</li>")
    html.append(f"<li>GT labels: {len(rally_info['gt_labels'])}</li>")
    html.append("</ul>")

    # Cosine similarity matrix
    if track_pair_cosines:
        html.append("<h2>Pairwise learned ReID cosine (median of all crop pairs)</h2>")
        track_ids_sorted = sorted(track_records.keys())
        html.append("<table><tr><th></th>")
        for tid in track_ids_sorted:
            html.append(f"<th>T{tid}</th>")
        html.append("</tr>")
        for tid_a in track_ids_sorted:
            html.append(f"<tr><th>T{tid_a}</th>")
            for tid_b in track_ids_sorted:
                if tid_a == tid_b:
                    html.append("<td>—</td>")
                    continue
                pair = (min(tid_a, tid_b), max(tid_a, tid_b))
                if pair in track_pair_cosines:
                    cos = track_pair_cosines[pair]["median_cosine"]
                    cls = "cos-high" if cos > 0.7 else "cos-low" if cos < 0.4 else "cos-mid"
                    html.append(f"<td class='{cls}'>{cos:+.3f}</td>")
                else:
                    html.append("<td>n/a</td>")
            html.append("</tr>")
        html.append("</table>")
        html.append("<div class='legend'>Green = ReID says different players (cos &lt; 0.4). Yellow = ambiguous. Red = ReID says same player (cos &gt; 0.7). For the failed-merge case (T1 vs T2 should be same player), the cell should ideally be RED — if it's yellow/green, the head can't see the similarity.</div>")

    # Per-track crop grids
    for tid in sorted(track_records.keys()):
        records = track_records[tid]
        html.append(f"<div class='track-section'>")
        html.append(f"<h2>Filtered track T{tid} — {len(records)} crops</h2>")
        if records:
            frame_range = (records[0]["meta"]["rally_frame"], records[-1]["meta"]["rally_frame"])
            html.append(f"<p>Frame range: {frame_range[0]}-{frame_range[1]}. Raw track lineage (closest BoT-SORT): {sorted(set(r['raw_track'] for r in records if r['raw_track'] is not None))}</p>")
        html.append("<div class='crop-grid'>")
        for r in records:
            m = r["meta"]
            crop_path = r.get("crop_filename")
            html.append("<div class='crop-card'>")
            if crop_path and (out_dir / "crops" / crop_path).exists():
                rally_frame = m['rally_frame']
                html.append(f"<img src='crops/{crop_path}' alt='T{tid} f{rally_frame}' />")
            html.append("<div class='meta'>")
            html.append(f"frame {m['rally_frame']}<br>")
            html.append(f"raw track {r.get('raw_track', '?')}<br>")
            if "crop_w" in m:
                html.append(f"crop {m['crop_w']}×{m['crop_h']}px<br>")
            if m.get("confidence") is not None:
                html.append(f"conf {m['confidence']:.2f}<br>")
            if "error" in m:
                html.append(f"<b style='color:red'>{m['error']}</b><br>")
            html.append("</div></div>")
        html.append("</div></div>")

    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rally", required=True, help="Rally UUID")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames per filtered track")
    args = parser.parse_args()

    rally = _load_rally(args.rally)
    rally_short = rally["rally_id"].split("-")[0]
    out_dir = OUT_ROOT / rally_short
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    # Group raw positions by frame for closest-raw-track lookup
    raw_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in rally["raw_positions"]:
        fn = int(p.get("frameNumber", -1))
        raw_by_frame[fn].append(p)

    # Group filtered positions by track id
    filtered_track_ids = sorted(
        {int(p.get("trackId", -1)) for p in rally["positions"] if p.get("trackId") is not None},
    )
    print(f"Rally {rally_short}: {len(rally['positions'])} positions, "
          f"filtered track ids = {filtered_track_ids}")

    video_path = get_video_path(rally["video_id"])
    if video_path is None:
        print(f"ERROR: video {rally['video_id']} not available", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}", file=sys.stderr)
        return 1
    video_fps = cap.get(cv2.CAP_PROP_FPS) or rally["fps"]

    device = _default_device()
    print(f"Using device: {device}")

    # Per-track: extract crops, save, compute embeddings
    track_records: dict[int, list[dict[str, Any]]] = {}
    track_embeddings: dict[int, np.ndarray] = {}
    for tid in filtered_track_ids:
        samples = _sample_frames(rally["positions"], tid, args.max_frames)
        if not samples:
            continue
        records: list[dict[str, Any]] = []
        crops: list[np.ndarray] = []
        for s in samples:
            raw_tid = _closest_raw_track(s, raw_by_frame)
            crop_filename = f"T{tid}_f{int(s['frameNumber']):04d}_raw{raw_tid if raw_tid is not None else 'none'}.jpg"
            crop_path = out_dir / "crops" / crop_filename
            crop, meta = _extract_and_save_crop(
                cap, rally["start_ms"], video_fps, s, crop_path,
            )
            records.append({
                "raw_track": raw_tid,
                "meta": meta,
                "crop_filename": crop_filename if crop is not None else None,
            })
            if crop is not None:
                crops.append(crop)
        track_records[tid] = records
        if crops:
            emb = extract_learned_embeddings(crops, device=device)
            if emb.shape[0] > 0:
                track_embeddings[tid] = emb
        print(f"  T{tid}: {len(records)} samples, {len(crops)} crops extracted, "
              f"{track_embeddings.get(tid, np.empty(0)).shape[0]} embeddings")
    cap.release()

    # Compute pairwise cosines
    track_pair_cosines: dict[tuple[int, int], dict[str, Any]] = {}
    track_ids_with_embeddings = sorted(track_embeddings.keys())
    for i, tid_a in enumerate(track_ids_with_embeddings):
        for tid_b in track_ids_with_embeddings[i + 1:]:
            ea = track_embeddings[tid_a]
            eb = track_embeddings[tid_b]
            scores = []
            for ra in ea:
                for rb in eb:
                    scores.append(float(np.dot(ra, rb)))
            if not scores:
                continue
            track_pair_cosines[(tid_a, tid_b)] = {
                "n_pairs": len(scores),
                "median_cosine": float(np.median(scores)),
                "mean_cosine": float(np.mean(scores)),
                "min_cosine": float(np.min(scores)),
                "max_cosine": float(np.max(scores)),
                "p10_cosine": float(np.percentile(scores, 10)),
                "p90_cosine": float(np.percentile(scores, 90)),
            }
            print(f"  T{tid_a} ↔ T{tid_b}: median cos = "
                  f"{track_pair_cosines[(tid_a, tid_b)]['median_cosine']:+.3f} "
                  f"(p10={track_pair_cosines[(tid_a, tid_b)]['p10_cosine']:+.3f}, "
                  f"p90={track_pair_cosines[(tid_a, tid_b)]['p90_cosine']:+.3f})")

    # Save JSON summary + HTML
    summary = {
        "rally_id": rally["rally_id"],
        "rally_short": rally_short,
        "primary_track_ids": rally["primary_track_ids"],
        "applied_full_mapping": rally["applied_full_mapping"],
        "track_pair_cosines": {
            f"T{a}_T{b}": v for (a, b), v in track_pair_cosines.items()
        },
        "track_records": {str(tid): records for tid, records in track_records.items()},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "cosines.json").write_text(json.dumps(summary, indent=2, default=str))

    _format_html(
        rally["rally_id"], rally, track_records, track_pair_cosines, out_dir,
    )

    print()
    print(f"Wrote {out_dir}/index.html — open in browser to inspect crops")
    print(f"Crops at {out_dir}/crops/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
