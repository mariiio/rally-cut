#!/usr/bin/env python3
"""Render a per-rally pid-overlay contact sheet for a single video.

For each rally in the video:
  - Sample the mid-rally frame (positions are densest there).
  - Draw each player's bbox with a per-pid color and label
    (``tid=N pid=M``) using the canonical pid map currently in DB.
  - Add a banner with the rally index, rally short id, frame number,
    and (if ``match_analysis_json.sideSwitches`` says so) "SWITCH".

Output: one JPG per fixture composited into a single grid PNG so the
whole match is visible in one image. Suitable for human eyeball
validation against retracked truth.

Usage::

    cd analysis
    uv run python scripts/render_match_overlay.py \\
        --video-id 5c756c41 --output /tmp/wawa_overlay.png

    # Also writes per-rally crops to /tmp/wawa_overlay/ for closer inspection.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psycopg

# Per-pid colors (BGR) — chosen for legibility against beach backgrounds.
PID_COLORS = {
    1: (50, 50, 255),    # red
    2: (50, 255, 255),   # yellow
    3: (50, 255, 50),    # green
    4: (255, 100, 255),  # magenta
}
UNKNOWN_COLOR = (180, 180, 180)


def _connect() -> psycopg.Connection:
    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5436/rallycut",
    )
    return psycopg.connect(db_url)


def resolve_full_video_id(prefix: str) -> str:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM videos WHERE id::text LIKE %s",
            [f"{prefix}%"],
        )
        rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"No video with prefix {prefix!r}")
    if len(rows) > 1:
        raise SystemExit(f"Ambiguous prefix {prefix!r}: {[r[0] for r in rows]}")
    return str(rows[0][0])


def fetch_video_meta(video_id: str) -> dict[str, Any]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT canonical_pid_map_json, fps, match_analysis_json "
            "FROM videos WHERE id = %s",
            [video_id],
        )
        row = cur.fetchone()
    if not row:
        raise SystemExit(f"Video {video_id} not found")
    canon = row[0] or {}
    ma = row[2] or {}

    # Per-rally track_to_player lives in match_analysis_json.rallies (the
    # solver's authoritative output). canonical_pid_map_json is the
    # within-team permutation written by the bypass path's seed-rally
    # logic and is NULL on the blind path. We prefer match_analysis_json
    # and fall back to canonical_pid_map_json so this script works on
    # both paths.
    track_to_player_by_rally: dict[str, dict[str, int]] = {}
    if isinstance(ma, dict):
        for entry in ma.get("rallies", []) or []:
            rid = entry.get("rallyId") or entry.get("rally_id")
            ttp = entry.get("trackToPlayer") or entry.get("track_to_player")
            if rid and isinstance(ttp, dict):
                track_to_player_by_rally[rid] = {
                    str(k): int(v) for k, v in ttp.items()
                }
    if isinstance(canon, dict):
        for rid, mp in (canon.get("rallies") or {}).items():
            if rid not in track_to_player_by_rally and isinstance(mp, dict):
                track_to_player_by_rally[rid] = {
                    str(k): int(v) for k, v in mp.items()
                }

    return {
        "canonical_map": track_to_player_by_rally,
        "fps": float(row[1] or 30.0),
        "side_switches": ma.get("sideSwitches", []) if isinstance(ma, dict) else [],
    }


def fetch_rallies(video_id: str) -> list[dict[str, Any]]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT r.id::text, r.start_ms, r.end_ms, p.positions_json "
            "FROM rallies r "
            "LEFT JOIN player_tracks p ON p.rally_id = r.id "
            "WHERE r.video_id = %s "
            "ORDER BY r.start_ms ASC",
            [video_id],
        )
        rows = cur.fetchall()
    rallies = []
    for rid, start_ms, end_ms, positions_json in rows:
        rallies.append({
            "rally_id": rid,
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "positions": positions_json or [],
        })
    return rallies


def find_video_path(video_id: str, meta: dict[str, Any]) -> Path:
    """Resolve a usable local video path.

    Mirrors the lookup ``rallycut.evaluation.tracking.db.get_video_path``
    does — proxy is preferred when present.
    """
    from rallycut.evaluation.tracking.db import get_video_path

    p = get_video_path(video_id)
    if p is None or not p.exists():
        raise SystemExit(f"No local video file for {video_id}")
    return p


def index_positions_by_frame(
    positions: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for p in positions:
        f = int(p.get("frameNumber", p.get("frame_number", -1)))
        if f < 0:
            continue
        out.setdefault(f, []).append(p)
    return out


def resolve_pid(track_id: int, canon_for_rally: dict[str, int] | None) -> int | None:
    if canon_for_rally is None:
        return None
    return canon_for_rally.get(str(track_id))


def draw_overlay(
    img: np.ndarray,
    boxes: list[dict[str, Any]],
    canon_for_rally: dict[str, int] | None,
    rally_idx: int,
    rally_id_short: str,
    frame_in_rally: int,
    max_frame: int,
    side_switch_at_this_rally: bool,
) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for b in boxes:
        tid = int(b.get("trackId", b.get("track_id", -1)))
        pid = resolve_pid(tid, canon_for_rally)
        cx = float(b.get("x", 0)) * w
        cy = float(b.get("y", 0)) * h
        bw = float(b.get("width", 0)) * w
        bh = float(b.get("height", 0)) * h
        x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
        x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
        color = PID_COLORS.get(pid, UNKNOWN_COLOR) if pid else UNKNOWN_COLOR
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
        label = f"t{tid}->p{pid if pid else '?'}"
        # Background pad for legibility.
        (lw, lh), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
        )
        ly = max(lh + 4, y0 - 6)
        cv2.rectangle(
            out, (x0, ly - lh - 4), (x0 + lw + 6, ly + 4), (0, 0, 0), -1,
        )
        cv2.putText(
            out, label, (x0 + 3, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

    banner_text = (
        f"rally #{rally_idx + 1}  id={rally_id_short}  "
        f"frame {frame_in_rally}/{max_frame}"
    )
    if side_switch_at_this_rally:
        banner_text += "  *** SIDE SWITCH ***"
    cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(
        out, banner_text, (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )
    return out


def make_contact_sheet(
    frames: list[np.ndarray],
    cols: int = 2,
    target_height: int = 540,
) -> np.ndarray:
    """Compose frames into a grid. All frames are downscaled to
    ``target_height`` pixels for compactness.
    """
    if not frames:
        raise SystemExit("No frames to compose")

    resized = []
    for f in frames:
        h, w = f.shape[:2]
        scale = target_height / h
        new_w = int(round(w * scale))
        resized.append(cv2.resize(f, (new_w, target_height)))

    # Pad each cell to the same width so rows align.
    max_w = max(f.shape[1] for f in resized)
    padded = []
    for f in resized:
        h, w = f.shape[:2]
        if w < max_w:
            pad = np.zeros((h, max_w - w, 3), dtype=f.dtype)
            f = np.hstack([f, pad])
        padded.append(f)

    rows: list[np.ndarray] = []
    for r in range(0, len(padded), cols):
        row_cells = padded[r:r + cols]
        # Pad final row to full width.
        while len(row_cells) < cols:
            row_cells.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row_cells))
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True,
                    help="Full or prefix video id")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output composite PNG path")
    ap.add_argument("--per-rally-dir", type=Path, default=None,
                    help="Optional: also write per-rally JPGs to this dir")
    ap.add_argument("--cols", type=int, default=3,
                    help="Contact-sheet grid columns (default 3)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    full_id = resolve_full_video_id(args.video_id)
    meta = fetch_video_meta(full_id)
    rallies = fetch_rallies(full_id)
    video_path = find_video_path(full_id, meta)

    print(f"video {full_id} ({video_path.name})")
    print(f"  rallies={len(rallies)} "
          f"side_switches={meta['side_switches']} fps={meta['fps']:.1f}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or meta["fps"]

    # ``sideSwitches`` is a list of rally indices where the switch BECOMES
    # active (rally i and onward are flipped). Convert to a per-rally
    # boolean for the banner so the user sees where the detector said the
    # flip happened.
    switches = set(meta["side_switches"])

    annotated_frames: list[np.ndarray] = []
    if args.per_rally_dir:
        args.per_rally_dir.mkdir(parents=True, exist_ok=True)

    for idx, r in enumerate(rallies):
        positions = r["positions"]
        by_frame = index_positions_by_frame(positions)
        if not by_frame:
            print(f"  rally {idx + 1}: no positions; skipping")
            continue

        # Pick the rally midpoint for visual stability.
        sorted_frames = sorted(by_frame.keys())
        mid_in_rally = sorted_frames[len(sorted_frames) // 2]
        f_in_video = int(round(r["start_ms"] / 1000.0 * video_fps)) + mid_in_rally
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_in_video)
        ok, img = cap.read()
        if not ok:
            print(f"  rally {idx + 1}: could not read frame {f_in_video}")
            continue

        canon_for_rally = meta["canonical_map"].get(r["rally_id"])
        boxes = by_frame[mid_in_rally]
        annotated = draw_overlay(
            img=img,
            boxes=boxes,
            canon_for_rally=canon_for_rally,
            rally_idx=idx,
            rally_id_short=r["rally_id"][:8],
            frame_in_rally=mid_in_rally,
            max_frame=sorted_frames[-1],
            side_switch_at_this_rally=(idx in switches),
        )
        annotated_frames.append(annotated)

        if args.per_rally_dir:
            out_path = args.per_rally_dir / (
                f"rally_{idx + 1:02d}_{r['rally_id'][:8]}.jpg"
            )
            cv2.imwrite(str(out_path), annotated)

        pid_summary = sorted(
            (int(b.get("trackId", b.get("track_id", -1))),
             resolve_pid(int(b.get("trackId", b.get("track_id", -1))), canon_for_rally))
            for b in boxes
        )
        print(f"  rally {idx + 1}: f={mid_in_rally} tids->pids={pid_summary}")

    cap.release()

    if not annotated_frames:
        raise SystemExit("No annotated frames produced")

    sheet = make_contact_sheet(annotated_frames, cols=args.cols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(f"\nwrote contact sheet to {args.output}")
    if args.per_rally_dir:
        print(f"per-rally JPGs in {args.per_rally_dir}")


if __name__ == "__main__":
    main()
