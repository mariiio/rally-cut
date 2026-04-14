"""Extract mid-flight ball crops for manual blur classification.

Saves individual 96×96 crops at GT-verified ball positions, and builds
contact sheets (grids of ~50 crops each) for multimodal visual review.

Usage
-----
    cd analysis
    uv run python scripts/extract_blur_crops.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import cv2
import numpy as np

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.video_resolver import VideoResolver

OUTPUT_DIR = Path("outputs/ball_3d_rig/blur_crops")
AUDIT_FILE = Path("outputs/ball_3d_rig/audit_rallies.json")

CROP_HALF = 48  # 96×96 crops
CONTACT_EXCLUSION = 5
TARGET_CROPS = 200
CROPS_PER_RALLY = 40
SHEET_COLS = 10
SHEET_ROWS = 5  # 50 per sheet


def main() -> None:
    audit = json.loads(AUDIT_FILE.read_text())
    rally_ids = [r["rally_id"] for r in audit["audit_rallies"]]
    meta = {r["rally_id"]: r for r in audit["audit_rallies"]}

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, pt.ground_truth_json, pt.action_ground_truth_json,
                   v.content_hash, pt.fps, r.video_id
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE r.id = ANY(%s)
              AND pt.ground_truth_json IS NOT NULL
        """, (rally_ids,))
        rows = cur.fetchall()

    resolver = VideoResolver()

    # Pick 5 rallies: 2 low, 2 mid, 1 high (mix of camera heights).
    by_tier: dict[str, list] = {"low": [], "mid": [], "high": []}
    for row in rows:
        rid = str(row[0])
        m = meta[rid]
        chash = row[3]
        if chash and resolver.is_cached(chash):
            by_tier[m["tier"]].append(row)

    selected: list = []
    for tier, target in [("low", 2), ("mid", 2), ("high", 1)]:
        selected.extend(by_tier[tier][:target])

    print(f"Selected {len(selected)} rallies for crop extraction")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_crops: list[dict[str, Any]] = []
    cap: cv2.VideoCapture | None = None
    cap_hash: str | None = None

    for row in selected:
        rid = str(row[0])
        gt = row[1]
        action_gt = row[2] or []
        chash = row[3]
        fps = float(row[4] or 30.0)
        vid = str(row[5])
        m = meta[rid]

        # Ball GT positions.
        positions = gt.get("positions", []) if isinstance(gt, dict) else gt or []
        ball_gt = [
            (int(p["frameNumber"]), float(p["x"]), float(p["y"]))
            for p in positions
            if isinstance(p, dict) and (p.get("label") or "").lower() == "ball"
        ]
        ball_gt.sort()

        contact_frames = {int(l["frame"]) for l in action_gt if "frame" in l}

        def in_contact(f: int) -> bool:
            return any(abs(f - c) <= CONTACT_EXCLUSION for c in contact_frames)

        mid_flight = [(f, x, y) for f, x, y in ball_gt if not in_contact(f)]
        if not mid_flight:
            continue

        # Uniform-stride sample.
        if len(mid_flight) > CROPS_PER_RALLY:
            idxs = np.linspace(0, len(mid_flight) - 1, CROPS_PER_RALLY, dtype=int)
            mid_flight = [mid_flight[i] for i in idxs]

        # Open video.
        if cap_hash != chash:
            if cap is not None:
                cap.release()
            path = resolver.get_cached_path(chash)
            cap = cv2.VideoCapture(str(path))
            cap_hash = chash

        for f, bx, by in mid_flight:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            cx = int(bx * w)
            cy = int(by * h)
            x0 = max(cx - CROP_HALF, 0)
            y0 = max(cy - CROP_HALF, 0)
            x1 = min(cx + CROP_HALF, w)
            y1 = min(cy + CROP_HALF, h)
            if x1 - x0 < CROP_HALF or y1 - y0 < CROP_HALF:
                continue
            crop = frame[y0:y1, x0:x1]
            idx = len(all_crops)
            crop_path = OUTPUT_DIR / f"crop_{idx:04d}.png"
            cv2.imwrite(str(crop_path), crop)
            all_crops.append({
                "idx": idx,
                "rally_id": rid,
                "video_id": vid,
                "tier": m["tier"],
                "camera_height_m": m.get("camera_height_m", 0.0),
                "frame": f,
                "ball_x": bx,
                "ball_y": by,
                "fps": fps,
            })

    if cap is not None:
        cap.release()

    print(f"Extracted {len(all_crops)} crops")

    # Build contact sheets (grids of SHEET_ROWS × SHEET_COLS).
    n_sheets = (len(all_crops) + SHEET_ROWS * SHEET_COLS - 1) // (SHEET_ROWS * SHEET_COLS)
    for sheet_idx in range(n_sheets):
        start = sheet_idx * SHEET_ROWS * SHEET_COLS
        end = min(start + SHEET_ROWS * SHEET_COLS, len(all_crops))
        batch = all_crops[start:end]

        # Read crops and tile.
        tiles: list[np.ndarray] = []
        for entry in batch:
            crop_path = OUTPUT_DIR / f"crop_{entry['idx']:04d}.png"
            img = cv2.imread(str(crop_path))
            if img is None:
                img = np.zeros((CROP_HALF * 2, CROP_HALF * 2, 3), dtype=np.uint8)
            # Resize to exact size if needed.
            img = cv2.resize(img, (CROP_HALF * 2, CROP_HALF * 2))
            # Add index number overlay.
            cv2.putText(
                img, str(entry["idx"]), (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1,
            )
            tiles.append(img)

        # Pad to fill the grid.
        while len(tiles) < SHEET_ROWS * SHEET_COLS:
            tiles.append(np.zeros_like(tiles[0]))

        # Assemble grid.
        rows_list = []
        for r in range(SHEET_ROWS):
            row_tiles = tiles[r * SHEET_COLS:(r + 1) * SHEET_COLS]
            rows_list.append(np.hstack(row_tiles))
        grid = np.vstack(rows_list)

        sheet_path = OUTPUT_DIR / f"sheet_{sheet_idx}.png"
        cv2.imwrite(str(sheet_path), grid)
        print(f"  Sheet {sheet_idx}: crops {start}-{end - 1} → {sheet_path}")

    # Save manifest.
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(all_crops, indent=2))
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
