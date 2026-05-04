"""
Validation spike: can YOLO11s-pose recover the near-side partner who is
heavily occluded by the foreground server in dd042609 r13/r18/r19?

For each rally:
  1. Extract frames at the serve-setup time (early in the rally).
  2. Run YOLO11s-pose at imgsz=1280.
  3. Find the foreground "server" track bbox from raw_positions_json
     (the largest bbox, x in midfield).
  4. Check if pose detects ≥2 distinct skeleton clusters inside that bbox.
  5. Also check the global pose-skeleton count vs. YOLO-bbox count for
     comparison: does pose see more persons in the same frame?

Output: per-rally per-frame report with keypoint clusters and bboxes,
plus a summary verdict on whether pose can recover the partner.

Read-only. No DB writes.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from rallycut.evaluation.tracking.db import get_connection
from rallycut.service.s3_utils import download_from_s3
from rallycut.tracking.pose_anchored_features import get_pose_model


VIDEO_ID = "dd042609-e22e-4f60-83ed-038897c88c32"
RALLY_PREFIXES = [("r13", "381d6268"), ("r18", "edb45683"), ("r19", "efdbf6b2")]

# Frames to sample per rally: serve-setup is usually frames 0-30.
# We also check mid-rally (50, 100) to see if pose can recover the partner
# *after* they emerge from behind the server.
FRAMES_TO_SAMPLE = [0, 10, 20, 30, 60, 100]


def _load_rally_metadata(rally_prefix: str) -> dict:
    with get_connection() as c, c.cursor() as cur:
        cur.execute(
            """
            SELECT r.id, r.start_ms, r.end_ms,
                   pt.raw_positions_json, pt.primary_track_ids, pt.fps
            FROM rallies r
            LEFT JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id = %s AND r.id LIKE %s
            """,
            (VIDEO_ID, f"{rally_prefix}%"),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"No rally found for prefix {rally_prefix}")
    return {
        "rally_id": row[0],
        "start_ms": row[1],
        "end_ms": row[2],
        "raw_positions": row[3] or [],
        "primary_track_ids": row[4],
        "fps": row[5] or 30.0,
    }


def _load_video_metadata() -> tuple[str, int, int, float]:
    with get_connection() as c, c.cursor() as cur:
        cur.execute(
            "SELECT proxy_s3_key, processed_s3_key, width, height, fps FROM videos WHERE id = %s",
            (VIDEO_ID,),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"Video {VIDEO_ID} not found")
    s3_key = row[0] or row[1]
    return s3_key, int(row[2]), int(row[3]), float(row[4])


def _foreground_track_id(raw_positions: list[dict]) -> int | None:
    """The 'server' track is the one with the largest median bbox area."""
    by_track = defaultdict(list)
    for p in raw_positions:
        by_track[p["trackId"]].append(p["width"] * p["height"])
    if not by_track:
        return None
    medians = {tid: float(np.median(areas)) for tid, areas in by_track.items()}
    return max(medians, key=medians.get)


def _bbox_at_frame(raw_positions: list[dict], track_id: int, frame: int) -> dict | None:
    matches = [p for p in raw_positions if p["trackId"] == track_id and p["frameNumber"] == frame]
    if not matches:
        # closest frame within ±2
        candidates = [
            p for p in raw_positions if p["trackId"] == track_id
            and abs(p["frameNumber"] - frame) <= 2
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p["frameNumber"] - frame))
    return matches[0]


def _yolo_count_at_frame(raw_positions: list[dict], frame: int) -> int:
    return len({p["trackId"] for p in raw_positions if p["frameNumber"] == frame})


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def _pose_detect(model, frame: np.ndarray) -> list[dict]:
    """Returns list of {bbox: [x1,y1,x2,y2], keypoints: [(x,y,conf)*17], conf: float}."""
    results = model.predict(frame, verbose=False, imgsz=1280, conf=0.25)
    out = []
    if not results:
        return out
    r0 = results[0]
    if r0.boxes is None or r0.keypoints is None:
        return out
    boxes = r0.boxes.xyxy.cpu().numpy()  # (N, 4)
    confs = r0.boxes.conf.cpu().numpy()  # (N,)
    kpts_xy = r0.keypoints.xy.cpu().numpy()  # (N, 17, 2)
    kpts_conf = r0.keypoints.conf.cpu().numpy() if r0.keypoints.conf is not None else None
    for i in range(len(boxes)):
        kp = []
        for j in range(kpts_xy.shape[1]):
            x, y = kpts_xy[i, j]
            c = float(kpts_conf[i, j]) if kpts_conf is not None else 1.0
            kp.append((float(x), float(y), c))
        out.append({
            "bbox": boxes[i].tolist(),
            "conf": float(confs[i]),
            "keypoints": kp,
        })
    return out


def _bbox_overlap(box_a: list[float], box_b: list[float]) -> float:
    """IoU of two xyxy bboxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _pose_inside_bbox(pose: dict, server_bbox_xyxy: list[float]) -> bool:
    """A pose detection is 'inside' the server bbox if its bbox center is inside."""
    x1, y1, x2, y2 = pose["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx1, sy1, sx2, sy2 = server_bbox_xyxy
    return sx1 <= cx <= sx2 and sy1 <= cy <= sy2


def _pose_overlaps_bbox(pose: dict, server_bbox_xyxy: list[float], min_iou: float = 0.10) -> bool:
    return _bbox_overlap(pose["bbox"], server_bbox_xyxy) >= min_iou


def main() -> None:
    print(f"=== Pose-occlusion recovery validation for {VIDEO_ID} ===\n")

    s3_key, w, h, fps = _load_video_metadata()
    print(f"Video: {s3_key} {w}x{h} @ {fps:.2f}fps\n")

    s3_endpoint = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
    bucket = os.environ.get("S3_BUCKET_NAME", "rallycut-dev")
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
        os.environ["S3_ENDPOINT"] = s3_endpoint
    print(f"S3 endpoint: {os.environ.get('S3_ENDPOINT')}, bucket: {bucket}\n")

    print("Loading YOLO11s-pose model...")
    pose_model = get_pose_model()
    print("Model loaded.\n")

    summary = []

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        local_video = download_from_s3(s3_key, bucket, td_path, label="[VALIDATE]")
        cap = cv2.VideoCapture(str(local_video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {local_video}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Opened video: {total_frames} frames @ {video_fps:.2f}fps\n")

        for label, prefix in RALLY_PREFIXES:
            print(f"================ {label} ({prefix}) ================")
            meta = _load_rally_metadata(prefix)
            rally_start_frame = int((meta["start_ms"] / 1000.0) * video_fps)
            rally_end_frame = int((meta["end_ms"] / 1000.0) * video_fps)
            rally_total_frames = rally_end_frame - rally_start_frame

            server_track_id = _foreground_track_id(meta["raw_positions"])
            print(f"  rally_id={meta['rally_id'][:8]} dur={(meta['end_ms']-meta['start_ms'])/1000:.1f}s")
            print(f"  start_frame={rally_start_frame} end_frame={rally_end_frame} total={rally_total_frames}")
            print(f"  primary_track_ids={meta['primary_track_ids']}")
            print(f"  inferred server_track_id (largest median area)={server_track_id}")
            print()

            for frame_offset in FRAMES_TO_SAMPLE:
                if frame_offset >= rally_total_frames:
                    continue
                global_frame = rally_start_frame + frame_offset
                frame = _read_frame(cap, global_frame)
                if frame is None:
                    print(f"  [f={frame_offset}] FRAME READ FAILED")
                    continue

                # YOLO baseline: how many tracks are present at this frame?
                yolo_n = _yolo_count_at_frame(meta["raw_positions"], frame_offset)

                # Find server bbox at this frame
                server_p = _bbox_at_frame(meta["raw_positions"], server_track_id, frame_offset) \
                    if server_track_id is not None else None
                if server_p is None:
                    print(f"  [f={frame_offset}] no server bbox at this frame — skipping")
                    continue

                # Convert normalized YOLO bbox (xywh-center) to xyxy in pixels
                fh, fw = frame.shape[:2]
                cx, cy, bw, bh = server_p["x"], server_p["y"], server_p["width"], server_p["height"]
                sx1 = (cx - bw / 2) * fw
                sy1 = (cy - bh / 2) * fh
                sx2 = (cx + bw / 2) * fw
                sy2 = (cy + bh / 2) * fh
                server_bbox_xyxy = [sx1, sy1, sx2, sy2]

                # Run pose detection
                poses = _pose_detect(pose_model, frame)

                # Filter: pose detections inside or heavily overlapping server bbox
                inside = [p for p in poses if _pose_inside_bbox(p, server_bbox_xyxy)]
                overlapping = [p for p in poses if _pose_overlaps_bbox(p, server_bbox_xyxy, 0.10)
                               and not _pose_inside_bbox(p, server_bbox_xyxy)]

                # Visible-keypoint counts (filter low-conf keypoints)
                def visible_kpts(p, thresh=0.3):
                    return sum(1 for _, _, c in p["keypoints"] if c >= thresh)

                print(f"  [f={frame_offset:3d}] yolo_tracks={yolo_n}  pose_total={len(poses)}  "
                      f"inside_server={len(inside)}  overlapping_server={len(overlapping)}")
                print(f"    server bbox: x=[{sx1:.0f}..{sx2:.0f}] y=[{sy1:.0f}..{sy2:.0f}] "
                      f"area={(sx2-sx1)*(sy2-sy1):.0f}px")
                for j, p in enumerate(inside):
                    bx1, by1, bx2, by2 = p["bbox"]
                    print(f"    INSIDE #{j}: bbox=[{bx1:.0f},{by1:.0f}..{bx2:.0f},{by2:.0f}] "
                          f"conf={p['conf']:.2f} visible_kpts={visible_kpts(p)}")
                for j, p in enumerate(overlapping):
                    bx1, by1, bx2, by2 = p["bbox"]
                    iou = _bbox_overlap(p["bbox"], server_bbox_xyxy)
                    print(f"    OVERLAP #{j}: bbox=[{bx1:.0f},{by1:.0f}..{bx2:.0f},{by2:.0f}] "
                          f"conf={p['conf']:.2f} IoU={iou:.2f} visible_kpts={visible_kpts(p)}")

                summary.append({
                    "rally": label,
                    "frame": frame_offset,
                    "yolo_n": yolo_n,
                    "pose_total": len(poses),
                    "pose_inside_server": len(inside),
                    "pose_overlapping_server": len(overlapping),
                })
            print()

        cap.release()

    # Final verdict
    print("\n=== SUMMARY ===")
    print(f"{'rally':<6} {'frame':<6} {'yolo_n':<8} {'pose_total':<11} {'inside_srv':<11} {'overlap_srv':<12}")
    for s in summary:
        print(f"{s['rally']:<6} {s['frame']:<6} {s['yolo_n']:<8} {s['pose_total']:<11} "
              f"{s['pose_inside_server']:<11} {s['pose_overlapping_server']:<12}")

    # Verdict: if pose finds ≥2 distinct skeletons inside the server bbox
    # in any frame of any rally, the recovery approach is feasible
    feasible_frames = [s for s in summary if s["pose_inside_server"] >= 2]
    if feasible_frames:
        print(f"\n✓ FEASIBLE: pose found ≥2 skeletons inside server bbox in "
              f"{len(feasible_frames)}/{len(summary)} sampled frames.")
    else:
        # Maybe the partner is not strictly inside but overlapping
        overlap_frames = [s for s in summary
                          if s["pose_inside_server"] + s["pose_overlapping_server"] >= 2]
        if overlap_frames:
            print(f"\n⚠ PARTIAL: pose found additional skeleton overlapping server bbox in "
                  f"{len(overlap_frames)}/{len(summary)} sampled frames "
                  "(partner partially visible at edges, not fully behind).")
        else:
            # Compare pose_total vs yolo_n: does pose see more persons globally?
            extra_persons = [s for s in summary if s["pose_total"] > s["yolo_n"]]
            if extra_persons:
                print(f"\n⚠ PARTIAL: pose sees more persons globally than YOLO in "
                      f"{len(extra_persons)}/{len(summary)} frames "
                      "(but not specifically the occluded partner).")
            else:
                print(f"\n✗ INFEASIBLE: pose detected the same person count as YOLO. "
                      "Partner is fully occluded; pose cannot recover them either.")


if __name__ == "__main__":
    main()
