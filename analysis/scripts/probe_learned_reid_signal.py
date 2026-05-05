"""Validate whether the fine-tuned OSNet ReID head can rescue missed merges.

Hypothesis (from the chimera_stitching_dd042609_2026_05_04 investigation
follow-up): the player_filter pipeline's `relink_primary_fragments`
HSV-Bhattacharyya appearance gate (≤ 0.20) is too strict on cases where
a player's appearance changes during a brief occlusion (lighting,
scale, partial visibility), so legitimate same-player fragments stay
unmerged. The learned ReID head (Session-3 SupCon-trained DINOv2+MLP,
128-d) was designed to be more invariant to those nuisance variations.
Question: would using it as a positive merge signal (additional cost
term, not just a veto) reliably distinguish same-player pairs from
different-player pairs on the user-labeled fixtures?

Approach (offline, no production code changes):
  1. For each user-labeled video, load GT samples (frame-level
     `playerId`, `cx`, `cy`).
  2. Match each GT label to the closest post-filter primary track at
     that frame → gives `(rally, post_remap_track_id, gt_pid, frame,
     bbox)` tuples.
  3. Build same-player and different-player TRACK PAIRS:
       - Same-player: two distinct post-remap track ids that GT
         assigns to the same gt_pid in the same rally → these are the
         FAILED MERGES (matcher gave them different PIDs but GT says
         same physical player).
       - Different-player: two distinct post-remap track ids that GT
         assigns to different gt_pids in the same rally → these are
         the CORRECT REJECTIONS (matcher correctly kept them apart).
  4. For each pair, extract bbox crops at sampled frames, run them
     through the learned ReID head, compute pairwise cosine similarity,
     aggregate to a per-pair score (median).
  5. Report cosine distributions for the two pair classes. If they
     separate cleanly (e.g., same-player pairs have median cos > 0.7,
     different-player pairs < 0.5), the learned head can be used as a
     positive merge signal in the production pipeline.

Output: per-pair scores printed as a table + summary statistics. JSON
dump at `analysis/reports/learned_reid_signal/<timestamp>.json` for
post-hoc analysis.

Read-only DB access. No mutation. ~5-10 min compute on a calibrated
panel (4 user-labeled videos, depending on crop count).
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
OUT_DIR = ANALYSIS_ROOT / "reports" / "learned_reid_signal"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("probe_learned_reid_signal")

# Per (track_id, gt_pid) we sample at most this many frames (uniformly
# across the rally) to compute embeddings. More samples → more stable
# median, but more compute. 8 frames balances stability and runtime.
MAX_FRAMES_PER_TRACK = 8

# Per-pair: we compute pairwise cosines between every embedding from
# track A and every embedding from track B, then take the MEDIAN as
# the pair's similarity score. Median is robust to single bad crops.
USE_MEDIAN = True


def _load_gt_samples(video_id: str) -> dict[str, list[dict[str, Any]]]:
    """Returns rally_id → list of {playerId, frame, cx, cy} GT labels."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT player_matching_gt_json FROM videos WHERE id = %s",
            [video_id],
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return {}
    gt = row[0] if isinstance(row[0], dict) else json.loads(cast(str, row[0]))
    out: dict[str, list[dict[str, Any]]] = {}
    rallies = gt.get("rallies", {}) or {}
    for rid, entry in rallies.items():
        labels = entry.get("labels", [])
        out[rid] = [
            {
                "playerId": int(lbl["playerId"]),
                "frame": int(lbl["frame"]),
                "cx": float(lbl["cx"]),
                "cy": float(lbl["cy"]),
            }
            for lbl in labels
            if "playerId" in lbl and "frame" in lbl
        ]
    return out


def _load_positions(video_id: str) -> dict[str, list[dict[str, Any]]]:
    """Returns rally_id → positions_json (post-remap)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT pt.rally_id, pt.positions_json
            FROM player_tracks pt
            JOIN rallies r ON r.id = pt.rally_id
            WHERE r.video_id = %s AND pt.positions_json IS NOT NULL
            """,
            [video_id],
        )
        rows = cur.fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for rid, pj in rows:
        if pj is None:
            continue
        out[str(rid)] = pj if isinstance(pj, list) else json.loads(cast(str, pj))
    return out


def _load_rally_meta(video_id: str) -> dict[str, dict[str, int]]:
    """Returns rally_id → {start_ms, end_ms, video_fps_approx_30 frame_count}."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id, r.start_ms, r.end_ms, pt.frame_count, pt.fps
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id = %s
            """,
            [video_id],
        )
        rows = cur.fetchall()
    return {
        str(rid): {
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "frame_count": int(frame_count) if frame_count is not None else 0,
            "fps": float(fps) if fps is not None else 30.0,
        }
        for rid, start_ms, end_ms, frame_count, fps in rows
    }


def _match_gt_to_track(
    gt_label: dict[str, Any],
    positions: list[dict[str, Any]],
    *,
    frame_window: int = 5,
) -> int | None:
    """Find the post-remap track_id at the labeled frame closest to (cx, cy)."""
    target_frame = gt_label["frame"]
    cx, cy = gt_label["cx"], gt_label["cy"]
    best_pos = None
    best_d = float("inf")
    for delta in range(0, frame_window + 1):
        candidates = [
            p for p in positions
            if (p.get("frameNumber") in {target_frame - delta, target_frame + delta})
        ]
        for p in candidates:
            px = p.get("x", 0.0) + p.get("width", 0.0) / 2.0
            py = p.get("y", 0.0) + p.get("height", 0.0) / 2.0
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d < best_d:
                best_d = d
                best_pos = p
        if best_pos is not None:
            break
    if best_pos is None:
        return None
    tid = best_pos.get("trackId")
    return int(tid) if tid is not None else None


def _sample_track_frames(
    positions: list[dict[str, Any]],
    track_id: int,
    n: int,
) -> list[dict[str, Any]]:
    """Uniformly sample up to n positions for this track."""
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


def _extract_crops(
    video_path: Path,
    rally_start_ms: int,
    rally_fps: float,
    samples: list[dict[str, Any]],
) -> list[np.ndarray]:
    """Open the video and extract BGR crops at the rally-relative frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or rally_fps or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rally_start_frame_in_video = int(rally_start_ms / 1000.0 * video_fps)
        crops: list[np.ndarray] = []
        for s in samples:
            f_in_rally = int(s.get("frameNumber", 0))
            f_in_video = rally_start_frame_in_video + f_in_rally
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_in_video)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            crop = extract_bbox_crop(
                frame,
                (s["x"] + s["width"] / 2, s["y"] + s["height"] / 2,
                 s["width"], s["height"]),
                frame_w, frame_h,
            )
            if crop is not None:
                crops.append(crop)
        return crops
    finally:
        cap.release()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both already L2-normalized


def _pair_cosine(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Aggregate cosine across all (a_i, b_j) pairs. Median is robust to noisy crops."""
    if emb_a.size == 0 or emb_b.size == 0:
        return float("nan")
    scores = []
    for i in range(emb_a.shape[0]):
        for j in range(emb_b.shape[0]):
            scores.append(float(np.dot(emb_a[i], emb_b[j])))
    if not scores:
        return float("nan")
    return float(np.median(scores)) if USE_MEDIAN else float(np.mean(scores))


def _build_track_to_gt_pid(
    gt_samples: list[dict[str, Any]],
    positions: list[dict[str, Any]],
) -> dict[int, set[int]]:
    """For each post-remap track_id, which gt_pids did it carry across GT labels?"""
    out: dict[int, set[int]] = defaultdict(set)
    for lbl in gt_samples:
        tid = _match_gt_to_track(lbl, positions)
        if tid is None:
            continue
        out[tid].add(lbl["playerId"])
    return dict(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos", nargs="+", required=True,
        help="Video IDs to probe (must have player_matching_gt_json).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=MAX_FRAMES_PER_TRACK,
        help="Max frames sampled per track for embedding aggregation.",
    )
    parser.add_argument(
        "--rally-id", type=str, default=None,
        help="Optional: probe only one rally (must match a GT-labeled rally).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = OUT_DIR / f"learned_reid_signal_{timestamp}.json"

    device = _default_device()
    print(f"Using device: {device}")
    print(f"Output: {out_path}")
    print()

    all_results: list[dict[str, Any]] = []

    for vid in args.videos:
        print(f"========== {vid[:8]} ==========")
        gt_per_rally = _load_gt_samples(vid)
        if not gt_per_rally:
            print(f"  No GT labels — skipping.")
            continue
        positions_per_rally = _load_positions(vid)
        rally_meta = _load_rally_meta(vid)
        video_path = get_video_path(vid)
        if video_path is None:
            print(f"  Video file unavailable — skipping.")
            continue

        same_pair_scores: list[dict[str, Any]] = []
        diff_pair_scores: list[dict[str, Any]] = []

        rallies_to_probe = (
            [args.rally_id] if args.rally_id else list(gt_per_rally.keys())
        )
        for rid in rallies_to_probe:
            gt_samples = gt_per_rally.get(rid, [])
            positions = positions_per_rally.get(rid, [])
            meta = rally_meta.get(rid)
            if not gt_samples or not positions or not meta:
                continue

            # For each post-filter track, what gt_pids does it carry?
            track_to_gt_pids = _build_track_to_gt_pid(gt_samples, positions)
            if not track_to_gt_pids:
                continue
            track_ids = sorted(track_to_gt_pids.keys())

            # Extract embeddings for each track once.
            track_embeddings: dict[int, np.ndarray] = {}
            for tid in track_ids:
                samples = _sample_track_frames(positions, tid, args.max_frames)
                if not samples:
                    continue
                crops = _extract_crops(
                    video_path, meta["start_ms"], meta["fps"], samples,
                )
                if not crops:
                    continue
                emb = extract_learned_embeddings(crops, device=device)
                if emb.shape[0] > 0:
                    track_embeddings[tid] = emb

            # Build pair classifications.
            for i, tid_a in enumerate(track_ids):
                if tid_a not in track_embeddings:
                    continue
                gt_a = track_to_gt_pids[tid_a]
                for tid_b in track_ids[i + 1:]:
                    if tid_b not in track_embeddings:
                        continue
                    gt_b = track_to_gt_pids[tid_b]
                    cos = _pair_cosine(
                        track_embeddings[tid_a], track_embeddings[tid_b],
                    )
                    if np.isnan(cos):
                        continue
                    # Same-player iff GT pid sets overlap (i.e., these
                    # two filtered tracks were both labeled the same
                    # physical player at some frame).
                    overlap = gt_a & gt_b
                    pair_record = {
                        "video_id": vid[:8],
                        "rally_id": rid[:8],
                        "track_a": tid_a,
                        "track_b": tid_b,
                        "gt_pids_a": sorted(gt_a),
                        "gt_pids_b": sorted(gt_b),
                        "cosine": cos,
                        "n_frames_a": int(track_embeddings[tid_a].shape[0]),
                        "n_frames_b": int(track_embeddings[tid_b].shape[0]),
                    }
                    if overlap:
                        same_pair_scores.append(pair_record)
                    else:
                        diff_pair_scores.append(pair_record)

        # Per-video summary
        if same_pair_scores or diff_pair_scores:
            print(f"  Same-player pairs (FAILED MERGES per GT): {len(same_pair_scores)}")
            for p in same_pair_scores:
                print(
                    f"    {p['rally_id']} T{p['track_a']}↔T{p['track_b']} "
                    f"cos={p['cosine']:+.3f} (gt_pids: A={p['gt_pids_a']}, B={p['gt_pids_b']})"
                )
            print(f"  Different-player pairs (CORRECT REJECTIONS per GT): {len(diff_pair_scores)}")
            for p in diff_pair_scores:
                print(
                    f"    {p['rally_id']} T{p['track_a']}↔T{p['track_b']} "
                    f"cos={p['cosine']:+.3f} (gt_pids: A={p['gt_pids_a']}, B={p['gt_pids_b']})"
                )
        else:
            print("  No usable pairs found.")
        print()

        all_results.append({
            "video_id": vid,
            "same_pair_scores": same_pair_scores,
            "diff_pair_scores": diff_pair_scores,
        })

    # Cross-video summary
    all_same = [
        s for r in all_results for s in r["same_pair_scores"]
    ]
    all_diff = [
        s for r in all_results for s in r["diff_pair_scores"]
    ]
    print("========== CROSS-VIDEO SUMMARY ==========")
    if all_same:
        same_cos = np.array([p["cosine"] for p in all_same])
        print(f"  Same-player pairs: n={len(all_same)}, "
              f"cos median={np.median(same_cos):+.3f}, "
              f"p10={np.percentile(same_cos, 10):+.3f}, "
              f"p90={np.percentile(same_cos, 90):+.3f}")
    if all_diff:
        diff_cos = np.array([p["cosine"] for p in all_diff])
        print(f"  Different-player pairs: n={len(all_diff)}, "
              f"cos median={np.median(diff_cos):+.3f}, "
              f"p10={np.percentile(diff_cos, 10):+.3f}, "
              f"p90={np.percentile(diff_cos, 90):+.3f}")
    if all_same and all_diff:
        same_cos = np.array([p["cosine"] for p in all_same])
        diff_cos = np.array([p["cosine"] for p in all_diff])
        # Find the best threshold that separates the two distributions.
        best_thresh = None
        best_acc = 0.0
        for t in np.linspace(min(diff_cos.min(), same_cos.min()),
                             max(diff_cos.max(), same_cos.max()), 50):
            same_correct = (same_cos >= t).sum()
            diff_correct = (diff_cos < t).sum()
            acc = (same_correct + diff_correct) / (len(same_cos) + len(diff_cos))
            if acc > best_acc:
                best_acc = acc
                best_thresh = t
        if best_thresh is not None:
            print(f"  Best separating threshold: cos={best_thresh:+.3f} "
                  f"(separates same vs different at {best_acc * 100:.1f}% accuracy)")

    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWrote per-pair details to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
