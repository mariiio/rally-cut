#!/usr/bin/env python3
"""Visualize what the player matching model sees.

Shows player crops, clothing regions, masked regions, and similarity matrices
for a given video. Helps understand why same-team players can be hard to tell apart.

Usage:
    uv run python scripts/visualize_player_appearance.py <video-id>
    uv run python scripts/visualize_player_appearance.py <video-id> --rally-index 3
    uv run python scripts/visualize_player_appearance.py <video-id> --output viz.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add analysis root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
from rallycut.tracking.player_features import (
    CLOTHING_SKIN_LOWER,
    CLOTHING_SKIN_UPPER,
    SKIN_HSV_LOWER,
    SKIN_HSV_UPPER,
    TrackAppearanceStats,
    _build_clothing_mask,
    _extract_hs_histogram,
    _extract_skin_tone,
    _extract_v_histogram,
    compute_track_similarity,
    extract_appearance_features,
)


def extract_crops_for_rally(
    video_path: Path,
    rally,
    num_frames: int = 5,
) -> dict[int, list[tuple[np.ndarray, np.ndarray, dict]]]:
    """Extract player crops and feature visualizations for a rally.

    Returns:
        Dict mapping track_id -> list of (crop_bgr, annotated_crop, feature_info).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get primary track IDs
    primary_ids = set(rally.primary_track_ids or [])
    if not primary_ids:
        # Fallback: find most common track IDs
        from collections import Counter

        track_counts = Counter(p.track_id for p in rally.positions)
        primary_ids = {tid for tid, _ in track_counts.most_common(4)}

    # Group positions by frame
    frames_by_track: dict[int, list] = {}
    for pos in rally.positions:
        if pos.track_id in primary_ids:
            frames_by_track.setdefault(pos.track_id, []).append(pos)

    # Sample frames evenly for each track
    results: dict[int, list[tuple[np.ndarray, np.ndarray, dict]]] = {}

    for track_id, positions in frames_by_track.items():
        if len(positions) < 3:
            continue

        # Sample evenly across track lifetime
        indices = np.linspace(0, len(positions) - 1, num_frames, dtype=int)
        sampled = [positions[i] for i in indices]

        track_crops: list[tuple[np.ndarray, np.ndarray, dict]] = []

        for pos in sampled:
            # Seek to frame
            rally_start_ms = rally.start_ms
            frame_ms = rally_start_ms + (pos.frame_number / fps) * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_ms)
            ret, frame = cap.read()
            if not ret:
                continue

            # Find bbox for this position (need width/height from positions)
            # Positions store (x, y) center — look for bbox in same frame
            bbox_w = pos.width if hasattr(pos, "width") and pos.width else 0.05
            bbox_h = pos.height if hasattr(pos, "height") and pos.height else 0.15

            cx, cy = pos.x, pos.y

            # Convert to pixel coordinates
            x1 = int((cx - bbox_w / 2) * frame_width)
            y1 = int((cy - bbox_h / 2) * frame_height)
            x2 = int((cx + bbox_w / 2) * frame_width)
            y2 = int((cy + bbox_h / 2) * frame_height)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width, x2)
            y2 = min(frame_height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            # Build annotated version showing what model sees
            annotated = _annotate_crop(crop)

            # Extract features for info display
            features = extract_appearance_features(
                frame, track_id, pos.frame_number,
                (cx, cy, bbox_w, bbox_h),
                frame_width, frame_height,
            )

            info = {
                "skin_tone": features.skin_tone_hsv,
                "dominant_color": features.dominant_color_hsv,
                "has_upper_hist": features.upper_body_hist is not None,
                "has_lower_hist": features.lower_body_hist is not None,
                "skin_pixels": features.skin_pixel_count,
            }

            track_crops.append((crop, annotated, info))

        if track_crops:
            results[track_id] = track_crops

    cap.release()
    return results


def _annotate_crop(crop: np.ndarray) -> np.ndarray:
    """Create an annotated version of a crop showing clothing/skin regions."""
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    clothing_mask = _build_clothing_mask(hsv)

    # Create visualization: original with colored overlays
    viz = crop.copy()

    # Draw region boundaries
    upper_top = int(h * 0.20)
    upper_bottom = int(h * 0.55)
    lower_top = int(h * 0.50)
    lower_bottom = int(h * 0.78)
    skin_bottom = int(h * 0.40)

    # Skin region: blue tint
    skin_mask = cv2.inRange(hsv[:skin_bottom], SKIN_HSV_LOWER, SKIN_HSV_UPPER)
    skin_overlay = np.zeros_like(viz[:skin_bottom])
    skin_overlay[skin_mask > 0] = [255, 100, 0]  # blue
    viz[:skin_bottom] = cv2.addWeighted(viz[:skin_bottom], 0.7, skin_overlay, 0.3, 0)

    # Upper clothing region: green tint on masked pixels
    upper_mask_region = clothing_mask[upper_top:upper_bottom]
    upper_overlay = np.zeros_like(viz[upper_top:upper_bottom])
    upper_overlay[upper_mask_region > 0] = [0, 200, 0]  # green
    viz[upper_top:upper_bottom] = cv2.addWeighted(
        viz[upper_top:upper_bottom], 0.7, upper_overlay, 0.3, 0,
    )

    # Lower clothing region: red tint on masked pixels (most important - 35% weight)
    lower_mask_region = clothing_mask[lower_top:lower_bottom]
    lower_overlay = np.zeros_like(viz[lower_top:lower_bottom])
    lower_overlay[lower_mask_region > 0] = [0, 0, 255]  # red
    viz[lower_top:lower_bottom] = cv2.addWeighted(
        viz[lower_top:lower_bottom], 0.7, lower_overlay, 0.3, 0,
    )

    # Draw horizontal lines for region boundaries
    line_color = (255, 255, 255)
    cv2.line(viz, (0, skin_bottom), (w, skin_bottom), line_color, 1)
    cv2.line(viz, (0, upper_top), (w, upper_top), line_color, 1)
    cv2.line(viz, (0, upper_bottom), (w, upper_bottom), line_color, 1)
    cv2.line(viz, (0, lower_top), (w, lower_top), line_color, 1)
    cv2.line(viz, (0, lower_bottom), (w, lower_bottom), line_color, 1)

    return viz


def build_similarity_matrix(
    track_stats: dict[int, TrackAppearanceStats],
) -> tuple[list[int], np.ndarray]:
    """Build pairwise similarity cost matrix."""
    track_ids = sorted(track_stats.keys())
    n = len(track_ids)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                cost = compute_track_similarity(
                    track_stats[track_ids[i]],
                    track_stats[track_ids[j]],
                )
                matrix[i, j] = cost

    return track_ids, matrix


def build_composite_image(
    crops_by_track: dict[int, list[tuple[np.ndarray, np.ndarray, dict]]],
    track_stats: dict[int, TrackAppearanceStats],
    rally_index: int,
    court_split_y: float | None = None,
) -> np.ndarray:
    """Build a composite image showing crops, annotations, and similarity matrix."""
    track_ids = sorted(crops_by_track.keys())
    n_tracks = len(track_ids)

    if n_tracks == 0:
        raise ValueError("No tracks to visualize")

    # Target crop size for display
    crop_h, crop_w = 200, 80

    # Resize all crops to uniform size
    def resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (crop_w, crop_h), interpolation=cv2.INTER_AREA)

    # Layout: for each track, show 5 original crops + 5 annotated crops
    n_samples = min(5, min(len(crops_by_track[tid]) for tid in track_ids))

    # Calculate image dimensions
    pad = 10
    label_h = 30
    section_gap = 20

    # Section 1: Original crops (n_tracks rows x n_samples cols)
    crops_section_w = n_samples * (crop_w + pad) + pad
    crops_section_h = n_tracks * (crop_h + label_h + pad) + pad

    # Section 2: Annotated crops (same size)
    annotated_section_w = crops_section_w
    annotated_section_h = crops_section_h

    # Section 3: Similarity matrix
    cell_size = 60
    matrix_w = (n_tracks + 1) * cell_size
    matrix_h = (n_tracks + 1) * cell_size + label_h

    total_w = pad + crops_section_w + section_gap + annotated_section_w + section_gap + matrix_w + pad
    total_h = max(crops_section_h, matrix_h) + 80  # extra for title

    # Create canvas
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 40  # dark gray bg

    # Title
    cv2.putText(
        canvas, f"Rally {rally_index} - Player Appearance Analysis",
        (pad, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
    )
    y_start = 50

    # Team info
    if court_split_y is not None:
        near_tracks = []
        far_tracks = []
        for tid in track_ids:
            crops_list = crops_by_track[tid]
            if crops_list:
                # Use first crop's info for position
                avg_y = np.mean([
                    p.y for p in []  # placeholder
                ])
        cv2.putText(
            canvas, f"Court split Y: {court_split_y:.2f}",
            (pad, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

    # Draw original crops
    cv2.putText(
        canvas, "Raw Crops",
        (pad, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )
    for i, tid in enumerate(track_ids):
        row_y = y_start + i * (crop_h + label_h + pad)

        # Track label
        cv2.putText(
            canvas, f"T{tid}",
            (pad, row_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        for j in range(n_samples):
            if j < len(crops_by_track[tid]):
                crop_orig, _, info = crops_by_track[tid][j]
                resized = resize(crop_orig)
                x_pos = pad + j * (crop_w + pad)
                y_pos = row_y + label_h
                canvas[y_pos:y_pos + crop_h, x_pos:x_pos + crop_w] = resized

    # Draw annotated crops
    x_offset = pad + crops_section_w + section_gap
    cv2.putText(
        canvas, "Model View (blue=skin, green=upper, red=lower/shorts)",
        (x_offset, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
    )
    for i, tid in enumerate(track_ids):
        row_y = y_start + i * (crop_h + label_h + pad)

        cv2.putText(
            canvas, f"T{tid}",
            (x_offset, row_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        for j in range(n_samples):
            if j < len(crops_by_track[tid]):
                _, annotated, info = crops_by_track[tid][j]
                resized = resize(annotated)
                x_pos = x_offset + j * (crop_w + pad)
                y_pos = row_y + label_h
                canvas[y_pos:y_pos + crop_h, x_pos:x_pos + crop_w] = resized

                # Show dominant color as small swatch
                if info.get("dominant_color") and j == 0:
                    dc = info["dominant_color"]
                    # Convert HSV to BGR for display
                    hsv_pixel = np.array([[[dc[0], dc[1], dc[2]]]], dtype=np.uint8)
                    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
                    color = tuple(int(c) for c in bgr_pixel[0, 0])
                    swatch_x = x_offset + n_samples * (crop_w + pad) + 5
                    swatch_y = row_y + label_h + 5
                    cv2.rectangle(canvas, (swatch_x, swatch_y),
                                  (swatch_x + 30, swatch_y + 30), color, -1)
                    cv2.rectangle(canvas, (swatch_x, swatch_y),
                                  (swatch_x + 30, swatch_y + 30), (255, 255, 255), 1)
                    cv2.putText(
                        canvas, "dom.",
                        (swatch_x, swatch_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1,
                    )

    # Draw similarity matrix
    matrix_x = x_offset + annotated_section_w + section_gap
    cv2.putText(
        canvas, "Pairwise Cost (lower=more similar)",
        (matrix_x, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
    )

    if len(track_stats) >= 2:
        matrix_track_ids, matrix = build_similarity_matrix(track_stats)
        matrix_y = y_start + label_h

        # Column headers
        for j, tid in enumerate(matrix_track_ids):
            cv2.putText(
                canvas, f"T{tid}",
                (matrix_x + (j + 1) * cell_size + 10, matrix_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )

        for i, tid_i in enumerate(matrix_track_ids):
            row_y = matrix_y + i * cell_size

            # Row header
            cv2.putText(
                canvas, f"T{tid_i}",
                (matrix_x + 5, row_y + cell_size // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )

            for j, tid_j in enumerate(matrix_track_ids):
                cell_x = matrix_x + (j + 1) * cell_size
                cell_y = row_y

                cost = matrix[i, j]

                if i == j:
                    # Diagonal — skip
                    color = (60, 60, 60)
                    cv2.rectangle(canvas, (cell_x, cell_y),
                                  (cell_x + cell_size - 2, cell_y + cell_size - 2),
                                  color, -1)
                    cv2.putText(canvas, "-", (cell_x + 22, cell_y + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                else:
                    # Color: green (high cost = different) to red (low cost = similar)
                    # Low cost = hard to distinguish = red (bad)
                    # High cost = easy to distinguish = green (good)
                    r = int(max(0, min(255, (1 - cost) * 2 * 255)))
                    g = int(max(0, min(255, cost * 2 * 255)))
                    color = (0, g, r)

                    cv2.rectangle(canvas, (cell_x, cell_y),
                                  (cell_x + cell_size - 2, cell_y + cell_size - 2),
                                  color, -1)
                    text = f"{cost:.2f}"
                    cv2.putText(canvas, text, (cell_x + 5, cell_y + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Legend
        legend_y = matrix_y + len(matrix_track_ids) * cell_size + 20
        cv2.putText(canvas, "Green = easy to tell apart",
                    (matrix_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        cv2.putText(canvas, "Red = hard to tell apart (same-team?)",
                    (matrix_x, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize player appearance features")
    parser.add_argument("video_id", help="Video ID to analyze")
    parser.add_argument("--rally-index", type=int, default=2,
                        help="Rally index to visualize (default: 2, skip first)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output image path (default: opens in window)")
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Number of frames to sample per track")
    args = parser.parse_args()

    print(f"Loading video {args.video_id}...")
    video_path = get_video_path(args.video_id)
    if video_path is None:
        print(f"ERROR: Video {args.video_id} not found")
        sys.exit(1)

    print(f"Video path: {video_path}")

    rallies = load_rallies_for_video(args.video_id)
    if not rallies:
        print("ERROR: No tracked rallies found")
        sys.exit(1)

    print(f"Found {len(rallies)} rallies")

    rally_idx = min(args.rally_index, len(rallies) - 1)
    rally = rallies[rally_idx]
    print(f"Analyzing rally {rally_idx} (ID: {rally.rally_id})")

    # Extract crops
    print("Extracting player crops...")
    crops_by_track = extract_crops_for_rally(
        video_path, rally, num_frames=args.num_frames,
    )

    if not crops_by_track:
        print("ERROR: No crops extracted")
        sys.exit(1)

    print(f"Extracted crops for {len(crops_by_track)} tracks: {sorted(crops_by_track.keys())}")

    # Build track appearance stats from extracted features
    print("Computing appearance features...")
    cap = cv2.VideoCapture(str(video_path))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    track_stats: dict[int, TrackAppearanceStats] = {}
    primary_ids = set(rally.primary_track_ids or [])
    if not primary_ids:
        from collections import Counter
        track_counts = Counter(p.track_id for p in rally.positions)
        primary_ids = {tid for tid, _ in track_counts.most_common(4)}

    # Sample more frames for stats
    for track_id in crops_by_track:
        positions = [p for p in rally.positions if p.track_id == track_id]
        if len(positions) < 3:
            continue

        indices = np.linspace(0, len(positions) - 1, 12, dtype=int)
        stats = TrackAppearanceStats(track_id=track_id)

        for idx in indices:
            pos = positions[idx]
            frame_ms = rally.start_ms + (pos.frame_number / fps) * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_ms)
            ret, frame = cap.read()
            if not ret:
                continue

            bbox_w = pos.width if hasattr(pos, "width") and pos.width else 0.05
            bbox_h = pos.height if hasattr(pos, "height") and pos.height else 0.15

            features = extract_appearance_features(
                frame, track_id, pos.frame_number,
                (pos.x, pos.y, bbox_w, bbox_h),
                frame_width, frame_height,
            )
            stats.features.append(features)

        stats.compute_averages()
        track_stats[track_id] = stats

    cap.release()

    # Print similarity matrix
    print("\n=== Pairwise Similarity Costs ===")
    print("(lower = more similar, harder to distinguish)")
    track_ids_sorted = sorted(track_stats.keys())
    header = "      " + "  ".join(f"T{tid:>3}" for tid in track_ids_sorted)
    print(header)
    for tid_i in track_ids_sorted:
        row = f"T{tid_i:>3}  "
        for tid_j in track_ids_sorted:
            if tid_i == tid_j:
                row += "   -  "
            else:
                cost = compute_track_similarity(track_stats[tid_i], track_stats[tid_j])
                row += f" {cost:.3f}"
        print(row)

    # Identify same-team pairs (low cost)
    print("\n=== Team Analysis ===")
    costs = []
    for i, tid_i in enumerate(track_ids_sorted):
        for j, tid_j in enumerate(track_ids_sorted):
            if i < j:
                cost = compute_track_similarity(track_stats[tid_i], track_stats[tid_j])
                costs.append((cost, tid_i, tid_j))

    costs.sort()
    print("Most similar pairs (likely same-team):")
    for cost, t1, t2 in costs[:2]:
        print(f"  T{t1} <-> T{t2}: cost={cost:.3f}")
    if len(costs) > 2:
        print("Most different pairs (likely cross-team):")
        for cost, t1, t2 in costs[-2:]:
            print(f"  T{t1} <-> T{t2}: cost={cost:.3f}")

    # Build composite image
    print("\nBuilding visualization...")
    composite = build_composite_image(
        crops_by_track, track_stats, rally_idx,
        court_split_y=rally.court_split_y,
    )

    if args.output:
        output_path = Path(args.output)
        cv2.imwrite(str(output_path), composite)
        print(f"Saved to {output_path}")
    else:
        output_path = Path(f"/tmp/player_appearance_{args.video_id}_rally{rally_idx}.png")
        cv2.imwrite(str(output_path), composite)
        print(f"Saved to {output_path}")
        # Try to open
        import subprocess
        subprocess.run(["open", str(output_path)], check=False)


if __name__ == "__main__":
    main()
