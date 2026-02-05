#!/usr/bin/env python3
"""Diagnostic script to verify label alignment for temporal model training."""

import numpy as np
from pathlib import Path
from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.training.sampler import generate_sequence_labels
from rallycut.temporal.features import FeatureCache
from rallycut.core.proxy import ProxyGenerator


def main():
    print("=" * 70)
    print("LABEL ALIGNMENT DIAGNOSTIC")
    print("=" * 70)

    # Load ground truth
    videos = load_evaluation_videos()
    print(f"\nLoaded {len(videos)} videos with ground truth")

    feature_cache = FeatureCache(cache_dir=Path("training_data/features"))
    stride = 48
    window_size = 16

    print("\n" + "=" * 70)
    print("1. WINDOW TIMESTAMP COMPUTATION CHECK")
    print("=" * 70)
    print(f"\nParameters: stride={stride}, window_size={window_size}")
    print(f"At 30fps:")
    print(f"  - Window duration: {window_size/30*1000:.0f}ms ({window_size/30:.2f}s)")
    print(f"  - Stride duration: {stride/30*1000:.0f}ms ({stride/30:.2f}s)")
    print(f"  - Windows per minute: {60*30/stride:.1f}")

    print("\n" + "=" * 70)
    print("2. RALLY FRACTION PER VIDEO")
    print("=" * 70)

    all_labels = []
    video_stats = []

    for video in videos:
        # Get cached features to match window count
        cached = feature_cache.get(video.content_hash, stride)
        if cached is None:
            print(f"  {video.filename}: NO CACHED FEATURES - SKIPPING")
            continue

        features, metadata = cached

        # Use proxy FPS (30fps for high-fps videos)
        original_fps = video.fps or 30.0
        fps = 30.0 if original_fps > ProxyGenerator.FPS_NORMALIZE_THRESHOLD else original_fps

        # Generate labels
        labels = generate_sequence_labels(
            rallies=video.ground_truth_rallies,
            video_duration_ms=video.duration_ms or 0,
            fps=fps,
            stride=stride,
            window_size=window_size,
            labeling_mode="center",
        )

        # Truncate to match feature count
        if len(labels) != len(features):
            min_len = min(len(labels), len(features))
            labels = labels[:min_len]

        rally_fraction = sum(labels) / len(labels) if labels else 0
        all_labels.extend(labels)

        # Calculate expected rally fraction from ground truth
        total_rally_ms = sum(r.end_ms - r.start_ms for r in video.ground_truth_rallies)
        expected_fraction = total_rally_ms / (video.duration_ms or 1)

        video_stats.append({
            'filename': video.filename,
            'num_rallies': len(video.ground_truth_rallies),
            'num_windows': len(labels),
            'rally_windows': sum(labels),
            'rally_fraction': rally_fraction,
            'expected_fraction': expected_fraction,
            'duration_s': (video.duration_ms or 0) / 1000,
            'fps': fps,
            'original_fps': original_fps,
        })

        print(f"  {video.filename[:20]:20s}: "
              f"rallies={len(video.ground_truth_rallies):2d}, "
              f"windows={len(labels):3d}, "
              f"rally_windows={sum(labels):3d} ({rally_fraction*100:5.1f}%), "
              f"expected={expected_fraction*100:5.1f}%, "
              f"fps={fps:.0f}")

    # Overall stats
    total_windows = len(all_labels)
    total_rally = sum(all_labels)
    overall_fraction = total_rally / total_windows if total_windows else 0

    print(f"\n  OVERALL: {total_rally}/{total_windows} windows = {overall_fraction*100:.1f}% RALLY")
    print(f"  Class balance: NO_RALLY={1-overall_fraction:.1%}, RALLY={overall_fraction:.1%}")

    print("\n" + "=" * 70)
    print("3. LABEL INSPECTION AROUND RALLY BOUNDARIES")
    print("=" * 70)

    # Pick a few videos to inspect
    for video in videos[:3]:
        cached = feature_cache.get(video.content_hash, stride)
        if cached is None:
            continue

        features, metadata = cached
        original_fps = video.fps or 30.0
        fps = 30.0 if original_fps > ProxyGenerator.FPS_NORMALIZE_THRESHOLD else original_fps

        labels = generate_sequence_labels(
            rallies=video.ground_truth_rallies,
            video_duration_ms=video.duration_ms or 0,
            fps=fps,
            stride=stride,
            window_size=window_size,
            labeling_mode="center",
        )
        labels = labels[:len(features)]

        print(f"\n  {video.filename}")
        print(f"  Original FPS: {original_fps:.1f}, Proxy FPS: {fps:.1f}")

        # Show first 2 rallies
        for i, rally in enumerate(video.ground_truth_rallies[:2]):
            print(f"\n    Rally {i+1}: {rally.start_ms/1000:.1f}s - {rally.end_ms/1000:.1f}s "
                  f"(duration: {(rally.end_ms-rally.start_ms)/1000:.1f}s)")

            # Find window indices around rally start
            start_frame = int(rally.start_ms / 1000 * fps)
            end_frame = int(rally.end_ms / 1000 * fps)

            start_window = start_frame // stride
            end_window = end_frame // stride

            # Show windows around start boundary
            window_range = range(max(0, start_window - 2), min(len(labels), start_window + 3))
            print(f"    Around START (window {start_window}):")
            for w in window_range:
                window_start_ms = (w * stride / fps) * 1000
                window_center_ms = ((w * stride + window_size//2) / fps) * 1000
                window_end_ms = ((w * stride + window_size) / fps) * 1000
                in_rally = rally.start_ms <= window_center_ms <= rally.end_ms
                label = labels[w] if w < len(labels) else -1
                marker = "<<<" if w == start_window else ""
                status = "MATCH" if (label == 1) == in_rally else "MISMATCH!"
                print(f"      w{w:3d}: {window_start_ms/1000:6.2f}s-{window_end_ms/1000:6.2f}s "
                      f"center={window_center_ms/1000:6.2f}s | label={label} | "
                      f"center_in_rally={in_rally} | {status} {marker}")

    print("\n" + "=" * 70)
    print("4. FEATURE CACHE METADATA CHECK")
    print("=" * 70)

    for video in videos[:3]:
        cached = feature_cache.get(video.content_hash, stride)
        if cached is None:
            continue
        features, metadata = cached
        print(f"\n  {video.filename}")
        print(f"    Features shape: {features.shape}")
        print(f"    Metadata: fps={metadata.fps}, stride={metadata.stride}, "
              f"window_size={metadata.window_size}")
        print(f"    Video duration: {video.duration_ms/1000:.1f}s")
        print(f"    Expected windows: {int(video.duration_ms/1000 * metadata.fps / stride)}")
        print(f"    Actual windows: {len(features)}")


if __name__ == "__main__":
    main()
