#!/usr/bin/env python3
"""Quick test script to detect match vs dead time segments."""

import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from rallycut.core.video import Video
from rallycut.core.config import get_config, set_config
from rallycut.analysis.game_state import GameStateAnalyzer

def main():
    video_path = Path.home() / "Desktop" / "match.mp4"

    print(f"Loading video: {video_path}")

    # Use 25% scale for very fast testing
    vid = Video(video_path, scale_factor=0.25)
    info = vid.info
    print(f"Video: {info.width}x{info.height} @ {info.fps:.2f}fps, {info.duration:.1f}s")

    # Configure device
    config = get_config()
    config.use_gpu = True
    set_config(config)
    device = config.get_device()
    print(f"Device: {device}")

    # Very large stride for quick test (analyze every ~2 seconds)
    stride = 64

    analyzer = GameStateAnalyzer(
        device=device,
        min_confidence=0.5,
        stride=stride,
    )

    print(f"\nAnalyzing with stride={stride} (every ~{stride/info.fps:.1f}s)...")

    # Only analyze first 2 minutes for quick test
    max_frames = int(120 * info.fps)  # 2 minutes

    # Get first chunk only
    chunks = list(vid.iter_chunks(chunk_duration_seconds=120))
    if not chunks:
        print("No chunks found")
        return

    chunk = chunks[0]
    print(f"Analyzing chunk: frames {chunk.start_frame}-{chunk.end_frame}")

    results = analyzer.analyze_chunk(chunk, stride=stride)

    print(f"\nFound {len(results)} classification windows:")
    for r in results[:20]:  # Show first 20
        time_start = r.start_frame / info.fps
        time_end = r.end_frame / info.fps
        print(f"  {time_start:6.1f}s - {time_end:6.1f}s: {r.state.name:8s} ({r.confidence:.2f})")

    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more")

    # Get segments
    segments = analyzer.get_segments(results, info.fps)
    print(f"\nMerged into {len(segments)} segments:")
    for seg in segments:
        print(f"  {seg.start_time:6.1f}s - {seg.end_time:6.1f}s ({seg.duration:5.1f}s): {seg.state.name}")

if __name__ == "__main__":
    main()
