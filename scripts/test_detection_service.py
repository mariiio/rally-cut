#!/usr/bin/env python3
"""Test the detection service locally before deploying to Modal."""

import json
import sys
from pathlib import Path

from rallycut.service import DetectionService, DetectionRequest


def main():
    # Get video path from argument or use default
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
    else:
        print("Usage: python scripts/test_detection_service.py <video_path_or_url>")
        print("\nExample:")
        print("  python scripts/test_detection_service.py /path/to/match.mp4")
        print("  python scripts/test_detection_service.py https://example.com/video.mp4")
        sys.exit(1)

    # Check if it's a local file or URL
    if video_source.startswith("http"):
        video_url = video_source
    else:
        # Convert local path to file:// URL for testing
        local_path = Path(video_source).resolve()
        if not local_path.exists():
            print(f"Error: File not found: {local_path}")
            sys.exit(1)
        video_url = f"file://{local_path}"

    print(f"Testing detection service with: {video_source}")
    print("-" * 60)

    # Create request
    request = DetectionRequest(
        video_url=video_url,
        config={
            "min_play_duration": 5.0,
            "use_proxy": True,
        },
    )

    # Run detection
    service = DetectionService(device="mps")  # Use MPS on Mac, change to "cuda" for GPU

    def progress(pct: float, msg: str):
        print(f"[{pct*100:5.1f}%] {msg}")

    response = service.detect(request, progress_callback=progress)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if response.status == "failed":
        print(f"Error: {response.error}")
        sys.exit(1)

    print(f"\nVideo: {response.video.duration:.1f}s @ {response.video.fps:.0f}fps")
    print(f"Processing time: {response.processing_time_seconds:.1f}s")
    print(f"Speed: {response.video.duration / response.processing_time_seconds:.1f}x realtime")

    print(f"\nStatistics:")
    print(f"  Rallies: {response.statistics.rally_count}")
    print(f"  Play time: {response.statistics.total_play_duration:.1f}s ({response.statistics.play_percentage:.1f}%)")
    print(f"  Dead time: {response.statistics.total_dead_time:.1f}s")
    print(f"  Avg rally: {response.statistics.avg_rally_duration:.1f}s")
    print(f"  Longest: {response.statistics.longest_rally_duration:.1f}s")

    print(f"\nTop 5 Highlights:")
    top_segments = sorted(response.segments, key=lambda s: s.highlight_rank)[:5]
    for seg in top_segments:
        print(f"  #{seg.highlight_rank}: {seg.start_time:.1f}s - {seg.end_time:.1f}s ({seg.duration:.1f}s) score={seg.highlight_score:.2f}")

    # Save full JSON output
    output_file = Path("detection_output.json")
    with open(output_file, "w") as f:
        json.dump(response.model_dump(mode="json"), f, indent=2, default=str)
    print(f"\nFull output saved to: {output_file}")


if __name__ == "__main__":
    main()
