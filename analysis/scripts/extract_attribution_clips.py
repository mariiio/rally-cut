"""Extract per-rally MP4 clips for the attribution viewer.

Reads the locked baseline JSON + fixture→video_id map, downloads each video's
proxy (480p) via VideoResolver, and uses ffmpeg to cut rally clips. Writes to
``reports/attribution_audit/{fixture}/clips/{rally_id}.mp4``.

Idempotent: skips clips already extracted.

Usage:
    uv run python scripts/extract_attribution_clips.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.video_resolver import VideoResolver

BASELINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "baseline_2026_04_24.json"
)
FIXTURE_MAP_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "fixture_video_ids_2026_04_24.json"
)
OUT_ROOT = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_audit"
)


def _extract_clip(src: Path, start_ms: int, end_ms: int, out: Path) -> None:
    """ffmpeg trim + faststart + force 30fps output so the HTML overlay
    (frame = round(time * 30)) stays synchronized with tracking frameNumbers.
    Tracker runs on 30fps proxies; some source proxies are 60fps and we
    downsample at clip time."""
    out.parent.mkdir(parents=True, exist_ok=True)
    duration_s = max((end_ms - start_ms) / 1000.0, 0.1)
    start_s = start_ms / 1000.0
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(src),
        "-t", f"{duration_s:.3f}",
        "-r", "30",  # force output 30fps regardless of source
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-an",
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    fixture_map = json.loads(FIXTURE_MAP_PATH.read_text())["fixtures"]
    baseline = json.loads(BASELINE_PATH.read_text())
    rallies = baseline["rallies"]

    # Pull proxy_s3_key per video from DB
    video_proxy: dict[str, tuple[str, str]] = {}  # video_id -> (proxy_s3_key, content_hash)
    with get_connection() as conn, conn.cursor() as cur:
        for name, info in fixture_map.items():
            vid = info["video_id"]
            cur.execute(
                "SELECT proxy_s3_key, content_hash FROM videos WHERE id = %s",
                (vid,),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                print(f"[{name}] MISSING proxy_s3_key, skipping")
                continue
            video_proxy[vid] = (row[0], row[1])
            print(f"[{name}] proxy={row[0]}")

    resolver = VideoResolver()
    # Download videos (content-hash cached)
    video_local: dict[str, Path] = {}
    for vid, (s3_key, content_hash) in video_proxy.items():
        # Use a different cache key for proxy to avoid collisions with full video
        # VideoResolver caches by content_hash; same hash for both proxy + full
        # would conflict. Use ext suffix trick: cache in a side dir.
        cache = resolver.cache_dir / f"{content_hash}_proxy.mp4"
        if cache.exists():
            print(f"  cached proxy: {vid[:8]}")
        else:
            t0 = time.time()
            print(f"  downloading proxy: {vid[:8]}...")
            resolver.s3.download_file(resolver.bucket_name, s3_key, str(cache))
            print(f"    {cache.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.1f}s")
        video_local[vid] = cache

    # Extract rally clips
    extracted = 0
    skipped = 0
    for rally in rallies:
        vid = rally["video_id"]
        if vid not in video_local:
            continue
        fx = rally["fixture"]
        rid = rally["rally_id"]
        out_path = OUT_ROOT / fx / "clips" / f"{rid}.mp4"
        if out_path.exists():
            skipped += 1
            continue
        src = video_local[vid]
        start_ms = rally["start_ms"]
        end_ms = rally["end_ms"]
        try:
            _extract_clip(src, start_ms, end_ms, out_path)
            extracted += 1
            if extracted % 5 == 0:
                print(f"  extracted {extracted} clips")
        except subprocess.CalledProcessError as e:
            print(f"  [err] {fx}/{rid[:8]}: {e}")

    print(f"\nDone. extracted={extracted}, skipped (cached)={skipped}")
    print(f"Clips at: {OUT_ROOT}/{{fixture}}/clips/{{rally_id}}.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
