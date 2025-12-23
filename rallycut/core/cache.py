"""Analysis caching for RallyCut."""

import hashlib
import json
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_dir

from rallycut.core.models import GameState, TimeSegment


class AnalysisCache:
    """Cache for video analysis results."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(user_cache_dir("rallycut")) / "analysis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_signature(self, video_path: Path) -> str:
        """Generate signature from file properties and partial content hash."""
        stat = video_path.stat()

        # Read first 1MB for content hash
        with open(video_path, "rb") as f:
            first_chunk = f.read(1024 * 1024)

        content_hash = hashlib.md5(first_chunk).hexdigest()[:16]

        # Combine: filename + size + mtime + content hash
        signature = f"{video_path.name}_{stat.st_size}_{int(stat.st_mtime)}_{content_hash}"
        return signature

    def get_cache_key(
        self, video_path: Path, stride: int, quick: bool, proxy: bool = False
    ) -> str:
        """Generate cache key from video signature + analysis settings."""
        file_sig = self._get_file_signature(video_path)
        if quick:
            mode = "quick"
        else:
            mode = f"ml_s{stride}"
            if proxy:
                mode += "_proxy"

        # Hash the combined key for a clean filename
        key_str = f"{file_sig}_{mode}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(
        self, video_path: Path, stride: int, quick: bool, proxy: bool = False
    ) -> Optional[list[TimeSegment]]:
        """Load cached segments if available."""
        try:
            cache_key = self.get_cache_key(video_path, stride, quick, proxy)
            cache_file = self._cache_path(cache_key)

            if not cache_file.exists():
                return None

            with open(cache_file) as f:
                data = json.load(f)

            # Validate cache structure
            if "segments" not in data:
                return None

            # Convert dicts back to TimeSegment objects
            segments = [
                TimeSegment(
                    start_frame=seg["start_frame"],
                    end_frame=seg["end_frame"],
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    state=GameState(seg["state"]),
                )
                for seg in data["segments"]
            ]

            return segments

        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    def set(
        self,
        video_path: Path,
        stride: int,
        quick: bool,
        segments: list[TimeSegment],
        proxy: bool = False,
    ) -> None:
        """Save segments to cache."""
        try:
            cache_key = self.get_cache_key(video_path, stride, quick, proxy)
            cache_file = self._cache_path(cache_key)

            data = {
                "video": str(video_path),
                "stride": stride,
                "quick": quick,
                "segments": [
                    {
                        "start_frame": seg.start_frame,
                        "end_frame": seg.end_frame,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "state": seg.state.value,
                    }
                    for seg in segments
                ],
            }

            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

        except OSError:
            # Silently fail on cache write errors
            pass

    def clear(self) -> int:
        """Clear all cached analysis results. Returns number of files deleted."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass
        return count
