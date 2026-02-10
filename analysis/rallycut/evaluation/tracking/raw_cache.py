"""Cache raw (unfiltered) positions for efficient grid search.

The key insight: YOLO+ByteTrack detection is slow (~seconds per rally),
but the filter pipeline is fast (~milliseconds). By caching raw positions,
we can re-run filtering with different configs without re-running detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


@dataclass
class CachedRallyData:
    """Cached raw data for a rally (before filtering)."""

    rally_id: str
    video_id: str
    raw_positions: list[PlayerPosition]  # Unfiltered YOLO+ByteTrack output
    ball_positions: list[BallPosition]
    video_fps: float
    frame_count: int
    video_width: int = 1920
    video_height: int = 1080
    start_ms: int = 0
    end_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "rally_id": self.rally_id,
            "video_id": self.video_id,
            "raw_positions": [
                {
                    "frame_number": p.frame_number,
                    "track_id": p.track_id,
                    "x": p.x,
                    "y": p.y,
                    "width": p.width,
                    "height": p.height,
                    "confidence": p.confidence,
                }
                for p in self.raw_positions
            ],
            "ball_positions": [
                {
                    "frame_number": bp.frame_number,
                    "x": bp.x,
                    "y": bp.y,
                    "confidence": bp.confidence,
                }
                for bp in self.ball_positions
            ],
            "video_fps": self.video_fps,
            "frame_count": self.frame_count,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedRallyData:
        """Create from dict."""
        raw_positions = [
            PlayerPosition(
                frame_number=p["frame_number"],
                track_id=p["track_id"],
                x=p["x"],
                y=p["y"],
                width=p["width"],
                height=p["height"],
                confidence=p["confidence"],
            )
            for p in data.get("raw_positions", [])
        ]

        ball_positions = [
            BallPosition(
                frame_number=bp["frame_number"],
                x=bp["x"],
                y=bp["y"],
                confidence=bp["confidence"],
            )
            for bp in data.get("ball_positions", [])
        ]

        return cls(
            rally_id=data["rally_id"],
            video_id=data["video_id"],
            raw_positions=raw_positions,
            ball_positions=ball_positions,
            video_fps=data.get("video_fps", 30.0),
            frame_count=data.get("frame_count", 0),
            video_width=data.get("video_width", 1920),
            video_height=data.get("video_height", 1080),
            start_ms=data.get("start_ms", 0),
            end_ms=data.get("end_ms", 0),
        )


@dataclass
class RawPositionCache:
    """Cache for raw tracking positions.

    Stores raw YOLO+ByteTrack output before filtering, enabling fast
    grid search over filter parameters.

    Default cache location: ~/.cache/rallycut/tracking_grid_search/
    """

    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "rallycut" / "tracking_grid_search"
    )

    def __post_init__(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, rally_id: str) -> Path:
        """Get cache file path for a rally.

        Uses a hash of the rally_id to avoid filesystem issues with UUIDs.
        """
        # Use first 16 chars of sha256 hash for shorter filenames
        hash_prefix = hashlib.sha256(rally_id.encode()).hexdigest()[:16]
        return self.cache_dir / f"raw_{hash_prefix}.json"

    def has(self, rally_id: str) -> bool:
        """Check if rally is cached."""
        return self._cache_path(rally_id).exists()

    def get(self, rally_id: str) -> CachedRallyData | None:
        """Get cached rally data.

        Args:
            rally_id: Rally identifier.

        Returns:
            CachedRallyData if found, None otherwise.
        """
        cache_path = self._cache_path(rally_id)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            cached = CachedRallyData.from_dict(data)

            # Verify rally_id matches (collision protection)
            if cached.rally_id != rally_id:
                logger.warning(
                    f"Cache collision: expected {rally_id}, got {cached.rally_id}"
                )
                return None

            return cached
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load cache for {rally_id}: {e}")
            return None

    def put(self, data: CachedRallyData) -> None:
        """Store rally data in cache.

        Args:
            data: Raw rally data to cache.
        """
        cache_path = self._cache_path(data.rally_id)

        try:
            with open(cache_path, "w") as f:
                json.dump(data.to_dict(), f)
            logger.debug(f"Cached raw positions for {data.rally_id[:8]}...")
        except OSError as e:
            logger.warning(f"Failed to cache {data.rally_id}: {e}")

    def delete(self, rally_id: str) -> bool:
        """Delete cached data for a rally.

        Args:
            rally_id: Rally identifier.

        Returns:
            True if deleted, False if not found.
        """
        cache_path = self._cache_path(rally_id)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cached data.

        Returns:
            Number of cache files deleted.
        """
        count = 0
        for cache_file in self.cache_dir.glob("raw_*.json"):
            cache_file.unlink()
            count += 1
        logger.info(f"Cleared {count} cached raw position files")
        return count

    def list_cached(self) -> list[str]:
        """List all cached rally IDs.

        Note: This returns the stored rally_ids, not the hash prefixes.

        Returns:
            List of rally IDs that have cached data.
        """
        rally_ids = []
        for cache_file in self.cache_dir.glob("raw_*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                rally_ids.append(data["rally_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return rally_ids

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats (count, total size, etc.).
        """
        cache_files = list(self.cache_dir.glob("raw_*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "count": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
        }
