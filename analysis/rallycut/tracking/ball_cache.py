"""
Persistent cache for ball validation results.

Caches ball validation results per segment to avoid redundant computation
on repeated runs.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_cache_dir

from rallycut.tracking.ball_features import SegmentBallFeatures
from rallycut.tracking.ball_validation import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class CachedValidation:
    """Cached validation result."""

    is_valid: bool
    confidence_adjustment: float
    features: dict | None  # Serialized SegmentBallFeatures

    def to_validation_result(self) -> ValidationResult:
        """Convert to ValidationResult."""
        features = None
        if self.features:
            features = SegmentBallFeatures(**self.features)

        return ValidationResult(
            is_valid=self.is_valid,
            confidence_adjustment=self.confidence_adjustment,
            features=features,
            early_terminated=False,  # Not stored, assume false
            frames_processed=0,  # Not stored
        )


class BallFeatureCache:
    """
    Persistent cache for ball validation results.

    Cache is keyed by video hash + segment bounds, allowing reuse
    across runs with the same video.
    """

    CACHE_VERSION = 1

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files.
                      Defaults to ~/.cache/rallycut/ball_features/
        """
        if cache_dir is None:
            cache_dir = Path(user_cache_dir("rallycut")) / "ball_features"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, CachedValidation] = {}

    def _segment_key(
        self,
        video_hash: str,
        start_time: float,
        end_time: float,
    ) -> str:
        """
        Generate cache key for a segment.

        Key includes segment bounds (rounded to 2 decimal places) for
        cache invalidation when boundaries change.

        Args:
            video_hash: Hash of the video file.
            start_time: Segment start time in seconds.
            end_time: Segment end time in seconds.

        Returns:
            Cache key string.
        """
        key_str = f"v{self.CACHE_VERSION}:{video_hash}:{start_time:.2f}:{end_time:.2f}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use first 2 chars as subdirectory for better filesystem performance
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.json"

    def get(
        self,
        video_hash: str,
        start_time: float,
        end_time: float,
    ) -> ValidationResult | None:
        """
        Get cached validation result.

        Args:
            video_hash: Hash of the video file.
            start_time: Segment start time in seconds.
            end_time: Segment end time in seconds.

        Returns:
            ValidationResult if cached, None otherwise.
        """
        key = self._segment_key(video_hash, start_time, end_time)

        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key].to_validation_result()

        # Check disk cache
        cache_file = self._cache_file(key)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            cached = CachedValidation(
                is_valid=data["is_valid"],
                confidence_adjustment=data["confidence_adjustment"],
                features=data.get("features"),
            )

            # Store in memory cache
            self._memory_cache[key] = cached

            return cached.to_validation_result()

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid cache file {cache_file}: {e}")
            cache_file.unlink(missing_ok=True)
            return None

    def put(
        self,
        video_hash: str,
        start_time: float,
        end_time: float,
        result: ValidationResult,
    ) -> None:
        """
        Store validation result in cache.

        Args:
            video_hash: Hash of the video file.
            start_time: Segment start time in seconds.
            end_time: Segment end time in seconds.
            result: Validation result to cache.
        """
        key = self._segment_key(video_hash, start_time, end_time)

        # Serialize features if present
        features_dict = None
        if result.features:
            features_dict = asdict(result.features)

        cached = CachedValidation(
            is_valid=result.is_valid,
            confidence_adjustment=result.confidence_adjustment,
            features=features_dict,
        )

        # Store in memory cache
        self._memory_cache[key] = cached

        # Store on disk
        cache_file = self._cache_file(key)
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "is_valid": cached.is_valid,
                    "confidence_adjustment": cached.confidence_adjustment,
                    "features": cached.features,
                }, f)
        except OSError as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")

    def has(
        self,
        video_hash: str,
        start_time: float,
        end_time: float,
    ) -> bool:
        """
        Check if validation result is cached.

        Args:
            video_hash: Hash of the video file.
            start_time: Segment start time in seconds.
            end_time: Segment end time in seconds.

        Returns:
            True if cached.
        """
        key = self._segment_key(video_hash, start_time, end_time)

        if key in self._memory_cache:
            return True

        return self._cache_file(key).exists()

    def clear(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of cache entries cleared.
        """
        count = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for cache_file in subdir.glob("*.json"):
                    cache_file.unlink()
                    count += 1

        self._memory_cache.clear()
        return count

    def get_cache_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_bytes = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for cache_file in subdir.glob("*.json"):
                    total_bytes += cache_file.stat().st_size

        return total_bytes / (1024 * 1024)

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        entry_count = 0
        total_bytes = 0

        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for cache_file in subdir.glob("*.json"):
                    entry_count += 1
                    total_bytes += cache_file.stat().st_size

        return {
            "entry_count": entry_count,
            "size_mb": total_bytes / (1024 * 1024),
            "memory_cache_size": len(self._memory_cache),
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance for convenience
_global_cache: BallFeatureCache | None = None


def get_ball_cache() -> BallFeatureCache:
    """Get global ball feature cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = BallFeatureCache()
    return _global_cache
