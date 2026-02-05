"""Feature extraction and caching for temporal modeling.

Extracts VideoMAE encoder features from video windows and caches them
for efficient training and inference of temporal models.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from platformdirs import user_cache_dir

from rallycut.core.video import Video

if TYPE_CHECKING:
    from lib.volleyball_ml.video_mae import GameStateClassifier


# Feature dimension for VideoMAE base model
FEATURE_DIM = 768
# Default window size for VideoMAE
WINDOW_SIZE = 16


@dataclass
class FeatureMetadata:
    """Metadata for cached features."""

    video_id: str
    content_hash: str
    stride: int
    num_windows: int
    fps: float
    duration_seconds: float
    feature_dim: int = FEATURE_DIM


def _get_cache_dir() -> Path:
    """Get the cache directory for features."""
    cache_dir = Path(user_cache_dir("rallycut")) / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _compute_cache_key(content_hash: str, stride: int) -> str:
    """Compute a unique cache key for video features.

    Args:
        content_hash: Video content hash (SHA-256).
        stride: Frame stride used for extraction.

    Returns:
        Cache key string.
    """
    key_data = f"{content_hash}:stride={stride}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


class FeatureCache:
    """Cache for extracted VideoMAE features.

    Features are cached as .npy files with metadata in companion .json files.
    The cache key is based on video content hash and extraction stride.

    Usage:
        cache = FeatureCache()

        # Check if features exist
        if cache.has(video_hash, stride=8):
            features, metadata = cache.get(video_hash, stride=8)
        else:
            features = extract_features_for_video(video_path, stride=8)
            cache.put(video_hash, stride=8, features, metadata)
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize feature cache.

        Args:
            cache_dir: Custom cache directory. If None, uses default.
        """
        self.cache_dir = cache_dir or _get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_paths(self, content_hash: str, stride: int) -> tuple[Path, Path]:
        """Get paths for feature array and metadata files."""
        cache_key = _compute_cache_key(content_hash, stride)
        feature_path = self.cache_dir / f"{cache_key}.npy"
        metadata_path = self.cache_dir / f"{cache_key}.json"
        return feature_path, metadata_path

    def has(self, content_hash: str, stride: int) -> bool:
        """Check if features are cached for a video.

        Args:
            content_hash: Video content hash.
            stride: Frame stride used for extraction.

        Returns:
            True if features are cached.
        """
        feature_path, metadata_path = self._get_paths(content_hash, stride)
        return feature_path.exists() and metadata_path.exists()

    def get(self, content_hash: str, stride: int) -> tuple[np.ndarray, FeatureMetadata] | None:
        """Get cached features for a video.

        Args:
            content_hash: Video content hash.
            stride: Frame stride used for extraction.

        Returns:
            Tuple of (features array, metadata) or None if not cached.
        """
        feature_path, metadata_path = self._get_paths(content_hash, stride)

        if not feature_path.exists() or not metadata_path.exists():
            return None

        try:
            features = np.load(feature_path)
            with open(metadata_path) as f:
                meta_dict = json.load(f)
            metadata = FeatureMetadata(**meta_dict)
            return features, metadata
        except (ValueError, json.JSONDecodeError, OSError):
            # Corrupted cache, remove it
            feature_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            return None

    def put(
        self,
        content_hash: str,
        stride: int,
        features: np.ndarray,
        metadata: FeatureMetadata,
    ) -> None:
        """Cache features for a video.

        Args:
            content_hash: Video content hash.
            stride: Frame stride used for extraction.
            features: Feature array of shape (num_windows, feature_dim).
            metadata: Feature metadata.
        """
        feature_path, metadata_path = self._get_paths(content_hash, stride)

        # Save features
        np.save(feature_path, features)

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

    def clear(self) -> int:
        """Clear all cached features.

        Returns:
            Number of cache entries deleted.
        """
        count = 0
        for path in self.cache_dir.glob("*.npy"):
            path.unlink()
            count += 1
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
        return count


def iter_frame_windows(
    video: Video,
    stride: int = 8,
    window_size: int = WINDOW_SIZE,
    subsample_factor: int = 1,
) -> Iterator[tuple[int, list[np.ndarray]]]:
    """Iterate over frame windows from a video.

    Args:
        video: Video wrapper.
        stride: Frame stride between windows.
        window_size: Number of frames per window.
        subsample_factor: FPS subsampling (e.g., 2 for 60fps -> 30fps).

    Yields:
        Tuples of (start_frame, frames) where frames is a list of BGR frames.
    """
    frame_buffer: list[np.ndarray] = []
    logical_idx = 0
    next_window_start = 0  # Next window should start at frame 0

    for frame_idx, frame in video.iter_frames():
        # Subsample high-FPS videos
        if subsample_factor > 1 and frame_idx % subsample_factor != 0:
            continue

        frame_buffer.append(frame)

        # Check if we have enough frames to emit a window starting at next_window_start
        # Window covers frames [next_window_start, next_window_start + window_size)
        # We can emit when logical_idx >= next_window_start + window_size - 1
        if logical_idx >= next_window_start + window_size - 1:
            # Extract the window (last window_size frames in buffer)
            window_frames = frame_buffer[-window_size:]
            yield next_window_start, window_frames

            # Schedule next window
            next_window_start += stride

            # Trim buffer - keep only frames that might be needed for next window
            # We need frames from next_window_start onwards
            frames_to_discard = next_window_start - (logical_idx - len(frame_buffer) + 1)
            if frames_to_discard > 0:
                frame_buffer = frame_buffer[frames_to_discard:]

        logical_idx += 1


def extract_features_for_video(
    video_path: str | Path,
    classifier: GameStateClassifier,
    stride: int = 8,
    batch_size: int = 8,
    pooling: str = "cls",
) -> tuple[np.ndarray, FeatureMetadata]:
    """Extract VideoMAE encoder features from a video.

    Features are extracted at the specified stride using a sliding window
    of 16 frames. High-FPS videos (>40fps) are automatically subsampled
    to ~30fps for consistent temporal resolution.

    Args:
        video_path: Path to video file.
        classifier: GameStateClassifier instance with loaded model.
        stride: Frame stride between windows (default 8 for fine resolution).
        batch_size: Number of windows to process at once.
        pooling: Feature pooling method ("cls" or "mean").

    Returns:
        Tuple of (features array, metadata).
        Features array has shape (num_windows, 768).
    """
    from rallycut.core.proxy import ProxyGenerator

    video_path = Path(video_path)
    video = Video(video_path)

    # Determine FPS subsampling for high-FPS videos
    fps = video.info.fps or 30.0
    subsample_factor = 2 if fps > ProxyGenerator.FPS_NORMALIZE_THRESHOLD else 1
    effective_fps = fps / subsample_factor

    # Collect all windows
    all_features: list[np.ndarray] = []
    window_batch: list[list[np.ndarray]] = []

    for start_frame, frames in iter_frame_windows(
        video, stride=stride, subsample_factor=subsample_factor
    ):
        window_batch.append(frames)

        if len(window_batch) >= batch_size:
            # Process batch
            batch_features = classifier.get_encoder_features_batch(window_batch, pooling=pooling)
            all_features.append(batch_features)
            window_batch = []

    # Process remaining windows
    if window_batch:
        batch_features = classifier.get_encoder_features_batch(window_batch, pooling=pooling)
        all_features.append(batch_features)

    # Combine all features
    if all_features:
        features = np.concatenate(all_features, axis=0)
    else:
        features = np.zeros((0, FEATURE_DIM), dtype=np.float32)

    # Create metadata
    metadata = FeatureMetadata(
        video_id=video_path.stem,
        content_hash="",  # Will be set by caller
        stride=stride,
        num_windows=len(features),
        fps=effective_fps,
        duration_seconds=video.info.duration or 0.0,
    )

    return features, metadata


def load_cached_features(
    video_path: str | Path,
    content_hash: str,
    classifier: GameStateClassifier,
    stride: int = 8,
    cache: FeatureCache | None = None,
) -> tuple[np.ndarray, FeatureMetadata]:
    """Load or extract and cache features for a video.

    Args:
        video_path: Path to video file.
        content_hash: Video content hash for caching.
        classifier: GameStateClassifier instance.
        stride: Frame stride for extraction.
        cache: Feature cache instance. If None, uses default cache.

    Returns:
        Tuple of (features array, metadata).
    """
    if cache is None:
        cache = FeatureCache()

    # Check cache first
    cached = cache.get(content_hash, stride)
    if cached is not None:
        return cached

    # Extract features
    features, metadata = extract_features_for_video(video_path, classifier, stride=stride)
    metadata.content_hash = content_hash

    # Cache for future use
    cache.put(content_hash, stride, features, metadata)

    return features, metadata


def subsample_features(
    features: np.ndarray,
    fine_stride: int,
    coarse_stride: int,
) -> np.ndarray:
    """Subsample features from fine stride to coarse stride.

    This allows caching at fine stride and using coarse stride for inference.

    Args:
        features: Feature array of shape (num_windows, feature_dim).
        fine_stride: The stride at which features were extracted.
        coarse_stride: The target coarse stride.

    Returns:
        Subsampled features array.
    """
    if coarse_stride <= fine_stride:
        return features

    ratio = coarse_stride // fine_stride
    return features[::ratio]
