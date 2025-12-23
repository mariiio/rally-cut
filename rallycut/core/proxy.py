"""Proxy video generation for faster ML analysis.

Creates optimized 480p@15fps proxy videos that are:
- 8x smaller than 1080p@30fps source
- Faster to decode and process
- Cached for reuse across analysis runs
"""

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from platformdirs import user_cache_dir


@dataclass
class ProxyConfig:
    """Configuration for proxy video generation."""

    height: int = 480  # 480p resolution (sufficient for 224x224 ML input)
    fps: Optional[int] = None  # None = keep original FPS (safest for ML accuracy)
    preset: str = "ultrafast"  # FFmpeg H.264 preset for fast decode
    crf: int = 28  # Quality (lower = better, larger file)


class FrameMapper:
    """Maps frame indices and timestamps between source and proxy videos.

    When source is 30fps and proxy is 15fps:
    - Source frame 0 -> Proxy frame 0
    - Source frame 30 -> Proxy frame 15
    - Proxy frame 15 -> Source frame 30
    """

    def __init__(self, source_fps: float, proxy_fps: float):
        self.source_fps = source_fps
        self.proxy_fps = proxy_fps
        self.ratio = source_fps / proxy_fps  # e.g., 2.0 for 30fps->15fps

    def source_to_proxy(self, source_frame: int) -> int:
        """Convert source frame index to proxy frame index."""
        return int(source_frame / self.ratio)

    def proxy_to_source(self, proxy_frame: int) -> int:
        """Convert proxy frame index to source frame index."""
        return int(proxy_frame * self.ratio)

    def time_to_proxy_frame(self, time_seconds: float) -> int:
        """Convert time in seconds to proxy frame index."""
        return int(time_seconds * self.proxy_fps)

    def proxy_frame_to_time(self, proxy_frame: int) -> float:
        """Convert proxy frame index to time in seconds."""
        return proxy_frame / self.proxy_fps

    def time_to_source_frame(self, time_seconds: float) -> int:
        """Convert time in seconds to source frame index."""
        return int(time_seconds * self.source_fps)

    def source_frame_to_time(self, source_frame: int) -> float:
        """Convert source frame index to time in seconds."""
        return source_frame / self.source_fps


class ProxyGenerator:
    """Generates and manages cached proxy videos for ML analysis."""

    def __init__(
        self,
        config: Optional[ProxyConfig] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.config = config or ProxyConfig()
        self.cache_dir = cache_dir or Path(user_cache_dir("rallycut")) / "proxies"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_signature(self, video_path: Path) -> str:
        """Get a signature for the video file (path + size + mtime)."""
        stat = video_path.stat()
        sig_str = f"{video_path.resolve()}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(sig_str.encode()).hexdigest()[:16]

    def get_proxy_path(self, source_video: Path) -> Path:
        """Get the cache path for a source video's proxy."""
        sig = self._get_file_signature(source_video)
        if self.config.fps:
            proxy_name = f"{sig}_{self.config.height}p_{self.config.fps}fps.mp4"
        else:
            proxy_name = f"{sig}_{self.config.height}p.mp4"
        return self.cache_dir / proxy_name

    def proxy_exists(self, source_video: Path) -> bool:
        """Check if a cached proxy exists for the source video."""
        return self.get_proxy_path(source_video).exists()

    def generate_proxy(
        self,
        source_video: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """Generate a proxy video using ffmpeg.

        Args:
            source_video: Path to source video
            progress_callback: Optional callback for progress updates

        Returns:
            Path to generated proxy video

        Raises:
            RuntimeError: If ffmpeg fails
        """
        proxy_path = self.get_proxy_path(source_video)

        # Skip if already exists
        if proxy_path.exists():
            if progress_callback:
                progress_callback(1.0, "Using cached proxy")
            return proxy_path

        if self.config.fps:
            proxy_desc = f"{self.config.height}p@{self.config.fps}fps"
            vf_filter = f"scale=-2:{self.config.height},fps={self.config.fps}"
        else:
            proxy_desc = f"{self.config.height}p"
            vf_filter = f"scale=-2:{self.config.height}"

        if progress_callback:
            progress_callback(0.0, f"Creating {proxy_desc} proxy...")

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(source_video),
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", self.config.preset,
            "-tune", "fastdecode",
            "-crf", str(self.config.crf),
            "-an",  # Strip audio
            "-y",  # Overwrite
            str(proxy_path),
        ]

        try:
            # Run ffmpeg with progress parsing
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for completion (progress tracking could be added here)
            _, stderr = process.communicate()

            if process.returncode != 0:
                # Clean up failed proxy
                proxy_path.unlink(missing_ok=True)
                raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

            if progress_callback:
                progress_callback(1.0, "Proxy created")

            return proxy_path

        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    def get_or_create(
        self,
        source_video: Path,
        source_fps: float,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> tuple[Path, FrameMapper]:
        """Get or create a proxy video with frame mapper.

        This is the main entry point for proxy generation.

        Args:
            source_video: Path to source video
            source_fps: FPS of source video (needed for frame mapping)
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (proxy_path, frame_mapper)
        """
        proxy_path = self.generate_proxy(source_video, progress_callback)
        # If no fps conversion, proxy has same fps as source
        proxy_fps = self.config.fps if self.config.fps else source_fps
        mapper = FrameMapper(source_fps, proxy_fps)
        return proxy_path, mapper

    def cleanup_proxy(self, source_video: Path) -> bool:
        """Remove a specific proxy from cache.

        Returns:
            True if proxy was removed, False if it didn't exist
        """
        proxy_path = self.get_proxy_path(source_video)
        if proxy_path.exists():
            proxy_path.unlink()
            return True
        return False

    def cleanup_all(self) -> int:
        """Clear all cached proxies.

        Returns:
            Number of proxies removed
        """
        count = 0
        for proxy_file in self.cache_dir.glob("*.mp4"):
            proxy_file.unlink()
            count += 1
        return count

    def get_cache_size_mb(self) -> float:
        """Get total size of cached proxies in MB."""
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.mp4"))
        return total_bytes / (1024 * 1024)
