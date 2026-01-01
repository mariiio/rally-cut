"""Proxy video generation for faster ML analysis.

Creates optimized proxy videos that are:
- Normalized to 30fps for optimal VideoMAE temporal dynamics
- Downscaled to 480p (sufficient for 224x224 ML input)
- Faster to decode and process
- Cached for reuse across analysis runs

FPS Normalization Rationale:
VideoMAE's 16-frame window needs ~0.5s of temporal content to recognize patterns.
- At 60fps: 16 frames = 0.27s (temporal dynamics too compressed, poor accuracy)
- At 30fps: 16 frames = 0.53s (matches training data, optimal accuracy)
Normalizing high-FPS videos to 30fps in the proxy reduces file size by ~30%
and decode overhead by ~50%, with no loss in detection accuracy.
"""

import hashlib
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir


@dataclass
class ProxyConfig:
    """Configuration for proxy video generation.

    Default settings are optimized for VideoMAE rally detection:
    - 480p resolution: sufficient for 224x224 ML input
    - 30fps: optimal for VideoMAE's 16-frame temporal window
    - ultrafast preset: prioritizes decode speed over file size
    """

    height: int = 480  # 480p resolution (sufficient for 224x224 ML input)
    fps: int = 30  # Normalize to 30fps for optimal ML temporal dynamics
    preset: str = "ultrafast"  # FFmpeg H.264 preset for fast decode
    crf: int = 24  # Quality (18-28 range, lower = better)
    keyint: int = 30  # Keyframe interval (1 second at 30fps, enables faster seeking)


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

    # Only normalize FPS if source exceeds this threshold
    # Matches the threshold in game_state.py for consistency
    FPS_NORMALIZE_THRESHOLD = 40.0

    def __init__(
        self,
        config: ProxyConfig | None = None,
        cache_dir: Path | None = None,
    ):
        self.config = config or ProxyConfig()
        self.cache_dir = cache_dir or Path(user_cache_dir("rallycut")) / "proxies"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_source_fps(self, video_path: Path) -> float:
        """Get source video FPS using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            fps_str = result.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den)
            return float(fps_str)
        except Exception:
            return 30.0  # Default fallback

    def _get_file_signature(self, video_path: Path) -> str:
        """Get a signature for the video file (path + size + mtime)."""
        stat = video_path.stat()
        sig_str = f"{video_path.resolve()}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(sig_str.encode()).hexdigest()[:16]

    def get_proxy_path(self, source_video: Path) -> Path:
        """Get the cache path for a source video's proxy.

        Note: This detects source FPS to determine the correct cache path.
        For repeated calls, use _get_proxy_path_for_fps directly.
        """
        source_fps = self._get_source_fps(source_video)
        should_normalize = source_fps > self.FPS_NORMALIZE_THRESHOLD
        return self._get_proxy_path_for_fps(source_video, should_normalize)

    def _get_proxy_path_for_fps(self, source_video: Path, fps_normalized: bool) -> Path:
        """Get cache path based on whether FPS was normalized."""
        sig = self._get_file_signature(source_video)
        if fps_normalized:
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
        progress_callback: Callable[[float, str], None] | None = None,
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
        # Detect source FPS to determine if normalization is needed
        source_fps = self._get_source_fps(source_video)
        should_normalize_fps = source_fps > self.FPS_NORMALIZE_THRESHOLD

        proxy_path = self._get_proxy_path_for_fps(source_video, should_normalize_fps)

        # Skip if already exists
        if proxy_path.exists():
            if progress_callback:
                progress_callback(1.0, "Court already set up!")
            return proxy_path

        # Build filter chain
        if should_normalize_fps:
            vf_filter = f"scale=-2:{self.config.height},fps={self.config.fps}"
        else:
            vf_filter = f"scale=-2:{self.config.height}"

        if progress_callback:
            progress_callback(0.0, "Setting up the court...")

        # Build ffmpeg command with optimized settings
        cmd = [
            "ffmpeg",
            "-i", str(source_video),
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", self.config.preset,
            "-tune", "fastdecode",
            "-crf", str(self.config.crf),
            "-g", str(self.config.keyint),  # Keyframe interval for faster seeking
            "-pix_fmt", "yuv420p",  # Ensure compatibility
            "-movflags", "+faststart",  # Optimize for streaming/seeking
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
                progress_callback(1.0, "Court is ready!")

            return proxy_path

        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    def get_or_create(
        self,
        source_video: Path,
        source_fps: float,
        progress_callback: Callable[[float, str], None] | None = None,
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
        # Determine actual proxy FPS based on whether normalization was applied
        should_normalize = source_fps > self.FPS_NORMALIZE_THRESHOLD
        proxy_fps = float(self.config.fps) if should_normalize else source_fps
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
