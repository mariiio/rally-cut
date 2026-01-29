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
import re
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
        return round(source_frame / self.ratio)

    def proxy_to_source(self, proxy_frame: int) -> int:
        """Convert proxy frame index to source frame index."""
        return round(proxy_frame * self.ratio)

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

    # Class-level default (for backward compatibility)
    # Instance uses config.fps_normalize_threshold if available
    FPS_NORMALIZE_THRESHOLD: float = 40.0

    def __init__(
        self,
        config: ProxyConfig | None = None,
        cache_dir: Path | None = None,
    ):
        self.config = config or ProxyConfig()
        self.cache_dir = cache_dir or Path(user_cache_dir("rallycut")) / "proxies"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Use config value, falling back to class constant for backward compatibility
        self._fps_threshold: float = getattr(
            self.config, 'fps_normalize_threshold', self.FPS_NORMALIZE_THRESHOLD
        )

    def _get_video_info(self, video_path: Path) -> tuple[float, int]:
        """Get source video FPS and height in a single ffprobe call.

        Returns:
            Tuple of (fps, height). Defaults to (30.0, 1080) on error.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,height",
            "-of", "csv=p=0",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Output format: "fps_num/fps_den,height" e.g. "30/1,1080"
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                fps_str, height_str = parts[0], parts[1]
                # Parse FPS
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_str)
                height = int(height_str)
                return fps, height
        except Exception:
            pass
        return 30.0, 1080  # Defaults

    def _get_source_fps(self, video_path: Path) -> float:
        """Get source video FPS using ffprobe."""
        fps, _ = self._get_video_info(video_path)
        return fps

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception:
            return 0.0  # Unknown duration

    def _get_source_height(self, video_path: Path) -> int:
        """Get source video height in pixels using ffprobe."""
        _, height = self._get_video_info(video_path)
        return height

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
        should_normalize = source_fps > self._fps_threshold
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
        # Get source video info in single ffprobe call
        source_fps, source_height = self._get_video_info(source_video)
        should_normalize_fps = source_fps > self._fps_threshold
        is_already_small = source_height <= 720

        # Skip proxy generation if input is already small enough (e.g., API-generated 720p proxy)
        # and doesn't need FPS normalization
        if is_already_small and not should_normalize_fps:
            return source_video

        proxy_path = self._get_proxy_path_for_fps(source_video, should_normalize_fps)

        # Skip if already exists
        if proxy_path.exists():
            return proxy_path

        # Build filter chain
        # If source is already small (<=720p), only normalize FPS without downscaling
        # Otherwise, downscale to config.height (480p) and optionally normalize FPS
        if is_already_small and should_normalize_fps:
            # Keep resolution, just normalize FPS (e.g., 720p@60fps -> 720p@30fps)
            vf_filter = f"fps={self.config.fps}"
        elif should_normalize_fps:
            vf_filter = f"scale=-2:{self.config.height},fps={self.config.fps}"
        else:
            vf_filter = f"scale=-2:{self.config.height}"

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

        # Get video duration for progress tracking
        duration = self._get_duration(source_video)

        # Timeout: 10 minutes max, or 10x video duration, whichever is greater
        timeout_seconds = max(600, int(duration * 10)) if duration > 0 else 600

        # Run ffmpeg with progress output to stderr
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Parse stderr for progress updates
            # FFmpeg outputs lines like: time=00:00:04.10
            time_pattern = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
            last_progress = 0.0
            stderr_output = b""

            if process.stderr:
                while True:
                    chunk = process.stderr.read(256)
                    if not chunk:
                        break
                    stderr_output += chunk

                    # Try to parse time from recent output
                    chunk_str = chunk.decode("utf-8", errors="ignore")
                    match = time_pattern.search(chunk_str)
                    if match and duration > 0 and progress_callback:
                        hours, minutes, seconds = match.groups()
                        current_time = (
                            int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                        )
                        progress = min(current_time / duration, 0.99)
                        # Only report if progress increased meaningfully
                        if progress - last_progress >= 0.02:
                            last_progress = progress
                            progress_callback(progress, "Preparing for analysis...")

            process.wait(timeout=timeout_seconds)

            if process.returncode != 0:
                # Clean up failed proxy
                proxy_path.unlink(missing_ok=True)
                raise RuntimeError(f"ffmpeg failed: {stderr_output.decode()}")

            return proxy_path

        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
        except subprocess.TimeoutExpired:
            # Kill the hung process
            if process is not None:
                process.kill()
                process.wait(timeout=5)
            proxy_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg timed out after {timeout_seconds}s")
        except Exception:
            # Ensure process is cleaned up on any error
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            proxy_path.unlink(missing_ok=True)
            raise

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
        should_normalize = source_fps > self._fps_threshold
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
