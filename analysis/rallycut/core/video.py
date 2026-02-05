"""Video abstraction layer for RallyCut."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from rallycut.core.models import VideoInfo

if TYPE_CHECKING:
    from types import TracebackType

    from rallycut.core.hwaccel import HWAccelDecoder

# Optional hardware acceleration
try:
    from rallycut.core.hwaccel import PYAV_AVAILABLE
    from rallycut.core.hwaccel import HWAccelDecoder as _HWAccelDecoder
    from rallycut.core.hwaccel import get_hwaccel_type as _get_hwaccel_type

    HWAccelDecoderClass: type[HWAccelDecoder] | None = _HWAccelDecoder
    get_hwaccel_type_fn: Callable[[str], str | None] | None = _get_hwaccel_type
except ImportError:
    PYAV_AVAILABLE = False
    HWAccelDecoderClass = None
    get_hwaccel_type_fn = None


class Video:
    """
    Video file wrapper with metadata and frame access.

    Supports optional hardware-accelerated decoding via PyAV:
    - NVDEC for NVIDIA CUDA GPUs (2-4x faster)
    - VideoToolbox for Apple Silicon (2-3x faster)

    Falls back to OpenCV if PyAV is not available or hardware acceleration fails.
    """

    def __init__(
        self,
        path: Path | str,
        use_hwaccel: bool = False,
        device: str | None = None,
    ):
        self._cap: cv2.VideoCapture | None = None
        self._hwaccel_decoder = None
        self._info: VideoInfo | None = None
        self._use_hwaccel = use_hwaccel and PYAV_AVAILABLE
        self._device = device

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        # Initialize hardware-accelerated decoder if requested
        if self._use_hwaccel and HWAccelDecoderClass is not None:
            try:
                hwaccel_type = get_hwaccel_type_fn(device) if get_hwaccel_type_fn and device else None
                self._hwaccel_decoder = HWAccelDecoderClass(
                    self.path, hwaccel_type=hwaccel_type, device=device
                )
            except Exception:
                # Fall back to OpenCV
                self._hwaccel_decoder = None
                self._use_hwaccel = False

    @property
    def info(self) -> VideoInfo:
        """Get video metadata (lazy loaded)."""
        if self._info is None:
            self._info = self._extract_info()
        return self._info

    def _extract_info(self) -> VideoInfo:
        """Extract video metadata using OpenCV."""
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {self.path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Decode fourcc to codec string
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            duration = frame_count / fps if fps > 0 else 0.0

            return VideoInfo(
                path=self.path,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                frame_count=frame_count,
                codec=codec,
            )
        finally:
            cap.release()

    def _get_capture(self) -> cv2.VideoCapture:
        """Get or create video capture object."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.path))
        return self._cap

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        """Read a specific frame by index."""
        cap = self._get_capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None

    def iter_frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Iterate over frames yielding (frame_idx, frame).

        If hardware acceleration is enabled and available, uses PyAV with
        NVDEC/VideoToolbox for 2-4x faster decoding.

        For OpenCV fallback:
        - step <= 16: Uses sequential reading (optimal for dense access)
        - step > 16: Uses seeking (optimal for sparse frame sampling)
        """
        # Use hardware-accelerated decoder if available
        if self._hwaccel_decoder is not None:
            yield from self._hwaccel_decoder.iter_frames(start_frame, end_frame, step)
            return

        # Fall back to OpenCV
        cap = self._get_capture()

        if end_frame is None:
            end_frame = self.info.frame_count

        # For sparse access (large step), seeking is faster than reading+discarding
        if step > 16:
            frame_idx = start_frame
            while frame_idx < end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_idx, frame
                frame_idx += step
            return

        # For dense access (small step), sequential reading is faster
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        next_yield_frame = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Only yield frames at step intervals
            if frame_idx == next_yield_frame:
                yield frame_idx, frame
                next_yield_frame += step

            frame_idx += 1

    def compute_content_hash(self, read_size: int = 1024 * 1024) -> str:
        """Compute a content hash for caching.

        The hash is based on filename, file size, modification time, and
        the first `read_size` bytes of the file. This provides a good balance
        between speed and uniqueness.

        Args:
            read_size: Number of bytes to read from start of file (default 1MB).

        Returns:
            Hex digest of the content hash.
        """
        stat = self.path.stat()
        hasher = hashlib.sha256()

        # Include file metadata in hash
        hasher.update(self.path.name.encode())
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime_ns).encode())

        # Include first N bytes of file content
        with open(self.path, "rb") as f:
            hasher.update(f.read(read_size))

        return hasher.hexdigest()[:16]  # 16 hex chars = 64 bits

    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._hwaccel_decoder is not None:
            self._hwaccel_decoder.close()
            self._hwaccel_decoder = None

    def __enter__(self) -> Video:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
