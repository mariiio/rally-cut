"""Video abstraction layer for RallyCut."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from rallycut.core.models import VideoInfo

# Optional hardware acceleration
try:
    from rallycut.core.hwaccel import PYAV_AVAILABLE, HWAccelDecoder, get_hwaccel_type
except ImportError:
    PYAV_AVAILABLE = False
    HWAccelDecoder = None
    get_hwaccel_type = None


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
        if self._use_hwaccel:
            try:
                hwaccel_type = get_hwaccel_type(device) if get_hwaccel_type else None
                self._hwaccel_decoder = HWAccelDecoder(
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

    def read_frames(
        self,
        start_frame: int,
        count: int,
        step: int = 1,
        resize: tuple[int, int] | None = None,
    ) -> list[np.ndarray]:
        """
        Read multiple frames starting from a position.

        Args:
            start_frame: Starting frame index
            count: Number of frames to read
            step: Step between frames
            resize: Optional (width, height) to resize frames for faster processing

        Returns:
            List of frames as numpy arrays
        """
        frames = []
        cap = self._get_capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(count):
            if step > 1 and i > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i * step)

            ret, frame = cap.read()
            if not ret:
                break

            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

            frames.append(frame)

        return frames

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
        - step > 16: Uses seeking (optimal for sparse access like motion detection)
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

    def iter_chunks(
        self,
        chunk_frames: int,
        overlap: int = 0,
    ) -> Iterator[tuple[int, list[np.ndarray]]]:
        """
        Iterate over video in chunks.

        Args:
            chunk_frames: Number of frames per chunk
            overlap: Number of overlapping frames between chunks

        Yields:
            Tuple of (start_frame_idx, list of frames)
        """
        cap = self._get_capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        total_frames = self.info.frame_count
        stride = chunk_frames - overlap
        start_frame = 0

        while start_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []

            for _ in range(chunk_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            if frames:
                yield start_frame, frames

            start_frame += stride

    def close(self):
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._hwaccel_decoder is not None:
            self._hwaccel_decoder.close()
            self._hwaccel_decoder = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()
