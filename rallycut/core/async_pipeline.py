"""Async pipeline for overlapping decode and inference.

This module provides a producer-consumer pipeline that overlaps frame
decoding with ML inference, hiding decode latency and achieving up to
1.3-1.5x speedup on typical workloads.

Architecture:
    [Video File] -> [Decoder Thread] -> [Frame Queue] -> [Inference Thread] -> [Results]

The decoder thread reads frames ahead of the inference thread, keeping
the GPU busy with inference while CPU handles decoding.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any

import numpy as np


@dataclass
class FrameBatch:
    """A batch of frames ready for inference."""

    frames: list[list[np.ndarray]]  # List of window frames
    start_frames: list[int]  # Start frame index for each window
    metadata: dict[str, Any] | None = None


@dataclass
class InferenceResult:
    """Result from inference on a batch."""

    states: list[Any]  # GameState values
    confidences: list[float]
    start_frames: list[int]
    end_frames: list[int]


class AsyncFrameDecoder:
    """
    Async frame decoder that reads ahead into a queue.

    Runs in a separate thread to overlap decoding with inference.
    """

    def __init__(
        self,
        video_path: Path,
        window_size: int = 16,
        stride: int = 8,
        frame_sample: int = 1,
        batch_size: int = 16,
        queue_size: int = 4,
        start_frame: int = 0,
        end_frame: int | None = None,
        resize: tuple[int, int] | None = None,
        use_hwaccel: bool = False,
        device: str | None = None,
    ):
        self.video_path = video_path
        self.window_size = window_size
        self.stride = stride
        self.frame_sample = frame_sample
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.resize = resize
        self.use_hwaccel = use_hwaccel
        self.device = device

        self._queue: Queue[FrameBatch | None] = Queue(maxsize=queue_size)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        """Start the decoder thread."""
        self._stop_event.clear()
        self._error = None
        self._thread = Thread(target=self._decode_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the decoder thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_batch(self, timeout: float = 5.0) -> FrameBatch | None:
        """
        Get the next batch of frames.

        Returns:
            FrameBatch or None if decoding is complete.

        Raises:
            Exception if decoder thread encountered an error.
        """
        if self._error is not None:
            raise self._error

        try:
            batch = self._queue.get(timeout=timeout)
            if batch is None:
                # End of stream
                return None
            return batch
        except Empty:
            if self._error is not None:
                raise self._error
            return None

    def _decode_loop(self) -> None:
        """Main decode loop running in separate thread."""
        import cv2

        from rallycut.core.video import Video

        try:
            with Video(
                self.video_path,
                use_hwaccel=self.use_hwaccel,
                device=self.device,
            ) as video:
                end_frame = self.end_frame or video.info.frame_count

                # Calculate frame span for windowing
                frame_span = (self.window_size - 1) * self.frame_sample + 1

                # Frame buffer for windowing
                frame_buffer: list[np.ndarray] = []
                buffer_start_idx = self.start_frame
                next_window_start = self.start_frame

                # Batch accumulation
                batch_frames: list[list[np.ndarray]] = []
                batch_starts: list[int] = []

                for frame_idx, frame in video.iter_frames(
                    start_frame=self.start_frame, end_frame=end_frame
                ):
                    if self._stop_event.is_set():
                        break

                    # Resize if needed
                    if self.resize is not None:
                        h, w = frame.shape[:2]
                        if (w, h) != self.resize:
                            frame = cv2.resize(
                                frame, self.resize, interpolation=cv2.INTER_AREA
                            )

                    frame_buffer.append(frame)

                    # Extract windows from buffer
                    while (
                        len(frame_buffer) >= frame_span
                        and next_window_start <= frame_idx - frame_span + 1
                    ):
                        buffer_offset = next_window_start - buffer_start_idx

                        if (
                            buffer_offset >= 0
                            and buffer_offset + frame_span <= len(frame_buffer)
                        ):
                            # Extract and subsample window
                            raw_frames = frame_buffer[
                                buffer_offset : buffer_offset + frame_span
                            ]
                            window_frames = raw_frames[:: self.frame_sample][
                                : self.window_size
                            ]

                            batch_frames.append(window_frames)
                            batch_starts.append(next_window_start)
                            next_window_start += self.stride

                            # Emit batch when full
                            if len(batch_frames) >= self.batch_size:
                                batch = FrameBatch(
                                    frames=batch_frames.copy(),
                                    start_frames=batch_starts.copy(),
                                )
                                batch_frames.clear()
                                batch_starts.clear()

                                # Put in queue, blocking if full
                                while not self._stop_event.is_set():
                                    try:
                                        self._queue.put(batch, timeout=0.1)
                                        break
                                    except Full:
                                        continue
                        else:
                            break

                    # Trim buffer
                    min_needed_frame = next_window_start
                    frames_to_drop = min_needed_frame - buffer_start_idx
                    if frames_to_drop > 0:
                        frame_buffer = frame_buffer[frames_to_drop:]
                        buffer_start_idx = min_needed_frame

                # Emit any remaining batch
                if batch_frames and not self._stop_event.is_set():
                    batch = FrameBatch(
                        frames=batch_frames,
                        start_frames=batch_starts,
                    )
                    while not self._stop_event.is_set():
                        try:
                            self._queue.put(batch, timeout=0.1)
                            break
                        except Full:
                            continue

        except Exception as e:
            self._error = e

        finally:
            # Signal end of stream
            try:
                self._queue.put(None, timeout=1.0)
            except Full:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class AsyncInferencePipeline:
    """
    Complete async pipeline for decode + inference.

    Combines AsyncFrameDecoder with a classifier to process
    video frames with overlapped decode and inference.
    """

    def __init__(
        self,
        video_path: Path,
        classifier: Any,  # GameStateClassifier or compatible
        window_size: int = 16,
        stride: int = 8,
        frame_sample: int = 1,
        batch_size: int = 16,
        queue_size: int = 4,
        start_frame: int = 0,
        end_frame: int | None = None,
        resize: tuple[int, int] | None = None,
        use_hwaccel: bool = False,
        device: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ):
        self.video_path = video_path
        self.classifier = classifier
        self.window_size = window_size
        self.stride = stride
        self.frame_sample = frame_sample
        self.batch_size = batch_size
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.resize = resize
        self.progress_callback = progress_callback

        self._decoder = AsyncFrameDecoder(
            video_path=video_path,
            window_size=window_size,
            stride=stride,
            frame_sample=frame_sample,
            batch_size=batch_size,
            queue_size=queue_size,
            start_frame=start_frame,
            end_frame=end_frame,
            resize=resize,
            use_hwaccel=use_hwaccel,
            device=device,
        )

    def run(self) -> Iterator[InferenceResult]:
        """
        Run the async pipeline.

        Yields:
            InferenceResult for each processed batch.
        """
        from rallycut.core.video import Video

        # Get total frames for progress
        with Video(self.video_path) as video:
            total_frames = self.end_frame or video.info.frame_count

        frame_span = (self.window_size - 1) * self.frame_sample + 1
        total_windows = max(
            1, (total_frames - self.start_frame - self.window_size) // self.stride + 1
        )

        processed_windows = 0

        with self._decoder:
            while True:
                batch = self._decoder.get_batch()
                if batch is None:
                    break

                # Run inference
                results = self.classifier.classify_segments_batch(batch.frames)

                # Build result
                states = []
                confidences = []
                end_frames = []

                for i, (state, confidence) in enumerate(results):
                    states.append(state)
                    confidences.append(confidence)
                    end_frames.append(batch.start_frames[i] + frame_span - 1)

                yield InferenceResult(
                    states=states,
                    confidences=confidences,
                    start_frames=batch.start_frames,
                    end_frames=end_frames,
                )

                processed_windows += len(batch.frames)

                if self.progress_callback:
                    progress = min(1.0, processed_windows / total_windows)
                    self.progress_callback(progress)


def analyze_region_async(
    video_path: Path,
    classifier: Any,
    start_frame: int,
    end_frame: int,
    stride: int = 8,
    frame_sample: int = 1,
    batch_size: int = 16,
    resize: tuple[int, int] | None = None,
    use_hwaccel: bool = False,
    device: str | None = None,
) -> list:
    """
    Analyze a video region using async pipeline.

    This is a convenience function that wraps AsyncInferencePipeline
    and returns results compatible with GameStateResult format.

    Args:
        video_path: Path to video file
        classifier: Classifier instance (GameStateClassifier or compatible)
        start_frame: Start frame of region
        end_frame: End frame of region
        stride: Frames between window starts
        frame_sample: Sample every Nth frame
        batch_size: Batch size for inference
        resize: Optional (width, height) to resize frames
        use_hwaccel: Use hardware-accelerated decoding
        device: Device for inference

    Returns:
        List of GameStateResult objects
    """
    from rallycut.core.models import GameStateResult

    pipeline = AsyncInferencePipeline(
        video_path=video_path,
        classifier=classifier,
        window_size=classifier.FRAME_WINDOW,
        stride=stride,
        frame_sample=frame_sample,
        batch_size=batch_size,
        start_frame=start_frame,
        end_frame=end_frame,
        resize=resize,
        use_hwaccel=use_hwaccel,
        device=device,
    )

    results = []
    for batch_result in pipeline.run():
        for i in range(len(batch_result.states)):
            results.append(
                GameStateResult(
                    state=batch_result.states[i],
                    confidence=batch_result.confidences[i],
                    start_frame=batch_result.start_frames[i],
                    end_frame=batch_result.end_frames[i],
                )
            )

    return results
