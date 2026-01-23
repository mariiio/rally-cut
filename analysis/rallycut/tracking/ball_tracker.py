"""
Ball tracking using ONNX inference for volleyball videos.

Based on fast-volleyball-tracking-inference:
https://github.com/asigatchov/fast-volleyball-tracking-inference

Uses lightweight ONNX models optimized for CPU (~100 FPS).
"""

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
MODEL_URL = "https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/master/models/VballNetV1b_seq9_grayscale_best.onnx"
MODEL_NAME = "VballNetV1b_seq9_grayscale_best.onnx"
MODEL_INPUT_HEIGHT = 288
MODEL_INPUT_WIDTH = 512
SEQUENCE_LENGTH = 9  # 9-frame temporal context


def _get_model_cache_dir() -> Path:
    """Get the cache directory for ball tracking models."""
    from platformdirs import user_cache_dir

    cache_dir = Path(user_cache_dir("rallycut")) / "models" / "ball_tracking"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_model(url: str, dest_path: Path) -> None:
    """Download model file from URL."""
    import httpx

    logger.info(f"Downloading ball tracking model to {dest_path}...")

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    if downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                        logger.debug(f"Downloaded {downloaded / 1024 / 1024:.1f}MB ({pct:.0f}%)")

    logger.info(f"Model downloaded: {dest_path.stat().st_size / 1024 / 1024:.1f}MB")


@dataclass
class BallPosition:
    """Single ball detection result."""

    frame_number: int
    x: float  # Normalized 0-1 (relative to video width)
    y: float  # Normalized 0-1 (relative to video height)
    confidence: float  # Detection confidence 0-1

    def to_dict(self) -> dict:
        return {
            "frameNumber": self.frame_number,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
        }


@dataclass
class BallTrackingResult:
    """Complete ball tracking result for a video segment."""

    positions: list[BallPosition] = field(default_factory=list)
    frame_count: int = 0
    video_fps: float = 30.0
    video_width: int = 0
    video_height: int = 0
    processing_time_ms: float = 0.0
    model_version: str = MODEL_NAME

    @property
    def detection_rate(self) -> float:
        """Percentage of frames with ball detected (confidence > 0.5)."""
        if self.frame_count == 0:
            return 0.0
        detected = sum(1 for p in self.positions if p.confidence > 0.5)
        return detected / self.frame_count

    def to_dict(self) -> dict:
        return {
            "positions": [p.to_dict() for p in self.positions],
            "frameCount": self.frame_count,
            "videoFps": self.video_fps,
            "videoWidth": self.video_width,
            "videoHeight": self.video_height,
            "detectionRate": self.detection_rate,
            "processingTimeMs": self.processing_time_ms,
            "modelVersion": self.model_version,
        }

    def to_json(self, path: Path) -> None:
        """Write result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "BallTrackingResult":
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        positions = [
            BallPosition(
                frame_number=p["frameNumber"],
                x=p["x"],
                y=p["y"],
                confidence=p["confidence"],
            )
            for p in data.get("positions", [])
        ]

        return cls(
            positions=positions,
            frame_count=data.get("frameCount", 0),
            video_fps=data.get("videoFps", 30.0),
            video_width=data.get("videoWidth", 0),
            video_height=data.get("videoHeight", 0),
            processing_time_ms=data.get("processingTimeMs", 0.0),
            model_version=data.get("modelVersion", MODEL_NAME),
        )


class BallTracker:
    """
    Ball tracker using ONNX inference.

    Uses 9-frame grayscale sequences at 288x512 resolution.
    Optimized for CPU (~100 FPS on standard hardware).
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize ball tracker.

        Args:
            model_path: Optional path to ONNX model. If not provided,
                       downloads to cache on first use.
        """
        self.model_path = model_path
        self._session: Any = None

    def _ensure_model(self) -> Path:
        """Ensure model is available, downloading if necessary."""
        if self.model_path and self.model_path.exists():
            return self.model_path

        cache_dir = _get_model_cache_dir()
        cached_path = cache_dir / MODEL_NAME

        if not cached_path.exists():
            _download_model(MODEL_URL, cached_path)

        self.model_path = cached_path
        return cached_path

    def _load_session(self) -> Any:
        """Load ONNX Runtime session."""
        if self._session is not None:
            return self._session

        import onnxruntime as ort

        model_path = self._ensure_model()

        # Configure session options for CPU performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Parallel ops within single inference

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        logger.info(f"Loaded ball tracking model: {model_path.name}")
        return self._session

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame for the model.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Grayscale frame resized to model input size (H=512, W=288)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to model input size (height=288, width=512)
        resized = cv2.resize(gray, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))

        return resized

    def _create_sequence(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Create input sequence for the model.

        Args:
            frames: List of 9 preprocessed grayscale frames

        Returns:
            Input tensor of shape (1, 9, 288, 512) normalized to 0-1
        """
        # Stack frames: (9, 288, 512)
        sequence = np.stack(frames, axis=0)

        # Normalize to 0-1
        sequence = sequence.astype(np.float32) / 255.0

        # Add batch dimension: (1, 9, 288, 512)
        sequence = np.expand_dims(sequence, axis=0)

        return sequence

    def _decode_heatmap(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[float, float, float]:
        """
        Decode a single heatmap to ball position using contour detection.

        Args:
            heatmap: 2D heatmap array (height, width)
            threshold: Threshold for binary detection

        Returns:
            Tuple of (x_norm, y_norm, confidence) where coordinates are 0-1 normalized
        """
        # Normalize heatmap to 0-1 if needed
        heatmap_norm = heatmap.astype(np.float32)
        if heatmap_norm.max() > 1.0:
            heatmap_norm = heatmap_norm / 255.0

        # Binary threshold
        _, binary = cv2.threshold(heatmap_norm, threshold, 1.0, cv2.THRESH_BINARY)
        binary_uint8 = (binary * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.5, 0.5, 0.0

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get centroid using moments
        moments = cv2.moments(largest_contour)
        if moments["m00"] == 0:
            return 0.5, 0.5, 0.0

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # Normalize to 0-1 (heatmap is in model space: height=288, width=512)
        x_norm = cx / heatmap.shape[1]
        y_norm = cy / heatmap.shape[0]

        # Clamp to valid range
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        # Confidence based on contour area and max heatmap value
        max_val = float(np.max(heatmap_norm))
        confidence = min(1.0, max_val)

        return x_norm, y_norm, confidence

    def _decode_output(
        self,
        output: np.ndarray,
        frame_number: int,
        video_width: int,
        video_height: int,
    ) -> list[BallPosition]:
        """
        Decode model output to ball positions.

        The VballNet model outputs heatmaps for 3 consecutive frames.
        Output shape: (batch, 3, height, width)

        Args:
            output: Model output tensor
            frame_number: Frame index of the first output frame
            video_width: Original video width
            video_height: Original video height

        Returns:
            List of BallPosition for each output frame
        """
        positions = []

        # Model outputs 3 heatmaps for the middle 3 frames of the 9-frame input
        # Output shape: (batch, 3, height, width) or (batch, height, width) for single frame
        if output.ndim == 4:
            # Multi-frame output: (batch, num_frames, height, width)
            num_frames = output.shape[1]
            for i in range(num_frames):
                heatmap = output[0, i, :, :]
                x_norm, y_norm, confidence = self._decode_heatmap(heatmap)
                positions.append(BallPosition(
                    frame_number=frame_number + i,
                    x=x_norm,
                    y=y_norm,
                    confidence=confidence,
                ))
        elif output.ndim == 3:
            # Single frame output: (batch, height, width)
            heatmap = output[0, :, :]
            x_norm, y_norm, confidence = self._decode_heatmap(heatmap)
            positions.append(BallPosition(
                frame_number=frame_number,
                x=x_norm,
                y=y_norm,
                confidence=confidence,
            ))
        else:
            # Unknown format
            logger.warning(f"Unknown output format: shape={output.shape}")
            positions.append(BallPosition(
                frame_number=frame_number,
                x=0.5,
                y=0.5,
                confidence=0.0,
            ))

        return positions

    def track_video(
        self,
        video_path: Path | str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> BallTrackingResult:
        """
        Track ball positions in a video segment.

        Args:
            video_path: Path to video file
            start_ms: Start time in milliseconds (optional)
            end_ms: End time in milliseconds (optional)
            progress_callback: Optional callback(progress: float) for progress updates

        Returns:
            BallTrackingResult with all detected positions
        """
        import time

        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load ONNX session
        session = self._load_session()

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate frame range
            start_frame = 0
            end_frame = total_frames

            if start_ms is not None:
                start_frame = int(start_ms / 1000 * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if end_ms is not None:
                end_frame = min(int(end_ms / 1000 * fps), total_frames)

            frames_to_process = end_frame - start_frame
            logger.info(
                f"Processing frames {start_frame}-{end_frame} "
                f"({frames_to_process} frames, {fps:.1f} fps)"
            )

            # Process frames with stride optimization
            # Model takes 9 frames and outputs 3 predictions for middle frames
            # So we can stride by 3 frames instead of 1 to avoid redundant inference
            output_frames = 3  # Model outputs 3 frames per inference
            positions: list[BallPosition] = []
            frame_buffer: list[np.ndarray] = []
            frame_idx = start_frame
            inference_count = 0

            # Get session info once
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Preprocess and add to buffer
                preprocessed = self._preprocess_frame(frame)
                frame_buffer.append(preprocessed)
                frame_idx += 1

                # Run inference when we have enough frames
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    # Create input sequence
                    input_tensor = self._create_sequence(frame_buffer)

                    # Run inference
                    outputs = session.run([output_name], {input_name: input_tensor})
                    output = outputs[0]
                    inference_count += 1

                    # Log output shape on first inference for debugging
                    if inference_count == 1:
                        logger.info(f"Model output shape: {output.shape}, dtype: {output.dtype}")
                        logger.info(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

                    # Decode output (prediction is for middle frames of sequence)
                    # Model outputs 3 heatmaps for the middle 3 frames
                    middle_frame = frame_idx - SEQUENCE_LENGTH + SEQUENCE_LENGTH // 2 - 1
                    decoded_positions = self._decode_output(
                        output, middle_frame, video_width, video_height
                    )
                    positions.extend(decoded_positions)

                    # Clear buffer and keep last (SEQUENCE_LENGTH - output_frames) frames
                    # This gives us a stride of output_frames for ~3x speedup
                    frame_buffer = frame_buffer[output_frames:]

                # Progress callback
                if progress_callback and frame_idx % 30 == 0:
                    progress = (frame_idx - start_frame) / frames_to_process
                    progress_callback(progress)

            # Final progress
            if progress_callback:
                progress_callback(1.0)

            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Completed {inference_count} inferences in {processing_time_ms/1000:.1f}s")

            return BallTrackingResult(
                positions=positions,
                frame_count=frames_to_process,
                video_fps=fps,
                video_width=video_width,
                video_height=video_height,
                processing_time_ms=processing_time_ms,
                model_version=MODEL_NAME,
            )

        finally:
            cap.release()

    def track_frames(
        self,
        frames: Iterator[np.ndarray],
        video_fps: float,
        video_width: int,
        video_height: int,
    ) -> Iterator[BallPosition]:
        """
        Track ball positions in a stream of frames.

        Args:
            frames: Iterator of BGR frames
            video_fps: Video frame rate
            video_width: Frame width
            video_height: Frame height

        Yields:
            BallPosition for each frame (after initial buffer fills)
        """
        output_frames = 3  # Model outputs 3 frames per inference
        session = self._load_session()
        frame_buffer: list[np.ndarray] = []
        frame_idx = 0

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        for frame in frames:
            preprocessed = self._preprocess_frame(frame)
            frame_buffer.append(preprocessed)
            frame_idx += 1

            if len(frame_buffer) == SEQUENCE_LENGTH:
                input_tensor = self._create_sequence(frame_buffer)
                outputs = session.run([output_name], {input_name: input_tensor})
                output = outputs[0]

                middle_frame = frame_idx - SEQUENCE_LENGTH + SEQUENCE_LENGTH // 2 - 1
                yield from self._decode_output(output, middle_frame, video_width, video_height)

                # Stride optimization: keep last (SEQUENCE_LENGTH - output_frames) frames
                frame_buffer = frame_buffer[output_frames:]
