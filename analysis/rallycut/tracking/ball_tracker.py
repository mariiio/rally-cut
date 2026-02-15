"""
Ball tracking using ONNX inference for volleyball videos.

Based on fast-volleyball-tracking-inference:
https://github.com/asigatchov/fast-volleyball-tracking-inference

Uses lightweight ONNX models optimized for CPU (~100 FPS).
"""

import json
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rallycut.tracking.ball_filter import BallFilterConfig

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Model repository base URL
MODEL_REPO_BASE = "https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/master/models"

# Model registry: model_id -> (filename, height, width)
# All models use 9-frame sequences and grayscale input
BALL_MODELS: dict[str, tuple[str, int, int]] = {
    "v2": ("VballNetV2_seq9_grayscale_320_h288_w512.onnx", 288, 512),
    # v1c excluded - requires recurrent hidden state input which is incompatible
    "fast": ("VballNetFastV1_seq9_grayscale_233_h288_w512.onnx", 288, 512),
}

# Default model configuration
# v2 is most consistent across videos (70.6% match rate on ground truth)
# fast is 3.6x faster but 1.5% less accurate
DEFAULT_BALL_MODEL = "v2"
MODEL_NAME = BALL_MODELS[DEFAULT_BALL_MODEL][0]
MODEL_INPUT_HEIGHT = BALL_MODELS[DEFAULT_BALL_MODEL][1]
MODEL_INPUT_WIDTH = BALL_MODELS[DEFAULT_BALL_MODEL][2]
SEQUENCE_LENGTH = 9  # 9-frame temporal context (all models)


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


def get_available_ball_models() -> list[str]:
    """Return list of available ball tracking model IDs.

    Includes VballNet variants (v2, fast) and WASB/ensemble options.
    """
    return list(BALL_MODELS.keys()) + ["wasb", "ensemble"]


def get_ball_model_info(model_id: str) -> tuple[str, int, int]:
    """Get model info (filename, height, width) for a model ID.

    Args:
        model_id: Model identifier (e.g., 'v2', 'fast')

    Returns:
        Tuple of (filename, height, width)

    Raises:
        ValueError: If model_id not in registry
    """
    if model_id not in BALL_MODELS:
        available = ", ".join(BALL_MODELS.keys())
        raise ValueError(f"Unknown ball model '{model_id}'. Available: {available}")
    return BALL_MODELS[model_id]


def ensure_ball_model(model_id: str) -> Path:
    """Ensure a ball tracking model is downloaded and return its path.

    Args:
        model_id: Model identifier (e.g., 'v2', 'fast')

    Returns:
        Path to the downloaded model file
    """
    filename, _, _ = get_ball_model_info(model_id)
    cache_dir = _get_model_cache_dir()
    cached_path = cache_dir / filename

    if not cached_path.exists():
        url = f"{MODEL_REPO_BASE}/{filename}"
        _download_model(url, cached_path)

    return cached_path


@dataclass
class HeatmapDecodingConfig:
    """Configuration for heatmap-to-position decoding.

    Uses contour centroid at a fixed threshold. Testing on ground truth data showed
    threshold=0.3 gives best results: 98% detection rate, 24% match rate at 50px.
    Adaptive threshold, weighted centroid, sub-pixel refinement, and multi-threshold
    were all tested and rejected (hurt accuracy on beach volleyball).
    """

    # Threshold for heatmap binarization
    # 0.3 tested better than 0.5 on beach volleyball ground truth
    threshold: float = 0.3


@dataclass
class BallPosition:
    """Single ball detection result."""

    frame_number: int
    x: float  # Normalized 0-1 (relative to video width)
    y: float  # Normalized 0-1 (relative to video height)
    confidence: float  # Detection confidence 0-1
    motion_energy: float = 0.0  # Motion energy at ball position (0-1)

    def to_dict(self) -> dict:
        d = {
            "frameNumber": self.frame_number,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
        }
        if self.motion_energy > 0:
            d["motionEnergy"] = self.motion_energy
        return d


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
    filtering_enabled: bool = False
    raw_positions: list[BallPosition] | None = None  # Before filtering (debug)

    @property
    def detection_rate(self) -> float:
        """Percentage of frames with ball detected (confidence > 0.5)."""
        if self.frame_count == 0:
            return 0.0
        detected = sum(1 for p in self.positions if p.confidence > 0.5)
        return detected / self.frame_count

    def to_dict(self) -> dict:
        result = {
            "positions": [p.to_dict() for p in self.positions],
            "frameCount": self.frame_count,
            "videoFps": self.video_fps,
            "videoWidth": self.video_width,
            "videoHeight": self.video_height,
            "detectionRate": self.detection_rate,
            "processingTimeMs": self.processing_time_ms,
            "modelVersion": self.model_version,
            "filteringEnabled": self.filtering_enabled,
        }
        if self.raw_positions is not None:
            result["rawPositions"] = [p.to_dict() for p in self.raw_positions]
        return result

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
                motion_energy=p.get("motionEnergy", 0.0),
            )
            for p in data.get("positions", [])
        ]

        raw_positions = None
        if "rawPositions" in data:
            raw_positions = [
                BallPosition(
                    frame_number=p["frameNumber"],
                    x=p["x"],
                    y=p["y"],
                    confidence=p["confidence"],
                    motion_energy=p.get("motionEnergy", 0.0),
                )
                for p in data["rawPositions"]
            ]

        return cls(
            positions=positions,
            frame_count=data.get("frameCount", 0),
            video_fps=data.get("videoFps", 30.0),
            video_width=data.get("videoWidth", 0),
            video_height=data.get("videoHeight", 0),
            processing_time_ms=data.get("processingTimeMs", 0.0),
            model_version=data.get("modelVersion", MODEL_NAME),
            filtering_enabled=data.get("filteringEnabled", False),
            raw_positions=raw_positions,
        )


class BallTracker:
    """
    Ball tracker using ONNX inference.

    Uses 9-frame grayscale sequences at 288x512 resolution.
    Optimized for CPU (~100 FPS on standard hardware).

    Supports multiple model variants via the `model` parameter:
    - 'v2' (default): VballNetV2 - most consistent across videos (70.6% match rate)
    - 'fast': VballNetFastV1 - 3.6x faster, 1.5% less accurate (69.1% match rate)
    """

    def __init__(
        self,
        model_path: Path | None = None,
        model: str = DEFAULT_BALL_MODEL,
        heatmap_config: HeatmapDecodingConfig | None = None,
    ):
        """
        Initialize ball tracker.

        Args:
            model_path: Optional path to ONNX model. If not provided,
                       downloads to cache on first use.
            model: Model variant to use ('v2', 'fast').
                  Ignored if model_path is provided.
            heatmap_config: Configuration for heatmap decoding. If not provided,
                           uses defaults (threshold=0.3, contour centroid).
        """
        self.model_path = model_path
        self.model_id = model
        self.heatmap_config = heatmap_config or HeatmapDecodingConfig()
        self._session: Any = None

        # Get model dimensions from registry
        if model_path is None:
            _, self._input_height, self._input_width = get_ball_model_info(model)
        else:
            # Use default dimensions for custom models
            self._input_height = MODEL_INPUT_HEIGHT
            self._input_width = MODEL_INPUT_WIDTH

    def _ensure_model(self) -> Path:
        """Ensure model is available, downloading if necessary."""
        if self.model_path and self.model_path.exists():
            return self.model_path

        # Use model registry to download the selected model
        cached_path = ensure_ball_model(self.model_id)
        self.model_path = cached_path
        return cached_path

    def _load_session(self) -> Any:
        """Load ONNX Runtime session with GPU fallback to CPU."""
        if self._session is not None:
            return self._session

        import onnxruntime as ort

        model_path = self._ensure_model()

        # Configure session options for performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Use available CPUs or environment override
        num_threads = int(os.environ.get("ONNX_NUM_THREADS", os.cpu_count() or 4))
        sess_options.intra_op_num_threads = num_threads

        # Try GPU providers first, fall back to CPU
        available_providers = ort.get_available_providers()
        preferred_providers = []

        # CUDA is typically 2-5x faster for this model
        if "CUDAExecutionProvider" in available_providers:
            preferred_providers.append("CUDAExecutionProvider")
        if "CoreMLExecutionProvider" in available_providers:
            preferred_providers.append("CoreMLExecutionProvider")
        # Always include CPU as fallback
        preferred_providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=preferred_providers,
        )

        # Log which provider was actually used
        active_provider = self._session.get_providers()[0] if self._session.get_providers() else "unknown"
        logger.info(f"Loaded ball tracking model: {model_path.name} (provider: {active_provider}, threads: {num_threads})")
        return self._session

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame for the model.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Grayscale frame resized to model input size
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to model input size (width, height for cv2.resize)
        resized = cv2.resize(gray, (self._input_width, self._input_height))

        return resized

    def _create_sequence(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Create input sequence for the model.

        Args:
            frames: List of 9 preprocessed grayscale frames

        Returns:
            Input tensor of shape (1, 9, H, W) normalized to 0-1
        """
        # Stack frames directly as float32 and normalize in one operation
        # Shape: (1, 9, H, W) - add batch dimension during stack
        sequence = np.stack(frames, axis=0).astype(np.float32, copy=False)
        sequence *= (1.0 / 255.0)  # In-place multiply is faster than divide
        return sequence[np.newaxis, ...]  # Add batch dimension

    def _decode_heatmap(
        self,
        heatmap: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Decode a single heatmap to ball position using contour centroid.

        Thresholds the heatmap, finds the largest contour, and returns its
        centroid as the ball position.

        Args:
            heatmap: 2D heatmap array (height, width)

        Returns:
            Tuple of (x_norm, y_norm, confidence) where coordinates are 0-1 normalized
        """
        threshold = self.heatmap_config.threshold

        # Contour-based centroid
        max_val = float(heatmap.max())
        if max_val > 1.0:
            # Scale threshold to match unnormalized range, avoiding division
            scaled_threshold = threshold * 255.0
            # Direct uint8 thresholding - single operation instead of three
            if heatmap.dtype != np.uint8:
                heatmap_uint8 = heatmap.astype(np.uint8, copy=False)
            else:
                heatmap_uint8 = heatmap
            _, binary_uint8 = cv2.threshold(heatmap_uint8, int(scaled_threshold), 255, cv2.THRESH_BINARY)
            # Normalize max_val for confidence calculation
            max_val = max_val / 255.0
        else:
            # Already normalized - convert to uint8 for contour detection
            _, binary = cv2.threshold(heatmap.astype(np.float32, copy=False), threshold, 1.0, cv2.THRESH_BINARY)
            binary_uint8 = (binary * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.5, 0.5, 0.0

        # Find largest contour - for single contour, skip max() overhead
        if len(contours) == 1:
            largest_contour = contours[0]
        else:
            largest_contour = max(contours, key=cv2.contourArea)

        # Get centroid using moments
        moments = cv2.moments(largest_contour)
        if moments["m00"] == 0:
            return 0.5, 0.5, 0.0

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # Normalize to 0-1 (heatmap is in model space)
        x_norm = cx / heatmap.shape[1]
        y_norm = cy / heatmap.shape[0]

        # Clamp to valid range
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        # Confidence based on max heatmap value (already normalized above)
        confidence = min(1.0, max_val)

        return x_norm, y_norm, confidence

    # Motion energy: half-width of the comparison patch (15x15 pixels)
    _ME_PATCH_HALF = 7

    def _compute_motion_energy(
        self,
        positions: list[BallPosition],
        frame_buffer: list[np.ndarray],
        prev_frame: np.ndarray | None,
    ) -> None:
        """Compute motion energy for detected positions in-place.

        Compares each detected position's patch against the previous frame's
        patch at the same location. High energy = real ball (temporal change),
        low energy = likely false positive at a static position.

        Args:
            positions: Decoded positions to annotate with motion_energy.
            frame_buffer: Preprocessed grayscale frames for the current window.
            prev_frame: Last frame from the previous window (for cross-window
                continuity), or None if this is the first window.
        """
        h = self._ME_PATCH_HALF
        w, ht = self._input_width, self._input_height

        for i, pos in enumerate(positions):
            if pos.confidence <= 0:
                continue

            # Determine the reference frame for differencing
            if i == 0:
                ref = prev_frame
            else:
                ref = frame_buffer[i - 1]
            if ref is None:
                continue

            px = max(h, min(int(pos.x * w), w - h - 1))
            py = max(h, min(int(pos.y * ht), ht - h - 1))
            cur = frame_buffer[i][py - h : py + h + 1, px - h : px + h + 1]
            prev = ref[py - h : py + h + 1, px - h : px + h + 1]
            pos.motion_energy = float(
                np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32))) / 255.0
            )

    def _decode_output(
        self,
        output: np.ndarray,
        frame_number: int,
    ) -> list[BallPosition]:
        """
        Decode model output to ball positions.

        The VballNet model outputs heatmaps for all 9 frames of the input sequence.
        Output shape: (batch, 9, height, width)

        Args:
            output: Model output tensor
            frame_number: Frame index of the first output frame

        Returns:
            List of BallPosition for each output frame
        """
        positions = []

        # Model outputs 9 heatmaps for all 9 frames of the input sequence
        # Output shape: (batch, 9, height, width) or (batch, height, width) for single frame
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
        filter_config: "BallFilterConfig | None" = None,
        enable_filtering: bool = True,
        preserve_raw: bool = False,
    ) -> BallTrackingResult:
        """
        Track ball positions in a video segment.

        Args:
            video_path: Path to video file
            start_ms: Start time in milliseconds (optional)
            end_ms: End time in milliseconds (optional)
            progress_callback: Optional callback(progress: float) for progress updates
            filter_config: Optional configuration for temporal filtering
            enable_filtering: Apply Kalman filter smoothing (default True)
            preserve_raw: Store raw positions in result for debugging

        Returns:
            BallTrackingResult with all detected positions
        """
        import time

        from rallycut.tracking.ball_filter import BallFilterConfig, BallTemporalFilter

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

            # Process frames in non-overlapping windows
            # Model takes 9 frames and outputs 9 predictions (one per input frame)
            positions: list[BallPosition] = []
            frame_buffer: list[np.ndarray] = []
            frame_idx = start_frame
            inference_count = 0
            prev_preprocessed: np.ndarray | None = None  # For motion energy

            # Get session info once
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        f"Frame read failed at index {frame_idx} "
                        f"(expected {end_frame - start_frame} frames, got {frame_idx - start_frame})"
                    )
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

                    # Decode output - model outputs heatmaps for all 9 input frames
                    first_frame = frame_idx - SEQUENCE_LENGTH
                    decoded_positions = self._decode_output(output, first_frame)

                    self._compute_motion_energy(
                        decoded_positions, frame_buffer, prev_preprocessed
                    )
                    positions.extend(decoded_positions)

                    # Update prev_preprocessed to last frame of this window
                    prev_preprocessed = frame_buffer[-1]

                    # Clear buffer completely for non-overlapping windows (stride by 9)
                    frame_buffer = []

                # Progress callback
                if progress_callback and frame_idx % 30 == 0:
                    progress = (frame_idx - start_frame) / frames_to_process
                    progress_callback(progress)

            # Handle remaining frames in buffer (partial final window)
            # Without this, up to 8 frames at the end of each segment are silently dropped
            if frame_buffer:
                real_frame_count = len(frame_buffer)
                # Pad with last frame repeated to fill the 9-frame window
                last_frame = frame_buffer[-1]
                while len(frame_buffer) < SEQUENCE_LENGTH:
                    frame_buffer.append(last_frame)

                input_tensor = self._create_sequence(frame_buffer)
                outputs = session.run([output_name], {input_name: input_tensor})
                output = outputs[0]
                inference_count += 1

                first_frame = frame_idx - real_frame_count
                decoded_positions = self._decode_output(output, first_frame)
                # Only keep positions for real frames (discard padded)
                decoded_positions = decoded_positions[:real_frame_count]

                self._compute_motion_energy(
                    decoded_positions, frame_buffer, prev_preprocessed
                )
                positions.extend(decoded_positions)
                frame_buffer = []

            # Final progress
            if progress_callback:
                progress_callback(1.0)

            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Completed {inference_count} inferences in {processing_time_ms/1000:.1f}s")

            # Apply temporal filtering if enabled
            raw_positions = None
            if enable_filtering:
                if preserve_raw:
                    raw_positions = positions.copy()
                config = filter_config or BallFilterConfig()
                temporal_filter = BallTemporalFilter(config)

                positions = temporal_filter.filter_batch(positions)

            # Get model filename for version string
            model_filename = BALL_MODELS.get(self.model_id, (MODEL_NAME,))[0]

            return BallTrackingResult(
                positions=positions,
                frame_count=frames_to_process,
                video_fps=fps,
                video_width=video_width,
                video_height=video_height,
                processing_time_ms=processing_time_ms,
                model_version=model_filename,
                filtering_enabled=enable_filtering,
                raw_positions=raw_positions,
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
        Track ball positions in a stream of frames (raw, unfiltered).

        Note: This streaming API returns raw detections without temporal filtering.
        Use track_video() for filtered results, or apply BallTemporalFilter manually.

        Args:
            frames: Iterator of BGR frames
            video_fps: Video frame rate
            video_width: Frame width
            video_height: Frame height

        Yields:
            BallPosition for each frame (after initial buffer fills)
        """
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

                first_frame = frame_idx - SEQUENCE_LENGTH
                yield from self._decode_output(output, first_frame)

                # Clear buffer completely for non-overlapping windows (stride by 9)
                frame_buffer = []

        # Handle remaining frames in buffer (partial final window)
        if frame_buffer:
            real_frame_count = len(frame_buffer)
            last_frame = frame_buffer[-1]
            while len(frame_buffer) < SEQUENCE_LENGTH:
                frame_buffer.append(last_frame)

            input_tensor = self._create_sequence(frame_buffer)
            outputs = session.run([output_name], {input_name: input_tensor})
            output = outputs[0]

            first_frame = frame_idx - real_frame_count
            decoded_positions = self._decode_output(output, first_frame)
            yield from decoded_positions[:real_frame_count]


def create_ball_tracker(
    model: str = DEFAULT_BALL_MODEL,
    **kwargs: Any,
) -> Any:
    """Factory function to create a ball tracker by model name.

    Supports VballNet variants (v2, fast), WASB HRNet, and ensemble.
    All returned trackers have a compatible track_video() interface.

    Args:
        model: Model identifier. VballNet: 'v2', 'fast'.
            WASB: 'wasb'. Ensemble: 'ensemble'.
        **kwargs: Additional keyword arguments passed to the tracker constructor.
            For WASB/ensemble: device, threshold, weights_path/wasb_weights.

    Returns:
        A tracker instance (BallTracker, WASBBallTracker, or EnsembleBallTracker).
    """
    if model == "wasb":
        from rallycut.tracking.wasb_model import WASBBallTracker

        return WASBBallTracker(
            weights_path=kwargs.get("weights_path"),
            device=kwargs.get("device"),
            threshold=kwargs.get("threshold", 0.3),
        )
    elif model == "ensemble":
        from rallycut.tracking.ball_ensemble import EnsembleBallTracker

        return EnsembleBallTracker(
            vballnet_model=kwargs.get("vballnet_model", DEFAULT_BALL_MODEL),
            wasb_weights=kwargs.get("wasb_weights"),
            wasb_device=kwargs.get("device"),
            wasb_threshold=kwargs.get("threshold", 0.3),
        )
    else:
        return BallTracker(model=model)
