"""
VideoMAE adapter for game state classification.

Adapted from volleyball_analytics for RallyCut CLI use.
Original: https://github.com/masouduut94/volleyball_analytics

Supports ONNX Runtime for faster inference (1.5-2x speedup).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from rallycut.core.models import GameState, GameStateResult
from rallycut.core.profiler import get_profiler

if TYPE_CHECKING:
    from collections.abc import Callable

    from rallycut.core.profiler import PerformanceProfiler

# Optional ONNX Runtime support
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def _get_onnx_cache_dir() -> Path:
    """Get the cache directory for ONNX models."""
    from platformdirs import user_cache_dir

    cache_dir = Path(user_cache_dir("rallycut")) / "models" / "onnx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_onnx_providers(device: str) -> list[str]:
    """Get ONNX Runtime execution providers based on device."""
    if device == "cuda":
        # Try CUDA first, fall back to CPU
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "mps":
        # CoreML for Apple Silicon, fall back to CPU
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]


class GameStateClassifier:
    """
    VideoMAE-based game state classification.

    Classifies video segments as SERVICE, PLAY, or NO_PLAY.
    Uses 16-frame windows as input.

    Supports ONNX Runtime for faster inference (1.5-2x speedup).
    Set use_onnx=True in config or pass to constructor.
    """

    FRAME_WINDOW = 16
    IMAGE_SIZE = 224
    # Trained model uses 3 classes
    LABEL_MAP = {0: GameState.NO_PLAY, 1: GameState.PLAY, 2: GameState.SERVICE}

    _model: Any  # VideoMAEForVideoClassification or compiled variant
    _processor: Any  # VideoMAEImageProcessor
    _onnx_session: Any  # ort.InferenceSession or None

    # Known model variants and their HuggingFace model IDs
    MODEL_VARIANTS: dict[str, str] = {
        "v1": "MCG-NJU/videomae-base-finetuned-kinetics",
        "v2": "OpenGVLab/VideoMAEv2-Base",
    }

    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "cpu",
        use_onnx: bool | None = None,
        model_variant: str | None = None,
    ):
        self.device = device
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._onnx_session = None
        self._onnx_export_attempted = False

        # Model variant: "v1" (kinetics-finetuned), "v2" (VideoMAEv2 base)
        # Can be overridden with RALLYCUT_VIDEOMAE_MODEL env var
        import os

        self._model_source_override: str | None = None
        env_model = os.environ.get("RALLYCUT_VIDEOMAE_MODEL")
        if env_model:
            self._model_source_override = env_model
            self.model_variant = "custom"
        elif model_variant is not None:
            self.model_variant = model_variant
        else:
            self.model_variant = "v1"

        # Determine ONNX usage:
        # - Explicit use_onnx parameter takes priority
        # - ONNX disabled on MPS (CoreML provider has compatibility issues)
        # - ONNX enabled on CUDA (good acceleration)
        # - ONNX disabled on CPU (no benefit)
        if use_onnx is not None:
            self._use_onnx = use_onnx
        elif os.environ.get("RALLYCUT_DISABLE_ONNX"):
            self._use_onnx = False
        elif device == "cuda" and ONNX_AVAILABLE:
            self._use_onnx = True  # ONNX beneficial on CUDA
        else:
            self._use_onnx = False  # Disable on MPS/CPU (native PyTorch is faster)

    def _load_model(self) -> None:
        """Lazy load the model with optimizations."""
        if self._model is not None:
            return

        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        # Try to load from local path first, fallback to HuggingFace
        use_local = self.model_path and self.model_path.exists()
        if use_local:
            model_source = str(self.model_path)
        elif self._model_source_override:
            model_source = self._model_source_override
        else:
            # Use model variant mapping
            default_model = "MCG-NJU/videomae-base-finetuned-kinetics"
            model_source = self.MODEL_VARIANTS.get(self.model_variant, default_model)

        # Use half precision for CUDA with compute capability >= 7.0 (Volta+)
        # - MPS has precision issues (Input type mismatch errors)
        # - Older CUDA GPUs (pre-Volta) don't have efficient FP16 Tensor Cores
        use_half = False
        if self.device == "cuda":
            try:
                cc = torch.cuda.get_device_capability()
                # Compute capability 7.0+ has Tensor Cores for efficient FP16
                use_half = cc[0] >= 7
            except Exception:
                use_half = False
        dtype = torch.float16 if use_half else torch.float32

        if use_local:
            # Local weights already have correct 3-class structure
            self._processor = VideoMAEImageProcessor.from_pretrained(model_source)
            self._model = VideoMAEForVideoClassification.from_pretrained(
                model_source,
                dtype=dtype,  # Updated from torch_dtype (deprecated)
                low_cpu_mem_usage=True,
            )
        elif self.model_variant == "v2":
            # VideoMAEv2 uses custom model code from OpenGVLab
            # v2 doesn't ship its own processor — use v1's processor (same preprocessing)
            from transformers import AutoModel

            self._processor = VideoMAEImageProcessor.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics"
            )
            self._model = AutoModel.from_pretrained(
                model_source,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            # HuggingFace Kinetics model — processor and model from same source
            self._processor = VideoMAEImageProcessor.from_pretrained(model_source)
            self._model = VideoMAEForVideoClassification.from_pretrained(
                model_source,
                num_labels=3,  # NO_PLAY, PLAY, SERVICE
                ignore_mismatched_sizes=True,
                dtype=dtype,  # Updated from torch_dtype (deprecated)
                low_cpu_mem_usage=True,
            )

        self._model.to(self.device)
        self._model.eval()

        # Try to compile model for faster inference (PyTorch 2.0+)
        # Note: torch.compile may not work with all backends
        # MPS compilation tested - slower due to compile overhead (96s vs 50s on 2min video)
        if hasattr(torch, "compile") and self.device == "cuda":
            # CUDA gets full torch.compile optimization with max-autotune
            try:
                self._model = torch.compile(
                    self._model,
                    mode="max-autotune",  # Best performance, longer warmup
                    fullgraph=True,  # Capture entire model for optimization
                )
            except Exception:
                pass  # Fall back to uncompiled model

        # Enable inference optimizations
        if hasattr(torch, "inference_mode"):
            self._use_inference_mode = True
        else:
            self._use_inference_mode = False

    def _get_onnx_path(self) -> Path:
        """Get the path for the cached ONNX model."""
        cache_dir = _get_onnx_cache_dir()
        # Use model source hash for unique naming
        if self.model_path and self.model_path.exists():
            model_id = f"local_{self.model_path.stem}"
        else:
            model_id = "kinetics_3class"
        return cache_dir / f"videomae_{model_id}.onnx"

    def _export_to_onnx(self) -> Path | None:
        """
        Export the PyTorch model to ONNX format.

        Returns:
            Path to the exported ONNX model, or None if export failed.
        """
        if not ONNX_AVAILABLE:
            return None

        import torch

        onnx_path = self._get_onnx_path()

        # Skip if already exported
        if onnx_path.exists():
            return onnx_path

        # Ensure PyTorch model is loaded
        self._load_model()

        if self._model is None:
            return None

        try:
            import onnx

            # Create dummy input matching expected shape
            # VideoMAE expects: (batch, num_frames, num_channels, height, width)
            model_dtype = next(self._model.parameters()).dtype
            dummy_input = torch.randn(
                1,
                self.FRAME_WINDOW,
                3,
                self.IMAGE_SIZE,
                self.IMAGE_SIZE,
                device=self.device,
                dtype=model_dtype,
            )

            # Export with dynamic batch size
            torch.onnx.export(
                self._model,
                (dummy_input,),
                str(onnx_path),
                input_names=["pixel_values"],
                output_names=["logits"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                },
                opset_version=17,
                do_constant_folding=True,
            )

            # Verify the exported model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            return onnx_path

        except Exception as e:
            # Clean up failed export
            if onnx_path.exists():
                onnx_path.unlink()
            # Log but don't raise - will fall back to PyTorch
            import warnings

            warnings.warn(f"ONNX export failed: {e}. Using PyTorch backend.")
            return None

    def _load_onnx_session(self) -> bool:
        """
        Load the ONNX Runtime session.

        Returns:
            True if session was loaded successfully, False otherwise.
        """
        if self._onnx_session is not None:
            return True

        if not self._use_onnx or not ONNX_AVAILABLE:
            return False

        if self._onnx_export_attempted:
            return False

        self._onnx_export_attempted = True

        # Try to export or load cached ONNX model
        cached_path = self._get_onnx_path()
        onnx_path: Path | None = cached_path if cached_path.exists() else None
        if onnx_path is None:
            onnx_path = self._export_to_onnx()

        if onnx_path is None or not onnx_path.exists():
            return False

        try:
            # Configure session options for performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Use available providers
            providers = _get_onnx_providers(self.device)

            self._onnx_session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=providers,
            )

            return True

        except Exception as e:
            import warnings

            warnings.warn(f"Failed to load ONNX session: {e}. Using PyTorch backend.")
            self._onnx_session = None
            return False

    def _onnx_inference(self, pixel_values: np.ndarray) -> np.ndarray:
        """
        Run inference using ONNX Runtime.

        Args:
            pixel_values: Input array of shape (batch, 16, 3, 224, 224)

        Returns:
            Logits array of shape (batch, 3)
        """
        if self._onnx_session is None:
            raise RuntimeError("ONNX session not loaded")

        # Ensure float32 for ONNX Runtime
        if pixel_values.dtype != np.float32:
            pixel_values = pixel_values.astype(np.float32)

        # Run inference
        outputs = self._onnx_session.run(
            ["logits"],
            {"pixel_values": pixel_values},
        )

        result: np.ndarray = outputs[0]
        return result

    def preprocess_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Preprocess frames for VideoMAE.

        Args:
            frames: List of BGR frames from OpenCV

        Returns:
            Preprocessed frames as numpy array
        """
        processed = []

        for frame in frames:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Only resize if needed (skip if already correct size)
            h, w = rgb.shape[:2]
            if h != self.IMAGE_SIZE or w != self.IMAGE_SIZE:
                rgb = cv2.resize(rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE))

            processed.append(rgb)

        return np.array(processed)

    def classify_segment(
        self, frames: list[np.ndarray]
    ) -> tuple[GameState, float]:
        """
        Classify a segment of frames.

        Args:
            frames: List of exactly 16 BGR frames

        Returns:
            Tuple of (GameState, confidence)
        """
        import torch

        self._load_model()
        assert self._model is not None
        assert self._processor is not None

        if len(frames) != self.FRAME_WINDOW:
            raise ValueError(
                f"Expected {self.FRAME_WINDOW} frames, got {len(frames)}"
            )

        # Preprocess
        processed_frames = self.preprocess_frames(frames)

        # Use processor
        inputs = self._processor(
            list(processed_frames),
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference with best available context manager
        ctx = torch.inference_mode() if getattr(self, '_use_inference_mode', False) else torch.no_grad()
        with ctx:
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = int(probs.argmax(-1).item())
            confidence = float(probs[0, predicted_class].item())

        return self.LABEL_MAP.get(predicted_class, GameState.NO_PLAY), confidence

    def classify_segments_batch(
        self, batch_frames: list[list[np.ndarray]]
    ) -> list[tuple[GameState, float, float, float, float]]:
        """
        Classify multiple segments in a single forward pass.

        Uses ONNX Runtime if available for 1.5-2x faster inference,
        falls back to PyTorch otherwise.

        Args:
            batch_frames: List of frame lists, each containing exactly 16 BGR frames

        Returns:
            List of (GameState, confidence, no_play_prob, play_prob, service_prob) tuples.
            The confidence is the max probability (same as before for backward compat).
            Individual class probabilities enable confidence-aware boundary detection.
        """
        if not batch_frames:
            return []

        profiler = get_profiler()
        batch_size = len(batch_frames)

        # Track overall batch processing with a stage
        with profiler.stage(
            "videomae_batch", batch_size=batch_size, parent="ml_analysis"
        ) as batch_stage:
            # Validate and preprocess all segments
            with profiler.time("videomae", "preprocess", batch_size=batch_size):
                batch_processed = []
                for frames in batch_frames:
                    if len(frames) != self.FRAME_WINDOW:
                        raise ValueError(
                            f"Expected {self.FRAME_WINDOW} frames per segment, "
                            f"got {len(frames)}"
                        )
                    batch_processed.append(self.preprocess_frames(frames))

            # Try ONNX inference first (faster)
            if self._use_onnx and self._load_onnx_session():
                results = self._classify_batch_onnx(
                    batch_processed, profiler, batch_size
                )
            else:
                # Fall back to PyTorch inference
                results = self._classify_batch_pytorch(
                    batch_processed, profiler, batch_size
                )

            batch_stage.items_processed = batch_size
            batch_stage.metadata["backend"] = (
                "onnx" if self._use_onnx and self._onnx_session else "pytorch"
            )

        return results

    def _classify_batch_onnx(
        self,
        batch_processed: list[np.ndarray],
        profiler: "PerformanceProfiler",
        batch_size: int,
    ) -> list[tuple[GameState, float, float, float, float]]:
        """Run batch inference using ONNX Runtime."""
        from scipy.special import softmax

        # Load processor for normalization (still need this for consistency)
        self._load_model()
        assert self._processor is not None

        # Process through the processor to get normalized pixel values
        with profiler.time("videomae", "processor", batch_size=batch_size):
            inputs = self._processor(
                [list(frames) for frames in batch_processed],
                return_tensors="np",  # Get numpy arrays for ONNX
            )

        # ONNX inference
        with profiler.time(
            "videomae", "inference_onnx", batch_size=batch_size, device=self.device
        ):
            pixel_values = inputs["pixel_values"]
            logits = self._onnx_inference(pixel_values)

            # Apply softmax to get probabilities
            # LABEL_MAP: 0=NO_PLAY, 1=PLAY, 2=SERVICE
            probs = softmax(logits, axis=-1)
            predicted_classes = probs.argmax(axis=-1).tolist()
            confidences = probs.max(axis=-1).tolist()

        # Map to GameState with full class probabilities
        results = []
        for i, (pred_class, conf) in enumerate(zip(predicted_classes, confidences)):
            state = self.LABEL_MAP.get(pred_class, GameState.NO_PLAY)
            no_play_prob = float(probs[i, 0])
            play_prob = float(probs[i, 1])
            service_prob = float(probs[i, 2])
            results.append((state, conf, no_play_prob, play_prob, service_prob))

        return results

    def _classify_batch_pytorch(
        self,
        batch_processed: list[np.ndarray],
        profiler: "PerformanceProfiler",
        batch_size: int,
    ) -> list[tuple[GameState, float, float, float, float]]:
        """Run batch inference using PyTorch."""
        import torch

        self._load_model()
        assert self._model is not None
        assert self._processor is not None

        # Process all segments through the processor
        with profiler.time("videomae", "processor", batch_size=batch_size):
            inputs = self._processor(
                [list(frames) for frames in batch_processed],
                return_tensors="pt",
            )

        # Move to device
        with profiler.time(
            "videomae", "to_device", batch_size=batch_size, device=self.device
        ):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Batch inference
        with profiler.time(
            "videomae", "inference", batch_size=batch_size, device=self.device
        ):
            ctx = (
                torch.inference_mode()
                if getattr(self, "_use_inference_mode", False)
                else torch.no_grad()
            )
            with ctx:
                outputs = self._model(**inputs)
                # LABEL_MAP: 0=NO_PLAY, 1=PLAY, 2=SERVICE
                probs = torch.softmax(outputs.logits, dim=-1)
                predicted_classes = probs.argmax(-1).tolist()
                confidences = probs.max(-1).values.tolist()
                # Extract individual class probabilities
                probs_np = probs.cpu().numpy()

        # Map to GameState with full class probabilities
        results = []
        for i, (pred_class, conf) in enumerate(zip(predicted_classes, confidences)):
            state = self.LABEL_MAP.get(pred_class, GameState.NO_PLAY)
            no_play_prob = float(probs_np[i, 0])
            play_prob = float(probs_np[i, 1])
            service_prob = float(probs_np[i, 2])
            results.append((state, conf, no_play_prob, play_prob, service_prob))

        return results

    def classify_video_frames(
        self,
        frames: list[np.ndarray],
        stride: int = 8,
        progress_callback: "Callable[[float], None] | None" = None,
    ) -> list[GameStateResult]:
        """
        Classify game states across a sequence of frames.

        Args:
            frames: List of BGR frames
            stride: Number of frames between classification windows
            progress_callback: Optional callback for progress updates

        Returns:
            List of GameStateResult for each window
        """
        results = []
        total_windows = max(1, (len(frames) - self.FRAME_WINDOW) // stride + 1)

        for i, start_idx in enumerate(
            range(0, len(frames) - self.FRAME_WINDOW + 1, stride)
        ):
            window_frames = frames[start_idx : start_idx + self.FRAME_WINDOW]
            state, confidence = self.classify_segment(window_frames)

            results.append(
                GameStateResult(
                    state=state,
                    confidence=confidence,
                    start_frame=start_idx,
                    end_frame=start_idx + self.FRAME_WINDOW - 1,
                )
            )

            if progress_callback:
                progress_callback((i + 1) / total_windows)

        return results

    def get_active_frame_ranges(
        self,
        results: list[GameStateResult],
        min_confidence: float = 0.7,
    ) -> list[tuple[int, int]]:
        """
        Extract frame ranges where game is active (SERVICE or PLAY).

        Merges adjacent active ranges for efficiency.

        Args:
            results: Classification results
            min_confidence: Minimum confidence threshold

        Returns:
            List of (start_frame, end_frame) tuples
        """
        active_ranges = []
        current_start = None
        current_end = None

        for result in results:
            is_active = (
                result.state in (GameState.SERVICE, GameState.PLAY)
                and result.confidence >= min_confidence
            )

            if is_active:
                if current_start is None:
                    current_start = result.start_frame
                current_end = result.end_frame
            else:
                if current_start is not None:
                    assert current_end is not None
                    active_ranges.append((current_start, current_end))
                    current_start = None
                    current_end = None

        # Don't forget last range
        if current_start is not None:
            assert current_end is not None
            active_ranges.append((current_start, current_end))

        return active_ranges

    def get_encoder_features_batch(
        self, batch_frames: list[list[np.ndarray]], pooling: str = "cls"
    ) -> np.ndarray:
        """
        Extract encoder features from video segments (before classification head).

        This method returns the 768-dimensional features from the VideoMAE encoder,
        which can be used for temporal modeling (BiLSTM, 1D Conv, etc.).

        Args:
            batch_frames: List of frame lists, each containing exactly 16 BGR frames.
            pooling: How to pool patch features. Options:
                - "cls": Use the CLS token (position 0) - default
                - "mean": Mean pooling over all patch tokens

        Returns:
            NumPy array of shape (batch_size, 768) with encoder features.
        """
        import torch

        if not batch_frames:
            return np.array([])

        self._load_model()
        assert self._model is not None
        assert self._processor is not None

        profiler = get_profiler()
        batch_size = len(batch_frames)

        with profiler.stage(
            "feature_extraction", batch_size=batch_size, parent="ml_analysis"
        ):
            # Validate and preprocess all segments
            with profiler.time("videomae", "preprocess", batch_size=batch_size):
                batch_processed = []
                for frames in batch_frames:
                    if len(frames) != self.FRAME_WINDOW:
                        raise ValueError(
                            f"Expected {self.FRAME_WINDOW} frames per segment, "
                            f"got {len(frames)}"
                        )
                    batch_processed.append(self.preprocess_frames(frames))

            # Process through processor
            with profiler.time("videomae", "processor", batch_size=batch_size):
                inputs = self._processor(
                    [list(frames) for frames in batch_processed],
                    return_tensors="pt",
                )

            # Move to device
            with profiler.time(
                "videomae", "to_device", batch_size=batch_size, device=self.device
            ):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract encoder features (not going through classifier head)
            with profiler.time(
                "videomae", "encoder_forward", batch_size=batch_size, device=self.device
            ):
                ctx = (
                    torch.inference_mode()
                    if getattr(self, "_use_inference_mode", False)
                    else torch.no_grad()
                )
                with ctx:
                    # Access the underlying VideoMAE encoder directly
                    # VideoMAEForVideoClassification has: videomae (encoder), fc_norm, classifier
                    encoder = self._model.videomae
                    outputs = encoder(pixel_values=inputs["pixel_values"])

                    # outputs.last_hidden_state: (batch, num_patches+1, hidden_size)
                    # Position 0 is the CLS token, rest are patch tokens
                    hidden_states = outputs.last_hidden_state

                    if pooling == "cls":
                        # Use CLS token
                        features = hidden_states[:, 0, :]  # (batch, 768)
                    elif pooling == "mean":
                        # Mean pooling over all tokens
                        features = hidden_states.mean(dim=1)  # (batch, 768)
                    else:
                        raise ValueError(f"Unknown pooling method: {pooling}")

                    # Apply the layer norm (fc_norm) for consistency with classification
                    features = self._model.fc_norm(features)
                    features_np: np.ndarray = features.cpu().numpy()

        return features_np

    def get_encoder_features(
        self, frames: list[np.ndarray], pooling: str = "cls"
    ) -> np.ndarray:
        """
        Extract encoder features from a single video segment.

        Args:
            frames: List of exactly 16 BGR frames.
            pooling: How to pool patch features ("cls" or "mean").

        Returns:
            NumPy array of shape (768,) with encoder features.
        """
        features = self.get_encoder_features_batch([frames], pooling=pooling)
        result: np.ndarray = features[0]
        return result
