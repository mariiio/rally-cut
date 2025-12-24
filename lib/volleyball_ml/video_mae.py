"""
VideoMAE adapter for game state classification.

Adapted from volleyball_analytics for RallyCut CLI use.
Original: https://github.com/masouduut94/volleyball_analytics
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from rallycut.core.config import get_config
from rallycut.core.models import GameState, GameStateResult
from rallycut.core.profiler import get_profiler


class GameStateClassifier:
    """
    VideoMAE-based game state classification.

    Classifies video segments as SERVICE, PLAY, or NO_PLAY.
    Uses 16-frame windows as input.
    """

    FRAME_WINDOW = 16
    IMAGE_SIZE = 224
    # Trained model uses 3 classes
    LABEL_MAP = {0: GameState.NO_PLAY, 1: GameState.PLAY, 2: GameState.SERVICE}

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.model_path = model_path
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load the model with optimizations."""
        if self._model is not None:
            return

        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        config = get_config()

        # Try to load from local path first, fallback to HuggingFace
        use_local = self.model_path and self.model_path.exists()
        if use_local:
            model_source = str(self.model_path)
        else:
            # Use pretrained model from HuggingFace
            # In production, this would be a fine-tuned volleyball model
            model_source = "MCG-NJU/videomae-base-finetuned-kinetics"

        self._processor = VideoMAEImageProcessor.from_pretrained(model_source)

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
            self._model = VideoMAEForVideoClassification.from_pretrained(
                model_source,
                dtype=dtype,  # Updated from torch_dtype (deprecated)
                low_cpu_mem_usage=True,
            )
        else:
            # HuggingFace Kinetics model needs classifier head replaced
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
        if hasattr(torch, "compile") and self.device != "mps":
            # MPS doesn't fully support torch.compile yet
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead")
            except Exception:
                pass  # Fall back to uncompiled model

        # Enable inference optimizations
        if hasattr(torch, "inference_mode"):
            self._use_inference_mode = True
        else:
            self._use_inference_mode = False

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
            predicted_class = probs.argmax(-1).item()
            confidence = probs[0, predicted_class].item()

        return self.LABEL_MAP.get(predicted_class, GameState.NO_PLAY), confidence

    def classify_segments_batch(
        self, batch_frames: list[list[np.ndarray]]
    ) -> list[tuple[GameState, float]]:
        """
        Classify multiple segments in a single forward pass.

        Args:
            batch_frames: List of frame lists, each containing exactly 16 BGR frames

        Returns:
            List of (GameState, confidence) tuples for each segment
        """
        import torch

        self._load_model()

        if not batch_frames:
            return []

        profiler = get_profiler()
        batch_size = len(batch_frames)

        # Validate and preprocess all segments
        with profiler.time("videomae", "preprocess", batch_size=batch_size):
            batch_processed = []
            for frames in batch_frames:
                if len(frames) != self.FRAME_WINDOW:
                    raise ValueError(
                        f"Expected {self.FRAME_WINDOW} frames per segment, got {len(frames)}"
                    )
                batch_processed.append(self.preprocess_frames(frames))

        # Process all segments through the processor
        # VideoMAE processor expects list of video frames
        with profiler.time("videomae", "processor", batch_size=batch_size):
            inputs = self._processor(
                [list(frames) for frames in batch_processed],
                return_tensors="pt",
            )

        # Move to device
        with profiler.time("videomae", "to_device", batch_size=batch_size, device=self.device):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Batch inference
        with profiler.time("videomae", "inference", batch_size=batch_size, device=self.device):
            ctx = torch.inference_mode() if getattr(self, '_use_inference_mode', False) else torch.no_grad()
            with ctx:
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predicted_classes = probs.argmax(-1).tolist()
                confidences = probs.max(-1).values.tolist()

        # Map to GameState
        results = []
        for pred_class, conf in zip(predicted_classes, confidences):
            state = self.LABEL_MAP.get(pred_class, GameState.NO_PLAY)
            results.append((state, conf))

        return results

    def classify_video_frames(
        self,
        frames: list[np.ndarray],
        stride: int = 8,
        progress_callback: Optional[callable] = None,
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
                    active_ranges.append((current_start, current_end))
                    current_start = None
                    current_end = None

        # Don't forget last range
        if current_start is not None:
            active_ranges.append((current_start, current_end))

        return active_ranges
