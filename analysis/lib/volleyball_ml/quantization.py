"""INT8 quantization for VideoMAE ONNX model.

Provides up to 2x inference speedup on compatible hardware (CUDA with INT8 Tensor Cores,
x86 with VNNI/AVX512). Falls back to FP32 if quantization is not supported.

Quantization methods:
1. Dynamic quantization: Fastest to apply, moderate speedup
2. Static quantization: Requires calibration data, best speedup

Usage:
    from lib.volleyball_ml.quantization import quantize_onnx_model, load_quantized_model

    # Quantize an existing ONNX model
    quant_path = quantize_onnx_model(onnx_path, method="dynamic")

    # Load and use quantized model
    session = load_quantized_model(quant_path, device="cuda")
"""

from pathlib import Path

# Check for quantization support
try:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False


def is_quantization_supported() -> bool:
    """Check if INT8 quantization is supported."""
    return QUANTIZATION_AVAILABLE


def quantize_onnx_model(
    onnx_path: Path | str,
    output_path: Path | str | None = None,
    method: str = "dynamic",
) -> Path | None:
    """
    Quantize an ONNX model to INT8.

    Args:
        onnx_path: Path to the FP32 ONNX model
        output_path: Path for quantized model (default: adds _int8 suffix)
        method: Quantization method ("dynamic" or "static")

    Returns:
        Path to quantized model, or None if quantization failed.
    """
    if not QUANTIZATION_AVAILABLE:
        import warnings

        warnings.warn("ONNX Runtime quantization not available. Install with: pip install onnxruntime")
        return None

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"
    else:
        output_path = Path(output_path)

    # Skip if already quantized
    if output_path.exists():
        return output_path

    try:
        if method == "dynamic":
            # Dynamic quantization - fastest to apply
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
                extra_options={
                    "ActivationSymmetric": True,
                    "WeightSymmetric": True,
                },
            )
        else:
            raise ValueError(f"Unsupported quantization method: {method}")

        return output_path

    except Exception as e:
        import warnings

        warnings.warn(f"Quantization failed: {e}")
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        return None


def load_quantized_model(
    onnx_path: Path | str,
    device: str = "cpu",
):
    """
    Load a quantized ONNX model with optimal execution providers.

    Args:
        onnx_path: Path to quantized ONNX model
        device: Device to run on ("cpu", "cuda")

    Returns:
        ONNX Runtime InferenceSession
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("ONNX Runtime required. Install with: pip install onnxruntime") from e

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_path}")

    # Configure session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Select providers based on device
    if device == "cuda":
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
    )

    return session


class QuantizedClassifier:
    """
    Quantized VideoMAE classifier using INT8 ONNX model.

    Provides the same interface as GameStateClassifier but uses
    a quantized model for faster inference.
    """

    FRAME_WINDOW = 16
    IMAGE_SIZE = 224
    LABEL_MAP = {0: "NO_PLAY", 1: "PLAY", 2: "SERVICE"}

    def __init__(
        self,
        onnx_path: Path | str,
        device: str = "cpu",
        quantize: bool = True,
    ):
        from rallycut.core.models import GameState

        self.device = device
        self._session = None
        self._processor = None

        self.LABEL_MAP = {
            0: GameState.NO_PLAY,
            1: GameState.PLAY,
            2: GameState.SERVICE,
        }

        # Quantize if requested
        onnx_path = Path(onnx_path)
        if quantize and QUANTIZATION_AVAILABLE:
            quant_path = quantize_onnx_model(onnx_path)
            if quant_path is not None:
                onnx_path = quant_path

        self._onnx_path = onnx_path

    def _load_session(self) -> None:
        """Load the ONNX session."""
        if self._session is not None:
            return

        self._session = load_quantized_model(self._onnx_path, self.device)

        # Load processor for preprocessing
        from transformers import VideoMAEImageProcessor

        self._processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )

    def classify_segments_batch(
        self, batch_frames: list[list]
    ) -> list[tuple]:
        """
        Classify multiple segments using quantized model.

        Args:
            batch_frames: List of frame lists (16 BGR frames each)

        Returns:
            List of (GameState, confidence) tuples
        """
        import numpy as np
        from scipy.special import softmax

        self._load_session()

        if not batch_frames:
            return []

        # Preprocess
        processed = []
        for frames in batch_frames:
            if len(frames) != self.FRAME_WINDOW:
                raise ValueError(f"Expected {self.FRAME_WINDOW} frames, got {len(frames)}")

            import cv2

            rgb_frames = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (self.IMAGE_SIZE, self.IMAGE_SIZE):
                    rgb = cv2.resize(rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                rgb_frames.append(rgb)
            processed.append(np.array(rgb_frames))

        # Use processor for normalization
        inputs = self._processor(
            [list(frames) for frames in processed],
            return_tensors="np",
        )

        # Run inference
        pixel_values = inputs["pixel_values"].astype(np.float32)
        outputs = self._session.run(["logits"], {"pixel_values": pixel_values})
        logits = outputs[0]

        # Get predictions
        probs = softmax(logits, axis=-1)
        predicted_classes = probs.argmax(axis=-1).tolist()
        confidences = probs.max(axis=-1).tolist()

        # Map to GameState
        results = []
        for pred_class, conf in zip(predicted_classes, confidences):
            state = self.LABEL_MAP.get(pred_class, self.LABEL_MAP[0])
            results.append((state, conf))

        return results


def get_quantization_info(onnx_path: Path | str) -> dict:
    """
    Get information about a quantized model.

    Returns:
        Dictionary with model info including size and quantization status.
    """
    onnx_path = Path(onnx_path)

    info = {
        "path": str(onnx_path),
        "exists": onnx_path.exists(),
        "size_mb": 0,
        "is_quantized": False,
    }

    if onnx_path.exists():
        info["size_mb"] = onnx_path.stat().st_size / (1024 * 1024)
        info["is_quantized"] = "_int8" in onnx_path.stem

    return info
