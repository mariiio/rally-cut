"""Export WASB HRNet to ONNX format for faster inference.

Usage:
    cd analysis
    uv run python scripts/export_wasb_onnx.py

    # Verify only (if ONNX already exists)
    uv run python scripts/export_wasb_onnx.py --verify-only

    # Custom weights path
    uv run python scripts/export_wasb_onnx.py --weights path/to/weights.pth.tar
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rallycut.tracking.wasb_model import (
    IMG_HEIGHT,
    IMG_WIDTH,
    WEIGHTS_DIR,
    HRNet,
    load_wasb_model,
)


class HRNetWrapper(nn.Module):
    """Wrapper to flatten HRNet's dict output to a single tensor for ONNX."""

    def __init__(self, hrnet: HRNet) -> None:
        super().__init__()
        self.hrnet = hrnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hrnet(x)
        return out[0]  # Extract scale 0: (B, 3, H, W)


def export_onnx(
    weights_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Export WASB HRNet to ONNX with dynamic batch axis.

    Returns:
        Path to the exported ONNX model.
    """
    if output_path is None:
        output_path = WEIGHTS_DIR / "wasb_volleyball_best.onnx"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading WASB HRNet weights...")
    model = load_wasb_model(weights_path, device="cpu")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e6:.1f}M")

    wrapper = HRNetWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 9, IMG_HEIGHT, IMG_WIDTH)

    print("Exporting to ONNX (opset 17, dynamic batch)...")
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["heatmaps"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "heatmaps": {0: "batch_size"},
        },
        dynamo=False,  # Legacy exporter required for CoreMLExecutionProvider
    )
    export_time = time.time() - t0
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Exported in {export_time:.1f}s: {output_path} ({file_size_mb:.1f}MB)")

    return output_path


def verify_onnx(onnx_path: Path, weights_path: Path | None = None) -> bool:
    """Verify ONNX model produces same outputs as PyTorch."""
    import onnxruntime as ort

    print(f"Verifying ONNX model: {onnx_path}")

    model = load_wasb_model(weights_path, device="cpu")
    wrapper = HRNetWrapper(model)
    wrapper.eval()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Test with multiple batch sizes
    for batch_size in [1, 4, 8]:
        test_input = torch.randn(batch_size, 9, IMG_HEIGHT, IMG_WIDTH)

        with torch.inference_mode():
            pytorch_out = wrapper(test_input).numpy()

        onnx_out = sess.run(None, {"input": test_input.numpy()})[0]

        max_diff = float(np.max(np.abs(pytorch_out - onnx_out)))
        mean_diff = float(np.mean(np.abs(pytorch_out - onnx_out)))

        status = "PASS" if max_diff < 1e-4 else "FAIL"
        print(f"  batch={batch_size}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f} [{status}]")

        if max_diff >= 1e-4:
            print(f"  VERIFICATION FAILED for batch_size={batch_size}")
            return False

    print("  All batch sizes verified successfully")
    return True


def benchmark_onnx(onnx_path: Path) -> None:
    """Quick benchmark comparing ONNX vs PyTorch speed."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = []
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    active_provider = sess.get_providers()[0] if sess.get_providers() else "unknown"
    print(f"\nBenchmark (provider: {active_provider}):")

    for batch_size in [1, 4, 8]:
        test_input = np.random.randn(batch_size, 9, IMG_HEIGHT, IMG_WIDTH).astype(np.float32)

        # Warmup
        for _ in range(3):
            sess.run(None, {"input": test_input})

        # Benchmark
        n_iters = 20
        t0 = time.time()
        for _ in range(n_iters):
            sess.run(None, {"input": test_input})
        elapsed = time.time() - t0

        fps = (n_iters * batch_size) / elapsed
        ms_per_batch = elapsed / n_iters * 1000
        print(f"  batch={batch_size}: {fps:.1f} FPS ({ms_per_batch:.1f}ms/batch)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WASB HRNet to ONNX")
    parser.add_argument("--weights", default=None, help="Path to PyTorch weights")
    parser.add_argument("--output", default=None, help="Output ONNX path")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing ONNX")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    args = parser.parse_args()

    weights_path = Path(args.weights) if args.weights else None
    output_path = Path(args.output) if args.output else WEIGHTS_DIR / "wasb_volleyball_best.onnx"

    if args.verify_only:
        if not output_path.exists():
            print(f"ONNX model not found: {output_path}")
            sys.exit(1)
        ok = verify_onnx(output_path, weights_path)
        if args.benchmark:
            benchmark_onnx(output_path)
        sys.exit(0 if ok else 1)

    onnx_path = export_onnx(weights_path, output_path)
    ok = verify_onnx(onnx_path, weights_path)

    if args.benchmark:
        benchmark_onnx(onnx_path)

    if not ok:
        print("\nExport completed but verification FAILED!")
        sys.exit(1)

    print("\nExport and verification successful!")


if __name__ == "__main__":
    main()
