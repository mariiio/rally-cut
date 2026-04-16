"""BGR-safe augmentations for crop training.

Operates on (H, W, 3) BGR uint8 numpy arrays. Output: (3, 224, 224) BGR uint8
torch tensor (CHW). The backbone wrapper handles BGR→RGB + ImageNet normalize.

Augmentations (matching the plan):
- random crop with ±5% bbox jitter, then resize to 224×224
- random horizontal flip (p=0.5)
- random brightness (β ∈ [-10, 10]) + contrast (α ∈ [0.9, 1.1])
- random erasing (p=0.3, area ∈ [0.02, 0.15], aspect ∈ [0.3, 3.3])
- NO hue/saturation jitter — kit color is partial signal we want to retain
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

OUTPUT_SIZE = 224


def _crop_with_jitter(img: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    """Random crop ±5% then resize back to 224×224."""
    h, w = img.shape[:2]
    crop_frac = rng.uniform(0.90, 1.0)  # 90-100% area kept (≈ ±5% jitter on each edge)
    new_h = max(16, int(h * crop_frac))
    new_w = max(8, int(w * crop_frac))
    top = int(rng.integers(0, max(1, h - new_h + 1)))
    left = int(rng.integers(0, max(1, w - new_w + 1)))
    return img[top : top + new_h, left : left + new_w]


def _brightness_contrast(img: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    alpha = rng.uniform(0.9, 1.1)
    beta = rng.uniform(-10.0, 10.0)
    out = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return out


def _random_erase(img: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    """Erase a random rectangle, fill with channel-wise mean."""
    h, w = img.shape[:2]
    area = h * w
    erase_area = rng.uniform(0.02, 0.15) * area
    aspect = rng.uniform(0.3, 3.3)
    eh = int(np.sqrt(erase_area / aspect))
    ew = int(np.sqrt(erase_area * aspect))
    if eh < 1 or ew < 1 or eh >= h or ew >= w:
        return img
    top = int(rng.integers(0, h - eh))
    left = int(rng.integers(0, w - ew))
    fill = np.array([
        int(img[..., 0].mean()), int(img[..., 1].mean()), int(img[..., 2].mean()),
    ], dtype=np.uint8)
    out = img.copy()
    out[top : top + eh, left : left + ew] = fill
    return out


def augment_train(img: NDArray[np.uint8], rng: np.random.Generator) -> torch.Tensor:
    """Apply full train-time augmentation, return a (3, 224, 224) BGR uint8 torch tensor."""
    if img is None or img.size == 0 or img.shape[0] < 4 or img.shape[1] < 4:
        # Fall back to a neutral 224×224 gray patch if input is degenerate.
        out = np.full((OUTPUT_SIZE, OUTPUT_SIZE, 3), 128, dtype=np.uint8)
        return torch.from_numpy(out).permute(2, 0, 1).contiguous()

    aug = _crop_with_jitter(img, rng)
    aug = cv2.resize(aug, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        aug = cv2.flip(aug, 1)
    aug = _brightness_contrast(aug, rng)
    if rng.random() < 0.3:
        aug = _random_erase(aug, rng)
    return torch.from_numpy(aug).permute(2, 0, 1).contiguous()


def augment_eval(img: NDArray[np.uint8]) -> torch.Tensor:
    """Eval-time: resize-only. (3, 224, 224) BGR uint8 torch tensor."""
    if img is None or img.size == 0 or img.shape[0] < 4 or img.shape[1] < 4:
        out = np.full((OUTPUT_SIZE, OUTPUT_SIZE, 3), 128, dtype=np.uint8)
        return torch.from_numpy(out).permute(2, 0, 1).contiguous()
    out = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(out).permute(2, 0, 1).contiguous()
