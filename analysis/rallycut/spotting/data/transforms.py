"""Temporal-consistent augmentation pipeline for video clips.

All transforms use batched tensor ops only — no per-frame Python loops.
Beach-specific: strong brightness/contrast jitter for sun and sand.
"""

from __future__ import annotations

import random

import torch
from torchvision import transforms


class ClipTransform:
    """Fast batched transforms for video clips. No per-frame loops."""

    def __init__(self, size: int = 224, is_train: bool = True) -> None:
        self.size = size
        self.is_train = is_train
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, frames: list[torch.Tensor]) -> torch.Tensor:
        if self.is_train:
            clip = self._train_transform(frames)
        else:
            clip = self._val_transform(frames)
        clip = (clip - self._mean) / self._std
        return clip

    def _train_transform(self, frames: list[torch.Tensor]) -> torch.Tensor:
        _, h, w = frames[0].shape

        # Random resized crop (same params, all frames)
        i, j, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
            frames[0], scale=(0.6, 1.0), ratio=(0.75, 1.33)
        )
        cropped = torch.stack([f[:, i : i + crop_h, j : j + crop_w] for f in frames])
        clip = torch.nn.functional.interpolate(
            cropped, size=(self.size, self.size), mode="bilinear", align_corners=False
        )

        # Horizontal flip
        if random.random() > 0.5:
            clip = clip.flip(-1)

        # Strong brightness (beach sun/sand) — batched multiply
        if random.random() > 0.3:
            clip = clip * random.uniform(0.5, 1.5)

        # Contrast — batched
        if random.random() > 0.3:
            mean = clip.mean(dim=(-2, -1), keepdim=True)
            factor = random.uniform(0.6, 1.4)
            clip = factor * clip + (1.0 - factor) * mean

        # Saturation — batched
        if random.random() > 0.5:
            gray = clip.mean(dim=1, keepdim=True)
            factor = random.uniform(0.5, 1.5)
            clip = factor * clip + (1.0 - factor) * gray

        # Small random translation (camera wobble — batched roll)
        if random.random() > 0.7:
            px = random.randint(-3, 3)
            py = random.randint(-3, 3)
            if px != 0 or py != 0:
                clip = torch.roll(clip, shifts=(py, px), dims=(-2, -1))

        clip = clip.clamp(0, 1)
        return clip

    def _val_transform(self, frames: list[torch.Tensor]) -> torch.Tensor:
        clip = torch.stack(frames)
        clip = torch.nn.functional.interpolate(
            clip, size=(self.size + 32, self.size + 32), mode="bilinear", align_corners=False
        )
        margin = 16
        clip = clip[:, :, margin : margin + self.size, margin : margin + self.size]
        return clip
