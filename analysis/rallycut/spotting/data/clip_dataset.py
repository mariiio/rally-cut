"""Unified clip dataset for training E2E-Spot.

Samples fixed-length clips from rallies with oversampling of event-containing
clips to combat class imbalance. Supports in-memory frame caching to
eliminate disk I/O during training.
"""

from __future__ import annotations

import functools
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from rallycut.spotting.data.beach import RallyInfo
from rallycut.spotting.data.transforms import ClipTransform

# Force unbuffered output for Modal log streaming
print = functools.partial(print, flush=True)


def preload_all_frames(rallies: list[RallyInfo]) -> dict[str, torch.Tensor]:
    """Load all frames as uint8 tensors (4x smaller than float32).

    Returns a dict mapping rally_id -> (N, 3, H, W) uint8 tensor.
    61K frames at 224×~400 × uint8 ≈ 16GB (fits in 48GB container).
    Converted to float32 per-batch in __getitem__.
    """
    cache: dict[str, torch.Tensor] = {}
    total_frames = 0
    t0 = time.time()

    for rally in rallies:
        frames: list[torch.Tensor] = []
        for i in range(rally.frame_count):
            path = rally.frame_dir / f"{i:06d}.jpg"
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Keep as uint8 to save 4x memory
                    tensor = torch.from_numpy(img).permute(2, 0, 1)
                    frames.append(tensor)
                    continue
            # Missing/failed frame: repeat last or black
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(torch.zeros(3, 224, 224, dtype=torch.uint8))

        cache[rally.rally_id] = torch.stack(frames)
        total_frames += len(frames)

    elapsed = time.time() - t0
    print(f"  Preloaded {total_frames} frames from {len(rallies)} rallies in {elapsed:.0f}s")
    return cache


class ClipDataset(Dataset):
    """Dataset that yields fixed-length video clips with per-frame labels.

    For training, clips containing events are oversampled by a configurable
    factor to address the extreme background/event class imbalance.

    When frame_cache is provided, frames are served from memory instead of
    reading from disk, eliminating I/O as a bottleneck.
    """

    def __init__(
        self,
        rallies: list[RallyInfo],
        clip_length: int = 96,
        transform: ClipTransform | None = None,
        oversample_events: int = 3,
        is_train: bool = True,
        frame_cache: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.rallies = rallies
        self.clip_length = clip_length
        self.transform = transform or ClipTransform(is_train=is_train)
        self.is_train = is_train
        self._frame_cache = frame_cache

        # Build clip index: (rally_idx, start_frame)
        self._clips: list[tuple[int, int]] = []
        self._build_clip_index(oversample_events if is_train else 1)

    def _build_clip_index(self, oversample_events: int) -> None:
        """Pre-compute clip start positions for all rallies."""
        stride = self.clip_length // 2 if self.is_train else self.clip_length

        for rally_idx, rally in enumerate(self.rallies):
            if rally.frame_count < self.clip_length:
                # Short rally: use what we have (padded during loading)
                self._clips.append((rally_idx, 0))
                continue

            for start in range(0, rally.frame_count - self.clip_length + 1, stride):
                clip_labels = rally.labels[start : start + self.clip_length]
                has_event = np.any(clip_labels > 0)

                if has_event:
                    # Oversample event-containing clips
                    for _ in range(oversample_events):
                        self._clips.append((rally_idx, start))
                else:
                    self._clips.append((rally_idx, start))

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rally_idx, start = self._clips[idx]
        rally = self.rallies[rally_idx]

        # Add random jitter to start position during training
        if self.is_train and rally.frame_count > self.clip_length:
            max_jitter = self.clip_length // 4
            jitter = random.randint(-max_jitter, max_jitter)
            start = max(0, min(start + jitter, rally.frame_count - self.clip_length))

        end = min(start + self.clip_length, rally.frame_count)
        actual_len = end - start

        # Load frames (from cache or disk)
        if self._frame_cache is not None and rally.rally_id in self._frame_cache:
            cached = self._frame_cache[rally.rally_id]  # (N, 3, H, W) uint8 tensor
            end_idx = min(end, cached.shape[0])
            # Batch slice + convert (one op, not per-frame loop)
            chunk = cached[start:end_idx].float().div_(255.0)
            frames = list(chunk.unbind(0))
        else:
            frames = _load_frames_from_disk(rally.frame_dir, start, end)

        # Get labels and offsets for this clip
        labels = rally.labels[start:end].copy()
        offsets = rally.offsets[start:end].copy()

        # Pad if needed (short rallies)
        if actual_len < self.clip_length:
            pad_len = self.clip_length - actual_len
            frames = frames + [frames[-1]] * pad_len  # repeat last frame
            labels = np.pad(labels, (0, pad_len), constant_values=0)
            offsets = np.pad(offsets, (0, pad_len), constant_values=0)

        # Apply transforms
        clip = self.transform(frames)  # (T, 3, H, W)

        # Create event mask (1 where there's a non-background label)
        event_mask = torch.from_numpy((labels > 0).astype(np.float32))

        return {
            "clip": clip,
            "labels": torch.from_numpy(labels),
            "offsets": torch.from_numpy(offsets),
            "event_mask": event_mask,
        }


def _load_frames_from_disk(frame_dir: Path, start: int, end: int) -> list[torch.Tensor]:
    """Load frames from pre-extracted JPGs.

    Returns list of (3, H, W) float tensors in [0, 1].
    """
    frames: list[torch.Tensor] = []
    for i in range(start, end):
        path = frame_dir / f"{i:06d}.jpg"
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                frames.append(tensor)
                continue
        # Missing frame: use black image or previous
        if frames:
            frames.append(frames[-1].clone())
        else:
            frames.append(torch.zeros(3, 224, 224))

    return frames


def collate_clips(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for ClipDataset batches."""
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "offsets": torch.stack([b["offsets"] for b in batch]),
        "event_mask": torch.stack([b["event_mask"] for b in batch]),
    }
