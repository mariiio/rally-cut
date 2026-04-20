"""Per-candidate crop dataset for contact classification.

Loads cached (player_crop, ball_patch, label, meta) tuples from disk.
Crops are pre-extracted by scripts/extract_crop_dataset.py to avoid
re-decoding video on every epoch (slow + non-reproducible if decoding
is non-deterministic).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CropContactDataset(Dataset):
    """Dataset of pre-extracted (player_crop[T,3,64,64], ball_patch[T,3,32,32], label).

    Crops stored as .npz files, one per candidate, under a root directory
    organized by video_id for easy held-out splitting.
    """

    def __init__(self, cache_root: Path, video_ids: list[str]) -> None:
        self.cache_root = Path(cache_root)
        self.video_ids = list(video_ids)
        self.items: list[tuple[str, str]] = []
        for vid in self.video_ids:
            vid_dir = self.cache_root / vid
            if not vid_dir.exists():
                continue
            for f in sorted(vid_dir.glob("*.npz")):
                self.items.append((vid, str(f)))
        if not self.items:
            raise ValueError(
                f"No crops found under {cache_root} for {len(self.video_ids)} videos"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        vid, path = self.items[idx]
        data = np.load(path)
        return {
            "video_id": vid,
            "rally_id": str(data["rally_id"]),
            "frame": int(data["frame"]),
            "player_crop": torch.from_numpy(data["player_crop"]).float(),
            "ball_patch": torch.from_numpy(data["ball_patch"]).float(),
            "label": int(data["label"]),
            "gbm_conf": float(data["gbm_conf"]),
            "source": str(data["source"]),
        }
