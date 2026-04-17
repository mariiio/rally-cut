"""PyTorch dataset for within-team ReID training.

Indexed by pair index. `__getitem__(i)` samples one crop per track from the
pair's window_frames, applies augmentation, and returns:

    - crop_a: (3, 224, 224) BGR uint8 tensor
    - crop_b: (3, 224, 224) BGR uint8 tensor
    - identity_a: long — encoding of (rally_id, canonical_a) — same id appearing
        in multiple batch slots gives SupCon real positive sets
    - identity_b: long — encoding of (rally_id, canonical_b)
    - team_a: long — encoding of (rally_id, team_a)
    - team_b: long — encoding of (rally_id, team_b)
    - tier_idx: long — 0=positive, 1=easy_neg, 2=mid, 3=gold (used for label
        smoothing and teammate-margin gating)
    - pair_idx: long — original pair index (debugging only)

Identity encoding: a deterministic int derived once from the corpus's full
(rally_id, canonical_id) string. The training process builds the mapping from
the manifest so different epochs see consistent integer labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment_eval, augment_train
from .manifest import Pair

logger = logging.getLogger("within_team_reid.data.dataset")

TIER_TO_IDX: dict[str, int] = {
    "positive": 0,
    "easy_neg": 1,
    "mid": 2,
    "gold": 3,
}


def build_identity_encoding(pairs: list[Pair]) -> dict[tuple[str, int], int]:
    """Stable mapping (rally_id, canonical_id) → int across the corpus."""
    seen: list[tuple[str, int]] = []
    seen_set: set[tuple[str, int]] = set()
    for p in pairs:
        for tk in (p.track_a, p.track_b):
            key = (p.rally_id, tk.canonical_id)
            if key not in seen_set:
                seen_set.add(key)
                seen.append(key)
    return {k: i for i, k in enumerate(seen)}


def build_team_encoding(pairs: list[Pair]) -> dict[tuple[str, int], int]:
    """Stable mapping (rally_id, team) → int across the corpus."""
    seen: list[tuple[str, int]] = []
    seen_set: set[tuple[str, int]] = set()
    for p in pairs:
        for tk in (p.track_a, p.track_b):
            key = (p.rally_id, tk.team)
            if key not in seen_set:
                seen_set.add(key)
                seen.append(key)
    return {k: i for i, k in enumerate(seen)}


def _load_crop(corpus_root: Path, rally_id: str, track_id: int, frame: int) -> np.ndarray | None:
    """Read crop JPEG. Returns BGR uint8 ndarray or None if missing/corrupt."""
    path = corpus_root / "crops" / rally_id / f"t{track_id}" / f"{frame:06d}.jpg"
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img if img is not None and img.size > 0 else None


def _sample_frame(rng: np.random.Generator, frames: tuple[int, ...]) -> int:
    """Random frame from window_frames. Caller guarantees non-empty."""
    idx = int(rng.integers(0, len(frames)))
    return int(frames[idx])


class PairDataset(Dataset):
    """Pair-indexed dataset; epoch length controlled by sampler."""

    def __init__(
        self,
        pairs: list[Pair],
        corpus_root: Path,
        identity_encoding: dict[tuple[str, int], int],
        team_encoding: dict[tuple[str, int], int],
        seed: int = 42,
        train_mode: bool = True,
    ) -> None:
        self.pairs = pairs
        self.corpus_root = corpus_root
        self.identity_encoding = identity_encoding
        self.team_encoding = team_encoding
        self.seed = seed
        self.train_mode = train_mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        # Per-worker seeded RNG so augmentation is reproducible within a process.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + worker_id * 1_000_003 + idx)

        pair = self.pairs[idx]

        crop_a, crop_b = self._sample_crop_pair(pair, rng)

        if self.train_mode:
            tensor_a = augment_train(crop_a, rng)
            tensor_b = augment_train(crop_b, rng)
        else:
            tensor_a = augment_eval(crop_a)
            tensor_b = augment_eval(crop_b)

        ident_a = self.identity_encoding[(pair.rally_id, pair.track_a.canonical_id)]
        ident_b = self.identity_encoding[(pair.rally_id, pair.track_b.canonical_id)]
        team_a = self.team_encoding[(pair.rally_id, pair.track_a.team)]
        team_b = self.team_encoding[(pair.rally_id, pair.track_b.team)]
        tier_idx = TIER_TO_IDX[pair.tier]

        return {
            "crop_a": tensor_a,
            "crop_b": tensor_b,
            "identity_a": torch.tensor(ident_a, dtype=torch.long),
            "identity_b": torch.tensor(ident_b, dtype=torch.long),
            "team_a": torch.tensor(team_a, dtype=torch.long),
            "team_b": torch.tensor(team_b, dtype=torch.long),
            "tier_idx": torch.tensor(tier_idx, dtype=torch.long),
            "pair_idx": torch.tensor(idx, dtype=torch.long),
        }

    def _sample_crop_pair(
        self,
        pair: Pair,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample one crop per track. Falls back to a neutral patch on missing files."""
        crop_a = self._sample_one(pair.rally_id, pair.track_a.track_id, pair.track_a.window_frames, rng)
        crop_b = self._sample_one(pair.rally_id, pair.track_b.track_id, pair.track_b.window_frames, rng)
        return crop_a, crop_b

    def _sample_one(
        self,
        rally_id: str,
        track_id: int,
        window_frames: tuple[int, ...],
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not window_frames:
            return np.full((224, 224, 3), 128, dtype=np.uint8)
        # Try a few frames if the first is missing on disk.
        for _attempt in range(5):
            frame = _sample_frame(rng, window_frames)
            crop = _load_crop(self.corpus_root, rally_id, track_id, frame)
            if crop is not None:
                return crop
        return np.full((224, 224, 3), 128, dtype=np.uint8)
