"""PyTorch dataset and collation for rally-level action sequences.

Wraps per-frame trajectory features and labels for variable-length
rally sequences. Handles padding, masking, and batching for Conv1d
temporal models (MS-TCN++, TemporalMaxer).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

# Ball feature indices (dims 0-5: ball_x, ball_y, ball_conf, ball_dx, ball_dy, ball_speed)
# Plus derived: court_ball [19:21], ball_det_density [21], ball_nearest_height_ratio [26]
_BALL_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 19, 20, 21, 26]
# Player slot indices for permutation: player_xy [6:14], ball_player_dist [14:18],
# player_team [22:26]
_PLAYER_SLOT_RANGES = [(6, 14, 2), (14, 18, 1), (22, 26, 1)]


class SequenceActionDataset(Dataset):
    """Dataset of rally-level trajectory sequences.

    Each item is a (features, labels) pair for one rally.
    Features: (num_frames, feature_dim) float32
    Labels: (num_frames,) int64

    Augmentations (when augment=True):
        - Gaussian noise on all features
        - Ball feature dropout: zero ball features for 5-15% of frames
        - Temporal jitter: shift sequence by ±3 frames
        - Speed perturbation: scale time axis by 0.85-1.15x
        - Player slot permutation: shuffle 4 player slots with p=0.1
    """

    def __init__(
        self,
        features_list: list[np.ndarray],
        labels_list: list[np.ndarray],
        augment: bool = False,
        noise_std: float = 0.01,
    ) -> None:
        assert len(features_list) == len(labels_list)
        self.features_list = features_list
        self.labels_list = labels_list
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.features_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features_list[idx].copy()  # (T, F), numpy
        labels = self.labels_list[idx].copy()  # (T,), numpy

        if self.augment:
            features, labels = self._augment(features, labels)

        return torch.from_numpy(features).float(), torch.from_numpy(labels).long()

    def _augment(
        self, features: np.ndarray, labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng()
        n_frames, n_feat = features.shape

        # 1. Ball feature dropout: zero ball features for 5-15% of frames
        if rng.random() < 0.5:
            drop_rate = rng.uniform(0.05, 0.15)
            drop_mask = rng.random(n_frames) < drop_rate
            for idx in _BALL_FEATURE_INDICES:
                if idx < n_feat:
                    features[drop_mask, idx] = 0.0

        # 2. Speed perturbation: resample time axis by 0.85-1.15x
        if rng.random() < 0.3:
            speed_factor = rng.uniform(0.85, 1.15)
            new_len = max(10, int(n_frames * speed_factor))
            old_t = np.linspace(0, 1, n_frames)
            new_t = np.linspace(0, 1, new_len)
            interp_func = interp1d(old_t, features, axis=0, kind="linear",
                                   fill_value="extrapolate")
            features = interp_func(new_t).astype(np.float32)
            label_interp = interp1d(old_t, labels.astype(float), axis=0,
                                    kind="nearest", fill_value="extrapolate")
            labels = label_interp(new_t).astype(np.int64)
            n_frames = new_len

        # 3. Temporal jitter: shift sequence by ±3 frames
        if rng.random() < 0.3:
            shift = rng.integers(-3, 4)  # [-3, 3]
            if shift != 0:
                features = np.roll(features, shift, axis=0)
                labels = np.roll(labels, shift, axis=0)
                if shift > 0:
                    features[:shift] = 0.0
                    labels[:shift] = 0
                else:
                    features[shift:] = 0.0
                    labels[shift:] = 0

        # 4. Player slot permutation: shuffle 4 player slots with p=0.1
        if rng.random() < 0.1:
            perm = rng.permutation(4)
            for start, end, stride in _PLAYER_SLOT_RANGES:
                if start >= n_feat:
                    break
                actual_end = min(end, n_feat)
                n_slots = (actual_end - start) // stride
                if n_slots != 4:
                    continue
                original = features[:, start:actual_end].copy()
                for i in range(4):
                    src_slice = slice(start + perm[i] * stride, start + perm[i] * stride + stride)
                    dst_slice = slice(start + i * stride, start + i * stride + stride)
                    if src_slice.stop <= n_feat and dst_slice.stop <= n_feat:
                        features[:, dst_slice] = original[:, perm[i] * stride:perm[i] * stride + stride]

        # 5. Gaussian noise (always, same as before)
        if self.noise_std > 0:
            features = features + rng.normal(0, self.noise_std, features.shape).astype(np.float32)

        return features, labels


def collate_rally_sequences(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length rally sequences with padding.

    Args:
        batch: List of (features, labels) tuples.
            features: (seq_len, feature_dim)
            labels: (seq_len,)

    Returns:
        features: (batch, feature_dim, max_len) — transposed for Conv1d
        labels: (batch, max_len) — padded with 0
        mask: (batch, 1, max_len) — 1 for valid positions
    """
    features_list, labels_list = zip(*batch)
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    feature_dim = features_list[0].shape[1]
    batch_size = len(batch)

    padded_features = torch.zeros(batch_size, feature_dim, max_len)
    padded_labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, 1, max_len)

    for i, (feat, lbl) in enumerate(zip(features_list, labels_list)):
        seq_len = feat.shape[0]
        padded_features[i, :, :seq_len] = feat.T  # (feat_dim, seq_len)
        padded_labels[i, :seq_len] = lbl
        mask[i, 0, :seq_len] = 1.0

    return padded_features, padded_labels, mask
