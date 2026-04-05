"""PyTorch dataset and collation for rally-level action sequences.

Wraps per-frame trajectory features and labels for variable-length
rally sequences. Handles padding, masking, and batching for Conv1d
temporal models (MS-TCN++, TemporalMaxer).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceActionDataset(Dataset):
    """Dataset of rally-level trajectory sequences.

    Each item is a (features, labels) pair for one rally.
    Features: (num_frames, feature_dim) float32
    Labels: (num_frames,) int64
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
        features = torch.from_numpy(self.features_list[idx]).float()
        labels = torch.from_numpy(self.labels_list[idx]).long()

        if self.augment and self.noise_std > 0:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise

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
