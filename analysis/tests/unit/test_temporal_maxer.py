"""Tests for TemporalMaxer model."""

from __future__ import annotations

import numpy as np
import torch

from rallycut.temporal.temporal_maxer.model import TemporalMaxer, TemporalMaxerConfig
from rallycut.temporal.temporal_maxer.training import (
    VideoSequenceDataset,
    collate_video_sequences,
    compute_tmse_loss,
)


class TestTemporalMaxerModel:
    def test_forward_shape(self) -> None:
        config = TemporalMaxerConfig(feature_dim=768, num_classes=2, num_layers=4)
        model = TemporalMaxer(config)

        # Input: (batch=1, feature_dim=768, T=100)
        x = torch.randn(1, 768, 100)
        logits = model(x)

        assert logits.shape == (1, 2, 100)

    def test_forward_with_mask(self) -> None:
        config = TemporalMaxerConfig(feature_dim=768, num_classes=2, num_layers=2)
        model = TemporalMaxer(config)

        x = torch.randn(2, 768, 50)
        mask = torch.ones(2, 1, 50)
        mask[1, :, 30:] = 0  # Second sequence shorter

        logits = model(x, mask)
        assert logits.shape == (2, 2, 50)

    def test_forward_short_sequence(self) -> None:
        config = TemporalMaxerConfig(feature_dim=768, num_classes=2, num_layers=2)
        model = TemporalMaxer(config)

        x = torch.randn(1, 768, 5)
        logits = model(x)
        assert logits.shape == (1, 2, 5)

    def test_small_feature_dim(self) -> None:
        config = TemporalMaxerConfig(feature_dim=32, hidden_dim=32, num_classes=2, num_layers=2)
        model = TemporalMaxer(config)

        x = torch.randn(1, 32, 20)
        logits = model(x)
        assert logits.shape == (1, 2, 20)

    def test_gradient_flow(self) -> None:
        """Verify gradients flow through the model."""
        config = TemporalMaxerConfig(feature_dim=32, hidden_dim=32, num_classes=2, num_layers=2)
        model = TemporalMaxer(config)

        x = torch.randn(1, 32, 20, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestTMSELoss:
    def test_constant_predictions_zero_loss(self) -> None:
        """Constant predictions should have zero TMSE."""
        logits = torch.zeros(1, 2, 10)
        logits[:, 0, :] = 1.0  # All same class
        mask = torch.ones(1, 1, 10)

        loss = compute_tmse_loss(logits, mask)
        assert loss.item() < 1e-6

    def test_varying_predictions_positive_loss(self) -> None:
        """Varying predictions should have positive TMSE."""
        logits = torch.zeros(1, 2, 10)
        logits[:, 0, ::2] = 2.0  # Alternating
        logits[:, 1, 1::2] = 2.0
        mask = torch.ones(1, 1, 10)

        loss = compute_tmse_loss(logits, mask)
        assert loss.item() > 0

    def test_masked_positions_ignored(self) -> None:
        """Masked positions should not contribute to loss."""
        logits = torch.zeros(1, 2, 10)
        logits[:, 0, ::2] = 2.0
        logits[:, 1, 1::2] = 2.0
        mask = torch.zeros(1, 1, 10)  # All masked

        loss = compute_tmse_loss(logits, mask)
        assert loss.item() == 0.0


class TestCollation:
    def test_collate_equal_length(self) -> None:
        dataset = VideoSequenceDataset(
            [np.random.randn(10, 32).astype(np.float32)] * 2,
            [np.zeros(10).astype(np.int64)] * 2,
        )
        batch = [dataset[0], dataset[1]]
        features, labels, mask = collate_video_sequences(batch)

        assert features.shape == (2, 32, 10)
        assert labels.shape == (2, 10)
        assert mask.shape == (2, 1, 10)
        assert torch.all(mask == 1.0)

    def test_collate_variable_length(self) -> None:
        dataset = VideoSequenceDataset(
            [
                np.random.randn(10, 32).astype(np.float32),
                np.random.randn(5, 32).astype(np.float32),
            ],
            [
                np.zeros(10).astype(np.int64),
                np.zeros(5).astype(np.int64),
            ],
        )
        batch = [dataset[0], dataset[1]]
        features, labels, mask = collate_video_sequences(batch)

        assert features.shape == (2, 32, 10)  # Padded to max_len=10
        assert labels.shape == (2, 10)
        assert mask.shape == (2, 1, 10)
        # Second sequence: first 5 valid, rest masked
        assert torch.all(mask[1, 0, :5] == 1.0)
        assert torch.all(mask[1, 0, 5:] == 0.0)

    def test_collate_empty(self) -> None:
        features, labels, mask = collate_video_sequences([])
        assert features.shape[0] == 0
