"""Tests for per-game self-supervised identity embedding."""

from __future__ import annotations

import numpy as np

from rallycut.tracking.appearance_descriptor import (
    AppearanceDescriptorStore,
    MultiRegionDescriptor,
)
from rallycut.tracking.identity_embedding import (
    EmbeddingConfig,
    IdentityEmbedding,
    _descriptor_to_feature,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _make_histogram(peak: int = 0) -> np.ndarray:
    hist = np.zeros((16, 8), dtype=np.float32)
    hist[peak % 16, 0] = 1.0
    return hist


def _make_descriptor(
    head_peak: int = 0,
    upper_peak: int = 4,
    shorts_peak: int = 8,
) -> MultiRegionDescriptor:
    return MultiRegionDescriptor(
        head=_make_histogram(head_peak),
        upper=_make_histogram(upper_peak),
        shorts=_make_histogram(shorts_peak),
    )


def _make_positions(
    track_id: int,
    frames: range,
    x: float = 0.5,
    y: float = 0.5,
) -> list[PlayerPosition]:
    return [
        PlayerPosition(f, track_id, x, y, 0.05, 0.15, 0.9)
        for f in frames
    ]


class TestDescriptorToFeature:
    def test_full_descriptor(self) -> None:
        desc = _make_descriptor()
        feat = _descriptor_to_feature(desc)
        assert feat is not None
        assert feat.shape == (384,)

    def test_partial_descriptor(self) -> None:
        desc = MultiRegionDescriptor(shorts=_make_histogram())
        feat = _descriptor_to_feature(desc)
        assert feat is not None
        assert feat.shape == (384,)
        # Head and upper should be zeros
        assert feat[:256].sum() == 0.0

    def test_empty_descriptor(self) -> None:
        desc = MultiRegionDescriptor()
        feat = _descriptor_to_feature(desc)
        assert feat is not None
        assert feat.sum() == 0.0


class TestIdentityEmbedding:
    def test_not_trained_initially(self) -> None:
        emb = IdentityEmbedding()
        assert not emb.is_trained

    def test_get_embedding_before_training(self) -> None:
        emb = IdentityEmbedding()
        desc = _make_descriptor()
        result = emb.get_embedding(desc)
        assert result is None

    def test_train_on_synthetic_data(self) -> None:
        """Should be able to train on simple synthetic data."""
        config = EmbeddingConfig(
            min_total_samples=10,
            epochs=10,
            batch_size=16,
        )
        emb = IdentityEmbedding(config)

        # Create synthetic training data with 4 distinct classes
        rng = np.random.default_rng(42)
        n_per_class = 20
        features = np.zeros((n_per_class * 4, 385), dtype=np.float32)
        labels = np.zeros(n_per_class * 4, dtype=np.int64)

        for cls in range(4):
            start = cls * n_per_class
            end = start + n_per_class
            # Each class has a distinct feature pattern
            features[start:end] = rng.normal(
                cls * 0.5, 0.1, (n_per_class, 385)
            ).astype(np.float32)
            labels[start:end] = cls

        acc = emb.train(features, labels)
        assert emb.is_trained
        assert acc > 0.0  # Should learn something

    def test_embedding_distance(self) -> None:
        """Trained model should produce embedding distances."""
        config = EmbeddingConfig(
            min_total_samples=10,
            epochs=20,
            batch_size=16,
        )
        emb = IdentityEmbedding(config)

        # Train on separable data
        rng = np.random.default_rng(42)
        features = np.zeros((40, 385), dtype=np.float32)
        labels = np.zeros(40, dtype=np.int64)
        for cls in range(2):
            features[cls * 20 : (cls + 1) * 20] = rng.normal(
                cls * 2, 0.1, (20, 385)
            ).astype(np.float32)
            labels[cls * 20 : (cls + 1) * 20] = cls

        emb.train(features, labels)

        # Same class should be closer than different class
        desc_a = _make_descriptor(head_peak=0)
        desc_b = _make_descriptor(head_peak=0)
        dist = emb.compute_embedding_distance(desc_a, desc_b)
        assert isinstance(dist, float)
        assert 0.0 <= dist <= 2.0

    def test_collect_training_data_insufficient(self) -> None:
        """Should return None when insufficient samples."""
        config = EmbeddingConfig(min_total_samples=100)
        emb = IdentityEmbedding(config)

        positions = (
            _make_positions(1, range(0, 10), x=0.2, y=0.3)
            + _make_positions(2, range(0, 10), x=0.8, y=0.7)
        )
        store = AppearanceDescriptorStore()
        # Only a few descriptors
        store.add(1, 0, _make_descriptor())
        store.add(2, 0, _make_descriptor(head_peak=5))

        result = emb.collect_training_data(positions, store, [1, 2])
        assert result is None  # Not enough samples

    def test_collect_training_data_sufficient(self) -> None:
        """Should collect features when enough well-separated frames."""
        config = EmbeddingConfig(
            min_total_samples=5,
            min_well_separated_distance=0.1,
            min_samples_per_track=2,
        )
        emb = IdentityEmbedding(config)

        # Create well-separated tracks
        positions = (
            _make_positions(1, range(0, 30), x=0.2, y=0.3)
            + _make_positions(2, range(0, 30), x=0.8, y=0.7)
        )
        store = AppearanceDescriptorStore()
        for f in range(0, 30, 3):
            store.add(1, f, _make_descriptor(head_peak=0))
            store.add(2, f, _make_descriptor(head_peak=8))

        result = emb.collect_training_data(positions, store, [1, 2])
        assert result is not None
        features, labels = result
        assert features.shape[1] == 385
        assert len(labels) == len(features)
        assert set(labels.tolist()) == {0, 1}
