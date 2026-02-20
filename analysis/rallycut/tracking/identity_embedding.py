"""Per-game self-supervised identity embedding for player tracking.

Trains a tiny MLP per video on well-separated frames where BoT-SORT's
track assignments are reliable. The MLP learns to map multi-region
histogram features to a 16-dim embedding that discriminates between
the 4 players. This embedding provides an additional signal for
identity resolution during ambiguous net interactions.

Architecture: 385 -> 64 -> 32 -> 16 (cross-entropy on 4 classes)
Input: 3 flattened histograms (3 * 128) + height estimate (1) = 385
Training: <0.5s on CPU for ~200 samples
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.appearance_descriptor import (
        AppearanceDescriptorStore,
        MultiRegionDescriptor,
    )
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Training parameters
MIN_SAMPLES_PER_TRACK = 10  # Minimum samples to include a track
MIN_WELL_SEPARATED_DISTANCE = 0.15  # Normalized image distance between all pairs
MIN_TOTAL_SAMPLES = 30  # Minimum total samples before training
FEATURE_DIM = 385  # 3 * 128 (HS histograms) + 1 (height)
EMBEDDING_DIM = 16
HIDDEN_1 = 64
HIDDEN_2 = 32


@dataclass
class EmbeddingConfig:
    """Configuration for identity embedding training."""

    min_samples_per_track: int = MIN_SAMPLES_PER_TRACK
    min_well_separated_distance: float = MIN_WELL_SEPARATED_DISTANCE
    min_total_samples: int = MIN_TOTAL_SAMPLES
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int = 32


class IdentityEmbedding:
    """Per-game identity embedding model.

    Trained on well-separated frames from the current video.
    Once trained, provides embedding distances for identity resolution.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._weights: dict[str, np.ndarray] = {}
        self._is_trained = False
        self._track_labels: dict[int, int] = {}  # track_id -> class (0-3)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def collect_training_data(
        self,
        positions: list[PlayerPosition],
        appearance_store: AppearanceDescriptorStore,
        primary_track_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Collect training features from well-separated frames.

        Args:
            positions: All player positions.
            appearance_store: Multi-region descriptor store.
            primary_track_ids: The 4 primary track IDs.

        Returns:
            (features, labels) arrays, or None if insufficient data.
        """
        if len(primary_track_ids) < 2:
            return None

        cfg = self.config

        # Assign class labels to tracks
        self._track_labels = {
            tid: i for i, tid in enumerate(sorted(primary_track_ids))
        }

        # Group positions by frame
        by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
        for p in positions:
            if p.track_id in self._track_labels:
                by_frame[p.frame_number].append(p)

        # Find well-separated frames (all primary tracks present, pairwise far apart)
        features_list: list[np.ndarray] = []
        labels_list: list[int] = []

        for frame_num in sorted(by_frame.keys()):
            frame_pos = by_frame[frame_num]
            present_tids = {p.track_id for p in frame_pos}

            # Need at least 2 primary tracks
            primary_present = present_tids & set(primary_track_ids)
            if len(primary_present) < 2:
                continue

            # Check all pairs are well-separated
            separated = True
            primary_positions = [
                p for p in frame_pos if p.track_id in primary_present
            ]
            for i in range(len(primary_positions)):
                for j in range(i + 1, len(primary_positions)):
                    pi, pj = primary_positions[i], primary_positions[j]
                    dist = ((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2) ** 0.5
                    if dist < cfg.min_well_separated_distance:
                        separated = False
                        break
                if not separated:
                    break

            if not separated:
                continue

            # Extract features for each player in this frame
            for p in primary_positions:
                desc = appearance_store.get(p.track_id, frame_num)
                if desc is None:
                    continue

                feat = _descriptor_to_feature(desc)
                if feat is not None:
                    # Height dimension reserved but zeroed — inference
                    # doesn't have height context, so train consistently
                    full_feat = np.append(feat, 0.0)
                    features_list.append(full_feat)
                    labels_list.append(self._track_labels[p.track_id])

        if len(features_list) < cfg.min_total_samples:
            logger.debug(
                f"Identity embedding: only {len(features_list)} samples "
                f"(need {cfg.min_total_samples}), skipping training"
            )
            return None

        features = np.array(features_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)

        logger.info(
            f"Identity embedding: {len(features)} training samples from "
            f"{len(self._track_labels)} tracks"
        )

        return features, labels

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Train the MLP on collected features.

        Args:
            features: (N, 385) feature array.
            labels: (N,) class labels (0 to num_classes-1).

        Returns:
            Final training accuracy.
        """
        cfg = self.config
        n_samples, n_features = features.shape
        # Use track_labels count when available (from collect_training_data),
        # fall back to labels array for direct train() calls
        n_classes = len(self._track_labels) if self._track_labels else int(labels.max()) + 1

        if n_classes < 2:
            return 0.0

        # Initialize weights (Xavier)
        rng = np.random.default_rng(42)
        self._weights = {
            "w1": rng.normal(0, (2 / n_features) ** 0.5, (n_features, HIDDEN_1)).astype(np.float32),
            "b1": np.zeros(HIDDEN_1, dtype=np.float32),
            "w2": rng.normal(0, (2 / HIDDEN_1) ** 0.5, (HIDDEN_1, HIDDEN_2)).astype(np.float32),
            "b2": np.zeros(HIDDEN_2, dtype=np.float32),
            "w3": rng.normal(0, (2 / HIDDEN_2) ** 0.5, (HIDDEN_2, EMBEDDING_DIM)).astype(np.float32),
            "b3": np.zeros(EMBEDDING_DIM, dtype=np.float32),
            "w_cls": rng.normal(0, (2 / EMBEDDING_DIM) ** 0.5, (EMBEDDING_DIM, n_classes)).astype(np.float32),
            "b_cls": np.zeros(n_classes, dtype=np.float32),
        }

        # Training loop
        lr = cfg.learning_rate
        best_acc = 0.0

        for epoch in range(cfg.epochs):
            # Shuffle
            perm = rng.permutation(n_samples)
            features_shuffled = features[perm]
            labels_shuffled = labels[perm]

            total_loss = 0.0
            correct = 0

            for i in range(0, n_samples, cfg.batch_size):
                batch_x = features_shuffled[i : i + cfg.batch_size]
                batch_y = labels_shuffled[i : i + cfg.batch_size]

                # Forward pass
                h1 = np.maximum(0, batch_x @ self._weights["w1"] + self._weights["b1"])
                h2 = np.maximum(0, h1 @ self._weights["w2"] + self._weights["b2"])
                embedding = np.maximum(0, h2 @ self._weights["w3"] + self._weights["b3"])
                logits = embedding @ self._weights["w_cls"] + self._weights["b_cls"]

                # Softmax + cross-entropy
                logits_max = logits - logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits_max)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                batch_n = len(batch_y)
                loss = -np.log(probs[np.arange(batch_n), batch_y] + 1e-8).mean()
                total_loss += loss * batch_n
                correct += (probs.argmax(axis=1) == batch_y).sum()

                # Backward pass (manual gradients)
                dlogits = probs.copy()
                dlogits[np.arange(batch_n), batch_y] -= 1
                dlogits /= batch_n

                dw_cls = embedding.T @ dlogits
                db_cls = dlogits.sum(axis=0)

                dembedding = dlogits @ self._weights["w_cls"].T
                dembedding *= (embedding > 0).astype(np.float32)

                dw3 = h2.T @ dembedding
                db3 = dembedding.sum(axis=0)

                dh2 = dembedding @ self._weights["w3"].T
                dh2 *= (h2 > 0).astype(np.float32)

                dw2 = h1.T @ dh2
                db2 = dh2.sum(axis=0)

                dh1 = dh2 @ self._weights["w2"].T
                dh1 *= (h1 > 0).astype(np.float32)

                dw1 = batch_x.T @ dh1
                db1 = dh1.sum(axis=0)

                # Gradient descent
                self._weights["w_cls"] -= lr * dw_cls
                self._weights["b_cls"] -= lr * db_cls
                self._weights["w3"] -= lr * dw3
                self._weights["b3"] -= lr * db3
                self._weights["w2"] -= lr * dw2
                self._weights["b2"] -= lr * db2
                self._weights["w1"] -= lr * dw1
                self._weights["b1"] -= lr * db1

            acc = correct / n_samples
            best_acc = max(best_acc, acc)

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"Embedding epoch {epoch + 1}: loss={total_loss / n_samples:.4f}, "
                    f"acc={acc:.3f}"
                )

        self._is_trained = True
        logger.info(f"Identity embedding trained: accuracy={best_acc:.3f}")
        return float(best_acc)

    def get_embedding(self, descriptor: MultiRegionDescriptor) -> np.ndarray | None:
        """Get embedding vector for a descriptor.

        Args:
            descriptor: Multi-region appearance descriptor.

        Returns:
            16-dim embedding vector, or None if not trained.
        """
        if not self._is_trained:
            return None

        feat = _descriptor_to_feature(descriptor)
        if feat is None:
            return None

        # Height dimension zeroed (consistent with training)
        full_feat = np.append(feat, 0.0).astype(np.float32)

        # Forward pass through embedding layers only
        h1: np.ndarray = np.maximum(0, full_feat @ self._weights["w1"] + self._weights["b1"])
        h2: np.ndarray = np.maximum(0, h1 @ self._weights["w2"] + self._weights["b2"])
        embedding: np.ndarray = np.maximum(0, h2 @ self._weights["w3"] + self._weights["b3"])

        # L2 normalize
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def compute_embedding_distance(
        self,
        desc_a: MultiRegionDescriptor,
        desc_b: MultiRegionDescriptor,
    ) -> float:
        """Compute embedding distance between two descriptors.

        Returns distance in [0, 2] (cosine distance of L2-normalized vectors).
        """
        emb_a = self.get_embedding(desc_a)
        emb_b = self.get_embedding(desc_b)

        if emb_a is None or emb_b is None:
            return 1.0  # No data — neutral distance

        # Cosine distance
        return float(1.0 - np.dot(emb_a, emb_b))


def _descriptor_to_feature(
    descriptor: MultiRegionDescriptor,
) -> np.ndarray | None:
    """Flatten a MultiRegionDescriptor to a 384-dim feature vector."""
    parts: list[np.ndarray] = []

    if descriptor.head is not None:
        parts.append(descriptor.head.flatten())
    else:
        parts.append(np.zeros(128, dtype=np.float32))

    if descriptor.upper is not None:
        parts.append(descriptor.upper.flatten())
    else:
        parts.append(np.zeros(128, dtype=np.float32))

    if descriptor.shorts is not None:
        parts.append(descriptor.shorts.flatten())
    else:
        parts.append(np.zeros(128, dtype=np.float32))

    result: np.ndarray = np.concatenate(parts)
    if result.shape[0] != 384:
        return None

    return result
