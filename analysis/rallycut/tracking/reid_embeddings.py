"""Few-shot ReID classifier fine-tuned on user-selected reference crops.

Extracts backbone features (DINOv2 ViT-S/14) from reference crops, trains a
small linear head as a 4-class classifier with data augmentation, then uses
the classifier to identify players at contact frames and across rallies.

Key insight: general models fail because beach volleyball players look similar.
Few-shot fine-tuning learns video-specific discriminative features (hat color,
tattoo, shorts pattern, body shape) from just 2-6 crops per player.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional  # noqa: N812
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_BATCH_SIZE = 32  # Batch size for backbone feature extraction & training
AUGMENTATIONS_PER_CROP = 20  # Augmented copies per original crop
TRAIN_EPOCHS = 60  # Training iterations over augmented data
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1  # Prevents overconfident predictions on few-shot data


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def _default_device() -> str:
    """Auto-detect best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Backbone (DINOv2 ViT-S/14) — frozen, shared singleton
# ---------------------------------------------------------------------------

_backbone_cache: dict[str, tuple[nn.Module, torch.Tensor, torch.Tensor, int]] = {}


def _get_backbone(
    device: str,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, int]:
    """Load frozen DINOv2 ViT-S/14 backbone (384-dim CLS token).

    Returns:
        (model, mean, std, embed_dim)
    """
    cache_key = f"dinov2:{device}"
    if cache_key not in _backbone_cache:
        logger.info("Loading DINOv2 ViT-S/14 backbone on %s...", device)
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        embed_dim = 384  # ViT-S/14

        _backbone_cache[cache_key] = (model, mean, std, embed_dim)

    return _backbone_cache[cache_key]


def extract_backbone_features(
    crops: list[NDArray[np.uint8]],
    device: str | None = None,
) -> NDArray[np.floating]:
    """Extract DINOv2 CLS features from BGR crops.

    Args:
        crops: Non-empty list of BGR images (any size, resized to 224x224).
        device: Torch device. Auto-detected if None.

    Returns:
        (N, 384) float32 features (L2-normalized).
    """
    if not crops:
        return np.empty((0, 384), dtype=np.float32)

    if device is None:
        device = _default_device()

    model, mean, std, _embed_dim = _get_backbone(device)

    batch = []
    for crop in crops:
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        batch.append(tensor)

    batch_tensor = torch.stack(batch).to(device)
    batch_tensor = (batch_tensor - mean) / std

    with torch.inference_mode():
        features = model(batch_tensor)

    features = functional.normalize(features, dim=1)
    return features.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Data augmentation for few-shot training
# ---------------------------------------------------------------------------


def _augment_crop(crop: NDArray[np.uint8], rng: np.random.Generator) -> NDArray:
    """Apply random augmentation to a BGR crop.

    Augmentations: horizontal flip, brightness/contrast jitter,
    hue shift, random crop (85-100%), slight rotation.
    """
    h, w = crop.shape[:2]
    aug: NDArray = crop.copy()

    # Horizontal flip (50%)
    if rng.random() < 0.5:
        aug = cv2.flip(aug, 1)

    # Brightness and contrast jitter
    alpha = rng.uniform(0.8, 1.2)  # contrast
    beta = rng.uniform(-15, 15)  # brightness
    aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Hue shift in HSV space
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV)
        hue_shift = rng.integers(-8, 9)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 180
        aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Random crop (85-100% of area)
    crop_frac = rng.uniform(0.85, 1.0)
    new_h = max(16, int(h * crop_frac))
    new_w = max(8, int(w * crop_frac))
    top = rng.integers(0, max(1, h - new_h + 1))
    left = rng.integers(0, max(1, w - new_w + 1))
    aug = aug[top : top + new_h, left : left + new_w]

    # Slight rotation (±5°)
    if rng.random() < 0.3:
        angle = rng.uniform(-5, 5)
        center = (aug.shape[1] // 2, aug.shape[0] // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug = cv2.warpAffine(
            aug, mat, (aug.shape[1], aug.shape[0]),
            borderMode=cv2.BORDER_REFLECT_101,
        )

    return aug


# ---------------------------------------------------------------------------
# Few-shot classifier
# ---------------------------------------------------------------------------


class PlayerReIDClassifier:
    """Few-shot player classifier fine-tuned on reference crops.

    Architecture: frozen DINOv2 backbone -> trainable linear head (384 -> N_players).
    Training: ~60 epochs on augmented reference crops (~seconds on CPU/MPS).
    Inference: extract backbone features -> linear head -> softmax probabilities.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or _default_device()
        self.head: nn.Linear | None = None
        self.player_ids: list[int] = []  # Ordered player IDs (index -> player_id)
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(
        self,
        crops_by_player: dict[int, list[NDArray[np.uint8]]],
        augmentations_per_crop: int = AUGMENTATIONS_PER_CROP,
        epochs: int = TRAIN_EPOCHS,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train the classifier on reference crops.

        Args:
            crops_by_player: {player_id: [BGR crop images]}.
                Typically 2-6 crops per player from the web UI.
            augmentations_per_crop: Number of augmented copies per original crop.
            epochs: Training epochs over the augmented dataset.
            verbose: Print training progress.

        Returns:
            Training stats: {"train_acc": float, "train_loss": float, "n_samples": int}
        """
        self.player_ids = sorted(crops_by_player.keys())
        n_classes = len(self.player_ids)
        if n_classes < 2:
            logger.warning("Need at least 2 players for few-shot training")
            return {"train_acc": 0.0, "train_loss": 0.0, "n_samples": 0}

        pid_to_idx = {pid: i for i, pid in enumerate(self.player_ids)}

        # Build augmented training set
        rng = np.random.default_rng(42)
        all_crops: list[NDArray] = []
        all_labels: list[int] = []
        n_originals = 0

        for pid in self.player_ids:
            for crop in crops_by_player[pid]:
                n_originals += 1
                all_crops.append(crop)
                all_labels.append(pid_to_idx[pid])
                for _ in range(augmentations_per_crop):
                    all_crops.append(_augment_crop(crop, rng))
                    all_labels.append(pid_to_idx[pid])

        n_samples = len(all_crops)
        logger.info(
            "Training ReID classifier: %d players, %d samples "
            "(%d originals x %d augmentations)",
            n_classes, n_samples, n_originals, augmentations_per_crop + 1,
        )

        # Extract backbone features for all crops (batched)
        all_features: list[NDArray[np.floating]] = []
        for i in range(0, n_samples, FEATURE_BATCH_SIZE):
            batch = all_crops[i : i + FEATURE_BATCH_SIZE]
            features = extract_backbone_features(batch, self.device)
            all_features.append(features)

        features_np = np.concatenate(all_features, axis=0)  # (N, 384)
        features_tensor = torch.from_numpy(features_np).to(self.device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=self.device)

        # Initialize linear head
        embed_dim = features_np.shape[1]
        self.head = nn.Linear(embed_dim, n_classes).to(self.device)

        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        self.head.train()
        final_loss = 0.0
        final_acc = 0.0

        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=self.device)
            features_shuffled = features_tensor[perm]
            labels_shuffled = labels_tensor[perm]

            epoch_loss = 0.0
            epoch_correct = 0

            for i in range(0, n_samples, FEATURE_BATCH_SIZE):
                batch_feats = features_shuffled[i : i + FEATURE_BATCH_SIZE]
                batch_labels = labels_shuffled[i : i + FEATURE_BATCH_SIZE]

                logits = self.head(batch_feats)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_feats.size(0)
                epoch_correct += (logits.argmax(1) == batch_labels).sum().item()

            final_loss = epoch_loss / n_samples
            final_acc = epoch_correct / n_samples

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "  Epoch %d/%d: loss=%.4f acc=%.1f%%",
                    epoch + 1, epochs, final_loss, final_acc * 100,
                )

        self.head.eval()
        self._trained = True

        logger.info("Training complete: acc=%.1f%% loss=%.4f", final_acc * 100, final_loss)
        return {
            "train_acc": final_acc,
            "train_loss": final_loss,
            "n_samples": n_samples,
        }

    def predict(
        self,
        crops: list[NDArray[np.uint8]],
    ) -> list[dict[int, float]]:
        """Predict player probabilities for candidate crops.

        Args:
            crops: List of BGR crop images.

        Returns:
            List of {player_id: probability} dicts, one per crop.
        """
        if not self._trained or self.head is None:
            raise RuntimeError("Classifier not trained. Call train() first.")

        if not crops:
            return []

        features = extract_backbone_features(crops, self.device)
        features_tensor = torch.from_numpy(features).to(self.device)

        with torch.inference_mode():
            logits = self.head(features_tensor)
            probs = functional.softmax(logits, dim=1).cpu().numpy()

        results: list[dict[int, float]] = []
        for i in range(len(crops)):
            prob_dict = {
                self.player_ids[j]: float(probs[i, j])
                for j in range(len(self.player_ids))
            }
            results.append(prob_dict)

        return results

    def predict_single(
        self,
        crop: NDArray[np.uint8],
    ) -> dict[int, float]:
        """Predict player probabilities for a single crop."""
        return self.predict([crop])[0]

    def classify(
        self,
        crops: list[NDArray[np.uint8]],
    ) -> list[int]:
        """Return the most likely player_id for each crop."""
        probs_list = self.predict(crops)
        return [max(probs, key=lambda pid: probs[pid]) for probs in probs_list]


# ---------------------------------------------------------------------------
# Reference crop loading from video
# ---------------------------------------------------------------------------


def extract_crops_from_video(
    video_path: Path,
    crop_infos: list[dict[str, Any]],
) -> dict[int, list[NDArray[np.uint8]]]:
    """Extract reference crops from video at stored bbox/frame positions.

    Seeks to each frame_ms in order. Seeking is intentional here: reference
    crops are sparse user-selected frames spread across the full video.

    Args:
        video_path: Path to the video file.
        crop_infos: List of dicts with keys:
            player_id (int), frame_ms (int),
            bbox_x, bbox_y, bbox_w, bbox_h (float, normalized 0-1).

    Returns:
        {player_id: [BGR crop images]}.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return {}

    # Sort by frame_ms for forward-only seeking
    sorted_infos = sorted(crop_infos, key=lambda c: c["frame_ms"])

    crops_by_player: dict[int, list[NDArray]] = {}

    for info in sorted_infos:
        cap.set(cv2.CAP_PROP_POS_MSEC, info["frame_ms"])
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        bx, by, bw, bh = info["bbox_x"], info["bbox_y"], info["bbox_w"], info["bbox_h"]

        x1 = max(0, int((bx - bw / 2) * w))
        y1 = max(0, int((by - bh / 2) * h))
        x2 = min(w, int((bx + bw / 2) * w))
        y2 = min(h, int((by + bh / 2) * h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
            continue

        pid = info["player_id"]
        crops_by_player.setdefault(pid, []).append(crop)

    cap.release()

    for pid, crops in crops_by_player.items():
        logger.info("Extracted %d reference crops for player %d", len(crops), pid)

    return crops_by_player


# ---------------------------------------------------------------------------
# Within-team ReID head (Session 4 — trained MLP 384→192→128 over frozen DINOv2)
# ---------------------------------------------------------------------------


HEAD_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "weights" / "within_team_reid" / "best.pt"
)


def _compute_head_sha() -> str:
    """SHA-12 of the checkpoint file for retrack-cache invalidation.

    Empty string if the file is missing — treated as ``learned_reid:disabled``.
    """
    import hashlib

    if not HEAD_CHECKPOINT_PATH.is_file():
        return ""
    h = hashlib.sha256(HEAD_CHECKPOINT_PATH.read_bytes())
    return h.hexdigest()[:12]


HEAD_SHA = _compute_head_sha()

_head_cache: dict[str, nn.Module | None] = {}
_head_load_warned = False


def _get_head(device: str) -> nn.Module | None:
    """Load the trained MLP head (Session 3 V3 epoch 2) lazily, once per device.

    Returns None on any failure (missing file, state-dict mismatch, import
    error). Callers treat None as "learned ReID disabled" — no crash.
    """
    global _head_load_warned
    cache_key = f"within_team_reid:{device}"
    if cache_key in _head_cache:
        return _head_cache[cache_key]

    try:
        from training.within_team_reid.model.head import MLPHead
    except Exception as exc:  # noqa: BLE001
        if not _head_load_warned:
            logger.warning(
                "Within-team ReID head: MLPHead import failed (%s); "
                "learned-ReID cost will be disabled.", exc,
            )
            _head_load_warned = True
        _head_cache[cache_key] = None
        return None

    if not HEAD_CHECKPOINT_PATH.is_file():
        if not _head_load_warned:
            logger.warning(
                "Within-team ReID head: checkpoint not found at %s; "
                "learned-ReID cost will be disabled.", HEAD_CHECKPOINT_PATH,
            )
            _head_load_warned = True
        _head_cache[cache_key] = None
        return None

    try:
        ckpt = torch.load(HEAD_CHECKPOINT_PATH, map_location=device, weights_only=False)
        head = MLPHead()
        head.load_state_dict(ckpt["head_state_dict"])
        head.to(device).eval()
        for p in head.parameters():
            p.requires_grad_(False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Within-team ReID head: load failed (%s); learned-ReID cost disabled.",
            exc,
        )
        _head_cache[cache_key] = None
        return None

    logger.info(
        "Within-team ReID head loaded on %s (sha=%s, params=%d)",
        device, HEAD_SHA, sum(p.numel() for p in head.parameters()),
    )
    _head_cache[cache_key] = head
    return head


def extract_learned_embeddings(
    crops: list[NDArray[np.uint8]],
    device: str | None = None,
) -> NDArray[np.floating]:
    """Extract Session-3 128-d within-team embeddings from BGR crops.

    Pipeline: DINOv2 ViT-S/14 (frozen, shared singleton) → 2-layer MLP head →
    L2-normalize. Returns ``(N, 128)`` float32. Returns empty ``(0, 128)`` if
    the head is unavailable (checkpoint missing, import failed) — callers
    must handle the empty case by skipping the learned-ReID path for that
    frame.
    """
    if not crops:
        return np.empty((0, 128), dtype=np.float32)

    if device is None:
        device = _default_device()

    head = _get_head(device)
    if head is None:
        return np.empty((0, 128), dtype=np.float32)

    feats_384 = extract_backbone_features(crops, device=device)
    if feats_384.shape[0] == 0:
        return np.empty((0, 128), dtype=np.float32)

    with torch.inference_mode():
        x = torch.from_numpy(feats_384).to(device)
        out = head(x)
    result: NDArray[np.floating] = out.cpu().numpy().astype(np.float32)
    return result
