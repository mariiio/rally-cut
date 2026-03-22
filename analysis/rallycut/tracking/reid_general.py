"""General volleyball ReID model — contrastive DINOv2 fine-tuning.

Learns player-discriminative embeddings from GT-labeled training data across
multiple videos. Unlike the per-video PlayerReIDClassifier (reid_embeddings.py),
this model generalizes to unseen videos without requiring user reference crops.

Architecture:
    DINOv2 ViT-S/14 backbone (last 2 blocks unfrozen)
    → projection head (384 → 128 → L2 normalize)

Training:
    SupCon (Supervised Contrastive) loss on video-grouped batches.
    Hard negatives: different players from the same video.
    Easy negatives: players from different videos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional  # noqa: N812
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Model constants
EMBED_DIM = 384  # DINOv2 ViT-S/14 CLS token
PROJ_DIM = 128  # Projection head output
WEIGHTS_PATH = Path(__file__).parent.parent.parent / "weights" / "reid" / "general_reid.pt"

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
DEFAULT_LR_BACKBONE = 1e-5
DEFAULT_LR_HEAD = 1e-3
DEFAULT_TEMPERATURE = 0.07
UNFREEZE_BLOCKS = 2  # Unfreeze last N ViT blocks


def _default_device() -> str:
    """Auto-detect best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class ProjectionHead(nn.Module):
    """MLP projection head: 384 → 384 → 128."""

    def __init__(self, in_dim: int = EMBED_DIM, out_dim: int = PROJ_DIM) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return functional.normalize(x, dim=1)


class GeneralReIDModel:
    """Contrastive-trained DINOv2 model for volleyball player ReID.

    Usage:
        # Training
        model = GeneralReIDModel()
        model.train_on_dataset(Path("reid_training_data"))

        # Inference (after training or loading weights)
        model = GeneralReIDModel(weights_path=WEIGHTS_PATH)
        embeddings = model.extract_embeddings(crops)  # (N, 128) L2-normalized
    """

    def __init__(
        self,
        weights_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.device = device or _default_device()
        self.backbone: nn.Module | None = None
        self.head: ProjectionHead | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

        if weights_path and weights_path.exists():
            self._load(weights_path)

    def _ensure_backbone(self) -> None:
        """Lazy-load DINOv2 backbone."""
        if self.backbone is not None:
            return

        logger.info("Loading DINOv2 ViT-S/14 for general ReID on %s...", self.device)
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.backbone = self.backbone.to(self.device)

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        if self.head is None:
            self.head = ProjectionHead().to(self.device)

    def _load(self, weights_path: Path) -> None:
        """Load trained weights."""
        self._ensure_backbone()
        assert self.backbone is not None
        assert self.head is not None

        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.backbone.load_state_dict(state["backbone"])
        self.head.load_state_dict(state["head"])
        self.backbone.eval()
        self.head.eval()
        logger.info("Loaded general ReID weights from %s", weights_path)

    def save(self, path: Path | None = None) -> None:
        """Save trained weights."""
        assert self.backbone is not None
        assert self.head is not None

        save_path = path or WEIGHTS_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "head": self.head.state_dict(),
            },
            save_path,
        )
        logger.info("Saved general ReID weights to %s", save_path)

    def extract_embeddings(
        self,
        crops: list[NDArray[np.uint8]],
        batch_size: int = 32,
    ) -> NDArray[np.floating]:
        """Extract (N, 128) L2-normalized embeddings from BGR crops.

        Args:
            crops: List of BGR images (any size, resized to 224x224).
            batch_size: Batch size for inference.

        Returns:
            (N, 128) float32 embeddings (L2-normalized).
        """
        if not crops:
            return np.empty((0, PROJ_DIM), dtype=np.float32)

        self._ensure_backbone()
        assert self.backbone is not None
        assert self.head is not None
        assert self._mean is not None
        assert self._std is not None

        self.backbone.eval()
        self.head.eval()

        all_embeddings: list[NDArray[np.floating]] = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            batch = []
            for crop in batch_crops:
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                batch.append(tensor)

            batch_tensor = torch.stack(batch).to(self.device)
            batch_tensor = (batch_tensor - self._mean) / self._std

            with torch.inference_mode():
                features = self.backbone(batch_tensor)  # (B, 384)
                embeddings = self.head(features)  # (B, 128) L2-normalized

            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def extract_single(self, crop: NDArray[np.uint8]) -> NDArray[np.floating]:
        """Extract embedding for a single BGR crop."""
        result: NDArray[np.floating] = self.extract_embeddings([crop])[0]
        return result

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def train_on_dataset(
        self,
        dataset_dir: Path,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lr_backbone: float = DEFAULT_LR_BACKBONE,
        lr_head: float = DEFAULT_LR_HEAD,
        temperature: float = DEFAULT_TEMPERATURE,
        save_path: Path | None = None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Train on extracted crops with SupCon loss.

        Dataset structure:
            dataset_dir/{video_id}/player_{pid}/{rally}_{frame}.jpg

        Args:
            dataset_dir: Directory with extracted training crops.
            epochs: Number of training epochs.
            batch_size: Batch size (will sample players_per_batch × crops_per_player).
            lr_backbone: Learning rate for backbone (last UNFREEZE_BLOCKS).
            lr_head: Learning rate for projection head.
            temperature: Temperature for SupCon loss.
            save_path: Where to save weights. Defaults to WEIGHTS_PATH.
            verbose: Print progress.

        Returns:
            Training stats dict.
        """
        self._ensure_backbone()
        assert self.backbone is not None
        assert self.head is not None
        assert self._mean is not None
        assert self._std is not None

        # Load dataset index
        samples = self._load_dataset_index(dataset_dir)
        if not samples:
            logger.error("No training data found in %s", dataset_dir)
            return {"loss": 0.0, "n_samples": 0}

        n_identities = len(set(s[1] for s in samples))
        logger.info(
            "Training general ReID: %d crops, %d identities",
            len(samples), n_identities,
        )

        # Freeze backbone except last N blocks
        self._set_backbone_freeze(unfreeze_blocks=UNFREEZE_BLOCKS)

        # Optimizer with different LRs
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": self.head.parameters(), "lr": lr_head},
        ])

        self.backbone.train()
        self.head.train()

        best_loss = float("inf")

        for epoch in range(epochs):
            epoch_loss = self._train_epoch(
                samples, batch_size, temperature, optimizer,
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss

            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    "  Epoch %d/%d: loss=%.4f", epoch + 1, epochs, epoch_loss,
                )

        self.backbone.eval()
        self.head.eval()

        # Save
        self.save(save_path)

        return {
            "loss": best_loss,
            "n_samples": len(samples),
            "n_identities": n_identities,
        }

    def _load_dataset_index(
        self,
        dataset_dir: Path,
    ) -> list[tuple[Path, str]]:
        """Load dataset index: list of (image_path, identity_label).

        Identity label format: "{video_id}_{player_id}" to keep per-video
        identities distinct (player 1 in video A ≠ player 1 in video B).
        """
        samples: list[tuple[Path, str]] = []

        for video_dir in sorted(dataset_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name

            for player_dir in sorted(video_dir.iterdir()):
                if not player_dir.is_dir() or not player_dir.name.startswith("player_"):
                    continue
                pid = player_dir.name.split("_")[1]
                identity = f"{video_id}_{pid}"

                for img_file in sorted(player_dir.glob("*.jpg")):
                    samples.append((img_file, identity))

        return samples

    def _train_epoch(
        self,
        samples: list[tuple[Path, str]],
        batch_size: int,
        temperature: float,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """One training epoch with SupCon loss."""
        assert self.backbone is not None
        assert self.head is not None
        assert self._mean is not None
        assert self._std is not None

        from rallycut.tracking.reid_embeddings import _augment_crop

        rng = np.random.default_rng()

        # Group samples by identity for mining
        identity_samples: dict[str, list[Path]] = {}
        for path, identity in samples:
            identity_samples.setdefault(identity, []).append(path)

        identities = list(identity_samples.keys())
        n_identities = len(identities)

        # Sample batches: pick identities, then pick crops per identity
        crops_per_identity = max(2, batch_size // min(16, n_identities))
        identities_per_batch = batch_size // crops_per_identity

        total_loss = 0.0
        n_batches = 0

        # Shuffle and iterate
        perm = rng.permutation(n_identities)
        for start in range(0, n_identities, identities_per_batch):
            batch_ids = perm[start : start + identities_per_batch]
            if len(batch_ids) < 2:
                continue

            batch_crops: list[torch.Tensor] = []
            batch_labels: list[int] = []

            for label_idx, id_idx in enumerate(batch_ids):
                identity = identities[id_idx]
                paths = identity_samples[identity]

                # Sample crops_per_identity from this identity
                chosen = rng.choice(
                    len(paths),
                    size=min(crops_per_identity, len(paths)),
                    replace=len(paths) < crops_per_identity,
                )

                for ci in chosen:
                    img = cv2.imread(str(paths[ci]))
                    if img is None:
                        continue
                    # Apply augmentation
                    img_u8 = np.asarray(img, dtype=np.uint8)
                    aug = _augment_crop(img_u8, rng)
                    aug = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
                    aug = cv2.resize(aug, (224, 224))
                    tensor = torch.from_numpy(aug).permute(2, 0, 1).float() / 255.0
                    batch_crops.append(tensor)
                    batch_labels.append(label_idx)

            if len(batch_crops) < 4:
                continue

            batch_tensor = torch.stack(batch_crops).to(self.device)
            batch_tensor = (batch_tensor - self._mean) / self._std
            labels_tensor = torch.tensor(batch_labels, device=self.device)

            # Forward
            features = self.backbone(batch_tensor)
            embeddings = self.head(features)  # L2-normalized

            # SupCon loss
            loss = self._supcon_loss(embeddings, labels_tensor, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @staticmethod
    def _supcon_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Supervised Contrastive Loss (SupCon).

        For each anchor, positive pairs are same-label samples.
        The loss encourages same-label samples to cluster.

        Args:
            embeddings: (N, D) L2-normalized embeddings.
            labels: (N,) integer labels.
            temperature: Scaling temperature.

        Returns:
            Scalar loss.
        """
        n = embeddings.size(0)
        # Pairwise cosine similarity (embeddings are L2-normalized)
        sim = torch.mm(embeddings, embeddings.t()) / temperature

        # Mask: same label = positive pair
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        # Remove self-pairs
        self_mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
        pos_mask = label_eq & self_mask

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Log-sum-exp over all non-self pairs (denominator)
        exp_sim = torch.exp(sim) * self_mask.float()
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average log-prob over positive pairs
        log_prob = sim - log_sum_exp
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        n_pos = pos_mask.float().sum(dim=1)

        # Avoid division by zero for anchors with no positives
        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        loss = -(pos_log_prob[valid] / n_pos[valid]).mean()
        return loss

    def _set_backbone_freeze(self, unfreeze_blocks: int = UNFREEZE_BLOCKS) -> None:
        """Freeze all backbone parameters except the last N transformer blocks."""
        assert self.backbone is not None

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Unfreeze last N blocks
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            n_blocks = len(blocks)
            for i in range(max(0, n_blocks - unfreeze_blocks), n_blocks):
                for p in blocks[i].parameters():
                    p.requires_grad_(True)

        # Always unfreeze the final norm layer
        norm = getattr(self.backbone, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(True)

        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            "Backbone: unfroze last %d blocks (%d/%d params trainable, %.1f%%)",
            unfreeze_blocks, n_trainable, n_total, n_trainable / n_total * 100,
        )
