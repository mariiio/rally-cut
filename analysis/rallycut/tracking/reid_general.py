"""General volleyball ReID model — OSNet-x1.0 with MSMT17 pretrained weights.

Learns player-discriminative embeddings from GT-labeled training data across
multiple videos. Unlike the per-video PlayerReIDClassifier (reid_embeddings.py),
this model generalizes to unseen videos without requiring user reference crops.

Architecture:
    OSNet-x1.0 backbone (MSMT17 pretrained, 512-dim)
    → projection head (512 → 128 → L2 normalize)

Training:
    SupCon (Supervised Contrastive) loss on video-grouped batches.
    Hard negatives: different players from the same video.
    Easy negatives: players from different videos.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional  # noqa: N812
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Model constants
EMBED_DIM = 512  # OSNet-x1.0 feature dimension
PROJ_DIM = 128  # Projection head output
WEIGHTS_PATH = Path(__file__).parent.parent.parent / "weights" / "reid" / "general_reid.pt"

# MSMT17 pretrained weights (auto-downloaded via gdown)
_MSMT17_GDRIVE_ID = "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x"
_MSMT17_NUM_CLASSES = 4101  # MSMT17 combineall identity count
_MSMT17_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "torch", "checkpoints", "osnet_x1_0_msmt17.pth",
)

# OSNet input: 256×128 (standard ReID dimensions)
_INPUT_HEIGHT = 256
_INPUT_WIDTH = 128

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_LR_BACKBONE = 1e-5
DEFAULT_LR_HEAD = 1e-3
DEFAULT_TEMPERATURE = 0.07


def _default_device() -> str:
    """Auto-detect best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class _OSNetFeatureExtractor(nn.Module):
    """Wrapper that always returns 512-dim features, even in train mode.

    OSNet's forward() returns classifier logits in train mode and features
    in eval mode. This wrapper bypasses the classifier to always return
    the 512-dim fc output, enabling gradient flow through the backbone
    during contrastive training.
    """

    def __init__(self, osnet: nn.Module) -> None:
        super().__init__()
        self.osnet = osnet
        self._feature_dim: int = getattr(osnet, "feature_dim", EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        featuremaps = getattr(self.osnet, "featuremaps")
        global_avgpool = getattr(self.osnet, "global_avgpool")
        fc = getattr(self.osnet, "fc", None)

        x = featuremaps(x)
        v: torch.Tensor = global_avgpool(x)
        v = v.view(v.size(0), -1)
        if fc is not None:
            v = fc(v)
        return v

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


def _load_osnet_backbone(device: str) -> tuple[nn.Module, int]:
    """Load OSNet-x1.0 with MSMT17 pretrained weights.

    Downloads weights from Google Drive on first call (17 MB).

    Returns:
        (model, feature_dim) where model always outputs (B, 512) features.
    """
    import torchreid

    # Download MSMT17 weights if not cached
    if not os.path.exists(_MSMT17_CACHE_PATH):
        import gdown

        os.makedirs(os.path.dirname(_MSMT17_CACHE_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={_MSMT17_GDRIVE_ID}"
        logger.info("Downloading OSNet-x1.0 MSMT17 weights...")
        gdown.download(url, _MSMT17_CACHE_PATH, quiet=True)

    # Build model with matching num_classes for weight loading
    osnet = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=_MSMT17_NUM_CLASSES,
        loss="softmax",
        pretrained=False,
    )

    # Load MSMT17 pretrained weights
    state = torch.load(_MSMT17_CACHE_PATH, map_location=device, weights_only=False)
    osnet.load_state_dict(state)

    # Wrap to always return 512-dim features (not classifier logits)
    model = _OSNetFeatureExtractor(osnet).to(device)
    feature_dim = model.feature_dim  # 512
    logger.info(
        "Loaded OSNet-x1.0 MSMT17 on %s (%d-dim, %.1fM params)",
        device, feature_dim, sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model, feature_dim


class ProjectionHead(nn.Module):
    """MLP projection head: 512 → BN → ReLU → 128 → BN → L2 (SimCLR pattern)."""

    def __init__(self, in_dim: int = EMBED_DIM, out_dim: int = PROJ_DIM) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return functional.normalize(x, dim=1)


class GeneralReIDModel:
    """Contrastive-trained OSNet model for volleyball player ReID.

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
        """Lazy-load OSNet backbone with MSMT17 weights."""
        if self.backbone is not None:
            return

        self.backbone, _feat_dim = _load_osnet_backbone(self.device)

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        if self.head is None:
            self.head = ProjectionHead().to(self.device)

    def _load(self, weights_path: Path) -> None:
        """Load fine-tuned weights (backbone + projection head)."""
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
        """Save fine-tuned weights."""
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
            crops: List of BGR images (any size, resized to 256×128).
            batch_size: Batch size for inference.

        Returns:
            (N, PROJ_DIM) float32 embeddings (L2-normalized).
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
                img = cv2.resize(img, (_INPUT_WIDTH, _INPUT_HEIGHT))
                tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                batch.append(tensor)

            batch_tensor = torch.stack(batch).to(self.device)
            batch_tensor = (batch_tensor - self._mean) / self._std

            with torch.inference_mode():
                features = self.backbone(batch_tensor)  # (B, 512)
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
            lr_backbone: Learning rate for backbone unfrozen layers.
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

        # Freeze backbone except last conv stage
        self._set_backbone_freeze()

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
        if save_path is not None:
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
                    aug = cv2.resize(aug, (_INPUT_WIDTH, _INPUT_HEIGHT))
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
        """
        n = embeddings.size(0)
        sim = torch.mm(embeddings, embeddings.t()) / temperature

        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        self_mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
        pos_mask = label_eq & self_mask

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * self_mask.float()
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        log_prob = sim - log_sum_exp
        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        n_pos = pos_mask.float().sum(dim=1)

        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        loss = -(pos_log_prob[valid] / n_pos[valid]).mean()
        return loss

    def _set_backbone_freeze(self) -> None:
        """Freeze backbone except the last conv stage (conv5) and fc layer."""
        assert self.backbone is not None

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # _OSNetFeatureExtractor wraps osnet — access inner model's named params.
        # OSNet structure: conv1, conv2, conv3, conv4, conv5, fc, classifier
        # Unfreeze conv5 (last convolutional stage) and fc (feature layer)
        for name, p in self.backbone.named_parameters():
            # With wrapper: params are named "osnet.conv5.xxx", "osnet.fc.xxx"
            if "conv5" in name or ".fc." in name or name.endswith(".fc"):
                p.requires_grad_(True)

        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            "Backbone: unfroze conv5+fc (%d/%d params trainable, %.1f%%)",
            n_trainable, n_total, n_trainable / n_total * 100,
        )
