"""DINOv2-based ReID for cross-rally player matching.

Drop-in alternative to `GeneralReIDModel` (OSNet) for fixtures where
OSNet's surveillance-trained embeddings are out-of-distribution
(e.g., beach VB in swimsuits with sand backgrounds — see
`blind_regime_ceiling_2026_05_02.md`).

Uses a frozen DINOv2 backbone via `torch.hub`. No fine-tuning, no
projection head — the raw CLS token is L2-normalized and used directly.

Backbone choice:
- `dinov2_vits14` (384-dim, smaller): faster, less expressive
- `dinov2_vitl14` (1024-dim, larger): the SOTA pick — robust embeddings
  on outdoor / sports / OOD imagery. Default.

Selection at runtime via `RALLYCUT_REID_BACKBONE` env var:
  - unset / "osnet": current OSNet (`GeneralReIDModel`)
  - "dinov2_vits14": DINOv2 ViT-S/14
  - "dinov2_vitl14": DINOv2 ViT-L/14 (recommended)

Interface intentionally matches `GeneralReIDModel.extract_embeddings`
so the existing match-players pipeline can swap backbones without
caller changes.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional  # noqa: N812
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

ENV_BACKBONE_FLAG = "RALLYCUT_REID_BACKBONE"

# DINOv2 standard input size (must be multiple of patch_size=14).
_DINOV2_INPUT_HW = (224, 224)

_BACKBONE_CACHE: dict[
    str, tuple[nn.Module, torch.Tensor, torch.Tensor, int]
] = {}


def get_backbone_choice() -> str:
    """Return the configured backbone name (lowercase).

    Recognized values:
      - "osnet" (default; uses GeneralReIDModel)
      - "dinov2_vits14"
      - "dinov2_vitl14"
    """
    return os.environ.get(ENV_BACKBONE_FLAG, "osnet").lower()


def is_dinov2_selected() -> bool:
    return get_backbone_choice().startswith("dinov2")


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _get_dinov2_backbone(
    name: str, device: str,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, int]:
    """Load + cache a DINOv2 backbone (frozen, eval mode).

    Cached per (name, device) so a single match-players run uses one
    weights load.
    """
    cache_key = f"{name}:{device}"
    if cache_key in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[cache_key]

    logger.info("Loading %s on %s ...", name, device)
    model = torch.hub.load("facebookresearch/dinov2", name)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    embed_dim = int(model.embed_dim)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    _BACKBONE_CACHE[cache_key] = (model, mean, std, embed_dim)
    logger.info(
        "Loaded %s on %s (%d-dim CLS token, frozen)",
        name, device, embed_dim,
    )
    return _BACKBONE_CACHE[cache_key]


class DinoV2ReIDModel:
    """Frozen-DINOv2 ReID — same interface as `GeneralReIDModel`.

    No training, no projection head. The CLS token is L2-normalized and
    returned as-is. Embedding dimensionality depends on the backbone:
      - dinov2_vits14: 384
      - dinov2_vitl14: 1024
    """

    def __init__(
        self,
        backbone: str = "dinov2_vitl14",
        device: str | None = None,
    ) -> None:
        if not backbone.startswith("dinov2"):
            raise ValueError(f"DinoV2ReIDModel requires a dinov2_* backbone, got {backbone!r}")
        self.backbone_name = backbone
        self.device = device or _default_device()
        # Lazy-load on first extract_embeddings call.
        self._loaded: tuple[nn.Module, torch.Tensor, torch.Tensor, int] | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded is None:
            self._loaded = _get_dinov2_backbone(self.backbone_name, self.device)

    @property
    def embed_dim(self) -> int:
        self._ensure_loaded()
        assert self._loaded is not None
        return self._loaded[3]

    def extract_embeddings(
        self,
        crops: list[NDArray[np.uint8]],
        batch_size: int = 16,
    ) -> NDArray[np.floating]:
        """Extract L2-normalized embeddings from BGR crops.

        Returns: (N, embed_dim) float32, L2-normalized along axis=1.
        """
        if not crops:
            self._ensure_loaded()
            return np.empty((0, self.embed_dim), dtype=np.float32)

        self._ensure_loaded()
        assert self._loaded is not None
        model, mean, std, _ = self._loaded

        out: list[NDArray[np.floating]] = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i: i + batch_size]
            tensors = []
            for crop in batch:
                if crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (_DINOV2_INPUT_HW[1], _DINOV2_INPUT_HW[0]))
                tensors.append(
                    torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                )
            if not tensors:
                continue
            x = torch.stack(tensors).to(self.device)
            x = (x - mean) / std
            with torch.inference_mode():
                feats = model(x)  # CLS token, (B, embed_dim)
            feats = functional.normalize(feats, p=2, dim=1)
            out.append(feats.detach().cpu().numpy().astype(np.float32))

        if not out:
            return np.empty((0, self.embed_dim), dtype=np.float32)
        return np.concatenate(out, axis=0)

    def extract_single(self, crop: NDArray[np.uint8]) -> NDArray[np.floating]:
        result: NDArray[np.floating] = self.extract_embeddings([crop])[0]
        return result
