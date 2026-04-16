"""Frozen DINOv2 ViT-S/14 backbone wrapper for training-time use.

Reuses the singleton from `rallycut.tracking.reid_embeddings._get_backbone` so
training and the production pipeline share one cached model instance. Returns
L2-normalized 384-d features from BGR uint8 input — byte-equivalent to
`extract_backbone_features` so eval cache + training-time embeddings align.

The backbone is held under torch.no_grad — head params get gradients, backbone
does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger("within_team_reid.model.backbone")


@dataclass
class BackboneRunner:
    """Stateful wrapper around DINOv2 ViT-S/14 for training-time forward."""

    device: torch.device
    embed_dim: int = 384

    def __post_init__(self) -> None:
        from rallycut.tracking.reid_embeddings import _get_backbone

        model, mean, std, dim = _get_backbone(self.device.type)
        self._model = model
        self._mean = mean       # (1, 3, 1, 1) on device
        self._std = std         # (1, 3, 1, 1) on device
        self.embed_dim = dim
        # Belt-and-suspenders: backbone must stay frozen.
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()
        logger.info(
            "BackboneRunner ready: dim=%d on %s (frozen)", self.embed_dim, self.device,
        )

    def forward(self, crops_bgr_uint8: torch.Tensor) -> torch.Tensor:
        """Forward (N, 3, 224, 224) BGR uint8 → (N, 384) L2-normed float32.

        Internally: BGR→RGB channel swap, /255, ImageNet mean/std, DINOv2 forward,
        L2-normalize. Matches `reid_embeddings.extract_backbone_features` semantics.
        """
        if crops_bgr_uint8.dtype != torch.uint8:
            raise TypeError(
                f"BackboneRunner expects uint8 input; got {crops_bgr_uint8.dtype}"
            )
        if crops_bgr_uint8.shape[1:] != (3, 224, 224):
            raise ValueError(
                f"BackboneRunner expects (N, 3, 224, 224); got {tuple(crops_bgr_uint8.shape)}"
            )

        x = crops_bgr_uint8.to(self.device).float() / 255.0
        # BGR → RGB: reverse channel dim
        x = x.flip(dims=[1])
        x = (x - self._mean) / self._std

        with torch.no_grad():
            feats = self._model(x)
        feats = F.normalize(feats, dim=1)
        return feats
