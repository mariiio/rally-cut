"""Tests for the frozen ResNet-18 backbone + MLP head."""
import torch

from rallycut.ml.crop_head.backbone import FrozenResNet18
from rallycut.ml.crop_head.head import CropHeadMLP


def test_backbone_output_shape() -> None:
    bb = FrozenResNet18()
    x = torch.rand(4, 3, 64, 64)
    out = bb(x)
    assert out.shape == (4, 512)


def test_backbone_is_frozen() -> None:
    bb = FrozenResNet18()
    assert all(not p.requires_grad for p in bb.parameters())


def test_backbone_deterministic_under_eval() -> None:
    """Frozen backbone with BN in eval mode must be deterministic."""
    bb = FrozenResNet18()
    x = torch.rand(2, 3, 64, 64)
    out_a = bb(x)
    out_b = bb(x)
    assert torch.allclose(out_a, out_b)


def test_mlp_head_shape() -> None:
    head = CropHeadMLP(d_in=1024)
    x = torch.rand(4, 9, 1024)
    out = head(x)
    assert out.shape == (4,)
