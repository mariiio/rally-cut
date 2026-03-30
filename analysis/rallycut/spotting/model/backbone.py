"""RegNet-Y backbone with GSM temporal shift modules.

Loads a pretrained RegNet-Y 200MF from timm and inserts GSM wrappers
around specified stages for lightweight temporal modeling.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from rallycut.spotting.config import BackboneConfig
from rallycut.spotting.model.gsm import GSMWrapper


class RegNetYGSM(nn.Module):
    """RegNet-Y backbone with Gated Shift Modules for temporal awareness.

    Processes clips of T frames, producing per-frame feature vectors.
    GSM modules are inserted after specified stages to enable temporal
    communication between neighboring frames without 3D convolutions.
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config

        # Load pretrained RegNet-Y from timm (features_only for stage access)
        base = timm.create_model(config.model_name, pretrained=config.pretrained)

        # Extract components
        self.stem = base.stem
        stages = [base.s1, base.s2, base.s3, base.s4]

        # Get input channels for each stage by checking stem/stage output channels
        stage_in_channels = self._get_stage_in_channels(base)

        # Wrap specified stages with GSM
        wrapped_stages: list[nn.Module] = []
        for i, stage in enumerate(stages):
            stage_idx = i + 1  # stages are 1-indexed
            if stage_idx in config.gsm_stages:
                wrapped = GSMWrapper(stage, stage_in_channels[i])  # type: ignore[arg-type]
                wrapped_stages.append(wrapped)
            else:
                wrapped_stages.append(stage)  # type: ignore[arg-type]

        self.stages = nn.ModuleList(wrapped_stages)
        self.final_conv = base.final_conv
        self.feature_dim = config.feature_dim

    @staticmethod
    def _get_stage_in_channels(base: nn.Module) -> list[int]:
        """Get input channel count for each of the 4 stages."""
        channels: list[int] = []
        # stem output = s1 input
        channels.append(base.stem.out_channels)  # type: ignore[union-attr, arg-type]
        # Each stage's input = previous stage's output
        for stage_name in ["s1", "s2", "s3"]:
            stage = getattr(base, stage_name)
            # Get output channels from the last block
            last_block = list(stage.children())[-1]
            if hasattr(last_block, "out_channels"):
                channels.append(last_block.out_channels)
            else:
                # Fallback: check conv3 or downsample
                for m in reversed(list(last_block.modules())):
                    if isinstance(m, nn.Conv2d):
                        channels.append(m.out_channels)
                        break
        return channels

    def set_num_frames(self, t: int) -> None:
        """Set the temporal dimension for GSM modules."""
        for stage in self.stages:
            if isinstance(stage, GSMWrapper):
                stage.set_num_frames(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-frame features from a clip.

        Args:
            x: (B, T, 3, H, W) clip tensor.

        Returns:
            (B, T, D) per-frame feature vectors where D = feature_dim.
        """
        b, t, c, h, w = x.shape
        self.set_num_frames(t)

        # Flatten batch and time for 2D processing
        x = x.reshape(b * t, c, h, w)

        # Run through backbone stages
        x = self.stem(x)  # type: ignore[operator]
        for stage in self.stages:
            x = stage(x)
        x = self.final_conv(x)  # type: ignore[operator]

        # Global average pool: (B*T, D, H', W') -> (B*T, D)
        x = x.mean(dim=[2, 3])

        # Reshape to (B, T, D)
        x = x.view(b, t, -1)
        return x
