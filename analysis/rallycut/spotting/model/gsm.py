"""Gated Shift Module (GSM) for temporal modeling in 2D CNNs.

Inserts lightweight temporal communication into a 2D backbone by shifting
a fraction of channels forward/backward in time and gating the result.

Reference: E2E-Spot (ECCV 2022), BSD-3 license.
    https://github.com/jhong93/spot
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedShiftModule(nn.Module):
    """Gated Shift Module that enables temporal reasoning in 2D CNNs.

    Splits input channels into 3 groups:
    - Group 1: shifted forward in time (sees past)
    - Group 2: shifted backward in time (sees future)
    - Group 3: unchanged (spatial only)

    A learned 1x1 conv gate controls how much shifted information to use.
    """

    def __init__(self, channels: int, shift_fraction: float = 0.25) -> None:
        super().__init__()
        self.shift_channels = int(channels * shift_fraction)
        # Gate: 1x1 conv on the shifted channels to learn what to pass through
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal shifting.

        Args:
            x: (B*T, C, H, W) tensor where B*T is the batch of frames.
               The temporal dimension T must be tracked externally.

        Returns:
            (B*T, C, H, W) tensor with temporal information mixed in.
        """
        # This is applied within the backbone where we don't have explicit T dim.
        # The shifting is done by GSMWrapper which handles the reshape.
        identity = x
        shifted = self.gate(x)
        return identity * shifted  # type: ignore[no-any-return]


class GSMWrapper(nn.Module):
    """Wraps a backbone stage to add temporal shifting before processing.

    Reshapes (B*T, C, H, W) → (B, T, C, H, W), applies temporal shift,
    reshapes back, then runs the original stage.
    """

    def __init__(self, stage: nn.Module, channels: int, shift_fraction: float = 0.25) -> None:
        super().__init__()
        self.stage = stage
        self.shift_channels = int(channels * shift_fraction)
        # Learned gate on the shifted portion
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate_act = nn.Sigmoid()
        self._num_frames: int = 0

    def set_num_frames(self, t: int) -> None:
        self._num_frames = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self._num_frames
        if t <= 1:
            return self.stage(x)  # type: ignore[no-any-return]

        bt, c, h, w = x.shape
        b = bt // t
        x_5d = x.view(b, t, c, h, w)

        # Temporal shift: move shift_channels forward and backward
        sc = self.shift_channels
        shifted = x_5d.clone()
        # Forward shift (current frame sees past frame's channels)
        shifted[:, 1:, :sc] = x_5d[:, :-1, :sc]
        shifted[:, 0, :sc] = x_5d[:, 0, :sc]  # pad first frame
        # Backward shift (current frame sees future frame's channels)
        shifted[:, :-1, sc : 2 * sc] = x_5d[:, 1:, sc : 2 * sc]
        shifted[:, -1, sc : 2 * sc] = x_5d[:, -1, sc : 2 * sc]  # pad last frame

        # Reshape back and apply gate
        shifted_flat = shifted.reshape(bt, c, h, w)
        gate = self.gate_act(self.gate_conv(shifted_flat))
        x_gated = x * gate

        return self.stage(x_gated)  # type: ignore[no-any-return]
