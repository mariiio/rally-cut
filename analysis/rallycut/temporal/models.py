"""Temporal models for sequence labeling in rally detection.

Progressive complexity:
- v1: LearnedSmoothing - Simple 1D Conv with learned threshold
- v2: ConvCRF - Multi-layer 1D CNN + CRF for transition learning
- v3: BiLSTMCRF - BiLSTM + CRF (if needed)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from torchcrf import CRF  # noqa: F401 (used for type hints only)


class LearnedSmoothing(nn.Module):
    """Temporal model v1: Linear projection + 1D convolution for temporal smoothing.

    This is the simplest temporal model that projects features to logits,
    applies learned smoothing, and outputs binary classifications.

    Architecture:
        input (batch, seq_len, feature_dim)
        -> Linear projection to 2D logits
        -> Softmax -> Conv1D smoothing -> learned threshold
        -> output

    Args:
        feature_dim: Input feature dimension (768 for VideoMAE).
        kernel_size: Size of smoothing kernel (default 5, covers ~8 seconds at stride 48).
        dropout: Dropout rate (default 0.3).
    """

    def __init__(
        self,
        feature_dim: int = 768,
        kernel_size: int = 5,
        dropout: float = 0.3,
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size

        # Class weight for BCE loss (handles imbalanced classes)
        # pos_weight > 1 increases recall, < 1 increases precision
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight]) if pos_weight is not None else None,
        )

        # Linear projection from features to 2-class logits
        self.classifier = nn.Linear(feature_dim, 2)

        # Learned smoothing kernel (initialized to Gaussian-like weights)
        self.conv = nn.Conv1d(
            in_channels=2,  # NO_RALLY and RALLY logits
            out_channels=2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=2,  # Separate smoothing for each class
        )

        # Initialize with Gaussian-like weights for smooth start
        self._init_weights()

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Learned threshold (initialized to 0.5)
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def _init_weights(self) -> None:
        """Initialize conv weights to Gaussian-like smoothing kernel."""
        with torch.no_grad():
            # Create Gaussian-like kernel
            sigma = self.kernel_size / 4
            x = torch.arange(self.kernel_size).float() - self.kernel_size // 2
            gaussian = torch.exp(-x ** 2 / (2 * sigma ** 2))
            gaussian = gaussian / gaussian.sum()

            # Apply to both channels
            for i in range(2):
                self.conv.weight[i, 0, :] = gaussian

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass with optional training loss.

        Args:
            features: Input features of shape (batch, seq_len, feature_dim)
            labels: Optional labels of shape (batch, seq_len) with values 0/1

        Returns:
            Dictionary with:
                - "predictions": Binary predictions (batch, seq_len)
                - "probs": Smoothed probabilities (batch, seq_len)
                - "emissions": Raw logits (batch, seq_len, 2)
                - "loss": Loss value if labels provided
        """
        # Project features to logits
        logits = self.classifier(features)  # (batch, seq_len, 2)

        # Transpose for conv: (batch, seq_len, 2) -> (batch, 2, seq_len)
        logits_t = logits.transpose(1, 2)

        # Apply learned smoothing on logits
        smoothed_logits = self.conv(logits_t)  # (batch, 2, seq_len)
        smoothed_logits = self.dropout(smoothed_logits)

        # Transpose back: (batch, 2, seq_len) -> (batch, seq_len, 2)
        smoothed_logits = smoothed_logits.transpose(1, 2)

        # Get rally logits and apply sigmoid for probabilities
        rally_logits = smoothed_logits[:, :, 1] - smoothed_logits[:, :, 0]  # (batch, seq_len)
        rally_probs = torch.sigmoid(rally_logits)

        # Apply learned threshold for predictions
        predictions = (rally_probs > self.threshold).long()

        result: dict[str, torch.Tensor] = {
            "predictions": predictions,
            "probs": rally_probs,
            "emissions": logits,
        }

        # Compute loss if training
        if labels is not None:
            # Binary cross entropy with logits (with optional class weighting)
            # pos_weight is a registered buffer (Tensor or None)
            pos_weight: torch.Tensor | None = self.pos_weight  # type: ignore[assignment]
            loss = F.binary_cross_entropy_with_logits(
                rally_logits, labels.float(), reduction="mean", pos_weight=pos_weight
            )
            result["loss"] = loss

        return result

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference.

        Args:
            features: Input features of shape (batch, seq_len, feature_dim)

        Returns:
            Binary predictions of shape (batch, seq_len)
        """
        with torch.no_grad():
            return self.forward(features)["predictions"]


class ConvCRF(nn.Module):
    """Temporal model v2: 1D CNN + CRF for sequence labeling.

    This model uses multi-layer 1D convolutions for temporal feature extraction
    followed by a CRF layer for learning transition dynamics.

    Architecture:
        input (batch, seq_len, feature_dim)
        -> Conv1D stack (with BatchNorm, ReLU, Dropout)
        -> Linear emission layer
        -> CRF for sequence decoding

    Args:
        feature_dim: Input feature dimension (768 for VideoMAE)
        hidden_dim: Hidden dimension for conv layers (default 128)
        num_layers: Number of conv layers (default 3)
        kernel_size: Conv kernel size (default 5)
        num_states: Number of output states (default 2: NO_RALLY, RALLY)
        dropout: Dropout rate (default 0.4)
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 5,
        num_states: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_states = num_states

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # 1D Conv stack
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.conv_stack = nn.Sequential(*layers)

        # Emission layer (logits for CRF)
        self.emission = nn.Linear(hidden_dim, num_states)

        # CRF for transition learning
        # Imported lazily to avoid hard dependency
        self._crf: CRF | None = None

    @property
    def crf(self) -> CRF:
        """Lazy-load CRF to avoid import issues during model creation."""
        if self._crf is None:
            try:
                from torchcrf import CRF
            except ImportError as e:
                raise ImportError(
                    "pytorch-crf is required for ConvCRF model. "
                    "Install with: pip install pytorch-crf"
                ) from e
            self._crf = CRF(self.num_states, batch_first=True).to(
                next(self.parameters()).device
            )
        return self._crf

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Input features of shape (batch, seq_len, feature_dim)
            labels: Optional labels of shape (batch, seq_len) with values 0/1

        Returns:
            Dictionary with:
                - "predictions": Predicted sequence (batch, seq_len)
                - "emissions": Emission logits (batch, seq_len, num_states)
                - "loss": Negative log-likelihood if labels provided
        """
        batch_size, seq_len, _ = features.shape

        # Project input
        x = self.input_proj(features)  # (batch, seq_len, hidden_dim)

        # Transpose for conv: (batch, seq_len, hidden) -> (batch, hidden, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_stack(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden)

        # Compute emission logits
        emissions = self.emission(x)  # (batch, seq_len, num_states)

        result: dict[str, torch.Tensor] = {"emissions": emissions}

        if labels is not None:
            # Training: compute negative log-likelihood
            # CRF forward returns log-likelihood, negate for loss
            loss = -self.crf(emissions, labels)
            result["loss"] = loss

            # Also compute predictions for metrics during training
            predictions = self.crf.decode(emissions)
            result["predictions"] = torch.tensor(predictions, device=features.device)
        else:
            # Inference: Viterbi decoding
            predictions = self.crf.decode(emissions)
            result["predictions"] = torch.tensor(predictions, device=features.device)

        return result

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference.

        Args:
            features: Input features of shape (batch, seq_len, feature_dim)

        Returns:
            Predicted labels of shape (batch, seq_len)
        """
        with torch.no_grad():
            return self.forward(features)["predictions"]


class BiLSTMCRF(nn.Module):
    """Temporal model v3: BiLSTM + CRF for sequence labeling.

    This model uses a bidirectional LSTM for temporal feature extraction
    followed by a CRF layer. Use this only if v2 (ConvCRF) doesn't meet
    success criteria.

    Architecture:
        input (batch, seq_len, feature_dim)
        -> Input projection
        -> BiLSTM
        -> Linear emission layer
        -> CRF for sequence decoding

    Args:
        feature_dim: Input feature dimension (768 for VideoMAE)
        hidden_dim: Hidden dimension for LSTM (default 128)
        num_layers: Number of LSTM layers (default 1)
        num_states: Number of output states (default 2: NO_RALLY, RALLY)
        dropout: Dropout rate (default 0.5)
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_states: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_states = num_states

        # Input projection to reduce dimensionality
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional doubles output
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Emission layer
        self.emission = nn.Linear(hidden_dim, num_states)

        # CRF (lazy loaded)
        self._crf: CRF | None = None

    @property
    def crf(self) -> CRF:
        """Lazy-load CRF."""
        if self._crf is None:
            try:
                from torchcrf import CRF
            except ImportError as e:
                raise ImportError(
                    "pytorch-crf is required for BiLSTMCRF model. "
                    "Install with: pip install pytorch-crf"
                ) from e
            self._crf = CRF(self.num_states, batch_first=True).to(
                next(self.parameters()).device
            )
        return self._crf

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Input features of shape (batch, seq_len, feature_dim)
            labels: Optional labels of shape (batch, seq_len)

        Returns:
            Dictionary with predictions, emissions, and optionally loss.
        """
        # Project input
        x = self.input_proj(features)  # (batch, seq_len, hidden_dim)

        # BiLSTM
        x, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Emission logits
        emissions = self.emission(x)  # (batch, seq_len, num_states)

        result: dict[str, torch.Tensor] = {"emissions": emissions}

        if labels is not None:
            loss = -self.crf(emissions, labels)
            result["loss"] = loss
            predictions = self.crf.decode(emissions)
            result["predictions"] = torch.tensor(predictions, device=features.device)
        else:
            predictions = self.crf.decode(emissions)
            result["predictions"] = torch.tensor(predictions, device=features.device)

        return result

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Inference method."""
        with torch.no_grad():
            return self.forward(features)["predictions"]


# Model registry for easy selection
TEMPORAL_MODELS = {
    "v1": LearnedSmoothing,
    "v2": ConvCRF,
    "v3": BiLSTMCRF,
}


def get_temporal_model(
    version: str,
    feature_dim: int = 768,
    *,
    # v1 (LearnedSmoothing) specific
    kernel_size: int = 5,
    dropout: float = 0.4,
    pos_weight: float | None = None,
    # v2/v3 specific
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_states: int = 2,
) -> nn.Module:
    """Create a temporal model by version.

    Args:
        version: Model version ("v1", "v2", "v3")
        feature_dim: Input feature dimension
        kernel_size: Conv kernel size (v1 only)
        dropout: Dropout rate (all models)
        pos_weight: Positive class weight for BCE loss (v1 only)
        hidden_dim: Hidden dimension (v2/v3 only)
        num_layers: Number of layers (v2/v3 only)
        num_states: Number of output states (v2/v3 only)

    Returns:
        Instantiated temporal model
    """
    if version not in TEMPORAL_MODELS:
        raise ValueError(f"Unknown model version: {version}. Choose from {list(TEMPORAL_MODELS.keys())}")

    if version == "v1":
        return LearnedSmoothing(
            feature_dim=feature_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            pos_weight=pos_weight,
        )
    elif version == "v2":
        return ConvCRF(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            num_states=num_states,
            dropout=dropout,
        )
    else:  # v3
        return BiLSTMCRF(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_states=num_states,
            dropout=dropout,
        )
