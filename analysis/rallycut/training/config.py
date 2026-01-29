"""Training configuration for VideoMAE fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for VideoMAE fine-tuning.

    Optimized for:
    - Small dataset (~311 rallies, ~2000 samples)
    - Local Mac training (MPS backend)
    - 3-class classification (NO_PLAY, PLAY, SERVICE)
    """

    # Model paths
    base_model_path: Path = field(
        default_factory=lambda: Path("weights/videomae/game_state_classifier")
    )
    output_dir: Path = field(
        default_factory=lambda: Path("weights/videomae/beach_volleyball")
    )

    # Training hyperparameters (optimized for small dataset + MPS)
    # Note: 1e-5 is stable for fine-tuning; use 5e-5 only for aggressive training
    learning_rate: float = 1e-5
    batch_size: int = 2  # Very small for MPS memory limits
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    num_epochs: int = 25
    warmup_ratio: float = 0.1
    weight_decay: float = 0.05

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0  # Gradient clipping threshold

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.01

    # Data
    train_split: float = 0.9
    val_split: float = 0.1
    seed: int = 42

    # VideoMAE input specs
    num_frames: int = 16
    image_size: int = 224
    fps: float = 30.0  # Target FPS for normalization

    # Sampling parameters
    samples_per_rally_play: int = 3  # PLAY samples from middle
    samples_per_rally_service: int = 1  # SERVICE from start
    samples_per_gap_no_play: int = 3  # NO_PLAY from gaps
    min_gap_duration_seconds: float = 2.0  # Min gap to sample NO_PLAY

    # MPS-specific settings
    use_mps: bool = True
    dataloader_num_workers: int = 0  # MPS issues with multiprocessing

    # Logging and checkpointing
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100  # Used when save_strategy="steps"
    save_strategy: str = "epoch"  # "epoch" for local, "steps" for Modal (preemption resilience)

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def window_duration_seconds(self) -> float:
        """Duration of a 16-frame window in seconds."""
        return self.num_frames / self.fps


# Label mapping (matches existing model)
LABEL_MAP = {
    0: "NO_PLAY",
    1: "PLAY",
    2: "SERVICE",
}

LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
