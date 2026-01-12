"""PyTorch Dataset for VideoMAE fine-tuning on beach volleyball."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor

from rallycut.training.config import TrainingConfig

if TYPE_CHECKING:
    from rallycut.training.sampler import TrainingSample


class BeachVolleyballDataset(Dataset[dict[str, Any]]):
    """Dataset for VideoMAE fine-tuning on beach volleyball.

    Loads 16-frame windows from videos based on TrainingSample specifications.
    Applies VideoMAE preprocessing (resize, normalize).
    """

    def __init__(
        self,
        samples: list[TrainingSample],
        video_paths: dict[str, Path],
        processor: VideoMAEImageProcessor | None = None,
        config: TrainingConfig | None = None,
        augment: bool = False,
    ):
        """Initialize dataset.

        Args:
            samples: List of TrainingSample objects
            video_paths: Mapping from video_id to local file path
            processor: VideoMAE image processor (loaded if not provided)
            config: Training configuration
            augment: Whether to apply data augmentation
        """
        self.samples = samples
        self.video_paths = video_paths
        self.config = config or TrainingConfig()
        self.augment = augment

        # Load processor
        if processor is None:
            model_path = self.config.base_model_path
            if model_path.exists():
                self.processor = VideoMAEImageProcessor.from_pretrained(str(model_path))
            else:
                self.processor = VideoMAEImageProcessor.from_pretrained(
                    "MCG-NJU/videomae-base-finetuned-kinetics"
                )
        else:
            self.processor = processor

        # Cache for video captures (lazy loading)
        self._video_caps: dict[str, cv2.VideoCapture] = {}

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a training sample.

        Returns:
            dict with 'pixel_values' (tensor) and 'labels' (int)
        """
        sample = self.samples[idx]

        # Load frames
        frames = self._load_frames(sample.video_id, sample.start_frame)

        # Apply augmentation if enabled
        if self.augment:
            frames = self._augment_frames(frames)

        # Process frames with VideoMAE processor
        inputs = self.processor(
            list(frames),
            return_tensors="pt",
        )

        # Remove batch dimension (added by processor)
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": sample.label,
        }

    def _load_frames(self, video_id: str, start_frame: int) -> np.ndarray:
        """Load 16 consecutive frames from video.

        Args:
            video_id: Video identifier
            start_frame: Starting frame number

        Returns:
            Array of shape (16, H, W, 3) in RGB format
        """
        video_path = self.video_paths.get(video_id)
        if video_path is None:
            raise ValueError(f"No video path for video_id: {video_id}")

        # Get or create video capture
        cap = self._get_video_capture(video_id, video_path)

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[np.ndarray] = []
        for _ in range(self.config.num_frames):
            ret, frame = cap.read()
            if not ret:
                # If we run out of frames, repeat last frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    # Create black frame as fallback
                    frames.append(
                        np.zeros(
                            (self.config.image_size, self.config.image_size, 3),
                            dtype=np.uint8,
                        )
                    )
            else:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        return np.array(frames)

    def _get_video_capture(self, video_id: str, video_path: Path) -> cv2.VideoCapture:
        """Get or create video capture for a video."""
        if video_id not in self._video_caps:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            self._video_caps[video_id] = cap
        return self._video_caps[video_id]

    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """Apply data augmentation to frames.

        Args:
            frames: Array of shape (16, H, W, 3)

        Returns:
            Augmented frames
        """
        # Random horizontal flip (50% chance)
        if np.random.random() < 0.5:
            frames = frames[:, :, ::-1, :]

        # Random brightness adjustment (-10% to +10%)
        brightness = np.random.uniform(0.9, 1.1)
        frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)

        # Random contrast adjustment
        contrast = np.random.uniform(0.9, 1.1)
        mean = frames.mean()
        frames = np.clip((frames - mean) * contrast + mean, 0, 255).astype(np.uint8)

        return frames

    def close(self) -> None:
        """Release video captures."""
        for cap in self._video_caps.values():
            cap.release()
        self._video_caps.clear()

    def __del__(self) -> None:
        """Clean up on deletion."""
        self.close()


def create_data_collator() -> Any:
    """Create a data collator for the trainer."""

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate batch of samples."""
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    return collate_fn
