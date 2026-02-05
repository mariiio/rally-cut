"""Generate training samples from ground truth rallies."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rallycut.core.proxy import ProxyGenerator
from rallycut.training.config import LABEL_TO_ID, TrainingConfig

if TYPE_CHECKING:
    from rallycut.evaluation.ground_truth import EvaluationVideo, GroundTruthRally


@dataclass
class TrainingSample:
    """A training sample: 16-frame window with label."""

    video_id: str
    start_frame: int
    label: int  # 0=NO_PLAY, 1=PLAY, 2=SERVICE
    label_name: str

    @property
    def end_frame(self) -> int:
        """End frame (exclusive)."""
        return self.start_frame + 16


def generate_training_samples(
    videos: list[EvaluationVideo],
    config: TrainingConfig | None = None,
    seed: int | None = None,
) -> list[TrainingSample]:
    """Generate balanced training samples from ground truth rallies.

    Sampling strategy:
    - PLAY: Windows from middle of rallies (avoiding boundaries)
    - SERVICE: Windows from first 1-2s of rallies
    - NO_PLAY: Windows from gaps between rallies

    Args:
        videos: List of videos with ground truth rallies
        config: Training configuration
        seed: Random seed for reproducibility

    Returns:
        List of TrainingSample objects
    """
    if config is None:
        config = TrainingConfig()

    if seed is not None:
        random.seed(seed)

    samples: list[TrainingSample] = []

    for video in videos:
        if not video.ground_truth_rallies:
            continue

        # Use proxy FPS for sampling (training uses 480p proxy videos)
        # High-FPS videos (>40fps) are normalized to 30fps in proxies (see proxy.py)
        # Low-FPS videos keep their original FPS
        original_fps = video.fps if video.fps else config.fps
        fps = config.fps if original_fps > ProxyGenerator.FPS_NORMALIZE_THRESHOLD else original_fps
        frame_count = int((video.duration_ms or 0) / 1000 * fps) if video.duration_ms else 0

        # Sort rallies by start time
        rallies = sorted(video.ground_truth_rallies, key=lambda r: r.start_ms)

        # Generate PLAY and SERVICE samples from rallies
        for rally in rallies:
            rally_samples = _sample_from_rally(
                video_id=video.id,
                rally=rally,
                fps=fps,
                config=config,
            )
            samples.extend(rally_samples)

        # Generate NO_PLAY samples from gaps
        gap_samples = _sample_from_gaps(
            video_id=video.id,
            rallies=rallies,
            fps=fps,
            frame_count=frame_count,
            config=config,
        )
        samples.extend(gap_samples)

    # Shuffle samples
    random.shuffle(samples)

    return samples


def _sample_from_rally(
    video_id: str,
    rally: GroundTruthRally,
    fps: float,
    config: TrainingConfig,
) -> list[TrainingSample]:
    """Extract PLAY and SERVICE samples from a rally."""
    samples: list[TrainingSample] = []

    start_frame = int(rally.start_ms / 1000 * fps)
    end_frame = int(rally.end_ms / 1000 * fps)
    rally_frames = end_frame - start_frame

    # Skip very short rallies
    if rally_frames < config.num_frames * 2:
        return samples

    # SERVICE samples: first 1.5 seconds of rally (serves can take 2-3s)
    service_end_frame = start_frame + int(1.5 * fps)  # First 1.5 seconds
    service_range = service_end_frame - start_frame - config.num_frames

    if service_range > 0:
        for _ in range(config.samples_per_rally_service):
            sample_start = start_frame + random.randint(0, service_range)
            samples.append(
                TrainingSample(
                    video_id=video_id,
                    start_frame=sample_start,
                    label=LABEL_TO_ID["SERVICE"],
                    label_name="SERVICE",
                )
            )

    # PLAY samples: middle of rally (excluding first/last 0.5s)
    boundary_frames = int(0.5 * fps)  # 0.5 second boundary
    play_start = start_frame + boundary_frames + int(1.0 * fps)  # After service
    play_end = end_frame - boundary_frames
    play_range = play_end - play_start - config.num_frames

    if play_range > 0:
        for _ in range(config.samples_per_rally_play):
            sample_start = play_start + random.randint(0, play_range)
            samples.append(
                TrainingSample(
                    video_id=video_id,
                    start_frame=sample_start,
                    label=LABEL_TO_ID["PLAY"],
                    label_name="PLAY",
                )
            )

    return samples


def _sample_from_gaps(
    video_id: str,
    rallies: list[GroundTruthRally],
    fps: float,
    frame_count: int,
    config: TrainingConfig,
) -> list[TrainingSample]:
    """Extract NO_PLAY samples from gaps between rallies."""
    samples: list[TrainingSample] = []

    min_gap_frames = int(config.min_gap_duration_seconds * fps)

    # Find gaps between rallies
    gaps: list[tuple[int, int]] = []

    # Gap before first rally
    if rallies:
        first_start = int(rallies[0].start_ms / 1000 * fps)
        if first_start > min_gap_frames + config.num_frames:
            gaps.append((0, first_start))

    # Gaps between rallies
    for i in range(len(rallies) - 1):
        gap_start = int(rallies[i].end_ms / 1000 * fps)
        gap_end = int(rallies[i + 1].start_ms / 1000 * fps)
        if gap_end - gap_start > min_gap_frames + config.num_frames:
            gaps.append((gap_start, gap_end))

    # Gap after last rally
    if rallies and frame_count > 0:
        last_end = int(rallies[-1].end_ms / 1000 * fps)
        if frame_count - last_end > min_gap_frames + config.num_frames:
            gaps.append((last_end, frame_count))

    # Sample from gaps
    for gap_start, gap_end in gaps:
        gap_range = gap_end - gap_start - config.num_frames
        if gap_range <= 0:
            continue

        # Take multiple samples from each gap
        num_samples = min(config.samples_per_gap_no_play, max(1, gap_range // (config.num_frames)))
        for _ in range(num_samples):
            sample_start = gap_start + random.randint(0, gap_range)
            samples.append(
                TrainingSample(
                    video_id=video_id,
                    start_frame=sample_start,
                    label=LABEL_TO_ID["NO_PLAY"],
                    label_name="NO_PLAY",
                )
            )

    return samples


def get_sample_statistics(samples: list[TrainingSample]) -> dict[str, int]:
    """Get statistics about the generated samples."""
    stats: dict[str, int] = {"total": len(samples)}

    for label_name in ["NO_PLAY", "PLAY", "SERVICE"]:
        count = sum(1 for s in samples if s.label_name == label_name)
        stats[label_name] = count

    return stats


# --- Sequence Labeling for Temporal Models ---


def generate_sequence_labels(
    rallies: list[GroundTruthRally],
    video_duration_ms: float,
    fps: float,
    stride: int = 48,
    window_size: int = 16,
    labeling_mode: str = "center",
    overlap_threshold: float = 0.5,
) -> list[int]:
    """Generate per-window RALLY/NO_RALLY labels for temporal modeling.

    This creates binary labels (0=NO_RALLY, 1=RALLY) for each window position
    in a video, suitable for training sequence labeling models.

    Args:
        rallies: List of ground truth rallies.
        video_duration_ms: Total video duration in milliseconds.
        fps: Video frames per second.
        stride: Frame stride between windows.
        window_size: Frames per window (default 16).
        labeling_mode: How to determine window label:
            - "center": Label based on whether window center is in a rally
            - "overlap": Label as RALLY if overlap ratio exceeds threshold
        overlap_threshold: Overlap ratio threshold for "overlap" mode.

    Returns:
        List of labels (0 or 1) for each window position.
    """

    total_frames = int(video_duration_ms / 1000 * fps)
    num_windows = max(1, (total_frames - window_size) // stride + 1)

    labels = []
    half_window = window_size // 2
    window_duration_ms = (window_size / fps) * 1000

    for i in range(num_windows):
        start_frame = i * stride
        end_frame = start_frame + window_size
        center_frame = start_frame + half_window

        # Convert frames to milliseconds
        window_start_ms = (start_frame / fps) * 1000
        window_end_ms = (end_frame / fps) * 1000
        center_ms = (center_frame / fps) * 1000

        is_rally = 0

        if labeling_mode == "center":
            # Check if center point is within any rally
            for rally in rallies:
                if rally.start_ms <= center_ms <= rally.end_ms:
                    is_rally = 1
                    break
        elif labeling_mode == "overlap":
            # Check overlap ratio with any rally
            for rally in rallies:
                overlap_start = max(window_start_ms, rally.start_ms)
                overlap_end = min(window_end_ms, rally.end_ms)
                overlap_duration = max(0, overlap_end - overlap_start)
                overlap_ratio = overlap_duration / window_duration_ms
                if overlap_ratio >= overlap_threshold:
                    is_rally = 1
                    break
        else:
            raise ValueError(f"Unknown labeling_mode: {labeling_mode}")

        labels.append(is_rally)

    return labels


def compute_class_weights(labels: list[int]) -> tuple[float, float]:
    """Compute inverse frequency class weights for balanced training.

    Args:
        labels: List of binary labels (0 or 1).

    Returns:
        Tuple of (weight_class_0, weight_class_1).
    """
    counts = np.bincount(labels, minlength=2)
    total = len(labels)

    # Inverse frequency weighting
    weights = total / (2.0 * counts.clip(min=1))

    return float(weights[0]), float(weights[1])


def augment_sequence(
    features: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    shift_range: int = 2,
    keep_ratio_range: tuple[float, float] = (0.8, 1.0),
    random_state: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply temporal data augmentation to a sequence.

    Augmentation operations:
    1. Random temporal shift (±shift_range windows)
    2. Random subsequence selection (keep_ratio_range of original)

    Args:
        features: Feature array of shape (seq_len, feature_dim).
        labels: Label array of shape (seq_len,).
        shift_range: Maximum shift in windows (default ±2).
        keep_ratio_range: Range of sequence length to keep (default 80-100%).
        random_state: Optional random state for reproducibility.

    Returns:
        Tuple of (augmented_features, augmented_labels).
    """
    rng = random_state or random
    features_np = np.array(features)
    labels_np = np.array(labels)

    seq_len = len(features_np)
    if seq_len < 10:  # Too short to augment meaningfully
        return features_np, labels_np

    # 1. Random temporal shift (50% probability)
    if rng.random() < 0.5:
        shift = rng.randint(-shift_range, shift_range)
        if shift != 0:
            features_np = np.roll(features_np, shift, axis=0)
            labels_np = np.roll(labels_np, shift, axis=0)

    # 2. Random subsequence (50% probability)
    if rng.random() < 0.5:
        keep_ratio = rng.uniform(*keep_ratio_range)
        keep_len = max(10, int(seq_len * keep_ratio))
        start_idx = rng.randint(0, seq_len - keep_len)
        features_np = features_np[start_idx : start_idx + keep_len]
        labels_np = labels_np[start_idx : start_idx + keep_len]

    return features_np, labels_np
