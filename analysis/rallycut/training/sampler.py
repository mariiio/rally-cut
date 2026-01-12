"""Generate training samples from ground truth rallies."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

        # Get video FPS (default to 30 if not available)
        fps = config.fps
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

    # SERVICE samples: first 1 second of rally
    service_end_frame = start_frame + int(1.0 * fps)  # First 1 second
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
