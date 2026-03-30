"""Beach volleyball data loading for E2E-Spot training.

Loads ground truth from the database, resolves video paths, and extracts
frames for training. Uses the existing train/held-out split infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress

from rallycut.spotting.config import ACTION_TO_IDX

console = Console()

FRAME_DIR = Path(__file__).resolve().parents[3] / "data" / "spotting_frames"


@dataclass
class RallyInfo:
    """Metadata for a rally used in clip sampling."""

    rally_id: str
    video_id: str
    frame_dir: Path
    frame_count: int
    fps: float
    labels: np.ndarray  # (frame_count,) int64, 0=background
    offsets: np.ndarray  # (frame_count,) float32, offset to nearest event
    gt_labels: list[Any] = field(default_factory=list)
    start_ms: int = 0


def build_frame_labels(
    gt_labels: list[Any],
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert GT labels to per-frame label and offset arrays.

    Args:
        gt_labels: List of ground truth contact labels.
        frame_count: Total frames in the rally.

    Returns:
        labels: (frame_count,) int64 array, 0=background, 1-6=action class.
        offsets: (frame_count,) float32 array, signed distance to nearest event.
    """
    labels = np.zeros(frame_count, dtype=np.int64)
    offsets = np.zeros(frame_count, dtype=np.float32)

    event_frames = []
    for gt in gt_labels:
        cls = ACTION_TO_IDX.get(gt.action, 0)
        if cls > 0 and 0 <= gt.frame < frame_count:
            labels[gt.frame] = cls
            event_frames.append(gt.frame)

    # Compute offset to nearest event for all frames (vectorized)
    if event_frames:
        event_arr = np.array(event_frames)
        all_frames = np.arange(frame_count)
        dists = event_arr[:, None] - all_frames[None, :]  # (E, F)
        nearest = np.argmin(np.abs(dists), axis=0)         # (F,)
        offsets = dists[nearest, np.arange(frame_count)].astype(np.float32)

    return labels, offsets


def extract_rally_frames(
    video_id: str,
    rally_id: str,
    start_ms: int,
    frame_count: int,
    target_height: int = 224,
) -> Path | None:
    """Extract frames for a rally from its video to disk.

    Args:
        video_id: Database video ID.
        rally_id: Rally ID for directory naming.
        start_ms: Rally start time in milliseconds.
        frame_count: Number of frames to extract.
        target_height: Resize height (width scales proportionally).

    Returns:
        Path to frame directory, or None if extraction failed.
    """
    from rallycut.evaluation.tracking.db import get_video_path

    rally_dir = FRAME_DIR / rally_id
    # Check if already extracted
    if rally_dir.exists() and len(list(rally_dir.glob("*.jpg"))) >= frame_count * 0.9:
        return rally_dir

    video_path = get_video_path(video_id)
    if video_path is None:
        console.print(f"  [red]Cannot resolve video {video_id}[/]")
        return None

    rally_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"  [red]Cannot open video {video_path}[/]")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_ms / 1000.0 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted = 0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to target height
        h, w = frame.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        frame = cv2.resize(frame, (new_w, target_height))

        cv2.imwrite(str(rally_dir / f"{i:06d}.jpg"), frame)
        extracted += 1

    cap.release()

    if extracted < frame_count * 0.5:
        console.print(f"  [yellow]Only extracted {extracted}/{frame_count} frames for {rally_id}[/]")

    return rally_dir


def load_beach_rallies(
    split: str = "all",
    extract_frames: bool = False,
    target_height: int = 224,
) -> list[RallyInfo]:
    """Load beach volleyball rallies with action ground truth.

    Args:
        split: "train", "held_out", or "all".
        extract_frames: If True, extract frames to disk for rallies that need it.
        target_height: Frame resize height.

    Returns:
        List of RallyInfo objects ready for clip sampling.
    """
    # Lazy imports: these require DB access and scripts/ on sys.path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from eval_action_detection import load_rallies_with_action_gt  # type: ignore[import-not-found]

    from rallycut.evaluation.split import video_split  # noqa: E402

    rallies = load_rallies_with_action_gt()

    if split != "all":
        rallies = [r for r in rallies if video_split(r.video_id) == split]

    console.print(f"  Loaded {len(rallies)} rallies ({split} split)")

    rally_infos: list[RallyInfo] = []
    skipped = 0

    if extract_frames:
        with Progress(console=console) as progress:
            task = progress.add_task("Extracting frames", total=len(rallies))
            for rally in rallies:
                frame_dir = extract_rally_frames(
                    rally.video_id,
                    rally.rally_id,
                    rally.start_ms,
                    rally.frame_count,
                    target_height,
                )
                progress.update(task, advance=1)

                if frame_dir is None:
                    skipped += 1
                    continue

                labels, offsets = build_frame_labels(rally.gt_labels, rally.frame_count)
                rally_infos.append(RallyInfo(
                    rally_id=rally.rally_id,
                    video_id=rally.video_id,
                    frame_dir=frame_dir,
                    frame_count=rally.frame_count,
                    fps=rally.fps,
                    labels=labels,
                    offsets=offsets,
                    gt_labels=rally.gt_labels,
                    start_ms=rally.start_ms,
                ))
    else:
        for rally in rallies:
            rally_dir = FRAME_DIR / rally.rally_id
            if not rally_dir.exists():
                skipped += 1
                continue

            labels, offsets = build_frame_labels(rally.gt_labels, rally.frame_count)
            rally_infos.append(RallyInfo(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                frame_dir=rally_dir,
                frame_count=rally.frame_count,
                fps=rally.fps,
                labels=labels,
                offsets=offsets,
                gt_labels=rally.gt_labels,
                start_ms=rally.start_ms,
            ))

    if skipped:
        console.print(f"  [yellow]Skipped {skipped} rallies (no frames)[/]")

    return rally_infos
