"""VNL-STES indoor volleyball dataset loader for pretraining.

VNL-STES provides 1,028 rally videos with 6,137 single-frame event
annotations (serve, receive, set, spike, block, score) at 25 FPS.

Dataset structure:
    data/external/vnl_stes/
    ├── class.txt           # One class name per line
    ├── train.json          # [{video, num_frames, fps, events: [{frame, label, x, y}]}]
    ├── val.json
    ├── test.json
    └── frames/
        ├── rally_0001/     # Pre-extracted JPGs at 224px height
        │   ├── 000000.jpg
        │   └── ...
        └── ...

Reference: https://github.com/hoangqnguyen/spot
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console

from rallycut.spotting.config import ACTION_TO_IDX

console = Console()

# VNL-STES data directory
VNL_STES_DIR = Path(__file__).resolve().parents[3] / "data" / "external" / "vnl_stes"

# Class mapping from VNL-STES to our beach volleyball labels
VNL_CLASS_MAP: dict[str, str] = {
    "serve": "serve",
    "receive": "receive",
    "set": "set",
    "spike": "attack",    # VNL-STES "spike" → our "attack"
    "block": "block",
    "score": "",           # VNL-STES "score" → ignore (no beach equivalent)
}


@dataclass
class VNLEvent:
    """A single event annotation from VNL-STES."""

    frame: int
    label: str        # Original VNL-STES label
    mapped_label: str  # Mapped to our label space
    x: float          # Normalized spatial x (0-1)
    y: float          # Normalized spatial y (0-1)


@dataclass
class VNLRally:
    """A rally from the VNL-STES dataset."""

    video_id: str
    frame_dir: Path
    num_frames: int
    fps: float
    events: list[VNLEvent] = field(default_factory=list)
    width: int = 1920
    height: int = 1080


def _load_split_json(split_path: Path) -> list[dict]:
    """Load a VNL-STES split JSON file."""
    if not split_path.exists():
        console.print(f"  [red]VNL-STES split file not found: {split_path}[/]")
        return []
    with open(split_path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def load_vnl_rallies(
    splits: list[str] | None = None,
    data_dir: Path | None = None,
) -> list[VNLRally]:
    """Load VNL-STES rally annotations.

    Args:
        splits: Which splits to load ("train", "val", "test"). Default: all.
        data_dir: Override VNL-STES data directory.

    Returns:
        List of VNLRally objects with mapped labels.
    """
    base = data_dir or VNL_STES_DIR
    if splits is None:
        splits = ["train", "val", "test"]

    frames_dir = base / "frames"
    if not frames_dir.exists():
        console.print(f"  [red]VNL-STES frames directory not found: {frames_dir}[/]")
        console.print("  Download from https://hoangqnguyen.github.io/stes/")
        return []

    rallies: list[VNLRally] = []
    skipped_no_frames = 0
    skipped_events = 0

    for split in splits:
        entries = _load_split_json(base / f"{split}.json")

        for entry in entries:
            video_id = entry["video"]
            rally_dir = frames_dir / video_id

            if not rally_dir.exists():
                skipped_no_frames += 1
                continue

            events: list[VNLEvent] = []
            for ev in entry.get("events", []):
                orig_label = ev["label"].lower()
                mapped = VNL_CLASS_MAP.get(orig_label, "")

                if not mapped:
                    skipped_events += 1
                    continue

                events.append(VNLEvent(
                    frame=ev["frame"],
                    label=orig_label,
                    mapped_label=mapped,
                    x=ev.get("x", 0.5),
                    y=ev.get("y", 0.5),
                ))

            rallies.append(VNLRally(
                video_id=video_id,
                frame_dir=rally_dir,
                num_frames=entry.get("num_frames", 0),
                fps=entry.get("fps", 25.0),
                events=events,
                width=entry.get("width", 1920),
                height=entry.get("height", 1080),
            ))

    console.print(
        f"  VNL-STES: {len(rallies)} rallies loaded from {splits}, "
        f"{sum(len(r.events) for r in rallies)} events"
    )
    if skipped_no_frames:
        console.print(f"  [yellow]Skipped {skipped_no_frames} rallies (no frames)[/]")
    if skipped_events:
        console.print(f"  [dim]Skipped {skipped_events} 'score' events (no mapping)[/]")

    return rallies


def vnl_to_rally_infos(rallies: list[VNLRally]) -> list:
    """Convert VNL-STES rallies to RallyInfo format for training.

    Creates per-frame label arrays using our class indices. The "dig" class
    (index 5) never appears in VNL-STES — it will be learned from beach data.
    """
    from rallycut.spotting.data.beach import RallyInfo

    rally_infos: list[RallyInfo] = []
    for rally in rallies:
        if rally.num_frames <= 0:
            continue

        labels = np.zeros(rally.num_frames, dtype=np.int64)
        offsets = np.zeros(rally.num_frames, dtype=np.float32)

        event_frames = []
        for ev in rally.events:
            cls = ACTION_TO_IDX.get(ev.mapped_label, 0)
            if cls > 0 and 0 <= ev.frame < rally.num_frames:
                labels[ev.frame] = cls
                event_frames.append(ev.frame)

        # Compute offset to nearest event (vectorized)
        if event_frames:
            event_arr = np.array(event_frames)
            all_frames = np.arange(rally.num_frames)
            dists = event_arr[:, None] - all_frames[None, :]
            nearest = np.argmin(np.abs(dists), axis=0)
            offsets = dists[nearest, np.arange(rally.num_frames)].astype(np.float32)

        rally_infos.append(RallyInfo(
            rally_id=f"vnl_{rally.video_id}",
            video_id=f"vnl_{rally.video_id}",
            frame_dir=rally.frame_dir,
            frame_count=rally.num_frames,
            fps=rally.fps,
            labels=labels,
            offsets=offsets,
        ))

    return rally_infos
