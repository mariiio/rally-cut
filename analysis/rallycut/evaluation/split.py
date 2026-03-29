"""Deterministic hash-based train/held-out split for evaluation.

Splits videos (not individual rallies) into train/held-out partitions
using a stable SHA-256 hash of the video ID. All rallies from one video
always land in the same partition to prevent data leakage.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol


class _HasVideoId(Protocol):
    video_id: str


SplitName = Literal["train", "held_out", "all"]


def _get_video_id(item: Any) -> str:
    """Extract video_id from an object or dict."""
    if isinstance(item, dict):
        return str(item["video_id"])
    return str(item.video_id)


def video_split(video_id: str, *, salt: str = "rallycut-v1") -> Literal["train", "held_out"]:
    """Return the split partition for a video ID.

    Uses SHA-256 hash modulo 4: bucket 0 → held_out (~25%), buckets 1-3 → train (~75%).
    Deterministic and stable across platforms/Python versions.
    """
    digest = hashlib.sha256(f"{salt}:{video_id}".encode()).digest()
    value = int.from_bytes(digest[:8], byteorder="big")
    return "held_out" if value % 4 == 0 else "train"


def split_rallies(
    rallies: Sequence[_HasVideoId],
    split: SplitName,
) -> list[Any]:
    """Filter rallies by their video's split partition."""
    if split == "all":
        return list(rallies)
    return [r for r in rallies if video_split(_get_video_id(r)) == split]


def add_split_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--split {train,held_out,all}`` argument to an argparse parser."""
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "held_out", "all"],
        default="all",
        help="Filter to train (~75%%) or held_out (~25%%) partition, or all (default)",
    )


def apply_split(
    rallies: Sequence[_HasVideoId],
    args: argparse.Namespace,
) -> list[Any]:
    """Apply the --split filter and print a summary."""
    split: SplitName = getattr(args, "split", "all")
    filtered = split_rallies(rallies, split)

    if split != "all":
        video_ids = {_get_video_id(r) for r in rallies}
        train_vids = {v for v in video_ids if video_split(v) == "train"}
        held_vids = video_ids - train_vids
        train_rallies = [r for r in rallies if video_split(_get_video_id(r)) == "train"]
        held_rallies = [r for r in rallies if video_split(_get_video_id(r)) == "held_out"]
        print(
            f"  Split: {len(train_vids)} train videos ({len(train_rallies)} rallies), "
            f"{len(held_vids)} held-out videos ({len(held_rallies)} rallies)"
        )
        print(f"  Using: {split} → {len(filtered)} rallies")

    return filtered
