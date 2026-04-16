"""Cache raw BoT-SORT output (positions + appearance data) for fast retrack eval.

Unlike RawPositionCache (grid search), this cache includes ColorHistogramStore
and AppearanceDescriptorStore so the full post-processing pipeline (Steps 0-5)
can be replayed without re-running YOLO+BoT-SORT detection.

All cached frame numbers are rally-relative (0-based). The caller normalizes
absolute frame numbers before storing, so replay passes start_frame=0.

Cache format per rally:
  {hash_prefix}_meta.json   — positions, ball, video metadata, config hash
  {hash_prefix}_appearance.npz — color histograms + appearance descriptors
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rallycut.tracking.appearance_descriptor import (
    AppearanceDescriptorStore,
    MultiRegionDescriptor,
)
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.color_repair import ColorHistogramStore, LearnedEmbeddingStore
from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


@dataclass
class CachedRetrackData:
    """Cached raw BoT-SORT output for a rally (before post-processing)."""

    rally_id: str
    video_id: str
    config_hash: str
    positions: list[PlayerPosition]
    ball_positions: list[BallPosition]
    video_fps: float
    video_width: int
    video_height: int
    frame_count: int
    # Session 4 — head SHA pinned to detect checkpoint drift. Empty string
    # when learned ReID was disabled at extraction time.
    head_sha: str = ""


def _serialize_positions(positions: list[PlayerPosition]) -> list[dict[str, Any]]:
    return [
        {
            "fn": p.frame_number,
            "tid": p.track_id,
            "x": p.x,
            "y": p.y,
            "w": p.width,
            "h": p.height,
            "c": p.confidence,
        }
        for p in positions
    ]


def _deserialize_positions(data: list[dict[str, Any]]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=p["fn"],
            track_id=p["tid"],
            x=p["x"],
            y=p["y"],
            width=p["w"],
            height=p["h"],
            confidence=p["c"],
        )
        for p in data
    ]


def _serialize_ball_positions(positions: list[BallPosition]) -> list[dict[str, Any]]:
    return [
        {
            "fn": bp.frame_number,
            "x": bp.x,
            "y": bp.y,
            "c": bp.confidence,
        }
        for bp in positions
    ]


def _deserialize_ball_positions(data: list[dict[str, Any]]) -> list[BallPosition]:
    return [
        BallPosition(
            frame_number=bp["fn"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp["c"],
        )
        for bp in data
    ]


def _serialize_color_store(store: ColorHistogramStore) -> dict[str, np.ndarray]:
    """Convert ColorHistogramStore to named arrays for npz storage."""
    arrays: dict[str, np.ndarray] = {}
    for (tid, fn), hist in store._histograms.items():
        arrays[f"c_{tid}_{fn}"] = hist
    return arrays


def _deserialize_color_store(arrays: dict[str, np.ndarray]) -> ColorHistogramStore:
    """Reconstruct ColorHistogramStore from npz arrays."""
    store = ColorHistogramStore()
    for key, hist in arrays.items():
        if not key.startswith("c_"):
            continue
        parts = key.split("_")
        tid = int(parts[1])
        fn = int(parts[2])
        store.add(tid, fn, hist)
    return store


def _serialize_appearance_store(
    store: AppearanceDescriptorStore,
) -> dict[str, np.ndarray]:
    """Convert AppearanceDescriptorStore to named arrays for npz storage."""
    arrays: dict[str, np.ndarray] = {}
    for (tid, fn), desc in store._descriptors.items():
        if desc.head is not None:
            arrays[f"ah_{tid}_{fn}"] = desc.head
        if desc.upper is not None:
            arrays[f"au_{tid}_{fn}"] = desc.upper
        if desc.shorts is not None:
            arrays[f"as_{tid}_{fn}"] = desc.shorts
    return arrays


def _deserialize_appearance_store(
    arrays: dict[str, np.ndarray],
) -> AppearanceDescriptorStore:
    """Reconstruct AppearanceDescriptorStore from npz arrays."""
    store = AppearanceDescriptorStore()
    # Collect all (tid, fn) pairs that have appearance data
    descriptors: dict[tuple[int, int], dict[str, np.ndarray | None]] = {}
    for key, arr in arrays.items():
        if not key.startswith("a"):
            continue
        parts = key.split("_")
        if len(parts) != 3:
            continue
        prefix = parts[0]  # "ah", "au", or "as"
        tid = int(parts[1])
        fn = int(parts[2])
        k = (tid, fn)
        if k not in descriptors:
            descriptors[k] = {"head": None, "upper": None, "shorts": None}
        if prefix == "ah":
            descriptors[k]["head"] = arr
        elif prefix == "au":
            descriptors[k]["upper"] = arr
        elif prefix == "as":
            descriptors[k]["shorts"] = arr

    for (tid, fn), parts_dict in descriptors.items():
        desc = MultiRegionDescriptor(
            head=parts_dict["head"],
            upper=parts_dict["upper"],
            shorts=parts_dict["shorts"],
        )
        store.add(tid, fn, desc)
    return store


class RetrackCache:
    """Cache for raw tracking data including appearance stores.

    Stores everything needed to replay the full post-processing pipeline
    without re-running YOLO+BoT-SORT:
    - Raw player positions (before Step 0)
    - Ball positions
    - Color histograms (ColorHistogramStore)
    - Appearance descriptors (AppearanceDescriptorStore)
    - Video metadata and config hash

    Cache location: ~/.cache/rallycut/retrack_cache/
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
    ) -> None:
        self.cache_dir = cache_dir or (
            Path.home() / ".cache" / "rallycut" / "retrack_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_prefix(self, rally_id: str, config_hash: str) -> str:
        """Deterministic file prefix from rally_id + config_hash."""
        combined = f"{rally_id}:{config_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _meta_path(self, prefix: str) -> Path:
        return self.cache_dir / f"{prefix}_meta.json"

    def _appearance_path(self, prefix: str) -> Path:
        return self.cache_dir / f"{prefix}_appearance.npz"

    def has(self, rally_id: str, config_hash: str) -> bool:
        """Check if a rally is cached with the given config."""
        prefix = self._key_prefix(rally_id, config_hash)
        return (
            self._meta_path(prefix).exists()
            and self._appearance_path(prefix).exists()
        )

    def get(
        self,
        rally_id: str,
        config_hash: str,
    ) -> (
        tuple[
            CachedRetrackData,
            ColorHistogramStore,
            AppearanceDescriptorStore,
            LearnedEmbeddingStore | None,
        ]
        | None
    ):
        """Load cached rally data and appearance stores.

        Returns:
            (CachedRetrackData, ColorHistogramStore, AppearanceDescriptorStore,
             LearnedEmbeddingStore | None) or None if not cached or corrupt.
            The fourth element is None when the cache was written without
            learned-ReID embeddings (i.e. ``WEIGHT_LEARNED_REID`` was 0 at
            extraction time).
        """
        prefix = self._key_prefix(rally_id, config_hash)
        meta_path = self._meta_path(prefix)
        appearance_path = self._appearance_path(prefix)

        if not meta_path.exists() or not appearance_path.exists():
            return None

        try:
            # Load metadata + positions
            with open(meta_path) as f:
                meta = json.load(f)

            # Verify rally_id matches (collision protection)
            if meta["rally_id"] != rally_id:
                logger.warning(
                    f"Cache collision: expected {rally_id}, got {meta['rally_id']}"
                )
                return None

            # Verify config hash matches
            if meta["config_hash"] != config_hash:
                logger.warning(
                    f"Config hash mismatch for {rally_id[:8]}: "
                    f"expected {config_hash[:8]}, got {meta['config_hash'][:8]}"
                )
                return None

            data = CachedRetrackData(
                rally_id=meta["rally_id"],
                video_id=meta["video_id"],
                config_hash=meta["config_hash"],
                positions=_deserialize_positions(meta["positions"]),
                ball_positions=_deserialize_ball_positions(
                    meta.get("ball_positions", [])
                ),
                video_fps=meta["video_fps"],
                video_width=meta["video_width"],
                video_height=meta["video_height"],
                frame_count=meta["frame_count"],
                head_sha=meta.get("head_sha", ""),
            )

            # Load appearance arrays
            with np.load(appearance_path) as npz:
                all_arrays = dict(npz)

            color_store = _deserialize_color_store(all_arrays)
            appearance_store = _deserialize_appearance_store(all_arrays)
            learned_arrays = {
                k: v for k, v in all_arrays.items() if k.startswith("l_")
            }
            learned_store = (
                LearnedEmbeddingStore.deserialize(learned_arrays)
                if learned_arrays
                else None
            )

            return data, color_store, appearance_store, learned_store

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to load retrack cache for {rally_id[:8]}: {e}")
            return None

    def put(
        self,
        data: CachedRetrackData,
        color_store: ColorHistogramStore | None,
        appearance_store: AppearanceDescriptorStore | None,
        learned_store: LearnedEmbeddingStore | None = None,
    ) -> None:
        """Store rally data and appearance stores in cache."""
        prefix = self._key_prefix(data.rally_id, data.config_hash)

        try:
            # Write metadata + positions
            meta = {
                "rally_id": data.rally_id,
                "video_id": data.video_id,
                "config_hash": data.config_hash,
                "positions": _serialize_positions(data.positions),
                "ball_positions": _serialize_ball_positions(data.ball_positions),
                "video_fps": data.video_fps,
                "video_width": data.video_width,
                "video_height": data.video_height,
                "frame_count": data.frame_count,
                "head_sha": data.head_sha,
            }
            with open(self._meta_path(prefix), "w") as f:
                json.dump(meta, f)

            # Write appearance arrays
            arrays: dict[str, np.ndarray] = {}
            if color_store is not None:
                arrays.update(_serialize_color_store(color_store))
            if appearance_store is not None:
                arrays.update(_serialize_appearance_store(appearance_store))
            if learned_store is not None and learned_store.has_data():
                arrays.update(learned_store.serialize())

            np.savez_compressed(self._appearance_path(prefix), **arrays)  # type: ignore[arg-type]

            logger.debug(
                f"Cached retrack data for {data.rally_id[:8]}... "
                f"({len(arrays)} arrays)"
            )
        except OSError as e:
            logger.warning(f"Failed to cache retrack data for {data.rally_id[:8]}: {e}")

    def clear(self) -> int:
        """Clear all cached data. Returns number of rally entries deleted."""
        count = 0
        for meta_file in self.cache_dir.glob("*_meta.json"):
            prefix = meta_file.stem.replace("_meta", "")
            meta_file.unlink(missing_ok=True)
            appearance_file = self._appearance_path(prefix)
            appearance_file.unlink(missing_ok=True)
            count += 1
        logger.info(f"Cleared {count} retrack cache entries")
        return count

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        meta_files = list(self.cache_dir.glob("*_meta.json"))
        total_size = 0
        for mf in meta_files:
            total_size += mf.stat().st_size
            prefix = mf.stem.replace("_meta", "")
            npz = self._appearance_path(prefix)
            if npz.exists():
                total_size += npz.stat().st_size

        return {
            "cache_dir": str(self.cache_dir),
            "count": len(meta_files),
            "total_size_mb": total_size / (1024 * 1024),
        }
