"""Ensemble ball tracker combining WASB HRNet + VballNet.

WASB provides higher precision (6-22px error) but lower recall.
VballNet provides higher recall but lower precision (40-92px error).
The ensemble uses WASB as primary and VballNet as fallback, achieving
79.4% match rate (+37.7pp over VballNet, +12.0pp over WASB alone).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rallycut.tracking.ball_filter import BallFilterConfig
    from rallycut.tracking.ball_tracker import BallTrackingResult

logger = logging.getLogger(__name__)


class EnsembleBallTracker:
    """Ball tracker that ensembles WASB HRNet + VballNet.

    Strategy: WASB-primary, VballNet-fallback. For each frame:
    - If WASB has a detection (confidence > 0), use it
    - Otherwise fall back to VballNet detection

    This achieves 79.4% match rate on beach volleyball GT (vs 41.7% VballNet,
    67.5% WASB alone) with 92.3% detection rate.
    """

    def __init__(
        self,
        vballnet_model: str = "v2",
        wasb_weights: Path | str | None = None,
        wasb_device: str | None = None,
        wasb_threshold: float = 0.3,
    ):
        self._vballnet_model = vballnet_model
        self._wasb_weights = wasb_weights
        self._wasb_device = wasb_device
        self._wasb_threshold = wasb_threshold

    def track_video(
        self,
        video_path: Path | str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
        filter_config: BallFilterConfig | None = None,
        enable_filtering: bool = True,
        preserve_raw: bool = False,
    ) -> BallTrackingResult:
        """Track ball using WASB + VballNet ensemble.

        Runs both trackers, merges per-frame (WASB priority), then
        optionally applies filtering.
        """
        from rallycut.tracking.ball_filter import BallTemporalFilter
        from rallycut.tracking.ball_tracker import BallPosition, BallTracker, BallTrackingResult
        from rallycut.tracking.wasb_model import WASBBallTracker

        start_time = time.time()

        # Run both trackers without filtering (we'll filter the merged result)
        logger.info("Ensemble: running WASB inference...")
        wasb_tracker = WASBBallTracker(
            weights_path=self._wasb_weights,
            device=self._wasb_device,
            threshold=self._wasb_threshold,
        )

        # Wrap progress callback to show first half for WASB, second half for VballNet
        def _wasb_cb(p: float) -> None:
            if progress_callback:
                progress_callback(p * 0.5)

        def _vnet_cb(p: float) -> None:
            if progress_callback:
                progress_callback(0.5 + p * 0.5)

        wasb_result = wasb_tracker.track_video(
            video_path, start_ms, end_ms,
            progress_callback=_wasb_cb if progress_callback else None,
            enable_filtering=False,
        )

        logger.info("Ensemble: running VballNet inference...")
        vnet_tracker = BallTracker(model=self._vballnet_model)
        vnet_result = vnet_tracker.track_video(
            video_path, start_ms, end_ms,
            progress_callback=_vnet_cb if progress_callback else None,
            enable_filtering=False,
        )

        # Pre-filter VballNet: remove stationary false positives before merging.
        # VballNet detects players as ball with low motion energy — these would
        # become fallback positions where WASB has no detection. Motion energy
        # filter zeroes their confidence so they're excluded from the merge.
        from rallycut.tracking.ball_filter import BallFilterConfig

        vnet_prefilter = BallFilterConfig(
            enable_motion_energy_filter=True,
            enable_segment_pruning=False,
            enable_oscillation_pruning=False,
            enable_exit_ghost_removal=False,
            enable_outlier_removal=False,
            enable_blip_removal=False,
            enable_interpolation=False,
        )
        vnet_prefiltered = BallTemporalFilter(vnet_prefilter).filter_batch(
            vnet_result.positions
        )

        # Merge: WASB-primary, VballNet-fallback (per frame)
        wasb_by_frame = {p.frame_number: p for p in wasb_result.positions}
        vnet_by_frame = {p.frame_number: p for p in vnet_prefiltered}

        all_frames = sorted(set(wasb_by_frame.keys()) | set(vnet_by_frame.keys()))
        merged: list[BallPosition] = []
        wasb_count = 0
        vnet_count = 0

        for f in all_frames:
            wasb_pos = wasb_by_frame.get(f)
            vnet_pos = vnet_by_frame.get(f)

            if wasb_pos is not None and wasb_pos.confidence > 0:
                # Use WASB detection; tag with high motion_energy to indicate source
                merged.append(BallPosition(
                    frame_number=f,
                    x=wasb_pos.x,
                    y=wasb_pos.y,
                    confidence=wasb_pos.confidence,
                    motion_energy=1.0,  # Tag: WASB source
                ))
                wasb_count += 1
            elif vnet_pos is not None and vnet_pos.confidence > 0:
                # Fall back to VballNet
                merged.append(BallPosition(
                    frame_number=f,
                    x=vnet_pos.x,
                    y=vnet_pos.y,
                    confidence=vnet_pos.confidence,
                    motion_energy=vnet_pos.motion_energy,  # Preserve original
                ))
                vnet_count += 1
            else:
                # Neither has a detection
                merged.append(BallPosition(
                    frame_number=f, x=0.5, y=0.5, confidence=0.0,
                ))

        logger.info(
            f"Ensemble merged: {wasb_count} WASB + {vnet_count} VballNet = "
            f"{wasb_count + vnet_count} total detections"
        )

        if progress_callback:
            progress_callback(1.0)

        processing_time_ms = (time.time() - start_time) * 1000

        # Apply filtering to merged result
        raw_positions = None
        if enable_filtering:
            if preserve_raw:
                raw_positions = merged.copy()
            if filter_config is not None:
                config = filter_config
            else:
                from rallycut.tracking.ball_filter import get_ensemble_filter_config

                config = get_ensemble_filter_config()
            temporal_filter = BallTemporalFilter(config)
            merged = temporal_filter.filter_batch(merged)

        return BallTrackingResult(
            positions=merged,
            frame_count=max(wasb_result.frame_count, vnet_result.frame_count),
            video_fps=wasb_result.video_fps,
            video_width=wasb_result.video_width,
            video_height=wasb_result.video_height,
            processing_time_ms=processing_time_ms,
            model_version="ensemble_wasb+vballnet",
            filtering_enabled=enable_filtering,
            raw_positions=raw_positions,
        )
