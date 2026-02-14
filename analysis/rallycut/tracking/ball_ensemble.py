"""Trajectory-aware ensemble for TrackNet + VballNet ball tracking.

Merges predictions from both models per frame using position-aware logic:
- Both detect nearby: weighted average of positions
- Both detect far apart: prefer VballNet (has motion_energy context)
- Only one detects: use that detection

Output feeds directly into BallFilter.filter_batch() for segment pruning.

Usage:
    from rallycut.tracking.ball_ensemble import ensemble_positions, run_ensemble

    # Merge existing predictions
    merged = ensemble_positions(tracknet_preds, vballnet_preds)

    # Full pipeline: run both models and merge
    merged = run_ensemble(video_path, start_ms, end_ms)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)

# Max normalized distance between detections to consider them "agreeing"
# 5% of screen — real ball is ~1-2% of screen width, so 5% allows for
# model disagreement on exact position while still agreeing on location
AGREEMENT_THRESHOLD = 0.05


@dataclass
class EnsembleConfig:
    """Configuration for TrackNet + VballNet ensemble."""

    tracknet_model: str = "best"
    vballnet_model: str = "v2"
    agreement_threshold: float = AGREEMENT_THRESHOLD


def ensemble_positions(
    tracknet_positions: list[BallPosition],
    vballnet_positions: list[BallPosition],
    agreement_threshold: float = AGREEMENT_THRESHOLD,
) -> list[BallPosition]:
    """Merge TrackNet and VballNet predictions per frame.

    Strategy per frame:
    - Both detect within agreement_threshold: average positions, take max confidence.
      Preserves VballNet's motion_energy for downstream filtering.
    - Both detect far apart: prefer VballNet (it has 9-frame temporal context
      and motion_energy, which TrackNet lacks).
    - Only one detects: use that detection.

    Args:
        tracknet_positions: TrackNet ball positions (motion_energy=0.0).
        vballnet_positions: VballNet raw ball positions (with motion_energy).
        agreement_threshold: Max distance (normalized) to merge positions.

    Returns:
        Merged list of BallPosition, sorted by frame number.
    """
    tracknet_by_frame: dict[int, BallPosition] = {}
    for bp in tracknet_positions:
        existing = tracknet_by_frame.get(bp.frame_number)
        if existing is None or bp.confidence > existing.confidence:
            tracknet_by_frame[bp.frame_number] = bp

    vballnet_by_frame: dict[int, BallPosition] = {}
    for bp in vballnet_positions:
        existing = vballnet_by_frame.get(bp.frame_number)
        if existing is None or bp.confidence > existing.confidence:
            vballnet_by_frame[bp.frame_number] = bp

    all_frames = sorted(set(tracknet_by_frame) | set(vballnet_by_frame))
    merged: list[BallPosition] = []

    for frame in all_frames:
        t = tracknet_by_frame.get(frame)
        v = vballnet_by_frame.get(frame)

        if t is not None and v is not None:
            dist = math.hypot(t.x - v.x, t.y - v.y)
            if dist <= agreement_threshold:
                # Both agree: weighted average, keep VballNet's motion_energy
                merged.append(BallPosition(
                    frame_number=frame,
                    x=(t.x + v.x) / 2,
                    y=(t.y + v.y) / 2,
                    confidence=max(t.confidence, v.confidence),
                    motion_energy=v.motion_energy,
                ))
            else:
                # Disagree: prefer VballNet (9-frame context + motion_energy)
                merged.append(v)
        elif v is not None:
            merged.append(v)
        elif t is not None:
            merged.append(t)

    return merged


def run_ensemble(
    video_path: Path,
    start_ms: int,
    end_ms: int,
    config: EnsembleConfig | None = None,
) -> list[BallPosition]:
    """Run both TrackNet and VballNet on a video segment, merge results.

    Requires scripts/eval_tracknet.py to be importable (run from analysis/ dir).

    Args:
        video_path: Path to video file.
        start_ms: Start of rally in milliseconds.
        end_ms: End of rally in milliseconds.
        config: Ensemble configuration.

    Returns:
        Merged list of BallPosition.
    """
    # Lazy imports: BallTracker is heavy, eval_tracknet is a script
    from rallycut.tracking.ball_tracker import BallTracker

    try:
        from scripts.eval_tracknet import load_tracknet, run_tracknet_inference
    except ImportError:
        raise ImportError(
            "run_ensemble() requires scripts/eval_tracknet.py to be importable. "
            "Run from the analysis/ directory."
        )

    cfg = config or EnsembleConfig()

    # Run VballNet
    logger.info("Running VballNet inference...")
    vballnet_tracker = BallTracker(model=cfg.vballnet_model)
    vballnet_result = vballnet_tracker.track_video(
        video_path,
        start_ms=start_ms,
        end_ms=end_ms,
        enable_filtering=False,
    )

    # Convert to rally-relative frame numbers
    vballnet_positions = vballnet_result.positions
    if vballnet_positions:
        first_frame = min(p.frame_number for p in vballnet_positions)
        vballnet_positions = [
            BallPosition(
                frame_number=p.frame_number - first_frame,
                x=p.x,
                y=p.y,
                confidence=p.confidence,
                motion_energy=p.motion_energy,
            )
            for p in vballnet_positions
        ]

    # Run TrackNet
    logger.info("Running TrackNet inference...")
    tracknet_model = load_tracknet(cfg.tracknet_model)
    tracknet_positions = run_tracknet_inference(
        tracknet_model, video_path, start_ms, end_ms
    )

    logger.info(
        f"VballNet: {len(vballnet_positions)} detections, "
        f"TrackNet: {len(tracknet_positions)} detections"
    )

    return ensemble_positions(
        tracknet_positions, vballnet_positions, cfg.agreement_threshold
    )
