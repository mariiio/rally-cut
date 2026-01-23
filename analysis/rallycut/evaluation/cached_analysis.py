"""Cached ML analysis for fast parameter iteration.

The key insight: ML inference is slow (~minutes), post-processing is instant (~ms).

This module separates these concerns:
1. `analyze_and_cache()` - Run VideoMAE once, cache raw results
2. `apply_post_processing()` - Apply heuristics to cached results instantly

This enables testing many parameter combinations without re-running ML.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import user_cache_dir

from rallycut.core.models import GameState, GameStateResult, TimeSegment
from rallycut.core.video import Video

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class CachedAnalysis:
    """Cached ML analysis results for a video."""

    video_id: str
    content_hash: str
    fps: float
    frame_count: int
    stride: int
    raw_results: list[GameStateResult]  # Raw ML classifications (before post-processing)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "video_id": self.video_id,
            "content_hash": self.content_hash,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "stride": self.stride,
            "raw_results": [
                {
                    "state": r.state.value,
                    "confidence": r.confidence,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "play_confidence": r.play_confidence,
                    "service_confidence": r.service_confidence,
                    "no_play_confidence": r.no_play_confidence,
                }
                for r in self.raw_results
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> CachedAnalysis:
        """Create from JSON dict."""
        raw_results = [
            GameStateResult(
                state=GameState(r["state"]),
                confidence=r["confidence"],
                start_frame=r["start_frame"],
                end_frame=r["end_frame"],
                play_confidence=r["play_confidence"],
                service_confidence=r["service_confidence"],
                no_play_confidence=r["no_play_confidence"],
            )
            for r in data["raw_results"]
        ]
        return cls(
            video_id=data["video_id"],
            content_hash=data["content_hash"],
            fps=data["fps"],
            frame_count=data["frame_count"],
            stride=data["stride"],
            raw_results=raw_results,
        )


@dataclass
class PostProcessingParams:
    """Parameters for post-processing heuristics."""

    min_gap_seconds: float = 5.0
    rally_continuation_seconds: float = 2.0
    min_play_duration: float = 1.0
    padding_seconds: float = 2.0
    padding_end_seconds: float = 3.0
    boundary_confidence_threshold: float = 0.35
    min_active_density: float = 0.25
    min_active_windows: int = 1

    @classmethod
    def from_defaults(cls) -> PostProcessingParams:
        """Create with default values from config."""
        from rallycut.core.config import get_config

        config = get_config()
        return cls(
            min_gap_seconds=config.segment.min_gap_seconds,
            rally_continuation_seconds=config.segment.rally_continuation_seconds,
            min_play_duration=config.segment.min_play_duration,
            padding_seconds=config.segment.padding_seconds,
            padding_end_seconds=config.segment.padding_end_seconds,
        )


class AnalysisCache:
    """Manages cached ML analysis results."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.cache/rallycut/evaluation/
        """
        self.cache_dir = cache_dir or Path(user_cache_dir("rallycut")) / "evaluation"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, content_hash: str, stride: int) -> str:
        """Generate cache key from content hash and stride.

        Different stride = different ML results (samples at different intervals).
        """
        return hashlib.sha256(f"{content_hash}:{stride}".encode()).hexdigest()[:16]

    def _get_cache_path(self, content_hash: str, stride: int) -> Path:
        """Get cache file path."""
        key = self._get_cache_key(content_hash, stride)
        return self.cache_dir / f"{key}.json"

    def get(self, content_hash: str, stride: int) -> CachedAnalysis | None:
        """Load cached analysis if available.

        Args:
            content_hash: Video content hash.
            stride: Stride used for analysis.

        Returns:
            CachedAnalysis if cached, None otherwise.
        """
        cache_path = self._get_cache_path(content_hash, stride)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            return CachedAnalysis.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            # Invalid cache file - remove it
            cache_path.unlink(missing_ok=True)
            return None

    def put(self, analysis: CachedAnalysis) -> None:
        """Save analysis to cache.

        Args:
            analysis: Analysis results to cache.
        """
        cache_path = self._get_cache_path(analysis.content_hash, analysis.stride)
        cache_path.write_text(json.dumps(analysis.to_dict(), indent=2))

    def has(self, content_hash: str, stride: int) -> bool:
        """Check if analysis is cached.

        Args:
            content_hash: Video content hash.
            stride: Stride used for analysis.

        Returns:
            True if cached.
        """
        return self._get_cache_path(content_hash, stride).exists()

    def clear(self) -> int:
        """Clear all cached analyses.

        Returns:
            Number of files deleted.
        """
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


def analyze_and_cache(
    video_path: Path,
    video_id: str,
    content_hash: str,
    stride: int | None = None,
    device: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cache: AnalysisCache | None = None,
    use_proxy: bool = True,
) -> CachedAnalysis:
    """Run ML analysis and cache results.

    This runs the expensive VideoMAE inference and caches the raw results
    for fast iteration with different post-processing parameters.

    Args:
        video_path: Path to video file.
        video_id: Video identifier for tracking.
        content_hash: SHA-256 hash of video content.
        stride: Frame stride for analysis. Defaults to config value.
        device: Device for ML inference (cuda, mps, cpu).
        progress_callback: Callback for progress updates.
        cache: AnalysisCache instance. Creates default if None.
        use_proxy: Whether to use proxy video for faster analysis.

    Returns:
        CachedAnalysis with raw ML results.
    """
    from rallycut.analysis.game_state import GameStateAnalyzer
    from rallycut.core.config import get_config, get_recommended_batch_size
    from rallycut.core.proxy import ProxyConfig, ProxyGenerator

    config = get_config()
    effective_stride = stride if stride is not None else config.game_state.stride
    effective_device = device or config.device

    # Check cache first
    if cache is None:
        cache = AnalysisCache()

    cached = cache.get(content_hash, effective_stride)
    if cached is not None:
        return cached

    # Initialize analyzer
    analyzer = GameStateAnalyzer(device=effective_device)

    # Get video info
    with Video(video_path) as video:
        source_fps = video.info.fps
        source_frame_count = video.info.frame_count

    # Run analysis (with or without proxy)
    if use_proxy:
        proxy_gen = ProxyGenerator(
            config=ProxyConfig(height=config.proxy.height, fps=config.proxy.fps),
            cache_dir=config.proxy_cache_dir,
        )

        def proxy_progress(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct * 0.1, msg)

        proxy_path, mapper = proxy_gen.get_or_create(video_path, source_fps, proxy_progress)

        with Video(proxy_path) as proxy_video:
            # Normalize stride for proxy FPS
            proxy_stride = int(round(effective_stride * (mapper.proxy_fps / 30.0)))
            batch_size = get_recommended_batch_size(effective_device)

            def analysis_progress(pct: float, msg: str) -> None:
                if progress_callback:
                    progress_callback(0.1 + pct * 0.9, msg)

            proxy_results_raw = analyzer.analyze_video(
                proxy_video,
                stride=max(1, proxy_stride),
                progress_callback=analysis_progress,
                batch_size=batch_size,
            )
            # analyze_video returns list[GameStateResult] when return_raw=False (default)
            proxy_results: list[GameStateResult] = proxy_results_raw  # type: ignore[assignment]

        # Map results back to source frame space
        results = [
            GameStateResult(
                state=r.state,
                confidence=r.confidence,
                start_frame=mapper.proxy_to_source(r.start_frame or 0),
                end_frame=mapper.proxy_to_source(r.end_frame or 0),
                play_confidence=r.play_confidence,
                service_confidence=r.service_confidence,
                no_play_confidence=r.no_play_confidence,
            )
            for r in proxy_results
        ]
        fps = source_fps
    else:
        with Video(video_path) as video:
            fps = video.info.fps
            batch_size = get_recommended_batch_size(effective_device)

            results_raw = analyzer.analyze_video(
                video,
                stride=effective_stride,
                progress_callback=progress_callback,
                batch_size=batch_size,
            )
            # analyze_video returns list[GameStateResult] when return_raw=False (default)
            results = results_raw  # type: ignore[assignment]

    # Create and cache analysis
    analysis = CachedAnalysis(
        video_id=video_id,
        content_hash=content_hash,
        fps=fps,
        frame_count=source_frame_count,
        stride=effective_stride,
        raw_results=results,
    )
    cache.put(analysis)

    return analysis


def apply_post_processing(
    cached: CachedAnalysis,
    params: PostProcessingParams | None = None,
) -> list[TimeSegment]:
    """Apply post-processing heuristics to cached ML results.

    This is the fast path - takes cached raw ML results and applies
    configurable post-processing to generate segments. ~1ms per call.

    Args:
        cached: Cached ML analysis results.
        params: Post-processing parameters. Defaults to config values.

    Returns:
        List of detected rally segments.
    """
    from rallycut.processing.cutter import (
        VideoCutter,
    )

    if params is None:
        params = PostProcessingParams.from_defaults()

    # Create a cutter with the specified parameters
    cutter = VideoCutter(
        min_gap_seconds=params.min_gap_seconds,
        rally_continuation_seconds=params.rally_continuation_seconds,
        min_play_duration=params.min_play_duration,
        padding_seconds=params.padding_seconds,
        padding_end_seconds=params.padding_end_seconds,
        stride=cached.stride,
        auto_stride=False,  # Don't re-normalize, results already in source frame space
    )

    # Override class-level thresholds if different from params
    # These are module-level constants in cutter.py, but we can work around
    # by directly calling the processing methods

    # Apply heuristics using the cutter's methods
    # Note: This reuses the existing well-tested logic
    segments, _suggested = cutter._get_segments_from_results(
        cached.raw_results,
        cached.fps,
        cached.frame_count,
    )

    return segments


def apply_post_processing_custom(
    cached: CachedAnalysis,
    params: PostProcessingParams,
) -> list[TimeSegment]:
    """Apply post-processing with full parameter customization.

    This version implements the heuristics directly to allow overriding
    all parameters including BOUNDARY_CONFIDENCE_THRESHOLD and MIN_ACTIVE_DENSITY
    which are constants in the cutter module.

    Args:
        cached: Cached ML analysis results.
        params: Post-processing parameters.

    Returns:
        List of detected rally segments.
    """
    fps = cached.fps
    results = cached.raw_results[:]  # Copy to avoid modifying cache

    if not results:
        return []

    # 1. Apply confidence-based boundary extension
    extended_results = _apply_confidence_extension(
        results, params.boundary_confidence_threshold
    )

    # 2. Apply rally continuation heuristic
    extended_results = _apply_rally_continuation(
        extended_results, fps, params.rally_continuation_seconds
    )

    # 3. Convert to segments with merging
    segments = _build_segments(
        extended_results,
        results,  # Original for density calculation
        fps,
        cached.frame_count,
        params,
    )

    return segments


def _apply_confidence_extension(
    results: list[GameStateResult],
    threshold: float,
) -> list[GameStateResult]:
    """Apply confidence-based boundary extension."""
    if len(results) < 3:
        return results

    extended = []
    for i, result in enumerate(results):
        if result.state == GameState.NO_PLAY:
            active_confidence = result.play_confidence + result.service_confidence

            prev_is_play = i > 0 and results[i - 1].state in (GameState.PLAY, GameState.SERVICE)
            next_is_play = (
                i < len(results) - 1
                and results[i + 1].state in (GameState.PLAY, GameState.SERVICE)
            )

            should_extend = False
            if prev_is_play and next_is_play:
                should_extend = active_confidence > 0.25
            elif prev_is_play or next_is_play:
                should_extend = active_confidence > threshold

            if should_extend:
                extended.append(
                    GameStateResult(
                        state=GameState.PLAY,
                        confidence=active_confidence,
                        start_frame=result.start_frame,
                        end_frame=result.end_frame,
                        play_confidence=result.play_confidence,
                        service_confidence=result.service_confidence,
                        no_play_confidence=result.no_play_confidence,
                    )
                )
                continue

        extended.append(result)

    return extended


def _apply_rally_continuation(
    results: list[GameStateResult],
    fps: float,
    continuation_seconds: float,
) -> list[GameStateResult]:
    """Apply rally continuation heuristic."""
    if continuation_seconds <= 0 or not results:
        return results

    min_no_play_frames = int(continuation_seconds * fps)

    # Build frame -> result mapping
    frame_to_result: dict[int, int] = {}
    for idx, r in enumerate(results):
        if r.start_frame is not None and r.end_frame is not None:
            for f in range(r.start_frame, r.end_frame + 1):
                frame_to_result[f] = idx

    if not frame_to_result:
        return results

    max_frame = max(frame_to_result.keys())
    in_rally = False
    consecutive_no_play = 0
    results_to_extend: set[int] = set()

    for frame in range(max_frame + 1):
        result_idx = frame_to_result.get(frame)
        if result_idx is None:
            if in_rally:
                consecutive_no_play += 1
                if consecutive_no_play >= min_no_play_frames:
                    in_rally = False
            continue

        result = results[result_idx]
        is_active = result.state in (GameState.PLAY, GameState.SERVICE)

        if is_active:
            in_rally = True
            consecutive_no_play = 0
        else:
            if in_rally:
                consecutive_no_play += 1
                if consecutive_no_play >= min_no_play_frames:
                    in_rally = False
                else:
                    results_to_extend.add(result_idx)

    extended = []
    for idx, result in enumerate(results):
        if idx in results_to_extend:
            extended.append(
                GameStateResult(
                    state=GameState.PLAY,
                    confidence=result.confidence * 0.8,
                    start_frame=result.start_frame,
                    end_frame=result.end_frame,
                    play_confidence=result.play_confidence,
                    service_confidence=result.service_confidence,
                    no_play_confidence=result.no_play_confidence,
                )
            )
        else:
            extended.append(result)

    return extended


def _build_segments(
    extended_results: list[GameStateResult],
    original_results: list[GameStateResult],
    fps: float,
    total_frames: int,
    params: PostProcessingParams,
) -> list[TimeSegment]:
    """Build and filter segments from extended results."""
    if not extended_results:
        return []

    min_no_play_frames = int(params.min_gap_seconds * fps)
    min_bridge_duration_frames = int(1.0 * fps)

    # First pass: merge adjacent same-state
    raw_segments: list[TimeSegment] = []
    current_state = extended_results[0].state
    current_start = extended_results[0].start_frame or 0
    current_end = extended_results[0].end_frame or 0

    for result in extended_results[1:]:
        if result.state == current_state:
            current_end = result.end_frame or current_end
        else:
            raw_segments.append(
                TimeSegment(
                    start_frame=current_start,
                    end_frame=current_end,
                    start_time=current_start / fps,
                    end_time=current_end / fps,
                    state=current_state,
                )
            )
            current_state = result.state
            current_start = result.start_frame or 0
            current_end = result.end_frame or 0

    raw_segments.append(
        TimeSegment(
            start_frame=current_start,
            end_frame=current_end,
            start_time=current_start / fps,
            end_time=current_end / fps,
            state=current_state,
        )
    )

    # Second pass: merge through short gaps
    segments = []
    i = 0
    while i < len(raw_segments):
        seg = raw_segments[i]

        if seg.state in (GameState.SERVICE, GameState.PLAY):
            merged_end = seg.end_frame
            j = i + 1

            while j < len(raw_segments):
                next_seg = raw_segments[j]

                if next_seg.state == GameState.NO_PLAY:
                    if next_seg.frame_count < min_no_play_frames:
                        if (
                            j + 1 < len(raw_segments)
                            and raw_segments[j + 1].state in (GameState.SERVICE, GameState.PLAY)
                        ):
                            after_gap = raw_segments[j + 1]
                            if after_gap.frame_count >= min_bridge_duration_frames:
                                merged_end = after_gap.end_frame
                                j += 2
                                continue
                    break
                elif next_seg.state in (GameState.SERVICE, GameState.PLAY):
                    merged_end = next_seg.end_frame
                    j += 1
                else:
                    break

            segments.append(
                TimeSegment(
                    start_frame=seg.start_frame,
                    end_frame=merged_end,
                    start_time=seg.start_frame / fps,
                    end_time=merged_end / fps,
                    state=seg.state,
                )
            )
            i = j
        else:
            segments.append(seg)
            i += 1

    # Filter and pad
    padding_start_frames = int(params.padding_seconds * fps)
    padding_end_frames = int(params.padding_end_seconds * fps)
    min_frames = int(params.min_play_duration * fps)

    play_segments = []
    for segment in segments:
        if segment.state not in (GameState.SERVICE, GameState.PLAY):
            continue

        # Count active windows from original results
        active_count = 0
        for r in original_results:
            r_start = r.start_frame if r.start_frame is not None else 0
            r_end = r.end_frame if r.end_frame is not None else 0
            if r_end >= segment.start_frame and r_start <= segment.end_frame:
                if r.state in (GameState.PLAY, GameState.SERVICE):
                    active_count += 1

        if active_count < params.min_active_windows:
            continue

        if active_count > 1 and segment.frame_count < min_frames:
            continue

        # Density filter
        if active_count > 1:
            segment_stride_intervals = max(1, segment.frame_count / 48)  # Use base stride
            active_density = active_count / segment_stride_intervals
            if active_density < params.min_active_density:
                continue

        padded_start = max(0, segment.start_frame - padding_start_frames)
        padded_end = min(segment.end_frame + padding_end_frames, total_frames - 1)

        if padded_start >= padded_end:
            continue

        play_segments.append(
            TimeSegment(
                start_frame=padded_start,
                end_frame=padded_end,
                start_time=padded_start / fps,
                end_time=padded_end / fps,
                state=segment.state,
            )
        )

    # Merge overlapping
    if not play_segments:
        return []

    sorted_segments = sorted(play_segments, key=lambda s: s.start_frame)
    merged = [sorted_segments[0]]

    for segment in sorted_segments[1:]:
        last = merged[-1]
        if segment.start_frame <= last.end_frame + 1:
            merged[-1] = TimeSegment(
                start_frame=last.start_frame,
                end_frame=max(last.end_frame, segment.end_frame),
                start_time=last.start_time,
                end_time=max(last.end_frame, segment.end_frame) / fps,
                state=last.state,
            )
        else:
            merged.append(segment)

    return merged
