"""Video cutting functionality for RallyCut."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rallycut.core.config import (
    MODEL_PRESETS,
    get_config,
    get_model_path,
    get_recommended_batch_size,
)
from rallycut.core.models import GameStateResult, RejectionReason, SuggestedSegment, TimeSegment
from rallycut.core.video import Video
from rallycut.processing.exporter import FFmpegExporter

if TYPE_CHECKING:
    from pathlib import Path as PathType

    from rallycut.analysis.game_state import GameStateAnalyzer
    from rallycut.core.proxy import ProxyGenerator
    from rallycut.temporal.processor import TemporalProcessor

# Default heuristic values (indoor model defaults)
# These can be overridden by MODEL_PRESETS when a model_variant is specified
DEFAULT_BOUNDARY_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_MIN_ACTIVE_DENSITY = 0.25

# Minimum number of PLAY/SERVICE windows required to form a valid segment
MIN_ACTIVE_WINDOWS = 1

# Minimum confidence to include as a suggestion (don't suggest obvious noise)
MIN_SUGGESTION_CONFIDENCE = 0.3


class VideoCutter:
    """Cuts video to remove dead time segments."""

    # Reference FPS for stride calibration (stride 32 is optimal at 30fps)
    REFERENCE_FPS = 30.0

    def __init__(
        self,
        device: str | None = None,
        padding_seconds: float | None = None,
        padding_end_seconds: float | None = None,
        min_play_duration: float | None = None,
        stride: int | None = None,
        limit_seconds: float | None = None,
        use_proxy: bool | None = None,
        min_gap_seconds: float | None = None,
        auto_stride: bool = True,
        rally_continuation_seconds: float | None = None,
        model_variant: str = "indoor",
        use_temporal_model: bool = False,
        temporal_model_path: Path | None = None,
        temporal_model_version: str = "v2",
        use_binary_head_decoder: bool = False,
        binary_head_model_path: Path | None = None,
    ):
        config = get_config()
        self.device = device or config.device
        self.model_variant = model_variant

        # Get model-specific preset if available
        preset = MODEL_PRESETS.get(model_variant, MODEL_PRESETS["indoor"])

        # Apply preset values, with explicit parameters taking precedence
        self.padding_seconds = padding_seconds if padding_seconds is not None else config.segment.padding_seconds
        self.padding_end_seconds = padding_end_seconds if padding_end_seconds is not None else config.segment.padding_end_seconds
        self.min_play_duration = min_play_duration if min_play_duration is not None else preset.get("min_play_duration", config.segment.min_play_duration)
        self.base_stride = stride if stride is not None else config.game_state.stride
        self.limit_seconds = limit_seconds
        self.use_proxy = use_proxy if use_proxy is not None else config.proxy.enabled
        self.min_gap_seconds = min_gap_seconds if min_gap_seconds is not None else preset.get("min_gap_seconds", config.segment.min_gap_seconds)
        self.auto_stride = auto_stride
        self.rally_continuation_seconds = rally_continuation_seconds if rally_continuation_seconds is not None else preset.get("rally_continuation_seconds", config.segment.rally_continuation_seconds)

        # Model-specific heuristic thresholds from preset
        self.boundary_confidence_threshold = preset.get("boundary_confidence_threshold", DEFAULT_BOUNDARY_CONFIDENCE_THRESHOLD)
        self.min_active_density = preset.get("min_active_density", DEFAULT_MIN_ACTIVE_DENSITY)

        # Model path based on variant
        self._model_path: PathType | None = get_model_path(model_variant)

        # Temporal model settings
        self.use_temporal_model = use_temporal_model
        self.temporal_model_path = temporal_model_path
        self.temporal_model_version = temporal_model_version

        # Binary head decoder settings
        self.use_binary_head_decoder = use_binary_head_decoder
        self.binary_head_model_path = binary_head_model_path

        self._analyzer: GameStateAnalyzer | None = None
        self._proxy_generator: ProxyGenerator | None = None
        self._temporal_processor: object | None = None
        self._binary_head_model: object | None = None
        self.exporter = FFmpegExporter()

    def _normalize_stride(self, fps: float) -> int:
        """
        Normalize stride based on video FPS for consistent temporal sampling.

        Stride 32 at 30fps = sample every ~1.07 seconds.
        For 60fps video, use stride 64 to maintain same temporal rate.
        """
        if not self.auto_stride:
            return self.base_stride

        # Scale stride proportionally to FPS
        normalized = int(round(self.base_stride * (fps / self.REFERENCE_FPS)))
        # Ensure minimum stride of 1
        return max(1, normalized)

    def _get_analyzer(self) -> GameStateAnalyzer:
        """Lazy load the game state analyzer."""
        if self._analyzer is None:
            from rallycut.analysis.game_state import GameStateAnalyzer

            self._analyzer = GameStateAnalyzer(
                device=self.device, model_path=self._model_path
            )
        return self._analyzer

    def _get_proxy_generator(self) -> ProxyGenerator:
        """Lazy load proxy generator."""
        if self._proxy_generator is None:
            from rallycut.core.proxy import ProxyConfig, ProxyGenerator

            config = get_config()
            self._proxy_generator = ProxyGenerator(
                config=ProxyConfig(
                    height=config.proxy.height,
                    fps=config.proxy.fps,
                ),
                cache_dir=config.proxy_cache_dir,
            )
        return self._proxy_generator

    def _get_temporal_processor(self) -> TemporalProcessor:
        """Lazy load temporal processor."""
        if self._temporal_processor is None:
            from rallycut.temporal.processor import TemporalProcessor, TemporalProcessorConfig

            config = TemporalProcessorConfig(
                model_path=self.temporal_model_path,
                model_version=self.temporal_model_version,
                coarse_stride=self.base_stride,
                device=self.device,
            )
            self._temporal_processor = TemporalProcessor(config)
        return self._temporal_processor  # type: ignore[return-value]

    def _apply_confidence_extension(
        self, results: list[GameStateResult]
    ) -> list[GameStateResult]:
        """
        Apply confidence-based boundary extension.

        Extends PLAY segments by treating low-confidence NO_PLAY classifications
        as PLAY if they're adjacent to PLAY and have significant play_confidence.

        This addresses the issue where model returns NO_PLAY with ~0.4 confidence
        for frames that are actually part of a rally but visually ambiguous.
        """
        from rallycut.core.models import GameState

        if len(results) < 3:
            return results

        extended = []
        for i, result in enumerate(results):
            # Check if this NO_PLAY should be extended to PLAY
            if result.state == GameState.NO_PLAY:
                # Calculate active confidence (PLAY + SERVICE probability)
                active_confidence = result.play_confidence + result.service_confidence

                # Check if this is adjacent to PLAY/SERVICE segments
                prev_is_play = (
                    i > 0 and results[i - 1].state in (GameState.PLAY, GameState.SERVICE)
                )
                next_is_play = (
                    i < len(results) - 1
                    and results[i + 1].state in (GameState.PLAY, GameState.SERVICE)
                )

                # Extend if:
                # 1. Adjacent to play AND active_confidence > threshold
                # 2. Between two play segments (gap bridging)
                should_extend = False
                if prev_is_play and next_is_play:
                    # Gap between plays - extend if any active confidence
                    should_extend = active_confidence > 0.25
                elif prev_is_play or next_is_play:
                    # Boundary - use model-specific threshold
                    should_extend = active_confidence > self.boundary_confidence_threshold

                if should_extend:
                    # Convert NO_PLAY to PLAY with reduced confidence
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
        self, results: list[GameStateResult], fps: float
    ) -> list[GameStateResult]:
        """
        Apply rally continuation heuristic.

        Once PLAY/SERVICE is detected, keep it active until we see
        rally_continuation_seconds of consecutive NO_PLAY predictions.
        This bridges gaps where the ML model incorrectly predicts NO_PLAY mid-rally.

        Optimized to iterate over results (O(n)) instead of frames (O(frames)).
        For a 30-minute video, this reduces iterations from ~54,000 to ~1,000.

        Args:
            results: List of GameStateResult from ML analysis
            fps: Video frames per second

        Returns:
            Modified list with rally continuation applied
        """
        from rallycut.core.models import GameState

        if self.rally_continuation_seconds <= 0 or not results:
            return results

        # Convert continuation threshold from seconds to frames
        min_no_play_frames = int(self.rally_continuation_seconds * fps)

        # Track rally state and which results to extend
        in_rally = False
        no_play_start_frame: int | None = None  # Frame where NO_PLAY streak started
        pending_no_play_indices: list[int] = []  # NO_PLAY results waiting to be extended
        results_to_extend: set[int] = set()

        for idx, result in enumerate(results):
            if result.start_frame is None or result.end_frame is None:
                continue

            is_active = result.state in (GameState.PLAY, GameState.SERVICE)

            if is_active:
                if in_rally and pending_no_play_indices:
                    # We're back to PLAY - extend all pending NO_PLAY results
                    results_to_extend.update(pending_no_play_indices)
                # Start or continue rally
                in_rally = True
                no_play_start_frame = None
                pending_no_play_indices = []
            else:
                # NO_PLAY result
                if in_rally:
                    if no_play_start_frame is None:
                        # Start of NO_PLAY streak
                        no_play_start_frame = result.start_frame

                    # Check if NO_PLAY streak exceeds threshold
                    no_play_duration = result.end_frame - no_play_start_frame
                    if no_play_duration >= min_no_play_frames:
                        # End the rally - don't extend pending NO_PLAY results
                        in_rally = False
                        no_play_start_frame = None
                        pending_no_play_indices = []
                    else:
                        # Still within threshold - mark for potential extension
                        pending_no_play_indices.append(idx)

        # Create modified results list
        extended = []
        for idx, result in enumerate(results):
            if idx in results_to_extend:
                # Convert NO_PLAY to PLAY
                extended.append(
                    GameStateResult(
                        state=GameState.PLAY,
                        confidence=result.confidence * 0.8,  # Reduce confidence for extended frames
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

    def _count_active_windows_in_range(
        self, results: list[GameStateResult], start_frame: int, end_frame: int
    ) -> tuple[int, int, float]:
        """Count PLAY and SERVICE windows that overlap with the given frame range.

        Returns:
            Tuple of (play_count, service_count, avg_confidence)
        """
        from rallycut.core.models import GameState

        play_count = 0
        service_count = 0
        total_confidence = 0.0
        for r in results:
            r_start = r.start_frame if r.start_frame is not None else 0
            r_end = r.end_frame if r.end_frame is not None else 0
            if r_end >= start_frame and r_start <= end_frame:
                if r.state == GameState.PLAY:
                    play_count += 1
                    total_confidence += r.confidence
                elif r.state == GameState.SERVICE:
                    service_count += 1
                    total_confidence += r.confidence
        active_count = play_count + service_count
        avg_confidence = total_confidence / active_count if active_count > 0 else 0.0
        return play_count, service_count, avg_confidence

    def _get_segments_from_results(
        self, results: list[GameStateResult], fps: float, total_frames: int | None = None
    ) -> tuple[list[TimeSegment], list[SuggestedSegment]]:
        """Convert analysis results to merged play segments with hysteresis.

        Returns:
            Tuple of (confirmed_segments, suggested_segments)
        """
        from rallycut.core.models import GameState

        if not results:
            return [], []

        suggested_segments: list[SuggestedSegment] = []

        # Apply confidence-based boundary extension before segment creation
        extended_results = self._apply_confidence_extension(results)

        # Apply rally continuation heuristic - keeps rallies active until N seconds of NO_PLAY
        extended_results = self._apply_rally_continuation(extended_results, fps)

        # Minimum NO_PLAY duration (in frames) before ending a PLAY segment
        # This prevents early rally termination from brief misclassifications
        min_no_play_frames = int(self.min_gap_seconds * fps)

        # First pass: merge adjacent same-state results
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

        # Second pass: merge short NO_PLAY gaps into adjacent PLAY segments
        # This extends rallies through brief detection gaps
        segments = []
        i = 0
        # Minimum duration for a segment to trigger gap-bridging (1.0 seconds)
        # This prevents isolated false positive windows from causing over-extension
        min_bridge_duration_frames = int(1.0 * fps)
        while i < len(raw_segments):
            seg = raw_segments[i]

            # If this is a PLAY/SERVICE segment, look ahead for short NO_PLAY gaps
            if seg.state in (GameState.SERVICE, GameState.PLAY):
                merged_end = seg.end_frame

                # Check following segments
                j = i + 1
                while j < len(raw_segments):
                    next_seg = raw_segments[j]

                    if next_seg.state == GameState.NO_PLAY:
                        # Short NO_PLAY gap - might be false detection
                        if next_seg.frame_count < min_no_play_frames:
                            # Check if there's another PLAY segment after
                            if j + 1 < len(raw_segments) and raw_segments[j + 1].state in (GameState.SERVICE, GameState.PLAY):
                                after_gap_seg = raw_segments[j + 1]
                                # Only bridge if the segment after the gap is substantial
                                # This prevents isolated short false positives from causing over-extension
                                # The gap itself being short (<min_no_play_frames) is the primary criteria
                                if after_gap_seg.frame_count >= min_bridge_duration_frames:
                                    # Merge through the gap
                                    merged_end = after_gap_seg.end_frame
                                    j += 2
                                    continue
                                else:
                                    # Segment is too short, but look ahead to see if there's
                                    # substantial PLAY (accumulated) within a short distance
                                    # This handles alternating SERVICE/PLAY that collectively form a rally
                                    # Key: only accumulate if gaps between play segments are SHORT
                                    accumulated_play_frames = after_gap_seg.frame_count
                                    last_play_idx = j + 1
                                    last_play_end = after_gap_seg.end_frame
                                    max_internal_gap_frames = int(1.5 * fps)  # Max gap between accumulated segments
                                    for lookahead in range(j + 2, min(j + 10, len(raw_segments))):
                                        look_seg = raw_segments[lookahead]
                                        if look_seg.state in (GameState.SERVICE, GameState.PLAY):
                                            accumulated_play_frames += look_seg.frame_count
                                            last_play_idx = lookahead
                                            last_play_end = look_seg.end_frame
                                        elif look_seg.state == GameState.NO_PLAY:
                                            if look_seg.frame_count >= min_no_play_frames:
                                                # Long NO_PLAY gap - stop looking
                                                break
                                            elif look_seg.frame_count >= max_internal_gap_frames:
                                                # Internal gap too long - isolated false positives
                                                # Don't accumulate across this gap
                                                break
                                    # If accumulated play is substantial, merge through
                                    if accumulated_play_frames >= min_bridge_duration_frames:
                                        merged_end = last_play_end
                                        j = last_play_idx + 1
                                        continue
                        # Long enough NO_PLAY or no substantial segment found - end the rally here
                        break
                    elif next_seg.state in (GameState.SERVICE, GameState.PLAY):
                        # Continue merging play segments
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

        # Filter for play segments
        padding_start_frames = int(self.padding_seconds * fps)
        padding_end_frames = int(self.padding_end_seconds * fps)
        min_frames = int(self.min_play_duration * fps)

        play_segments = []
        for segment in segments:
            if segment.state not in (GameState.SERVICE, GameState.PLAY):
                continue

            # Filter out segments with too few active windows (isolated false positives)
            # Count windows from ORIGINAL results (before extension) to avoid counting
            # extended NO_PLAY windows that were converted to PLAY
            play_count, service_count, avg_confidence = self._count_active_windows_in_range(
                results, segment.start_frame, segment.end_frame
            )
            active_window_count = play_count + service_count

            # Require minimum number of active windows
            if active_window_count < MIN_ACTIVE_WINDOWS:
                if avg_confidence >= MIN_SUGGESTION_CONFIDENCE:
                    suggested_segments.append(SuggestedSegment(
                        start_frame=segment.start_frame,
                        end_frame=segment.end_frame,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        state=segment.state,
                        rejection_reason=RejectionReason.INSUFFICIENT_WINDOWS,
                        avg_confidence=avg_confidence,
                        active_window_count=active_window_count,
                    ))
                continue

            # Apply min_play_duration only to multi-window segments
            # Single confident windows should be kept (will be padded to reasonable length)
            if active_window_count > 1 and segment.frame_count < min_frames:
                if avg_confidence >= MIN_SUGGESTION_CONFIDENCE:
                    suggested_segments.append(SuggestedSegment(
                        start_frame=segment.start_frame,
                        end_frame=segment.end_frame,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        state=segment.state,
                        rejection_reason=RejectionReason.TOO_SHORT,
                        avg_confidence=avg_confidence,
                        active_window_count=active_window_count,
                    ))
                continue

            # Filter out sparse detections - require minimum density of active windows
            # This catches cases like 2 isolated windows spanning a 5-second gap
            # Skip for single-window segments (density doesn't apply)
            if active_window_count > 1:
                stride_frames = self.base_stride
                segment_stride_intervals = max(1, segment.frame_count / stride_frames)
                active_density = active_window_count / segment_stride_intervals
                if active_density < self.min_active_density:
                    if avg_confidence >= MIN_SUGGESTION_CONFIDENCE:
                        suggested_segments.append(SuggestedSegment(
                            start_frame=segment.start_frame,
                            end_frame=segment.end_frame,
                            start_time=segment.start_time,
                            end_time=segment.end_time,
                            state=segment.state,
                            rejection_reason=RejectionReason.SPARSE_DENSITY,
                            avg_confidence=avg_confidence,
                            active_window_count=active_window_count,
                            active_density=active_density,
                        ))
                    continue

            padded_start = max(0, segment.start_frame - padding_start_frames)
            padded_end = segment.end_frame + padding_end_frames
            if total_frames is not None:
                padded_start = min(padded_start, total_frames - 1)
                padded_end = min(padded_end, total_frames - 1)
                # Skip segments that are entirely outside video bounds
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

        # Merge overlapping segments
        if not play_segments:
            return [], suggested_segments

        sorted_segments = sorted(play_segments, key=lambda s: s.start_frame)
        merged = [sorted_segments[0]]

        for segment in sorted_segments[1:]:
            last = merged[-1]
            # Merge if segments overlap or are adjacent
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

        return merged, suggested_segments

    def _analyze_with_temporal_model(
        self,
        input_path: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[list[TimeSegment], list[SuggestedSegment]]:
        """Analyze video using temporal model.

        Args:
            input_path: Path to input video.
            progress_callback: Progress callback.

        Returns:
            Tuple of (confirmed_segments, empty suggested_segments).
        """
        from rallycut.temporal.processor import rally_segments_to_time_segments

        # Get source video info
        with Video(input_path) as video:
            source_fps = video.info.fps
            source_frame_count = video.info.frame_count
            content_hash = video.compute_content_hash()

        # Use proxy for temporal model too if enabled
        if self.use_proxy:
            proxy_gen = self._get_proxy_generator()

            def proxy_progress(pct: float, msg: str) -> None:
                if progress_callback:
                    progress_callback(pct * 0.1, msg)

            proxy_path, mapper = proxy_gen.get_or_create(
                input_path, source_fps, proxy_progress
            )

            # Run temporal model on proxy
            processor = self._get_temporal_processor()

            def temporal_progress(pct: float, msg: str) -> None:
                if progress_callback:
                    progress_callback(0.1 + pct * 0.9, msg)

            with Video(proxy_path) as proxy_video:
                result = processor.process_video(
                    proxy_video,
                    content_hash=content_hash,
                    progress_callback=temporal_progress,
                )

            # Convert segments and map back to source frame space
            proxy_segments = rally_segments_to_time_segments(
                result.segments,
                fps=mapper.proxy_fps,
                padding_start=self.padding_seconds,
                padding_end=self.padding_end_seconds,
                total_frames=int(mapper.proxy_fps * mapper.source_duration),
            )

            # Map segment times back to source
            segments = []
            for seg in proxy_segments:
                source_start = mapper.proxy_to_source(seg.start_frame)
                source_end = mapper.proxy_to_source(seg.end_frame)
                segments.append(
                    TimeSegment(
                        start_frame=source_start,
                        end_frame=min(source_end, source_frame_count - 1),
                        start_time=source_start / source_fps,
                        end_time=min(source_end, source_frame_count - 1) / source_fps,
                        state=seg.state,
                    )
                )
        else:
            # Run temporal model on source video
            processor = self._get_temporal_processor()

            with Video(input_path) as video:
                result = processor.process_video(
                    video,
                    content_hash=content_hash,
                    progress_callback=progress_callback,
                )

            segments = rally_segments_to_time_segments(
                result.segments,
                fps=source_fps,
                padding_start=self.padding_seconds,
                padding_end=self.padding_end_seconds,
                total_frames=source_frame_count,
            )

        # Temporal model doesn't produce suggested segments
        return segments, []

    def _analyze_with_binary_head_decoder(
        self,
        input_path: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[list[TimeSegment], list[SuggestedSegment]]:
        """Analyze video using binary head + deterministic decoder.

        Args:
            input_path: Path to input video.
            progress_callback: Progress callback.

        Returns:
            Tuple of (confirmed_segments, empty suggested_segments).
        """
        from rallycut.core.models import GameState
        from rallycut.temporal.deterministic_decoder import DecoderConfig
        from rallycut.temporal.features import FeatureCache
        from rallycut.temporal.inference import load_binary_head_model, run_binary_head_decoder

        # Get source video info
        with Video(input_path) as video:
            source_fps = video.info.fps
            source_frame_count = video.info.frame_count
            content_hash = video.compute_content_hash()

        # Determine binary head model path
        if self.binary_head_model_path is not None:
            model_path = self.binary_head_model_path
        else:
            # Default path
            from rallycut.core.config import get_config
            config = get_config()
            model_path = config.weights_dir / "binary_head" / "best_binary_head.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Binary head model not found at {model_path}. "
                "Train one with 'rallycut train binary-head' first."
            )

        # Load model if not cached
        if self._binary_head_model is None:
            if progress_callback:
                progress_callback(0.0, "Loading binary head model...")
            self._binary_head_model = load_binary_head_model(model_path, self.device)

        # Load cached features
        if progress_callback:
            progress_callback(0.05, "Loading cached features...")

        cache = FeatureCache(cache_dir=config.feature_cache_dir)
        stride = self.base_stride

        cached = cache.get(content_hash, stride)
        if cached is None:
            raise ValueError(
                f"No cached features found for video (stride={stride}). "
                "Run 'rallycut train extract-features' first."
            )

        features, _ = cached

        # Configure decoder
        decoder_config = DecoderConfig(
            fps=source_fps,
            stride=stride,
        )

        # Run binary head + decoder
        if progress_callback:
            progress_callback(0.1, "Running binary head inference...")

        result = run_binary_head_decoder(
            features=features,
            model=self._binary_head_model,
            config=decoder_config,
            device=self.device,
        )

        if progress_callback:
            progress_callback(0.9, "Converting segments...")

        # Convert decoder segments to TimeSegments with padding
        padding_start_frames = int(self.padding_seconds * source_fps)
        padding_end_frames = int(self.padding_end_seconds * source_fps)

        segments = []
        for start_time, end_time in result.segments:
            start_frame = int(start_time * source_fps)
            end_frame = int(end_time * source_fps)

            # Apply padding
            padded_start = max(0, start_frame - padding_start_frames)
            padded_end = min(end_frame + padding_end_frames, source_frame_count - 1)

            segments.append(
                TimeSegment(
                    start_frame=padded_start,
                    end_frame=padded_end,
                    start_time=padded_start / source_fps,
                    end_time=padded_end / source_fps,
                    state=GameState.PLAY,
                )
            )

        # Merge overlapping segments
        if segments:
            sorted_segments = sorted(segments, key=lambda s: s.start_frame)
            merged = [sorted_segments[0]]

            for segment in sorted_segments[1:]:
                last = merged[-1]
                if segment.start_frame <= last.end_frame + 1:
                    merged[-1] = TimeSegment(
                        start_frame=last.start_frame,
                        end_frame=max(last.end_frame, segment.end_frame),
                        start_time=last.start_time,
                        end_time=max(last.end_frame, segment.end_frame) / source_fps,
                        state=last.state,
                    )
                else:
                    merged.append(segment)
            segments = merged

        if progress_callback:
            progress_callback(1.0, "Complete")

        # Binary head decoder doesn't produce suggested segments
        return segments, []

    def analyze_only(
        self,
        input_path: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[list[TimeSegment], list[SuggestedSegment]]:
        """
        Analyze video to find play segments without generating output.

        Args:
            input_path: Path to input video
            progress_callback: Callback for progress updates (percentage, message)

        Returns:
            Tuple of (confirmed_segments, suggested_segments)
        """
        # Use binary head decoder if enabled (takes priority)
        if self.use_binary_head_decoder:
            return self._analyze_with_binary_head_decoder(input_path, progress_callback)

        # Use temporal model if enabled
        if self.use_temporal_model:
            return self._analyze_with_temporal_model(input_path, progress_callback)

        from rallycut.core.models import GameStateResult

        analyzer = self._get_analyzer()

        # Get source video info for FPS, frame count, and frame mapping
        with Video(input_path) as video:
            source_fps = video.info.fps
            source_frame_count = video.info.frame_count

        # Use proxy for faster analysis if enabled
        if self.use_proxy:
            proxy_gen = self._get_proxy_generator()

            def proxy_progress(pct: float, msg: str) -> None:
                if progress_callback:
                    progress_callback(pct * 0.1, msg)

            proxy_path, mapper = proxy_gen.get_or_create(
                input_path, source_fps, proxy_progress
            )

            # Run analysis on proxy video
            with Video(proxy_path) as proxy_video:
                # Normalize stride based on proxy FPS
                effective_stride = self._normalize_stride(mapper.proxy_fps)
                batch_size = get_recommended_batch_size(self.device)

                def analysis_progress(pct: float, msg: str) -> None:
                    if progress_callback:
                        progress_callback(0.1 + pct * 0.9, msg)

                proxy_results_raw = analyzer.analyze_video(
                    proxy_video,
                    stride=effective_stride,
                    progress_callback=analysis_progress,
                    limit_seconds=self.limit_seconds,
                    batch_size=batch_size,
                )
                # When return_raw=False (default), returns list[GameStateResult]
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
            # Run analysis directly on source video
            with Video(input_path) as video:
                fps = video.info.fps
                effective_stride = self._normalize_stride(fps)
                batch_size = get_recommended_batch_size(self.device)

                direct_results = analyzer.analyze_video(
                    video,
                    stride=effective_stride,
                    progress_callback=progress_callback,
                    limit_seconds=self.limit_seconds,
                    batch_size=batch_size,
                )
                # When return_raw=False (default), returns list[GameStateResult]
                results = direct_results  # type: ignore[assignment]

        # Convert to merged segments (clamp to source video duration)
        merged_segments, suggested_segments = self._get_segments_from_results(results, fps, source_frame_count)

        return merged_segments, suggested_segments

    def analyze_with_diagnostics(
        self,
        input_path: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict:
        """
        Analyze video and return all intermediate results for diagnostics.

        Returns:
            Dictionary with:
            - raw_results: ML classifications before smoothing
            - smoothed_results: ML classifications after smoothing
            - raw_segments: Segments before min_duration filter
            - final_segments: Segments after filtering
            - fps: Video FPS used for analysis
        """
        from rallycut.core.models import GameStateResult

        analyzer = self._get_analyzer()

        # Get source video info
        with Video(input_path) as video:
            source_fps = video.info.fps
            source_frame_count = video.info.frame_count

        # Run analysis (with or without proxy)
        if self.use_proxy:
            proxy_gen = self._get_proxy_generator()

            def proxy_progress(pct: float, msg: str) -> None:
                if progress_callback:
                    progress_callback(pct * 0.1, msg)

            proxy_path, mapper = proxy_gen.get_or_create(
                input_path, source_fps, proxy_progress
            )

            with Video(proxy_path) as proxy_video:
                effective_stride = self._normalize_stride(mapper.proxy_fps)
                batch_size = get_recommended_batch_size(self.device)

                def analysis_progress(pct: float, msg: str) -> None:
                    if progress_callback:
                        progress_callback(0.1 + pct * 0.9, msg)

                result_tuple = analyzer.analyze_video(
                    proxy_video,
                    stride=effective_stride,
                    progress_callback=analysis_progress,
                    limit_seconds=self.limit_seconds,
                    batch_size=batch_size,
                    return_raw=True,
                )
                # When return_raw=True, returns tuple[list, list]
                assert isinstance(result_tuple, tuple)
                smoothed_proxy = result_tuple[0]
                raw_proxy = result_tuple[1]

            # Map results back to source frame space
            raw_results = [
                GameStateResult(
                    state=r.state,
                    confidence=r.confidence,
                    start_frame=mapper.proxy_to_source(r.start_frame or 0),
                    end_frame=mapper.proxy_to_source(r.end_frame or 0),
                    play_confidence=r.play_confidence,
                    service_confidence=r.service_confidence,
                    no_play_confidence=r.no_play_confidence,
                )
                for r in raw_proxy
            ]
            smoothed_results = [
                GameStateResult(
                    state=r.state,
                    confidence=r.confidence,
                    start_frame=mapper.proxy_to_source(r.start_frame or 0),
                    end_frame=mapper.proxy_to_source(r.end_frame or 0),
                    play_confidence=r.play_confidence,
                    service_confidence=r.service_confidence,
                    no_play_confidence=r.no_play_confidence,
                )
                for r in smoothed_proxy
            ]
            fps = source_fps
        else:
            with Video(input_path) as video:
                fps = video.info.fps
                effective_stride = self._normalize_stride(fps)
                batch_size = get_recommended_batch_size(self.device)

                result_tuple = analyzer.analyze_video(
                    video,
                    stride=effective_stride,
                    progress_callback=progress_callback,
                    limit_seconds=self.limit_seconds,
                    batch_size=batch_size,
                    return_raw=True,
                )
                # When return_raw=True, returns tuple[list, list]
                assert isinstance(result_tuple, tuple)
                smoothed_results = result_tuple[0]
                raw_results = result_tuple[1]

        # Get segments before min_duration filter (by temporarily setting it to 0)
        original_min_play = self.min_play_duration
        self.min_play_duration = 0.0
        raw_segments, _ = self._get_segments_from_results(smoothed_results, fps, source_frame_count)
        self.min_play_duration = original_min_play

        # Get final segments with actual filter
        final_segments, suggested_segments = self._get_segments_from_results(smoothed_results, fps, source_frame_count)

        return {
            "raw_results": raw_results,
            "smoothed_results": smoothed_results,
            "raw_segments": raw_segments,
            "final_segments": final_segments,
            "suggested_segments": suggested_segments,
            "fps": fps,
        }

    def cut_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[Path, list[TimeSegment]]:
        """
        Cut video to remove dead time.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Callback for progress updates (percentage, message)

        Returns:
            Tuple of (output_path, list of kept segments)
        """
        # Phase 1: Analyze video (0-70% of progress)
        def analysis_progress(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct * 0.7, f"Analyzing: {msg}")

        merged_segments, _ = self.analyze_only(input_path, analysis_progress)

        if not merged_segments:
            raise ValueError("No play segments detected in video")

        # Phase 2: Export video (70-100% of progress)
        def export_progress(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(0.7 + pct * 0.3, f"Exporting: {msg}")

        self.exporter.export_segments(
            input_path=input_path,
            output_path=output_path,
            segments=merged_segments,
            progress_callback=export_progress,
        )

        return output_path, merged_segments

    def get_cut_stats(
        self,
        original_duration: float,
        segments: list[TimeSegment],
    ) -> dict:
        """
        Calculate statistics about the cut.

        Args:
            original_duration: Original video duration in seconds
            segments: List of kept segments

        Returns:
            Dictionary with cut statistics
        """
        kept_duration = sum(s.duration for s in segments)
        removed_duration = original_duration - kept_duration

        return {
            "original_duration": original_duration,
            "kept_duration": kept_duration,
            "removed_duration": removed_duration,
            "kept_percentage": (kept_duration / original_duration * 100)
            if original_duration > 0
            else 0,
            "removed_percentage": (removed_duration / original_duration * 100)
            if original_duration > 0
            else 0,
            "segment_count": len(segments),
        }
