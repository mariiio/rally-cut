"""Video cutting functionality for RallyCut."""

from pathlib import Path
from typing import Callable, Optional

from rallycut.core.config import get_config, get_recommended_batch_size
from rallycut.core.models import TimeSegment
from rallycut.core.video import Video
from rallycut.processing.exporter import FFmpegExporter


class VideoCutter:
    """Cuts video to remove dead time segments."""

    # Reference FPS for stride calibration (stride 32 is optimal at 30fps)
    REFERENCE_FPS = 30.0

    def __init__(
        self,
        device: Optional[str] = None,
        padding_seconds: Optional[float] = None,
        padding_end_seconds: Optional[float] = None,
        min_play_duration: Optional[float] = None,
        stride: Optional[int] = None,
        use_quick_mode: bool = False,
        use_two_pass: bool = False,
        limit_seconds: Optional[float] = None,
        use_proxy: Optional[bool] = None,
        min_gap_seconds: Optional[float] = None,
        auto_stride: bool = True,
    ):
        config = get_config()
        self.device = device or config.device
        self.padding_seconds = padding_seconds if padding_seconds is not None else config.segment.padding_seconds
        # End padding defaults to start padding + 0.5s for smoother endings
        self.padding_end_seconds = padding_end_seconds if padding_end_seconds is not None else (self.padding_seconds + 0.5)
        self.min_play_duration = min_play_duration if min_play_duration is not None else config.segment.min_play_duration
        self.base_stride = stride if stride is not None else config.game_state.stride
        self.use_quick_mode = use_quick_mode
        self.use_two_pass = use_two_pass
        self.limit_seconds = limit_seconds
        self.use_proxy = use_proxy if use_proxy is not None else config.proxy.enabled
        self.min_gap_seconds = min_gap_seconds if min_gap_seconds is not None else 3.0
        self.auto_stride = auto_stride

        self._analyzer = None
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

    def _get_analyzer(self):
        """Lazy load the appropriate analyzer."""
        if self._analyzer is None:
            if self.use_quick_mode:
                from rallycut.analysis.motion_detector import MotionDetector
                self._analyzer = MotionDetector()
            elif self.use_two_pass:
                from rallycut.analysis.two_pass import TwoPassAnalyzer
                self._analyzer = TwoPassAnalyzer(
                    device=self.device,
                    use_proxy=self.use_proxy,
                )
            else:
                from rallycut.analysis.game_state import GameStateAnalyzer
                self._analyzer = GameStateAnalyzer(device=self.device)
        return self._analyzer

    def _get_segments_from_results(self, results, fps) -> list[TimeSegment]:
        """Convert analysis results to merged play segments with hysteresis."""
        from rallycut.core.models import GameState

        if not results:
            return []

        # Minimum NO_PLAY duration (in frames) before ending a PLAY segment
        # This prevents early rally termination from brief misclassifications
        min_no_play_frames = int(self.min_gap_seconds * fps)

        # First pass: merge adjacent same-state results
        raw_segments = []
        current_state = results[0].state
        current_start = results[0].start_frame
        current_end = results[0].end_frame

        for result in results[1:]:
            if result.state == current_state:
                current_end = result.end_frame
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
                current_start = result.start_frame
                current_end = result.end_frame

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
                                if after_gap_seg.frame_count >= min_bridge_duration_frames:
                                    # Merge through the gap
                                    merged_end = after_gap_seg.end_frame
                                    j += 2
                                    continue
                                else:
                                    # Segment is too short, but look ahead to see if there's
                                    # substantial PLAY (accumulated) within a short distance
                                    # This handles alternating SERVICE/PLAY that collectively form a rally
                                    # Key: only accumulate if gaps between play segments are SHORT (<1.5s)
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
            if segment.frame_count < min_frames:
                continue

            padded_start = max(0, segment.start_frame - padding_start_frames)
            padded_end = segment.end_frame + padding_end_frames

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

    def analyze_only(
        self,
        input_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> list[TimeSegment]:
        """
        Analyze video to find play segments without generating output.

        Args:
            input_path: Path to input video
            progress_callback: Callback for progress updates (percentage, message)

        Returns:
            List of play segments
        """
        analyzer = self._get_analyzer()

        # Run analysis (proxy handling is now done by TwoPassAnalyzer)
        with Video(input_path) as video:
            fps = video.info.fps
            # Normalize stride based on video FPS
            effective_stride = self._normalize_stride(fps)
            # Auto-scale batch size for GPU
            batch_size = get_recommended_batch_size(self.device)

            results = analyzer.analyze_video(
                video,
                stride=effective_stride,
                progress_callback=progress_callback,
                limit_seconds=self.limit_seconds,
                batch_size=batch_size,
            )

        # Convert to merged segments
        merged_segments = self._get_segments_from_results(results, fps)

        return merged_segments

    def cut_video(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
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
        def analysis_progress(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct * 0.7, f"Analyzing: {msg}")

        merged_segments = self.analyze_only(input_path, analysis_progress)

        if not merged_segments:
            raise ValueError("No play segments detected in video")

        # Phase 2: Export video (70-100% of progress)
        def export_progress(pct: float, msg: str):
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
