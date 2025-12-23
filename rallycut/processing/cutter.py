"""Video cutting functionality for RallyCut."""

from pathlib import Path
from typing import Callable, Optional

from rallycut.core.config import get_config
from rallycut.core.models import TimeSegment
from rallycut.core.video import Video
from rallycut.processing.exporter import FFmpegExporter


class VideoCutter:
    """Cuts video to remove dead time segments."""

    def __init__(
        self,
        device: Optional[str] = None,
        padding_seconds: Optional[float] = None,
        min_play_duration: Optional[float] = None,
        stride: int = 16,
        use_quick_mode: bool = False,
        use_two_pass: bool = False,
        limit_seconds: Optional[float] = None,
        use_proxy: bool = True,  # Use proxy for ML analysis (handled by TwoPassAnalyzer)
        min_gap_seconds: float = 1.5,
    ):
        config = get_config()
        self.device = device or config.device
        self.padding_seconds = padding_seconds or config.padding_seconds
        self.min_play_duration = min_play_duration or config.min_play_duration
        self.stride = stride
        self.use_quick_mode = use_quick_mode
        self.use_two_pass = use_two_pass
        self.limit_seconds = limit_seconds
        self.use_proxy = use_proxy
        self.min_gap_seconds = min_gap_seconds  # Min NO_PLAY gap before ending rally

        self._analyzer = None
        self.exporter = FFmpegExporter()

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
                                # Merge through the gap
                                merged_end = raw_segments[j + 1].end_frame
                                j += 2
                                continue
                        # Long enough NO_PLAY - end the rally here
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
        padding_frames = int(self.padding_seconds * fps)
        min_frames = int(self.min_play_duration * fps)

        play_segments = []
        for segment in segments:
            if segment.state not in (GameState.SERVICE, GameState.PLAY):
                continue
            if segment.frame_count < min_frames:
                continue

            padded_start = max(0, segment.start_frame - padding_frames)
            padded_end = segment.end_frame + padding_frames

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
            results = analyzer.analyze_video(
                video,
                stride=self.stride,
                progress_callback=progress_callback,
                limit_seconds=self.limit_seconds,
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
