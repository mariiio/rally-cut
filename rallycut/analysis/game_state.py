"""Game state analysis for RallyCut."""

from pathlib import Path
from typing import Callable, Optional

from rallycut.core.config import get_config
from rallycut.core.models import GameState, GameStateResult, TimeSegment
from rallycut.core.video import Video


class GameStateAnalyzer:
    """Analyzes video to classify game states (SERVICE, PLAY, NO_PLAY)."""

    WINDOW_SIZE = 16  # VideoMAE expects 16 frames
    ANALYSIS_SIZE = (224, 224)  # VideoMAE input size - resize for speed

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[Path] = None,
        use_resize: bool = True,
    ):
        config = get_config()
        self.device = device or config.device
        self.model_path = model_path or config.videomae_model_path
        self.use_resize = use_resize
        self._classifier = None

    def _get_classifier(self):
        """Lazy load the classifier."""
        if self._classifier is None:
            from lib.volleyball_ml.video_mae import GameStateClassifier

            self._classifier = GameStateClassifier(
                model_path=self.model_path,
                device=self.device,
            )
        return self._classifier

    def analyze_video(
        self,
        video: Video,
        stride: int = 8,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        limit_seconds: Optional[float] = None,
        batch_size: int = 8,
    ) -> list[GameStateResult]:
        """
        Analyze entire video for game states using batch processing.

        Uses sequential frame reading with buffering for 10-100x faster
        performance compared to seeking to each frame.

        Args:
            video: Video to analyze
            stride: Number of frames between classification windows
            progress_callback: Callback for progress updates (percentage, message)
            limit_seconds: Only analyze first N seconds (for testing)
            batch_size: Number of windows to process in each batch (default 8)

        Returns:
            List of GameStateResult for each analyzed window
        """
        classifier = self._get_classifier()
        results = []

        total_frames = video.info.frame_count

        # Apply limit if specified
        if limit_seconds is not None:
            max_frames = min(total_frames, int(limit_seconds * video.info.fps))
        else:
            max_frames = total_frames

        # Calculate total windows
        total_windows = max(0, (max_frames - self.WINDOW_SIZE) // stride + 1)

        if total_windows == 0:
            return results

        # Check if resize is needed (skip for optimized proxy already at 224x224)
        target_size = self.ANALYSIS_SIZE if self.use_resize else None
        needs_resize = None  # Will be determined from first frame

        # Sequential reading with frame buffer
        # Buffer holds frames for overlapping windows
        frame_buffer = []
        buffer_start_idx = 0
        next_window_start = 0
        windows_processed = 0

        for frame_idx, frame in video.iter_frames(end_frame=max_frames):
            # Check on first frame if resize is needed
            if needs_resize is None and target_size:
                import cv2
                h, w = frame.shape[:2]
                needs_resize = (w != target_size[0] or h != target_size[1])
            elif needs_resize is None:
                needs_resize = False

            # Only resize if needed (skip for optimized proxy)
            if needs_resize:
                import cv2
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            frame_buffer.append(frame)

            # Check if we can process any windows
            while (len(frame_buffer) >= self.WINDOW_SIZE and
                   next_window_start <= frame_idx - self.WINDOW_SIZE + 1):

                # Calculate buffer offset for this window
                buffer_offset = next_window_start - buffer_start_idx

                if buffer_offset >= 0 and buffer_offset + self.WINDOW_SIZE <= len(frame_buffer):
                    # Extract window frames from buffer
                    window_frames = frame_buffer[buffer_offset:buffer_offset + self.WINDOW_SIZE]

                    # Collect batch
                    batch_frames = [window_frames]
                    batch_starts = [next_window_start]

                    # Try to add more windows to batch
                    temp_start = next_window_start + stride
                    while (len(batch_frames) < batch_size and
                           temp_start <= frame_idx - self.WINDOW_SIZE + 1):
                        temp_offset = temp_start - buffer_start_idx
                        if temp_offset >= 0 and temp_offset + self.WINDOW_SIZE <= len(frame_buffer):
                            batch_frames.append(frame_buffer[temp_offset:temp_offset + self.WINDOW_SIZE])
                            batch_starts.append(temp_start)
                            temp_start += stride
                        else:
                            break

                    # Process batch
                    batch_results = classifier.classify_segments_batch(batch_frames)

                    for i, (state, confidence) in enumerate(batch_results):
                        start_frame = batch_starts[i]
                        results.append(
                            GameStateResult(
                                state=state,
                                confidence=confidence,
                                start_frame=start_frame,
                                end_frame=start_frame + self.WINDOW_SIZE - 1,
                            )
                        )

                    next_window_start = batch_starts[-1] + stride
                    windows_processed += len(batch_frames)

                    # Report progress
                    if progress_callback and windows_processed % batch_size == 0:
                        progress = windows_processed / total_windows
                        progress_callback(progress, f"Window {windows_processed}/{total_windows}")
                else:
                    break

            # Trim buffer to save memory (keep only what we need for future windows)
            min_needed_frame = next_window_start
            frames_to_drop = min_needed_frame - buffer_start_idx
            if frames_to_drop > 0:
                frame_buffer = frame_buffer[frames_to_drop:]
                buffer_start_idx = min_needed_frame

        # Process any remaining windows
        while next_window_start <= max_frames - self.WINDOW_SIZE:
            buffer_offset = next_window_start - buffer_start_idx
            if buffer_offset >= 0 and buffer_offset + self.WINDOW_SIZE <= len(frame_buffer):
                window_frames = frame_buffer[buffer_offset:buffer_offset + self.WINDOW_SIZE]
                batch_results = classifier.classify_segments_batch([window_frames])

                for state, confidence in batch_results:
                    results.append(
                        GameStateResult(
                            state=state,
                            confidence=confidence,
                            start_frame=next_window_start,
                            end_frame=next_window_start + self.WINDOW_SIZE - 1,
                        )
                    )
                windows_processed += 1
            next_window_start += stride

        if progress_callback:
            progress_callback(1.0, f"Window {windows_processed}/{total_windows}")

        return results

    def get_segments(
        self,
        results: list[GameStateResult],
        fps: float,
    ) -> list[TimeSegment]:
        """
        Convert classification results into merged time segments.

        Adjacent results with the same state are merged into single segments.

        Args:
            results: List of classification results
            fps: Video frame rate

        Returns:
            List of TimeSegment with merged consecutive states
        """
        if not results:
            return []

        segments = []
        current_state = results[0].state
        current_start = results[0].start_frame
        current_end = results[0].end_frame

        for result in results[1:]:
            if result.state == current_state:
                # Extend current segment
                current_end = result.end_frame
            else:
                # Save current segment and start new one
                segments.append(
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

        # Don't forget the last segment
        segments.append(
            TimeSegment(
                start_frame=current_start,
                end_frame=current_end,
                start_time=current_start / fps,
                end_time=current_end / fps,
                state=current_state,
            )
        )

        return segments

    def get_play_segments(
        self,
        segments: list[TimeSegment],
        padding_seconds: float,
        min_duration_seconds: float,
        fps: float,
    ) -> list[TimeSegment]:
        """
        Filter segments to only include play-related segments.

        Args:
            segments: All time segments
            padding_seconds: Padding to add before/after each segment
            min_duration_seconds: Minimum segment duration to keep
            fps: Video frame rate

        Returns:
            List of play segments (SERVICE or PLAY) with padding applied
        """
        padding_frames = int(padding_seconds * fps)
        min_frames = int(min_duration_seconds * fps)

        play_segments = []

        for segment in segments:
            # Only keep SERVICE or PLAY segments
            if segment.state not in (GameState.SERVICE, GameState.PLAY):
                continue

            # Check minimum duration
            if segment.frame_count < min_frames:
                continue

            # Apply padding
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

        return play_segments

    def merge_overlapping_segments(
        self,
        segments: list[TimeSegment],
        fps: float,
    ) -> list[TimeSegment]:
        """
        Merge overlapping or adjacent segments.

        Args:
            segments: List of segments (may overlap after padding)
            fps: Video frame rate

        Returns:
            List of merged non-overlapping segments
        """
        if not segments:
            return []

        # Sort by start frame
        sorted_segments = sorted(segments, key=lambda s: s.start_frame)

        merged = [sorted_segments[0]]

        for segment in sorted_segments[1:]:
            last = merged[-1]

            # Check if overlapping or adjacent
            if segment.start_frame <= last.end_frame + 1:
                # Merge by extending end frame
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
