"""Game state analysis for RallyCut."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rallycut.core.config import get_config
from rallycut.core.models import GameState, GameStateResult
from rallycut.core.video import Video


class GameStateAnalyzer:
    """Analyzes video to classify game states (SERVICE, PLAY, NO_PLAY)."""

    def __init__(
        self,
        device: str | None = None,
        model_path: Path | None = None,
        enable_temporal_smoothing: bool | None = None,
        temporal_smoothing_window: int | None = None,
    ):
        config = get_config()
        self.device = device or config.device
        self.model_path = model_path or config.videomae_model_path
        self.enable_temporal_smoothing = (
            enable_temporal_smoothing
            if enable_temporal_smoothing is not None
            else config.game_state.enable_temporal_smoothing
        )
        self.temporal_smoothing_window = (
            temporal_smoothing_window
            if temporal_smoothing_window is not None
            else config.game_state.temporal_smoothing_window
        )
        self._classifier: Any = None

    def _get_classifier(self) -> Any:
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
        stride: int | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        limit_seconds: float | None = None,
        batch_size: int | None = None,
        return_raw: bool = False,
    ) -> list[GameStateResult] | tuple[list[GameStateResult], list[GameStateResult]]:
        """
        Analyze entire video for game states using batch processing.

        Uses sequential frame reading with buffering for 10-100x faster
        performance compared to seeking to each frame.

        Args:
            video: Video to analyze
            stride: Number of frames between classification windows
            progress_callback: Callback for progress updates (percentage, message)
            limit_seconds: Only analyze first N seconds (for testing)
            batch_size: Number of windows to process in each batch
            return_raw: If True, returns tuple of (smoothed_results, raw_results)

        Returns:
            List of GameStateResult for each analyzed window, or
            tuple of (smoothed_results, raw_results) if return_raw=True
        """
        config = get_config()
        stride = stride or config.game_state.stride
        batch_size = batch_size or config.game_state.batch_size
        window_size = config.game_state.window_size
        target_size = config.game_state.analysis_size

        classifier = self._get_classifier()
        results: list[GameStateResult] = []

        total_frames = video.info.frame_count
        fps = video.info.fps

        # FPS normalization: subsample high-FPS videos to ~30fps
        # The model's 16-frame window needs ~0.5s of content to recognize patterns
        # At 60fps, 16 frames = 0.27s (too short); subsampling fixes this
        target_fps = 30.0
        if fps > 40:  # Likely 50fps or 60fps
            frame_step = round(fps / target_fps)
        else:
            frame_step = 1

        # Apply limit if specified
        if limit_seconds is not None:
            max_frames = min(total_frames, int(limit_seconds * fps))
        else:
            max_frames = total_frames

        # Adjust stride for subsampling (stride is in source frames, need to convert)
        effective_stride = stride // frame_step if frame_step > 1 else stride
        effective_stride = max(1, effective_stride)

        # Calculate total windows (in subsampled frame space)
        effective_max_frames = max_frames // frame_step
        total_windows = max(0, (effective_max_frames - window_size) // effective_stride + 1)

        if total_windows == 0:
            return results

        # Check if resize is needed (skip for optimized proxy already at 224x224)
        needs_resize = None  # Will be determined from first frame

        # Sequential reading with frame buffer
        # Buffer holds frames for overlapping windows (in subsampled space)
        frame_buffer = []
        buffer_start_idx = 0
        next_window_start = 0
        windows_processed = 0
        subsampled_idx = 0  # Index in subsampled frame space

        for frame_idx, frame in video.iter_frames(end_frame=max_frames, step=frame_step):
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

            # Check if we can process any windows (using subsampled indices)
            while (len(frame_buffer) >= window_size and
                   next_window_start <= subsampled_idx - window_size + 1):

                # Calculate buffer offset for this window
                buffer_offset = next_window_start - buffer_start_idx

                if buffer_offset >= 0 and buffer_offset + window_size <= len(frame_buffer):
                    # Extract window frames from buffer
                    window_frames = frame_buffer[buffer_offset:buffer_offset + window_size]

                    # Collect batch
                    batch_frames = [window_frames]
                    batch_starts = [next_window_start]

                    # Try to add more windows to batch
                    temp_start = next_window_start + effective_stride
                    while (len(batch_frames) < batch_size and
                           temp_start <= subsampled_idx - window_size + 1):
                        temp_offset = temp_start - buffer_start_idx
                        if temp_offset >= 0 and temp_offset + window_size <= len(frame_buffer):
                            batch_frames.append(frame_buffer[temp_offset:temp_offset + window_size])
                            batch_starts.append(temp_start)
                            temp_start += effective_stride
                        else:
                            break

                    # Process batch
                    batch_results = classifier.classify_segments_batch(batch_frames)

                    for i, (state, confidence, no_play_prob, play_prob, service_prob) in enumerate(batch_results):
                        # Convert subsampled frame index back to source frame index
                        subsampled_start = batch_starts[i]
                        source_start_frame = subsampled_start * frame_step
                        source_end_frame = source_start_frame + (window_size - 1) * frame_step
                        results.append(
                            GameStateResult(
                                state=state,
                                confidence=confidence,
                                start_frame=source_start_frame,
                                end_frame=source_end_frame,
                                play_confidence=play_prob,
                                service_confidence=service_prob,
                                no_play_confidence=no_play_prob,
                            )
                        )

                    next_window_start = batch_starts[-1] + effective_stride
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

            subsampled_idx += 1

        # Process any remaining windows
        while next_window_start <= effective_max_frames - window_size:
            buffer_offset = next_window_start - buffer_start_idx
            if buffer_offset >= 0 and buffer_offset + window_size <= len(frame_buffer):
                window_frames = frame_buffer[buffer_offset:buffer_offset + window_size]
                batch_results = classifier.classify_segments_batch([window_frames])

                for state, confidence, no_play_prob, play_prob, service_prob in batch_results:
                    # Convert subsampled frame index back to source frame index
                    source_start_frame = next_window_start * frame_step
                    source_end_frame = source_start_frame + (window_size - 1) * frame_step
                    results.append(
                        GameStateResult(
                            state=state,
                            confidence=confidence,
                            start_frame=source_start_frame,
                            end_frame=source_end_frame,
                            play_confidence=play_prob,
                            service_confidence=service_prob,
                            no_play_confidence=no_play_prob,
                        )
                    )
                windows_processed += 1
            next_window_start += effective_stride

        if progress_callback:
            progress_callback(1.0, f"Window {windows_processed}/{total_windows}")

        # Store raw results before smoothing (for diagnostics)
        raw_results = list(results) if return_raw else None

        # Apply temporal smoothing if enabled
        if self.enable_temporal_smoothing and results:
            results = self._smooth_results(results, self.temporal_smoothing_window)

        if return_raw:
            assert raw_results is not None
            return results, raw_results
        return results

    def smooth_results(
        self, results: list[GameStateResult], window_size: int | None = None
    ) -> list[GameStateResult]:
        """Public method to apply temporal smoothing to results."""
        ws = window_size if window_size is not None else self.temporal_smoothing_window
        return self._smooth_results(results, ws)

    def _smooth_results(
        self, results: list[GameStateResult], window_size: int = 5
    ) -> list[GameStateResult]:
        """
        Apply temporal smoothing to fix isolated classification errors.

        Uses a sliding median filter to smooth out isolated state flips.
        For example: PLAY-NO_PLAY-PLAY -> PLAY-PLAY-PLAY

        This allows using a larger stride while maintaining accuracy,
        since isolated errors are corrected by the majority vote.

        Args:
            results: List of GameStateResult to smooth
            window_size: Size of smoothing window (must be odd, default 5)

        Returns:
            Smoothed list of GameStateResult
        """
        if len(results) < window_size:
            return results

        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1

        half_window = window_size // 2
        smoothed = []

        for i, result in enumerate(results):
            # Get window of states around this result
            window_start = max(0, i - half_window)
            window_end = min(len(results), i + half_window + 1)

            # Count states in window
            play_count = 0
            no_play_count = 0
            service_count = 0

            for j in range(window_start, window_end):
                state = results[j].state
                if state == GameState.PLAY:
                    play_count += 1
                elif state == GameState.NO_PLAY:
                    no_play_count += 1
                elif state == GameState.SERVICE:
                    service_count += 1

            # Determine majority state (SERVICE treated as PLAY for voting)
            active_count = play_count + service_count

            if active_count > no_play_count:
                # Majority is active (PLAY/SERVICE) - keep original if active, else PLAY
                if result.state in (GameState.PLAY, GameState.SERVICE):
                    new_state = result.state
                else:
                    new_state = GameState.PLAY
            else:
                new_state = GameState.NO_PLAY

            # Create new result with smoothed state
            if new_state != result.state:
                smoothed.append(
                    GameStateResult(
                        state=new_state,
                        confidence=result.confidence * 0.9,  # Slightly reduce confidence
                        start_frame=result.start_frame,
                        end_frame=result.end_frame,
                        play_confidence=result.play_confidence,
                        service_confidence=result.service_confidence,
                        no_play_confidence=result.no_play_confidence,
                    )
                )
            else:
                smoothed.append(result)

        return smoothed
