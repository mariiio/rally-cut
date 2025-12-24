"""Two-pass analysis for faster ML processing.

First pass: Quick motion detection to find candidate regions.
Second pass: ML analysis only on motion regions (skips 60-70% of video).

Optimizations:
- Adaptive stride: Use stride 8 near boundaries, stride 16 in middle of regions
- Adaptive frame sampling: Sample every 2nd frame in middle zones for wider temporal context
- Proxy video: Use 480p@15fps proxy for ML analysis (2-4x faster decode)
- Temporal smoothing: Fix isolated classification errors with median filter
- Confidence-based skipping: Skip ML for high-confidence motion regions
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from rallycut.core.config import get_config
from rallycut.core.models import GameState, GameStateResult
from rallycut.core.profiler import get_profiler
from rallycut.core.video import Video


@dataclass
class MotionRegion:
    """A region of video with detected motion."""

    start_frame: int
    end_frame: int
    avg_motion: float  # Average motion score (0-1)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


class TwoPassAnalyzer:
    """
    Two-pass analysis: quick motion scan then ML on motion regions only.

    Typical beach volleyball matches have 60-70% dead time. By first scanning
    for motion, we can skip ML analysis on static regions and achieve 2-3x speedup.

    With proxy enabled (default), Pass 2 uses a 480p@15fps proxy for 2-4x faster
    frame decode while maintaining ML accuracy (model uses 224x224 anyway).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        motion_stride: Optional[int] = None,
        ml_stride: Optional[int] = None,
        motion_padding_seconds: Optional[float] = None,
        boundary_seconds: Optional[float] = None,
        use_proxy: Optional[bool] = None,
    ):
        config = get_config()
        self.device = device or config.device
        self.motion_stride = motion_stride or config.two_pass.motion_stride
        self.ml_stride = ml_stride or config.two_pass.ml_stride
        self.motion_padding_seconds = motion_padding_seconds or config.two_pass.motion_padding_seconds
        self.boundary_seconds = boundary_seconds or config.two_pass.boundary_seconds
        self.use_proxy = use_proxy if use_proxy is not None else config.proxy.enabled

        self._motion_detector = None
        self._ml_analyzer = None
        self._proxy_generator = None

    def _get_motion_detector(self):
        """Lazy load motion detector."""
        if self._motion_detector is None:
            from rallycut.analysis.motion_detector import MotionDetector

            config = get_config()
            self._motion_detector = MotionDetector(
                high_motion_threshold=config.two_pass.motion_high_threshold,
                low_motion_threshold=config.two_pass.motion_low_threshold,
            )
        return self._motion_detector

    def _get_ml_analyzer(self):
        """Lazy load ML analyzer."""
        if self._ml_analyzer is None:
            from rallycut.analysis.game_state import GameStateAnalyzer

            self._ml_analyzer = GameStateAnalyzer(device=self.device)
        return self._ml_analyzer

    def _get_proxy_generator(self):
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

    def analyze_video(
        self,
        video: Video,
        stride: int = 8,  # This is the ML stride, motion uses motion_stride
        progress_callback: Optional[Callable[[float, str], None]] = None,
        limit_seconds: Optional[float] = None,
        batch_size: int = 16,  # Larger batch for better GPU utilization
    ) -> list[GameStateResult]:
        """
        Analyze video using two-pass approach.

        Args:
            video: Video to analyze
            stride: Frame stride for ML analysis (motion uses larger stride)
            progress_callback: Callback for progress updates
            limit_seconds: Only analyze first N seconds
            batch_size: Batch size for ML inference

        Returns:
            List of GameStateResult
        """
        import time

        profiler = get_profiler()
        fps = video.info.fps
        total_frames = video.info.frame_count

        if limit_seconds is not None:
            total_frames = min(total_frames, int(limit_seconds * fps))

        if progress_callback:
            progress_callback(0.0, f"Pass 1/2: Motion scan ({total_frames} frames)")

        # Pass 1: Quick motion scan (10% of progress)
        motion_start = time.time()

        def motion_progress(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct * 0.1, f"Pass 1/2: {msg}")

        motion_regions = self._detect_motion_regions(
            video, fps, total_frames, motion_progress
        )

        motion_time = time.time() - motion_start

        # Record motion detection timing
        from rallycut.core.profiler import TimingEntry
        profiler._entries.append(TimingEntry(
            component="motion",
            operation="detect_regions",
            duration_seconds=motion_time,
            metadata={"regions": len(motion_regions) if motion_regions else 0},
        ))

        if not motion_regions:
            # No motion detected - return all as NO_PLAY
            if progress_callback:
                progress_callback(1.0, "No motion detected")
            return [
                GameStateResult(
                    state=GameState.NO_PLAY,
                    confidence=0.9,
                    start_frame=0,
                    end_frame=total_frames - 1,
                )
            ]

        # Calculate how much of video has motion
        motion_frames = sum(r.duration_frames for r in motion_regions)
        motion_ratio = motion_frames / total_frames

        if progress_callback:
            progress_callback(
                0.1,
                f"Pass 2/2: ML on {len(motion_regions)} regions ({motion_ratio:.0%} of video, motion scan: {motion_time:.1f}s)"
            )

        # Pass 2: ML analysis on motion regions only (90% of progress)
        # Optionally use proxy video for faster decoding
        ml_start = time.time()

        if self.use_proxy and video.path is not None:
            # Generate/get proxy and frame mapper
            proxy_gen = self._get_proxy_generator()

            def proxy_progress(pct: float, msg: str):
                if progress_callback:
                    progress_callback(0.1 + pct * 0.05, f"Proxy: {msg}")

            proxy_path, mapper = proxy_gen.get_or_create(
                video.path, fps, proxy_progress
            )

            # Map motion regions to proxy frame space
            proxy_regions = [
                MotionRegion(
                    start_frame=mapper.source_to_proxy(r.start_frame),
                    end_frame=mapper.source_to_proxy(r.end_frame),
                    avg_motion=r.avg_motion,
                )
                for r in motion_regions
            ]
            proxy_total_frames = mapper.source_to_proxy(total_frames)

            # Run ML analysis on proxy
            with Video(proxy_path) as proxy_video:
                proxy_results = self._analyze_motion_regions(
                    proxy_video,
                    proxy_regions,
                    stride,
                    batch_size,
                    mapper.proxy_fps,
                    proxy_total_frames,
                    progress_callback,
                )

            # Map results back to source frame space
            results = [
                GameStateResult(
                    state=r.state,
                    confidence=r.confidence,
                    start_frame=mapper.proxy_to_source(r.start_frame),
                    end_frame=mapper.proxy_to_source(r.end_frame),
                )
                for r in proxy_results
            ]
        else:
            # Original behavior without proxy
            results = self._analyze_motion_regions(
                video,
                motion_regions,
                stride,
                batch_size,
                fps,
                total_frames,
                progress_callback,
            )

        ml_time = time.time() - ml_start

        # Apply temporal smoothing if enabled
        config = get_config()
        if config.two_pass.enable_temporal_smoothing:
            results = self._smooth_results(
                results, window_size=config.two_pass.temporal_smoothing_window
            )

        if progress_callback:
            progress_callback(1.0, f"Done (motion: {motion_time:.1f}s, ML: {ml_time:.1f}s)")

        return results

    def _detect_motion_regions(
        self,
        video: Video,
        fps: float,
        total_frames: int,
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> list[MotionRegion]:
        """
        Detect regions with motion using fast motion detector.

        Returns:
            List of MotionRegion objects with motion scores
        """
        motion_detector = self._get_motion_detector()

        # Run motion detection with larger stride for speed
        results = motion_detector.analyze_video(
            video,
            stride=self.motion_stride,
            progress_callback=progress_callback,
            limit_seconds=total_frames / fps,
        )

        # Find PLAY segments from motion results, tracking confidence scores
        regions: list[tuple[int, int, list[float]]] = []  # (start, end, confidences)
        current_start = None
        current_confidences: list[float] = []

        for result in results:
            if result.state == GameState.PLAY:
                if current_start is None:
                    current_start = result.start_frame
                    current_confidences = []
                current_end = result.end_frame
                current_confidences.append(result.confidence)
            else:
                if current_start is not None:
                    regions.append((current_start, current_end, current_confidences))
                    current_start = None
                    current_confidences = []

        # Don't forget last region
        if current_start is not None:
            regions.append((current_start, current_end, current_confidences))

        # Add padding around each region and merge overlapping
        padding_frames = int(self.motion_padding_seconds * fps)
        padded_regions: list[tuple[int, int, float]] = []  # (start, end, avg_motion)

        for start, end, confidences in regions:
            padded_start = max(0, start - padding_frames)
            padded_end = min(total_frames, end + padding_frames)
            avg_motion = sum(confidences) / len(confidences) if confidences else 0.5
            padded_regions.append((padded_start, padded_end, avg_motion))

        # Merge overlapping regions
        if not padded_regions:
            return []

        padded_regions.sort(key=lambda x: x[0])
        merged: list[tuple[int, int, float]] = [padded_regions[0]]

        for start, end, avg_motion in padded_regions[1:]:
            last_start, last_end, last_avg = merged[-1]
            if start <= last_end:
                # Overlapping - merge, weighted average of motion scores
                new_end = max(last_end, end)
                # Weight by duration
                last_dur = last_end - last_start
                new_dur = end - start
                weighted_avg = (last_avg * last_dur + avg_motion * new_dur) / (last_dur + new_dur)
                merged[-1] = (last_start, new_end, weighted_avg)
            else:
                merged.append((start, end, avg_motion))

        return [
            MotionRegion(start_frame=start, end_frame=end, avg_motion=avg_motion)
            for start, end, avg_motion in merged
        ]

    def _analyze_motion_regions(
        self,
        video: Video,
        regions: list[MotionRegion],
        stride: int,
        batch_size: int,
        fps: float,
        total_frames: int,
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> list[GameStateResult]:
        """
        Run ML analysis on motion regions using adaptive stride/sampling.

        Uses zone-based analysis:
        - Boundary zones (first/last 2 sec): stride 8, sample every frame
        - Middle zone: stride 16, sample every 2nd frame (wider temporal window)

        Confidence-based skipping: If a region has very high motion confidence
        (>0.90 by default), skip ML entirely and classify as PLAY directly.
        This saves 20-40% of ML calls on typical videos.

        Returns:
            List of GameStateResult covering entire video
        """
        ml_analyzer = self._get_ml_analyzer()
        config = get_config()
        all_results = []

        boundary_frames = int(self.boundary_seconds * fps)
        window_size = config.game_state.window_size
        min_region_for_zones = boundary_frames * 2 + 32  # Need space for middle zone

        # Confidence-based skipping thresholds
        enable_skip = config.two_pass.enable_confidence_skip
        skip_threshold = config.two_pass.skip_ml_high_threshold

        # Separate regions into high-confidence (skip ML) and normal (need ML)
        skip_regions = []
        ml_regions = []
        for region in regions:
            if enable_skip and region.avg_motion >= skip_threshold:
                skip_regions.append(region)
            else:
                ml_regions.append(region)

        # Calculate total windows (approximate for progress) - only for ML regions
        total_windows = 0
        for region in ml_regions:
            if region.duration_frames < min_region_for_zones:
                # All boundary: stride 8
                total_windows += max(0, (region.duration_frames - window_size) // stride + 1)
            else:
                # Boundaries + middle
                boundary_windows = 2 * max(0, (boundary_frames - window_size) // stride + 1)
                middle_frames = region.duration_frames - 2 * boundary_frames
                middle_windows = max(0, (middle_frames - 31) // 16 + 1)  # stride 16, needs 31 frames
                total_windows += boundary_windows + middle_windows

        processed_windows = 0
        skipped_count = len(skip_regions)

        # Add NO_PLAY for gap before first region
        if regions and regions[0].start_frame > 0:
            all_results.append(
                GameStateResult(
                    state=GameState.NO_PLAY,
                    confidence=0.9,
                    start_frame=0,
                    end_frame=regions[0].start_frame - 1,
                )
            )

        prev_end = 0
        ml_region_idx = 0

        for region_idx, region in enumerate(regions):
            start, end = region.start_frame, region.end_frame

            # Add NO_PLAY gap between regions
            if start > prev_end:
                all_results.append(
                    GameStateResult(
                        state=GameState.NO_PLAY,
                        confidence=0.9,
                        start_frame=prev_end,
                        end_frame=start - 1,
                    )
                )

            # Check if this region should skip ML (high motion confidence)
            is_skip_region = region in skip_regions
            region_frames = end - start

            if is_skip_region:
                # High confidence motion - skip ML and classify as PLAY directly
                all_results.append(
                    GameStateResult(
                        state=GameState.PLAY,
                        confidence=region.avg_motion,  # Use motion confidence
                        start_frame=start,
                        end_frame=end,
                    )
                )
                region_windows = 0
            elif region_frames < window_size:
                # Too short for ML window - assume PLAY since motion detected
                all_results.append(
                    GameStateResult(
                        state=GameState.PLAY,
                        confidence=0.7,
                        start_frame=start,
                        end_frame=end,
                    )
                )
                region_windows = 0
            elif region_frames < min_region_for_zones:
                # Short region: analyze entirely with high precision (boundary settings)
                region_results = self._analyze_region(
                    video, ml_analyzer, start, end,
                    stride=stride, frame_sample=1, batch_size=batch_size
                )
                all_results.extend(region_results)
                region_windows = max(0, (region_frames - window_size) // stride + 1)
                ml_region_idx += 1
            else:
                # Zone-based analysis: boundary + middle + boundary
                # Start boundary: high precision
                start_boundary_end = start + boundary_frames
                start_results = self._analyze_region(
                    video, ml_analyzer, start, start_boundary_end,
                    stride=stride, frame_sample=1, batch_size=batch_size
                )
                all_results.extend(start_results)

                # Middle zone: lower precision, wider temporal window
                end_boundary_start = end - boundary_frames
                middle_results = self._analyze_region(
                    video, ml_analyzer, start_boundary_end, end_boundary_start,
                    stride=16, frame_sample=2, batch_size=batch_size
                )
                all_results.extend(middle_results)

                # End boundary: high precision
                end_results = self._analyze_region(
                    video, ml_analyzer, end_boundary_start, end,
                    stride=stride, frame_sample=1, batch_size=batch_size
                )
                all_results.extend(end_results)

                # Calculate windows for progress
                boundary_windows = 2 * max(0, (boundary_frames - window_size) // stride + 1)
                middle_frames = end_boundary_start - start_boundary_end
                middle_windows = max(0, (middle_frames - 31) // 16 + 1)
                region_windows = boundary_windows + middle_windows
                ml_region_idx += 1

            processed_windows += region_windows

            if progress_callback and total_windows > 0:
                progress = 0.1 + 0.9 * (processed_windows / total_windows)
                skip_msg = f" (skipped {skipped_count} high-conf)" if skipped_count > 0 else ""
                progress_callback(
                    progress, f"ML region {ml_region_idx}/{len(ml_regions)}{skip_msg}"
                )

            prev_end = end

        # Add NO_PLAY for gap after last region
        if regions and regions[-1].end_frame < total_frames:
            all_results.append(
                GameStateResult(
                    state=GameState.NO_PLAY,
                    confidence=0.9,
                    start_frame=regions[-1].end_frame,
                    end_frame=total_frames - 1,
                )
            )

        return all_results

    def _analyze_region(
        self,
        video: Video,
        ml_analyzer,
        start_frame: int,
        end_frame: int,
        stride: int,
        frame_sample: int = 1,
        batch_size: int = 16,
    ) -> list[GameStateResult]:
        """
        Analyze a specific region with ML using sequential reading.

        Args:
            video: Video to analyze
            ml_analyzer: GameStateAnalyzer instance
            start_frame: Start frame of region
            end_frame: End frame of region
            stride: Frames between window starts
            frame_sample: Sample every Nth frame for the 16-frame window.
                         1 = consecutive frames (0.5s window at 30fps)
                         2 = every 2nd frame (1.0s window at 30fps)
                         3 = every 3rd frame (1.5s window at 30fps)
            batch_size: Batch size for ML inference

        Returns:
            List of GameStateResult for the region
        """
        import cv2

        config = get_config()
        profiler = get_profiler()
        classifier = ml_analyzer._get_classifier()
        results = []

        window_size = config.game_state.window_size
        target_size = config.game_state.analysis_size

        # Calculate frame span needed for one window with sampling
        # e.g., frame_sample=2: need frames 0,2,4,...,30 = 31 frame span
        frame_span = (window_size - 1) * frame_sample + 1

        # Sequential reading with frame buffer for this region
        frame_buffer = []
        buffer_start_idx = start_frame
        next_window_start = start_frame

        # Check first frame to see if resize is needed
        needs_resize = None
        decode_time = 0.0
        resize_time = 0.0
        frames_decoded = 0
        import time as _time

        for frame_idx, frame in video.iter_frames(start_frame=start_frame, end_frame=end_frame):
            t0 = _time.perf_counter()
            frames_decoded += 1

            # Check on first frame if resize is needed (proxy might already be 224x224)
            if needs_resize is None:
                h, w = frame.shape[:2]
                needs_resize = (w != target_size[0] or h != target_size[1])

            decode_time += _time.perf_counter() - t0

            # Only resize if needed (skip for optimized proxy)
            if needs_resize:
                t1 = _time.perf_counter()
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                resize_time += _time.perf_counter() - t1

            frame_buffer.append(frame)

            # Check if we can process any windows
            # Need frame_span frames in buffer for subsampling
            while (len(frame_buffer) >= frame_span and
                   next_window_start <= frame_idx - frame_span + 1):

                # Calculate buffer offset for this window
                buffer_offset = next_window_start - buffer_start_idx

                if buffer_offset >= 0 and buffer_offset + frame_span <= len(frame_buffer):
                    # Extract and subsample window frames from buffer
                    raw_frames = frame_buffer[buffer_offset:buffer_offset + frame_span]
                    window_frames = raw_frames[::frame_sample][:window_size]

                    # Collect batch
                    batch_frames = [window_frames]
                    batch_starts = [next_window_start]

                    # Try to add more windows to batch
                    temp_start = next_window_start + stride
                    while (len(batch_frames) < batch_size and
                           temp_start <= frame_idx - frame_span + 1 and
                           temp_start < end_frame - frame_span + 1):
                        temp_offset = temp_start - buffer_start_idx
                        if temp_offset >= 0 and temp_offset + frame_span <= len(frame_buffer):
                            temp_raw = frame_buffer[temp_offset:temp_offset + frame_span]
                            batch_frames.append(temp_raw[::frame_sample][:window_size])
                            batch_starts.append(temp_start)
                            temp_start += stride
                        else:
                            break

                    # Process batch
                    batch_results = classifier.classify_segments_batch(batch_frames)

                    for i, (state, confidence) in enumerate(batch_results):
                        frame_start = batch_starts[i]
                        # end_frame reflects the actual temporal span
                        actual_end = frame_start + frame_span - 1
                        results.append(
                            GameStateResult(
                                state=state,
                                confidence=confidence,
                                start_frame=frame_start,
                                end_frame=actual_end,
                            )
                        )

                    next_window_start = batch_starts[-1] + stride
                else:
                    break

            # Trim buffer to save memory (keep only what we need for future windows)
            min_needed_frame = next_window_start
            frames_to_drop = min_needed_frame - buffer_start_idx
            if frames_to_drop > 0:
                frame_buffer = frame_buffer[frames_to_drop:]
                buffer_start_idx = min_needed_frame

        # Process any remaining windows
        while next_window_start <= end_frame - frame_span:
            buffer_offset = next_window_start - buffer_start_idx
            if buffer_offset >= 0 and buffer_offset + frame_span <= len(frame_buffer):
                raw_frames = frame_buffer[buffer_offset:buffer_offset + frame_span]
                window_frames = raw_frames[::frame_sample][:window_size]
                batch_results = classifier.classify_segments_batch([window_frames])

                for state, confidence in batch_results:
                    actual_end = next_window_start + frame_span - 1
                    results.append(
                        GameStateResult(
                            state=state,
                            confidence=confidence,
                            start_frame=next_window_start,
                            end_frame=actual_end,
                        )
                    )
            next_window_start += stride

        # Record accumulated timing from frame processing
        from rallycut.core.profiler import TimingEntry
        profiler._entries.append(TimingEntry(
            component="decode",
            operation="read_frames",
            duration_seconds=decode_time,
            metadata={"frames": frames_decoded},
        ))
        if resize_time > 0:
            profiler._entries.append(TimingEntry(
                component="decode",
                operation="resize",
                duration_seconds=resize_time,
                metadata={"frames": frames_decoded},
            ))

        return results

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
                    )
                )
            else:
                smoothed.append(result)

        return smoothed
