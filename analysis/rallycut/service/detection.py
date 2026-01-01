"""Core detection service for cloud deployment."""

import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from rallycut.core.models import GameState
from rallycut.core.video import Video
from rallycut.processing.cutter import VideoCutter
from rallycut.service.schemas import (
    DetectedSegment,
    DetectionConfig,
    DetectionRequest,
    DetectionResponse,
    MatchStatistics,
    SegmentType,
    VideoMetadata,
)
from rallycut.service.storage import cleanup_temp, download_video


class DetectionService:
    """
    Cloud detection service - analysis only, no video cutting.

    Wraps VideoCutter.analyze_only() for cloud deployment.
    Downloads video from cloud storage, runs detection, returns JSON results.
    """

    def __init__(
        self,
        device: str = "cuda",
        temp_dir: Path | None = None,
    ):
        """
        Initialize detection service.

        Args:
            device: PyTorch device (cuda, mps, cpu)
            temp_dir: Directory for temporary video files
        """
        self.device = device
        self.temp_dir = temp_dir or Path("/tmp/rallycut")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def detect(
        self,
        request: DetectionRequest,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> DetectionResponse:
        """
        Run rally detection on a video from cloud storage.

        Args:
            request: Detection request with video URL and config
            progress_callback: Optional callback for progress updates (0-1, message)

        Returns:
            DetectionResponse with segments, scores, and statistics
        """
        job_id = request.job_id or f"det_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        try:
            # Phase 1: Download video (0-10% progress)
            if progress_callback:
                progress_callback(0.0, "Receiving the serve...")

            local_path = download_video(
                request.video_url,
                self.temp_dir,
                progress_callback=(
                    lambda p, m: progress_callback(p * 0.1, m)
                    if progress_callback
                    else None
                ),
            )

            # Get video metadata
            with Video(local_path) as video:
                video_info = video.info

            video_meta = VideoMetadata(
                source_url=request.video_url,
                duration=video_info.duration,
                fps=video_info.fps,
                width=video_info.width,
                height=video_info.height,
                frame_count=video_info.frame_count,
            )

            # Phase 2: Run analysis (10-90% progress)
            config = request.config or DetectionConfig()
            cutter = VideoCutter(
                device=self.device,
                padding_seconds=config.padding_seconds,
                min_play_duration=config.min_play_duration,
                min_gap_seconds=config.min_gap_seconds,
                use_proxy=config.use_proxy,
                stride=config.stride,
            )

            def analysis_progress(pct: float, msg: str) -> None:
                if progress_callback:
                    # Map 0-1 to 0.1-0.9
                    progress_callback(0.1 + pct * 0.8, msg)

            segments = cutter.analyze_only(local_path, analysis_progress)

            # Phase 3: Build response (90-100% progress)
            if progress_callback:
                progress_callback(0.9, "Wrapping up the match...")

            detected_segments = []
            for i, seg in enumerate(segments):
                # Map GameState to SegmentType
                if seg.state in (GameState.PLAY, GameState.SERVICE):
                    seg_type = SegmentType.RALLY
                else:
                    seg_type = SegmentType.DEAD_TIME

                detected_segments.append(
                    DetectedSegment(
                        segment_id=i + 1,
                        segment_type=seg_type,
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        start_frame=seg.start_frame,
                        end_frame=seg.end_frame,
                        duration=seg.duration,
                    )
                )

            # Calculate statistics
            stats = cutter.get_cut_stats(video_info.duration, segments)
            durations = [s.duration for s in segments] if segments else [0.0]

            match_stats = MatchStatistics(
                total_segments=stats["segment_count"],
                rally_count=stats["segment_count"],
                total_play_duration=stats["kept_duration"],
                total_dead_time=stats["removed_duration"],
                play_percentage=stats["kept_percentage"],
                avg_rally_duration=(
                    stats["kept_duration"] / max(1, stats["segment_count"])
                ),
                longest_rally_duration=max(durations),
                shortest_rally_duration=min(durations) if durations else 0.0,
            )

            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(1.0, "Match point!")

            return DetectionResponse(
                job_id=job_id,
                status="completed",
                video=video_meta,
                segments=detected_segments,
                statistics=match_stats,
                processing_time_seconds=processing_time,
                created_at=datetime.utcnow(),
            )

        except Exception as e:
            return DetectionResponse(
                job_id=job_id,
                status="failed",
                video=None,
                segments=[],
                statistics=None,
                processing_time_seconds=time.time() - start_time,
                created_at=datetime.utcnow(),
                error=str(e),
            )

        finally:
            # Cleanup temp files
            cleanup_temp(self.temp_dir)
