"""FFmpeg video export functionality for RallyCut."""

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from rallycut.core.models import TimeSegment


class FFmpegExporter:
    """Exports video segments using FFmpeg."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is available."""
        try:
            subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
            )

    def export_segments(
        self,
        input_path: Path,
        output_path: Path,
        segments: list[TimeSegment],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """
        Export video segments to a single output file.

        Uses FFmpeg's concat demuxer for efficient concatenation.

        Args:
            input_path: Path to source video
            output_path: Path for output video
            segments: List of segments to include
            progress_callback: Callback for progress updates

        Returns:
            Path to output video
        """
        if not segments:
            raise ValueError("No segments to export")

        # Create temporary directory for segment files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            segment_files = []

            # Extract each segment
            total_segments = len(segments)
            for i, segment in enumerate(segments):
                if progress_callback:
                    progress_callback(
                        i / total_segments * 0.8,
                        f"Extracting segment {i + 1}/{total_segments}",
                    )

                segment_file = temp_path / f"segment_{i:04d}.mp4"
                self._extract_segment(
                    input_path,
                    segment_file,
                    segment.start_time,
                    segment.end_time,
                )
                segment_files.append(segment_file)

            # Create concat file list
            if progress_callback:
                progress_callback(0.8, "Concatenating segments")

            concat_file = temp_path / "concat.txt"
            with open(concat_file, "w") as f:
                for segment_file in segment_files:
                    f.write(f"file '{segment_file}'\n")

            # Concatenate segments
            self._concatenate(concat_file, output_path)

            if progress_callback:
                progress_callback(1.0, "Export complete")

        return output_path

    def _extract_segment(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ):
        """Extract a single segment from the video."""
        duration = end_time - start_time

        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output
            "-ss", str(start_time),  # Seek position
            "-i", str(input_path),
            "-t", str(duration),  # Duration
            "-c", "copy",  # Stream copy (fast, no re-encoding)
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # If stream copy fails, try with re-encoding
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-ss", str(start_time),
                "-i", str(input_path),
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg segment extraction failed: {result.stderr}"
                )

    def _concatenate(self, concat_file: Path, output_path: Path):
        """Concatenate segments using FFmpeg concat demuxer."""
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concatenation failed: {result.stderr}")

    def export_clip(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> Path:
        """
        Export a single clip from the video.

        Args:
            input_path: Path to source video
            output_path: Path for output video
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Path to output video
        """
        self._extract_segment(input_path, output_path, start_time, end_time)
        return output_path
